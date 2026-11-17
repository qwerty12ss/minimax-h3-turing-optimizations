"""Private AWQ adaptation of Luna's row-sharded, two-slot Qwen experiment.

Preserves native NVFP4 bytes, scales and smoothing; no GGUF conversion.
Derived scheduling/TP design: Luna-AI-Infra, GPL-3.0-only.
"""
import contextlib,copy,dataclasses,threading,time,types,weakref
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F

CACHE=None
PROJ={'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'}

def version(t):
    try:return t._version
    except RuntimeError:return 'inference'

def key(model):
    def item(t):
        if hasattr(t,'_qdata'):
            q=t._qdata
            metadata=tuple((f.name, (getattr(t._params,f.name).data_ptr(),version(getattr(t._params,f.name)),tuple(getattr(t._params,f.name).shape))if torch.is_tensor(getattr(t._params,f.name))else str(getattr(t._params,f.name)))for f in dataclasses.fields(t._params)if f.name!='orig_dtype')
            return (q.data_ptr(),version(q),tuple(q.shape),str(q.dtype),t._layout_cls,metadata)
        return (t.data_ptr(),version(t),tuple(t.shape),str(t.dtype))
    return tuple((n,item(t))for n,t in list(model.named_parameters())+list(model.named_buffers()))

class Pack:
    def __init__(self,layer,rank):
        self.layout=[];self.quant={};offset=0;parts=[]
        def add(name,t):
            nonlocal offset
            t=t.detach().contiguous();offset=(offset+255)//256*256;size=t.numel()*t.element_size()
            self.layout.append((name,offset,size,t.dtype,tuple(t.shape)));parts.append(t);offset+=size
        for name,t in list(layer.named_parameters())+list(layer.named_buffers()):
            module,_,field=name.rpartition('.');rows=None
            if module in PROJ and field=='weight':
                assert t.shape[0]%2==0;h=t.shape[0]//2;rows=slice(rank*h,(rank+1)*h)
            if hasattr(t,'_qdata'):
                assert t._layout_cls=='TensorCoreNVFP4Layout' and rows is not None
                params={};add(name+'#q',t._qdata[rows])
                for f in dataclasses.fields(t._params):
                    v=getattr(t._params,f.name)
                    if torch.is_tensor(v):
                        if v.ndim>0 and v.shape[0]==t.shape[0]:v=v[rows]
                        add(name+'#'+f.name,v);params[f.name]=None
                    else:params[f.name]=v
                params['orig_shape']=(h,t.shape[1]);self.quant[name]=(t._layout_cls,type(t._params),params)
            else:
                if module in PROJ and field=='bias':
                    assert t.numel()%2==0;h=t.numel()//2;t=t[rank*h:(rank+1)*h]
                add(name,t)
        self.host=torch.empty(offset,dtype=torch.uint8,pin_memory=True)
        for (_,start,size,_,_),t in zip(self.layout,parts):self.host[start:start+size].copy_(t.reshape(-1).view(torch.uint8))
    def bind(self,buffer):
        from comfy_kitchen.tensor.base import QuantizedTensor
        values={name:buffer[start:start+size].view(dtype).view(shape)for name,start,size,dtype,shape in self.layout}
        for name,(layout,cls,fields)in self.quant.items():
            params={k:values.pop(name+'#'+k)if v is None else v for k,v in fields.items()}
            values[name]=QuantizedTensor(values.pop(name+'#q'),layout,cls(**params))
        return values

class Bank:
    def __init__(self,packs,device):
        self.packs=packs;self.device=device
        with torch.cuda.device(device):
            self.stream=torch.cuda.Stream();self.buffers=[torch.empty(max(p.host.numel()for p in packs),dtype=torch.uint8,device=device)for _ in range(2)]
        self.loaded=[None,None];self.done=[None,None];self.ready=[None,None]
    def prefetch(self,index,slot):
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
            if self.done[slot]is not None:self.stream.wait_event(self.done[slot])
            p=self.packs[index];self.buffers[slot][:p.host.numel()].copy_(p.host,non_blocking=True)
            self.ready[slot]=torch.cuda.Event();self.ready[slot].record()
        self.loaded[slot]=index
    def acquire(self,index):
        if index==0:self.prefetch(0,0);self.prefetch(1,1)
        slot=self.loaded.index(index);torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        if index+1<len(self.packs)and index+1 not in self.loaded:self.prefetch(index+1,1-slot)
        return slot,self.packs[index].bind(self.buffers[slot])
    def finish(self,slot):
        self.done[slot]=torch.cuda.Event();self.done[slot].record()

class Exchange:
    def __init__(self):self.barrier=threading.Barrier(2,timeout=60);self.values=[None,None]
    def gather(self,r,v):
        v=v.contiguous();e=torch.cuda.Event();e.record();self.values[r]=(v,e);self.barrier.wait()
        peer,ready=self.values[1-r];torch.cuda.current_stream(r).wait_event(ready)
        received=peer.to(device=v.device,non_blocking=True);peer.record_stream(torch.cuda.current_stream(r))
        result=torch.cat((v,received)if r==0 else(received,v),-1)
        self.barrier.wait()
        return result

def module_copy(layer,views):
    # Shallow copies of module trees; the original model's fields are never rebound.
    def clone(module,prefix=''):
        m=copy.copy(module);m._parameters=dict(module._parameters);m._buffers=dict(module._buffers);m._modules={}
        for name,t in m._parameters.items():
            if t is not None:m._parameters[name]=views[prefix+name]
        for name,t in m._buffers.items():
            if t is not None:m._buffers[name]=views[prefix+name]
        for name,child in module._modules.items():m._modules[name]=clone(child,prefix+name+'.')
        if hasattr(m,'comfy_cast_weights'):m.comfy_cast_weights=True
        return m
    return clone(layer)

class Session:
    def __init__(self,transformer,packs):
        self.t=transformer;self.packs=packs;self.banks=[];self.restores=[];self.pool=ThreadPoolExecutor(2);self.exchange=Exchange()
        self.streams=[torch.cuda.Stream(device=i)for i in range(2)];self.static={};self.calls=0
    def worker(self,r,i,x,mask,freqs,attention,ready):
        from comfy.text_encoders.llama import apply_rope
        try:
            with torch.inference_mode(),torch.cuda.device(r),torch.cuda.stream(self.streams[r]):
                self.streams[r].wait_event(ready);slot,views=self.banks[r].acquire(i);layer=module_copy(self.t.model.layers[i],views);a=layer.self_attn
                b,s,_=x.shape;res=x;x=layer.input_layernorm(x);h=a.num_heads//2;kv=a.num_kv_heads//2;d=a.head_dim
                q=a.q_proj(x).view(b,s,h,d).transpose(1,2);k=a.k_proj(x).view(b,s,kv,d).transpose(1,2);v=a.v_proj(x).view(b,s,kv,d).transpose(1,2)
                if a.q_norm is not None:q=a.q_norm(q)
                if a.k_norm is not None:k=a.k_norm(k)
                q,k=apply_rope(q,k,freqs_cis=freqs)
                ctx=attention(q,k,v,h,mask=mask,skip_reshape=True,**({'enable_gqa':True}if h!=kv else{}))
                x=res+self.exchange.gather(r,a.o_proj(self.exchange.gather(r,ctx)))
                res=x;x=layer.post_attention_layernorm(x);g=layer.mlp.gate_proj(x);u=layer.mlp.up_proj(x)
                x=res+self.exchange.gather(r,layer.mlp.down_proj(self.exchange.gather(r,layer.mlp.activation(g)*u)))
                self.banks[r].finish(slot);e=torch.cuda.Event();e.record();return x,e
        except BaseException:self.exchange.barrier.abort();raise
    def block(self,i,x,attention_mask=None,freqs_cis=None,optimized_attention=None,past_key_value=None):
        assert past_key_value is None
        inputs=[]
        for r in range(2):
            with torch.cuda.device(r):
                def move(v):
                    if torch.is_tensor(v):return v.to(r,non_blocking=True)
                    if isinstance(v,tuple):return tuple(move(z)for z in v)
                    if isinstance(v,list):return [move(z)for z in v]
                    return v
                value=move(x);mask=move(attention_mask);freq=move(freqs_cis)
                e=torch.cuda.Event();e.record();inputs.append((r,i,value,mask,freq,optimized_attention,e))
        futures=[self.pool.submit(self.worker,*v)for v in inputs];result=[f.result()for f in futures]
        for r,(v,e)in enumerate(result):torch.cuda.current_stream(r).wait_event(e);v.record_stream(torch.cuda.current_stream(r))
        self.calls+=1
        return result[0][0],None
    def __enter__(self):
        self.banks=[Bank(p,torch.device('cuda',r))for r,p in enumerate(self.packs)]
        # Static embedding, vision and norms keep their original operations.
        for name,module in self.t.named_modules():
            if name.startswith('model.layers.'):continue
            for mapping in [module._parameters,module._buffers]:
                for field,t in list(mapping.items()):
                    if t is not None:self.restores.append((mapping,field,t));mapping[field]=t.to('cuda:0')
            if hasattr(module,'comfy_cast_weights'):
                self.restores.append((module.__dict__,'comfy_cast_weights',module.comfy_cast_weights));module.comfy_cast_weights=True
        for i,layer in enumerate(self.t.model.layers):
            original=layer.__dict__.get('forward',None);self.restores.append((layer.__dict__,'forward',original))
            layer.forward=functools_partial(self.block,i)
        return self
    def __exit__(self,*exc):
        self.pool.shutdown(wait=True)
        for r in range(2):torch.cuda.synchronize(r)
        for mapping,k,v in reversed(self.restores):
            if v is None and k=='forward':mapping.pop(k,None)
            else:mapping[k]=v
        self.restores=[];self.banks=[]

from functools import partial as functools_partial

def wrap(clip,transformer,stats):
    own=clip.clone();original=own.encode_from_tokens.__func__
    def encode(self,*args,**kwargs):
        global CACHE
        import comfy.model_management as mm
        with torch.inference_mode():
            t=time.perf_counter();mm.unload_all_models();stats['qwen_previous_unload_seconds']=time.perf_counter()-t
            t=time.perf_counter();fingerprint=key(transformer)
            hit=CACHE is not None and CACHE['owner']() is transformer and CACHE['key']==fingerprint
            if not hit:
                # Drop old pinned buffers before allocating a new generation.
                CACHE=None
                packs=[[Pack(layer,r)for layer in transformer.model.layers]for r in range(2)]
                CACHE={'owner':weakref.ref(transformer),'key':fingerprint,'packs':packs}
            stats['qwen_pack_seconds']=time.perf_counter()-t;stats['qwen_pack_hit']=hit
            t=time.perf_counter()
            stats['qwen_retained_pack_bytes']=sum(p.host.numel()for ps in CACHE['packs']for p in ps)
            with Session(transformer,CACHE['packs'])as session:out=original(self,*args,**kwargs)
            stats['qwen_tp_session_seconds']=time.perf_counter()-t;stats['qwen_tp_blocks']=session.calls
            return out
    own.encode_from_tokens=types.MethodType(encode,own);own.load_model=types.MethodType(lambda self,tokens={}:self.patcher,own)
    return own
