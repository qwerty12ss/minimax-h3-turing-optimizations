"""Private immutable CPU rank packs with unchanged native TP2 arithmetic."""
import dataclasses,time,types,hashlib
import torch

def version(t):
    try:return t._version
    except RuntimeError:return 'inference'

def signature(t):
    # ComfyUI may rebind Parameter/QuantizedTensor wrappers during offload.
    # The original runtime guard admits the unchanged CPU qdata storage.
    if hasattr(t,'_qdata'):
        q=t._qdata;s=t._params.scale
        return ('quant',q.data_ptr(),version(q),tuple(q.shape),tuple(q.stride()),str(q.dtype),t._layout_cls,
                hashlib.sha256(s.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    return ('plain',tuple(t.shape),str(t.dtype),hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

class RankPack:
    def __init__(self,shard,rank):
        self.items=[];self.weights={};self.loras={};parts=[];offset=0
        def add(name,t):
            nonlocal offset
            t=t.detach().contiguous();offset=(offset+255)//256*256;size=t.numel()*t.element_size()
            self.items.append((name,offset,size,t.dtype,tuple(t.shape)));parts.append(t);offset+=size
        for name in ['qkv','op','fc1','fc2']:
            t=shard[name][rank];assert hasattr(t,'_qdata');add(name+'#q',t._qdata);fields={}
            for f in dataclasses.fields(t._params):
                v=getattr(t._params,f.name)
                if torch.is_tensor(v):add(name+'#'+f.name,v);fields[f.name]=None
                else:fields[f.name]=v
            self.weights[name]=(t._layout_cls,type(t._params),fields)
        for name in ['n1','n2','qn','kn']:add(name,shard[name][rank])
        for name in ['lqkv','lop','lfc1','lfc2']:
            up,down,scale=shard[name][rank];assert scale==1.;add(name+'#up',up);add(name+'#down',down)
        self.host=torch.empty(offset,dtype=torch.uint8,pin_memory=True)
        for (_,start,size,_,_),t in zip(self.items,parts):self.host[start:start+size].copy_(t.reshape(-1).view(torch.uint8))
    def to(self,device):
        from comfy_kitchen.tensor.base import QuantizedTensor
        gpu=self.host.to(device,non_blocking=True)
        values={name:gpu[start:start+size].view(dtype).view(shape)for name,start,size,dtype,shape in self.items}
        result={}
        for name,(layout,cls,fields)in self.weights.items():
            p={k:values[name+'#'+k]if v is None else v for k,v in fields.items()};result[name]=QuantizedTensor(values[name+'#q'],layout,cls(**p))
        for name in ['n1','n2','qn','kn']:result[name]=values[name]
        for name in ['lqkv','lop','lfc1','lfc2']:result[name]=(values[name+'#up'],values[name+'#down'],1.)
        return result

def install(state,stats,enabled):
    if getattr(state,'_luna_pack_cache_installed',False):return
    state._luna_pack_cache_installed=True
    core=state._runtime_core;original=core.shard_block;cache={};owner=state._h3_public_pdd_owner
    def shard(block,d0,d1,lora=None):
        if not enabled():return original(block,d0,d1,lora=lora)
        if state._h3_public_pdd_owner is not owner:
            raise RuntimeError('The immutable PDD owner changed; restart the isolated trial')
        sources=[block.attn.qkv_proj.weight,block.attn.out_proj.weight,block.mlp.fc1.weight,block.mlp.fc2.weight,block.norm1.weight,block.norm2.weight,block.attn.q_norm.weight,block.attn.k_norm.weight]
        fingerprint=tuple(signature(t)for t in sources)
        slot=next(i for i,b in enumerate(state.patcher.model.diffusion_model.blocks)if b.adaln_proj is block.adaln_proj)
        if not 0 <= slot < 50 or len(state.patcher.model.diffusion_model.blocks) != 50:
            raise RuntimeError('This bounded cache is only admitted for the original 50-block H3 model')
        # Original BF16 adapter identities are admitted by the existing owner.
        # Half temporary IDs differ per preparation and are deliberately not used.
        hit=slot in cache and cache[slot][0]==fingerprint;t=time.perf_counter()
        if not hit:
            # One entry per block, released BEFORE rebuilding. Parameter rebinding
            # must never retain an unbounded history of 20GB generations.
            cache.pop(slot,None)
            cpu=original(block,torch.device('cpu'),torch.device('cpu'),lora=lora)
            packs=[RankPack(cpu,r)for r in range(2)]
            template={k:v for k,v in cpu.items()if k not in ['qkv','op','fc1','fc2','n1','n2','qn','kn','lqkv','lop','lfc1','lfc2']}
            cache[slot]=(fingerprint,packs,template)
        retained=sum(p.host.numel()for _,ps,_ in cache.values()for p in ps)
        if len(cache)>50 or retained>22*2**30:
            cache.clear()
            raise RuntimeError('Experimental DiT CPU pack cache exceeded its fixed memory budget')
        _,packs,template=cache[slot];ranks=[p.to(d)for p,d in zip(packs,[d0,d1])]
        out=dict(template,dev0=d0,dev1=d1)
        for k in ranks[0]:out[k]=tuple(r[k]for r in ranks)
        s=stats();s['pack_hits']=s.get('pack_hits',0)+int(hit);s['pack_misses']=s.get('pack_misses',0)+int(not hit);s['pack_copy_seconds']=s.get('pack_copy_seconds',0)+time.perf_counter()-t
        s['retained_pack_bytes']=retained;s['retained_pack_blocks']=len(cache)
        return out
    # Instance-owned namespace; no global TP2 replacement or source edits.
    state._runtime_core=types.SimpleNamespace(**dict(vars(core),shard_block=shard))
