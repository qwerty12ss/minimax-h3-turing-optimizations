"""Same frozen CK functions, admitting other rows only after the same selectors."""
import ast
import threading

def make(ck,extract):
    nodes,proof=extract.derive()
    namespace=dict(vars(ck));original=ck._int8_linear_turing_quantized
    counts={0:0,1:0};lock=threading.Lock()
    def observed(*args,**kwargs):
        q,w,sx,sw,bias,dtype=args
        assert not kwargs and q.device==w.device==sx.device==sw.device
        assert bias is None and dtype is ck.torch.float16
        assert 3<=q.shape[0]<=8192 and q.shape[1]==5376 and w.shape[0] in (10752,14336)
        result=original(*args,**kwargs)
        assert result is not None,'No unqualified fallback permitted'
        with lock:counts[q.device.index]+=1
        return result
    namespace['_int8_linear_turing_quantized']=observed
    exec(compile(ast.Module(body=nodes,type_ignores=[]),ck.__file__,'exec',dont_inherit=True),namespace)
    used={n.id for node in nodes for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    bindings={name:vars(ck)[name] for name in used if name in vars(ck)}
    def check():assert all(getattr(ck,name) is value for name,value in bindings.items())
    def guard(fn):
        def call(x,w,ws,*args,**kwargs):
            check()
            assert not args and x.dtype is ck.torch.float16 and x.ndim==2
            assert 3<=x.shape[0]<=8192 and x.shape[1]==5376 and w.shape[0] in (10752,14336) and w.shape[1]==5376
            assert kwargs.get('out_dtype') is ck.torch.float16 and kwargs.get('convrot') is True
            assert kwargs.get('convrot_groupsize')==256 and kwargs.get('bias') is None
            assert kwargs.get('residual') is None and kwargs.get('input_act') in (None,'none')
            assert ck._prefer_turing_fused_int8(x.shape[0],w.shape[0],5376)
            assert ck._cuda_device_should_use_turing_kernels(x.get_device())
            assert hasattr(ck._C,'cutlass_turing_int8_dequant') and ck._convrot_fused_shared_memory_fits(x,5376,256)
            return fn(x,w,ws,**kwargs)
        return call
    def snapshot():
        with lock:return dict(counts)
    proof=dict(proof,supported_rows='3..8192 with original fused selector and quantizer eligibility; 1/2 use original Half copy and original linear',extension=True)
    return *(guard(namespace[node.name]) for node in nodes),check,snapshot,proof
