"""Explicit two-rank ownership of shared q/scales/Z for QKV and FC1."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import torch
import torch.nn.functional as F




def functions():
    from . import extract
    import comfy_kitchen.backends.cuda as ck
    from .general_functions import make
    return make(ck,extract)


def peer(t,destination,producer,consumer):
    consumer.wait_stream(producer)
    with torch.cuda.device(destination),torch.cuda.stream(consumer):
        out=torch.empty_like(t,device=destination)
        out.copy_(t,non_blocking=True)
    return out


class Shared:
    def __init__(self,original):
        self.original=original
        self.quant,self.post,self.baseline,self.check,self.snapshot,self.proof=functions()
        self.frames=[];self.active=None;self.counts={'quant':0,'post':0,'down':0,'up':0,'packet_bytes':0,'phases':0}
        self.validated=set()
        self.epilogue=None

    def call(self,fn,x,w,packet=None):
        kw=dict(out_dtype=torch.float16,convrot=True,convrot_groupsize=256)
        if packet is not None:kw.update(x_qdata=packet[0],x_scale=packet[1])
        return fn(x,w._qdata,w._params.scale,**kw)

    def prepare(self,sh,h,phase,st0,st1):
        assert phase in ('qkv','fc1') and h.dtype==torch.float16 and h.is_contiguous()
        m,k=h.shape
        assert k==5376 and m>=3
        self.check()
        frame={'streams':(st0,st1),'weights':sh[phase],'lora':sh['l'+phase],'packets':[[],[]],'z':[[],[]],
               'cursor':[0,0],'phase':phase,'small':{},'chunk_sizes':[min(8192,m-start) for start in range(0,m,8192)]}
        a,b=frame['lora']
        key=(a[1].data_ptr(),b[1].data_ptr())
        if key not in self.validated:
            assert torch.equal(a[1].cpu().view(torch.uint8),b[1].cpu().view(torch.uint8))
            self.validated.add(key)
        with torch.cuda.device(0),torch.cuda.stream(st0):
            for start in range(0,m,8192):
                x=h[start:start+8192]
                if x.shape[0]<3:
                    frame['packets'][0].append(None);frame['z'][0].append(None)
                    frame['small'][start]=x
                    continue
                packet=self.call(self.quant,x,frame['weights'][0])
                z=F.linear(x,a[1])
                frame['packets'][0].append(packet);frame['z'][0].append(z)
                self.counts['quant']+=1;self.counts['down']+=1
        for packet,z in zip(frame['packets'][0],frame['z'][0]):
            if packet is None:
                frame['packets'][1].append(None);frame['z'][1].append(None)
                continue
            copied=tuple(peer(t,1,st0,st1) for t in packet)
            frame['packets'][1].append(copied)
            frame['z'][1].append(peer(z,1,st0,st1))
            self.counts['packet_bytes']+=sum(t.numel()*t.element_size() for t in (*packet,z))
        with torch.cuda.device(1),torch.cuda.stream(st1):
            # Metadata only. No original fallback may consume this payload.
            placeholder=torch.empty_like(h,device=1)
        for start,small in frame['small'].items():
            st1.wait_stream(st0)
            with torch.cuda.device(1),torch.cuda.stream(st1):
                placeholder[start:start+small.shape[0]].copy_(small,non_blocking=True)
            self.counts['packet_bytes']+=small.numel()*small.element_size()
        self.frames.append(frame);self.active=frame;self.counts['phases']+=1
        return placeholder

    def linear(self,x,w,lora):
        frame=self.active
        if frame is None or not any(w is a for a in frame['weights']):return self.original(x,w,lora)
        rank=x.device.index
        assert w is frame['weights'][rank] and lora is frame['lora'][rank]
        index=frame['cursor'][rank]
        assert index<len(frame['chunk_sizes']) and x.shape[0]==frame['chunk_sizes'][index]
        packet=frame['packets'][rank][index];z=frame['z'][rank][index]
        if packet is None:
            frame['cursor'][rank]+=1
            return self.original(x,w,lora)
        y=self.call(self.post,x,w,packet)
        frame['cursor'][rank]+=1;self.counts['post']+=1;self.counts['up']+=1
        up,down,scale=lora
        delta=F.linear(z,up)
        return self.epilogue(y,delta,scale) if self.epilogue else y+delta*scale

    def finish(self):
        for frame in self.frames:assert frame['cursor']==[len(frame['chunk_sizes'])]*2
        # Enqueue a source-side fence covering all peer reads before owner release.
        pairs={tuple(int(s.cuda_stream) for s in frame['streams']):frame['streams'] for frame in self.frames}
        for a,b in pairs.values():
            a.wait_stream(b);b.wait_stream(a)
        self.frames.clear();self.active=None
