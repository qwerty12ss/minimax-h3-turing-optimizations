"""Lazy packet consumption inside the original QKV/RoPE and FC1/SwiGLU/FC2 loops."""
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import torch.nn.functional as F
from .baseline import Shared as Baseline
from .packet_pipeline import linear_packet


class PacketView:
    dtype=torch.float16
    ndim=2
    def __init__(self,m,k,rank=1):self.shape=(m,k);self.device=torch.device('cuda',rank)
    def __getitem__(self,key):
        assert isinstance(key,slice) and key.step in (None,1)
        start,stop,_=key.indices(self.shape[0])
        return PacketView(max(0,stop-start),self.shape[1],self.device.index)
    def is_contiguous(self):return True
    def get_device(self):return self.device.index


class Shared(Baseline):
    def __init__(self,original,slots=2):
        super().__init__(original)
        # Complete both lazy peer-access setups on one host thread before the
        # producer and consumers can enter their first concurrent copy paths.
        with torch.cuda.device(0):
            peer_token = torch.ones(1, device=0)
        with torch.cuda.device(1):
            peer_reply = torch.empty(1, device=1)
        peer_reply.copy_(peer_token)
        peer_token.copy_(peer_reply)
        for rank in (0, 1):
            torch.cuda.synchronize(rank)
        del peer_token, peer_reply
        self.nslots=slots
        self.proof=dict(self.proof,metadata_only_Half_placeholder=False,explicit_packet_API=True,
                        packet_slots=slots,postquant_runtime='Original CK quantized Turing helper; no high-level x placeholder',
                        source_prefix_unchanged=True)
        self.proof['serialized_peer_access_setup'] = True
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='h3-packet-producer')
        self.producer=torch.cuda.Stream(device=0)
        self.copy0=torch.cuda.Stream(device=0);self.copy1=torch.cuda.Stream(device=1)

    def _drain(self):
        frame=self.active
        if frame is None:return
        assert frame['cursor']==[len(frame['chunk_sizes'])]*2
        frame['future'].result()
        frame['streams'][0].wait_event(frame['last_copied'])
        self.producer.wait_stream(frame['streams'][0]);self.producer.wait_stream(self.copy0)
        self.producer.wait_stream(self.copy1)
        self.copy1.wait_stream(frame['streams'][1])
        self.frames.clear();self.active=None

    def prepare(self,sh,h,phase,st0,st1):
        self._drain()
        assert phase in ('qkv','fc1') and h.dtype==torch.float16 and h.is_contiguous()
        m,k=h.shape;assert k==5376 and m>=3
        self.check()
        loras=sh['l'+phase]
        key=(loras[0][1].data_ptr(),loras[1][1].data_ptr())
        if key not in self.validated:
            assert torch.equal(loras[0][1].cpu().view(torch.uint8),loras[1][1].cpu().view(torch.uint8))
            self.validated.add(key)
        ready=torch.cuda.Event();ready.record(st0)
        h.record_stream(self.producer);h.record_stream(self.copy0)
        frame=dict(h=h,weights=sh[phase],lora=loras,phase=phase,streams=(st0,st1),ready=ready,
                   cursor=[0,0],chunk_sizes=[min(8192,m-s) for s in range(0,m,8192)],
                   packets={},condition=threading.Condition(),error=None)
        self.frames.append(frame);self.active=frame;self.counts['phases']+=1
        frame['future']=self.executor.submit(self._produce,frame)
        return PacketView(m,k)

    def _produce(self,frame):
        cond=frame['condition'];slots=[None]*self.nslots
        try:
            with torch.inference_mode(),torch.cuda.device(0):
                self.producer.wait_event(frame['ready'])
                for index,m in enumerate(frame['chunk_sizes']):
                    slot=index%self.nslots;old=slots[slot]
                    if old is not None:
                        with cond:
                            assert cond.wait_for(lambda:all(x is not None for x in old['done']),timeout=60), 'Consumer did not release packet slot'
                            del frame['packets'][index-self.nslots]
                        self.producer.wait_event(old['done'][0]);self.producer.wait_event(old['copied'])
                        self.copy1.wait_event(old['done'][1])
                        old['source']=None
                    x=frame['h'][index*8192:index*8192+m]
                    with torch.cuda.stream(self.producer):
                        if m>=3:
                            q,s=self.call(self.quant,x,frame['weights'][0]);z=F.linear(x,frame['lora'][0][1])
                            source=(q,s,z);self.counts['quant']+=1;self.counts['down']+=1
                        else:source=(x,)
                        produced=torch.cuda.Event();produced.record(self.producer)
                    self.copy0.wait_event(produced);self.copy1.wait_event(produced)
                    with torch.cuda.stream(self.copy0),torch.cuda.device(1),torch.cuda.stream(self.copy1):
                        if old is not None and len(old['buffers'])==len(source):buffers=old['buffers']
                        else:buffers=tuple(torch.empty((8192,*t.shape[1:]),device=1,dtype=t.dtype) for t in source)
                        dest=tuple(t[:m] for t in buffers)
                        for a,b in zip(dest,source):a.copy_(b,non_blocking=True)
                        copied=torch.cuda.Event();copied.record(self.copy1)
                    packet=dict(source=source,dest=dest,buffers=buffers,produced=produced,copied=copied,done=[None,None])
                    frame['last_copied']=copied
                    self.counts['packet_bytes']+=sum(t.numel()*t.element_size() for t in source)
                    slots[slot]=packet
                    with cond:frame['packets'][index]=packet;cond.notify_all()
                return True
        except BaseException as exc:
            with cond:frame['error']=exc;cond.notify_all()
            raise

    def linear(self,x,w,lora):
        frame=self.active
        if frame is None or not any(w is a for a in frame['weights']):return self.original(x,w,lora)
        rank=x.device.index;index=frame['cursor'][rank]
        assert w is frame['weights'][rank] and lora is frame['lora'][rank]
        assert x.shape[0]==frame['chunk_sizes'][index]
        cond=frame['condition']
        with cond:
            assert cond.wait_for(lambda:index in frame['packets'] or frame['error'] is not None,timeout=60), 'Producer did not deliver packet'
            if frame['error'] is not None:raise frame['error']
            packet=frame['packets'][index]
        stream=torch.cuda.current_stream(rank)
        stream.wait_event(packet['produced'] if rank==0 else packet['copied'])
        data=packet['source'] if rank==0 else packet['dest']
        if x.shape[0]<3:y=self.original(data[0],w,lora)
        else:
            epi=self.epilogue or (lambda y,d,s:y+d*s)
            y=linear_packet(*data,w,lora,epi)
            self.counts['post']+=1;self.counts['up']+=1
        done=torch.cuda.Event();done.record(stream)
        with cond:
            packet['done'][rank]=done;frame['cursor'][rank]+=1;cond.notify_all()
        return y

    def finish(self):self._drain()
