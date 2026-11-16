"""Transactional, instance-only preparation owner; no denoising hooks/kernels.

Installation/admission and the correctness-only node are separate modules. This
class never edits the original modules or computation callables. There are no
host executor workers in this preparation protocol.
"""
import threading
import types

from .transform import compile_prepare,compile_release,require

AB_BYTES_PER_RANK=573440000
LORA_KEYS=('lqkv','lop','lfc1','lfc2')


class InjectedPrepublishFailure(RuntimeError):pass


class Owner:
    def __init__(self,state,runtime,torch,adapters,identity_guard,emit,inject=False):
        require(not state.shards and not state._runtime_owned_backups and not state.patcher.backup,
                'Candidate must install before original preparation')
        require(type(inject) is bool and len(adapters)==200,'Exact static 200-adapter trunk required')
        self.state=state;self.runtime=runtime;self.torch=torch;self.adapters=adapters
        self.guard=identity_guard;self.emit=emit;self.inject=inject
        self.lock=threading.Lock();self.phase='empty';self.poison=False;self.cleanup_uncertain=False
        self.pending=None;self.sources=None;self.counts=None;self.ab_bytes=[0,0]
        self.pending_validated=False;self.records=[];self.current=None;self.retained=None
        self.prepares=0;self.cache_hits=0;self.releases=0;self.expected_injection_observed=False
        self.original_prepare=state.prepare;self.saved_original_release=state.release
        self.original_release=types.MethodType(compile_release(runtime,self._event),state)
        helpers={'_candidate_pending':self._pending,'_candidate_loras':self._loras,
                 '_candidate_append':self._append,'_candidate_validate_pending':self._validate_pending,
                 '_candidate_rollback':self._rollback,'_emit':self._event}
        self.derived_prepare=compile_prepare(runtime,helpers)

    def _event(self,event,**fields):
        self.emit(event,method='activation_bypass_original_base',activation_bypass=True,
                  offline_bake=False,all_timings_invalid_for_performance=True,**fields)

    def _pending(self,state):
        require(state is self.state and self.phase=='preparing' and self.pending is None,
                'A single unpublished generation is required')
        self.pending=[];self.sources=[];self.counts=[0,0];self.ab_bytes=[0,0]
        self.pending_validated=False
        return self.counts,self.pending,self.sources

    def _loras(self,state,index,source_owners):
        require(state is self.state and source_owners is self.sources and self.pending is not None
                and len(self.pending)==index,'Original block order and owned sources required')
        # The original adapters stay CPU BF16. Explicit Half conversion is
        # required because the original _lora_shard .to(device) preserves dtype.
        result={}
        for name,suffix in self.runtime.LARGE_SUFFIXES.items():
            key=f'diffusion_model.blocks.{index}.{suffix}';record=self.adapters[key]
            a,b=record['A'],record['B']
            require(a.device.type==b.device.type=='cpu' and a.dtype is b.dtype is self.torch.bfloat16,
                    'Original converted CPU BF16 adapter changed')
            result[name]=(b.to(dtype=self.torch.float16),a.to(dtype=self.torch.float16),1.)
        # Retain every Half CPU producer through the original final BOTH-device
        # fence. No ephemeral-pointer cache or detached storage ticket is used.
        source_owners.append(result)
        return result

    def _append(self,state,pending,counts,shard):
        require(state is self.state and pending is self.pending and counts is self.counts
                and not state.shards and len(pending)<50,'No partial generation may be published')
        for name in LORA_KEYS:
            require(len(shard[name])==2,'Exactly two A/B rank tuples required')
            for rank,value in enumerate(shard[name]):
                require(type(value) is tuple and len(value)==3 and value[2]==1.,'Original scale1 A/B tuple required')
                up,down,_=value
                require(up.dtype is down.dtype is self.torch.float16
                        and up.device==down.device==state.devices[rank],
                        'Every A/B shard must be Half on its original rank')
                extra=self.runtime._bytes(up)+self.runtime._bytes(down)
                self.ab_bytes[rank]+=extra;counts[rank]+=extra
        pending.append(shard)
        if self.inject and len(pending)==1:
            require(not state.shards,'CPU injection must precede any publication')
            self.expected_injection_observed=True
            raise InjectedPrepublishFailure('Intentional CPU failure after first pending block, before publication')

    def _validate_pending(self,state,pending,counts,sources):
        require(state is self.state and pending is self.pending and counts is self.counts
                and sources is self.sources and len(pending)==len(sources)==50 and not state.shards,
                'All50 pending blocks and sources must complete before publication')
        require(self.ab_bytes==[AB_BYTES_PER_RANK]*2 and len(state._runtime_owned_backups)==108
                and set(state.patcher.backup)==set(state._runtime_remaining_keys),
                'Exact A/B memory and original108 standard backups required')
        require(all(type(n) is int and n>AB_BYTES_PER_RANK for n in counts),'Truthful base+norm+AB memory accounting required')
        self.pending_validated=True

    def _rollback(self,state,pending,sources):
        require(state is self.state,'Rollback owner changed')
        self.poison=True
        # Original release performs the BOTH-device drain before restoring all
        # original108 backups. Pending shards/sources remain strongly owned.
        try:self.original_release()
        except BaseException:
            self.cleanup_uncertain=True;self.retained=(pending,sources)
            raise
        require(not state.shards and not state._runtime_owned_backups and not state.patcher.backup,
                'Original rollback left resident shards/backups')
        if pending is not None:pending.clear()
        if sources is not None:sources.clear()
        self.pending=None;self.sources=None;self.counts=None;self.phase='poisoned'

    def prepare(self,loaded):
        require(self.lock.acquire(blocking=False),'Concurrent candidate prepare/release is unsupported')
        try:
            require(not self.poison and not self.cleanup_uncertain,'Failed generation cannot be rearmed')
            self.guard()
            if self.state.shards:
                require(self.phase=='ready' and self.current is not None and len(self.state.shards)==50
                        and len(self.state._runtime_owned_backups)==108
                        and tuple(self.state.bytes_per_rank)==tuple(self.current['resident_bytes_per_rank']),
                        'Cached candidate generation is incomplete')
                self.cache_hits+=1;self._event('cache_hit',generation=self.state._runtime_generation)
                return
            require(self.phase=='empty' and self.pending is None,'Fresh complete generation required')
            self.phase='preparing';self.prepares+=1
            try:
                self.derived_prepare(self.state,loaded)
                require(self.pending_validated and self.pending is not None and len(self.pending)==50
                        and not self.state.shards,'Original preparation did not finish unpublished')
                # Its final original _sync already completed both devices.
                # Validate immutable source/owner identities before the final
                # single-GIL publication; state.shards is assigned LAST.
                self.guard()
                record=dict(generation=self.state._runtime_generation,blocks=50,
                            large_original_bases=200,rank_lora_tuples=400,
                            AB_bytes_per_rank=list(self.ab_bytes),resident_bytes_per_rank=list(self.counts))
                self.state.bytes_per_rank=tuple(self.counts)
                self.state.shards=self.pending
                self.current=record;self.records.append(dict(record));self.phase='ready'
                self.pending=None;self.sources.clear();self.sources=None;self.counts=None
                self._event('prepared_published',**record)
            except BaseException:
                self.poison=True
                # The derived function's own exception path rolls back. A
                # guard/publication failure after it returns still needs the
                # same protected rollback before process termination.
                if self.phase!='poisoned' and not self.cleanup_uncertain:
                    self._rollback(self.state,self.pending,self.sources)
                raise
        except BaseException:
            self.poison=True
            raise
        finally:self.lock.release()

    def release(self):
        require(self.lock.acquire(blocking=False),'Concurrent candidate prepare/release is unsupported')
        try:
            require(not self.cleanup_uncertain,'Uncertain cleanup requires process termination')
            if self.poison:
                require(not self.state.shards and not self.state._runtime_owned_backups and not self.state.patcher.backup,
                        'Poisoned generation still owns model resources')
                return
            self.guard()
            self.original_release()
            require(not self.state.shards and not self.state._runtime_owned_backups and not self.state.patcher.backup
                    and tuple(self.state.bytes_per_rank)==(0,0),'Original release failed to restore empty state')
            self.phase='empty';self.current=None;self.releases+=1
            self._event('released_candidate',generation=self.state._runtime_generation)
        except BaseException:
            self.poison=True;raise
        finally:self.lock.release()
