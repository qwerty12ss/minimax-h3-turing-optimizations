"""CPU regression: wrapper rebinding must reuse packs, replacements must release them.

This exercises cache ownership without loading H3 or launching CUDA kernels.
Native GPU output equivalence is checked separately with complete red-car jobs.
"""
import gc
import importlib.util
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch
import torch

spec=importlib.util.spec_from_file_location('private_packs',Path(__file__).with_name('packed_shards.py'))
packs=importlib.util.module_from_spec(spec);spec.loader.exec_module(packs)

class Quant:
    def __init__(self,raw):
        self._qdata=raw;self._layout_cls='native-test-layout'
        self._params=types.SimpleNamespace(scale=torch.tensor([0.125]))

def block():
    weight=lambda:types.SimpleNamespace(weight=Quant(torch.ones((2,4),dtype=torch.int8)))
    norm=lambda:types.SimpleNamespace(weight=torch.ones(4))
    return types.SimpleNamespace(adaln_proj=object(),attn=types.SimpleNamespace(qkv_proj=weight(),out_proj=weight(),q_norm=norm(),k_norm=norm()),mlp=types.SimpleNamespace(fc1=weight(),fc2=weight()),norm1=norm(),norm2=norm())

class CacheLifetime(unittest.TestCase):
    def test_rebind_and_replacement_are_bounded(self):
        blocks=[block()for _ in range(50)];calls=[];references=[];stats={};mode=[True]
        def original(b,d0,d1,lora=None):calls.append((b,d0,d1));return {'qkv':(None,None),'marker':b.adaln_proj}
        class TinyPack:
            def __init__(self,shard,rank):
                self.host=torch.empty(16,dtype=torch.uint8);references.append(weakref.ref(self))
            def to(self,device):return {'qkv':torch.ones(2)}
        state=types.SimpleNamespace(_runtime_core=types.SimpleNamespace(shard_block=original),_h3_public_pdd_owner=object(),patcher=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(blocks=blocks))))
        with patch.object(packs,'RankPack',TinyPack):
            packs.install(state,lambda:stats,lambda:mode[0]);shard=state._runtime_core.shard_block
            for b in blocks:shard(b,'cpu','cpu')
            self.assertEqual(len(calls),50);self.assertEqual(stats['retained_pack_bytes'],1600)
            # Offload may create new Qt wrappers around exactly the same storage.
            for b in blocks:
                for projection in [b.attn.qkv_proj,b.attn.out_proj,b.mlp.fc1,b.mlp.fc2]:projection.weight=Quant(projection.weight._qdata)
                b.norm1.weight=b.norm1.weight.clone()
                shard(b,'cpu','cpu')
            self.assertEqual(len(calls),50);self.assertEqual(stats['pack_hits'],50)
            # Real content changes require replacement. Keep only one generation.
            for generation in range(3):
                for b in blocks:
                    b.mlp.fc2.weight=Quant(torch.full((2,4),generation+2,dtype=torch.int8))
                    shard(b,'cpu','cpu')
                gc.collect()
                self.assertEqual(sum(r()is not None for r in references),100)
                self.assertEqual(stats['retained_pack_blocks'],50)
                self.assertEqual(stats['retained_pack_bytes'],1600)
            # In-place changes to scale/norm values invalidate cached bytes too.
            before=len(calls);blocks[0].norm1.weight[0]=2;shard(blocks[0],'cpu','cpu')
            blocks[0].mlp.fc2.weight._params.scale[0]=0.25;shard(blocks[0],'cpu','cpu')
            self.assertEqual(len(calls),before+2)
            mode[0]=False;shard(blocks[0],'cpu','cpu');self.assertEqual(len(calls),before+3)

if __name__=='__main__':unittest.main()
