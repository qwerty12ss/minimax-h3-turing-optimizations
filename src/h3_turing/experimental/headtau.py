"""Reuse the async prepare function and replace only its versioned fused export."""
import ctypes
from pathlib import Path
import types

import torch


class Routing:
    def __init__(self, native, path=None):
        self.native = native
        self.original_prepare = native.prepare
        self.library = ctypes.CDLL(str(path or Path(__file__).with_name('libheadtau_v1.so')))
        self.export = self.library.star7_sol_sm75_prepare_quantized_routes_headtau_v1
        self.export.argtypes = ([ctypes.c_uint64] * 15 + [ctypes.c_int64] * 9 + [ctypes.c_int] * 4
                                + [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_uint64] * 2)
        self.export.restype = ctypes.c_int

    def prepare(self, q, k, v, *, tau_heads, tau=1., **kwargs):
        assert tau_heads.dtype == torch.float32 and tau_heads.device == q.device
        assert tau_heads.is_contiguous() and tau_heads.shape == (q.shape[1],)
        native = self.native
        library = native._load_library()
        export = self.export

        class Proxy:
            def star7_sol_sm75_prepare_quantized_routes(self, *args):
                return export(*args, tau_heads.data_ptr())

            def __getattr__(self, name):
                return getattr(library, name)

        original = self.original_prepare
        function = types.FunctionType(original.__code__,
            dict(original.__globals__, _load_library=lambda: Proxy()),
            original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        return function(q, k, v, tau=tau, **kwargs)
