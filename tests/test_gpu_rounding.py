"""Small synthetic checks; explicit opt-in prevents accidental GPU work."""
import os
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("H3_RUN_GPU_TESTS") != "1":
        pytest.skip("Set H3_RUN_GPU_TESTS=1 to run CUDA qualification checks")
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("Published operators require an SM75 CUDA device")
    return torch


@pytest.mark.parametrize("scale", [1.0, 64.0, 256.0])
@pytest.mark.parametrize("inplace", [False, True])
def test_scale_add_retains_intermediate_half_rounding(gpu, scale, inplace):
    torch = gpu
    from h3_turing.pointwise import scale_add
    gen = torch.Generator(device="cuda").manual_seed(42)
    y = torch.randn(3, 1025, generator=gen, device="cuda", dtype=torch.float16)
    d = torch.randn(y.shape, generator=gen, device="cuda", dtype=torch.float16) * 0.01
    expected = (y.float() + (d.float() * scale).half().float()).half()
    actual, _ = scale_add(y, d, scale, inplace=inplace)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert (actual.data_ptr() == y.data_ptr()) is inplace


@pytest.mark.parametrize("reuse_delta", [False, True])
def test_dequant_pdd_preserves_both_half_boundaries(gpu, reuse_delta):
    torch = gpu
    from h3_turing.experimental.exact_dequant_pdd import finish
    gen = torch.Generator(device="cuda").manual_seed(43)
    acc = torch.randint(-100000, 100000, (17, 257), generator=gen, device="cuda", dtype=torch.int32)
    xs = torch.rand(17, 1, generator=gen, device="cuda") * 0.01
    ws = torch.rand(257, generator=gen, device="cuda") * 0.01
    d = torch.randn(17, 257, generator=gen, device="cuda", dtype=torch.float16) * 0.01
    expected = (((acc.float() * xs) * ws).half().float() + (d.float() * 64).half().float()).half()
    actual, _ = finish(acc, xs, ws, d, 64, out=d if reuse_delta else None)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert (actual.data_ptr() == d.data_ptr()) is reuse_delta
