import random

import pytest

from h3_turing.norm_mod.contract import CONFIGURATIONS, validate_configuration
from h3_turing.norm_mod.row_map import describe_segments
from h3_turing.rope.contract import CHUNKS, H, S, D, cases, metric, ptx_proof
from h3_turing.vae.spatial import Group, make_groups, partition_groups


class Tiles:
    def split_tiles(self, length):
        starts = list(range(0, length, 8))
        return starts, [min(8, length - n) for n in starts], None


def test_grouping_preserves_owner_windows_and_order():
    groups = make_groups(Tiles(), 17, 41, batch=4)
    assert len(groups) == 6
    for row in range(3):
        selected = [g for g in groups if g.row == row]
        assert [g.cost for g in selected] == [4, 2]
        assert [w for g in selected for w in g.x_windows] == [(x, min(8, 41 - x)) for x in range(0, 41, 8)]
        assert {g.y_start for g in selected} == {row * 8}
    with pytest.raises(ValueError):
        make_groups(Tiles(), 17, 41, batch=0)


def test_partition_has_globally_minimal_peak_load():
    rng = random.Random(111)
    for n in range(1, 9):
        for _ in range(8):
            costs = [rng.randrange(1, 10) for _ in range(n)]
            groups = tuple(Group(0, i, 0, 1, ((0, 1),) * c) for i, c in enumerate(costs))
            result = partition_groups(groups)
            assert result[0] == 0 and len(result) == n and set(result) <= {0, 1}
            loads = [sum(c for c, rank in zip(costs, result) if rank == r) for r in (0, 1)]
            reachable = {costs[0]}
            for c in costs[1:]:
                reachable |= {s + c for s in reachable}
            assert max(loads) == min(max(s, sum(costs) - s) for s in reachable)
            assert sum(loads) == sum(costs)
            assert result == partition_groups(groups)
    for invalid in ((), (Group(0, 0, 0, 1, ((0, 1),)),) * 21):
        with pytest.raises(RuntimeError):
            partition_groups(invalid)


class DummyTensor:
    ndim = 1
    dtype = "int64"
    device = "cpu"
    def __init__(self, n):
        self.shape = (n,)
    def stride(self):
        return (1,)


def test_modulation_exact_cover_and_negative_rows():
    desc = describe_segments([(0, 2, -4), (2, 5, DummyTensor(3))], 5, 4, DummyTensor)
    assert desc[0]["row"] == -4 and desc[1]["shape"] == [3]
    for segments in [[(1, 5, 0)], [(0, 2, 0), (1, 5, 0)], [(0, 4, 0)],
                     [(0, 5, 4)], [(0, 5, -5)], [(0, 5, DummyTensor(4))], [(0, 5, True)]]:
        with pytest.raises(RuntimeError):
            describe_segments(segments, 5, 4, DummyTensor)


def test_launch_contract_and_complete_rope_fixture_plan():
    for block, warps in CONFIGURATIONS:
        validate_configuration(block, warps)
    with pytest.raises(RuntimeError):
        validate_configuration(4096, 16)
    assert sum(count for _, count in CHUNKS) == S
    assert all(start + count == next_start for (start, count), (next_start, _) in zip(CHUNKS, CHUNKS[1:]))
    plan = cases()
    assert len(plan) == 76 and len({p["id"] for p in plan}) == 76
    assert {p["fixture"] for p in plan} >= {"all_half_codes", "rf_signedzero_edges", "float32_payloads"}
    good = dict(elements=H * S * D, mismatches=0, bitwise_equal=True, signed_zero_aware=True, nan_payload_aware=True)
    metric(good)
    for key in ("signed_zero_aware", "nan_payload_aware", "bitwise_equal"):
        with pytest.raises(ValueError):
            metric({**good, key: False})
    with pytest.raises(ValueError):
        ptx_proof({"target_sm75": True, "non_ftz": True})
