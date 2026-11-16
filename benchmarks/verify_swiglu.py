"""Bit comparisons include signed zero and NaN payloads; timing follows equality."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import comfy_kitchen.backends.cuda as ck

from h3_turing.swiglu_convrot.joint import Tables, prepare, ck_functions

HERE = Path('benchmark-output')
HERE.mkdir(exist_ok=True)
quant, _, proof = ck_functions()
report = dict(state='running', proof=proof, devices=[])


def bits(a, b):
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def metric(a, b):
    dtype = {torch.float16: torch.int16, torch.float32: torch.int32, torch.int8: torch.int8}[a.dtype]
    mismatches = int((a.view(dtype) != b.view(dtype)).sum())
    return dict(bitwise=mismatches == 0, mismatches=mismatches, elements=a.numel())


def original(g):
    gate, up = g.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float() / 256.).half()


def check_device(rank):
    with torch.cuda.device(rank):
        bank = Tables()
        weight = torch.empty((5376, 7168), dtype=torch.int8, device=rank)
        ws = torch.ones((5376,), dtype=torch.float32, device=rank)
        reference = lambda a: quant(a, weight, ws, out_dtype=torch.float16, convrot=True, convrot_groupsize=256)
        rows = []
        for mode in ('default', 'nondefault'):
            stream = torch.cuda.default_stream(rank) if mode == 'default' else torch.cuda.Stream(rank)
            with torch.cuda.stream(stream):
                cases = [('random', m, scale) for m in (8192, 4650, 3034) for scale in (.125, 8.)]
                cases += [('allbits', 10, rotation) for rotation in (0, 17, 32769)]
                cases += [('zero', 17, 0), ('subnormal', 17, 0)]
                for kind, m, setting in cases:
                    torch.manual_seed(2026110107 + rank)
                    if kind == 'random':
                        g = (torch.randn((m, 14336), device=rank) * setting).half()
                    elif kind == 'allbits':
                        index = torch.arange(m * 7168, device=rank, dtype=torch.int32)
                        gate = (index % 65536).to(torch.int16).view(torch.float16)
                        up = ((index + setting) % 65536).to(torch.int16).view(torch.float16)
                        g = torch.cat((gate.reshape(m, 7168), up.reshape(m, 7168)), dim=1)
                        del index, gate, up
                    else:
                        g = torch.zeros((m, 14336), device=rank, dtype=torch.float16)
                        if kind == 'subnormal':
                            g.view(torch.int16)[:, ::2] = 1
                            g.view(torch.int16)[:, 1::2] = -32767
                    frozen = g.clone()
                    a0 = original(g)
                    q0, s0 = reference(a0)
                    a1 = original(g)
                    q1, s1 = reference(a1)
                    row = dict(mode=mode, kind=kind, M=m, setting=setting,
                        reference_repeat=[bits(a0, a1), bits(q0, q1), bits(s0, s1)], variants=[])
                    del a1, q1, s1
                    for warps in (4, 8):
                        a, q, s = prepare(g, bank, warps)
                        result = dict(warps=warps, activation=metric(a, a0), q=metric(q, q0), scales=metric(s, s0))
                        a2, q2, s2 = prepare(g, bank, warps)
                        result['repeat'] = [bits(a, a2), bits(q, q2), bits(s, s2)]
                        result['passed'] = all(result[n]['bitwise'] for n in ('activation', 'q', 'scales')) and all(result['repeat'])
                        row['variants'].append(result)
                        del a, q, s, a2, q2, s2
                    row['input_unchanged'] = bits(frozen, g)
                    row['passed'] = all(row['reference_repeat']) and row['input_unchanged'] and all(v['passed'] for v in row['variants'])
                    rows.append(row)
                    print(json.dumps(dict(rank=rank, **row)), flush=True)
                    del g, frozen, a0, q0, s0
        torch.cuda.synchronize(rank)
        return dict(rank=rank, name=torch.cuda.get_device_name(rank), cases=rows,
                    table_bytes=sum(t.numel() * t.element_size() for t, e in bank.tables.values()),
                    passed=all(r['passed'] for r in rows))


with ThreadPoolExecutor(max_workers=2) as pool:
    report['devices'] = list(pool.map(check_device, (0, 1)))
report['state'] = 'completed'
report['passed'] = all(r['passed'] for r in report['devices'])
(HERE / 'operator-results.json').write_text(json.dumps(report, indent=2) + '\n')
assert report['passed'], 'Inspect byte-level operator comparisons before block tests'
