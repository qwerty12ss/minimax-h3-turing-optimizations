# Measurement definitions and limits

The primary full-job metric is ComfyUI `execution_start` to `execution_success`. It includes more than sampling. Do not substitute the sampler stage time or API polling elapsed time for full-job latency. Rows labeled `cold`, `diagnostic_replay` or `measurement_class=diagnostic_replay` are not ordinary hot measurements.

The 8-second anchor is 960 × 512, 192 output frames at 24 fps, Euler / BasicGuider, eight PDD steps, video sigma shift 12, audio sigma shift 3 and PDD strengths 1. Model filenames and attention step routing are in [anchor_8s.json](../configs/anchor_8s.json). Later qualified profiles use a separate FP16 low-rank PDD branch. Exploratory baked/runtime-merge arms remain distinct.

## Bracketed full-job experiments

| Result file | Public measurement rows | Seed | Estimator |
|---|---|---:|---|
| `shared_pipeline.json` | `followups.csv`, series `next-shared-confirm-20261101` | 2026110116 | Candidate median of 2 hot jobs; control is the mean of before/after arm medians, each from 2 hot jobs |
| `temporal_vae.json` | `followups.csv`, series `next-selected-initialized-20261101` | 2026110123 | Same bracket estimator |
| `swiglu_full.json` | `swiglu_full.csv` | 2026110123 | Candidate median of 2 hot jobs; control is the pooled median of all 4 control hot jobs |

Each experiment contains three arms: control-before, candidate, control-after. Each arm has one cold and two hot jobs. Percentage time reduction is `100 * (1 - candidate / control)`. `python tools/summarize_results.py` recomputes the three headline results directly from the CSV rows. Different controls, seeds and histories prevent adding their percentage reductions.

The shared pipeline saves about 3.047 s / 1.813% on its own bracket. The selected temporal VAE profile saves about 1.688 s / 1.021% full-job time; its video decode stage falls by about 9.02%. Isolated VAE tests on longer/resized tensors reused or cropped a source latent and do not constitute independent-scene quality evidence.

The joint SwiGLU/ConvRot profile saves 3.6815 s / 2.253% full-job time in its bracket. Block-only measurements show about 5.356% reduction at 29,226 rows and 4.425% at 60,378 rows; those percentages are **block timings**, not full-job speedups. Its SiLU lookup table costs 256 KiB per GPU. The captured full-job maximum reserved-memory difference is about 0.449 GiB on rank 0; allocated state also includes the table.

## Accuracy evidence

The joint preparation operator ledger contains 44 logical cases on each of two GPUs (88 executed variants). It checks Half activation bits and quantization/scales, including NaNs, signed zeros and edge fixtures. Full-job SwiGLU testing records 99/99 raw descriptor checks passing across nine jobs, plus successful complete media decoding and identical decoded audio/video for that particular case. The media and original input tensors are not distributed, so this is retained historical evidence rather than a new verification of the packaged build.

The shared and temporal ledgers explicitly contain `all_99_fingerprints_match=false`. They preserve complete fingerprint groups and candidate matches to complete control outputs. The temporal series has a cold-control outlier. Matching one valid complete control group does not establish that all baseline repeats were identical, and these files must not be summarized as 99/99 globally equal.

Operator L2/RMSE comparisons against high-precision recomputation refer to the **already INT8-quantized weights**, not the original BF16 model weights. A relative L2 change is not a percentage change in perceptual video quality.

## Experimental status

Exact dequant/PDD fusion preserved both original Half boundaries in real captures and rounding-edge tests, but block improvements of roughly 0.539% / 0.979% missed the 1% promotion gate. No end-to-end benefit is claimed.

Head-dependent attention thresholds improved the calibration group but failed the independent acceptance gate; one exact-pair aggregate was slightly worse. Quantizer clipping also failed generalization: an independent-scene aggregate L2 increased about 0.84%, with a roughly 22.62% regression at one tested point. These sources are retained under `experimental/` for research, not enabled as defaults.

## Other ledgers

`campaign2.csv` retains 58 multi-arm exploratory/qualification observations; `accuracy_night.csv` retains 66 observations; `followups.csv` has 32 rows including 5 diagnostic replays; `pdd_operators.csv` has 84 operator measurements; `cmp50.csv` has 42 secondary-hardware observations. Row counts, allowed columns and original-table hashes are in [table_manifest.json](../results/table_manifest.json). Scene labels in those tables are test identifiers, without original prompts or assets. These ledgers are not a single controlled experiment.
