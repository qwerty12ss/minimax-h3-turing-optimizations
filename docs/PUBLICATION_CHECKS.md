# Publication review: November 16, 2026

The release is an explicit file allowlist exported into a separate repository with a new root commit. Original Git branches, tags, history and Git LFS objects were not imported. Commit attribution uses the owner's GitHub noreply address.

Included material is limited to reviewed code, synthetic/operator tests, build and analysis tools, benchmark parameters, numeric result ledgers, documentation and required dependency license texts. No conversation exports, creative prompts, input/output media, project databases, remote inventories, SSH configuration, environment files, model weights or native binaries are included.

The content review checked file types and UTF-8 contents; emails; machine/GPU identifiers; network addresses; local/remote deployment paths; unrelated project references; and suspicious authentication material. Hardware specifications intentionally retained in `configs/hardware.json` are anonymized benchmark parameters.

Gitleaks 8.30.1 was run with the full built-in rule set, 100% report redaction and inline allow directives disabled, against the release files and the new Git history before publication. No secret findings were reported. This is a recorded review result, not a guarantee against every possible undiscovered secret.

Validation on the publication workstation:

- CPU/metadata suite: **10 passed**.
- Synthetic GPU suite: **8 skipped**, because qualification requires explicit opt-in and the target CUDA/Triton SM75 environment.
- Public result summarizer: reproduced the three bracketed full-job summaries from CSV.
- Python wheel: built successfully without downloading GPU dependencies; native source/include files are present.
- GPU compilation, historical video reproduction and performance remeasurement were **not performed** for this release. Published numerical results retain their original measurement scope and dates.

The original research repository remains separate and private. Subsequent contributions need their own content/secret review; this review covers the initial curated release.

The subsequent standalone deployment adds source-pinned TP2/PDD integration, an isolated installer, public model URLs/checksums and synthetic workflows. Its fresh installation, GPU checks, video/audio execution and additional publication scan are recorded in [DEPLOYMENT_VALIDATION.md](DEPLOYMENT_VALIDATION.md). The original release's test counts above remain a historical receipt.

The November 17 optional [Luna-derived recipe](RECIPE_LUNA.md) adds explicitly selected compute/test/license files and a sanitized numerical ledger. Its measured AWQ, DiT-pack and FC2 sources are byte-identical to the completed experiment; the public wrapper removes raw diagnostic dumps and exposes scalar statistics. Local validation: **31 CPU tests passed**, **8 GPU tests skipped**, and the separate cache-lifetime regression passed. The newly packaged wrapper was not rerun end to end on GPUs; historical timings are labeled accordingly. File-content and Git-history secret scans are repeated before publication. No original input, creative prompt, generated media, raw experiment history, connection settings or machine identifiers are included.
