# Recorded benchmark machine

| Item | Recorded configuration |
|---|---|
| CPU | 2 × Intel Xeon E5-2686 v4 @ 2.30 GHz |
| CPU topology | 18 cores/socket, 2 threads/core; 36 physical cores / 72 logical CPUs; 2 NUMA nodes |
| System RAM | Nominal 128 GB; Linux MemTotal 131,815,596 KiB (about 125.71 GiB) |
| Primary benchmark GPUs | 2 × NVIDIA GeForce RTX 2080 Ti, Turing SM75 |
| GPU memory | Modified 22 GiB / 22,528 MiB per card; these are not stock 11-GB cards |
| Power limits | Rank 0: 207 W; rank 1: 215 W |
| PCIe per card | Current and maximum: 8.0 GT/s ×16, corresponding to PCIe 3.0 ×16 |
| GPU interconnect | Recorded topology label NV2; peer access in both directions |
| NUMA locality | Both GPUs attached to NUMA node 1; CPU and preferred memory bound to the local NUMA node |

PCIe width/speed came from device link observations in sysfs. NV2 is a topology observation. Neither is a measured application bandwidth figure. No machine names, addresses, GPU UUIDs, device bus addresses or remote paths are included in the configuration export.

The primary software capture records Python 3.12.3, PyTorch 2.13.0+cu130, CUDA build 13.0, NVIDIA driver 580.178.04 and comfy-kitchen 0.2.39. The captured ComfyUI revision is `63be2f9b82ad0dac88a7ceaa28a9cf3154125a18`. These values describe captured runs from October 30–November 2, 2026, not a claim that every exploratory run used an identical environment. Operator source SHA256 checks are the compatibility authority where present.

`results/cmp50.csv` is a **separate secondary-hardware ledger** for an NVIDIA CMP 50HX with 20-GB memory. Its rows are not RTX 2080 Ti measurements. No general hardware comparison is inferred from that exploratory table.

The RTX 3090 reference reported a 24-GB card, the same CPU model/count and 128-GB system RAM. Its PCIe link, power limit and actual Sage `auto` kernel were not recorded in the retained benchmark evidence.

Machine-readable configuration: [hardware.json](../configs/hardware.json).
