# Phase 0 — Environment Audit Report

**Project:** BRACE-CPU — Budgeted Risk-Adaptive Computing Engine  
**Date:** 2026-11-05  
**Auditor:** Autonomous research agent

---

## 1. Hardware

| Attribute | Value |
|---|---|
| Operating system | Ubuntu 26.04.1 LTS (Resolute Raccoon) |
| Kernel | Linux 7.0.0-42-generic #42-Ubuntu SMP PREEMPT_DYNAMIC x86_64 |
| CPU model | Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz |
| CPU family / model | 6 / 94 (Skylake) |
| Sockets | 1 |
| Cores per socket | 4 |
| Threads per core | 1 |
| Logical CPUs | 4 |
| NUMA nodes | 1 (node 0: CPUs 0–3) |
| Max CPU frequency | 3600 MHz |
| SIMD features | AVX, AVX2, FMA, SSE4.1, SSE4.2, F16C |
| Virtualization | VT-x |
| Physical RAM | 7.2 GiB (7,503,312 kB total) |
| Available RAM | ~5.3 GiB (MemAvailable) at audit time |
| Free RAM | ~1.5 GiB (MemFree) at audit time |
| Swap | 511 MiB total, ~208 MiB free |
| Disk (project filesystem) | 112 GiB total, 88 GiB available, 20% used |
| Project path | /home/mukesh/Documents/research/BRACE-CPU |
| Project dir writable | Yes (verified) |
| GPU | None |
| CUDA | Not available |

## 2. Software

| Attribute | Value |
|---|---|
| Python version | 3.14.4 (main, Aug 20 2026, GCC 15.2.0) |
| System pip | Not installed (python3-pip not present on base image) |
| ensurepip | Available (pip 25.1.1 wheel) |
| apt-get / apt | Available |
| git | Available |
| gcc / g++ | Not verified as needed (not required for pure Python/NumPy/PyTorch workflow) |

## 3. Virtual environment (isolated, inside project)

| Attribute | Value |
|---|---|
| Location | `/home/mukesh/Documents/research/BRACE-CPU/.venv` |
| Creation method | `python3 -m venv .venv` (bootstrapped via ensurepip) |
| Base Python | 3.14.4 |
| pip inside venv | 25.1.1 |

## 4. Python packages installed (CPU-only)

| Package | Version | Notes |
|---|---|---|
| numpy | 2.5.4 | Numerical arrays |
| scipy | 1.18.1 | Scientific utilities |
| matplotlib | 3.11.2 | Plotting / figures |
| PyYAML | (installed) | Configuration files |
| tqdm | (installed) | Progress bars |
| pytest | (installed) | Automated testing |
| torch | 2.15.0+cpu | CPU-only build from https://download.pytorch.org/whl/cpu |
| torch.cuda.is_available() | False | Confirmed no CUDA |

## 5. Threading / parallelism defaults

| Attribute | Value |
|---|---|
| `os.cpu_count()` | 4 |
| `torch.get_num_threads()` | 4 (default, equals logical CPU count) |
| NumPy/BLAS thread count | Not instrumented (threadpoolctl not installed) |

## 6. Resource constraints and planned limits

Per the project directive (8 GiB RAM, CPU-only):

- Target maximum project-process RAM: **5.5 GiB**.
- Start with **one CPU worker**; increase to two only after measurement.
- Experiments run **sequentially**.
- Small batches; small model architectures (MLP, 10K–100K params).
- float32 for neural models; float64 for numerical reference calculations.
- Avoid swap as a substitute for RAM.
- Monitor memory before and during training.

## 7. Decisions made

- Base system Python left unmodified; all dependencies installed in an isolated project venv.
- CPU-only PyTorch wheel selected explicitly; no CUDA attempted.
- No large dependencies installed without demonstrated necessity.
- `sudo` not used; all operations performed as user `mukesh` in writable project directory.

## 8. Risks / limitations

- 4 logical cores only; CPU-bound numerical solve + neural inference will be the primary bottleneck. Thread counts should be tuned (Phase 20 CPU performance engineering).
- 7.2 GiB total RAM is modest; concurrent model + dataset + solver must be managed to stay under the 5.5 GiB process ceiling.
- No GPU acceleration; long-horizon rollouts and sweeps over many seeds/budgets will take significant wall-clock time. Experiments must be kept small and sequential.

---

*End of Phase 0 environment report.*
