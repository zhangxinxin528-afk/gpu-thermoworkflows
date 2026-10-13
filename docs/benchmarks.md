# Tested environments and measurements

Two evidence categories are kept distinct: prior application runs and tests of
the newly generalized v0.1 tool. A successful historical workflow run does not by
itself test every option in this public adapter.

## Historical GPU4PySCF application, 2026-10-08

RTX 4070 Laptop 8 GB, WSL Ubuntu 24.04; PySCF 2.14.0,
GPU4PySCF 1.8.1, CuPy 13.4.1. Six existing fixed geometries of 24-atom
Ni–sulfate–water inner/outer clusters were compared under DF-UKS/PBE0-D3(BJ),
def2-SVP, water IEF-PCM, grid level 4, complete grid/auxiliary gradient response.

- Maximum absolute energy difference: 2.956e-10 Ha.
- Maximum gradient-component difference: 1.488e-08 Ha/Bohr.
- Three historical CPU/GPU timings: 204.7/58.7, 416.9/85.1, 476.4/103.9 seconds.
- Speed ratios: 3.49, 4.90 and 4.58, including GPU process startup.

CPU timing was taken from previous saved runs, not simultaneous paired timing.
The initial CUDA compilation/setup overhead was excluded from representative warm
timing. A separate Ni Tutton water-fragment batch later completed 186 GPU points.

## Historical CP2K CUDA application, 2026-10-13

CP2K 2026.2 (`git:c92cc08`), GCC 14.4.0, CUDA build, A100-SXM4-80GB.
Fixed 78-atom crystal; PBE-D3(BJ), 1200/100 Ry, EPS_SCF 1e-10.
Flags included `offload_cuda` and `dbcsr_acc`; the GPU grid backend was confirmed.
The earlier application CPU executable's revision differed, so build/revision
information is retained rather than attributing every numeric difference solely
to CPU versus GPU.

Nine selected engine/thread records passed the application's energy/force checks:
[historical_cp2k_points.csv](../benchmarks/historical_cp2k_points.csv).

- Warm cloud point / historical local CPU timing: 2.666-fold speed ratio.
- Same pair, same cloud runtime: serial 133.35 s, concurrent 83.62 s,
  throughput ratio 1.595.
- Concurrent real pending batches:

| Workers | Threads each | Points | Seconds | Points/hour | GPU mean % | Peak GPU memory GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 4 | 124.62 | 115.55 | 65.56 | 4.24 |
| 6 | 2 | 6 | 164.27 | 131.49 | 72.93 | 6.33 |
| 8 | 1 | 8 | 221.57 | 129.98 | 73.24 | 8.43 |
| 12 | 1 | 12 | 304.73 | 141.76 | 76.11 | 12.61 |
| 16 | 1 | 16 | 358.68 | 160.59 | 84.36 | 16.80 |
| 24 | 1 | 24 | 541.42 | 159.58 | 80.39 | 25.18 |

Different concurrency batches used different coordinates. Only the paired test
supports a same-coordinate concurrent comparison. 70 accepted new points were
imported into the research queue. Throughput is a measurement for that system,
hardware and method; choose new resource settings from your own workload.

## Newly generalized tool

Release QA results are written to `validation/release_validation.json` and the
accompanying test log. They include synthetic unit/integration tests, installation
checks and any real engine smoke checks actually executed. Engine smoke tests use
the small water settings, not the 78-atom production budget. Missing backend tests
remain explicitly untested, without carrying over the historical pass status.

Release-candidate checks completed on 2026-10-13:

| Check | Environment | Result |
| --- | --- | --- |
| 15 offline unit/integration tests | Windows, Python 3.12.14, NumPy 2.3.5 | PASS |
| Same 15 tests | WSL Ubuntu 24.04, Python 3.12.3 | PASS |
| Wheel installation and 19-point offline run/analysis | Windows isolated target install | PASS |
| Real fixed-water energy/force | Existing CP2K 2026.2 CPU executable, 2 threads | PASS |
| Real fixed-water SMD energy/force | GPU4PySCF 1.8.1, RTX 4070 Laptop | PASS |

The real GPU smoke check took about 213 seconds in a fresh worker including Python
imports, library discovery, SCF and full gradient response. This is an installation
check, not a warm-throughput benchmark. The real CP2K executable's `--version`
formatter failed; its successful energy/force output identified version 2026.2.
That diagnostic and the executable hash were retained. This new adapter's CUDA
CP2K execution on A100 was not rerun; the A100 table above is historical evidence.
