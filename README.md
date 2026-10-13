# GPU Thermoworkflows

[![Workflow tests](https://github.com/zhangxinxin528-afk/gpu-thermoworkflows/actions/workflows/tests.yml/badge.svg)](https://github.com/zhangxinxin528-afk/gpu-thermoworkflows/actions/workflows/tests.yml)

A small, resumable workflow for **fixed-geometry energy and force calculations**
with separately installed CP2K and GPU4PySCF, followed by numerical Hessians and
harmonic thermal corrections. Developed from the Ni Tutton salt research workflow.

Our contribution is configuration, execution, checkpointed queue management,
provenance checks and postprocessing. CUDA electronic-structure kernels are
provided by the upstream simulation engines. Version 0.1.0 includes JSON configuration,
three examples, a CLI, tested-environment records and CUDA build recipes.

中文说明: [README.zh-CN.md](README.zh-CN.md).

## What you can use it for

- Generate and run molecular or crystal Cartesian force-point batches from a fixed XYZ structure.
- Pause long queues, resume verified points, and retain original attempts when a point fails.
- Compare identical-input CPU/GPU engine results before reusing a new build.
- Assemble numerical Hessians, molecular frequencies or Gamma-cell modes, and RRHO/qRRHO thermal estimates.
- Reproduce the documented CP2K CUDA build and adapt resource settings to your hardware.

The execution layer can be reused for other structures through the configuration
file. Select the appropriate electronic method, species and thermodynamic model for
your chemistry. The small examples demonstrate the interface; the measured Ni Tutton
application performance is documented separately in [benchmarks](docs/benchmarks.md).

## Install and verify in a minute

Python 3.10+ and NumPy are sufficient for the offline example:

```sh
git clone https://github.com/zhangxinxin528-afk/gpu-thermoworkflows.git
cd gpu-thermoworkflows
python -m venv .venv
```

Activate the environment: Linux/WSL `source .venv/bin/activate`; Windows
PowerShell `.\.venv\Scripts\Activate.ps1`. Then install and run:

```sh
python -m pip install .
gtw --version
gtw doctor
gtw prepare examples/water/mock.json
gtw run examples/water/mock.json
gtw analyze examples/water/mock.json
```

The offline water example executes 19 **synthetic harmonic-bond force points**.
Expect `COMPLETE`, 3 internal vibrational modes, and a JSON thermal summary.
It checks installation and the pipeline; it is not a DFT result or water spectrum.
Use `python -m gpu_thermoworkflows` in place of `gtw` if needed.
The [v0.1.0 release](https://github.com/zhangxinxin528-afk/gpu-thermoworkflows/releases/tag/v0.1.0)
also provides an installable wheel and a source bundle containing these examples.

For a quick checkpoint/resume demonstration, use a fresh work directory:

```sh
gtw run examples/water/checkpoint.json --max-tasks 3
gtw status examples/water/checkpoint.json
gtw resume examples/water/checkpoint.json
```

The temporary budget stops after at most 3 new points. Completed points are reused
only after their scientific fingerprint, coordinates and artifact hashes agree.

## Real calculation smoke tests

**CP2K** must already be installed. Put `cp2k.psmp` on PATH and set its data directory:

```sh
export CP2K_DATA_DIR=/path/to/cp2k/share/cp2k/data
gtw doctor --config examples/water/cp2k.json
gtw run examples/water/cp2k.json
```

PowerShell environment equivalent: `$env:CP2K_DATA_DIR = '/path/to/data'`.
Run the real Linux engines from Linux or WSL, not a Windows Python process.
The CP2K example calculates one fixed PBE molecular energy/force point, using
300/40 Ry and a 14 Å MT box for a small installation test. Those settings are not
the high-cutoff Ni Tutton production settings.

**GPU4PySCF** requires an NVIDIA driver and a compatible CUDA Python environment:

```sh
python -m pip install -r environment/gpu-cu12-tested.txt
python -m pip install .
gtw doctor --config examples/water/gpu4pyscf.json
gtw run examples/water/gpu4pyscf.json
```

The real GPU example runs a fixed PBE-D3(BJ)/def2-SVP water point with SMD water.
The pinned recipe includes NVIDIA runtime wheels; a full historical package
snapshot is in `environment/gpu-cu12-lock.txt`.
`doctor` is an inventory probe; a successful `run` confirms SCF and force execution.
Choose versions appropriate to your driver using the upstream GPU4PySCF guide:
https://github.com/pyscf/gpu4pyscf .

## One configuration, one set of scientific inputs

Paths are relative to the JSON configuration file. Environment variables such as
`${CP2K_DATA_DIR}` are supported. Settings are grouped as follows:

| Group | Configure here |
| --- | --- |
| `paths` | XYZ structure, output directory, CP2K input template |
| `calculation` | Charge, multiplicity, periodic cell, displacement value/unit, full Hessian or center only |
| `method` | Functional, basis/potentials, cutoff, SCF limits, solvation and gradient response |
| `resources` | Worker count, threads, memory minimum, scratch directory, GPU device, timeout |
| `execution` | CP2K argv list, input check, new-point budget |
| `analysis` | Temperature, rigid-mode projection, molecular symmetry, standard state, formula units, low-frequency sensitivity |

The runtime validates the JSON fields it uses. See [configuration.md](docs/configuration.md)
for the full interface. A single XYZ frame is used; coordinates remain fixed except
for prescribed finite differences. There is no geometry or cell optimizer.

## Commands

```text
gtw doctor [--config FILE]                 dependency / device inventory
gtw prepare FILE                         snapshot inputs and generate force jobs
gtw run FILE [--max-tasks N]              execute a new queue in the foreground
gtw pause FILE                           stop submissions; finish active points
gtw resume FILE [--max-tasks N]           verify and continue the saved queue
gtw status FILE                          inspect saved state and actual supervisor
gtw analyze FILE [--temperature K]        Hessian, frequencies and thermal estimates
gtw compare REFERENCE.json CANDIDATE.json same-input force consistency check
```

Call `pause` from a second terminal. Keep the first terminal open until `PAUSED`.
This is a cooperative pause at force-point boundaries; a long current SCF may need
time to finish. A crashed point is preserved as an attempt and rerun on resume;
v0.1 does not automatically project an unfinished wavefunction into a new task.
Existing CP2K point checkpoints remain in the raw attempt directory.

An OS-held run-directory lock prevents simultaneous supervisors on one host.
The status probe checks PID creation identity (and Linux boot identity), so an old
`RUNNING` record alone does not prove that a calculation is alive. A changed
physical configuration or engine version requires a new output directory.

## Outputs and scientific scope

```text
runs/CASE/
  manifest.json             scientific inputs and SHA256 fingerprint
  config_snapshot.json      resolved configuration
  runtime.json              engine build/version identity
  status.json               queue state and supervisor identity
  tasks/ID/attempts/.../    retained input, raw logs, failure/launch records
  tasks/ID/result.json      energy (Ha), forces (Ha/Bohr), coordinates (Å)
  tasks/ID/result.sha256     saved result integrity
  analysis/                 raw Hessian, signed frequencies and thermal summary
```

For molecular cases, translation and rotation are projected out. For periodic
cells, only three translations are removed; v0.1 reports a Gamma-cell harmonic
estimate, not Brillouin-zone integrated phonon thermodynamics. Imaginary modes and
Hessian asymmetry are reported directly. Significant imaginary modes prevent a
formal thermal correction; the program does not re-optimize the structure.

RRHO and optional 50/100 cm⁻¹ Grimme entropy interpolation are reported. The
interpolation changes entropy only. Molecular 1 bar and 1 M standard-state
corrections are explicit; symmetry numbers must be supplied by the user. Electronic
spin entropy defaults to off and can be enabled deliberately.

This release produces fixed-configuration thermal estimates. A salt solubility
prediction additionally needs a balanced thermodynamic cycle, consistent solid and
aqueous standard states, species definitions and an activity model. The project-specific
Ni Tutton dissolution/activity model and laboratory data are not bundled here.

## Build CP2K CUDA

[docs/cuda-build.md](docs/cuda-build.md) contains a configurable build recipe and
the tested upstream source version. No engine binaries or CUDA libraries are
distributed in this repository. [docs/benchmarks.md](docs/benchmarks.md) distinguishes
historical application measurements from tests of this new public adapter.

## Tests, license and citation

```sh
python -m unittest discover -s tests -v
```

Tests cover queue resume/pause, hash mismatches, retained failed attempts, CP2K
parsing, input rendering and analytically known harmonic modes. The real engine
smoke checks are separate from synthetic unit tests.

Original workflow code: [MIT](LICENSE). Upstream engines keep their own licenses:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite the released workflow version
using [CITATION.cff](CITATION.cff), and cite the simulation engines and methods used.
The GitHub maintainer is recorded in the citation metadata. Cite a tagged version;
no Zenodo DOI has been assigned to this release.

The local source release is packaged with `python tools/package_release.py` after
QA is recorded. The wheel installs the command-line tool; use the source archive
for examples, build recipes and documentation.
