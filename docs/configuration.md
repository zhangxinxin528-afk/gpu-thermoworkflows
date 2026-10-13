# Configuration interface, schema version 1

JSON is read with UTF-8 or UTF-8 BOM. Relative paths are resolved from the config
file, not the shell working directory. Environment-variable paths must be set
before preparing a queue. No command is run through a shell.

## Required fields

`schema_version: 1`, a simple `name`, `engine: mock|cp2k|gpu4pyscf`,
`paths.work_dir`, `paths.structure`, `calculation.displacement.value` and
`calculation.displacement.unit: bohr|angstrom`, and a `method` object.

XYZ input is one frame: atom count, comment, then element and three angstrom
coordinates. A changed XYZ file or input template is a changed scientific input.

## Calculation

| Key | Default / meaning |
| --- | --- |
| `charge` | Integer, default 0 |
| `multiplicity` | Positive integer, default 1 |
| `periodic` | false; GPU molecular adapter requires false |
| `cell_A` | Three Cartesian cell vectors in Å; required if periodic |
| `hessian` | true; false gives a single center force point |
| `geometry_optimization` | false; true is rejected |
| `displacement` | Positive value, explicit unit; ± on each Cartesian coordinate |

Full Hessian: 1 center plus 6N displaced points. Every point starts from the
specified structure. This release does not perform symmetry reduction.

## Method

CP2K methods are template parameters. The water template uses `xc`, `cutoff_Ry`,
`rel_cutoff_Ry`, `ngrids`, `eps_scf`, `max_scf`, `poisson_solver`, `basis_file`,
`potential_file`, and `kinds`. Each element in `kinds` has `basis` and `potential`,
with optional `magnetization`. Templates substitute `{{COORD}}`, `{{CELL}}`,
`{{KINDS}}`, `{{CHARGE}}`, `{{MULTIPLICITY}}`, `{{UKS}}`, `{{PERIODIC}}`, plus scalar
method keys. Templates must specify `RUN_TYPE ENERGY_FORCE` and contain no MD,
geometry/cell optimization or band-path execution. All parameters and source
files are fingerprinted. Provide a matching `PERIODIC` Poisson solver for crystals.

GPU4PySCF methods:

| Key | Meaning / default |
| --- | --- |
| `basis`, `xc` | Required PySCF basis / functional names |
| `dispersion` | e.g. d3bj; null for none |
| `density_fit`, `auxbasis` | true; optional auxiliary basis |
| `grid_level` | 4 |
| `conv_tol`, `conv_tol_grad`, `max_cycle` | 1e-10, 1e-7, 120 |
| `solvent`, `solvation` | null for gas; solvent name with SMD or PCM |
| `dielectric`, `pcm_method` | Required dielectric for PCM; IEF-PCM method default |
| `grid_response`, `auxbasis_response` | true, true |

RKS is selected for singlets, UKS for open shells. The spin expectation is
recorded for UKS. SCF convergence is required. The tool does not assert a
universal spin-purity criterion for different chemistries.

## Resources and execution

`resources.workers` (1), `threads_per_worker` (1), `memory_mb_per_worker` (1500),
`min_available_memory_mb` (0), `scratch_dir` (OS temporary directory),
`timeout_seconds` (600), `gpu_device` (0), `min_gpu_free_memory_mb` (0).
The GPU adapter uses one selected GPU and can share it among persistent workers;
choose concurrency from actual memory and throughput tests. CP2K accepts its MPI
launch prefix as `execution.cp2k_command`, e.g. `["mpiexec", "-n", "2", "cp2k.psmp"]`.
Version probes for complex MPI wrappers may require a direct executable wrapper.
If an upstream executable fails while formatting `--version`, the error is
retained and its SHA256 still pins the runtime. Scientific acceptance continues
to require successful input checks, SCF, normal termination and a full force array.

`execution.input_check` defaults to true. `max_tasks_per_run` defaults to null.
The CLI `--max-tasks N` overrides the budget for that invocation. Resource fields
are operational and can change at resume; physical settings and engine identities
cannot silently change inside the same work directory.

CP2K timeout covers each subprocess. GPU4PySCF cannot safely cancel an in-process
CUDA kernel; v0.1 uses its SCF max-cycle limit. `timeout_seconds` is not a hard GPU
wall-time limit. A cooperative pause finishes the active GPU point.
CUDA libraries installed as NVIDIA pip wheels are discovered inside the active
Python environment and added to spawned workers' loader path. The old
GPU4PySCF cuSOLVER lookup is resolved from that environment when necessary.
No global CUDA installation or running external task is modified.
`memory_mb_per_worker` sets PySCF's working-memory parameter; it is not an OS-enforced
CP2K memory cap. The host-memory minimum is checked before each point.

## Analysis

`temperature_K` (298.15), `symmetry_number` (1), `pressure_Pa` (100000),
`standard_state` (`1bar` or molecular `1M`), `formula_units_per_cell` (1),
`qrrho_cutoffs_cm_1` ([50,100]), `include_spin_entropy` (false),
`imaginary_tolerance_cm_1` (5), `zero_tolerance_cm_1` (1e-5),
`max_Hessian_antisymmetry` (0.005), `atomic_masses_amu` (optional element overrides).

Masses for common elements are supplied. Provide explicit isotope or other
element masses when needed. Translations/rotations are removed by rank-aware
mass-weighted projection. Signed imaginary frequencies are retained. Significant
imaginary modes suppress thermal estimates; small nonpositive modes excluded from
an estimate are listed explicitly. A tolerance is a reporting choice, not a
minimum-geometry proof. Periodic cells use a Gamma-only harmonic Helmholtz term
divided by the chosen formula-unit count; there is no concentration correction.

Molecular RRHO includes translation/rotation/vibration at the chosen pressure,
plus explicit optional 1 M conversion. qRRHO modifies only the vibrational entropy.
Solvent contributions come from the chosen electronic model. A gas-style RRHO
correction on a continuum-solvent configuration is a model choice, not a solution
ensemble free energy. `analyze --temperature K` reuses forces and changes only the
thermal evaluation, without new SCF calls.
