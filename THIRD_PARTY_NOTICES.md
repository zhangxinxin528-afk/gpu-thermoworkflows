# Upstream software and attribution

This release contains our Python workflow code, examples, configuration recipes,
tests and summarized measurements. It does not contain CP2K, GPU4PySCF, NVIDIA
runtime binaries, upstream source archives, model weights or laboratory data.
The MIT license applies to this repository's original workflow code only.

| Dependency / project | Purpose | Upstream |
| --- | --- | --- |
| CP2K | Periodic / molecular energy and forces, native CUDA kernels | https://github.com/cp2k/cp2k |
| GPU4PySCF | GPU molecular SCF, gradients and solvent response | https://github.com/pyscf/gpu4pyscf |
| PySCF | Molecular objects and CPU quantum chemistry interfaces | https://github.com/pyscf/pyscf |
| NumPy | Numerical Hessians and matrix algebra | https://numpy.org/ |
| Grimme low-frequency entropy treatment | qRRHO entropy interpolation | https://doi.org/10.1002/chem.201200497 |

CP2K source headers identify GPL-2.0-or-later; its LICENSE reproduces GPL version 2.
Source: https://github.com/cp2k/cp2k/blob/master/LICENSE . Compilation uses CP2K's
existing CUDA support. This project does not claim authorship of those kernels.

If you separately distribute CP2K executables, include the applicable notices
and GPL license and provide the complete corresponding source for that binary,
including modifications and build scripts, using a GPL-compliant distribution
method. Check the licenses of bundled dependencies as well. The simplest release
format for this repository is workflow source plus a recipe that users execute
against separately obtained upstream software.

When publishing scientific results cite the engine version and method used:

- CP2K: Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020),
  https://doi.org/10.1063/5.0007045 .
- PySCF: Sun et al., *J. Chem. Phys.* **153**, 024109 (2020),
  https://doi.org/10.1063/5.0006074 .
- GPU4PySCF: consult the upstream README's citation section for the methods and
  version used, https://github.com/pyscf/gpu4pyscf .
- qRRHO: Grimme, *Chem. Eur. J.* **18**, 9955–9964 (2012),
  https://doi.org/10.1002/chem.201200497 .

The verified GitHub maintainer account is recorded in CITATION.cff. Publication
author names can be added when supplied. The repository URL is included; a future
archive DOI should only be added after it has actually been assigned.
