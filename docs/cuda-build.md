# Reproducible CP2K CUDA build recipe

Upstream instructions: https://manual.cp2k.org/trunk/technologies/accelerators/cuda.html
and https://github.com/cp2k/cp2k-containers .

Historical build: official CP2K **2026.2** release archive,
https://github.com/cp2k/cp2k/releases/download/v2026.2/cp2k-2026.2.tar.bz2 .
Archive SHA256:
`f9bd86f580f57a53a0768c0045d1417f9f9a1d66d851ed7f662f496200043373`.
The build used GCC 14.4.0, CUDA toolkit 12.9, CMake/Ninja and numerical libraries
from an existing compatible CP2K environment. Both compute capabilities 80 and
89 were selected. A100 execution was tested; that alone does not validate this
CP2K binary's performance on the local RTX 4070.

## Dependencies

The `environment/cuda-build-tools.yml` recipe supplies compilers/build tools/CUDA
headers and libraries. It is not the entire CP2K dependency stack. Also supply
compatible MPI, DBCSR, BLAS/LAPACK/ScaLAPACK, FFTW, LibXC, LIBXSMM/LIBXS as required
by your build. Use CP2K's toolchain or a compatible package-manager environment.
An NVIDIA kernel driver is supplied by the host and is not bundled here.

Obtain and unpack the upstream source yourself. Verify the archive hash. Build
outside its source tree; use a separate install prefix.

```sh
python tools/build_cp2k_cuda.py \
  --source /path/to/cp2k-2026.2 \
  --build /path/to/cp2k-cuda-build \
  --install-prefix /path/to/cp2k-cuda-install \
  --cuda-root /path/to/cuda-toolkit \
  --dependency-prefix /path/to/cp2k-dependencies \
  --architectures '80;89' --jobs 4 --dry-run
```

Inspect the printed argv lists, then remove `--dry-run` to build. If using conda
compilers, specify `--cc`, `--cxx`, `--fc` explicitly and activate that toolchain.
`--cmake` accepts the desired CMake executable. `--install` adds an install step.
Build logs and per-stage SHA256 values are retained in the build directory.

After compilation:

1. Inspect `cp2k.psmp --version` for version and CUDA flags.
2. Run the appropriate upstream regression checks for the modules you use.
3. Run the small fixed-water smoke example.
4. Compare a CPU build and CUDA build on identical physical inputs using `gtw compare`.
5. Measure throughput over representative force points; GPU utilization is not
   a substitute for wall-time and numerical agreement.

The repository supplies a build recipe, not a standalone executable. If an engine
binary is added to a public release, supply its GPL-compliant corresponding source,
build scripts and licenses for redistributed dependencies; see THIRD_PARTY_NOTICES.md.
