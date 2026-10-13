#!/usr/bin/env python3
"""Explicit, recorded CP2K CUDA build. Upstream source/dependencies are supplied by the user."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "build", "install_prefix", "cuda_root"):
        p.add_argument("--" + flag.replace("_", "-"), required=True, type=Path)
    p.add_argument("--dependency-prefix", action="append", type=Path, default=[])
    p.add_argument("--architectures", default="80;89")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cmake", default="cmake")
    p.add_argument("--cc")
    p.add_argument("--cxx")
    p.add_argument("--fc")
    p.add_argument("--dry-run", action="store_true", help="print recipe without invoking build tools")
    p.add_argument("--install", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"\d+(;\d+)*", a.architectures) or a.jobs < 1:
        p.error("architectures must be a semicolon list of CUDA capabilities; jobs must be positive")
    source, build = a.source.resolve(), a.build.resolve()
    if build == source or build.is_relative_to(source):
        p.error("Use an independent build directory outside the source tree")
    if not a.dry_run and not (source / "CMakeLists.txt").exists():
        p.error("source must be an unpacked CP2K source tree")
    command = [a.cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=Release", "-DCP2K_USE_MPI=ON", "-DCP2K_USE_MPI_F08=ON",
               "-DCP2K_USE_LIBXC=ON", "-DCP2K_USE_FFTW3=ON", "-DCP2K_USE_ACCEL=CUDA",
               "-DCP2K_ENABLE_GRID_GPU=ON", "-DCP2K_ENABLE_DBM_GPU=ON", "-DCP2K_ENABLE_PW_GPU=ON",
               "-DCP2K_USE_DBCSR_CONFIG=ON", "-DCP2K_USE_LIBXS=ON", "-DCP2K_USE_LIBXSMM=ON",
               f"-DCMAKE_CUDA_ARCHITECTURES={a.architectures}", f"-DCUDAToolkit_ROOT={a.cuda_root.resolve()}",
               f"-DCMAKE_INSTALL_PREFIX={a.install_prefix.resolve()}", "-DBUILD_TESTING=ON"]
    if a.dependency_prefix:
        command.append("-DCMAKE_PREFIX_PATH=" + ";".join(str(x.resolve()) for x in a.dependency_prefix))
    for key, value in (("C", a.cc), ("CXX", a.cxx), ("Fortran", a.fc)):
        if value:
            command.append(f"-DCMAKE_{key}_COMPILER={value}")
    commands = [command, [a.cmake, "--build", str(build), "--target", "cp2k-bin", "-j", str(a.jobs)]]
    if a.install:
        commands.append([a.cmake, "--install", str(build)])
    manifest = {"commands": commands, "upstream": "https://github.com/cp2k/cp2k",
                "license": "GPL-2.0-or-later", "dry_run": a.dry_run, "started_unix": time.time()}
    print(json.dumps(manifest, indent=2))
    if a.dry_run:
        return 0
    build.mkdir(parents=True, exist_ok=True)
    record = build / "workflow_build_record.json"
    for i, cmd in enumerate(commands):
        log = build / f"workflow_build_stage_{i}_{time.time_ns()}.log"
        with log.open("w") as output:
            result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
        manifest[f"stage_{i}"] = {"returncode": result.returncode, "log": log.name,
                                   "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        record.write_text(json.dumps(manifest, indent=2))
        if result.returncode:
            return result.returncode
    manifest["ended_unix"] = time.time()
    record.write_text(json.dumps(manifest, indent=2))
    print("Build complete. Inspect --version, run upstream checks, then a fixed-force smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
