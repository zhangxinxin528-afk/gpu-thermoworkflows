"""Fixed force adapters. No geometry optimizer is used."""
import functools
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from .config import BOHR_A, coordinates, render
from .storage import available_memory_mb, digest, identity, read, save, sha


@functools.lru_cache(maxsize=8)
def cp2k_version(command):
    completed = subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=15)
    # Some upstream builds fail while formatting --version, although ENERGY_FORCE
    # works. Preserve that diagnostic; the executable hash still pins the runtime.
    return {"returncode": completed.returncode, "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip()}


def runtime(c):
    if c["engine"] == "cp2k":
        command = c["execution"]["cp2k_command"]
        executable = shutil.which(command[0])
        if not executable:
            raise ValueError(f"CP2K executable not found: {command[0]}")
        return {"engine": "cp2k", "version": cp2k_version(tuple(command)),
                "executable_sha256": sha(executable), "argv_prefix": command[1:]}
    if c["engine"] == "gpu4pyscf":
        names = ["pyscf", "gpu4pyscf-cuda12x", "gpu4pyscf-cuda11x", "gpu4pyscf-cuda13x", "gpu4pyscf", "cupy-cuda12x", "cupy-cuda11x", "cupy-cuda13x", "pyscf-dispersion"]
        versions = {}
        for name in names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        return {"engine": "gpu4pyscf", "versions": versions}
    return {"engine": "mock", "model": "synthetic-all-pairs-harmonic-bonds-v1"}


def runtime_fingerprint(info):
    if info["engine"] == "cp2k":
        # A failed Fortran version formatter may print ASLR-dependent addresses.
        # Those diagnostic addresses must not invalidate identical-binary resume.
        stable = {"engine": "cp2k", "executable_sha256": info["executable_sha256"],
                  "argv_prefix": info["argv_prefix"]}
        if info["version"]["returncode"] == 0:
            stable["version_stdout"] = info["version"]["stdout"]
        return digest(stable)
    return digest(info)


def parse_cp2k(text, atoms):
    if "PROGRAM ENDED AT" not in text or "SCF run NOT converged" in text or "SCF run converged" not in text:
        raise ValueError("CP2K output lacks successful SCF and normal termination")
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
    energies = re.findall(r"ENERGY\| Total FORCE_EVAL.*?energy \[hartree\]\s+(" + number + ")", text)
    if not energies:
        raise ValueError("CP2K total energy was not found")
    rows = re.findall(r"^\s*FORCES\|\s+\d+\s+(" + number + r")\s+(" + number + r")\s+(" + number + r")\s+" + number + r"\s*$", text, re.M)
    if len(rows) < atoms:
        # Conventional ATOMIC FORCES blocks: atom, kind, element, Fx, Fy, Fz.
        blocks = re.findall(r"ATOMIC FORCES in \[a\.u\.\](.*?)(?:SUM OF ATOMIC FORCES|\n\s*ENERGY\||\Z)", text, re.S)
        if blocks:
            rows = re.findall(r"^\s*\d+\s+\d+\s+[A-Za-z_]+\s+(" + number + r")\s+(" + number + r")\s+(" + number + r")\s*$", blocks[-1], re.M)
    if len(rows) < atoms:
        raise ValueError(f"Expected {atoms} atomic forces, found {len(rows)}")
    cv = lambda value: float(value.replace("D", "E").replace("d", "e"))
    return cv(energies[-1]), [[cv(x) for x in row] for row in rows[-atoms:]]


def cp2k(c, task, attempt, scratch):
    text = render(c, task)
    (attempt / "input.inp").write_text(text, encoding="utf-8")
    command = c["execution"]["cp2k_command"]
    env = dict(os.environ, OMP_NUM_THREADS=str(c["resources"]["threads_per_worker"]),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TMPDIR=str(scratch))
    timeout = c["resources"]["timeout_seconds"]
    if c["execution"].get("input_check", True):
        with (attempt / "check.log").open("w") as log:
            value = subprocess.run([*command, "--check", "-i", "input.inp"], cwd=attempt,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if value.returncode:
            raise RuntimeError("CP2K input check failed; see attempt/check.log")
    with (attempt / "launcher.log").open("w") as log:
        process = subprocess.Popen([*command, "-i", "input.inp", "-o", "output.out"],
                                   cwd=attempt, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(attempt / "external_process.json", {"identity": identity(process.pid), "command": command})
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"CP2K exited {code}; see raw attempt logs")
    energy, forces = parse_cp2k((attempt / "output.out").read_text(errors="replace"), len(c["atoms"]))
    raw = (attempt / "output.out").read_text(errors="replace")
    version_header = next((line.strip() for line in raw.splitlines() if "CP2K| version string:" in line), None)
    return energy, forces, {"scf_converged": True, "runtime": runtime(c),
                            "output_version_header": version_header}, ["input.inp", "output.out"]


def gpu4pyscf(c, task, attempt, scratch):
    # A process pool keeps imported CUDA libraries alive across force points.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(c["resources"]["gpu_device"])
    os.environ["PYSCF_TMPDIR"] = str(scratch)
    import numpy as np
    import cupy as cp
    from pyscf import dft, gto, lib
    from pyscf.solvent import smd
    from .cuda import import_gpu_backend
    _gpu4pyscf = import_gpu_backend()
    method = c["method"]
    lib.num_threads(c["resources"]["threads_per_worker"])
    lib.param.TMPDIR = str(scratch)
    free = cp.cuda.runtime.memGetInfo()[0] / 1024 ** 2
    if free < c["resources"].get("min_gpu_free_memory_mb", 0):
        raise RuntimeError(f"Available GPU memory {free:.0f} MiB is below configured minimum")
    mol = gto.M(atom=[(a[0], r) for a, r in zip(c["atoms"], coordinates(c, task))], unit="Angstrom",
                charge=c["calculation"]["charge"], spin=c["calculation"]["multiplicity"] - 1,
                basis=method["basis"], max_memory=c["resources"]["memory_mb_per_worker"],
                verbose=4, output=str(attempt / "engine.log"))
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    if method.get("density_fit", True):
        mf = mf.density_fit(auxbasis=method.get("auxbasis"))
    mf.xc = method["xc"]
    mf.disp = method.get("dispersion")
    mf.grids.level = method.get("grid_level", 4)
    mf.conv_tol = method.get("conv_tol", 1e-10)
    mf.conv_tol_grad = method.get("conv_tol_grad", 1e-7)
    mf.max_cycle = method.get("max_cycle", 120)
    solvent = method.get("solvent")
    if solvent:
        if method.get("solvation", "SMD") == "SMD":
            mf = mf.SMD(smd.SMD(mol, solvent=solvent))
        elif method["solvation"] == "PCM":
            mf = mf.PCM()
            mf.with_solvent.method = method.get("pcm_method", "IEF-PCM")
            mf.with_solvent.eps = method["dielectric"]
        else:
            raise ValueError("solvation must be SMD or PCM")
    gpu = mf.to_gpu()
    gpu.chkfile = str(attempt / "state.chk")
    energy = float(gpu.kernel())
    if not gpu.converged:
        raise RuntimeError("SCF did not converge")
    gradient = gpu.nuc_grad_method()
    gradient.grid_response = method.get("grid_response", True)
    if method.get("density_fit", True):
        gradient.auxbasis_response = method.get("auxbasis_response", True)
    value = gradient.kernel()
    cp.cuda.Stream.null.synchronize()
    forces = -(value.get() if hasattr(value, "get") else np.asarray(value))
    metadata = {"scf_converged": True, "runtime": runtime(c),
                "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "grid_response": gradient.grid_response}
    if mol.spin:
        metadata["spin_square"] = float(gpu.spin_square()[0])
    mol.stdout.close()
    cp.get_default_memory_pool().free_all_blocks()
    return energy, forces.tolist(), metadata, ["engine.log", "state.chk"]


def mock(c, task, attempt, scratch):
    """Synthetic bond springs for plumbing tests; no electronic structure."""
    import numpy as np
    time.sleep(c["method"].get("delay_seconds", 0))
    reference = np.array([a[1] for a in c["atoms"]]) / BOHR_A
    point = np.array(coordinates(c, task)) / BOHR_A
    forces = np.zeros_like(point)
    energy = 0.0
    k = c["method"].get("spring_constant_Ha_Bohr2", 0.2)
    for i in range(len(point)):
        for j in range(i):
            vector = point[i] - point[j]
            r = np.linalg.norm(vector)
            r0 = np.linalg.norm(reference[i] - reference[j])
            if r <= 0 or r0 <= 0:
                raise ValueError("Synthetic model requires non-overlapping atoms")
            gradient = k * (r - r0) * vector / r
            energy += 0.5 * k * (r - r0) ** 2
            forces[i] -= gradient
            forces[j] += gradient
    (attempt / "synthetic.log").write_text("Synthetic harmonic bonds; not DFT data.\n")
    return energy, forces.tolist(), {"synthetic": True, "runtime": runtime(c)}, ["synthetic.log"]


def execute(c, task, fingerprint):
    folder = Path(c["paths"]["work_dir"]) / "tasks" / task["id"]
    attempts = folder / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"{time.time_ns()}_{os.getpid()}"
    attempt.mkdir()
    started = time.time()
    save(attempt / "launch.json", {"identity": identity(), "task": task, "started_unix": started})
    os.environ["OMP_NUM_THREADS"] = str(c["resources"]["threads_per_worker"])
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    root = c["resources"].get("scratch_dir")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="gtw_", dir=root))
    try:
        memory = available_memory_mb()
        if memory is not None and memory < c["resources"]["min_available_memory_mb"]:
            raise RuntimeError(f"Available host memory {memory:.0f} MiB is below configured minimum")
        functions = {"cp2k": cp2k, "gpu4pyscf": gpu4pyscf, "mock": mock}
        energy, forces, metadata, names = functions[c["engine"]](c, task, attempt, scratch)
        import numpy as np
        array = np.array(forces)
        if array.shape != (len(c["atoms"]), 3) or not np.isfinite(array).all() or not np.isfinite(energy):
            raise ValueError("Non-finite energy or malformed force array")
        artifacts = {str((attempt / name).relative_to(folder)): sha(attempt / name) for name in names}
        result = {"task": task, "science_sha256": fingerprint, "energy_Ha": energy,
                  "forces_Ha_Bohr": forces, "coordinates_A": coordinates(c, task),
                  "geometry_optimized": False, "metadata": metadata,
                  "elapsed_seconds": time.time() - started, "artifacts": artifacts}
        save(folder / "result.json", result)
        (folder / "result.sha256").write_text(sha(folder / "result.json") + "\n")
        return task["id"]
    except BaseException:
        save(attempt / "failure.json", {"error": traceback.format_exc(), "time_unix": time.time(),
                                        "scratch": str(scratch)})
        raise
    finally:
        scratch_base = Path(root or tempfile.gettempdir()).resolve()
        if not scratch.resolve().is_relative_to(scratch_base):
            raise RuntimeError("Scratch cleanup target escaped the configured temporary directory")
        shutil.rmtree(scratch)


def verified(c, task, fingerprint):
    folder = Path(c["paths"]["work_dir"]) / "tasks" / task["id"]
    if not (folder / "result.json").exists():
        return False
    checksum = folder / "result.sha256"
    if not checksum.exists() or checksum.read_text().strip() != sha(folder / "result.json"):
        raise ValueError(f"Result checksum mismatch: {task['id']}; results preserved")
    result = read(folder / "result.json")
    if result["science_sha256"] != fingerprint or result["task"] != task or result["geometry_optimized"]:
        raise ValueError(f"Result provenance mismatch: {task['id']}")
    if result["coordinates_A"] != coordinates(c, task):
        raise ValueError(f"Result coordinate mismatch: {task['id']}")
    for relative, expected in result["artifacts"].items():
        file = (folder / relative).resolve()
        if not file.is_relative_to(folder.resolve()) or sha(file) != expected:
            raise ValueError(f"Artifact checksum mismatch: {task['id']}/{relative}")
    return True
