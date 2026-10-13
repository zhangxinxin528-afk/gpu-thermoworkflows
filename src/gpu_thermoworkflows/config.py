"""Configuration, immutable scientific fingerprints and fixed displacements."""
import copy
import math
import os
from pathlib import Path
import re
from .storage import digest, read, save, sha

BOHR_A = 0.529177210903


def load(path):
    path = Path(path).resolve()
    c = read(path)
    if c.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if c.get("engine") not in ("mock", "cp2k", "gpu4pyscf"):
        raise ValueError("engine must be mock, cp2k or gpu4pyscf")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", c.get("name", "")):
        raise ValueError("name must contain only letters, digits, _, . or -")
    c = copy.deepcopy(c)
    for key in ("work_dir", "structure", "template"):
        if key in c["paths"]:
            c["paths"][key] = resolve(c["paths"][key], path.parent)
    if "scratch_dir" in c.get("resources", {}) and c["resources"]["scratch_dir"]:
        c["resources"]["scratch_dir"] = resolve(c["resources"]["scratch_dir"], path.parent)
    for key in ("basis_file", "potential_file", "dispersion_file", "restart_file"):
        if c.get("method", {}).get(key):
            c["method"][key] = resolve(c["method"][key], path.parent)
    c["config_path"] = str(path)
    calc = c["calculation"]
    if calc.get("geometry_optimization", False):
        raise ValueError("This tool executes fixed geometries. geometry_optimization must be false.")
    calc.setdefault("geometry_optimization", False)
    calc.setdefault("charge", 0)
    calc.setdefault("multiplicity", 1)
    calc.setdefault("periodic", False)
    calc.setdefault("hessian", True)
    if not isinstance(calc["charge"], int) or not isinstance(calc["multiplicity"], int) or calc["multiplicity"] < 1:
        raise ValueError("charge must be integer; multiplicity must be positive integer")
    d = calc["displacement"]
    if d["unit"] not in ("angstrom", "bohr") or not math.isfinite(d["value"]) or d["value"] <= 0:
        raise ValueError("displacement: positive value and unit angstrom or bohr required")
    r = c.setdefault("resources", {})
    defaults = dict(workers=1, threads_per_worker=1, memory_mb_per_worker=1500,
                    min_available_memory_mb=0, timeout_seconds=600, gpu_device=0)
    for key, value in defaults.items():
        r.setdefault(key, value)
    for key in ("workers", "threads_per_worker", "memory_mb_per_worker", "timeout_seconds"):
        if not isinstance(r[key], int) or r[key] < 1:
            raise ValueError(f"resources.{key} must be a positive integer")
    if c["engine"] == "gpu4pyscf" and calc["periodic"]:
        raise ValueError("The GPU4PySCF adapter in v0.1 supports molecular calculations only.")
    execution = c.setdefault("execution", {})
    execution.setdefault("max_tasks_per_run", None)
    limit = execution["max_tasks_per_run"]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("max_tasks_per_run must be positive integer or null")
    if c["engine"] == "cp2k":
        command = execution.get("cp2k_command", ["cp2k.psmp"])
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("cp2k_command must be a nonempty argv list (no shell command)")
        execution["cp2k_command"] = [os.path.expandvars(x) for x in command]
        if "template" not in c["paths"]:
            raise ValueError("CP2K requires paths.template")
        template = Path(c["paths"]["template"]).read_text()
        if re.search(r"\b(?:GEO_OPT|CELL_OPT|BAND|MD)\b", strip_comments(template), re.I):
            raise ValueError("CP2K template contains a non-fixed run type or optimization section")
        if not re.search(r"RUN_TYPE\s+ENERGY_FORCE\b", strip_comments(template), re.I):
            raise ValueError("CP2K template must specify RUN_TYPE ENERGY_FORCE")
    c["atoms"] = xyz(c["paths"]["structure"])
    if calc["periodic"]:
        cell = calc.get("cell_A")
        if not cell or len(cell) != 3 or any(len(v) != 3 for v in cell):
            raise ValueError("Periodic calculations require a 3x3 cell_A in angstrom")
    # Force finite, JSON-compatible values; no permissive NaN scientific inputs.
    digest(c)
    return c


def resolve(value, base):
    expanded = os.path.expandvars(os.path.expanduser(value))
    if re.search(r"\$\{|\$[A-Za-z_]", expanded):
        raise ValueError(f"Set the environment variable in path: {value}")
    p = Path(expanded)
    return str((p if p.is_absolute() else base / p).resolve())


def strip_comments(text):
    return "\n".join(line.split("!", 1)[0].split("#", 1)[0] for line in text.splitlines())


def xyz(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    n = int(lines[0])
    if n <= 0 or len(lines) < n + 2 or any(x.strip() for x in lines[n+2:]):
        raise ValueError("XYZ must contain exactly one frame")
    atoms = []
    for line in lines[2:n+2]:
        fields = line.split()
        if len(fields) != 4 or not re.fullmatch(r"[A-Z][a-z]?", fields[0]):
            raise ValueError("XYZ atom lines must be ELEMENT X Y Z (angstrom)")
        coords = list(map(float, fields[1:]))
        if not all(math.isfinite(x) for x in coords):
            raise ValueError("XYZ has non-finite coordinates")
        atoms.append([fields[0], coords])
    return atoms


def science(c):
    method = copy.deepcopy(c["method"])
    sources = {"structure": sha(c["paths"]["structure"])}
    if "template" in c["paths"]:
        sources["template"] = sha(c["paths"]["template"])
    for key in ("basis_file", "potential_file", "dispersion_file", "restart_file"):
        if method.get(key):
            sources[key] = sha(method[key])
            method[key] = {"filename": Path(method[key]).name, "sha256": sources[key]}
    return {"engine": c["engine"], "atoms": c["atoms"], "method": method,
            "calculation": c["calculation"], "sources": sources}


def tasks(c):
    jobs = [{"id": "center", "coordinate": None, "sign": 0}]
    if c["calculation"]["hessian"]:
        for q in range(3 * len(c["atoms"])):
            for sign, label in ((-1, "minus"), (1, "plus")):
                jobs.append({"id": f"q{q:04d}_{label}", "coordinate": q, "sign": sign})
    return jobs


def coordinates(c, task):
    value = copy.deepcopy([a[1] for a in c["atoms"]])
    if task["coordinate"] is not None:
        q = task["coordinate"]
        d = c["calculation"]["displacement"]
        step_A = d["value"] * (BOHR_A if d["unit"] == "bohr" else 1)
        value[q // 3][q % 3] += task["sign"] * step_A
    return value


def render(c, task):
    text = Path(c["paths"]["template"]).read_text()
    calc = c["calculation"]
    cell = calc.get("cell_A", [[20, 0, 0], [0, 20, 0], [0, 0, 20]])
    mapping = dict(c["method"])
    kind_lines = []
    for element, values in c["method"].get("kinds", {}).items():
        if not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError("CP2K kind keys must be element symbols")
        kind_lines += [f"    &KIND {element}", f"      ELEMENT {element}",
                       f"      BASIS_SET {values['basis']}", f"      POTENTIAL {values['potential']}"]
        if "magnetization" in values:
            kind_lines.append(f"      MAGNETIZATION {float(values['magnetization'])}")
        kind_lines.append("    &END KIND")
    mapping["KINDS"] = "\n".join(kind_lines)
    mapping.update(CHARGE=calc["charge"], MULTIPLICITY=calc["multiplicity"],
                   UKS="TRUE" if calc["multiplicity"] > 1 else "FALSE",
                   PERIODIC="XYZ" if calc["periodic"] else "NONE",
                   COORD="\n".join(f"      {a[0]} " + " ".join(f"{v:.14f}" for v in point)
                                      for a, point in zip(c["atoms"], coordinates(c, task))),
                   CELL="\n".join(f"      {axis} " + " ".join(f"{v:.14f}" for v in vector)
                                     for axis, vector in zip("ABC", cell)))
    def replace(match):
        key = match.group(1)
        if key not in mapping:
            raise ValueError(f"Missing template parameter: method.{key}")
        value = str(mapping[key])
        if key not in ("COORD", "CELL", "KINDS") and ("\n" in value or "\r" in value):
            raise ValueError(f"Multiline scalar template value: {key}")
        return value
    return re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", replace, text)


def prepare(c):
    work = Path(c["paths"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(science(c))
    manifest = work / "manifest.json"
    if manifest.exists():
        old = read(manifest)
        if old["science_sha256"] != fingerprint:
            raise ValueError("Scientific inputs changed. Choose a new work_dir; existing results are preserved.")
        return old
    jobs = tasks(c)
    for task in jobs:
        directory = work / "tasks" / task["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if c["engine"] == "cp2k":
            (directory / "prepared.inp").write_text(render(c, task), encoding="utf-8")
    value = {"schema_version": 1, "science_sha256": fingerprint, "science": science(c),
             "engine": c["engine"], "tasks": jobs, "geometry_optimization": False}
    save(manifest, value)
    save(work / "config_snapshot.json", c)
    save(work / "status.json", {"state": "PREPARED", "completed": 0, "total": len(jobs)})
    return value
