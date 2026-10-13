"""Unified public entry point."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from . import __version__
from .config import load, prepare
from .storage import RunLock


def doctor(config=None):
    report = {"tool_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
              "numpy": importlib.metadata.version("numpy"), "backends": {}}
    cp2k = (config or {}).get("execution", {}).get("cp2k_command", ["cp2k.psmp"])
    executable = shutil.which(cp2k[0])
    report["backends"]["cp2k"] = {"found": bool(executable), "executable": executable}
    if executable:
        try:
            result = subprocess.run([*cp2k, "--version"], capture_output=True, text=True, timeout=15)
            report["backends"]["cp2k"].update(version=(result.stdout+result.stderr).strip(),
                                              version_probe_success=result.returncode == 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            report["backends"]["cp2k"]["probe_error"] = str(error)
    report["backends"]["gpu4pyscf"] = {"importable": bool(importlib.util.find_spec("gpu4pyscf")),
                                         "cupy_importable": bool(importlib.util.find_spec("cupy"))}
    smi = shutil.which("nvidia-smi")
    if not smi and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        smi = "/usr/lib/wsl/lib/nvidia-smi"
    if smi:
        try:
            p = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            report["gpu_inventory"] = p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
        except subprocess.TimeoutExpired:
            report["gpu_inventory"] = "Probe timed out"
    report["note"] = "Inventory only. Use a fixed center calculation to check energy/force execution."
    return report


def parser():
    p = argparse.ArgumentParser(description="Fixed-geometry CP2K / GPU4PySCF workflow")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="inspect dependencies without running SCF")
    d.add_argument("--config")
    for name in ("prepare", "run", "pause", "resume", "status", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("config", help="JSON configuration file")
        if name == "analyze":
            s.add_argument("--temperature", type=float)
        if name in ("run", "resume"):
            s.add_argument("--max-tasks", type=int, help="temporary execution budget; does not change physical inputs")
    s = sub.add_parser("compare", help="compare same-geometry force records from different engine builds")
    s.add_argument("reference")
    s.add_argument("candidate")
    s.add_argument("--energy-tolerance", type=float, default=1e-7)
    s.add_argument("--force-tolerance", type=float, default=1e-5)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            value = doctor(load(args.config) if args.config else None)
        elif args.command == "compare":
            from .analysis import compare
            value = compare(args.reference, args.candidate, args.energy_tolerance, args.force_tolerance)
        else:
            c = load(args.config)
            work = c["paths"]["work_dir"]
            if args.command == "prepare":
                with RunLock(work):
                    manifest = prepare(c)
                value = {"state": "PREPARED", "work_dir": work, "tasks": len(manifest["tasks"]),
                         "science_sha256": manifest["science_sha256"]}
            elif args.command in ("run", "resume"):
                from .runner import run
                if args.max_tasks is not None:
                    if args.max_tasks < 1:
                        raise ValueError("--max-tasks must be positive")
                    c["execution"]["max_tasks_per_run"] = args.max_tasks
                value = run(c, resume=args.command == "resume")
            elif args.command == "pause":
                from .runner import request_pause
                value = request_pause(c)
            elif args.command == "status":
                from .runner import status
                value = status(c)
            else:
                from .analysis import analyze
                with RunLock(work):
                    value = analyze(c, args.temperature)
        print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        if value.get("state") == "FAILED" or value.get("accepted") is False:
            return 1
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f"gtw: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
