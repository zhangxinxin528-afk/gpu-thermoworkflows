#!/usr/bin/env python3
"""Run the small fixed-water example in an isolated validation directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["mock", "cp2k", "gpu4pyscf"], required=True)
    p.add_argument("--cp2k-executable", default="cp2k.psmp")
    p.add_argument("--cp2k-data", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (a.output or root / "validation" / ("local_smoke_" + a.backend + "_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((root / "examples/water" / f"{a.backend}.json").read_text())
    cfg["paths"]["structure"] = str(root / "examples/water/water.xyz")
    cfg["paths"]["work_dir"] = str(out / "run")
    cfg["resources"]["scratch_dir"] = str(out / "scratch")
    if a.backend == "cp2k":
        if not a.cp2k_data:
            p.error("--cp2k-data is required for a real CP2K check")
        cfg["paths"]["template"] = str(root / "examples/water/cp2k.inp.template")
        cfg["method"]["basis_file"] = str(a.cp2k_data.resolve() / "BASIS_MOLOPT")
        cfg["method"]["potential_file"] = str(a.cp2k_data.resolve() / "GTH_POTENTIALS")
        cfg["execution"]["cp2k_command"] = [a.cp2k_executable]
    config = out / "config.json"
    config.write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    commands = [[sys.executable, "-m", "gpu_thermoworkflows", "run", str(config)]]
    if a.backend == "mock":
        commands.append([sys.executable, "-m", "gpu_thermoworkflows", "analyze", str(config)])
    report = {"backend": a.backend, "started_unix": time.time(), "geometry_optimization": False, "stages": []}
    for index, command in enumerate(commands):
        value = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
        (out / f"stage_{index}.stdout").write_text(value.stdout)
        (out / f"stage_{index}.stderr").write_text(value.stderr)
        report["stages"].append({"argv": command, "returncode": value.returncode})
        if value.returncode:
            report.update(state="FAILED", output_directory=str(out))
            (out / "smoke_result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            return value.returncode
    report.update(state="COMPLETE", output_directory=str(out), ended_unix=time.time())
    (out / "smoke_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
