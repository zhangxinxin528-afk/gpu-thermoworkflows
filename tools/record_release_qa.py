#!/usr/bin/env python3
"""Summarize completed local checks without publishing machine paths or lab data."""
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    validation = root / "validation"
    latest_code_change = max(p.stat().st_mtime for directory in (root / "src", root / "tests") for p in directory.rglob("*.py"))
    tests = {}
    for platform in ("windows", "linux"):
        file = validation / f"{platform}-tests.log"
        text = file.read_text(encoding="utf-8-sig")
        assert file.stat().st_mtime >= latest_code_change, f"{platform} tests precede the latest code changes"
        match = re.search(r"Ran (\d+) tests in ([0-9.]+)s", text)
        assert match and text.rstrip().endswith("OK"), f"{platform} tests not confirmed"
        tests[platform] = {"state": "PASS", "tests": int(match[1]), "seconds": float(match[2]),
                           "log_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    checks = {}
    for backend in ("cp2k", "gpu4pyscf"):
        candidates = sorted(validation.glob("local_smoke_" + backend + "_*/smoke_result.json"))
        successful = [p for p in candidates if json.loads(p.read_text())["state"] == "COMPLETE"]
        assert successful, f"No completed {backend} smoke test"
        folder = successful[-1].parent
        result = json.loads((folder / "run/tasks/center/result.json").read_text())
        assert not result["geometry_optimized"] and result["metadata"]["scf_converged"]
        checks[backend] = {"state": "PASS", "fixed_atoms": len(result["coordinates_A"]),
                           "geometry_optimized": False, "energy_Ha": result["energy_Ha"],
                           "forces_Ha_Bohr": result["forces_Ha_Bohr"],
                           "elapsed_seconds": result["elapsed_seconds"],
                           "raw_artifact_hashes": list(result["artifacts"].values())}
        if backend == "gpu4pyscf":
            checks[backend].update(versions=result["metadata"]["runtime"]["versions"],
                                    gpu=result["metadata"]["gpu_name"])
        else:
            checks[backend].update(output_version_header=result["metadata"]["output_version_header"],
                                    executable_sha256=result["metadata"]["runtime"]["executable_sha256"],
                                    version_probe_returncode=result["metadata"]["runtime"]["version"]["returncode"],
                                    backend="CPU; CUDA execution is historical evidence")
    read = lambda name: json.loads((validation / name).read_text(encoding="utf-8-sig"))
    installed = read("wheel-run.json")
    thermal = read("wheel-analysis.json")
    console = read("wheel-console-status.json")
    assert installed["state"] == "COMPLETE" and installed["completed"] == 19
    assert console["state"] == "COMPLETE" and thermal["synthetic"]
    wheel = next((root / "dist").glob("*.whl"))
    source = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / "src/gpu_thermoworkflows").glob("*.py")}
    with zipfile.ZipFile(wheel) as z:
        for path, checksum in source.items():
            assert hashlib.sha256(z.read(path.removeprefix("src/"))).hexdigest() == checksum
    report = {"state": "PASS", "version": "0.1.0", "recorded_UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "release QA evidence; publication status is reported by GitHub",
              "offline_tests": tests, "real_smoke_checks": checks,
              "wheel_install_and_console": {"state": "PASS", "synthetic_force_points": 19, "internal_modes": 3},
              "source_modules_sha256": source,
              "new_cuda_cp2k_cloud_test": "NOT_RUN; historical A100 measurements are separately labeled",
              "upstream_binaries_included": False, "laboratory_data_included": False}
    (validation / "release_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (validation / "smoke_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"state": report["state"], "offline_tests_per_platform": tests,
                      "real_smoke_backends": list(checks), "source_modules": len(source)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
