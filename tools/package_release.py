#!/usr/bin/env python3
"""Create a source-only release archive with checksums and verified QA records."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (a.output or root / "dist/gpu-thermoworkflows-0.1.0-source.zip").resolve()
    qa_file = root / "validation/release_validation.json"
    if not qa_file.exists() or json.loads(qa_file.read_text())["state"] != "PASS":
        raise SystemExit("Complete and save release_validation.json before packaging.")
    qa = json.loads(qa_file.read_text())
    for relative, expected in qa["source_modules_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit("Source changed after QA: " + relative)
    fixed = ["pyproject.toml", "MANIFEST.in", "README.md", "README.zh-CN.md", "LICENSE",
             "THIRD_PARTY_NOTICES.md", "CITATION.cff", "CHANGELOG.md", ".gitignore", ".gitattributes"]
    files = [root / name for name in fixed]
    for directory in ("src", "examples", "docs", "tools", "tests", "environment", "benchmarks", ".github"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc"
                     and not any(part.endswith(".egg-info") for part in path.parts))
    for name in ("release_validation.json", "windows-tests.log", "linux-tests.log", "smoke_checks.json"):
        path = root / "validation" / name
        if path.exists():
            files.append(path)
    manifest = {str(path.relative_to(root)).replace("\\", "/"):
                {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                for path in sorted(set(files))}
    for relative in manifest:
        if relative.endswith((".exe", ".so", ".wfn", ".chk")) or "local_smoke_" in relative:
            raise SystemExit(f"Unexpected runtime or private artifact: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):
            z.write(path, "gpu-thermoworkflows/" + str(path.relative_to(root)).replace("\\", "/"))
        z.writestr("gpu-thermoworkflows/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2))
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "  " + output.name + "\n")
    with zipfile.ZipFile(output) as z:
        for name, record in manifest.items():
            data = z.read("gpu-thermoworkflows/" + name)
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
    print(json.dumps({"archive": str(output), "sha256": checksum,
                      "files": len(manifest), "archive_round_trip_verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
