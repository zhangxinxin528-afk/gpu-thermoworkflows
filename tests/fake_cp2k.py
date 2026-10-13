"""A small subprocess fixture for adapter tests. Never used as a scientific engine."""
import os
from pathlib import Path
import re
import sys

if "--version" in sys.argv:
    if os.environ.get("GTW_FAKE_VERSION_FAIL"):
        print("Deliberate fixture version-format failure at " + hex(id(object())), file=sys.stderr)
        raise SystemExit(3)
    print("CP2K adapter-test fixture; NOT electronic structure")
elif "--check" in sys.argv:
    print("Fixture input check")
else:
    marker = Path.cwd().parents[1] / "trigger-failure-once"
    if marker.exists():
        marker.unlink()
        print("Deliberate fixture failure", file=sys.stderr)
        raise SystemExit(7)
    text = Path("input.inp").read_text()
    block = re.search(r"&COORD\s*\n(.*?)&END COORD", text, re.S).group(1)
    n = sum(bool(line.strip()) for line in block.splitlines())
    lines = ["SCF run converged", "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -1.0"]
    lines += [f"FORCES| {i+1} 0.0 0.0 0.0 0.0" for i in range(n)]
    lines += ["PROGRAM ENDED AT fixture"]
    Path("output.out").write_text("\n".join(lines))
