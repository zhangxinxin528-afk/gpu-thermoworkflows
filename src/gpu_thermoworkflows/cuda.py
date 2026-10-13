"""Find CUDA libraries shipped by pip inside the active Python environment."""
import ctypes.util
import os
from pathlib import Path
import sysconfig


def wheel_library_dirs():
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    return [str(p) for p in sorted(root.glob("*/lib")) if p.is_dir()]


def configure_spawn_environment():
    # Done before worker spawn: glibc reads the loader path at process startup.
    directories = wheel_library_dirs()
    if directories and os.name != "nt":
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([*directories, *filter(None, existing.split(":"))]))
    return directories


def import_gpu_backend():
    # Older GPU4PySCF versions ask find_library('cusolver'). Its pip-wheel path
    # may not be in ldconfig's database. Restore the function after import.
    previous = ctypes.util.find_library
    def find(name):
        value = previous(name)
        if value is None and name == "cusolver":
            matches = [p for d in wheel_library_dirs() for p in Path(d).glob("libcusolver.so.*")]
            if len(matches) == 1:
                return str(matches[0])
        return value
    ctypes.util.find_library = find
    try:
        import gpu4pyscf
        return gpu4pyscf
    finally:
        ctypes.util.find_library = previous
