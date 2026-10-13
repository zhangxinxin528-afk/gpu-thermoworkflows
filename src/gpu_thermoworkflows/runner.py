"""Bounded foreground supervisor with cooperative checkpointed pause."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from pathlib import Path
import signal
import time
from .config import prepare
from .engines import execute, runtime, runtime_fingerprint, verified
from .storage import alive, digest, identity, read, RunLock, save


def request_pause(c):
    work = Path(c["paths"]["work_dir"])
    if not (work / "manifest.json").exists():
        raise ValueError("No prepared work directory")
    save(work / "control.json", {"pause_requested": True, "time_unix": time.time()})
    current = read(work / "status.json")
    if not alive(current.get("supervisor")):
        current.update(state="PAUSED", pause_mode="between force points", updated_unix=time.time())
        save(work / "status.json", current)
    return {"state": "PAUSE_REQUESTED", "mode": "finish active points; preserve all attempts and results"}


def status(c):
    work = Path(c["paths"]["work_dir"])
    value = read(work / "status.json")
    value["supervisor_alive"] = alive(value.get("supervisor"))
    value["pause_requested"] = (read(work / "control.json").get("pause_requested", False)
                                if (work / "control.json").exists() else False)
    if value["state"] == "RUNNING" and not value["supervisor_alive"]:
        value["observed_state"] = "INTERRUPTED; saved RUNNING record has no matching process"
    return value


def run(c, resume=False):
    work = Path(c["paths"]["work_dir"])
    with RunLock(work):
        manifest = prepare(c)
        previous = read(work / "status.json")
        if previous["state"] in ("PAUSED", "FAILED", "BUDGET_STOP") and not resume:
            raise ValueError("Use resume to continue the saved queue")
        completed = {t["id"] for t in manifest["tasks"] if verified(c, t, manifest["science_sha256"])}
        save(work / "control.json", {"pause_requested": False, "resumed": resume, "time_unix": time.time()})
        runtime_info = runtime(c)
        if c["engine"] == "gpu4pyscf":
            from .cuda import configure_spawn_environment
            configure_spawn_environment()
        fingerprint = runtime_fingerprint(runtime_info)
        runtime_file = work / "runtime.json"
        if runtime_file.exists() and runtime_fingerprint(read(runtime_file)["runtime"]) != fingerprint:
            raise ValueError("Engine runtime changed. Use a new work_dir and compare force results before reuse.")
        save(runtime_file, {"sha256": fingerprint, "runtime": runtime_info})
        record = {"state": "RUNNING", "supervisor": identity(), "completed": len(completed),
                  "total": len(manifest["tasks"]), "active": [], "started_unix": time.time(), "errors": []}
        pending = [t for t in manifest["tasks"] if t["id"] not in completed]
        limit = c["execution"]["max_tasks_per_run"]
        selected = pending if limit is None else pending[:limit]
        queue = iter(selected)
        finished = False
        def snapshot():
            record["updated_unix"] = time.time()
            save(work / "status.json", record)
        def pause_signal(*_):
            save(work / "control.json", {"pause_requested": True, "time_unix": time.time()})
        previous_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, pause_signal)
        try:
            snapshot()
            # spawn prevents inherited CUDA contexts and supports Windows.
            with ProcessPoolExecutor(max_workers=c["resources"]["workers"],
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {}
                while futures or not finished:
                    paused = read(work / "control.json").get("pause_requested", False)
                    while not paused and not record["errors"] and not finished and len(futures) < c["resources"]["workers"]:
                        task = next(queue, None)
                        if task is None:
                            finished = True
                            break
                        futures[pool.submit(execute, c, task, manifest["science_sha256"])] = task
                    if paused or record["errors"]:
                        finished = True
                    record["active"] = [t["id"] for t in futures.values()]
                    snapshot()
                    if not futures:
                        break
                    done, _ = wait(futures, timeout=0.3, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = futures.pop(future)
                        try:
                            future.result()
                            verified(c, task, manifest["science_sha256"])
                            completed.add(task["id"])
                        except Exception as error:
                            record["errors"].append({"task": task["id"], "error": str(error)})
                    record["completed"] = len(completed)
            paused = read(work / "control.json").get("pause_requested", False)
            record.update(active=[], state=("FAILED" if record["errors"] else
                          "COMPLETE" if len(completed) == record["total"] else
                          "PAUSED" if paused else "BUDGET_STOP"))
            snapshot()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        return record
