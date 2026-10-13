"""Meaningful offline tests: queues, provenance, parser and harmonic physics."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np

from gpu_thermoworkflows.analysis import AMU, BOHR_A, C, EH_J, MASSES, analyze, compare
from gpu_thermoworkflows.config import load, prepare, render, tasks
from gpu_thermoworkflows.engines import parse_cp2k
from gpu_thermoworkflows.storage import alive, identity, read, RunLock, save, sha

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_base = Path(os.environ.get("GTW_TEST_TMP", tempfile.gettempdir())).resolve()
        self.temp_base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_base)
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(read(ROOT / "examples/water/mock.json"))
        self.config["paths"] = {"work_dir": "run", "structure": "water.xyz"}
        self.config["resources"]["scratch_dir"] = str(self.root / "scratch")
        (self.root / "water.xyz").write_text((ROOT / "examples/water/water.xyz").read_text())
        self.path = self.root / "config.json"
        save(self.path, self.config)
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + self.env.get("PYTHONPATH", "")

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.temp_base))
        self.temp.cleanup()

    def cli(self, *args, good=True):
        p = subprocess.run([sys.executable, "-m", "gpu_thermoworkflows", *map(str, args)],
                           capture_output=True, text=True, env=self.env, timeout=40)
        if good:
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        else:
            self.assertNotEqual(p.returncode, 0)
        return p

    def test_complete_and_analyze_water(self):
        r = json.loads(self.cli("run", self.path).stdout)
        self.assertEqual((r["state"], r["completed"], r["total"]), ("COMPLETE", 19, 19))
        result = json.loads(self.cli("analyze", self.path).stdout)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["removed_rigid_modes"], 6)
        self.assertEqual(len(result["signed_frequencies_cm_1"]), 3)
        self.assertGreater(min(result["signed_frequencies_cm_1"]), 0)
        self.assertLess(result["Hessian_relative_antisymmetry"], 1e-4)
        # Re-evaluating temperature reuses all force records.
        before = sha(self.root / "run/tasks/center/result.json")
        self.cli("analyze", self.path, "--temperature", 350)
        self.assertEqual(before, sha(self.root / "run/tasks/center/result.json"))

    def test_resume_skips_verified_points_and_budget(self):
        a = json.loads(self.cli("run", self.path, "--max-tasks", 3).stdout)
        self.assertEqual((a["state"], a["completed"]), ("BUDGET_STOP", 3))
        p = self.root / "run/tasks/center/result.json"
        stamp, checksum = p.stat().st_mtime_ns, sha(p)
        self.cli("run", self.path, good=False)
        b = json.loads(self.cli("resume", self.path).stdout)
        self.assertEqual(b["completed"], 19)
        self.assertEqual((stamp, checksum), (p.stat().st_mtime_ns, sha(p)))

    def test_cooperative_pause_and_resume(self):
        self.config["method"]["delay_seconds"] = 0.2
        save(self.path, self.config)
        p = subprocess.Popen([sys.executable, "-m", "gpu_thermoworkflows", "run", str(self.path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)
        try:
            deadline = time.time() + 15
            while time.time() < deadline:
                file = self.root / "run/status.json"
                if file.exists() and read(file).get("active"):
                    break
                if p.poll() is not None:
                    self.fail(p.communicate())
                time.sleep(0.05)
            else:
                self.fail("Supervisor did not start")
            self.cli("pause", self.path)
            stdout, stderr = p.communicate(timeout=20)
            self.assertEqual(p.returncode, 0, stderr)
            saved = json.loads(stdout)
            self.assertEqual(saved["state"], "PAUSED")
            self.assertGreater(saved["completed"], 0)
            self.assertLess(saved["completed"], 19)
            self.assertEqual(json.loads(self.cli("resume", self.path).stdout)["state"], "COMPLETE")
        finally:
            if p.poll() is None:
                p.terminate()
                p.communicate(timeout=20)

    def test_changed_inputs_rejected_and_preserved(self):
        self.cli("run", self.path, "--max-tasks", 1)
        result = self.root / "run/tasks/center/result.json"
        checksum = sha(result)
        self.config["calculation"]["displacement"]["value"] = 0.006
        save(self.path, self.config)
        p = self.cli("resume", self.path, good=False)
        self.assertIn("Scientific inputs changed", p.stderr)
        self.assertEqual(sha(result), checksum)

    def test_corrupted_result_is_not_reused(self):
        self.cli("run", self.path, "--max-tasks", 1)
        p = self.root / "run/tasks/center/result.json"
        value = read(p)
        value["energy_Ha"] += 1
        save(p, value)
        r = self.cli("resume", self.path, good=False)
        self.assertIn("checksum mismatch", r.stderr)

    def test_atom_count_linear_projection_known_frequency(self):
        (self.root / "water.xyz").write_text("2\nFixed diatomic\nH -0.4 0 0\nH 0.4 0 0\n")
        self.cli("run", self.path)
        c = load(self.path)
        self.assertEqual(len(tasks(c)), 13)
        r = analyze(c)
        self.assertEqual(r["removed_rigid_modes"], 5)
        self.assertEqual(len(r["signed_frequencies_cm_1"]), 1)
        factor = np.sqrt(EH_J/(AMU*(BOHR_A*1e-10)**2))/(2*np.pi*C*100)
        expected = factor*np.sqrt(2*self.config["method"]["spring_constant_Ha_Bohr2"]/MASSES["H"])
        self.assertAlmostEqual(r["signed_frequencies_cm_1"][0], expected, delta=1e-7)

    def test_optimization_nonfinite_and_unknown_backend_rejected(self):
        for change in ("optimization", "nan", "backend"):
            c = copy.deepcopy(self.config)
            if change == "optimization":
                c["calculation"]["geometry_optimization"] = True
            elif change == "nan":
                c["calculation"]["displacement"]["value"] = float("nan")
            else:
                c["engine"] = "unknown"
            # Deliberately allow invalid JSON NaN only in this negative test.
            self.path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load(self.path)

    def test_lock_and_process_identity(self):
        work = self.root / "lock"
        with RunLock(work):
            with self.assertRaises(RuntimeError):
                with RunLock(work):
                    pass
        record = identity()
        self.assertTrue(alive(record))
        changed = dict(record, start_ticks="not the saved process")
        self.assertFalse(alive(changed))

    def test_cp2k_parser_both_formats_and_failure(self):
        header = "SCF run converged\nENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -76.0\n"
        a = header + "FORCES| 1 0.1 0.2 0.3 0.374\nPROGRAM ENDED AT now\n"
        self.assertEqual(parse_cp2k(a, 1), (-76.0, [[0.1, 0.2, 0.3]]))
        b = header + "ATOMIC FORCES in [a.u.]\n 1 1 O 1.0D-1 0.2 0.3\nSUM OF ATOMIC FORCES\nPROGRAM ENDED AT now\n"
        self.assertEqual(parse_cp2k(b, 1), (-76.0, [[0.1, 0.2, 0.3]]))
        with self.assertRaises(ValueError):
            parse_cp2k(a.replace("SCF run converged", "SCF run NOT converged"), 1)
        with self.assertRaises(ValueError):
            parse_cp2k(a, 2)

    def cp2k_config(self):
        c = copy.deepcopy(read(ROOT / "examples/water/cp2k.json"))
        c["paths"] = {"structure": "water.xyz", "work_dir": "run", "template": "input.template"}
        (self.root / "input.template").write_text((ROOT / "examples/water/cp2k.inp.template").read_text())
        for key in ("basis_file", "potential_file"):
            (self.root / key).write_text("test data file")
            c["method"][key] = key
        c["execution"]["cp2k_command"] = [sys.executable, str(ROOT / "tests/fake_cp2k.py")]
        c["resources"]["scratch_dir"] = str(self.root / "scratch")
        return c

    def test_template_config_controls_methods_coordinates(self):
        c = self.cp2k_config()
        c["method"]["cutoff_Ry"] = 456
        save(self.path, c)
        prepared = load(self.path)
        text = render(prepared, {"id": "center", "coordinate": None, "sign": 0})
        self.assertIn("CUTOFF [Ry] 456", text)
        self.assertIn("&KIND O", text)
        self.assertNotIn("{{", text)
        (self.root / "input.template").write_text(text.replace("ENERGY_FORCE", "GEO_OPT"))
        with self.assertRaises(ValueError):
            load(self.path)

    def test_external_cp2k_adapter_and_failed_attempt_retention(self):
        c = self.cp2k_config()
        save(self.path, c)
        # The fake executable fails once in each fresh point; resume must retain it.
        self.cli("prepare", self.path)
        marker = self.root / "run/tasks/center/trigger-failure-once"
        marker.write_text("test only")
        failed = self.cli("run", self.path, good=False)
        self.assertIn('"state": "FAILED"', failed.stdout)
        complete = json.loads(self.cli("resume", self.path).stdout)
        self.assertEqual(complete["state"], "COMPLETE")
        attempts = list((self.root / "run/tasks/center/attempts").iterdir())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sum((p / "failure.json").exists() for p in attempts), 1)
        result = self.root / "run/tasks/center/result.json"
        self.assertTrue(compare(result, result)["accepted"])

    def test_analysis_requires_complete_hessian(self):
        self.cli("run", self.path, "--max-tasks", 2)
        self.assertIn("Missing force point", self.cli("analyze", self.path, good=False).stderr)

    def test_version_probe_failure_is_retained_with_successful_force_output(self):
        save(self.path, self.cp2k_config())
        self.env["GTW_FAKE_VERSION_FAIL"] = "1"
        self.assertEqual(json.loads(self.cli("run", self.path).stdout)["state"], "COMPLETE")
        r = read(self.root / "run/tasks/center/result.json")
        self.assertEqual(r["metadata"]["runtime"]["version"]["returncode"], 3)
        self.assertTrue(r["metadata"]["scf_converged"])
        checksum = sha(self.root / "run/tasks/center/result.json")
        self.assertEqual(json.loads(self.cli("resume", self.path).stdout)["state"], "COMPLETE")
        self.assertEqual(checksum, sha(self.root / "run/tasks/center/result.json"))

    def test_standard_state_conversion_is_explicit(self):
        self.cli("run", self.path)
        c = load(self.path)
        original = analyze(c)["treatments"]["RRHO"]["G_correction_kJ_mol"]
        c["analysis"]["standard_state"] = "1M"
        altered = analyze(c)["treatments"]["RRHO"]
        self.assertAlmostEqual(altered["G_correction_kJ_mol"]-original,
                               altered["concentration_conversion_kJ_mol"], places=10)
        self.assertGreater(altered["concentration_conversion_kJ_mol"], 7)

    def test_cuda_build_recipe_dry_run(self):
        p = subprocess.run([sys.executable, str(ROOT / "tools/build_cp2k_cuda.py"),
                            "--source", str(self.root / "source"), "--build", str(self.root / "build"),
                            "--install-prefix", str(self.root / "install"), "--cuda-root", str(self.root / "cuda"),
                            "--architectures", "80;89", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        report = json.loads(p.stdout)
        self.assertIn("-DCP2K_USE_ACCEL=CUDA", report["commands"][0])
        self.assertFalse((self.root / "build").exists())


if __name__ == "__main__":
    unittest.main()
