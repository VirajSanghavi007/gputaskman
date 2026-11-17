"""Smoke and integration tests for gputaskman-app on a real machine.

Each class skips cleanly (never fails) when what it needs is missing: Windows with a GPU adapter, and for the
window test pywebview with a working GUI backend. CI runners without a GPU skip them.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def gpus_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        from gputaskman.desktop.sampler import list_gpus
        return bool(list_gpus())
    except Exception:
        return False


def webview_available() -> bool:
    if importlib.util.find_spec("webview") is None:
        return False
    return sys.platform != "win32" or importlib.util.find_spec("clr") is not None


GPUS = gpus_available()


@unittest.skipUnless(GPUS, "needs Windows with a GPU adapter")
class SamplerIntegration(unittest.TestCase):
    def test_real_sources_produce_state(self):
        from gputaskman.desktop.sampler import Sampler, list_gpus

        gpus = list_gpus()
        s = Sampler(gpus, 0.5)
        s.start()
        try:
            deadline = time.monotonic() + 30
            while s.ticks < 2 and time.monotonic() < deadline:
                time.sleep(0.1)
        finally:
            s.stop()
        st = s.state()
        self.assertTrue(st["ready"])
        self.assertIsNone(st["error"])
        self.assertEqual(len(st["gpus"]), len(gpus))
        self.assertIsNotNone(st["system"])
        g0 = st["gpus"][0]
        self.assertNotIn("error", g0)
        self.assertIsInstance(g0["vram_total"], int)
        self.assertGreater(g0["vram_total"], 0)
        self.assertGreaterEqual(len(st["history"]["gpu0.vram"]), 2)
        json.dumps(st)  # what pywebview hands the page must be JSON-serializable


@unittest.skipUnless(sys.platform == "win32", "Windows process semantics")
class EndTaskIntegration(unittest.TestCase):
    def spawn(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: p.poll() is None and p.kill())
        import psutil
        return p, psutil.Process(p.pid)

    def test_ends_a_real_process(self):
        from gputaskman.desktop.bridge import end_task

        p, ps = self.spawn()
        res = end_task(p.pid, ps.name(), ps.create_time())
        self.assertTrue(res["ok"], res)
        self.assertIsNotNone(p.wait(5))

    def test_refuses_wrong_name_and_reused_pid(self):
        from gputaskman.desktop.bridge import end_task

        p, ps = self.spawn()
        self.assertFalse(end_task(p.pid, "notepad.exe", ps.create_time())["ok"])
        self.assertFalse(end_task(p.pid, ps.name(), ps.create_time() - 100)["ok"])
        self.assertIsNone(p.poll())  # still running: nothing was ended

    def test_exited_process(self):
        from gputaskman.desktop.bridge import end_task

        p, ps = self.spawn()
        name = ps.name()
        p.kill()
        p.wait(5)
        self.assertTrue(end_task(p.pid, name)["ok"])


@unittest.skipUnless(GPUS and webview_available(), "needs Windows with a GPU and pywebview + pythonnet")
class AppSmoke(unittest.TestCase):
    def test_window_opens_and_renders_rows(self):
        out = subprocess.run([sys.executable, "-m", "gputaskman.desktop.main", "--selftest", "5"],
                             cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(result["ready"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["title"], "GPU Task Manager")
        self.assertGreaterEqual(result["gpus"], 1)
        self.assertGreaterEqual(result["sampler_ticks"], 2)
        self.assertEqual(result["rows"], result["processes"])  # empty search: every process is a row


@unittest.skipUnless(sys.platform == "win32", "Windows taskbar identity")
class AppIdIntegration(unittest.TestCase):
    def test_real_shell32_sets_the_process_app_id(self):
        code = "\n".join([
            "import ctypes, ctypes.wintypes as w",
            "from gputaskman.desktop.main import set_app_id",
            "print(set_app_id())",
            "p = w.LPWSTR()",
            "ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(p))",
            "print(p.value)",
        ])
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=ENV, capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["True", "VirajSanghavi.GPUTaskManager"])


class CliSmoke(unittest.TestCase):
    def test_help(self):
        out = subprocess.run([sys.executable, "-m", "gputaskman.desktop.main", "--help"],
                             cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0)
        self.assertIn("--interval", out.stdout)
        self.assertNotIn("--selftest", out.stdout)

    def test_bad_interval_exits_2(self):
        out = subprocess.run([sys.executable, "-m", "gputaskman.desktop.main", "-i", "0"],
                             cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 2)
        self.assertIn("interval", out.stderr)


if __name__ == "__main__":
    unittest.main()
