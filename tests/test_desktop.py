"""Unit tests for the desktop app (gputaskman-app). Hardware-independent: no GPU, no window."""

import json
import math
import sys
import threading
import time
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import psutil

from gputaskman.desktop import bridge, main, model, sampler
from gputaskman.desktop.model import (History, _num, _sum_measured, error_view, gpu_view, merge_processes,
                                      process_row, record_history, system_view, top_engine)


def proc(**kw):
    base = {"pid": 10, "name": "app.exe", "host": None, "kind": "app", "runtimes": ["CUDA"],
            "vram_bytes": 100, "shared_bytes": 5, "gpu_pct": 1.0, "engines": {"3D": 1.0},
            "started_at": 1000.5}
    base.update(kw)
    return base


def snap(**kw):
    base = {"gpu": "Test GPU", "backend": "wddm", "driver": "1.0", "cuda": "13.0",
            "adapter": {"shared_system": 4000}, "vram_total": 1000, "vram_used": 300, "shared_used": 20,
            "attributed": 250, "device": {"util_gpu": 7, "util_mem": 2, "enc": 0, "dec": 1, "temp": 50,
                                          "power_w": 3.5, "power_limit_w": 80.0, "fan": None, "pstate": 8,
                                          "sm_clock": 180, "sm_clock_max": 3000, "mem_clock": 400,
                                          "pcie_rx_kbs": 10, "pcie_tx_kbs": 2, "driver_model": "WDDM",
                                          "pci_bus": "0:2:0"},
            "processes": [proc()]}
    base.update(kw)
    return base


class HistoryTests(unittest.TestCase):
    def test_keeps_last_n_and_gaps(self):
        h = History(3)
        for v in (1, None, 3, 4):
            h.push("a", v)
        self.assertEqual(h.get("a"), [None, 3, 4])
        self.assertEqual(h.to_dict(), {"a": [None, 3, 4]})
        self.assertEqual(h.get("missing"), [])

    def test_size_one_is_allowed_zero_is_not(self):
        h = History(1)
        h.push("a", 1)
        h.push("a", 2)
        self.assertEqual(h.get("a"), [2])
        with self.assertRaises(ValueError):
            History(0)

    def test_default_size(self):
        self.assertEqual(History().size, model.HISTORY)
        self.assertEqual(model.HISTORY, 60)


class NumTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(_num(0), 0)
        self.assertEqual(_num(2.5), 2.5)
        self.assertIsNone(_num(True))
        self.assertIsNone(_num(False))
        self.assertIsNone(_num("3"))
        self.assertIsNone(_num(None))
        self.assertIsNone(_num(float("nan")))
        self.assertIsNone(_num(float("inf")))
        self.assertIsNone(_num(float("-inf")))

    def test_sum_measured(self):
        self.assertIsNone(_sum_measured([]))
        self.assertIsNone(_sum_measured([None, None]))
        self.assertEqual(_sum_measured([None, 0]), 0)
        self.assertEqual(_sum_measured([1, None, 2]), 3)


class TopEngineTests(unittest.TestCase):
    def test_busiest(self):
        self.assertEqual(top_engine({"3D": 5.0, "Copy": 9.0}), ("Copy", 9.0))

    def test_ties_pick_a_stable_name(self):
        self.assertEqual(top_engine({"A": 5.0, "B": 5.0}), ("B", 5.0))

    def test_idle_or_unknown(self):
        self.assertIsNone(top_engine(None))
        self.assertIsNone(top_engine({}))
        self.assertIsNone(top_engine({"3D": 0.0}))


class ProcessRowTests(unittest.TestCase):
    def test_flatten(self):
        r = process_row(proc(runtimes=["CUDA", "cuDNN"], inside="model.gguf", engine_unavailable_reason="why",
                             llm={"segments": [{"label": "weights", "bytes": 50, "kind": "weights", "exact": 1}],
                                  "models": [{"label": "L"}, {"name": "N"}]}), 2)
        self.assertEqual(r["pid"], 10)
        self.assertEqual(r["name"], "app.exe")
        self.assertEqual(r["kind"], "app")
        self.assertEqual(r["runtime"], "CUDA+cuDNN")
        self.assertEqual(r["gpu"], 2)
        self.assertEqual(r["gpu_pct"], 1.0)
        self.assertEqual(r["engine"], "GPU 2 - 3D")
        self.assertEqual(r["vram"], 100)
        self.assertIsNone(r["vram_inferred"])
        self.assertIsNone(r["vram_inferred_by"])
        self.assertEqual(r["shared"], 5)
        self.assertEqual(r["inside"], "model.gguf")
        self.assertEqual(r["engine_reason"], "why")
        self.assertEqual(r["segments"], [{"label": "weights", "bytes": 50, "kind": "weights", "exact": True}])
        self.assertEqual(r["models"], ["L", "N"])
        self.assertEqual(r["started_at"], 1000.5)
        self.assertIsNone(r["host"])

    def test_missing_fields(self):
        r = process_row({"pid": 1}, 0)
        self.assertEqual(r["name"], "?")
        self.assertEqual(r["kind"], "")
        self.assertEqual(r["runtime"], "")
        self.assertEqual(r["inside"], "")
        self.assertIsNone(r["vram"])
        self.assertIsNone(r["gpu_pct"])
        self.assertIsNone(r["engine"])
        self.assertEqual(r["segments"], [])
        self.assertEqual(r["models"], [])

    def test_inferred_only_when_not_measured(self):
        r = process_row(proc(vram_bytes=None, vram_inferred_bytes=70, vram_inferred_by="wsl"), 0)
        self.assertIsNone(r["vram"])
        self.assertEqual(r["vram_inferred"], 70)
        self.assertEqual(r["vram_inferred_by"], "wsl")
        r = process_row(proc(vram_bytes=0, vram_inferred_bytes=70, vram_inferred_by="wsl"), 0)
        self.assertEqual(r["vram"], 0)  # a measured zero stays zero
        self.assertIsNone(r["vram_inferred"])
        self.assertIsNone(r["vram_inferred_by"])

    def test_segment_unknown_bytes(self):
        r = process_row(proc(llm={"segments": [{"label": "kv", "bytes": None, "kind": "kv"}]}), 0)
        self.assertIsNone(r["segments"][0]["bytes"])
        self.assertFalse(r["segments"][0]["exact"])


class GpuViewTests(unittest.TestCase):
    def test_fields(self):
        g = gpu_view(snap(host_status="h", process_status=None, memory_note="m", counter_status=""), 1)
        expect = {"index": 1, "name": "Test GPU", "backend": "wddm", "driver": "1.0", "cuda": "13.0",
                  "driver_model": "WDDM", "pci": "0:2:0", "util": 7, "util_mem": 2, "enc": 0, "dec": 1,
                  "vram_used": 300, "vram_total": 1000, "shared_used": 20, "shared_total": 4000,
                  "attributed": 250, "temp": 50, "power_w": 3.5, "power_limit_w": 80.0, "fan": None,
                  "pstate": 8, "sm_clock": 180, "sm_clock_max": 3000, "mem_clock": 400, "pcie_rx_kbs": 10,
                  "pcie_tx_kbs": 2, "notes": ["h", "m"]}
        for k, v in expect.items():
            self.assertEqual(g[k], v, k)
        self.assertEqual(len(g["processes"]), 1)
        self.assertEqual(g["processes"][0]["gpu"], 1)

    def test_every_note_source(self):
        g = gpu_view(snap(host_status="a", process_status="b", memory_note="c", counter_status="d"), 0)
        self.assertEqual(g["notes"], ["a", "b", "c", "d"])

    def test_name_fallbacks(self):
        self.assertEqual(gpu_view({"device": {"name": "Dev"}}, 0)["name"], "Dev")
        g = gpu_view({}, 3)
        self.assertEqual(g["name"], "GPU 3")
        self.assertIsNone(g["util"])
        self.assertEqual(g["processes"], [])
        self.assertEqual(g["notes"], [])

    def test_error_view(self):
        e = error_view(2, "Gone", "boom")
        self.assertEqual((e["index"], e["name"], e["error"], e["notes"], e["processes"]), (2, "Gone", "boom", ["boom"], []))
        for k in ("util", "vram_used", "vram_total", "shared_used", "shared_total", "temp", "power_w"):
            self.assertIsNone(e[k], k)


class MergeTests(unittest.TestCase):
    def rows(self, i, *ps):
        return {"index": i, "processes": [process_row(p, i) for p in ps]}

    def test_sums_measured_across_gpus(self):
        out = merge_processes([
            self.rows(0, proc(vram_bytes=100, shared_bytes=None, gpu_pct=2.0, engines={"3D": 2.0}, kind="",
                              runtimes=[], inside="", started_at=None)),
            self.rows(1, proc(vram_bytes=None, shared_bytes=7, gpu_pct=9.0, engines={"Copy": 9.0}, kind="ai",
                              runtimes=["CUDA"], inside="m.gguf", started_at=5.0,
                              engine_unavailable_reason="r")),
        ])
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertEqual(r["vram"], 100)
        self.assertEqual(r["shared"], 7)
        self.assertEqual(r["gpu_pct"], 9.0)
        self.assertEqual(r["engine"], "GPU 1 - Copy")
        self.assertEqual(r["gpus"], [0, 1])
        self.assertEqual(r["kind"], "ai")
        self.assertEqual(r["runtime"], "CUDA")
        self.assertEqual(r["inside"], "m.gguf")
        self.assertEqual(r["started_at"], 5.0)
        self.assertEqual(r["engine_reason"], "r")
        self.assertEqual((r["pid"], r["name"], r["host"]), (10, "app.exe", None))

    def test_both_measured_add(self):
        out = merge_processes([self.rows(0, proc(vram_bytes=100)), self.rows(1, proc(vram_bytes=50))])
        self.assertEqual(out[0]["vram"], 150)
        self.assertEqual(out[0]["shared"], 10)

    def test_unknown_stays_unknown(self):
        out = merge_processes([self.rows(0, proc(vram_bytes=None, shared_bytes=None, gpu_pct=None, engines={},
                                                     started_at=None))])
        r = out[0]
        self.assertIsNone(r["vram"])
        self.assertIsNone(r["shared"])
        self.assertIsNone(r["gpu_pct"])
        self.assertIsNone(r["engine"])
        self.assertIsNone(r["vram_inferred"])
        self.assertIsNone(r["started_at"])

    def test_unknown_pct_never_wins_the_engine(self):
        out = merge_processes([self.rows(0, proc(gpu_pct=None, engines={"3D": 5.0})),
                               self.rows(1, proc(gpu_pct=2.0, engines={"Copy": 2.0}))])
        self.assertEqual(out[0]["engine"], "GPU 1 - Copy")

    def test_inferred_kept_apart(self):
        out = merge_processes([
            self.rows(0, proc(vram_bytes=None, vram_inferred_bytes=30, vram_inferred_by="a")),
            self.rows(1, proc(vram_bytes=None, vram_inferred_bytes=80, vram_inferred_by="b")),
            self.rows(2, proc(vram_bytes=40)),
        ])
        r = out[0]
        self.assertEqual(r["vram"], 40)  # inferred bytes never added to the measured total
        self.assertEqual(r["vram_inferred"], 80)
        self.assertEqual(r["vram_inferred_by"], "a")

    def test_zero_pct_beats_unknown(self):
        out = merge_processes([self.rows(0, proc(gpu_pct=None, engines={})),
                               self.rows(1, proc(gpu_pct=0.0, engines={"3D": 0.0}))])
        self.assertEqual(out[0]["gpu_pct"], 0.0)

    def test_distinct_processes_and_hosts(self):
        out = merge_processes([self.rows(0, proc(pid=1), proc(pid=2), proc(pid=1, host="wsl"),
                                         proc(pid=1, name="other.exe"))])
        self.assertEqual(len(out), 4)

    def test_segments_and_models_concatenate(self):
        llm1 = {"segments": [{"label": "w", "bytes": 1, "kind": "weights"}], "models": [{"label": "A"}]}
        llm2 = {"segments": [{"label": "kv", "bytes": 2, "kind": "kv"}], "models": [{"label": "B"}]}
        out = merge_processes([self.rows(0, proc(llm=llm1)), self.rows(1, proc(llm=llm2))])
        self.assertEqual([s["label"] for s in out[0]["segments"]], ["w", "kv"])
        self.assertEqual(out[0]["models"], ["A", "B"])

    def test_empty(self):
        self.assertEqual(merge_processes([]), [])
        self.assertEqual(merge_processes([{"index": 0, "processes": []}]), [])


class SystemTests(unittest.TestCase):
    def test_view(self):
        v = system_view({"cpu": {"pct": 12.5, "per_core": [1, None, float("nan")], "logical": 24, "physical": 12},
                         "ram": {"total": 100, "used": 60, "available": 40, "swap_total": 10, "swap_used": 1}})
        self.assertEqual(v, {"cpu_pct": 12.5, "per_core": [1, None, None], "logical": 24, "physical": 12,
                             "ram_used": 60, "ram_total": 100, "ram_available": 40, "swap_used": 1,
                             "swap_total": 10})

    def test_missing(self):
        self.assertIsNone(system_view(None))
        self.assertIsNone(system_view({}))
        v = system_view({"cpu": None, "ram": None, "x": 1})
        self.assertIsNone(v["cpu_pct"])
        self.assertEqual(v["per_core"], [])

    def test_record_history(self):
        h = History(5)
        gpus = [gpu_view(snap(), 0), error_view(1, "x", "e")]
        record_history(h, gpus, {"cpu_pct": 3, "ram_used": 9})
        record_history(h, gpus, None)
        d = h.to_dict()
        self.assertEqual(d["gpu0.util"], [7, 7])
        self.assertEqual(d["gpu0.vram"], [300, 300])
        self.assertEqual(d["gpu0.shared"], [20, 20])
        self.assertEqual(d["gpu0.temp"], [50, 50])
        self.assertEqual(d["gpu0.power"], [3.5, 3.5])
        self.assertEqual(d["gpu1.util"], [None, None])
        self.assertEqual(d["cpu"], [3, None])
        self.assertEqual(d["ram"], [9, None])


class FakeSources:
    """open_source stand-in: per-GPU callables, open failures, and close tracking."""

    def __init__(self, samples, open_errors=()):
        self.samples, self.open_errors = samples, set(open_errors)
        self.opened, self.closed = [], []

    def __call__(self, gpu, system):
        if gpu in self.open_errors:
            raise OSError(f"no gpu {gpu}")
        self.opened.append((gpu, system))

        @contextmanager
        def cm():
            try:
                yield self.samples[gpu]
            finally:
                self.closed.append(gpu)
        return cm()


class SamplerTests(unittest.TestCase):
    def test_bad_interval(self):
        for bad in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                sampler.Sampler(["a"], bad)

    def test_initial_state(self):
        s = sampler.Sampler(["a"], 2.0, history_size=4)
        st = s.state()
        self.assertFalse(st["ready"])
        self.assertEqual(st["interval"], 2.0)
        self.assertEqual(st["gpus"], [])
        self.assertIsNone(st["error"])
        self.assertEqual(s.history.size, 4)
        self.assertEqual(s.ticks, 0)

    def test_tick(self):
        s = sampler.Sampler(["g0", "g1", "g2"], 1.0)
        system = {"cpu": {"pct": 5}, "ram": {"used": 1, "total": 2}}

        def boom():
            raise RuntimeError("reset")
        st = s.tick({0: lambda: snap(system=system), 1: boom}, {2: "open failed"})
        self.assertTrue(st["ready"])
        self.assertIsNone(st["error"])
        self.assertEqual(st["interval"], 1.0)
        self.assertEqual([g.get("error") for g in st["gpus"]], [None, "RuntimeError: reset", "open failed"])
        self.assertEqual([g["name"] for g in st["gpus"]], ["Test GPU", "g1", "g2"])
        self.assertEqual(st["system"]["cpu_pct"], 5)
        self.assertEqual(len(st["processes"]), 1)
        self.assertEqual(st["history"]["gpu0.util"], [7])
        self.assertEqual(st["history"]["gpu1.util"], [None])
        self.assertAlmostEqual(st["taken"], time.time(), delta=5)
        self.assertIs(s.state(), st)
        self.assertEqual(s.ticks, 1)

    def test_tick_without_system(self):
        s = sampler.Sampler(["g0"], 1.0)
        st = s.tick({0: lambda: snap(system=None)}, {})
        self.assertIsNone(st["system"])
        self.assertEqual(st["history"]["cpu"], [None])

    def test_set_error_keeps_last_state(self):
        s = sampler.Sampler(["g0"], 1.0)
        s.tick({0: snap}, {})
        s._set_error("bad")
        self.assertEqual(s.state()["error"], "bad")
        self.assertTrue(s.state()["ready"])

    def test_thread_runs_and_closes_sources(self):
        fake = FakeSources({0: lambda: snap(system={"cpu": {"pct": 1}}), 1: lambda: snap(gpu="B")}, open_errors={2})
        s = sampler.Sampler(["a", "b", "c"], 0.01, open_source=fake)
        s.start()
        deadline = time.monotonic() + 5
        while s.ticks < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        s.stop()
        self.assertGreaterEqual(s.ticks, 3)
        self.assertEqual(fake.opened, [(0, True), (1, False)])
        self.assertEqual(sorted(fake.closed), [0, 1])
        st = s.state()
        self.assertEqual(st["gpus"][2]["error"], "OSError: no gpu 2")
        self.assertEqual(st["gpus"][1]["name"], "B")
        with self.assertRaises(RuntimeError):
            s.start()

    def test_tick_failure_is_reported_not_fatal(self):
        s = sampler.Sampler(["a"], 0.01, open_source=FakeSources({0: snap}))
        calls = []

        def bad_tick(samplers, failed):
            calls.append(1)
            raise ValueError("bad tick")
        s.tick = bad_tick
        s.start()
        deadline = time.monotonic() + 5
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        s.stop()
        self.assertGreaterEqual(len(calls), 2)  # kept looping after the first failure
        self.assertEqual(s.state()["error"], "ValueError: bad tick")

    def test_stop_before_open_returns(self):
        fake = FakeSources({0: snap})
        s = sampler.Sampler(["a"], 1.0, open_source=fake)
        s._stop.set()
        s._run()
        self.assertEqual(fake.opened, [])

    def test_interval_paces_ticks(self):
        s = sampler.Sampler(["a"], 0.2, open_source=FakeSources({0: snap}))
        s.start()
        time.sleep(0.5)
        s.stop()
        self.assertGreaterEqual(s.ticks, 2)
        self.assertLessEqual(s.ticks, 4)

    def test_stop_without_start(self):
        sampler.Sampler(["a"]).stop()

    def test_default_source_and_list_gpus(self):
        with mock.patch("gputaskman.record.source", return_value="cm") as src:
            self.assertEqual(sampler.default_source(1, system=False), "cm")
        src.assert_called_once_with(1, system=False)
        adapters = [types.SimpleNamespace(name="B"), types.SimpleNamespace(name="A")]
        with mock.patch("gputaskman.collector.enumerate_adapters", return_value=adapters), \
             mock.patch("gputaskman.collector.ordered_adapters", side_effect=lambda a: list(reversed(a))):
            self.assertEqual(sampler.list_gpus(), ["A", "B"])


class FakeProc:
    def __init__(self, name="app.exe", created=1000.0, wait_exc=None):
        self._name, self._created, self._wait_exc = name, created, wait_exc
        self.terminated = False
        self.waited = None

    def name(self):
        return self._name

    def create_time(self):
        return self._created

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.waited = timeout
        if self._wait_exc:
            raise self._wait_exc


class EndTaskTests(unittest.TestCase):
    def run_end(self, fake, pid=50, name="app.exe", started=None, **kw):
        with mock.patch.object(bridge.psutil, "Process", return_value=fake) as p:
            res = bridge.end_task(pid, name, started, own_pid=1, **kw)
        return res, p

    def test_invalid_pids(self):
        for bad in (None, "5", 5.0, True, -1):
            res = bridge.end_task(bad, "x", own_pid=1)
            self.assertFalse(res["ok"], bad)
            self.assertIn("valid", res["message"])

    def test_protected_and_own(self):
        for pid in (0, 4):
            res = bridge.end_task(pid, "System", own_pid=1)
            self.assertFalse(res["ok"])
            self.assertIn("system process", res["message"])
        res = bridge.end_task(77, "python.exe", own_pid=77)
        self.assertFalse(res["ok"])
        self.assertIn("itself", res["message"])

    def test_own_pid_defaults_to_this_process(self):
        import os
        res = bridge.end_task(os.getpid(), "python.exe")
        self.assertFalse(res["ok"])
        self.assertIn("itself", res["message"])

    def test_success(self):
        fake = FakeProc()
        res, p = self.run_end(fake, name="APP.EXE", started=1000.4, wait=2.5)
        self.assertEqual(res, {"ok": True, "message": "Ended app.exe (PID 50)."})
        p.assert_called_once_with(50)
        self.assertTrue(fake.terminated)
        self.assertEqual(fake.waited, 2.5)

    def test_default_wait(self):
        fake = FakeProc()
        self.run_end(fake)
        self.assertEqual(fake.waited, 3.0)

    def test_name_mismatch(self):
        fake = FakeProc(name="other.exe")
        res, _ = self.run_end(fake)
        self.assertFalse(res["ok"])
        self.assertIn("now 'other.exe'", res["message"])
        self.assertFalse(fake.terminated)

    def test_none_name_never_matches_real_name(self):
        res, _ = self.run_end(FakeProc(), name=None)
        self.assertFalse(res["ok"])

    def test_reused_pid(self):
        fake = FakeProc(created=2000.0)
        res, _ = self.run_end(fake, started=1000.0)
        self.assertFalse(res["ok"])
        self.assertIn("reused", res["message"])
        self.assertFalse(fake.terminated)

    def test_start_tolerance_edges(self):
        self.assertEqual(bridge.START_TOLERANCE, 1.0)
        res, _ = self.run_end(FakeProc(created=1001.0), started=1000.0)
        self.assertTrue(res["ok"])
        res, _ = self.run_end(FakeProc(created=999.0), started=1000.0)
        self.assertTrue(res["ok"])
        res, _ = self.run_end(FakeProc(created=1001.01), started=1000.0)
        self.assertFalse(res["ok"])

    def test_started_at_zero_is_still_checked(self):
        res, _ = self.run_end(FakeProc(created=1000.0), started=0)
        self.assertFalse(res["ok"])

    def test_started_at_as_string_number(self):
        res, _ = self.run_end(FakeProc(created=1000.0), started="1000.2")
        self.assertTrue(res["ok"])

    def test_still_running(self):
        res, _ = self.run_end(FakeProc(wait_exc=psutil.TimeoutExpired(3)))
        self.assertFalse(res["ok"])
        self.assertIn("still running", res["message"])

    def test_already_gone(self):
        with mock.patch.object(bridge.psutil, "Process", side_effect=psutil.NoSuchProcess(50)):
            res = bridge.end_task(50, "app.exe", own_pid=1)
        self.assertEqual(res, {"ok": True, "message": "app.exe (PID 50) has already exited."})

    def test_access_denied(self):
        fake = FakeProc()
        fake.terminate = mock.Mock(side_effect=psutil.AccessDenied(50))
        res, _ = self.run_end(fake)
        self.assertFalse(res["ok"])
        self.assertIn("administrator", res["message"])


class ApiTests(unittest.TestCase):
    def test_methods(self):
        s = types.SimpleNamespace(state=lambda: {"ready": True}, interval=2.0)
        api = bridge.Api(s, ["G"])
        self.assertEqual(api.get_state(), {"ready": True})
        with mock.patch.object(bridge, "app_version", return_value="9.9"):
            self.assertEqual(api.get_info(), {"version": "9.9", "gpus": ["G"], "interval": 2.0})
        with mock.patch.object(bridge, "end_task", return_value={"ok": True}) as et:
            self.assertEqual(api.end_task(5, "x", 1.0), {"ok": True})
        et.assert_called_once_with(5, "x", 1.0)

    def test_only_intended_methods_are_public(self):
        public = {n for n in dir(bridge.Api) if not n.startswith("_")}
        self.assertEqual(public, {"get_state", "get_info", "end_task"})

    def test_app_version(self):
        from importlib.metadata import PackageNotFoundError
        with mock.patch("importlib.metadata.version", return_value="1.2.3"):
            self.assertEqual(bridge.app_version(), "1.2.3")
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            self.assertEqual(bridge.app_version(), "dev")


class FakeWindow:
    def __init__(self, js_result='{"ready": true, "rows": 3}', exc=None):
        self.js_result, self.exc = js_result, exc
        self.destroyed = False
        self.evaluated = None

    def evaluate_js(self, code):
        self.evaluated = code
        if self.exc:
            raise self.exc
        return self.js_result

    def destroy(self):
        self.destroyed = True


class MainTests(unittest.TestCase):
    def test_parse_args(self):
        a = main.parse_args([])
        self.assertEqual((a.interval, a.debug, a.selftest), (1.0, False, None))
        a = main.parse_args(["-i", "0.5", "--debug", "--selftest", "3"])
        self.assertEqual((a.interval, a.debug, a.selftest), (0.5, True, 3.0))

    def test_parse_args_rejects(self):
        for argv in (["-i", "0"], ["-i", "-1"], ["-i", "nan"], ["-i", "inf"],
                     ["--selftest", "0"], ["--selftest", "-2"], ["--selftest", "nan"]):
            with self.assertRaises(SystemExit, msg=argv), mock.patch("sys.stderr"):
                main.parse_args(argv)

    def test_run_selftest(self):
        w, result = FakeWindow(), {}
        with mock.patch.object(threading.Event, "wait") as wait:
            main.run_selftest(w, 2.5, result)
        wait.assert_called_once_with(2.5)
        self.assertEqual(result, {"ready": True, "rows": 3})
        self.assertIn("gtmSelfTest", w.evaluated)
        self.assertTrue(w.destroyed)

    def test_run_selftest_error_still_closes(self):
        w, result = FakeWindow(exc=RuntimeError("js")), {}
        with mock.patch.object(threading.Event, "wait"):
            main.run_selftest(w, 1, result)
        self.assertEqual(result, {"error": "RuntimeError: js"})
        self.assertTrue(w.destroyed)

    def fake_webview(self, js='{"ready": true}'):
        wv = types.ModuleType("webview")
        wv.window = FakeWindow(js)
        wv.created, wv.started = None, None

        def create_window(*a, **kw):
            wv.created = (a, kw)
            return wv.window

        def start(func=None, args=None, debug=False):
            wv.started = {"func": func, "args": args, "debug": debug}
            if func:
                func(*args)
        wv.create_window, wv.start = create_window, start
        return wv

    def run_main(self, argv, wv, ticks=2):
        fake_sampler = mock.Mock(ticks=ticks)
        with mock.patch.dict(sys.modules, {"webview": wv}), \
             mock.patch("gputaskman.desktop.sampler.list_gpus", return_value=["G0"]), \
             mock.patch("gputaskman.desktop.sampler.Sampler", return_value=fake_sampler) as S, \
             mock.patch.object(threading.Event, "wait"), \
             mock.patch("builtins.print") as pr:
            code = main.main(argv)
        return code, fake_sampler, S, pr

    def test_main_normal(self):
        wv = self.fake_webview()
        code, smp, S, pr = self.run_main(["-i", "2", "--debug"], wv)
        self.assertEqual(code, 0)
        S.assert_called_once_with(["G0"], 2.0)
        smp.start.assert_called_once()
        smp.stop.assert_called_once()
        a, kw = wv.created
        self.assertEqual(a, (main.TITLE,))
        self.assertTrue(kw["url"].endswith("index.html"))
        self.assertEqual((kw["width"], kw["height"], kw["min_size"]), (1180, 760, (760, 500)))
        self.assertIsInstance(kw["js_api"], bridge.Api)
        self.assertEqual(wv.started, {"func": None, "args": None, "debug": True})
        pr.assert_not_called()

    def test_main_selftest_pass_and_fail(self):
        code, smp, _, pr = self.run_main(["--selftest", "1"], self.fake_webview('{"ready": true}'), ticks=4)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(pr.call_args[0][0]), {"ready": True, "sampler_ticks": 4})
        code, *_ = self.run_main(["--selftest", "1"], self.fake_webview('{"ready": false}'))
        self.assertEqual(code, 1)
        code, *_ = self.run_main(["--selftest", "1"], self.fake_webview('{"ready": true, "error": "x"}'))
        self.assertEqual(code, 1)

    def test_main_selftest_passes_debug(self):
        wv = self.fake_webview()
        self.run_main(["--selftest", "1", "--debug"], wv)
        self.assertTrue(wv.started["debug"])
        self.assertIs(wv.started["func"], main.run_selftest)
        self.assertEqual(wv.started["args"][1], 1.0)

    def test_main_stops_sampler_when_window_fails(self):
        wv = self.fake_webview()
        wv.start = mock.Mock(side_effect=RuntimeError("no gui"))
        with self.assertRaises(RuntimeError):
            _, smp, _, _ = self.run_main([], wv)
        # the Sampler mock is recreated per run_main; check via a direct run
        smp = mock.Mock(ticks=0)
        with mock.patch.dict(sys.modules, {"webview": wv}), \
             mock.patch("gputaskman.desktop.sampler.list_gpus", return_value=[]), \
             mock.patch("gputaskman.desktop.sampler.Sampler", return_value=smp), \
             self.assertRaises(RuntimeError):
            main.main([])
        smp.stop.assert_called_once()

    def test_main_without_pywebview(self):
        with mock.patch.dict(sys.modules, {"webview": None}), self.assertRaises(SystemExit) as cm:
            main.main([])
        self.assertIn("pywebview", str(cm.exception.code))

    def test_main_unsupported_platform(self):
        with mock.patch.object(main.sys, "platform", "sunos5"), self.assertRaises(SystemExit) as cm:
            main.main([])
        self.assertIn("supports", str(cm.exception.code))

    def test_ui_files_exist(self):
        for f in ("index.html", "app.css", "app.js"):
            self.assertTrue((main.UI_DIR / f).is_file(), f)
        html = (main.UI_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('src="app.js"', html)
        self.assertIn('href="app.css"', html)
        js = (main.UI_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("window.gtmSelfTest", js)
        for name in ("get_state", "get_info", "end_task"):
            self.assertIn(f"api().{name}(", js)

    def test_no_em_dashes_in_new_files(self):
        root = main.UI_DIR.parent
        for path in [*root.glob("*.py"), *main.UI_DIR.iterdir()]:
            self.assertNotIn("—", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
