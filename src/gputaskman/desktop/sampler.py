# SPDX-License-Identifier: MIT
"""Background sampling for the desktop window: one `record.source` per GPU, read on a fixed interval."""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack

from .model import HISTORY, History, error_view, gpu_view, merge_processes, record_history, system_view


def list_gpus() -> list[str]:
    """Adapter names in the same order `--list-gpus` numbers them."""
    from ..collector import enumerate_adapters, ordered_adapters

    return [a.name for a in ordered_adapters(enumerate_adapters())]


def default_source(gpu: int, system: bool):
    from ..record import source

    return source(gpu, system=system)


class Sampler:
    """Reads every GPU each interval on its own thread; `state()` hands the window the latest view.

    One GPU failing (driver reset, adapter removed) turns that GPU's card into an error with the reason;
    the others keep updating. System RAM/CPU is read once per tick, by the first GPU's source.
    """

    def __init__(self, gpus: list[str], interval: float = 1.0, *, open_source=default_source,
                 history_size: int = HISTORY) -> None:
        if not interval > 0:
            raise ValueError("interval must be greater than zero")
        self.gpus = list(gpus)
        self.interval = interval
        self._open_source = open_source
        self.history = History(history_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0
        self._state: dict = {"ready": False, "gpus": [], "processes": [], "system": None, "history": {},
                             "interval": interval, "taken": None, "error": None}

    # --- one tick, separate from the thread so it is testable without timing ---
    def tick(self, samplers: dict[int, object], failed: dict[int, str]) -> dict:
        views, system = [], None
        for i, name in enumerate(self.gpus):
            if i in failed:
                views.append(error_view(i, name, failed[i]))
                continue
            try:
                snap = samplers[i]()
            except Exception as error:  # one GPU failing must not stop the others
                views.append(error_view(i, name, f"{type(error).__name__}: {error}"))
                continue
            views.append(gpu_view(snap, i))
            if snap.get("system") is not None:
                system = system_view(snap["system"])
        record_history(self.history, views, system)
        state = {"ready": True, "gpus": views, "processes": merge_processes(views), "system": system,
                 "history": self.history.to_dict(), "interval": self.interval, "taken": time.time(), "error": None}
        with self._lock:
            self._state = state
            self.ticks += 1
        return state

    def state(self) -> dict:
        with self._lock:
            return self._state

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state = {**self._state, "error": message}

    def _run(self) -> None:
        samplers, failed = {}, {}
        with ExitStack() as stack:
            for i, _ in enumerate(self.gpus):
                if self._stop.is_set():
                    return
                try:
                    samplers[i] = stack.enter_context(self._open_source(i, system=(i == 0)))
                except Exception as error:
                    failed[i] = f"{type(error).__name__}: {error}"
            due = time.monotonic()
            while not self._stop.is_set():
                try:
                    self.tick(samplers, failed)
                except Exception as error:  # keep the window alive; show why it is not updating
                    self._set_error(f"{type(error).__name__}: {error}")
                due = max(due + self.interval, time.monotonic())
                self._stop.wait(max(0.0, due - time.monotonic()))

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._thread = threading.Thread(target=self._run, name="gputaskman-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
