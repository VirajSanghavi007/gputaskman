# SPDX-License-Identifier: MIT
"""The object the window's JavaScript calls (pywebview js_api). Public methods only; state stays private."""

from __future__ import annotations

import os

import psutil

PROTECTED_PIDS = {0, 4}  # System Idle Process, System
START_TOLERANCE = 1.0  # seconds: started_at is rounded to ms, create_time is a float


def app_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("gputaskman")
    except PackageNotFoundError:
        return "dev"


def end_task(pid, name, started_at=None, *, own_pid: int | None = None, wait: float = 3.0) -> dict:
    """Terminate one process the table showed, after checking it is still that process.

    Refuses protected and own PIDs, a name that no longer matches, and a PID reused by a newer process
    (start time differs), so a stale table row can never end the wrong program.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
        return {"ok": False, "message": "Not a valid process ID."}
    if pid in PROTECTED_PIDS:
        return {"ok": False, "message": f"PID {pid} is a Windows system process and cannot be ended."}
    if pid == own_pid:
        return {"ok": False, "message": "That is GPU Task Manager itself. Close the window instead."}
    try:
        proc = psutil.Process(pid)
        actual = proc.name()
        if (actual or "").lower() != (name or "").lower():
            return {"ok": False, "message": f"PID {pid} is now {actual!r}, not {name!r}. Nothing was ended."}
        if started_at is not None and abs(proc.create_time() - float(started_at)) > START_TOLERANCE:
            return {"ok": False, "message": f"PID {pid} was reused by a newer {actual}. Nothing was ended."}
        proc.terminate()
        try:
            proc.wait(wait)
        except psutil.TimeoutExpired:
            return {"ok": False, "message": f"{actual} (PID {pid}) was asked to end but is still running."}
    except psutil.NoSuchProcess:
        return {"ok": True, "message": f"{name} (PID {pid}) has already exited."}
    except psutil.AccessDenied:
        return {"ok": False, "message": f"Access denied ending {name} (PID {pid}). It may need administrator rights."}
    return {"ok": True, "message": f"Ended {actual} (PID {pid})."}


class Api:
    def __init__(self, sampler, gpus: list[str]) -> None:
        self._sampler = sampler
        self._gpus = gpus

    def get_state(self) -> dict:
        return self._sampler.state()

    def get_info(self) -> dict:
        return {"version": app_version(), "gpus": self._gpus, "interval": self._sampler.interval}

    def end_task(self, pid, name, started_at=None) -> dict:
        return end_task(pid, name, started_at)
