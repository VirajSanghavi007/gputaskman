# SPDX-License-Identifier: MIT
"""Turn `--json` snapshot dicts into what the desktop window draws.

Pure functions only (no sampling, no GUI), so every rule here is unit tested. The provenance rules from
AGENTS.md hold: unknown stays None (the UI shows n/a, never 0), only measured bytes are summed, and an
inferred VRAM figure is shown with its own flag and never added to a total.
"""

from __future__ import annotations

from collections import deque

HISTORY = 60  # samples kept per series: one minute at the default 1 s interval


class History:
    """Fixed-length series per key, for the Performance graphs. None (unknown) is kept as a gap, not a zero."""

    def __init__(self, size: int = HISTORY) -> None:
        if size < 1:
            raise ValueError("history size must be at least 1")
        self.size = size
        self._series: dict[str, deque] = {}

    def push(self, key: str, value: float | None) -> None:
        series = self._series.get(key)
        if series is None:
            series = self._series[key] = deque(maxlen=self.size)
        series.append(value)

    def get(self, key: str) -> list:
        return list(self._series.get(key, ()))

    def to_dict(self) -> dict[str, list]:
        return {key: list(series) for key, series in self._series.items()}


def _num(value):
    """A finite number, or None. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def top_engine(engines: dict | None) -> tuple[str, float] | None:
    """The busiest engine as (name, pct), or None when engines are unknown or all idle."""
    if not engines:
        return None
    name, pct = max(engines.items(), key=lambda item: (item[1], item[0]))
    return (name, pct) if _num(pct) is not None and pct > 0 else None


def process_row(p: dict, gpu_index: int) -> dict:
    """One process on one GPU, flattened for the table."""
    measured = _num(p.get("vram_bytes"))
    inferred = _num(p.get("vram_inferred_bytes")) if measured is None else None
    engine = top_engine(p.get("engines"))
    llm = p.get("llm") or {}
    return {
        "pid": p.get("pid"),
        "name": p.get("name") or "?",
        "host": p.get("host"),
        "kind": p.get("kind") or "",
        "runtime": "+".join(p.get("runtimes") or []),
        "gpu": gpu_index,
        "gpu_pct": _num(p.get("gpu_pct")),
        "engine": f"GPU {gpu_index} - {engine[0]}" if engine else None,
        "vram": measured,
        "vram_inferred": inferred,
        "vram_inferred_by": p.get("vram_inferred_by") if inferred is not None else None,
        "shared": _num(p.get("shared_bytes")),
        "inside": p.get("inside") or "",
        "engine_reason": p.get("engine_unavailable_reason"),
        "segments": [{"label": s.get("label"), "bytes": _num(s.get("bytes")), "kind": s.get("kind"),
                      "exact": bool(s.get("exact"))} for s in llm.get("segments") or []],
        "models": [m.get("label") or m.get("name") for m in llm.get("models") or []],
        "started_at": p.get("started_at"),
    }


def gpu_view(snap: dict, index: int) -> dict:
    """One GPU's card for the Performance page plus its processes."""
    device = snap.get("device") or {}
    adapter = snap.get("adapter") or {}
    notes = [n for n in (snap.get("host_status"), snap.get("process_status"), snap.get("memory_note"),
                         snap.get("counter_status")) if n]
    return {
        "index": index,
        "name": snap.get("gpu") or device.get("name") or f"GPU {index}",
        "backend": snap.get("backend"),
        "driver": snap.get("driver"),
        "cuda": snap.get("cuda"),
        "driver_model": device.get("driver_model"),
        "pci": device.get("pci_bus"),
        "util": _num(device.get("util_gpu")),
        "util_mem": _num(device.get("util_mem")),
        "enc": _num(device.get("enc")),
        "dec": _num(device.get("dec")),
        "vram_used": _num(snap.get("vram_used")),
        "vram_total": _num(snap.get("vram_total")),
        "shared_used": _num(snap.get("shared_used")),
        "shared_total": _num(adapter.get("shared_system")),
        "attributed": _num(snap.get("attributed")),
        "temp": _num(device.get("temp")),
        "power_w": _num(device.get("power_w")),
        "power_limit_w": _num(device.get("power_limit_w")),
        "fan": _num(device.get("fan")),
        "pstate": device.get("pstate"),
        "sm_clock": _num(device.get("sm_clock")),
        "sm_clock_max": _num(device.get("sm_clock_max")),
        "mem_clock": _num(device.get("mem_clock")),
        "pcie_rx_kbs": _num(device.get("pcie_rx_kbs")),
        "pcie_tx_kbs": _num(device.get("pcie_tx_kbs")),
        "notes": notes,
        "processes": [process_row(p, index) for p in snap.get("processes") or []],
    }


def error_view(index: int, name: str, error: str) -> dict:
    """A GPU that could not be sampled: every reading unknown, with the reason."""
    return {"index": index, "name": name, "error": error, "notes": [error], "processes": [],
            "util": None, "vram_used": None, "vram_total": None, "shared_used": None, "shared_total": None,
            "temp": None, "power_w": None}


def _sum_measured(values: list) -> int | None:
    """Sum of the known values; None when none is known. Unknown never counts as zero."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def merge_processes(gpus: list[dict]) -> list[dict]:
    """One row per process across all GPUs, like Task Manager's Processes tab.

    Measured VRAM and shared bytes are summed over GPUs; inferred VRAM is kept apart and never added to the
    measured figure. GPU% is the busiest GPU's, and the engine column names that GPU and engine.
    """
    merged: dict[tuple, dict] = {}
    for gpu in gpus:
        for row in gpu["processes"]:
            key = (row["host"], row["pid"], row["name"])
            parts = merged.setdefault(key, [])
            parts.append(row)
    out = []
    for (host, pid, name), parts in merged.items():
        busiest = max(parts, key=lambda r: r["gpu_pct"] if r["gpu_pct"] is not None else -1.0)
        pcts = [r["gpu_pct"] for r in parts if r["gpu_pct"] is not None]
        inferred = [r["vram_inferred"] for r in parts if r["vram_inferred"] is not None]
        out.append({
            "pid": pid, "name": name, "host": host,
            "kind": next((r["kind"] for r in parts if r["kind"]), ""),
            "runtime": next((r["runtime"] for r in parts if r["runtime"]), ""),
            "gpus": sorted({r["gpu"] for r in parts}),
            "gpu_pct": max(pcts) if pcts else None,
            "engine": busiest["engine"],
            "vram": _sum_measured([r["vram"] for r in parts]),
            "vram_inferred": max(inferred) if inferred else None,
            "shared": _sum_measured([r["shared"] for r in parts]),
            "inside": next((r["inside"] for r in parts if r["inside"]), ""),
            "segments": [s for r in parts for s in r["segments"]],
            "models": [m for r in parts for m in r["models"]],
            "started_at": next((r["started_at"] for r in parts if r["started_at"] is not None), None),
            "vram_inferred_by": next((r["vram_inferred_by"] for r in parts if r["vram_inferred_by"]), None),
            "engine_reason": next((r["engine_reason"] for r in parts if r["engine_reason"]), None),
        })
    return out


def system_view(system: dict | None) -> dict | None:
    if not system:
        return None
    cpu = system.get("cpu") or {}
    ram = system.get("ram") or {}
    return {
        "cpu_pct": _num(cpu.get("pct")),
        "per_core": [_num(c) for c in cpu.get("per_core") or []],
        "logical": cpu.get("logical"),
        "physical": cpu.get("physical"),
        "ram_used": _num(ram.get("used")),
        "ram_total": _num(ram.get("total")),
        "ram_available": _num(ram.get("available")),
        "swap_used": _num(ram.get("swap_used")),
        "swap_total": _num(ram.get("swap_total")),
    }


def record_history(history: History, gpus: list[dict], system: dict | None) -> None:
    """Append this tick's readings. A GPU that failed this tick still gets a gap so series stay aligned."""
    for gpu in gpus:
        i = gpu["index"]
        history.push(f"gpu{i}.util", gpu.get("util"))
        history.push(f"gpu{i}.vram", gpu.get("vram_used"))
        history.push(f"gpu{i}.shared", gpu.get("shared_used"))
        history.push(f"gpu{i}.temp", gpu.get("temp"))
        history.push(f"gpu{i}.power", gpu.get("power_w"))
    history.push("cpu", system.get("cpu_pct") if system else None)
    history.push("ram", system.get("ram_used") if system else None)
