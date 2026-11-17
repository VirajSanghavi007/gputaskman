# SPDX-License-Identifier: MIT
"""gputaskman-app: open the GPU Task Manager window."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from pathlib import Path

UI_DIR = Path(__file__).parent / "ui"
ICON = Path(__file__).parent / "icon.ico"
TITLE = "GPU Task Manager"
# Windows groups taskbar buttons by AppUserModelID. Without our own, the window joins pythonw.exe's group and
# shows Python's icon. A Start Menu shortcut stamped with the same ID shares the button and its pin.
APP_ID = "VirajSanghavi.GPUTaskManager"


def set_app_id() -> bool:
    """Give this process its own taskbar identity on Windows. Returns whether it was set."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID) == 0  # S_OK
    except Exception:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="gputaskman-app", description="GPU Task Manager: a desktop window over gputaskman.")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    ap.add_argument("--debug", action="store_true", help="enable the web inspector (right-click > Inspect)")
    # Smoke tests: open the real window, wait, print what the page rendered as JSON, close, and exit.
    ap.add_argument("--selftest", type=float, metavar="SECONDS", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--interval must be a finite number greater than zero")
    if args.selftest is not None and (not math.isfinite(args.selftest) or args.selftest <= 0):
        ap.error("--selftest must be a positive number of seconds")
    return args


def run_selftest(window, seconds: float, result: dict) -> None:
    """Wait, ask the page what it drew, then close the window. Runs on pywebview's start thread."""
    threading.Event().wait(seconds)
    try:
        result.update(json.loads(window.evaluate_js("JSON.stringify(window.gtmSelfTest())")))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    window.destroy()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"):
        sys.exit("gputaskman-app supports Windows, Linux and macOS only.")
    try:
        import webview
    except ImportError:
        sys.exit("gputaskman-app needs pywebview: install with  uv tool install \"gputaskman[app]\"")

    from .bridge import Api
    from .sampler import Sampler, list_gpus

    set_app_id()
    gpus = list_gpus()
    sampler = Sampler(gpus, args.interval)
    sampler.start()
    window = webview.create_window(TITLE, url=str(UI_DIR / "index.html"), js_api=Api(sampler, gpus),
                                   width=1180, height=760, min_size=(760, 500))
    result: dict = {}
    icon = str(ICON) if ICON.is_file() else None
    try:
        if args.selftest is not None:
            webview.start(run_selftest, (window, args.selftest, result), debug=args.debug, icon=icon)
        else:
            webview.start(debug=args.debug, icon=icon)
    finally:
        sampler.stop()
    if args.selftest is not None:
        result["sampler_ticks"] = sampler.ticks
        print(json.dumps(result))
        return 0 if result.get("ready") and not result.get("error") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
