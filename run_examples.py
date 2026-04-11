#!/usr/bin/env python3
"""
Cycle through every example in ``examples/`` for visual regression checking.

Each example runs as a subprocess for ``--dwell`` seconds (default 5), then
gets killed so the next one can start. If stdin is a TTY, pressing any key
skips ahead early; otherwise the runner auto-advances on a timer.

    uv run python run_examples.py                 # 5s per example
    uv run python run_examples.py --dwell 10      # 10s per example
    uv run python run_examples.py --filter 03     # only examples containing "03"

Kept in Python rather than bash for clarity, argparse, and cross-platform
support (POSIX termios + Windows msvcrt for the keypress-to-skip path).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"


def discover_examples(filter_str: str | None) -> list[Path]:
    pattern = re.compile(r"^\d{2}_.*\.py$")
    files = sorted(p for p in EXAMPLES_DIR.iterdir() if pattern.match(p.name))
    if filter_str:
        files = [p for p in files if filter_str in p.name]
    return files


def wait_or_keypress(dwell: float) -> str:
    """Sleep for `dwell` seconds, or return early if a key is pressed.

    Returns "key" if a keypress skipped the wait, "timeout" otherwise.
    Falls back to plain sleep when stdin is not a terminal.
    """
    if not sys.stdin.isatty():
        time.sleep(dwell)
        return "timeout"

    if sys.platform == "win32":
        import msvcrt

        deadline = time.monotonic() + dwell
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                msvcrt.getwch()
                return "key"
            time.sleep(0.05)
        return "timeout"

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        deadline = time.monotonic() + dwell
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if ready:
                sys.stdin.read(1)
                return "key"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwell", type=float, default=5.0, help="seconds per example")
    parser.add_argument("--filter", type=str, default=None, help="substring match")
    args = parser.parse_args()

    files = discover_examples(args.filter)
    if not files:
        print("No examples matched.", file=sys.stderr)
        return 1

    total = len(files)
    tty = sys.stdin.isatty()
    print(
        f"Running {total} examples ({args.dwell:.0f}s each"
        f"{', press any key to skip' if tty else ', auto-advance'}).\n"
    )

    for i, path in enumerate(files, start=1):
        rel = path.relative_to(ROOT)
        print(f"[{i}/{total}] starting {rel}")
        proc = subprocess.Popen(
            [sys.executable, str(path)],
            cwd=str(ROOT),
        )
        reason = wait_or_keypress(args.dwell)
        if proc.poll() is None:
            print(f"[{i}/{total}] killing (dwell {reason})")
            kill_proc(proc)
        else:
            print(f"[{i}/{total}] exited on its own")
        print()

    print(f"All {total} examples done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
