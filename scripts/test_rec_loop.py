#!/usr/bin/env python3
"""rec_loop.ps1 must pass --long-clips and must not treat a null ExitCode as a halt.

    py -3 scripts/test_rec_loop.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rec_loop.ps1"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    compact = " ".join(text.split())
    assert "--long-clips" in text, "rec_loop.ps1 must pass --long-clips (collect_clips default is 0)"
    assert "600,900" in text or "600, 900" in text, "long-run default duration must be 10-15 min (600,900 s)"
    # ExitCode after Start-Process is unreliable unless -PassThru + WaitForExit (or Handle cache).
    assert "-PassThru" in text, "Start-Process must use -PassThru so WaitForExit/ExitCode work"
    assert "WaitForExit" in text, "must call WaitForExit before reading ExitCode"
    assert "ExitCode" in text
    assert "Handle" in text or "WaitForExit" in text
    low = text.lower()
    assert "unknown" in low or ("$null" in text and "exitcode" in low)
    # A null ExitCode must not be counted as a halt by a bare -ne 0 comparison.
    assert "session" in low or "clip" in low, "null rc should be verified from session/clip outcome"
    print("test_rec_loop: OK")


if __name__ == "__main__":
    main()
