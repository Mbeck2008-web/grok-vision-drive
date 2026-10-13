#!/usr/bin/env python3
"""Resident set and pipe-safe logs. No BeamNG."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.runtime.logio import safe_print  # noqa: E402
from python.run_vision import _rss_mb, _rss_mb_proc, _rss_mb_windows  # noqa: E402


def check_rss_is_live() -> None:
    if sys.platform.startswith("linux"):
        assert _rss_mb_proc() > 0.0, _rss_mb_proc()
        assert _rss_mb() > 0.0, _rss_mb()
    else:
        assert _rss_mb() >= 0.0
    if os_name_not_windows():
        assert _rss_mb_windows() == 0.0


def os_name_not_windows() -> bool:
    return __import__("os").name != "nt"


def check_arrow_survives_cp1252_pipe() -> None:
    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="strict", newline="\n")
    old = sys.stdout
    sys.stdout = wrapper
    try:
        safe_print("[GVD] beamngpy attached GVD→BeamNG vehicle-space convert")
        wrapper.flush()
    finally:
        sys.stdout = old
    text = buf.getvalue().decode("cp1252")
    assert "GVD" in text and "BeamNG" in text, text
    src = (ROOT / "python" / "sensors" / "cameras.py").read_text(encoding="utf-8")
    assert "safe_print(" in src
    assert "GVD→BeamNG" in src


def main() -> None:
    check_rss_is_live()
    check_arrow_survives_cp1252_pipe()
    print("test_host_metrics: OK")


if __name__ == "__main__":
    main()
