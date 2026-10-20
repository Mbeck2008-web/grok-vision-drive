#!/usr/bin/env python3
"""The offline runner skips Windows-only and BeamNG-dependent scripts cleanly.

A check skipped inside a passing script (``gvd_test_env.skip_check``) is
returned to the runner and counted, not folded into a plain pass.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_offline_suite as suite  # noqa: E402


def check_skip_tally() -> None:
    """skip_check lines reach the runner; a script that skips nothing reports none."""
    with tempfile.TemporaryDirectory() as tmp:
        skipper = Path(tmp) / "skipper.py"
        skipper.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
            "from gvd_test_env import skip_check\n"
            "skip_check('needs_gpu', 'no gpu here')\n"
            "skip_check('needs_torch', 'torch missing')\n",
            encoding="utf-8",
        )
        plain = Path(tmp) / "plain.py"
        plain.write_text("print('nothing skipped')\n", encoding="utf-8")
        status, code, skips = suite._run([sys.executable, str(skipper)], "skipper.py")
        assert (status, code) == ("pass", 0), (status, code)
        assert skips == [
            "SKIP-CHECK skipper.py needs_gpu: no gpu here",
            "SKIP-CHECK skipper.py needs_torch: torch missing",
        ], skips
        status, code, skips = suite._run([sys.executable, str(plain)], "plain.py")
        assert (status, code, skips) == ("pass", 0, []), (status, code, skips)
    print("skip tally: in-script skips are returned to the runner OK")


def main() -> None:
    scripts = list((ROOT / "scripts").glob("test_*.py"))
    assert scripts, "expected scripts/test_*.py"
    assert (ROOT / ".github" / "workflows" / "tests.yml").is_file()
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert "run_offline_suite.py" in workflow
    assert "xvfb-run -a python scripts/run_offline_suite.py" in workflow

    windows = ROOT / "scripts" / "test_m6_retail.py"
    assert suite._skip_reason(windows) is None
    # This file mentions the marker inside a string. That is not a directive.
    assert suite._skip_reason(Path(__file__)) is None

    fake_win = Path("/tmp/gvd_ci_windows_marker.py")
    fake_bng = Path("/tmp/gvd_ci_beamng_marker.py")
    fake_win.write_text("# gvd-ci: requires windows\n", encoding="utf-8")
    fake_bng.write_text("# gvd-ci: requires beamng\n", encoding="utf-8")
    try:
        if sys.platform == "win32":
            assert suite._skip_reason(fake_win) is None
        else:
            reason = suite._skip_reason(fake_win)
            assert reason and reason.startswith("windows-only"), reason
        if suite._beamng_installed():
            assert suite._skip_reason(fake_bng) is None
        else:
            reason = suite._skip_reason(fake_bng)
            assert reason and reason.startswith("BeamNG-dependent"), reason
    finally:
        fake_win.unlink(missing_ok=True)
        fake_bng.unlink(missing_ok=True)
    check_skip_tally()
    print("test_offline_suite: OK")


if __name__ == "__main__":
    main()
