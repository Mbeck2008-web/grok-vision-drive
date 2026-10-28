#!/usr/bin/env python3
"""The offline runner skips Windows-only and BeamNG-dependent scripts cleanly.

A check skipped inside a passing script (``gvd_test_env.skip_check``) is
returned to the runner and counted, not folded into a plain pass.

The test-only collector root variable (``GVD_TEST_DATA_DIR``) is named only
in gvd_test_env.py, this file, and docs (*.md): no product file reads it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_offline_suite as suite  # noqa: E402

TEST_DATA_LITERAL = "GVD_TEST_DATA_DIR"
TEST_DATA_ALLOWED = {"scripts/gvd_test_env.py", "scripts/test_offline_suite.py"}


def _tracked_files() -> list[str]:
    """Repo-relative tracked paths (``git ls-files``), else a walk of the source folders."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
        names = [name for name in out.decode("utf-8", errors="replace").split("\0") if name]
        if names:
            return names
    except (OSError, subprocess.SubprocessError):
        pass
    names = []
    for folder in ("python", "scripts", "beamng_mod"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file():
                names.append(path.relative_to(ROOT).as_posix())
    for pattern in ("*.bat", "*.py"):
        names.extend(path.name for path in ROOT.glob(pattern) if path.is_file())
    return names


def check_test_data_env_only_in_tests() -> None:
    import gvd_test_env

    assert gvd_test_env.TEST_DATA_ENV == TEST_DATA_LITERAL
    assert suite.TEST_DATA_ENV == TEST_DATA_LITERAL
    names = _tracked_files()
    assert "scripts/run_offline_suite.py" in names and "scripts/gvd_test_env.py" in names, len(names)
    needle = TEST_DATA_LITERAL.encode("ascii")
    found = []
    for name in names:
        try:
            data = (ROOT / name).read_bytes()
        except OSError:
            continue
        if needle in data:
            found.append(name)
    bad = [name for name in found if name not in TEST_DATA_ALLOWED and not name.lower().endswith(".md")]
    assert not bad, f"{TEST_DATA_LITERAL} named outside the test allow-list: {bad}"
    assert not [name for name in found if name.startswith("python/")], found
    assert "scripts/gvd_test_env.py" in found, found
    print(f"{TEST_DATA_LITERAL}: named only in {sorted(found)} ({len(names)} tracked files checked)")


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


def check_thread_env() -> None:
    """Each test process gets one OpenMP / MKL thread unless the caller chose a count."""
    from gvd_test_env import THREAD_ENV, isolated_env

    with tempfile.TemporaryDirectory() as tmp:
        env = isolated_env({"PATH": "/bin"}, Path(tmp) / "a")
        assert all(env[name] == "1" for name in THREAD_ENV), env
        env = isolated_env({"PATH": "/bin", "OMP_NUM_THREADS": "4"}, Path(tmp) / "b")
        assert env["OMP_NUM_THREADS"] == "4" and env["MKL_NUM_THREADS"] == "1", env
    print("thread env: one OpenMP / MKL thread per test unless set OK")


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

    with tempfile.TemporaryDirectory() as tmp:
        fake_win = Path(tmp) / "gvd_ci_windows_marker.py"
        fake_bng = Path(tmp) / "gvd_ci_beamng_marker.py"
        fake_win.write_text("# gvd-ci: requires windows\n", encoding="utf-8")
        fake_bng.write_text("# gvd-ci: requires beamng\n", encoding="utf-8")
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
    check_skip_tally()
    check_thread_env()
    check_test_data_env_only_in_tests()
    print("test_offline_suite: OK")


if __name__ == "__main__":
    main()
