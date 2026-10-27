#!/usr/bin/env python3
"""The suite's torch / torchvision pin check: exact versions, local suffix allowed,
enforced only on Windows (the dev PC target), logged everywhere."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_offline_suite as suite  # noqa: E402


def check_pin_matches() -> None:
    assert suite.TORCH_PINS == {"torch": "2.14.1", "torchvision": "0.29.1"}, suite.TORCH_PINS
    for ok in ("2.14.1", "2.14.1+cu126", "2.14.1+cpu"):
        assert suite.pin_matches(ok, "2.14.1"), ok
    for bad in (None, "", "2.14.0", "2.14.10", "2.14", "12.14.1", "2.14.1rc1", "2.14.1.post1", "2.14.1+", "2.15.0+cu126"):
        assert not suite.pin_matches(bad, "2.14.1"), bad
    print("pin_matches: exact version, optional +local suffix only")


def check_problems_and_platform() -> None:
    good = {"torch": "2.14.1+cu126", "torchvision": "0.29.1+cu126"}
    assert suite.torch_pin_problems(good) == []
    assert not suite.torch_pins_fatal(good, "win32")
    assert suite.torch_version_line(good) == "[suite] torch 2.14.1+cu126, torchvision 0.29.1+cu126"

    missing = {"torch": "2.14.1+cu126", "torchvision": None}
    assert suite.torch_pin_problems(missing) == ["torchvision is not installed, need 0.29.1"], suite.torch_pin_problems(missing)
    assert suite.torch_version_line(missing) == "[suite] torch 2.14.1+cu126, torchvision not installed"
    assert suite.torch_pins_fatal(missing, "win32")
    assert not suite.torch_pins_fatal(missing, "linux")

    wrong = {"torch": "2.13.0", "torchvision": "0.29.1"}
    assert suite.torch_pin_problems(wrong) == ["torch is 2.13.0, need 2.14.1"], suite.torch_pin_problems(wrong)
    assert suite.torch_pins_fatal(wrong, "win32")
    assert not suite.torch_pins_fatal(wrong, "linux")
    assert not suite.torch_pins_fatal({}, "linux") and suite.torch_pins_fatal({}, "win32")
    print("pins: fatal only on win32; Linux logs the line")


def check_child_reader() -> None:
    versions = suite._torch_versions()
    assert set(versions) == set(suite.TORCH_PINS), versions
    assert all(v is None or isinstance(v, str) for v in versions.values()), versions
    print(f"child reader: {suite.torch_version_line(versions)}")


def main() -> None:
    check_pin_matches()
    check_problems_and_platform()
    check_child_reader()
    print("test_torch_pins: OK")


if __name__ == "__main__":
    main()
