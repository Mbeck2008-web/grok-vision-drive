#!/usr/bin/env python3
"""Offline pin for GVD alpha 1.7.13 (no BeamNG). Canonical spots must stay in lockstep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_release_zip as rel  # noqa: E402
import python as gvd  # noqa: E402

PIN = "1.7.13"
CHANNEL = "alpha"


def main() -> None:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").splitlines()[0].strip()
    assert version == PIN, version
    assert gvd.__version__ == PIN, gvd.__version__
    assert gvd.__release__ == CHANNEL, gvd.__release__
    assert rel.VERSION == PIN, rel.VERSION
    assert rel.RELEASE_CHANNEL == CHANNEL, rel.RELEASE_CHANNEL
    assert rel.RELEASE_LABEL == f"{PIN}-{CHANNEL}", rel.RELEASE_LABEL

    app = json.loads(
        (ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "GVD" / "app.json").read_text(
            encoding="utf-8"
        )
    )
    strip = json.loads(
        (
            ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "gvd_strip" / "app.json"
        ).read_text(encoding="utf-8")
    )
    assert app["version"] == PIN, app["version"]
    assert strip["version"] == PIN, strip["version"]

    # The mod stamps its version on gvd_ego.json; the supervisor refuses below its minimum.
    import re

    from python.control.echo_link import MIN_MOD_VERSION, parse_version

    main_lua = (ROOT / "beamng_mod" / "lua" / "ge" / "extensions" / "gvd" / "main.lua").read_text(encoding="utf-8")
    m = re.search(r"M\.MOD_VERSION\s*=\s*'([^']+)'", main_lua)
    assert m and m.group(1) == PIN, m and m.group(1)
    assert parse_version(MIN_MOD_VERSION) <= parse_version(PIN), (MIN_MOD_VERSION, PIN)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # The alpha warning and the pin open the README, ahead of the quick start.
    assert f"This is an **alpha** (`VERSION` {PIN}). It may break." in readme
    assert readme.index("This is an **alpha**") < readme.index("## Quick start")
    assert f"Canonical pin **{PIN}**" in readme
    major, minor = PIN.split(".")[:2]
    assert f"Point bumps (`{major}.{minor}.x`) are fixes and small UI." in readme
    assert f"A main alpha bump (`{major}.x.0`) is features, core, or a UI overhaul." in readme
    assert f"{PIN}-alpha-<sha>" in readme
    assert "Soft Esc parked" in readme
    assert "m6-<sha>" not in readme

    banner = rel.version_text(PIN)
    assert f"Alpha {PIN}: may break / not work. Improves with fixes." in banner
    print("test_version: OK")


if __name__ == "__main__":
    main()
