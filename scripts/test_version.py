#!/usr/bin/env python3
"""Offline pin for GVD alpha 1.0.1 (no BeamNG). Canonical spots must stay in lockstep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_release_zip as rel  # noqa: E402
import python as gvd  # noqa: E402

PIN = "1.0.1"
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

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"## Alpha {PIN}" in readme
    assert "This is **alpha** — may break / not work; improves with fixes." in readme
    assert "**point** bumps" in readme
    assert "fixes / small UI" in readme
    assert "**main alpha** bump" in readme
    assert "features / core / UI overhaul" in readme
    assert f"{PIN}-alpha-<sha>" in readme
    assert "Soft Esc parked" in readme
    assert "m6-<sha>" not in readme

    banner = rel.version_text(PIN)
    assert f"Alpha {PIN}: may break / not work. Improves with fixes." in banner
    print("test_version: OK")


if __name__ == "__main__":
    main()
