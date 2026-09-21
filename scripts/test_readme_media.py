#!/usr/bin/env python3
"""Offline checks for README media (synthetic cabin + in-game HUD)."""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MEDIA = ROOT / "docs" / "media"
CABIN = MEDIA / "gvd_cabin_synthetic.png"
HUD = MEDIA / "gvd_ingame_ui_synthetic.png"
HUD_DRIVE = MEDIA / "gvd_ingame_ui_drive_synthetic.png"
SOURCE = MEDIA / "SOURCE.md"
README = ROOT / "README.md"
APP_ICON = ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "GVD" / "app.png"

CHROME_RE = re.compile(r"tesla|\bfsd\b|full self[- ]driving", re.I)


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path.name
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)


def main() -> None:
    assert CABIN.is_file() and HUD.is_file() and HUD_DRIVE.is_file() and SOURCE.is_file()
    cw, ch = _png_size(CABIN)
    hw, hh = _png_size(HUD)
    dw, dh = _png_size(HUD_DRIVE)
    iw, ih = _png_size(APP_ICON)
    assert (cw, ch) == (1280, 800), (cw, ch)
    assert iw == 64 and ih == 64, "app.png is the 64px icon, not the HUD shot"
    assert hw >= 280 and hh >= 400, (hw, hh)
    assert dw >= 280 and dh >= 400, (dw, dh)
    assert (hw, hh) != (iw, ih)

    import cv2

    cabin = cv2.imread(str(CABIN))
    hud = cv2.imread(str(HUD))
    drive = cv2.imread(str(HUD_DRIVE))
    assert cabin is not None and hud is not None and drive is not None
    assert int(cabin.mean()) < 80, "cabin must stay a void-stage render"
    assert 8 < float(hud.mean()) < 60, hud.mean()
    assert 8 < float(drive.mean()) < 60, drive.mean()

    # Ice token present (BGR ~ 212,196,158).
    ice_c = ((cabin[:, :, 0] > 170) & (cabin[:, :, 1] > 150) & (cabin[:, :, 2] > 120)).sum()
    ice_h = ((hud[:, :, 0] > 140) & (hud[:, :, 1] > 130) & (hud[:, :, 2] > 100)).sum()
    ice_d = ((drive[:, :, 0] > 140) & (drive[:, :, 1] > 130) & (drive[:, :, 2] > 100)).sum()
    assert ice_c > 200, ice_c
    assert ice_h > 200, ice_h
    assert ice_d > 200, ice_d

    readme = README.read_text(encoding="utf-8")
    src = SOURCE.read_text(encoding="utf-8")
    shot_block = readme.split("## Screenshots", 1)[1].split("## Audiences", 1)[0]
    for text in (shot_block, src):
        assert "gvd_cabin_synthetic.png" in text
        assert "gvd_ingame_ui_synthetic.png" in text
        assert "gvd_ingame_ui_drive_synthetic.png" in text
        assert "synthetic" in text.lower()
        assert CHROME_RE.search(text) is None, "screenshots/SOURCE must not add Tesla/FSD chrome"

    # Captions must not claim a live photo.
    assert "synthetic" in shot_block.lower()
    assert "not a live" in shot_block.lower()
    assert "Soft Esc parked" in shot_block
    assert "Not Engage" in shot_block
    assert "DISENGAGED" in shot_block
    assert "DRIVE" in shot_block
    assert "Not a live Engage" in shot_block

    assert "docs/media/gvd_cabin_synthetic.png" in readme
    assert "docs/media/gvd_ingame_ui_synthetic.png" in readme
    assert "docs/media/gvd_ingame_ui_drive_synthetic.png" in readme
    assert "render_readme_media.py" in readme

    # Parked cabin: regenerating with engaged=False stays the README dest.
    from python.viz.stage import smoke

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "cabin.png"
        out = smoke(use_perception=True, engaged=False, out=dest, write_bus=False)
        img = cv2.imread(str(out))
        assert img is not None and img.shape[1] == 1280 and img.shape[0] == 800
        assert int(img.mean()) < 80

    print("test_readme_media: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"test_readme_media: FAIL — {e}")
        sys.exit(1)
