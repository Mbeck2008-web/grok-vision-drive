#!/usr/bin/env python3
"""Render README media under docs/media/ (synthetic cabin + in-game HUD).

No recent live IQU45 GVD viz/UI photos exist in-repo. Cabin comes from
``python.viz.stage.smoke`` (parked, engaged=False). In-game UI is a raster of
``beamng_mod/ui/modules/apps/GVD/app.html`` in DISENGAGED / Soft Esc parked
(Chrome headless, OpenCV fallback). Captions must stay **synthetic**.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MEDIA = ROOT / "docs" / "media"
CABIN = MEDIA / "gvd_cabin_synthetic.png"
HUD = MEDIA / "gvd_ingame_ui_synthetic.png"
APP_HTML = ROOT / "beamng_mod" / "ui" / "modules" / "apps" / "GVD" / "app.html"

# Tiny on-frame stamp so GitHub unfurls stay honest without the README caption.
STAMP = "synthetic"


def _stamp(img, text: str = STAMP, *, corner: str = "tr") -> None:
    import cv2

    h, w = img.shape[:2]
    scale = 0.42 if w >= 800 else 0.38
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 8
    if corner == "tr":
        x, y = w - tw - pad, 16
    else:
        x, y = pad, h - pad
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (114, 124, 134), thick, cv2.LINE_AA)


def render_cabin(dest: Path = CABIN) -> Path:
    from python.viz.stage import smoke
    import cv2

    out = smoke(use_perception=True, engaged=False, out=dest, write_bus=False)
    img = cv2.imread(str(out))
    if img is None:
        raise RuntimeError(f"cabin write failed: {out}")
    _stamp(img, corner="tr")
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return out


def _app_css() -> str:
    html = APP_HTML.read_text(encoding="utf-8")
    m = re.search(r"<style>(.*?)</style>", html, re.S)
    if not m:
        raise RuntimeError("GVD app.html has no <style> block")
    return m.group(1).strip()


def _hud_html() -> str:
    """Parked DISENGAGED snapshot. Values match app.js with supervisor live, engage off."""
    css = _app_css()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GVD Apps HUD synthetic</title>
<style>
html, body {{
  margin: 0;
  background: #07080a;
}}
{_app_css_override(css)}
</style>
</head>
<body>
<div class="gvd-app bngApp">
  <div class="gvd-row gvd-head">
    <span class="gvd-brand">GVD</span>
    <span class="gvd-word">VISION</span>
    <span class="gvd-grow"></span>
    <span class="gvd-link is-live">
      <b class="gvd-dot"></b>link live
    </span>
  </div>

  <div class="gvd-state">
    <div class="gvd-state-main">DISENGAGED</div>
    <div class="gvd-state-sub">standby</div>
  </div>

  <button type="button" class="gvd-engage">
    <span>ENGAGE</span><i>ALT+G</i>
  </button>

  <div class="gvd-sec">
    <div class="gvd-sec-h">WHEEL / PEDALS <i class="gvd-note">applied electrics (no player device)</i></div>
    <div class="gvd-axis">
      <span>wheel</span>
      <div class="gvd-track">
        <i class="gvd-mid"></i>
        <i class="gvd-mark" style="left:50%"></i>
      </div>
      <b>0.00</b>
    </div>
    <div class="gvd-axis">
      <span>gas</span>
      <div class="gvd-track">
        <i class="gvd-fill" style="width:0%"></i>
      </div>
      <b>0.00</b>
    </div>
    <div class="gvd-axis">
      <span>brake</span>
      <div class="gvd-track">
        <i class="gvd-fill" style="width:0%"></i>
      </div>
      <b>0.00</b>
    </div>
    <div class="gvd-sub">applied wheel 0.00  gas 0.00  brake 0.00</div>
  </div>

  <div class="gvd-sec">
    <div class="gvd-sec-h">VIEW</div>
    <div class="gvd-chips">
      <button type="button" class="gvd-chip is-on">Path</button>
      <button type="button" class="gvd-chip">Ghosts</button>
    </div>
    <div class="gvd-sub">conf 0.90 - 2.0 m - preview path - 3 tracked</div>
  </div>

  <div class="gvd-sec">
    <div class="gvd-sec-h">POLICY <i class="gvd-note">nets, path, planner always run - engage takes the wheel</i></div>
    <div class="gvd-seg">
      <button type="button" class="gvd-segbtn is-on">modular</button>
      <button type="button" class="gvd-segbtn">e2e</button>
      <button type="button" class="gvd-segbtn">shadow</button>
    </div>
    <div class="gvd-sub">e2e stub - veto none</div>
  </div>

  <div class="gvd-sec">
    <div class="gvd-sec-h">
      SENSING <i class="gvd-note">GVD VISION screen</i>
      <button type="button" class="gvd-mini">1</button>
      <button type="button" class="gvd-mini is-on">2</button>
    </div>
    <div class="gvd-kv"><span>cameras</span><b>window - 1/8 feeds - retail: cam_main only</b></div>
    <div class="gvd-kv"><span>vision</span><b>on - screen 2</b></div>
  </div>

  <button type="button" class="gvd-nerdbtn">+ nerd</button>

  <div class="gvd-foot">Sim toy - camera-only - never a real car. synthetic raster.</div>
</div>
</body>
</html>
"""


def _app_css_override(css: str) -> str:
    # In-game tile is 330x440 with scroll. README shot is the same width, full
    # HUD unclipped so Policy / Sensing are visible without faking Esc menus.
    extra = """
.gvd-app {
  height: auto !important;
  width: 330px !important;
  min-height: 440px;
  overflow: visible !important;
  margin: 16px;
}
"""
    return css + extra


def _chrome_bin() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _crop_tile(img):
    """Crop to the HUD tile (drop the void page margin)."""
    import numpy as np

    gray = img.mean(axis=2)
    mask = gray > 16
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if not rows.any() or not cols.any():
        return img
    y0, y1 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
    x0, x1 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())
    pad = 4
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0], y1 + pad)
    x1 = min(img.shape[1], x1 + pad)
    tile = img[y0:y1, x0:x1]
    if tile.size == 0:
        return img
    return np.ascontiguousarray(tile)


def render_hud_chrome(dest: Path, html_path: Path, shot: Path) -> bool:
    chrome = _chrome_bin()
    if not chrome:
        return False
    cmd = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        "--force-color-profile=srgb",
        "--force-device-scale-factor=2",
        "--window-size=380,780",
        "--virtual-time-budget=1500",
        f"--screenshot={shot}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, check=False, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError):
        pass
    if not shot.is_file() or shot.stat().st_size < 1000:
        return False
    import cv2

    img = cv2.imread(str(shot))
    if img is None:
        return False
    img = _crop_tile(img)
    if img.shape[0] < 300 or img.shape[1] < 260:
        return False
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return dest.is_file()


def render_hud_opencv(dest: Path) -> Path:
    """Fallback HUD card using GVD void/ice tokens (not live CEF)."""
    import cv2
    import numpy as np

    from python.viz.stage import CORRIDOR, ICE, ICE_HI, VOID

    w, h = 330, 560
    img = np.full((h, w, 3), VOID, dtype=np.uint8)
    panel = (23, 19, 15)
    line = (46, 39, 30)
    dim = (140, 130, 121)
    faint = (106, 97, 89)
    paper = (209, 198, 184)

    def rect(x, y, ww, hh, color, thickness=-1):
        cv2.rectangle(img, (x, y), (x + ww, y + hh), color, thickness, cv2.LINE_AA)

    def text(s, x, y, scale=0.38, color=paper, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    rect(0, 0, w, h, (16, 13, 10))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), line, 1, cv2.LINE_AA)
    y = 22
    text("GVD", 10, y, 0.5, ICE, 1)
    text("VISION", 58, y, 0.38, faint)
    text("link live", w - 92, y, 0.32, ICE)
    cv2.circle(img, (w - 102, y - 4), 3, ICE, -1, cv2.LINE_AA)
    y = 40
    cv2.line(img, (10, y), (w - 10, y), line, 1, cv2.LINE_AA)
    y = 52
    rect(8, y, w - 16, 28, panel)
    cv2.rectangle(img, (8, y), (10, y + 28), faint, -1)
    text("DISENGAGED", 16, y + 19, 0.45, dim)
    text("standby", w - 78, y + 18, 0.32, faint)
    y = 90
    rect(8, y, w - 16, 32, (34, 28, 20))
    cv2.rectangle(img, (8, y), (w - 8, y + 32), (94, 82, 53), 1, cv2.LINE_AA)
    text("ENGAGE", 118, y + 21, 0.45, ICE_HI)
    text("ALT+G", w - 62, y + 20, 0.32, faint)
    y = 136
    text("WHEEL / PEDALS", 10, y, 0.32, faint)
    y = 150

    def axis(label, yy, centered=False):
        text(label, 10, yy + 10, 0.32, faint)
        rect(58, yy, 200, 10, (29, 24, 18))
        cv2.rectangle(img, (58, yy), (258, yy + 10), line, 1, cv2.LINE_AA)
        if centered:
            cv2.line(img, (158, yy), (158, yy + 10), (59, 51, 42), 1)
            cv2.rectangle(img, (157, yy - 1), (159, yy + 11), ICE_HI, -1)
        text("0.00", 268, yy + 10, 0.32, paper)

    axis("wheel", y, centered=True)
    axis("gas", y + 22)
    axis("brake", y + 44)
    text("applied wheel 0.00  gas 0.00  brake 0.00", 10, y + 70, 0.32, dim)
    y = 250
    cv2.line(img, (10, y), (w - 10, y), line, 1, cv2.LINE_AA)
    text("VIEW", 10, y + 16, 0.32, faint)
    rect(10, y + 24, 150, 22, (60, 50, 22))
    cv2.rectangle(img, (10, y + 24), (160, y + 46), (134, 118, 75), 1, cv2.LINE_AA)
    text("PATH", 64, y + 40, 0.35, ICE_HI)
    rect(170, y + 24, 150, 22, panel)
    cv2.rectangle(img, (170, y + 24), (320, y + 46), line, 1, cv2.LINE_AA)
    text("GHOSTS", 214, y + 40, 0.35, dim)
    text("conf 0.90 - 2.0 m - preview path - 3 tracked", 10, y + 64, 0.32, dim)
    y = 340
    cv2.line(img, (10, y), (w - 10, y), line, 1, cv2.LINE_AA)
    text("POLICY", 10, y + 16, 0.32, faint)
    for i, (name, on) in enumerate((("modular", True), ("e2e", False), ("shadow", False))):
        x = 10 + i * 104
        rect(x, y + 24, 98, 22, (60, 50, 22) if on else panel)
        cv2.rectangle(img, (x, y + 24), (x + 98, y + 46), (134, 118, 75) if on else line, 1, cv2.LINE_AA)
        text(name.upper(), x + (18 if name != "modular" else 10), y + 40, 0.32, ICE_HI if on else dim)
    text("e2e stub - veto none", 10, y + 66, 0.32, dim)
    y = 430
    cv2.line(img, (10, y), (w - 10, y), line, 1, cv2.LINE_AA)
    text("SENSING", 10, y + 16, 0.32, faint)
    text("GVD VISION screen", 90, y + 16, 0.28, faint)
    rect(268, y + 4, 20, 18, panel)
    text("1", 274, y + 17, 0.32, dim)
    rect(292, y + 4, 20, 18, (60, 50, 22))
    text("2", 298, y + 17, 0.32, ICE_HI)
    text("cameras", 10, y + 40, 0.32, faint)
    text("window - 1/8 - retail: cam_main only", 78, y + 40, 0.28, paper)
    text("vision", 10, y + 58, 0.32, faint)
    text("on - screen 2", 78, y + 58, 0.32, paper)
    y = 510
    text("+ nerd", 10, y, 0.32, faint)
    text("Sim toy - camera-only - never a real car.", 10, y + 28, 0.28, faint)
    text("synthetic raster", 10, h - 14, 0.32, CORRIDOR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return dest


def render_hud(dest: Path = HUD) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        html_path = tmp / "gvd_hud_parked.html"
        html_path.write_text(_hud_html(), encoding="utf-8")
        shot = tmp / "hud.png"
        if render_hud_chrome(dest, html_path, shot):
            return dest
    return render_hud_opencv(dest)


def main() -> int:
    MEDIA.mkdir(parents=True, exist_ok=True)
    cabin = render_cabin()
    hud = render_hud()
    print(f"[GVD] README media cabin -> {cabin}")
    print(f"[GVD] README media HUD   -> {hud}")
    print("[GVD] captions: synthetic. Soft Esc parked. Not Engage. Not live IQU45 photos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
