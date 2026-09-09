#!/usr/bin/env python3
"""Offline M6 retail-package checks (no BeamNG, no weights, no network)."""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_release_zip as rel  # noqa: E402
from python.control.actuate import (  # noqa: E402
    DriveCommand,
    cmd_path,
    engage_path,
    make_actuator,
    read_engage_flag,
    write_engage_flag,
)
from python.runtime.hw_probe import HwReport, cam_claim  # noqa: E402
from python.runtime.state_io import state_path  # noqa: E402
from python.sensors.cameras import CAM_IDS, WindowBackend  # noqa: E402

CHROME_RE = re.compile(r"tesla|\bfsd\b|full self[- ]driving|autopilot", re.I)


def _copy_repo(dst: Path) -> None:
    ignore = shutil.ignore_patterns(".git", "dist", "__pycache__", ".venv", "venv", "data")
    for item in ROOT.iterdir():
        if item.name in (".git", "dist", ".venv", "venv"):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, ignore=ignore)
        else:
            shutil.copy2(item, dst / item.name)


def check_release_zip() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "repo"
        tmp.mkdir()
        _copy_repo(tmp)
        # Plant everything that must never ship.
        (tmp / "models").mkdir(exist_ok=True)
        (tmp / "models" / "e2e_current.onnx").write_bytes(b"\x00" * 64)
        (tmp / "models" / "yolov8n.pt").write_bytes(b"\x00" * 64)
        (tmp / "python" / "weights.safetensors").write_bytes(b"\x00" * 16)
        (tmp / "data" / "clips").mkdir(parents=True, exist_ok=True)
        (tmp / "data" / "clips" / "clip_x.mp4").write_bytes(b"\x00" * 16)
        (tmp / "python" / "__pycache__").mkdir()
        (tmp / "python" / "__pycache__" / "run_vision.cpython-312.pyc").write_bytes(b"\x00")
        (tmp / ".git").mkdir()
        (tmp / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (tmp / "dist").mkdir()
        (tmp / "dist" / "old.zip").write_bytes(b"PK")

        files = rel.collect_files(tmp)
        out = Path(td) / "gvd-retail-test.zip"
        zip_path, names = rel.build(out, version="test", flat=False, root=tmp, files=files)
        problems = rel.verify(zip_path, flat=False, version="test")
        assert not problems, problems

        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            assert members and all(m.startswith("gvd-retail-test/") for m in members), members[:5]
            rels = [m.split("/", 1)[1] for m in members]
            for must in rel.MUST_HAVE:
                assert must in rels, f"missing {must}"
            for bad in (".onnx", ".pt", ".safetensors", ".pyc", ".mp4"):
                assert not any(r.endswith(bad) for r in rels), bad
            assert not any(r.startswith((".git/", "data/", "dist/", "scripts/")) for r in rels)
            assert not any("clips" in r for r in rels)
            assert "requirements-beamng.txt" not in rels
            assert "requirements-perception.txt" not in rels
            assert "requirements-viz.txt" not in rels
            # Mod entry point + UI icon survive; tests do not.
            assert "beamng_mod/scripts/gvd/modScript.lua" in rels
            assert "beamng_mod/ui/modules/apps/GVD/app.png" in rels
            assert "models/.gitkeep" in rels
            assert not any(r.startswith("scripts/test_") for r in rels)
            # Windows-friendly: .bat files are CRLF inside the zip.
            for r in ("install.bat", "uninstall.bat", "play_gvd.bat"):
                data = zf.read(f"gvd-retail-test/{r}")
                assert b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n"), r
            ver = zf.read("gvd-retail-test/VERSION.txt").decode("utf-8")
            assert "retail package test" in ver
            assert "1-cam window capture only" in ver
            assert "cannot drive the car" in ver

        # Flat variant has no top-level folder.
        flat = Path(td) / "flat.zip"
        rel.build(flat, version="test", flat=True, root=tmp, files=files)
        assert not rel.verify(flat, flat=True, version="test")
        with zipfile.ZipFile(flat) as zf:
            assert "install.bat" in zf.namelist()

    # The real checkout must package cleanly too (manifest only; no dist/ write).
    real = rel.collect_files(ROOT)
    assert "install.bat" in real and "python/run_vision.py" in real
    assert not any(f.endswith((".onnx", ".pt")) for f in real)


def check_boot_line_honesty() -> None:
    def rep(backend: str) -> HwReport:
        return HwReport(cpu="i9-9900K", ram_gb=32, dgpu="GTX 1080 Ti", dgpu_vram_gb=11.0, igpu="UHD630", qsv=True, backend=backend)

    retail = rep("window")
    line = retail.boot_line()
    assert retail.is_retail
    assert "backend=window" in line and "cams=1/8" in line and "retail" in line and "not 8" in line, line
    assert not CHROME_RE.search(line), line

    tech = rep("beamngpy")
    assert not tech.is_retail
    assert "cams<=8" in tech.boot_line() and "tech" in tech.boot_line()
    assert "cams=0/8" in rep("stub").boot_line()
    assert "8" not in cam_claim("window").split("(")[0].replace("1/8", "")  # never "8 cams" on retail


def check_window_backend_one_cam() -> None:
    class _FakeCam:
        def grab(self):
            return np.full((360, 640, 3), 90, dtype=np.uint8)

    wb = WindowBackend(long_side=320)
    wb._cam = _FakeCam()
    wb._region = {"left": 0, "top": 0, "width": 640, "height": 360}
    bundle = wb.grab()
    assert bundle.backend == "window"
    assert bundle.note.startswith("retail: 1 window"), bundle.note
    health = bundle.health_str()
    assert set(health) == set(CAM_IDS)
    ok = [k for k, v in health.items() if v == "ok"]
    assert ok == ["main"], health
    assert set(bundle.frames) <= {"main", "cam_main"}, list(bundle.frames)
    assert bundle.main_bgr() is not None and max(bundle.main_bgr().shape[:2]) <= 320

    # No capture library at all → honest missing, never synthesized frames.
    wb2 = WindowBackend(long_side=320)
    wb2._region = {"left": 0, "top": 0, "width": 10, "height": 10}
    empty = wb2.grab()
    assert empty.main_bgr() is None
    assert all(v == "missing" for v in empty.health_str().values())


def check_retail_actuator_is_sink() -> None:
    act = make_actuator(None, prefer_beamngpy=True)
    assert act.name == "cmd_json"
    out = act.apply(DriveCommand(steer=0.1, throttle=0.3, brake=0.0, seq=1, reason="ok"))
    assert out.applied is False and out.reason == "cmd_json_sink", out
    payload = json.loads(cmd_path().read_text(encoding="utf-8"))
    assert payload["applied"] is False and payload["reason"] == "cmd_json_sink"


def check_engage_path_contract() -> None:
    p = engage_path()
    assert p.parts[-3:] == ("Documents", "GVD", "gvd_engage.json"), p
    assert p.parent == state_path().parent == cmd_path().parent
    # Lua-shaped payload (Alt+A) must read back as engaged.
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"engaged":true,"mtime":1700000000}', encoding="utf-8")
    assert read_engage_flag(default=False) is True
    # Supervisor disengage payload: engaged=false with float mtime (what Lua adopts as OFF).
    write_engage_flag(False)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["engaged"] is False and isinstance(data["mtime"], float)
    assert read_engage_flag(default=True) is False


def _py_string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append(node.value)
    return out


def check_no_chrome() -> None:
    # Player-facing mod files: strict.
    for f in (ROOT / "beamng_mod").rglob("*"):
        if f.suffix.lower() in (".lua", ".js", ".html", ".json"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            assert not CHROME_RE.search(text), f"chrome in {f.relative_to(ROOT)}"
    # Launchers and the release tooling banner.
    for f in ("install.bat", "uninstall.bat", "play_gvd.bat", "requirements-retail.txt"):
        assert not CHROME_RE.search((ROOT / f).read_text(encoding="utf-8", errors="ignore")), f
    # Runtime strings players see (titles, prints); docstrings/comments may still state the honesty rule.
    for f in [ROOT / "python" / "run_vision.py", ROOT / "python" / "runtime" / "hw_probe.py", *(ROOT / "python" / "viz").glob("*.py")]:
        for s in _py_string_literals(f):
            assert not CHROME_RE.search(s), f"chrome string in {f.relative_to(ROOT)}: {s!r}"
    # Release VERSION.txt banner only names the honesty rule, never a product label.
    ver = rel.version_text("test")
    assert "Not Tesla FSD" in ver and "Full Self" not in ver


def check_player_docs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "**M6" in readme, "README Status must be M6"
    assert "## Player guide" in readme
    assert "make_release_zip" in readme
    assert "requirements-retail.txt" in readme
    # Soft: retail zip ships base+retail only
    assert "requirements-beamng.txt" not in (ROOT / "scripts" / "make_release_zip.py").read_text()
    assert '"requirements*.txt"' not in (ROOT / "scripts" / "make_release_zip.py").read_text()
    bat = (ROOT / "play_gvd.bat").read_text(encoding="utf-8", errors="ignore")
    assert 'set "GVD_BACKEND=window"' in bat and "--backend %GVD_BACKEND%" in bat
    assert "--backend auto" not in bat
    reqs = (ROOT / "requirements-retail.txt").read_text(encoding="utf-8")
    assert "mss" in reqs and "beamngpy" not in reqs.replace("# No beamngpy here", "")


def main() -> None:
    check_release_zip()
    check_boot_line_honesty()
    check_window_backend_one_cam()
    check_retail_actuator_is_sink()
    check_engage_path_contract()
    check_no_chrome()
    check_player_docs()
    print("test_m6_retail: OK")


if __name__ == "__main__":
    main()
