#!/usr/bin/env python3
"""Build the GVD retail release zip (M6).

Windows-friendly: double-click ``scripts\\make_release_zip.bat`` or run
``python scripts/make_release_zip.py``. Also works on Linux/macOS (stdlib only).

Ships the **retail** slice a player double-clicks:
  install.bat / uninstall.bat / play_gvd.bat, beamng_mod/, python/ (runtime),
  config/, requirements.txt + requirements-retail.txt, LICENSE, README.md, docs/*.md, VERSION.txt.

Never ships: data/clips/, weights (*.onnx *.pt *.pth *.bin *.safetensors …),
.git, __pycache__, venvs, scripts/ (tests + this tool), dist/.

Honesty: retail = 1-cam window capture only. 8-cam rig needs BeamNG.tech + BeamNGpy.
No Tesla / FSD chrome. Not real-vehicle control.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RELEASE_LABEL = "m6"

# Top-level files (exact) and globs relative to repo root.
INCLUDE_FILES = (
    "install.bat",
    "uninstall.bat",
    "play_gvd.bat",
    "LICENSE",
    "README.md",
)
INCLUDE_GLOBS = (
    # Retail zip: base + retail only (no beamng/perception/viz extras).
    "requirements.txt",
    "requirements-retail.txt",
    "docs/*.md",
    "models/.gitkeep",
)
# Directories copied recursively (subject to EXCLUDE_* below).
INCLUDE_DIRS = ("beamng_mod", "python", "config")

# Must exist in the final zip or the build fails (retail entry points).
MUST_HAVE = (
    "install.bat",
    "uninstall.bat",
    "play_gvd.bat",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "requirements-retail.txt",
    "beamng_mod/scripts/gvd/modScript.lua",
    "beamng_mod/lua/ge/extensions/gvd/main.lua",
    "beamng_mod/ui/modules/apps/GVD/app.json",
    "python/run_vision.py",
    "python/sensors/cameras.py",
    "config/cameras.yaml",
    "config/hardware.yaml",
    "VERSION.txt",
)

# Directory names skipped anywhere in the tree (tooling/cache junk only —
# NOT "scripts": beamng_mod/scripts/gvd/modScript.lua is the mod entry point).
EXCLUDE_DIR_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}
# Repo-root folders that never ship (tests/tooling, clips, build output).
EXCLUDE_REL_PREFIXES = ("scripts/", "data/", "dist/")
# Weight blobs + junk, matched against the file name and the repo-relative path.
EXCLUDE_FILE_GLOBS = (
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.engine",
    "*.trt",
    "*.ckpt",
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*.zip",
    "*.mp4",
    "*.mkv",
    ".DS_Store",
    "Thumbs.db",
    ".gitignore",
    ".gitattributes",
    "data/clips/*",
)
# Never allowed in the built archive (belt-and-suspenders verification).
FORBIDDEN_ARCHIVE_GLOBS = (
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.pyc",
    "*/.git/*",
    ".git/*",
    "*/data/clips/*",
    "data/clips/*",
    "*/scripts/test_*",
    "scripts/test_*",
)

HONESTY_LINES = (
    "Retail = 1-cam window capture only (main/cam_main). 8-cam rig needs BeamNG.tech + BeamNGpy.",
    "Retail drives the sim car through Documents/GVD/gvd_cmd.json -> mod Lua input.event on the player vehicle; "
    "Tech keeps BeamNGpy direct control (preferred).",
    "Vision only at inference. Optional LiDAR/radar/IMU/Foxglove are a bus the planner ignores. GPS is a nav hint, not localization. Not Tesla FSD, no Tesla logos. Sim toy, never a real car.",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        return out.strip() or None
    except Exception:
        return None


def default_version() -> str:
    tag = _git("describe", "--tags", "--exact-match")
    if tag:
        return tag
    sha = _git("rev-parse", "--short", "HEAD") or "dev"
    dirty = _git("status", "--porcelain")
    return f"{RELEASE_LABEL}-{sha}{'-dirty' if dirty else ''}"


def _excluded_dir(name: str) -> bool:
    return name in EXCLUDE_DIR_NAMES


def _excluded_file(rel: str) -> bool:
    if rel.startswith(EXCLUDE_REL_PREFIXES):
        return True
    name = rel.rsplit("/", 1)[-1]
    for pat in EXCLUDE_FILE_GLOBS:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
            return True
    return False


def collect_files(root: Path = ROOT) -> list[str]:
    """Return sorted repo-relative POSIX paths to package (excluding weights/clips/tests)."""
    picked: set[str] = set()

    for f in INCLUDE_FILES:
        if (root / f).is_file():
            picked.add(f)

    for pat in INCLUDE_GLOBS:
        for p in root.glob(pat):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if not _excluded_file(rel):
                    picked.add(rel)

    for d in INCLUDE_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(n for n in dirnames if not _excluded_dir(n))
            for fn in filenames:
                rel = (Path(dirpath) / fn).relative_to(root).as_posix()
                if _excluded_file(rel):
                    continue
                picked.add(rel)

    return sorted(picked)


def _to_crlf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def version_text(version: str) -> str:
    sha = _git("rev-parse", "HEAD") or "unknown"
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"Grok Vision Drive (GVD) retail package {version}",
        f"built {now}",
        f"git {sha}",
        "",
        *HONESTY_LINES,
        "",
        "Install: double-click install.bat -> restart BeamNG -> enable GVD in Mod Manager.",
        "Play:    double-click play_gvd.bat (needs Python 3 + pip install -r requirements-retail.txt).",
        "Remove:  double-click uninstall.bat.",
    ]
    return "\r\n".join(lines) + "\r\n"


def build(
    out: Path,
    *,
    version: str,
    flat: bool = False,
    root: Path = ROOT,
    files: list[str] | None = None,
) -> tuple[Path, list[str]]:
    files = files if files is not None else collect_files(root)
    prefix = "" if flat else f"gvd-retail-{version}/"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    names: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            src = root / rel
            data = src.read_bytes()
            if rel.lower().endswith(".bat"):
                data = _to_crlf(data)
            arc = prefix + rel
            zf.writestr(_zinfo(arc, src), data)
            names.append(rel)
        arc = prefix + "VERSION.txt"
        zf.writestr(arc, version_text(version))
        names.append("VERSION.txt")
    return out, names


def _zinfo(arcname: str, src: Path) -> zipfile.ZipInfo:
    st = src.stat()
    t = _dt.datetime.fromtimestamp(st.st_mtime)
    zi = zipfile.ZipInfo(arcname, date_time=(max(t.year, 1980), t.month, t.day, t.hour, t.minute, t.second))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


def verify(zip_path: Path, *, flat: bool, version: str) -> list[str]:
    """Return a list of problems (empty == OK)."""
    problems: list[str] = []
    prefix = "" if flat else f"gvd-retail-{version}/"
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            problems.append(f"corrupt member: {bad}")
        names = zf.namelist()
        stripped = [n[len(prefix):] if n.startswith(prefix) else n for n in names]
        for n in names:
            if prefix and not n.startswith(prefix):
                problems.append(f"member outside top-level folder: {n}")
            for pat in FORBIDDEN_ARCHIVE_GLOBS:
                if fnmatch.fnmatch(n, pat) or fnmatch.fnmatch(n.rsplit("/", 1)[-1], pat):
                    problems.append(f"forbidden member: {n}")
                    break
        for must in MUST_HAVE:
            if must not in stripped:
                problems.append(f"missing must-have: {must}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the GVD retail release zip (no weights, no clips, no .git).")
    ap.add_argument("--version", default=None, help="Version label (default: exact git tag, else m6-<sha>)")
    ap.add_argument("--out", default=None, help="Output path (default: dist/gvd-retail-<version>.zip)")
    ap.add_argument("--flat", action="store_true", help="No top-level gvd-retail-<version>/ folder inside the zip")
    ap.add_argument("--list", action="store_true", help="Print the manifest and exit without writing")
    args = ap.parse_args(argv)

    version = args.version or default_version()
    files = collect_files()

    if args.list:
        for f in files:
            print(f)
        print("VERSION.txt  (generated)")
        print(f"[GVD] {len(files) + 1} files would be packaged as gvd-retail-{version}.zip")
        return 0

    out = Path(args.out) if args.out else ROOT / "dist" / f"gvd-retail-{version}.zip"
    zip_path, names = build(out, version=version, flat=args.flat, files=files)
    problems = verify(zip_path, flat=args.flat, version=version)
    if problems:
        for p in problems:
            print(f"[GVD] RELEASE CHECK FAILED: {p}")
        try:
            zip_path.unlink()
        except Exception:
            pass
        return 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[GVD] release zip -> {zip_path}  ({len(names)} files, {size_mb:.2f} MB)")
    print("[GVD] excluded: data/clips/, weights (*.onnx *.pt *.pth *.bin *.safetensors), .git, scripts/, __pycache__")
    for line in HONESTY_LINES:
        print(f"[GVD] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
