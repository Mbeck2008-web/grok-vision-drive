#!/usr/bin/env bash
# GVD Cloud Agent install: idempotent setup for the Linux-runnable Python slice
# (perception, planning, viz, and the offline smoke/test suite). BeamNG.drive
# itself is Windows-only, so this prepares everything that runs headless here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# System libraries: OpenCV runtime (libGL/glib/X), ffmpeg for M4 clip encode,
# and a Lua 5.1 / LuaJIT interpreter so the mod's Lua drive-bus harness runs.
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq --no-install-recommends \
    python3-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg lua5.1 luajit
fi

# Python virtualenv with the retail (1-cam window) + viz runtime deps.
# Windows-only wheels in requirements-retail.txt are guarded by platform markers
# and are simply skipped on Linux.
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-retail.txt -r requirements-viz.txt

echo "[gvd-install] done: $(python --version) | $(python -c 'import cv2; print("opencv", cv2.__version__)')"
