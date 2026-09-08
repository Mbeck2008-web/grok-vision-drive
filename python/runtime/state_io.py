"""Read/write Documents/GVD/gvd_state.json for BeamNG path ribbon + HUD."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

def gvd_docs_dir() -> Path:
    home = Path(os.path.expanduser("~"))
    d = home / "Documents" / "GVD"
    d.mkdir(parents=True, exist_ok=True)
    return d

def state_path() -> Path:
    return gvd_docs_dir() / "gvd_state.json"

def steer_preview_path_ego(steer_deg: float = 0.0, length_m: float = 15.0, step: float = 1.0) -> list[dict[str, float]]:
    """Short debug ribbon in vehicle frame (+X right, +Y forward, +Z up)."""
    curvature = (steer_deg / 30.0) * 0.05
    pts: list[dict[str, float]] = []
    x = 0.0
    y = 0.0
    heading = 0.0
    n = int(length_m / step)
    for _ in range(n + 1):
        pts.append({"x": x, "y": y, "z": 0.0})
        y += step
        heading += curvature
        x += math.sin(heading) * step * 0.15
    return pts

def default_state(**overrides: Any) -> dict[str, Any]:
    steer = float(overrides.get("steer_deg", 0.0))
    st: dict[str, Any] = {
        "schema": 1,
        "policy": "modular",
        "engaged": False,
        "disengage_reason": "none",
        "loop_hz": 0.0,
        "camera_hz": 0.0,
        "infer_ms": 0.0,
        "viz_ms": 0.0,
        "heartbeat_ms": 0.0,
        "heartbeat_unix": int(time.time()),  # match Lua os.time() seconds
            "heartbeat_mtime": time.time(),
        "gpu_vram_used_gb": 0.0,
        "gpu_vram_total_gb": 11.0,
        "gpu_name": "GTX 1080 Ti",
        "cam_health": {
            "main": "missing",
            "wide": "missing",
            "pillarL": "missing",
            "pillarR": "missing",
            "repeatL": "missing",
            "repeatR": "missing",
            "rear": "missing",
        },
        "lane_conf": 0.0,
        "bev_coverage": 0.0,
        "objects_n": 0,
        "tracks_n": 0,
        "ego": {
            "speed_mps": 0.0,
            "steer_deg": steer,
            "throttle": 0.0,
            "brake": 0.0,
            "yaw_rate": 0.0,
            "accel": 0.0,
        },
        "planner": {
            "corridor_width": 2.0,
            "curvature": 0.0,
            "target_v": 0.0,
            "ttc_lead": None,
            "aeb": "off",
        },
        "path_ego": steer_preview_path_ego(steer),
        "path_width": 2.0,
        "path_conf": 0.4,
        "path_debug_preview": True,
        "gvd_show_path": True,
        "show_agent_ghosts": False,
        "agents": [],
        "tracks": [],
        "lanes_bev": [],
        "occupancy": None,
        "last_clip_trigger": "none",
        "missing_state_keys": [
            "tracks",
            "lanes_bev",
            "occupancy",
            "real path_ego from planner",
        ],
    }
    st.update(overrides)
    return st

def write_state(state: dict[str, Any], path: Path | None = None) -> Path:
    p = path or state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["heartbeat_unix"] = int(time.time())  # match Lua os.time() seconds
        state["heartbeat_mtime"] = time.time()  # high-res; Lua may use file mtime later
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p

def read_state(path: Path | None = None) -> dict[str, Any] | None:
    p = path or state_path()
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
