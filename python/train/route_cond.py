"""Navigation-command encoding for ``--route-cond``. Not ``--ctl-cond``.

The 5-vector is one-hot ``follow/left/right/straight`` plus a scaled
``next_jct_m``. No ego speed or control GT is included.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROUTE_CMDS = ("follow", "left", "right", "straight")
ROUTE_COND_DIM = 5
DIST_SCALE = 150.0


def encode_route(cmd: str, next_jct_m: float | None) -> np.ndarray:
    """``(5,)`` float32: one-hot command plus clipped ``next_jct_m / 150``.

    Unknown or masked commands become ``follow``. A missing distance is ``1.0``
    (treated as ``DIST_SCALE`` metres away).
    """
    label = str(cmd or "").strip().lower()
    if label not in ROUTE_CMDS:
        label = "follow"
    vec = np.zeros(ROUTE_COND_DIM, dtype=np.float32)
    vec[ROUTE_CMDS.index(label)] = 1.0
    if next_jct_m is None:
        dist = 1.0
    else:
        try:
            raw = float(next_jct_m)
            dist = 1.0 if not math.isfinite(raw) else min(max(raw / DIST_SCALE, 0.0), 2.0)
        except (TypeError, ValueError):
            dist = 1.0
    vec[4] = dist
    return vec


def cmd_from_nav(nav: Mapping[str, Any] | None) -> tuple[str, float | None]:
    """Map a ``route_intent`` / ``nav`` dict onto a training command."""
    if not isinstance(nav, Mapping):
        return "follow", None
    turn = str(nav.get("turn") or "").strip().lower()
    if turn not in ("left", "right", "straight"):
        turn = "follow"
    dist = nav.get("next_jct_m")
    try:
        metres = None if dist is None else float(dist)
        if metres is not None and not math.isfinite(metres):
            metres = None
    except (TypeError, ValueError):
        metres = None
    return turn, metres


def vec_from_nav(nav: Mapping[str, Any] | None) -> np.ndarray:
    cmd, dist = cmd_from_nav(nav)
    return encode_route(cmd, dist)


def runtime_route_vec(nav: Mapping[str, Any] | None, intent_result: Mapping[str, Any] | None) -> np.ndarray:
    """Encode the #203 schema after ``merge_nav_intent``. Missing intent is follow."""
    from python.sensors.route_intent import merge_nav_intent

    merged = merge_nav_intent(dict(nav) if isinstance(nav, Mapping) else {}, intent_result)
    return vec_from_nav(merged)


def _tkey(stamp: Any) -> str:
    try:
        return f"{float(stamp):.4f}"
    except (TypeError, ValueError):
        return str(stamp)


def _index_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and "t" in row:
                out[_tkey(row.get("t"))] = row
    return out


def attach_route_cmds(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Copy ``intent.jsonl`` onto rows as ``route_cmd`` / ``next_jct_m``."""
    caches: dict[str, dict[str, dict[str, Any]]] = {}
    counts = {name: 0 for name in (*ROUTE_CMDS, "mask", "missing")}
    for row in rows:
        root = str(row.get("source_root") or "")
        if root not in caches:
            caches[root] = _index_sidecar(Path(root) / "intent.jsonl") if root else {}
        rec = caches[root].get(_tkey(row.get("t")))
        if rec is None:
            row["route_cmd"] = "follow"
            row["next_jct_m"] = None
            counts["missing"] += 1
            continue
        cmd = str(rec.get("cmd") or "follow").strip().lower()
        if cmd not in (*ROUTE_CMDS, "mask"):
            cmd = "follow"
        row["route_cmd"] = cmd
        row["next_jct_m"] = rec.get("next_jct_m")
        counts[cmd] += 1
    return counts


def route_counts_line(counts: Mapping[str, int]) -> str:
    parts = " ".join(f"{name}={int(counts.get(name, 0))}" for name in (*ROUTE_CMDS, "mask", "missing"))
    return f"[GVD] route-cond: {parts}"


def per_command_metrics(samples: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Holdout ``n`` and mean error grouped by command."""
    out = {cmd: {"n": 0, "sum_err": 0.0, "mean_err": None} for cmd in ROUTE_CMDS}
    for sample in samples:
        cmd = str(sample.get("cmd") or "follow").strip().lower()
        if cmd not in ROUTE_CMDS:
            cmd = "follow"
        try:
            err = float(sample.get("err") or 0.0)
        except (TypeError, ValueError):
            err = 0.0
        out[cmd]["n"] += 1
        out[cmd]["sum_err"] += err
    for cmd, val in out.items():
        if val["n"]:
            val["mean_err"] = val["sum_err"] / val["n"]
    return out
