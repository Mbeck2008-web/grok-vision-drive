"""SceneNet + sparse occupancy outputs -> a C++ viz scene (perception only).

Torch-free and ground-truth-free: inputs are network outputs as numpy arrays
(``ScenePrediction.as_dict`` fields plus occ head logits on queried voxels).
Never import labelers, ``occ_sparse`` labels, ``raw/*_depth`` or ``*_sem``.

Two steps (docs/specs/occ_sparse.md, "Viz schema"):

* :func:`scene_state` -> supervisor-state keys schema v1 of
  ``python/viz/cpp_publisher.py`` already reads (``lanes_ext``, ``road_edges``,
  ``tracks``, ``signs``) plus ``occ_cells`` / ``stop_lines`` / ``crosswalks``.
* :func:`extend_message` adds the v1 additions (``occ``, ``stop_lines``,
  ``crosswalks``, track ``motion``) to a built v1 message.

Colours stay the renderer's job. Michael's rule: white paint white, yellow
paint yellow, red only for ``edges`` (road edges and curbs).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

OCC_CLASSES = ("unknown", "drivable", "vehicle")  # mirrors python.train.occ_sparse
VOXEL_M = 0.5
GRID_MIN = (-16.0, -16.0, -1.0)
GRID_SHAPE = (64, 112, 8)
OBJECT_CLASSES = ("vehicle", "car", "truck", "bus", "pedestrian", "bike")  # mirrors scene_net
SIGN_CLASSES = ("stop_sign", "yield", "traffic_light", "speed_limit")
MOTION_STATES = ("unknown", "same_way", "oncoming", "crossing", "alongside", "parked")
PAINTS = ("white", "yellow")
LINE_KINDS = ("detected", "predicted")  # v1 lane/edge ``kind``
EDGE_SRCS = ("occ", "curb")  # optional ``src``: occ-derived edge or curbs head
EXTRA_KEYS = ("occ", "stop_lines", "crosswalks")
# Required top-level keys of gvd_scene v1 (docs/specs/viz_cpp.md at f3555bf). ``lead_id`` is optional.
V1_REQUIRED = frozenset({"v", "type", "seq", "t", "ego", "aeb", "lanes", "edges", "tracks", "signs", "path", "path_conf"})
MAX_OCC = 800  # ~51 B per cell as JSON: keeps one message under the 60 kB datagram
PARKED_MPS = 0.5
EDGE_BIN_M = 1.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def voxel_centers(flat: np.ndarray) -> np.ndarray:
    ix, iy, iz = np.unravel_index(np.asarray(flat, np.int64), GRID_SHAPE)
    return np.stack([ix, iy, iz], 1).astype(np.float32) * VOXEL_M + np.asarray(GRID_MIN, np.float32) + VOXEL_M / 2


def occ_cells(voxel_ids: Any, logits: Any, *, thresh: float = 0.5) -> list[dict[str, Any]]:
    """Occupied queried voxels with a class. ``logits`` is ``(k, 1 + len(OCC_CLASSES))``."""
    ids = np.asarray(voxel_ids, np.int64).reshape(-1)
    lg = np.asarray(logits, np.float32).reshape(len(ids), -1)
    keep = _sigmoid(lg[:, 0]) > float(thresh)
    if not keep.any():
        return []
    c = voxel_centers(ids[keep])
    cls = lg[keep, 1:].argmax(1)
    out = [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "z": round(float(p[2]), 2),
            "cls": OCC_CLASSES[int(k)]} for p, k in zip(c, cls)]
    return out[:MAX_OCC]


def edges_from_occ(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Left/right road edge polylines: outermost drivable cell per 1 m forward bin.

    A side needs two bins to make a line. Kind ``occ_edge`` (drawn red).
    """
    drv = [(float(c["x"]), float(c["y"])) for c in cells if c.get("cls") == "drivable"]
    if not drv:
        return []
    bins: dict[int, list[float]] = {}
    for x, y in drv:
        bins.setdefault(int(math.floor(y / EDGE_BIN_M)), []).append(x)
    rows = sorted(bins.items())
    half = VOXEL_M / 2
    left = [[round(min(xs) - half, 2), round((b + 0.5) * EDGE_BIN_M, 2)] for b, xs in rows]
    right = [[round(max(xs) + half, 2), round((b + 0.5) * EDGE_BIN_M, 2)] for b, xs in rows]
    return [{"pts": pts, "kind": "detected", "src": "occ"} for pts in (left, right) if len(pts) >= 2]


def motion_bucket(vx: float, vy: float) -> str:
    """Legend bucket from a ground velocity (ego frame, m/s; +y = ego heading).

    Under 0.5 m/s parked; within 30 deg of +y same_way; beyond 150 deg oncoming;
    60-120 deg crossing; the rest by the sign of vy. ``alongside`` is decided by
    :class:`MotionTracker` from position.
    """
    speed = math.hypot(vx, vy)
    if speed < PARKED_MPS:
        return "parked"
    d = abs(math.degrees(math.atan2(vx, vy)))
    if d <= 30.0:
        return "same_way"
    if d >= 150.0:
        return "oncoming"
    if 60.0 <= d <= 120.0:
        return "crossing"
    return "same_way" if vy > 0 else "oncoming"


class MotionTracker:
    """Motion state from object positions over frames, never from one frame.

    Each tick, objects are matched to the previous tick's tracks by nearest
    position (<= ``MATCH_M``). Ground velocity = displacement / dt plus ego
    forward speed (ego frame; yaw rate ignored over the short window). A track
    reports ``unknown`` until it has ``MIN_FRAMES`` ticks spanning
    ``MIN_SPAN_S``. ``alongside``: same_way and beside ego (|y| < 6, |x| > 1.5).
    """

    MATCH_M = 3.0
    MIN_FRAMES = 3
    MIN_SPAN_S = 0.2
    KEEP = 8

    def __init__(self) -> None:
        self._tracks: dict[int, list[tuple[float, float, float, float]]] = {}
        self._next = 0

    def update(self, objects: Sequence[Mapping[str, Any]], t: float, ego_speed_mps: float = 0.0) -> list[dict[str, Any]]:
        prev = {tid: h[-1] for tid, h in self._tracks.items()}
        used: set[int] = set()
        out: list[dict[str, Any]] = []
        alive: dict[int, list[tuple[float, float, float, float]]] = {}
        for ob in objects:
            x, y = float(ob["x"]), float(ob["y"])
            best, best_d = None, self.MATCH_M
            for tid, (_t, px, py, _v) in prev.items():
                d = math.hypot(x - px, y - py)
                if tid not in used and d <= best_d:
                    best, best_d = tid, d
            if best is None:
                best = self._next
                self._next += 1
                hist: list[tuple[float, float, float, float]] = []
            else:
                hist = list(self._tracks[best])
            used.add(best)
            hist = (hist + [(float(t), x, y, float(ego_speed_mps))])[-self.KEEP:]
            alive[best] = hist
            out.append({**dict(ob), "id": best, "motion": self._motion(hist)})
        self._tracks = alive
        return out

    def _motion(self, hist: Sequence[tuple[float, float, float, float]]) -> str:
        if len(hist) < self.MIN_FRAMES or hist[-1][0] - hist[0][0] < self.MIN_SPAN_S:
            return "unknown"
        (t0, x0, y0, _), (t1, x1, y1, _) = hist[0], hist[-1]
        dt = t1 - t0
        ego_v = float(np.mean([h[3] for h in hist]))
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt + ego_v
        state = motion_bucket(vx, vy)
        if state == "same_way" and abs(y1) < 6.0 and abs(x1) > 1.5:
            return "alongside"
        return state


def _paint(probs: Any, i: int) -> str:
    """``lane_paint`` = per-lane P(yellow). Missing head -> white (never guessed yellow)."""
    if probs is None:
        return "white"
    p = np.asarray(probs, np.float32).reshape(-1)
    return "yellow" if i < len(p) and float(p[i]) > 0.5 else "white"


def scene_state(
    pred: Mapping[str, Any],
    *,
    occ: tuple[Any, Any] | None = None,
    tracker: MotionTracker | None = None,
    t: float = 0.0,
    ego_speed_mps: float = 0.0,
    thresh: float = 0.5,
) -> dict[str, Any]:
    """Supervisor-state keys for the viz from one tick of network outputs.

    ``pred`` keys (all optional, batch dim dropped): ``lanes (L,P,2)``,
    ``lane_valid (L)`` logits, ``curbs (C,P,2)``, ``curb_valid``, ``objects (N,8)``,
    ``object_class (N,K)``, ``object_valid``, ``signs (S,4)``, ``sign_class``,
    ``sign_valid``, ``lane_paint (L)`` P(yellow), ``stop_line (2)`` (y_m, valid
    logit), ``crosswalks (M,4)`` (x0, y0, x1, y1). ``occ`` = (voxel ids, logits).
    """
    def arr(key: str) -> np.ndarray | None:
        v = pred.get(key)
        return None if v is None else np.asarray(v, np.float32)

    def valid(key: str, n: int) -> np.ndarray:
        v = arr(key)
        return np.ones(n, bool) if v is None else _sigmoid(v.reshape(-1)[:n]) > thresh

    st: dict[str, Any] = {"lanes_ext": [], "road_edges": [], "tracks": [], "signs": [],
                          "occ_cells": [], "stop_lines": [], "crosswalks": []}
    lanes = arr("lanes")
    if lanes is not None:
        ok = valid("lane_valid", len(lanes))
        for i, ln in enumerate(lanes):
            if ok[i]:
                st["lanes_ext"].append({"points": ln.tolist(), "paint": _paint(pred.get("lane_paint"), i), "kind": "detected"})
    curbs = arr("curbs")
    if curbs is not None:
        ok = valid("curb_valid", len(curbs))
        st["road_edges"] += [{"points": c.tolist(), "kind": "detected", "src": "curb"} for i, c in enumerate(curbs) if ok[i]]
    if occ is not None:
        st["occ_cells"] = occ_cells(occ[0], occ[1], thresh=thresh)
        st["road_edges"] += [{"points": e["pts"], "kind": e["kind"], "src": e["src"]} for e in edges_from_occ(st["occ_cells"])]
    objs = arr("objects")
    if objs is not None:
        ok = valid("object_valid", len(objs))
        cls = arr("object_class")
        for i, o in enumerate(objs):
            if not ok[i]:
                continue
            name = OBJECT_CLASSES[int(cls[i].argmax())] if cls is not None else "vehicle"
            st["tracks"].append({"id": i, "class": name, "x": float(o[0]), "y": float(o[1]), "yaw": float(o[2]),
                                 "speed_mps": float(o[5])})
        # Motion needs history: without a tracker every track is ``unknown``.
        st["tracks"] = (tracker.update(st["tracks"], t, ego_speed_mps) if tracker is not None
                        else [{**tr, "motion": "unknown"} for tr in st["tracks"]])
    signs = arr("signs")
    if signs is not None:
        ok = valid("sign_valid", len(signs))
        scls = arr("sign_class")
        for i, s in enumerate(signs):
            if ok[i]:
                name = SIGN_CLASSES[int(scls[i].argmax())] if scls is not None else "stop_sign"
                st["signs"].append({"class": name, "x": float(s[0]), "y": float(s[1])})
    stop = arr("stop_line")
    if stop is not None and stop.size >= 2 and _sigmoid(stop.reshape(-1)[1]) > thresh:
        y = round(float(stop.reshape(-1)[0]), 2)
        st["stop_lines"].append({"pts": [[-1.8, y], [1.8, y]]})
    cw = arr("crosswalks")
    if cw is not None:
        st["crosswalks"] = [{"pts": [[round(float(c[0]), 2), round(float(c[1]), 2)], [round(float(c[2]), 2), round(float(c[3]), 2)]]}
                            for c in cw.reshape(-1, 4)]
    return st


def extend_message(msg: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Add the v1 additions to a built v1 message (in place, returned)."""
    msg["occ"] = [dict(c) for c in state.get("occ_cells") or []][:MAX_OCC]
    msg["stop_lines"] = [dict(s) for s in state.get("stop_lines") or []]
    msg["crosswalks"] = [dict(c) for c in state.get("crosswalks") or []]
    by_id = {t.get("id"): t.get("motion") for t in state.get("tracks") or [] if t.get("motion")}
    for tr in msg.get("tracks") or []:
        if by_id.get(tr.get("id")):
            tr["motion"] = by_id[tr.get("id")]
    return msg


def build_message(state: Mapping[str, Any], *, seq: int = 0, t: float = 0.0) -> dict[str, Any]:
    """Full message: the v1 publisher's builder when present, else the same v1 fields here."""
    try:
        from python.viz.cpp_publisher import build_scene_message
    except ImportError:
        build_scene_message = None
    if build_scene_message is not None:
        return extend_message(build_scene_message(dict(state), seq=seq, t=t), state)
    msg: dict[str, Any] = {
        "v": 1, "type": "gvd_scene", "seq": int(seq), "t": float(t),
        "ego": {"engaged": False, "speed_mps": 0.0}, "aeb": "off", "signs": [],
        "lanes": [{"pts": ln["points"], "paint": ln["paint"], "kind": ln["kind"]} for ln in state.get("lanes_ext") or []],
        "edges": [{"pts": e["points"], "kind": e["kind"], "src": e["src"]} for e in state.get("road_edges") or []],
        "tracks": [{k2: tr[k1] for k1, k2 in (("id", "id"), ("class", "cls"), ("x", "x"), ("y", "y"), ("yaw", "yaw"), ("speed_mps", "v"))}
                   for tr in state.get("tracks") or []],
        "path": {"pts": []}, "path_conf": 1.0,
    }
    msg["signs"] = [{"cls": s["class"], "x": s["x"], "y": s["y"], "state": "unknown"} for s in state.get("signs") or []]
    return extend_message(msg, state)


def validate_message(msg: Mapping[str, Any]) -> list[str]:
    """Problems with a viz message (empty list = valid). Pins the documented schema."""
    bad: list[str] = []
    if msg.get("v") != 1 or msg.get("type") != "gvd_scene":
        bad.append("header")
    missing = V1_REQUIRED - set(msg)
    if missing:
        bad.append(f"missing v1 keys {sorted(missing)}")
    for key in ("lanes", "edges", "tracks", "signs") + EXTRA_KEYS:
        if not isinstance(msg.get(key), list):
            bad.append(f"{key} not a list")
    for ln in msg.get("lanes") or []:
        if ln.get("paint") not in PAINTS or ln.get("kind") not in LINE_KINDS or len(ln.get("pts") or []) < 2:
            bad.append(f"lane {ln.get('paint')}")
    for ed in msg.get("edges") or []:
        if ed.get("kind") not in LINE_KINDS or ed.get("src", "curb") not in EDGE_SRCS or len(ed.get("pts") or []) < 2:
            bad.append(f"edge {ed.get('kind')}")
        if "paint" in ed:
            bad.append("edges carry no paint (always red)")
    for c in msg.get("occ") or []:
        if c.get("cls") not in OCC_CLASSES or not all(isinstance(c.get(a), float) for a in "xyz"):
            bad.append("occ cell")
    for tr in msg.get("tracks") or []:
        if "motion" in tr and tr["motion"] not in MOTION_STATES:
            bad.append(f"motion {tr['motion']}")
    for group in ("stop_lines", "crosswalks"):
        for s in msg.get(group) or []:
            if len(s.get("pts") or []) != 2:
                bad.append(group)
    return bad


def occ_to_viz(outputs: Mapping[str, Any], *, tracker: MotionTracker | None = None, seq: int = 0, t: float = 0.0,
               ego_speed_mps: float = 0.0) -> dict[str, Any]:
    """One tick of network outputs -> gvd_scene v1 message plus the optional fields.

    ``outputs`` is :func:`scene_state`'s ``pred``; ``outputs["occ"]`` (optional) is
    ``(voxel ids, logits)``. Pass the same ``tracker`` every tick for motion state.
    """
    pred = {k: v for k, v in outputs.items() if k != "occ"}
    st = scene_state(pred, occ=outputs.get("occ"), tracker=tracker, t=t, ego_speed_mps=ego_speed_mps)
    msg = build_message(st, seq=seq, t=t)
    msg["ego"]["speed_mps"] = round(float(ego_speed_mps), 2)
    return msg
