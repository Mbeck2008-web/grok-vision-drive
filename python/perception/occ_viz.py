"""Sparse occupancy output -> viz scene state (docs/specs/occ_sparse.md, "viz outputs").

Runtime, perception only: the input is what ``occ_sparse.infer_sparse``
returned (head outputs only: cells, classes). No ground truth is imported or read here.
numpy only; no torch, no scipy.

Everything is derived from occupied cells by plain geometry, in the viz
ego frame (x right, y forward, metres; yaw radians from +x, pi/2 = same
heading as ego), the frame of ``docs/specs/viz_cpp.md``:

* ``occ_cells``  every occupied cell: ``[x, y, z, class_name]``.
* ``lanes_ext``  per lane-paint class, 8-connected bird's-eye components;
  each becomes one polyline ordered along its principal axis (cells binned
  at one voxel, perpendicular offset averaged). ``paint`` is the class:
  white paint -> "white", yellow paint -> "yellow". Nothing else picks a colour.
* ``road_edges`` the left and right boundary of drivable cells, one point per
  0.4 m forward row, split where a row has no drivable cell for more than
  ``EDGE_GAP_ROWS`` rows. Curb cells (when the net predicts them) are
  edge polylines too, ``kind: "curb"``. Edges are the only red lines.
* ``crosswalks`` oriented rectangle (4 corners, closed) per component.
* ``stop_lines`` polyline per component.
* ``tracks``     per frame: vehicle components -> centroid and oriented
  extent. :class:`OccTracker` links them over frames (ego-motion
  compensated) and only then gives ``speed_mps``, ``yaw`` from velocity and
  ``motion`` in {same_way, oncoming, crossing, alongside, parked}.

:func:`occ_to_viz` wraps all of it as a ``gvd_scene`` v1 message
(``docs/specs/viz_cpp.md`` at f3555bf) with documented optional extras.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

CLASS_NAMES = ("drivable", "vehicle", "lane_white", "lane_yellow", "crosswalk", "stop_line", "curb")
MOTION_STATES = ("same_way", "oncoming", "crossing", "alongside", "parked")
PARKED_MPS = 0.5
# |cos| of the heading difference above which a mover is along the ego axis.
ALONG_COS = 0.7
# alongside: same way, beside the ego rather than ahead or behind.
ALONGSIDE_X_M = 1.5
ALONGSIDE_Y_M = 6.0
MIN_CELLS = {"lane": 3, "edge": 2, "vehicle": 2, "crosswalk": 3, "stop_line": 2}
EDGE_GAP_ROWS = 2
MAX_OCC_CELLS = 4096


def _cid(name: str) -> int:
    return CLASS_NAMES.index(name)


def bev_components(ij: np.ndarray) -> np.ndarray:
    """8-connected components of integer bird's-eye cells (N,2). Returns a label per row."""
    n = len(ij)
    labels = np.full(n, -1, np.int64)
    if n == 0:
        return labels
    where = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}
    cur = 0
    for s in range(n):
        if labels[s] >= 0:
            continue
        labels[s] = cur
        stack = [s]
        while stack:
            k = stack.pop()
            a, b = int(ij[k, 0]), int(ij[k, 1])
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    j = where.get((a + da, b + db))
                    if j is not None and labels[j] < 0:
                        labels[j] = cur
                        stack.append(j)
        cur += 1
    return labels


def _axis(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit principal axis (ties and single cells -> +y)."""
    c = xy.mean(0)
    d = xy - c
    if len(xy) < 2 or not np.any(d):
        return c, np.array([0.0, 1.0])
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    a = vt[0]
    if a[1] < 0 or (abs(a[1]) < 1e-9 and a[0] < 0):
        a = -a
    return c, a


def polyline(xy: np.ndarray, step: float) -> list[list[float]]:
    """Ordered centreline of one component: bin along the principal axis, average across."""
    c, a = _axis(xy)
    n = np.array([-a[1], a[0]])
    s = (xy - c) @ a
    t = (xy - c) @ n
    b = np.floor(s / step).astype(np.int64)
    out = []
    for k in np.unique(b):
        m = b == k
        p = c + a * float(s[m].mean()) + n * float(t[m].mean())
        out.append([round(float(p[0]), 2), round(float(p[1]), 2)])
    return out


def _box(xy: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    c, a = _axis(xy)
    n = np.array([-a[1], a[0]])
    s, t = (xy - c) @ a, (xy - c) @ n
    s0, s1 = float(s.min()) - voxel / 2, float(s.max()) + voxel / 2
    t0, t1 = float(t.min()) - voxel / 2, float(t.max()) + voxel / 2
    center = c + a * (s0 + s1) / 2 + n * (t0 + t1) / 2
    return center, a, s1 - s0, t1 - t0


def motion_state(x: float, y: float, yaw: float, speed: float) -> str:
    """Motion relative to the ego (ego heading is +y, yaw pi/2)."""
    if not math.isfinite(speed) or speed < PARKED_MPS:
        return "parked"
    along = math.sin(yaw)  # cos of the angle between the mover's heading and +y
    if along > ALONG_COS:
        return "alongside" if abs(x) > ALONGSIDE_X_M and abs(y) < ALONGSIDE_Y_M else "same_way"
    if along < -ALONG_COS:
        return "oncoming"
    return "crossing"


def _edges(xy: np.ndarray, voxel: float) -> list[dict[str, Any]]:
    if len(xy) < MIN_CELLS["edge"]:
        return []
    row = np.floor(xy[:, 1] / voxel).astype(np.int64)
    rows = np.unique(row)
    left, right = [], []
    for r in rows:
        xs = xy[row == r, 0]
        yc = (r + 0.5) * voxel
        left.append((r, float(xs.min()) - voxel / 2, yc))
        right.append((r, float(xs.max()) + voxel / 2, yc))
    out = []
    for side in (left, right):
        seg: list[list[float]] = []
        prev = None
        for r, x, y in side:
            if prev is not None and r - prev > EDGE_GAP_ROWS + 1:
                if len(seg) >= 2:
                    out.append({"pts": seg, "kind": "occ"})
                seg = []
            seg.append([round(x, 2), round(y, 2)])
            prev = r
        if len(seg) >= 2:
            out.append({"pts": seg, "kind": "occ"})
    return out


def viz_state_from_cells(centers: np.ndarray, cls: np.ndarray, *, voxel_m: float = 0.4) -> dict[str, Any]:
    """One frame's scene fields from occupied cells (centres (N,3), class id (N,)).

    ``tracks`` here are single-frame detections (no id, speed or motion);
    :class:`OccTracker` turns them into tracks over time.
    """
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    cls = np.asarray(cls, np.int64).reshape(-1)
    xy = centers[:, :2]
    ij = np.floor(xy / voxel_m).astype(np.int64)
    state: dict[str, Any] = {"occ_source": "camera", "lanes_ext": [], "road_edges": [], "tracks": [],
                             "crosswalks": [], "stop_lines": []}
    keep = np.argsort(-centers[:, 2], kind="stable")[:MAX_OCC_CELLS] if len(centers) > MAX_OCC_CELLS else slice(None)
    state["occ_cells"] = [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2),
                           CLASS_NAMES[int(c)] if 0 <= int(c) < len(CLASS_NAMES) else "unknown"]
                          for p, c in zip(centers[keep], cls[keep])]

    def comps(name: str):
        m = np.flatnonzero(cls == _cid(name))
        if not len(m):
            return
        # one bird's-eye cell per (i, j): z-stacked cells are one footprint cell
        uij, first = np.unique(ij[m], axis=0, return_index=True)
        lab = bev_components(uij)
        bev_rows = m[first]
        for k in range(int(lab.max()) + 1):
            sel = lab == k
            members = np.flatnonzero(np.isin(ij[m], uij[sel]).all(1))
            yield xy[bev_rows[sel]], m[members]

    for name, paint in (("lane_white", "white"), ("lane_yellow", "yellow")):
        for pts, _ in comps(name) or []:
            if len(pts) >= MIN_CELLS["lane"]:
                state["lanes_ext"].append({"pts": polyline(pts, voxel_m), "paint": paint, "kind": "occ"})
    drv = np.flatnonzero(cls == _cid("drivable"))
    if len(drv):
        uij = np.unique(ij[drv], axis=0)
        state["road_edges"].extend(_edges((uij + 0.5) * voxel_m, voxel_m))
    for pts, _ in comps("curb") or []:
        if len(pts) >= MIN_CELLS["edge"]:
            state["road_edges"].append({"pts": polyline(pts, voxel_m), "kind": "curb"})
    for pts, _ in comps("stop_line") or []:
        if len(pts) >= MIN_CELLS["stop_line"]:
            state["stop_lines"].append({"pts": polyline(pts, voxel_m)})
    for pts, _ in comps("crosswalk") or []:
        if len(pts) >= MIN_CELLS["crosswalk"]:
            c, a, ln, wd = _box(pts, voxel_m)
            n = np.array([-a[1], a[0]])
            corners = [c + a * sa * ln / 2 + n * sb * wd / 2 for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
            state["crosswalks"].append({"pts": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in corners]})
    for pts, _ in comps("vehicle") or []:
        if len(pts) < MIN_CELLS["vehicle"]:
            continue
        c, a, ln, wd = _box(pts, voxel_m)
        if wd > ln:
            a, ln, wd = np.array([-a[1], a[0]]), wd, ln
        state["tracks"].append({"class": "vehicle", "x": round(float(c[0]), 2), "y": round(float(c[1]), 2),
                                "axis_yaw": round(math.atan2(float(a[1]), float(a[0])), 3),
                                "length": round(ln, 2), "width": round(wd, 2)})
    return state


def viz_state_from_infer(res: Mapping[str, Any], grid: Any) -> dict[str, Any]:
    """``infer_sparse`` result -> scene-state fields."""
    idx = np.asarray(res["idx"]).reshape(-1, 3)
    sem = np.asarray(res["sem"]).reshape(len(idx), -1)
    cls = sem.argmax(1) if len(idx) else np.zeros(0, np.int64)
    return viz_state_from_cells(grid.centers(idx), cls, voxel_m=float(grid.voxel_m))


# Display colours per drawn thing. Michael's rule: white paint white, yellow
# paint yellow, red ONLY for road edges and curbs. Object colours follow the
# Tesla City Streets legend (viz_semantics.md 1.1) except parked, which the
# legend draws red: red is reserved, so parked is orange (as PR #232 M1 draws it).
VIZ_COLORS = {
    "lane:white": "white", "lane:yellow": "yellow", "edge": "red", "curb": "red",
    "crosswalk": "white", "stop_line": "white",
    "same_way": "green", "alongside": "blue", "oncoming": "amber", "crossing": "purple", "parked": "orange",
}


def color_of(kind: str, rec: Mapping[str, Any]) -> str:
    """Colour for one message element (``kind`` is lanes/edges/tracks/crosswalks/stop_lines)."""
    if kind == "lanes":
        return VIZ_COLORS["lane:yellow" if rec.get("paint") == "yellow" else "lane:white"]
    if kind == "edges":
        return VIZ_COLORS["curb" if rec.get("src") == "curb" else "edge"]
    if kind == "tracks":
        return VIZ_COLORS.get(str(rec.get("motion")), "white")
    return VIZ_COLORS["crosswalk" if kind == "crosswalks" else "stop_line"]


class OccTracker:
    """Vehicle detections over frames -> tracks with id, velocity, heading and motion.

    Positions are kept in an odometry frame integrated from ego speed and yaw
    rate, so a parked car stays still while the ego drives past it. Velocity
    is a least-squares line through the last ``history`` positions. No motion
    state (and no ``v``) until a track has ``min_frames`` detections spanning
    ``min_span_s``: one frame never decides motion.
    """

    def __init__(self, *, gate_m: float = 3.0, min_frames: int = 3, min_span_s: float = 0.2,
                 max_age_s: float = 0.6, history: int = 10) -> None:
        self.gate_m, self.min_frames, self.min_span_s = gate_m, min_frames, min_span_s
        self.max_age_s, self.history = max_age_s, history
        self.pose = np.zeros(3)  # X, Y, psi of the ego in the odometry frame (psi CCW, 0 = +y forward)
        self.t: float | None = None
        self.tracks: dict[int, dict[str, Any]] = {}
        self._next = 0

    def _rot(self, psi: float) -> np.ndarray:
        c, s = math.cos(psi), math.sin(psi)
        return np.array([[c, -s], [s, c]])

    def update(self, dets: list[Mapping[str, Any]], t: float, *, ego_speed_mps: float = 0.0,
               yaw_rate: float = 0.0) -> list[dict[str, Any]]:
        if self.t is not None:
            dt = max(0.0, float(t) - self.t)
            self.pose[:2] += self._rot(self.pose[2]) @ np.array([0.0, float(ego_speed_mps) * dt])
            self.pose[2] += float(yaw_rate) * dt
        self.t = float(t)
        R = self._rot(self.pose[2])
        odo = [R @ np.array([d["x"], d["y"]]) + self.pose[:2] for d in dets]
        # predicted positions of live tracks
        live = {k: v for k, v in self.tracks.items() if self.t - v["hist"][-1][0] <= self.max_age_s}
        pred = {k: v["hist"][-1][1] + v.get("vel", np.zeros(2)) * (self.t - v["hist"][-1][0]) for k, v in live.items()}
        pairs = sorted((float(np.linalg.norm(p - pred[k])), i, k) for i, p in enumerate(odo) for k in pred)
        used_d, used_k, match = set(), set(), {}
        for dist, i, k in pairs:
            if dist > self.gate_m or i in used_d or k in used_k:
                continue
            match[i] = k
            used_d.add(i)
            used_k.add(k)
        out = []
        for i, d in enumerate(dets):
            k = match.get(i)
            if k is None:
                k = self._next
                self._next += 1
                live[k] = {"hist": []}
            tr = live[k]
            tr["hist"] = (tr["hist"] + [(self.t, odo[i])])[-self.history:]
            row = {"id": k, "class": d.get("class", "vehicle"), "x": d["x"], "y": d["y"],
                   "length": d.get("length"), "width": d.get("width")}
            ts = np.array([h[0] for h in tr["hist"]])
            yaw = float(d.get("axis_yaw", math.pi / 2))
            if len(ts) >= self.min_frames and ts[-1] - ts[0] >= self.min_span_s:
                P = np.array([h[1] for h in tr["hist"]])
                tc = ts - ts.mean()
                vel = (tc[:, None] * (P - P.mean(0))).sum(0) / float((tc * tc).sum())
                tr["vel"] = vel
                v_ego = R.T @ vel  # absolute velocity, ego-frame axes
                speed = float(np.hypot(*v_ego))
                if speed >= PARKED_MPS:
                    yaw = math.atan2(float(v_ego[1]), float(v_ego[0]))
                row["speed_mps"] = round(speed, 2)
                row["motion"] = motion_state(float(d["x"]), float(d["y"]), yaw, speed)
            row["yaw"] = round(yaw, 3)
            out.append(row)
        self.tracks = {k: v for k, v in live.items() if self.t - v["hist"][-1][0] <= self.max_age_s}
        return out


V1_KEYS = frozenset({"v", "type", "seq", "t", "ego", "lead_id", "aeb", "lanes", "edges", "tracks", "signs", "path",
                     "path_conf"})
V1_CAPS = {"tracks": 64, "lanes": 16, "edges": 8, "points": 120}


def _cap_pts(pts: list[list[float]]) -> list[list[float]]:
    n = V1_CAPS["points"]
    if len(pts) <= n:
        return pts
    step = len(pts) / float(n)
    return [pts[int(i * step)] for i in range(n - 1)] + [pts[-1]]


def occ_to_viz(outputs: Mapping[str, Any], *, grid: Any = None, t: float = 0.0, seq: int = 0,
               ego_speed_mps: float = 0.0, yaw_rate: float = 0.0, tracker: OccTracker | None = None) -> dict[str, Any]:
    """Occupancy outputs -> a ``gvd_scene`` v1 message (viz_cpp.md @ f3555bf) plus optional extras.

    ``outputs`` is an ``infer_sparse`` result (needs ``grid``) or a frame
    from :func:`viz_state_from_cells`. v1 fields carry lanes, edges and
    tracks. Extras v1 ignores: ``lanes[].src``, ``edges[].src``
    (``drivable`` | ``curb``), ``tracks[].motion``, and ``occ`` =
    ``{cells, crosswalks, stop_lines}``. ``ego``/``aeb``/``path``/``signs``
    are neutral placeholders: the supervisor owns them.
    """
    frame = viz_state_from_infer(outputs, grid) if "idx" in outputs else dict(outputs)
    tracker = tracker if tracker is not None else OccTracker()
    tracks = tracker.update(frame.get("tracks", []), t, ego_speed_mps=ego_speed_mps, yaw_rate=yaw_rate)
    msg: dict[str, Any] = {
        "v": 1, "type": "gvd_scene", "seq": int(seq), "t": round(float(t), 4),
        "ego": {"engaged": False, "speed_mps": round(float(ego_speed_mps), 2)},
        "aeb": "off", "lanes": [], "edges": [], "tracks": [], "signs": [], "path": {"pts": []}, "path_conf": 0.0,
    }
    for ln in frame.get("lanes_ext", [])[:V1_CAPS["lanes"]]:
        msg["lanes"].append({"pts": _cap_pts(ln["pts"]), "paint": ln["paint"], "kind": "detected", "src": "occ"})
    for ed in frame.get("road_edges", [])[:V1_CAPS["edges"]]:
        msg["edges"].append({"pts": _cap_pts(ed["pts"]), "kind": "detected",
                             "src": "curb" if ed.get("kind") == "curb" else "drivable"})
    for tr in tracks[:V1_CAPS["tracks"]]:
        row = {"id": int(tr["id"]), "cls": str(tr["class"]), "x": tr["x"], "y": tr["y"], "yaw": tr["yaw"]}
        if tr.get("length") is not None:
            row["l"], row["w"] = tr["length"], tr["width"]
        if "speed_mps" in tr:
            row["v"] = tr["speed_mps"]
        if "motion" in tr:
            row["motion"] = tr["motion"]
        msg["tracks"].append(row)
    msg["occ"] = {"cells": frame.get("occ_cells", []), "crosswalks": frame.get("crosswalks", []),
                  "stop_lines": frame.get("stop_lines", [])}
    return msg


def validate_v1(msg: Mapping[str, Any]) -> list[str]:
    """Required v1 fields (viz_cpp.md @ f3555bf). Unknown keys are allowed and ignored, as the C++ parser does."""
    errs: list[str] = []

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def pts_ok(p, n=2):
        return isinstance(p, list) and len(p) >= n and len(p) <= V1_CAPS["points"] and all(
            isinstance(q, list) and len(q) == 2 and num(q[0]) and num(q[1]) for q in p)

    if msg.get("v") != 1 or msg.get("type") != "gvd_scene":
        errs.append("v/type")
    if not num(msg.get("seq")) or not num(msg.get("t")):
        errs.append("seq/t")
    ego = msg.get("ego")
    if not isinstance(ego, Mapping) or not isinstance(ego.get("engaged"), bool) or not num(ego.get("speed_mps")):
        errs.append("ego")
    if not isinstance(msg.get("aeb"), str):
        errs.append("aeb")
    for key in ("lanes", "edges", "tracks", "signs"):
        if not isinstance(msg.get(key), list):
            errs.append(key)
    if len(msg.get("lanes") or []) > V1_CAPS["lanes"] or len(msg.get("edges") or []) > V1_CAPS["edges"] \
            or len(msg.get("tracks") or []) > V1_CAPS["tracks"]:
        errs.append("caps")
    for ln in msg.get("lanes") or []:
        if ln.get("paint") not in ("white", "yellow") or ln.get("kind") not in ("detected", "predicted") \
                or not pts_ok(ln.get("pts")):
            errs.append(f"lane {ln}")
    for ed in msg.get("edges") or []:
        if ed.get("kind") not in ("detected", "predicted") or not pts_ok(ed.get("pts")):
            errs.append(f"edge {ed}")
    for tr in msg.get("tracks") or []:
        if not isinstance(tr.get("cls"), str) or not all(num(tr.get(k)) for k in ("x", "y")) \
                or ("yaw" in tr and not num(tr["yaw"])) or ("motion" in tr and tr["motion"] not in MOTION_STATES):
            errs.append(f"track {tr}")
    path = msg.get("path")
    if not isinstance(path, Mapping) or not isinstance(path.get("pts"), list):
        errs.append("path")
    return errs
