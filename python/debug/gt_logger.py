"""Sim-only ground-truth logger.

Off unless ``GVD_GT_LOG=1`` or the supervisor ``--gt-log`` flag. Each tick
appends one JSON object under ``logs/gt_<stamp>.jsonl``. Road edges are read
once from BeamNG (``get_roads`` / ``get_road_edges``) and then queried
locally, so a later tick does not sit on the research socket. Missing road
data is a null offset, not a guessed lane.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from python.debug.lane_targets import LOOKAHEADS_M, lookahead_xs
from python.debug.gt_geom import (
    CONNECT_HEADING,
    CurvePreview,
    PreviewMemory,
    RoadFollow,
    RoadHit,
    RoadPoly,
    ego_center_x,
    heading_from_dir,
    lane_still_beside,
    parse_road_edges,
    project_follow,
    road_width_floor,
    project_road,
    update_curve_preview,
)

FRAME_EVERY_S = 4.0
DEPART_BURST = 3


def gt_log_enabled(flag: bool = False) -> bool:
    if flag:
        return True
    raw = os.environ.get("GVD_GT_LOG", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def logs_dir(root: Path | None = None) -> Path:
    """Live GT logs. Default is ``E:\\GVD_data\\logs`` (or ``GVD_DATA_DIR``)."""
    if root is not None:
        base = Path(root)
    else:
        from python.data.data_root import logs_dir as data_logs
        from python.data.data_root import use_data_disk

        if use_data_disk():
            base = data_logs()
        else:
            base = Path(__file__).resolve().parents[2] / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def read_sim_time(bng: Any) -> float | None:
    """Simulation clock, if this BeamNGpy build exposes one. None if it does not."""
    if bng is None:
        return None
    for name in ("get_gamestate", "get_game_state"):
        fn = getattr(bng, name, None)
        if not callable(fn):
            continue
        try:
            state = fn()
        except Exception:
            continue
        if isinstance(state, dict):
            for key in ("time", "simTime", "simulationTime"):
                found = _num(state.get(key))
                if found is not None:
                    return found
    return None


def _exc_text(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


def _call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    try:
        return fn(*args, **kwargs), ""
    except TypeError:
        try:
            return fn(*args), ""
        except Exception as exc:
            return None, _exc_text(exc)
    except Exception as exc:
        return None, _exc_text(exc)


def polys_from_network(spec: Any) -> list[RoadPoly]:
    """Roads from ``get_road_network(include_edges=True)``."""
    if isinstance(spec, dict) and "roads" in spec and isinstance(spec["roads"], dict):
        spec = spec["roads"]
    elif isinstance(spec, dict) and "network" in spec and isinstance(spec["network"], dict):
        spec = spec["network"]
    if not isinstance(spec, dict):
        return []
    polys: list[RoadPoly] = []
    for road_id, road in spec.items():
        edges = None
        if isinstance(road, dict):
            driv = road.get("drivability", road.get("drivable", 1))
            try:
                if float(driv) <= 0.0:
                    continue
            except (TypeError, ValueError):
                pass
            edges = road.get("edges") or road.get("nodes")
        elif isinstance(road, list):
            edges = road
        poly = parse_road_edges(str(road_id), edges)
        if poly is not None:
            if isinstance(road, dict):
                try:
                    count = int(float(road.get("lanesLeft") or 0)) + int(float(road.get("lanesRight") or 0))
                except (TypeError, ValueError):
                    count = 0
                if 1 <= count <= 8:
                    poly.lanes = count
            polys.append(road_width_floor(poly, road))
    return polys


def raw_road_network(bng: Any) -> tuple[Any, Any, str]:
    """``(road_network spec, get_roads dict or None, note)``.

    The spec is whatever ``get_road_network`` returned, before it is parsed
    into centerlines. ``get_roads`` is included when that call succeeds.
    """
    if bng is None:
        return None, None, "no beamng"
    notes: list[str] = []
    spec = None
    owners = []
    scenario = getattr(bng, "scenario", None)
    if scenario is not None:
        owners.append(scenario)
    owners.append(bng)
    for owner in owners:
        fn = getattr(owner, "get_road_network", None)
        if not callable(fn):
            continue
        got, note = _call(fn, include_edges=True, drivable_only=True)
        if note:
            notes.append(f"get_road_network {note}")
            got, note2 = _call(fn)
            if note2:
                notes.append(f"get_road_network() {note2}")
        if got is not None:
            spec = got
            break
    roads = None
    get_roads = getattr(bng, "get_roads", None)
    if callable(get_roads):
        got, note = _call(get_roads)
        if note:
            notes.append(f"get_roads failed: {note}")
        elif isinstance(got, dict):
            roads = got
    if spec is None and roads is None:
        return None, None, "; ".join(notes) if notes else "no road network"
    return spec, roads, ""


def load_road_polys(bng: Any) -> tuple[list[RoadPoly], str]:
    """One snapshot of the level's road network.

    BeamNGpy 1.36 deprecates ``get_roads``; Tech 0.39 raises ``BNGError``
    from it. ``scenario.get_road_network(include_edges=True)`` is the call
    that returns left/middle/right edges. The note keeps the exception text.
    """
    if bng is None:
        return [], "no beamng"
    notes: list[str] = []
    owners = []
    scenario = getattr(bng, "scenario", None)
    if scenario is not None:
        owners.append(scenario)
    owners.append(bng)
    for owner in owners:
        fn = getattr(owner, "get_road_network", None)
        if not callable(fn):
            continue
        spec, note = _call(fn, include_edges=True, drivable_only=True)
        if note:
            notes.append(f"get_road_network {note}")
            spec, note2 = _call(fn)
            if note2:
                notes.append(f"get_road_network() {note2}")
        polys = polys_from_network(spec)
        if polys:
            return polys, ""
        if spec is not None and not note:
            notes.append("get_road_network returned no edges")
    get_roads = getattr(bng, "get_roads", None)
    get_edges = getattr(bng, "get_road_edges", None)
    if callable(get_roads):
        roads, note = _call(get_roads)
        if note:
            notes.append(f"get_roads failed: {note}")
        elif isinstance(roads, dict) and roads and callable(get_edges):
            polys = []
            for road_id in roads:
                edges, edge_note = _call(get_edges, road_id)
                if edge_note or edges is None:
                    continue
                poly = parse_road_edges(str(road_id), edges)
                if poly is not None:
                    polys.append(poly)
            if polys:
                return polys, ""
            notes.append("get_road_edges returned no centerlines")
        elif not note:
            notes.append("get_roads returned no roads")
    elif not notes:
        notes.append("beamngpy has no get_road_network or get_roads")
    return [], "; ".join(notes) if notes else "no road network"


def _x_at_y(poly: list, y_m: float) -> float | None:
    pts: list[tuple[float, float]] = []
    for raw in poly or []:
        if isinstance(raw, dict):
            x, y = raw.get("x"), raw.get("y")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            x, y = raw[0], raw[1]
        else:
            continue
        try:
            pts.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[1])
    if y_m < pts[0][1] or y_m > pts[-1][1]:
        # Nearest end if the line does not reach the lookahead.
        end = pts[0] if abs(pts[0][1] - y_m) < abs(pts[-1][1] - y_m) else pts[-1]
        if abs(end[1] - y_m) > 4.0:
            return None
        return end[0]
    for a, b in zip(pts, pts[1:]):
        if a[1] <= y_m <= b[1] or b[1] <= y_m <= a[1]:
            dy = b[1] - a[1]
            if abs(dy) < 1e-6:
                return (a[0] + b[0]) * 0.5
            t = (y_m - a[1]) / dy
            return a[0] + t * (b[0] - a[0])
    return None


def detected_lane_xs(lanes_bev: list | None, y_m: float = 8.0) -> tuple[float | None, float | None, float | None]:
    """Left, right, and mean x of detected polylines at ``y_m`` (x right).

    A boundary past 1.9 lane widths (6.65 m) is the next road or a
    side-camera miss. The ecf5d35 log's det_right_x of +8..+16 m was that
    miss. It is not the ego lane, so it does not count here.
    """
    from python.planning.path_predictor import LANE_BOUNDARY_WIDTHS, lane_point_plausible

    xs: list[float] = []
    for poly in lanes_bev or []:
        x = _x_at_y(poly, y_m)
        if x is None:
            continue
        if not lane_point_plausible(x, y_m, widths=LANE_BOUNDARY_WIDTHS):
            continue
        xs.append(x)
    if not xs:
        return None, None, None
    return min(xs), max(xs), sum(xs) / len(xs)


def _pose(vdata: Any) -> tuple[tuple[float, float, float] | None, float | None, float | None]:
    if vdata is None:
        return None, None, None
    pos = getattr(vdata, "pos", None)
    direction = getattr(vdata, "dir", None)
    speed = _num(getattr(vdata, "speed_mps", None))
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None, None, speed
    try:
        xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
    except (TypeError, ValueError):
        return None, None, speed
    heading = None
    if isinstance(direction, (list, tuple)) and len(direction) >= 2:
        try:
            heading = heading_from_dir(float(direction[0]), float(direction[1]))
        except (TypeError, ValueError):
            heading = None
    return xyz, heading, speed


def _draw_bev(lanes_bev: list | None, lane_x: float | None) -> Any:
    import cv2
    import numpy as np

    size = 220
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    cx, cy = size // 2, size - 28
    scale = 6.0  # px per metre

    def pix(x: float, y: float) -> tuple[int, int]:
        return int(cx + x * scale), int(cy - y * scale)

    cv2.line(img, (cx, 8), (cx, size - 8), (220, 220, 220), 1)
    cv2.circle(img, (cx, cy), 4, (40, 40, 40), -1, cv2.LINE_AA)
    for poly in lanes_bev or []:
        pts = []
        for p in poly or []:
            if isinstance(p, dict):
                x, y = p.get("x"), p.get("y")
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
            else:
                continue
            try:
                pts.append(pix(float(x), float(y)))
            except (TypeError, ValueError):
                continue
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (40, 140, 60), 2, cv2.LINE_AA)
    if lane_x is not None:
        try:
            x = float(lane_x)
            y_px = 8.0
        except (TypeError, ValueError):
            x = None
        if x is not None and x == x:
            cv2.circle(img, pix(x, y_px), 4, (40, 90, 200), -1, cv2.LINE_AA)
    return img


def save_debug_frame(
    path: Path,
    *,
    stitch: Any = None,
    lanes_bev: list | None = None,
    hud: list[str] | None = None,
    lane_x: float | None = None,
) -> Path | None:
    """Stitch (if we have one) with a bird's-eye of the lane lines in the corner."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    bev = _draw_bev(lanes_bev, lane_x)
    if stitch is None:
        canvas = bev
    else:
        frame = np.asarray(stitch)
        if frame.ndim != 3 or frame.shape[2] != 3:
            canvas = bev
        else:
            canvas = frame.copy()
            bh, bw = bev.shape[:2]
            h, w = canvas.shape[:2]
            if h >= bh + 8 and w >= bw + 8:
                canvas[8 : 8 + bh, w - bw - 8 : w - 8] = bev
    y = 22
    for line in hud or []:
        cv2.putText(
            canvas, line[:80], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, line[:80], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (20, 20, 20), 1, cv2.LINE_AA,
        )
        y += 18
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        return None
    return path


class GroundTruthLogger:
    """Append-only session. ``path`` is the JSONL file."""

    def __init__(self, path: Path, frames: Path) -> None:
        self.path = Path(path)
        self.frames = Path(frames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames.mkdir(parents=True, exist_ok=True)
        self._roads: list[RoadPoly] | None = None
        self._road_note = "road network pending"
        self._road_try_wall = 0.0
        self._prev_damage: float | None = None
        self._last_sim: float | None = None
        self._wall0 = time.time()
        self._last_frame_wall = 0.0
        self._burst = 0
        self._was_out = False
        self._ticks = 0
        self._preview = PreviewMemory()
        self._follow = RoadFollow()
        # Set by curve_preview. write() uses it so gt_center is the road
        # the wheel accepted, not a crossing polyline chosen the same tick.
        self._followed: RoadHit | None = None
        self._last_good: RoadHit | None = None

    @classmethod
    def open(cls, root: Path | None = None) -> GroundTruthLogger:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        folder = logs_dir(root)
        path = folder / f"gt_{stamp}.jsonl"
        n = 2
        while path.exists():
            path = folder / f"gt_{stamp}_{n}.jsonl"
            n += 1
        frames = path.with_suffix("")
        return cls(path, frames)

    def _ensure_roads(self, bng: Any, socket_busy: bool) -> None:
        if self._roads is not None or socket_busy or bng is None:
            return
        now = time.time()
        if self._road_try_wall and (now - self._road_try_wall) < 2.0:
            return
        self._road_try_wall = now
        polys, note = load_road_polys(bng)
        if polys:
            self._roads = polys
            self._road_note = ""
            print(f"[GVD] gt roads={len(polys)}", flush=True)
        else:
            self._road_note = note
            print(f"[GVD] gt roads: {note}", flush=True)

    def center_x(self, vdata: Any, bng: Any = None, socket_busy: bool = False) -> float | None:
        """Road-network center at 8 m, the same number the log stores as ``gt_center_x``.

        Off-road or a missing pose is None. The caller does not invent a lane.
        """
        self._ensure_roads(bng, socket_busy)
        if not self._roads:
            return None
        pos, heading, _speed = _pose(vdata)
        if pos is None:
            return None
        hit = project_road(self._roads, pos[0], pos[1], heading)
        if hit is None or hit.off_road or hit.lane_offset_m is None:
            return None
        if hit.lane_width_m is None or float(hit.lane_width_m) <= 0.0:
            return None
        return float(ego_center_x(hit.lane_offset_m, hit.heading_err_rad, 8.0))

    def curve_preview(
        self,
        vdata: Any,
        bng: Any = None,
        socket_busy: bool = False,
        dt_s: float | None = None,
    ) -> CurvePreview | None:
        """Lane sample for the wheel, held on the road we are already on.

        A crossing polyline does not replace that road. The end of the
        mesh, with no continuation, is lost immediately: the caller brakes
        and the last curve stays on the wheel. ``gt_center_x`` logged from
        ``write`` uses this same hit, so it matches ``lane_x`` instead of
        the rejected polyline.
        """
        self._ensure_roads(bng, socket_busy)
        dt = 0.1 if dt_s is None else dt_s
        self._followed = None
        if not self._roads:
            return None
        pos, heading, speed_pose = _pose(vdata)
        if pos is None or heading is None:
            return update_curve_preview(
                self._preview,
                offset_m=None,
                heading_err_rad=None,
                curvature=0.0,
                road_id="",
                beside=False,
                dt_s=dt,
                discontinuous=True,
            )
        hit = project_follow(
            self._roads, pos[0], pos[1], heading, self._follow, speed_mps=speed_pose or 0.0,
        )
        # A rejected crossing road still carries that road's offset. Log the
        # last accepted polyline instead, which is what lane_x is held to.
        bad_jump = (
            hit is not None
            and hit.discontinuous
            and hit.heading_err_rad is not None
            and abs(hit.heading_err_rad) > CONNECT_HEADING
        )
        if hit is not None and not bad_jump:
            self._last_good = hit
            self._followed = hit
        else:
            self._followed = self._last_good
        if hit is None or hit.heading_err_rad is None or hit.lane_offset_m is None:
            return update_curve_preview(
                self._preview,
                offset_m=None,
                heading_err_rad=None,
                curvature=0.0,
                road_id="",
                beside=False,
                dt_s=dt,
                discontinuous=True,
            )
        kappa = float(hit.curvature)
        # The walk died at the end of the mesh. Keep the bend we already had
        # instead of commanding a straight line off the end.
        if hit.ahead_m < 4.0 and self._preview.offset_m is not None:
            kappa = float(self._preview.curvature)
        aligned = abs(hit.heading_err_rad) <= CONNECT_HEADING
        beside = (not hit.discontinuous) and (lane_still_beside(hit) or aligned)
        return update_curve_preview(
            self._preview,
            offset_m=hit.lane_offset_m,
            heading_err_rad=hit.heading_err_rad,
            curvature=kappa,
            road_id=hit.road_id,
            beside=beside,
            dt_s=dt,
            discontinuous=bool(hit.discontinuous),
        )

    def write(
        self,
        *,
        bng: Any = None,
        vdata: Any = None,
        engaged: bool = False,
        lane_x: float | None = None,
        lane_src: str = "none",
        steer: float = 0.0,
        steer_why: str = "idle",
        frame_cmd_ms: float = -1.0,
        frame_age_s: float | None = None,
        stitch: Any = None,
        lanes_bev: list | None = None,
        socket_busy: bool = False,
        model_lane_x: float | None = None,
        model_conf: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_roads(bng, socket_busy)
        pos, heading, speed = _pose(vdata)
        sim = None if socket_busy else read_sim_time(bng)
        if sim is None:
            t = time.time() - self._wall0
            t_source = "wall"
        else:
            self._last_sim = sim
            t = sim
            t_source = "sim"
        hit: RoadHit | None = self._followed
        if hit is None and pos is not None and self._roads:
            hit = project_road(self._roads, pos[0], pos[1], heading)
        damage = _num(getattr(vdata, "damage", None)) if vdata is not None else None
        delta = None
        if damage is not None and self._prev_damage is not None:
            delta = damage - self._prev_damage
        if damage is not None:
            self._prev_damage = damage
        lane_off = None if hit is None else hit.lane_offset_m
        width = None if hit is None else hit.lane_width_m
        gt_center = None
        gt_left = None
        gt_right = None
        ahead = None
        if hit is not None and lane_off is not None and width is not None:
            gt_center = ego_center_x(lane_off, hit.heading_err_rad, 8.0)
            half = float(width) * 0.5
            gt_left = gt_center - half
            gt_right = gt_center + half
            ahead = lookahead_xs(lane_off, hit.heading_err_rad, width, ys=LOOKAHEADS_M)
        det_left, det_right, det_center = detected_lane_xs(lanes_bev, 8.0)
        in_lane = None
        if hit is not None and width is not None and width > 0.0:
            in_lane = (not hit.off_road) and abs(lane_off or 0.0) <= width * 0.5
        out = bool(engaged and in_lane is False)
        departed = out and not self._was_out
        self._was_out = out
        if departed:
            self._burst = DEPART_BURST
        row: dict[str, Any] = {
            "t": t,
            "t_source": t_source,
            "wall": time.time(),
            "pos": None if pos is None else [pos[0], pos[1], pos[2]],
            "heading_rad": heading,
            "speed_mps": speed,
            "road_id": None if hit is None else hit.road_id,
            "road_offset_m": None if hit is None else hit.offset_m,
            "road_heading_err_rad": None if hit is None else hit.heading_err_rad,
            "lane_id": None if hit is None else hit.lane_id,
            "lane_offset_m": lane_off,
            "lane_width_m": width,
            "off_road": None if hit is None else hit.off_road,
            "dist_m": None if hit is None else hit.dist_m,
            "in_lane": in_lane,
            "damage": damage,
            "damage_delta": delta,
            "lane_x": None if lane_x is None else float(lane_x),
            "model_lane_x": None if model_lane_x is None else float(model_lane_x),
            "model_conf": None if model_conf is None else float(model_conf),
            "gt_center_x": gt_center,
            "gt_left_x": gt_left,
            "gt_right_x": gt_right,
            "gt_lookahead_m": None if ahead is None else ahead["y_m"],
            "gt_center_xs": None if ahead is None else ahead["center_x"],
            "gt_left_xs": None if ahead is None else ahead["left_x"],
            "gt_right_xs": None if ahead is None else ahead["right_x"],
            "det_left_x": det_left,
            "det_right_x": det_right,
            "det_center_x": det_center,
            "lane_src": str(lane_src or "none"),
            "steer": float(steer),
            "steer_why": str(steer_why or ""),
            "frame_cmd_ms": float(frame_cmd_ms),
            "frame_age_s": None if frame_age_s is None else float(frame_age_s),
            "engaged": bool(engaged),
            "road_note": self._road_note,
            "departed": departed,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._ticks += 1
        self._maybe_frame(row, stitch, lanes_bev, lane_x, departed)
        from python.debug.remote_status import note_live

        note_live(
            {
                "lane_x": row.get("lane_x"),
                "model_lane_x": row.get("model_lane_x"),
                "model_conf": row.get("model_conf"),
                "gt_center_x": row.get("gt_center_x"),
                "lane_src": row.get("lane_src"),
                "engaged": row.get("engaged"),
                "speed_mps": row.get("speed_mps"),
                "log": str(self.path),
            }
        )
        return row

    def _maybe_frame(
        self,
        row: dict[str, Any],
        stitch: Any,
        lanes_bev: list | None,
        lane_x: float | None,
        departed: bool,
    ) -> None:
        now = time.time()
        periodic = (now - self._last_frame_wall) >= FRAME_EVERY_S
        burst = self._burst > 0
        if not periodic and not burst:
            return
        if periodic:
            self._last_frame_wall = now
        if burst:
            self._burst -= 1
        tag = "depart" if departed or burst else "frame"
        name = f"{tag}_{self._ticks:06d}.jpg"
        off = row.get("lane_offset_m")
        off_s = "na" if off is None else f"{float(off):+.2f}"
        model_x = row.get("model_lane_x")
        model_c = row.get("model_conf")
        gt_c = row.get("gt_center_x")
        hud = [
            f"t={float(row['t']):.2f} engaged={int(bool(row['engaged']))} in_lane={row.get('in_lane')}",
            f"offset={off_s} lane_x={row.get('lane_x')} src={row.get('lane_src')}",
            (
                f"model_x={'na' if model_x is None else f'{float(model_x):+.2f}'} "
                f"conf={'na' if model_c is None else f'{float(model_c):.2f}'} "
                f"gt={'na' if gt_c is None else f'{float(gt_c):+.2f}'}"
            ),
            f"steer={float(row['steer']):+.3f} {row.get('steer_why')} off_road={row.get('off_road')}",
        ]
        save_debug_frame(self.frames / name, stitch=stitch, lanes_bev=lanes_bev, hud=hud, lane_x=lane_x)
