"""Offline clip QC, training-time driving-target masks, and highway lane-spread report.

Training/offline only: runtime (``run_vision.py``, ``perception/*``, ``planning/*``)
never imports this module and stays strictly camera-only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from python.collect.clip_mode import MIN_MAIN_HZ, MIN_MOVING_MPS, SHARP_CURVE_PER_M
from python.collect.liveness import STALL_SPEED_MPS, TELEPORT_SETTLE_S, bad_row_reasons
from python.collect.signal_labels import _as_signal_list, _norm_cls, _xy
from python.collect.static_signs import load_static_catalog, world_signs
from python.debug.gt_geom import lane_center_offsets
from python.debug.lane_targets import SLOT_LEFT, SLOT_RIGHT, lookahead_slot
from python.train.event_mask import event_mask, read_impacts
from python.train.frame_quality import (
    QualityRoad,
    build_road_grid,
    classify_wrong_way,
    load_quality_roads,
    pose_from_row,
)
from python.train.intersection_labels import stop_line_target

CLIP_CLASSES = ("train", "pretrain_only", "reject")
EXCLUDED_TARGET_FIELDS = ("turnsignal", "blinker", "blinkers")

# Signal classes that always mask driving targets (AI ignores stop/yield & runs red lights).
MASKED_SIGNAL_CLASSES = frozenset({"traffic_light", "stop_sign", "yield"})

TELEPORT_MASK_S = float(TELEPORT_SETTLE_S)  # 5.0 s post-teleport settle window
MAP_LOAD_WARMUP_S = 20.0
SIGNAL_MASK_RADIUS_M = 25.0
SIGNAL_MAX_DZ_M = 3.0
STOPLINE_MIN_Y_M = -5.0
STOPLINE_MAX_Y_M = 25.0

UTURN_MIN_DEG = 150.0
UTURN_WINDOW_S = 15.0
UTURN_MAX_NET_DISP_M = 18.0

ABRUPT_YAW_RATE_RAD_S = 1.0
ABRUPT_LAT_ACC_MPS2 = 6.0
ABRUPT_STEER_RATE_S = 2.5

STUCK_MIN_THROTTLE = 0.15
STUCK_MAX_BRAKE = 0.10
STUCK_THROTTLE_S = 2.0

CORNER_MIN_CURV_PER_M = float(SHARP_CURVE_PER_M)  # 0.02 / m (R <= 50 m)
OVER_LINE_MARGIN_M = 0.25

MIN_CLIP_SPAN_S = 10.0
MIN_TRAIN_ROWS = 20
MIN_TRAIN_SPAN_S = 5.0
MAX_TRAIN_STUCK_SHARE = 0.25
MAX_TRAIN_WRONG_WAY_SHARE = 0.10
MAX_TRAIN_UTURN_SHARE = 0.25

REASON_ORDER = (
    "settle_5s",
    "map_warmup",
    "impact",
    "near_signal",
    "stuck",
    "uturn",
    "abrupt_turn",
    "over_line_corner",
    "wrong_way",
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _wrap_angle(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _row_heading_rad(row: Mapping[str, Any]) -> float | None:
    pose = pose_from_row(row)
    if pose is not None:
        return pose[2]
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    quat = ego.get("quat") if isinstance(ego, Mapping) else None
    if isinstance(quat, (list, tuple)) and len(quat) >= 4:
        try:
            qx, qy, qz, qw = (float(quat[k]) for k in range(4))
        except (TypeError, ValueError):
            return None
        # BeamNG forward = quat * (0, 1, 0) in world XY
        fx = 2.0 * (qx * qy - qw * qz)
        fy = 1.0 - 2.0 * (qx * qx + qz * qz)
        if math.hypot(fx, fy) > 1e-6:
            return math.atan2(fy, fx)
    return None


def _row_pose(row: Mapping[str, Any]) -> tuple[float, float, float, float | None] | None:
    pose = pose_from_row(row)
    if pose is not None:
        return pose
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    pos = ego.get("pos") if isinstance(ego, Mapping) else row.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        return None
    x, y = _num(pos[0]), _num(pos[1])
    h = _row_heading_rad(row)
    if x is None or y is None or h is None:
        return None
    z = _num(pos[2]) if len(pos) >= 3 else None
    return x, y, h, z


def _row_xyz(row: Mapping[str, Any]) -> tuple[float, float, float | None] | None:
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    pos = ego.get("pos") if isinstance(ego, Mapping) else row.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        return None
    x, y = _num(pos[0]), _num(pos[1])
    if x is None or y is None:
        return None
    z = _num(pos[2]) if len(pos) >= 3 else None
    return x, y, z


def _extract_world_signals(clip_dir: Path) -> tuple[list[tuple[str, float, float, float | None]], bool]:
    """Deduplicated ``(cls, x, y, z)`` for stoplights / stop signs / yield signs."""
    points: list[tuple[str, float, float, float | None]] = []
    seen: set[tuple[str, int, int, int]] = set()
    has_file = False

    sig_path = clip_dir / "signals.jsonl"
    if sig_path.is_file():
        has_file = True
        for entry in _read_jsonl(sig_path):
            raw = entry.get("signals")
            if raw is None and any(k in entry for k in ("pos", "instance", "kind", "cls")):
                raw = [entry]
            for sig in _as_signal_list(raw):
                cls = _norm_cls(sig.get("cls") or sig.get("kind") or sig.get("type") or sig.get("instance"))
                if cls not in MASKED_SIGNAL_CLASSES:
                    continue
                xy = _xy(sig.get("pos") or sig.get("position"))
                if xy is None:
                    continue
                pos_obj = sig.get("pos") or sig.get("position")
                z = None
                if isinstance(pos_obj, Mapping):
                    z = _num(pos_obj.get("z"))
                elif isinstance(pos_obj, (list, tuple)) and len(pos_obj) >= 3:
                    z = _num(pos_obj[2])
                key = (cls, int(round(xy[0] * 10)), int(round(xy[1] * 10)), int(round((z or 0.0) * 10)))
                if key not in seen:
                    seen.add(key)
                    points.append((cls, float(xy[0]), float(xy[1]), z))

    static_path = clip_dir / "static_signs.json"
    if static_path.is_file():
        has_file = True
        try:
            catalog = load_static_catalog(static_path)
            for item in world_signs(catalog):
                cls = _norm_cls(item.get("cls") or item.get("kind"))
                if cls not in MASKED_SIGNAL_CLASSES:
                    continue
                xy = _xy(item.get("pos"))
                if xy is None:
                    continue
                pos_obj = item.get("pos")
                z = _num(pos_obj[2]) if isinstance(pos_obj, (list, tuple)) and len(pos_obj) >= 3 else None
                key = (cls, int(round(xy[0] * 10)), int(round(xy[1] * 10)), int(round((z or 0.0) * 10)))
                if key not in seen:
                    seen.add(key)
                    points.append((cls, float(xy[0]), float(xy[1]), z))
        except Exception:
            pass

    return points, has_file


def _load_sidecar_by_row(clip_dir: Path, filename: str) -> tuple[dict[int, dict[str, Any]], bool]:
    path = clip_dir / filename
    if not path.is_file():
        return {}, False
    by_i: dict[int, dict[str, Any]] = {}
    for entry in _read_jsonl(path):
        rid = entry.get("i")
        if isinstance(rid, int) and not isinstance(rid, bool):
            by_i[rid] = entry
    return by_i, True


def _resolve_road_dump(clip_dir: Path, rows: list[Mapping[str, Any]], road_dump: Path | str | None) -> Path | None:
    if road_dump is not None:
        cand = Path(road_dump)
        return cand if cand.is_file() else None
    map_name = None
    for r in rows:
        if r.get("map"):
            map_name = str(r["map"])
            break
    if not map_name:
        return None
    for base in (clip_dir, clip_dir.parent, clip_dir.parent / "roads", clip_dir.parent.parent / "roads"):
        for ext in (".json.gz", ".json"):
            cand = base / f"{map_name}{ext}"
            if cand.is_file():
                return cand
    return None


def _compute_highway_spread(
    rows: list[Mapping[str, Any]],
    row_road_info: list[dict[str, Any] | None],
) -> dict[str, Any]:
    offsets: list[float] = []
    norm_positions: list[float] = []
    lane_counts: dict[str, int] = {}
    hwy_rows = 0
    classified = 0
    multi_lane = 0
    rightmost_hits = 0
    leftmost_hits = 0

    for row, info in zip(rows, row_road_info):
        if str(row.get("spawn_type") or "").strip().lower() != "highway":
            continue
        hwy_rows += 1
        if info is None or info.get("road") is None:
            continue
        road: QualityRoad = info["road"]
        offset = _num(info.get("offset_m"))
        align = _num(info.get("align"))
        if offset is None or align is None:
            continue
        classified += 1
        eff_oneway = bool(info.get("effective_oneway"))
        if eff_oneway:
            n_lanes = max(1, int(road.lanes or (road.lanes_left + road.lanes_right) or 1))
            travel_offset = offset
            centers = lane_center_offsets(road.width_m, n_lanes)
        else:
            # Two-way US RHT: +offset is right side when align >= 0, -offset when align < 0.
            if align >= 0.0:
                n_lanes = max(1, int(road.lanes_right or max(1, (road.lanes or 2) // 2)))
                travel_offset = offset
                half_w = road.width_m * 0.5
                lane_w = half_w / float(n_lanes)
                centers = [(k + 0.5) * lane_w for k in range(n_lanes)]
            else:
                n_lanes = max(1, int(road.lanes_left or max(1, (road.lanes or 2) // 2)))
                travel_offset = -offset
                half_w = road.width_m * 0.5
                lane_w = half_w / float(n_lanes)
                centers = [(k + 0.5) * lane_w for k in range(n_lanes)]
        offsets.append(float(travel_offset))
        lane_idx = min(range(n_lanes), key=lambda idx: abs(travel_offset - centers[idx]))
        key = str(lane_idx)
        lane_counts[key] = lane_counts.get(key, 0) + 1
        if n_lanes >= 2:
            multi_lane += 1
            norm = float(lane_idx) / float(n_lanes - 1)
            norm_positions.append(norm)
            if lane_idx == n_lanes - 1:
                rightmost_hits += 1
            if lane_idx == 0:
                leftmost_hits += 1

    out: dict[str, Any] = {
        "highway_rows": int(hwy_rows),
        "classified_rows": int(classified),
        "multi_lane_rows": int(multi_lane),
        "lane_index_counts": dict(sorted(lane_counts.items(), key=lambda kv: int(kv[0]))),
        "rightmost_lane_share": round(rightmost_hits / multi_lane, 4) if multi_lane else None,
        "leftmost_lane_share": round(leftmost_hits / multi_lane, 4) if multi_lane else None,
        "norm_lane_pos_mean": round(float(np.mean(norm_positions)), 4) if norm_positions else None,
        "norm_lane_pos_std": round(float(np.std(norm_positions)), 4) if norm_positions else None,
    }
    if offsets:
        arr = np.asarray(offsets, dtype=np.float64)
        out["offset_m"] = {
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3),
            "p10": round(float(np.percentile(arr, 10)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "max": round(float(arr.max()), 3),
        }
    else:
        out["offset_m"] = None
    return out


def format_highway_spread(spread: Mapping[str, Any]) -> str:
    hwy = int(spread.get("highway_rows") or 0)
    cls = int(spread.get("classified_rows") or 0)
    if hwy == 0:
        return "highway_rows=0"
    rm = spread.get("rightmost_lane_share")
    lm = spread.get("leftmost_lane_share")
    rm_s = f"{100.0 * float(rm):.1f}%" if rm is not None else "n/a"
    lm_s = f"{100.0 * float(lm):.1f}%" if lm is not None else "n/a"
    off = spread.get("offset_m")
    if isinstance(off, Mapping) and off.get("mean") is not None:
        std_v = off.get("std")
        off_s = f"{float(off['mean']):+.2f}±{float(std_v):.2f}m" if std_v is not None else f"{float(off['mean']):+.2f}m"
    else:
        off_s = "n/a"
    return (
        f"highway_rows={hwy} classified={cls} lanes={spread.get('lane_index_counts') or {}} "
        f"rightmost={rm_s} leftmost={lm_s} offset={off_s}"
    )


def merge_highway_spreads(spreads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate clip-level highway spreads into one session-level report."""
    total_hwy = 0
    total_cls = 0
    total_multi = 0
    counts: dict[str, int] = {}
    rm_num = 0.0
    lm_num = 0.0
    norm_sum = 0.0
    off_sum = 0.0
    off_sq_sum = 0.0
    off_n = 0
    for sp in spreads:
        hwy = int(sp.get("highway_rows") or 0)
        cls = int(sp.get("classified_rows") or 0)
        multi = int(sp.get("multi_lane_rows") or 0)
        total_hwy += hwy
        total_cls += cls
        total_multi += multi
        for k, v in (sp.get("lane_index_counts") or {}).items():
            counts[str(k)] = counts.get(str(k), 0) + int(v)
        if multi > 0:
            if sp.get("rightmost_lane_share") is not None:
                rm_num += float(sp["rightmost_lane_share"]) * multi
            if sp.get("leftmost_lane_share") is not None:
                lm_num += float(sp["leftmost_lane_share"]) * multi
            if sp.get("norm_lane_pos_mean") is not None:
                norm_sum += float(sp["norm_lane_pos_mean"]) * multi
        off = sp.get("offset_m")
        if isinstance(off, Mapping) and cls > 0 and off.get("mean") is not None:
            m_val = float(off["mean"])
            s_val = float(off.get("std") or 0.0)
            off_sum += m_val * cls
            off_sq_sum += (s_val * s_val + m_val * m_val) * cls
            off_n += cls
    return {
        "highway_rows": total_hwy,
        "classified_rows": total_cls,
        "multi_lane_rows": total_multi,
        "lane_index_counts": dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
        "rightmost_lane_share": round(rm_num / total_multi, 4) if total_multi else None,
        "leftmost_lane_share": round(lm_num / total_multi, 4) if total_multi else None,
        "norm_lane_pos_mean": round(norm_sum / total_multi, 4) if total_multi else None,
        "offset_m": (
            {
                "mean": round(off_sum / off_n, 3),
                "std": round(math.sqrt(max(0.0, (off_sq_sum / off_n) - (off_sum / off_n) ** 2)), 3),
            }
            if off_n
            else None
        ),
    }


def analyze_rows(
    rows: list[Mapping[str, Any]],
    *,
    collect: Mapping[str, Any] | None = None,
    impacts: Iterable[Mapping[str, Any]] = (),
    world_signals: Iterable[tuple[str, float, float, float | None]] = (),
    has_signal_files: bool = False,
    signs_by_i: Mapping[int, Mapping[str, Any]] | None = None,
    stoplines_by_i: Mapping[int, Mapping[str, Any]] | None = None,
    gt_by_i: Mapping[int, Mapping[str, Any]] | None = None,
    intent_by_i: Mapping[int, Mapping[str, Any]] | None = None,
    roads: list[QualityRoad] | None = None,
    settle_s: float = TELEPORT_MASK_S,
    warmup_s: float = MAP_LOAD_WARMUP_S,
    signal_radius_m: float = SIGNAL_MASK_RADIUS_M,
) -> dict[str, Any]:
    """Compute per-row driving-target mask reasons and road/highway metadata."""
    n = len(rows)
    reasons_per_row: list[set[str]] = [set() for _ in range(n)]
    if n == 0:
        return {
            "rows": [],
            "reason_counts": {k: 0 for k in REASON_ORDER},
            "masked_rows": 0,
            "usable_train_rows": 0,
            "usable_train_s": 0.0,
            "span_s": 0.0,
            "highway_lane_spread": _compute_highway_spread([], []),
        }

    ts = [_num(r.get("t")) for r in rows]
    first_t = next((t for t in ts if t is not None), 0.0)
    valid_ts = [t for t in ts if t is not None]
    span_s = (max(valid_ts) - min(valid_ts)) if len(valid_ts) >= 2 else 0.0

    # 1. Post-teleport settle_5s windows + settle/recover flags
    settle_intervals: list[tuple[float, float]] = []
    if collect and isinstance(collect.get("settle_windows"), list):
        for sw in collect["settle_windows"]:
            if not isinstance(sw, Mapping):
                continue
            t0 = _num(sw.get("t0"))
            i0_raw = sw.get("i0")
            i0_idx = None
            if isinstance(i0_raw, int) and not isinstance(i0_raw, bool):
                for j, row in enumerate(rows):
                    rid = row.get("i")
                    if isinstance(rid, int) and not isinstance(rid, bool) and rid == i0_raw:
                        i0_idx = j
                        break
            if t0 is None and i0_idx is not None:
                t0 = ts[i0_idx]
            if t0 is None:
                continue
            # Anchor at first recorded frame inside/starting the window so initial spawn
            # wait that happened before first state row still masks a full settle_s of frames.
            # When i0 is None (SettleWindow.start before frame 0) and t0 is earlier than
            # the first recorded row, anchor to first_t so a full settle_s of frames is masked.
            anchor = t0
            if i0_idx is not None and ts[i0_idx] is not None and ts[i0_idx] > t0:
                anchor = ts[i0_idx]
            elif i0_raw is None and first_t is not None and t0 < first_t:
                anchor = first_t
            settle_intervals.append((min(t0, anchor), anchor + float(settle_s)))
    # Also detect settle_t0 / settle=True runs / segment or sim_t resets directly on rows
    in_settle_run = False
    prev_seg = rows[0].get("segment")
    prev_sim = _num(rows[0].get("sim_t"))
    for idx, row in enumerate(rows):
        t = ts[idx]
        seg = row.get("segment")
        sim = _num(row.get("sim_t"))
        sw_row = row.get("settle_window")
        if isinstance(sw_row, Mapping):
            st0 = _num(sw_row.get("t0"))
            if st0 is not None:
                anchor = max(st0, t) if (idx == 0 and t is not None) else st0
                settle_intervals.append((min(st0, anchor), anchor + float(settle_s)))
        st0_row = _num(row.get("settle_t0"))
        if st0_row is not None and t is not None:
            anchor = max(st0_row, t) if idx == 0 else st0_row
            settle_intervals.append((min(st0_row, anchor), anchor + float(settle_s)))
        is_settle = bool(row.get("settle")) or bool(row.get("recover"))
        if is_settle and not in_settle_run and t is not None:
            settle_intervals.append((t, t + float(settle_s)))
        in_settle_run = is_settle
        if idx > 0 and t is not None:
            if (seg is not None and prev_seg is not None and seg != prev_seg) or (
                sim is not None and prev_sim is not None and sim < prev_sim - 0.5
            ):
                settle_intervals.append((t, t + float(settle_s)))
        if seg is not None:
            prev_seg = seg
        if sim is not None:
            prev_sim = sim

    # 2. Map-load warmup 20 s
    map_load_times: list[float] = [first_t]
    prev_map = rows[0].get("map")
    for idx in range(1, n):
        t = ts[idx]
        cur_map = rows[idx].get("map")
        if t is not None and cur_map and prev_map and cur_map != prev_map:
            map_load_times.append(t)
        if cur_map:
            prev_map = cur_map
    if collect and isinstance(collect.get("settle_windows"), list):
        for sw in collect["settle_windows"]:
            if isinstance(sw, Mapping) and str(sw.get("source") or "") == "level_load":
                t0 = _num(sw.get("t0"))
                if t0 is not None:
                    map_load_times.append(t0)

    # 3. Impacts + liveness bad_row_reasons
    ev_flags, _ev_counts = event_mask(list(rows), list(impacts))
    bad_liveness = bad_row_reasons(rows)
    prev_dmg: float | None = None

    # 4. Road grid for wrong_way / over_line_corner / highway spread
    grid = build_road_grid(roads) if roads else None
    road_by_id = {r.road_id: r for r in roads} if roads else {}
    road_info: list[dict[str, Any] | None] = [None] * n
    sig_pts = list(world_signals)
    signs_map = signs_by_i or {}
    stoplines_map = stoplines_by_i or {}
    gt_map = gt_by_i or {}
    intent_map = intent_by_i or {}

    poses: list[tuple[float, float, float, float | None] | None] = [_row_pose(r) for r in rows]
    xyzs: list[tuple[float, float, float | None] | None] = [_row_xyz(r) for r in rows]

    # Pre-pass: settle_5s, map_warmup, impact, near_signal, wrong_way, road_info
    for idx, row in enumerate(rows):
        t = ts[idx]
        rid = row.get("i")
        rid_int = rid if isinstance(rid, int) and not isinstance(rid, bool) else None

        if bool(row.get("settle")) or bool(row.get("recover")):
            reasons_per_row[idx].add("settle_5s")
        elif float(settle_s) > 0.0 and t is not None:
            for a, b in settle_intervals:
                if a - 1e-6 <= t < b - 1e-6:
                    reasons_per_row[idx].add("settle_5s")
                    break

        if float(warmup_s) > 0.0 and t is not None:
            for mt in map_load_times:
                if mt - 1e-6 <= t < mt + float(warmup_s) - 1e-6:
                    reasons_per_row[idx].add("map_warmup")
                    break

        ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
        dmg = _num(ego.get("damage")) if isinstance(ego, Mapping) else None
        dmg_spike = prev_dmg is not None and dmg is not None and (dmg - prev_dmg) > 0.05
        if dmg is not None:
            prev_dmg = dmg
        live = bad_liveness.get(rid_int) if rid_int is not None else None
        if ev_flags[idx] or bool(row.get("impact")) or dmg_spike or live == "damage":
            # event_mask includes settle/stall too; only tag "impact" on real impact/damage
            impact_near = False
            if t is not None:
                for imp in impacts:
                    if not isinstance(imp, Mapping):
                        continue
                    t_imp = _num(imp.get("t"))
                    if t_imp is not None and abs(t - t_imp) <= 5.0:
                        impact_near = True
                        break
            if bool(row.get("impact")) or dmg_spike or live == "damage" or impact_near:
                reasons_per_row[idx].add("impact")

        # Near stoplights / stop signs / yield signs
        xyz = xyzs[idx]
        if xyz is not None and sig_pts:
            ex, ey, ez = xyz
            r2 = float(signal_radius_m) ** 2
            for _cls, sx, sy, sz in sig_pts:
                if ez is not None and sz is not None and abs(ez - sz) > SIGNAL_MAX_DZ_M:
                    continue
                if (ex - sx) ** 2 + (ey - sy) ** 2 <= r2:
                    reasons_per_row[idx].add("near_signal")
                    break

        if "near_signal" not in reasons_per_row[idx]:
            sign_src = []
            if rid_int is not None and rid_int in signs_map:
                sign_src.extend(signs_map[rid_int].get("signs") or [])
            tech = row.get("tech") if isinstance(row.get("tech"), Mapping) else {}
            if isinstance(tech, Mapping) and isinstance(tech.get("signs"), list):
                sign_src.extend(tech.get("signs") or [])
            if isinstance(row.get("signs"), list):
                sign_src.extend(row.get("signs") or [])
            for sg in sign_src:
                if not isinstance(sg, Mapping):
                    continue
                cls = _norm_cls(sg.get("cls") or sg.get("kind") or sg.get("type"))
                if cls not in MASKED_SIGNAL_CLASSES:
                    continue
                xm, ym = _num(sg.get("x_m")), _num(sg.get("y_m"))
                dm = _num(sg.get("d_m"))
                dist = math.hypot(xm, ym) if (xm is not None and ym is not None) else dm
                if dist is not None and dist <= float(signal_radius_m):
                    reasons_per_row[idx].add("near_signal")
                    break

        if "near_signal" not in reasons_per_row[idx]:
            sl_row = stoplines_map.get(rid_int) if rid_int is not None else None
            y_sl: float | None = None
            sl_valid = False
            bars = (sl_row or {}).get("bars") or row.get("bars") or (sl_row or {}).get("stop_lines") or row.get("stop_lines")
            if isinstance(bars, list):
                vals, msk = stop_line_target(bars)
                if float(msk[0]) > 0.5:
                    y_sl = float(vals[0])
                    sl_valid = True
            if not sl_valid:
                sl = (sl_row or {}).get("stop_line_target") or row.get("stop_line_target") or row.get("stop_line")
                if isinstance(sl, Mapping) and sl.get("valid") is True:
                    y_sl = _num(sl.get("y_m"))
                    sl_valid = y_sl is not None
                elif isinstance(sl, (list, tuple, np.ndarray)) and len(sl) >= 2:
                    sl_arr = np.asarray(sl, dtype=np.float64).reshape(-1)
                    sl_m = row.get("stop_line_valid")
                    m_ok = True if sl_m is None else bool(np.asarray(sl_m, dtype=np.float64).reshape(-1)[0] > 0.5)
                    if m_ok and float(sl_arr[1]) > 0.5 and math.isfinite(float(sl_arr[0])):
                        y_sl = float(sl_arr[0])
                        sl_valid = True
            if sl_valid and y_sl is not None and STOPLINE_MIN_Y_M <= y_sl <= STOPLINE_MAX_Y_M:
                reasons_per_row[idx].add("near_signal")

        if "near_signal" not in reasons_per_row[idx] and not has_signal_files and not signs_map:
            intent_row = intent_map.get(rid_int) if rid_int is not None else None
            if intent_row:
                jct_m = _num(intent_row.get("next_jct_m"))
                if jct_m is not None and jct_m <= float(signal_radius_m):
                    reasons_per_row[idx].add("near_signal")

        # Road-network projection
        pose = poses[idx]
        if roads and grid is not None and pose is not None:
            info = classify_wrong_way(pose[0], pose[1], pose[2], roads, grid, pose[3])
            if info.get("road_id") is not None and info["road_id"] in road_by_id:
                info = dict(info)
                info["road"] = road_by_id[info["road_id"]]
            road_info[idx] = info
            if info.get("verdict") == "wrong_way":
                reasons_per_row[idx].add("wrong_way")

    # 5. Stuck (stall flag, bad_row_reasons, or throttle while stationary outside settle/signal)
    stuck_throttle_run: list[int] = []
    for idx, row in enumerate(rows):
        rid = row.get("i")
        rid_int = rid if isinstance(rid, int) and not isinstance(rid, bool) else None
        live = bad_liveness.get(rid_int) if rid_int is not None else None
        if bool(row.get("stall")) or live == "tipped" or (
            live == "stationary"
            and "near_signal" not in reasons_per_row[idx]
            and "settle_5s" not in reasons_per_row[idx]
        ):
            reasons_per_row[idx].add("stuck")
            stuck_throttle_run.clear()
            continue
        ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
        veh = row.get("veh") if isinstance(row.get("veh"), Mapping) else {}
        pedals = row.get("pedals") if isinstance(row.get("pedals"), Mapping) else {}
        sp = _num(ego.get("speed_mps"))
        thr = _num(pedals.get("throttle_ai")) if pedals.get("throttle_ai") is not None else _num(veh.get("throttle_ai") or pedals.get("throttle"))
        brk = _num(pedals.get("brake_ai")) if pedals.get("brake_ai") is not None else _num(veh.get("brake_ai") or pedals.get("brake"))
        if (
            sp is not None
            and thr is not None
            and brk is not None
            and sp < STALL_SPEED_MPS
            and thr >= STUCK_MIN_THROTTLE
            and brk <= STUCK_MAX_BRAKE
            and "near_signal" not in reasons_per_row[idx]
            and "settle_5s" not in reasons_per_row[idx]
        ):
            stuck_throttle_run.append(idx)
        else:
            if len(stuck_throttle_run) >= 2:
                t_a, t_b = ts[stuck_throttle_run[0]], ts[stuck_throttle_run[-1]]
                if t_a is not None and t_b is not None and (t_b - t_a) >= STUCK_THROTTLE_S - 1e-6:
                    for k in stuck_throttle_run:
                        reasons_per_row[k].add("stuck")
            stuck_throttle_run.clear()
    if len(stuck_throttle_run) >= 2:
        t_a, t_b = ts[stuck_throttle_run[0]], ts[stuck_throttle_run[-1]]
        if t_a is not None and t_b is not None and (t_b - t_a) >= STUCK_THROTTLE_S - 1e-6:
            for k in stuck_throttle_run:
                reasons_per_row[k].add("stuck")

    # 6. Abrupt turn & Over-the-line in tight corners
    for idx, row in enumerate(rows):
        ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
        veh = row.get("veh") if isinstance(row.get("veh"), Mapping) else {}
        wheel = row.get("wheel") if isinstance(row.get("wheel"), Mapping) else {}
        sp = _num(ego.get("speed_mps"))
        yr = _num(ego.get("yaw_rate"))
        st = _num(wheel.get("steer_ai")) if wheel.get("steer_ai") is not None else _num(veh.get("steer_ai") or wheel.get("steer"))
        t = ts[idx]
        abrupt = False
        if yr is not None and abs(yr) > ABRUPT_YAW_RATE_RAD_S:
            abrupt = True
        if yr is not None and sp is not None and abs(yr * sp) > ABRUPT_LAT_ACC_MPS2:
            abrupt = True
        if idx > 0 and t is not None and ts[idx - 1] is not None:
            dt = t - ts[idx - 1]
            if 0.02 <= dt <= 0.5 and sp is not None and sp >= 2.0:
                prev_veh = rows[idx - 1].get("veh") if isinstance(rows[idx - 1].get("veh"), Mapping) else {}
                prev_wheel = rows[idx - 1].get("wheel") if isinstance(rows[idx - 1].get("wheel"), Mapping) else {}
                prev_st = _num(prev_wheel.get("steer_ai")) if prev_wheel.get("steer_ai") is not None else _num(prev_veh.get("steer_ai") or prev_wheel.get("steer"))
                if st is not None and prev_st is not None and abs(st - prev_st) / dt > ABRUPT_STEER_RATE_S:
                    abrupt = True
                h0 = poses[idx - 1][2] if poses[idx - 1] is not None else None
                h1 = poses[idx][2] if poses[idx] is not None else None
                if h0 is not None and h1 is not None and abs(_wrap_angle(h1 - h0)) / dt > ABRUPT_YAW_RATE_RAD_S:
                    abrupt = True
        if abrupt:
            reasons_per_row[idx].add("abrupt_turn")

        # Over-the-line in tight corners
        curv = abs(yr) / sp if (yr is not None and sp is not None and sp >= 2.0) else 0.0
        if curv >= CORNER_MIN_CURV_PER_M:
            rid = row.get("i")
            rid_int = rid if isinstance(rid, int) and not isinstance(rid, bool) else None
            gt_row = gt_map.get(rid_int) if rid_int is not None else None
            over_line = False
            if gt_row and gt_row.get("off_road") is True:
                over_line = True
            tech = (gt_row or {}).get("tech") if isinstance((gt_row or {}).get("tech"), Mapping) else row.get("tech")
            if isinstance(tech, Mapping):
                lgt = tech.get("lane_gt")
                lgt_v = tech.get("lane_gt_valid")
                if isinstance(lgt, list) and isinstance(lgt_v, list) and len(lgt) >= 3 and len(lgt_v) >= 3:
                    il, ir = lookahead_slot(0, SLOT_LEFT), lookahead_slot(0, SLOT_RIGHT)
                    vl, vr = _num(lgt_v[il]), _num(lgt_v[ir])
                    xl, xr = _num(lgt[il]), _num(lgt[ir])
                    if (vl or 0.0) > 0.5 and xl is not None and xl > -OVER_LINE_MARGIN_M:
                        over_line = True
                    if (vr or 0.0) > 0.5 and xr is not None and xr < OVER_LINE_MARGIN_M:
                        over_line = True
            info = road_info[idx]
            if info is not None and info.get("road") is not None:
                road_obj: QualityRoad = info["road"]
                off_m = _num(info.get("offset_m"))
                align = _num(info.get("align"))
                if info.get("verdict") == "wrong_way":
                    over_line = True
                elif off_m is not None:
                    half_w = road_obj.width_m * 0.5
                    if abs(off_m) > half_w - OVER_LINE_MARGIN_M:
                        over_line = True
                    elif not bool(info.get("effective_oneway")) and align is not None:
                        # Two-way centerline crossing or wrong-side cut
                        if abs(off_m) < OVER_LINE_MARGIN_M or (align > 0.3 and off_m < 0.0) or (align < -0.3 and off_m > 0.0):
                            over_line = True
            if over_line:
                reasons_per_row[idx].add("over_line_corner")

    # 7. U-turns (intent sidecar OR tight 150+ deg heading reversal within <=15 s & <=18 m net displacement)
    for idx, row in enumerate(rows):
        rid = row.get("i")
        rid_int = rid if isinstance(rid, int) and not isinstance(rid, bool) else None
        intent_row = intent_map.get(rid_int) if rid_int is not None else None
        if intent_row and str(intent_row.get("reason") or "").strip().lower() == "uturn":
            reasons_per_row[idx].add("uturn")

    start = 0
    while start < n:
        end = start + 1
        seg = rows[start].get("segment")
        while end < n and rows[end].get("segment") == seg:
            end += 1
        # Accumulate wrapped yaw deltas across segment [start, end)
        cum_yaw = [0.0] * (end - start)
        for k in range(1, end - start):
            p0 = poses[start + k - 1]
            p1 = poses[start + k]
            dy = _wrap_angle(p1[2] - p0[2]) if (p0 is not None and p1 is not None) else 0.0
            cum_yaw[k] = cum_yaw[k - 1] + dy
        j = 0
        for i_rel in range(end - start):
            t_i = ts[start + i_rel]
            p_i = poses[start + i_rel]
            if t_i is None or p_i is None:
                continue
            while j < i_rel:
                t_j = ts[start + j]
                if t_j is None or (t_i - t_j) > UTURN_WINDOW_S:
                    j += 1
                else:
                    break
            for a_rel in range(j, i_rel):
                t_a = ts[start + a_rel]
                p_a = poses[start + a_rel]
                if t_a is None or p_a is None:
                    continue
                d_yaw_deg = math.degrees(abs(cum_yaw[i_rel] - cum_yaw[a_rel]))
                if d_yaw_deg >= UTURN_MIN_DEG:
                    net_disp = math.hypot(p_i[0] - p_a[0], p_i[1] - p_a[1])
                    if net_disp <= UTURN_MAX_NET_DISP_M:
                        for k in range(start + a_rel, start + i_rel + 1):
                            reasons_per_row[k].add("uturn")
                        break
        start = end

    # Build summary
    reason_counts = {name: 0 for name in REASON_ORDER}
    out_rows: list[dict[str, Any]] = []
    masked_rows = 0
    usable_ts: list[float] = []
    for idx, row in enumerate(rows):
        ordered = [r for r in REASON_ORDER if r in reasons_per_row[idx]]
        for r in ordered:
            reason_counts[r] += 1
        masked = bool(ordered)
        if masked:
            masked_rows += 1
        elif ts[idx] is not None:
            usable_ts.append(float(ts[idx]))
        rid = row.get("i")
        out_rows.append(
            {
                "i": rid if rid is not None else idx,
                "t": ts[idx],
                "masked": masked,
                "reasons": ordered,
            }
        )

    dt_est = span_s / max(1, n - 1) if n >= 2 else 0.0
    usable_train_rows = n - masked_rows
    usable_train_s = float(usable_train_rows) * dt_est if n >= 2 else 0.0

    return {
        "rows": out_rows,
        "reason_counts": reason_counts,
        "masked_rows": int(masked_rows),
        "usable_train_rows": int(usable_train_rows),
        "usable_train_s": round(usable_train_s, 3),
        "span_s": round(span_s, 3),
        "highway_lane_spread": _compute_highway_spread(rows, road_info),
    }


def _check_raw_images(clip_dir: Path) -> tuple[bool, list[str]]:
    """Verify no lossy JPEGs and sample PNG readability if ``raw/`` exists."""
    import cv2

    raw = clip_dir / "raw"
    if not raw.is_dir():
        return True, []
    reasons: list[str] = []
    lossy = [p for p in raw.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")]
    if lossy:
        reasons.append("lossy_frames")
    pngs = sorted(raw.rglob("*.png"))
    if not pngs:
        reasons.append("missing_cameras")
        return False, reasons
    main_dir = raw / "main"
    sample_pool = sorted(main_dir.rglob("*.png")) if main_dir.is_dir() else pngs
    for f in sample_pool[:2] + sample_pool[-2:]:
        if cv2.imread(str(f), cv2.IMREAD_UNCHANGED) is None:
            reasons.append("undecodable_frames")
            break
    return not reasons, reasons


def classify_clip(
    clip_dir: Path,
    analysis: Mapping[str, Any],
    raw_rows: list[Mapping[str, Any]],
    *,
    tags: Mapping[str, Any] | None = None,
    collect: Mapping[str, Any] | None = None,
    check_images: bool = True,
) -> tuple[str, list[str]]:
    """Classify a clip into ``train``, ``pretrain_only``, or ``reject`` + reason codes."""
    tags = tags or {}
    collect = collect or {}
    n = len(raw_rows)
    if n == 0:
        return "reject", ["no_state_rows"]

    reject_reasons: list[str] = []
    pretrain_reasons: list[str] = []

    span_s = float(analysis.get("span_s") or 0.0)
    if span_s < MIN_CLIP_SPAN_S:
        reject_reasons.append("span_too_short")

    if check_images:
        _ok_img, img_reasons = _check_raw_images(clip_dir)
        reject_reasons.extend(img_reasons)

    rc = analysis.get("reason_counts") or {}
    stuck_share = float(rc.get("stuck") or 0) / float(n)
    wrong_way_share = float(rc.get("wrong_way") or 0) / float(n)
    uturn_share = float(rc.get("uturn") or 0) / float(n)

    speeds = sorted(
        s for s in (_num((r.get("ego") or {}).get("speed_mps")) for r in raw_rows) if s is not None
    )
    median_speed = speeds[len(speeds) // 2] if speeds else 0.0
    if not speeds or (stuck_share >= 0.80 and median_speed < STALL_SPEED_MPS):
        reject_reasons.append("all_stuck_or_empty")

    exit_code = tags.get("collector_exit")
    if isinstance(exit_code, int) and exit_code != 0:
        if span_s < MIN_CLIP_SPAN_S:
            reject_reasons.append("collector_failed_empty")
        else:
            pretrain_reasons.append("collector_nonzero_exit")

    if reject_reasons:
        return "reject", list(dict.fromkeys(reject_reasons))

    mains = sum(1 for r in raw_rows if "main" in (r.get("cams") or {}))
    # Only enforce main_hz when cams paths are recorded in state.jsonl
    if any(r.get("cams") for r in raw_rows):
        hz = mains / span_s if span_s > 0 else 0.0
        if hz < MIN_MAIN_HZ:
            pretrain_reasons.append("low_main_hz")

    planned_s = _num(tags.get("duration_s")) or _num(collect.get("duration_s"))
    if planned_s is not None and span_s < 0.5 * planned_s:
        pretrain_reasons.append("short_span")

    if median_speed < MIN_MOVING_MPS:
        pretrain_reasons.append("low_median_speed")

    if int(rc.get("impact") or 0) > 0 or any(
        (_num((r.get("ego") or {}).get("damage")) or 0.0) > 0.0 for r in raw_rows
    ):
        pretrain_reasons.append("ego_damaged_or_impact")

    if tags.get("edge_case_observed") is False:
        pretrain_reasons.append("edge_case_not_observed")

    if stuck_share > MAX_TRAIN_STUCK_SHARE:
        pretrain_reasons.append("excessive_stuck")
    if wrong_way_share > MAX_TRAIN_WRONG_WAY_SHARE:
        pretrain_reasons.append("excessive_wrong_way")
    if uturn_share > MAX_TRAIN_UTURN_SHARE:
        pretrain_reasons.append("excessive_uturn")

    usable_rows = int(analysis.get("usable_train_rows") or 0)
    usable_s = float(analysis.get("usable_train_s") or 0.0)
    if usable_rows < MIN_TRAIN_ROWS or usable_s < MIN_TRAIN_SPAN_S:
        pretrain_reasons.append("insufficient_clean_driving")

    if pretrain_reasons:
        return "pretrain_only", list(dict.fromkeys(pretrain_reasons))
    return "train", []


def analyze_clip(
    clip_dir: Path | str,
    *,
    road_dump: Path | str | None = None,
    roads: list[QualityRoad] | None = None,
    check_images: bool = True,
    settle_s: float = TELEPORT_MASK_S,
    warmup_s: float = MAP_LOAD_WARMUP_S,
    signal_radius_m: float = SIGNAL_MASK_RADIUS_M,
) -> dict[str, Any]:
    """Run full read-only offline QC on one Recorder v2 clip directory."""
    clip = Path(clip_dir)
    rows = _read_jsonl(clip / "state.jsonl")
    collect = _read_json(clip / "collect.json")
    tags = _read_json(clip / "clip_tags.json")
    impacts = read_impacts(clip)
    world_sigs, has_sig_files = _extract_world_signals(clip)
    signs_by_i, has_signs_file = _load_sidecar_by_row(clip, "signs.jsonl")
    stoplines_by_i, _ = _load_sidecar_by_row(clip, "stoplines.jsonl")
    gt_by_i, _ = _load_sidecar_by_row(clip, "gt.jsonl")
    intent_by_i, _ = _load_sidecar_by_row(clip, "intent.jsonl")

    loaded_roads = roads
    if loaded_roads is None:
        rd_path = _resolve_road_dump(clip, rows, road_dump)
        if rd_path is not None:
            loaded_roads = load_quality_roads(rd_path)

    analysis = analyze_rows(
        rows,
        collect=collect,
        impacts=impacts,
        world_signals=world_sigs,
        has_signal_files=(has_sig_files or has_signs_file),
        signs_by_i=signs_by_i,
        stoplines_by_i=stoplines_by_i,
        gt_by_i=gt_by_i,
        intent_by_i=intent_by_i,
        roads=loaded_roads,
        settle_s=settle_s,
        warmup_s=warmup_s,
        signal_radius_m=signal_radius_m,
    )
    clip_class, reason_codes = classify_clip(
        clip,
        analysis,
        rows,
        tags=tags,
        collect=collect,
        check_images=check_images,
    )
    return {
        "kind": "gvd_clip_qc",
        "clip": clip.name,
        "clip_class": clip_class,
        "reason_codes": reason_codes,
        "total_rows": len(rows),
        "masked_rows": analysis["masked_rows"],
        "usable_train_rows": analysis["usable_train_rows"],
        "usable_train_s": analysis["usable_train_s"],
        "span_s": analysis["span_s"],
        "reason_counts": analysis["reason_counts"],
        "highway_lane_spread": analysis["highway_lane_spread"],
        "excluded_target_fields": list(EXCLUDED_TARGET_FIELDS),
        "rows": analysis["rows"],
    }


def write_clip_qc(clip_dir: Path | str, result: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write ``clip_qc.json`` (summary) and ``clip_qc_mask.jsonl`` (for ``--mask-files``)."""
    clip = Path(clip_dir)
    summary = {k: v for k, v in result.items() if k != "rows"}
    qc_path = clip / "clip_qc.json"
    qc_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    mask_path = clip / "clip_qc_mask.jsonl"
    lines = [
        json.dumps({"i": r["i"], "reasons": r["reasons"]}) + "\n"
        for r in (result.get("rows") or [])
        if r.get("masked")
    ]
    mask_path.write_text("".join(lines), encoding="utf-8")
    return qc_path, mask_path


def apply_clip_qc_masks(
    state_rows: list[dict[str, Any]],
    kept_rows: list[dict[str, Any]],
    *,
    root: Path | str | None = None,
    road_dump: Path | str | None = None,
    settle_s: float = TELEPORT_MASK_S,
    warmup_s: float = MAP_LOAD_WARMUP_S,
    signal_radius_m: float = SIGNAL_MASK_RADIUS_M,
) -> dict[str, Any]:
    """Mark driving-target mask flags in-place on ``state_rows`` and ``kept_rows``."""
    full = list(state_rows)
    if not full:
        return {"total_rows": 0, "masked_rows": 0, "reason_counts": {k: 0 for k in REASON_ORDER}}

    # Group by source_root when multi-strip training passes combined rows
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in full:
        key = str(r.get("source_root") or (Path(root) if root is not None else ""))
        groups.setdefault(key, []).append(r)

    masked_keys: dict[tuple[str, int], list[str]] = {}
    agg_counts = {k: 0 for k in REASON_ORDER}
    total_masked = 0
    roads_cache: dict[str, list[QualityRoad] | None] = {}

    for src_key, sub in groups.items():
        src_dir = Path(src_key) if src_key else (Path(root) if root is not None else None)
        collect = _read_json(src_dir / "collect.json") if src_dir else {}
        impacts = read_impacts(src_dir) if src_dir else []
        world_sigs, has_sig = _extract_world_signals(src_dir) if src_dir else ([], False)
        signs_i, has_signs = _load_sidecar_by_row(src_dir, "signs.jsonl") if src_dir else ({}, False)
        stops_i, _ = _load_sidecar_by_row(src_dir, "stoplines.jsonl") if src_dir else ({}, False)
        gt_i, _ = _load_sidecar_by_row(src_dir, "gt.jsonl") if src_dir else ({}, False)
        intent_i, _ = _load_sidecar_by_row(src_dir, "intent.jsonl") if src_dir else ({}, False)
        loaded_roads = None
        if src_dir is not None:
            rd = _resolve_road_dump(src_dir, sub, road_dump)
            if rd is not None:
                rd_s = str(rd)
                if rd_s not in roads_cache:
                    roads_cache[rd_s] = load_quality_roads(rd)
                loaded_roads = roads_cache[rd_s]
        elif road_dump is not None:
            rd_s = str(road_dump)
            if rd_s not in roads_cache and Path(rd_s).is_file():
                roads_cache[rd_s] = load_quality_roads(rd_s)
            loaded_roads = roads_cache.get(rd_s)

        analysis = analyze_rows(
            sub,
            collect=collect,
            impacts=impacts,
            world_signals=world_sigs,
            has_signal_files=(has_sig or has_signs),
            signs_by_i=signs_i,
            stoplines_by_i=stops_i,
            gt_by_i=gt_i,
            intent_by_i=intent_i,
            roads=loaded_roads,
            settle_s=settle_s,
            warmup_s=warmup_s,
            signal_radius_m=signal_radius_m,
        )
        for name, cnt in analysis["reason_counts"].items():
            agg_counts[name] = agg_counts.get(name, 0) + int(cnt)
        total_masked += int(analysis["masked_rows"])
        for row_obj, item in zip(sub, analysis["rows"]):
            if item["masked"]:
                row_obj["clip_qc_masked"] = True
                row_obj["driving_target_masked"] = True
                row_obj["clip_qc_reasons"] = list(item["reasons"])
                rid = row_obj.get("i")
                if isinstance(rid, int) and not isinstance(rid, bool):
                    masked_keys[(src_key, rid)] = list(item["reasons"])

    for row_obj in kept_rows:
        key = (
            str(row_obj.get("source_root") or (Path(root) if root is not None else "")),
            row_obj.get("i"),
        )
        if key in masked_keys:
            row_obj["clip_qc_masked"] = True
            row_obj["driving_target_masked"] = True
            row_obj["clip_qc_reasons"] = list(masked_keys[key])

    return {
        "total_rows": len(full),
        "masked_rows": int(total_masked),
        "reason_counts": agg_counts,
        "masked_ids": {rid for _src, rid in masked_keys},
    }
