#!/usr/bin/env python3
"""Offline checks for scripts/check_traffic_box.py (synthetic state.jsonl)."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_traffic_box as ctb  # noqa: E402


# half[i] pairs with AX[i] = [right, dir, up]: width, length, height.
HALF = [1.0, 2.2, 0.7]
AX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# A small, stable origin-to-centre offset (live box, not spawn).
CTR_OFFSET = (0.3, -0.4, 0.5)
# Enough points that a 5 m step yields >= MIN_MOVING_BOXED samples past MOVE_M.
LIVE_N = 14


def _box(
    ctr: tuple[float, float, float],
    ax: list | None = None,
    half: list | None = None,
) -> dict:
    return {
        "ctr": [ctr[0], ctr[1], ctr[2]],
        "half": list(half if half is not None else HALF),
        "ax": [list(a) for a in (ax or AX)],
    }


def _live_ctr(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    return (pos[0] + CTR_OFFSET[0], pos[1] + CTR_OFFSET[1], pos[2] + CTR_OFFSET[2])


def _orient(yaw: float) -> tuple[tuple[float, float, float], tuple[float, float, float], list]:
    """Yaw 0 is +Y (dir [0,1,0], axes = identity)."""
    direction = (math.sin(yaw), math.cos(yaw), 0.0)
    up = (0.0, 0.0, 1.0)
    right = (direction[1], -direction[0], 0.0)
    return direction, up, [list(right), list(direction), list(up)]


def _car(
    name: str,
    pos: tuple[float, float, float],
    box: dict | None,
    *,
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict:
    return {
        "name": name,
        "pos": [pos[0], pos[1], pos[2]],
        "vel": [0.0, 8.0, 0.0],
        "dir": [direction[0], direction[1], direction[2]],
        "up": [up[0], up[1], up[2]],
        "box": box,
    }


def _write(root: Path, cars_per_tick: list, extras: list | None = None) -> Path:
    """Write state.jsonl. A tick may be traffic=null (the 2 Hz gaps)."""
    path = root / "state.jsonl"
    lines = []
    for i, cars in enumerate(cars_per_tick):
        row: dict = {"t": float(i) * 0.1, "i": i}
        row["traffic"] = None if cars is None else cars
        if extras and i < len(extras) and extras[i]:
            row.update(extras[i])
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_rows(root: Path, rows: list[dict]) -> Path:
    path = root / "state.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _path(
    n: int = LIVE_N,
    start: tuple[float, float, float] = (0.0, 0.0, 0.5),
    step: float = 5.0,
) -> list[tuple[float, float, float]]:
    return [(start[0], start[1] + i * step, start[2]) for i in range(n)]


def _turning_companion(i: int, name: str = "gvd_traffic_axes") -> dict:
    """Live boxed sample i of a car that turns 30° after 20 m.

    Within nine samples this yields >= MIN_MOVING_BOXED moving boxed
    points whose yaw span is 30°, so a straight-only companion can
    still let a LIVE test exit 0 under the axes-proven gate.
    """
    start = (200.0, 0.0, 0.5)
    step = 5.0
    turn_from = 4
    x, y, z = start
    for k in range(i):
        yw = 0.0 if k < turn_from else math.radians(30.0)
        x += step * math.sin(yw)
        y += step * math.cos(yw)
    yaw = 0.0 if i < turn_from else math.radians(30.0)
    pos = (x, y, z)
    direction, up, ax = _orient(yaw)
    return _car(name, pos, _box(_live_ctr(pos), ax=ax), direction=direction, up=up)


def _add_axes_companion(ticks: list) -> list:
    present = 0
    out = []
    for cars in ticks:
        if cars is None:
            out.append(None)
            continue
        out.append(list(cars) + [_turning_companion(present)])
        present += 1
    return out


def _add_axes_companion_rows(rows: list[dict]) -> list[dict]:
    present = 0
    out = []
    for row in rows:
        r = dict(row)
        traffic = r.get("traffic")
        if isinstance(traffic, list):
            r["traffic"] = list(traffic) + [_turning_companion(present)]
            present += 1
        out.append(r)
    return out


def _straight_then_turn_poses(
    straight_m: float = 40.0,
    step: float = 5.0,
    turn_deg: float = 30.0,
    turn_n: int = 8,
    start: tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> list[tuple[tuple[float, float, float], float]]:
    poses: list[tuple[tuple[float, float, float], float]] = []
    x, y, z = start
    yaw = 0.0
    dist = 0.0
    while dist <= straight_m + 1e-9:
        poses.append(((x, y, z), yaw))
        x += step * math.sin(yaw)
        y += step * math.cos(yaw)
        dist += step
    yaw = math.radians(turn_deg)
    for _ in range(turn_n):
        x += step * math.sin(yaw)
        y += step * math.cos(yaw)
        poses.append(((x, y, z), yaw))
    return poses


def _ticks_from_poses(
    poses: list[tuple[tuple[float, float, float], float]],
    *,
    name: str = "gvd_traffic_0",
    flip_ax: bool = False,
    half: list | None = None,
    boxed: bool | list[bool] | None = None,
) -> list:
    ticks = []
    for i, (pos, yaw) in enumerate(poses):
        direction, up, ax = _orient(yaw)
        if flip_ax:
            ax = [[-c for c in a] for a in ax]
        has_box = True if boxed is None else (boxed if isinstance(boxed, bool) else boxed[i])
        box = _box(_live_ctr(pos), ax=ax, half=half) if has_box else None
        ticks.append([_car(name, pos, box, direction=direction, up=up)])
    return ticks


def _run_ticks(ticks: list, extras: list | None = None) -> tuple[int, dict, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, ticks, extras)
        rc = ctb.main([str(root)])
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        text = ctb.format_report(summary, "synthetic")
        return rc, summary, text



def _dist_from(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _assert_mixed_axes_caveat(text: str) -> None:
    """Y10: mixed PASS must name the straight car, not the turning companion."""
    assert "Axes are UNPROVEN for gvd_traffic_0" in text
    assert "Long run may proceed on this evidence; axes are unproven for gvd_traffic_0" in text
    assert "UNPROVEN for gvd_traffic_axes" not in text
    assert "unproven for gvd_traffic_axes" not in text


def check_live_box_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = _path()
        ticks: list = []
        for i, pos in enumerate(path):
            if i % 2 == 1:
                ticks.append(None)  # traffic is only on some ticks
                continue
            ticks.append([_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))])
        # Evens of LIVE_N=14 at 5 m: 0,10,...,30; only 5 points past 10 m.
        # Add enough even ticks past MOVE_M (continue the same stride).
        for j, pos in enumerate(_path(n=8, start=(0.0, 35.0, 0.5), step=5.0), start=len(path)):
            if j % 2 == 1:
                ticks.append(None)
                continue
            ticks.append([_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))])
        _write(root, _add_axes_companion(ticks))
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(ctb.resolve_state(root)))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and summary["status"] == "LIVE", summary
        assert stats["proven_live"] and not stats["spawn_stuck"] and not stats["unproven"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED, stats
        expected = (CTR_OFFSET[0] ** 2 + CTR_OFFSET[1] ** 2 + CTR_OFFSET[2] ** 2) ** 0.5
        assert stats["max_ctr_pos_m"] is not None
        assert abs(stats["max_ctr_pos_m"] - expected) < 1e-9, stats
        assert stats["moved_m"] > ctb.MOVE_M
        text = ctb.format_report(summary, "synthetic")
        assert "box.ctr follows pos" not in text
        assert "IQU45 smoke check" in text
        _assert_mixed_axes_caveat(text)
    print("live box passes")


def check_spawn_stuck_flagged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spawn_ctr = (10.0, 20.0, 0.8)
        ticks = []
        for pos in _path(start=(10.0, 20.0, 0.5)):
            ticks.append([_car("gvd_traffic_0", pos, _box(spawn_ctr))])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert not summary["ok"] and summary["status"] == "SPAWN-STUCK"
        assert summary["stuck"] == ["gvd_traffic_0"]
        assert stats["spawn_stuck"] and stats["moving"]
        assert stats["max_ctr_pos_m"] is not None and stats["max_ctr_pos_m"] > ctb.MOVE_M
        text = ctb.format_report(summary, "synthetic")
        assert "SPAWN-STUCK" in text and "Do not start the long run" in text
        assert "Keep only half" in text
        assert "|(pos-ctr)·ax_i|" in text
    print("spawn-stuck ctr flagged")


def check_wobbling_stuck_box_fails() -> None:
    """A box stuck at spawn with a ~0.3 m wobble must not pass as live."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spawn = (0.0, 0.0, 0.8)
        ticks = []
        for i, pos in enumerate(_path()):
            wobble = (0.3 * math.sin(i), 0.3 * math.cos(i), 0.0)
            ctr = (spawn[0] + wobble[0], spawn[1] + wobble[1], spawn[2] + wobble[2])
            ticks.append([_car("gvd_traffic_0", pos, _box(ctr))])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert stats["spawn_stuck"] and summary["status"] == "SPAWN-STUCK"
        # The wobble alone is well under 1 m, so a first-centre-span check
        # with a 1e-12..1 m radius would miss this; |ctr-pos| must catch it.
        assert stats["ctr_span_m"] is not None and stats["ctr_span_m"] < 1.0
        assert stats["max_ctr_pos_m"] is not None and stats["max_ctr_pos_m"] > ctb.MOVE_M
    print("wobbling stuck box fails")


def check_two_spawn_stuck_box_fails() -> None:
    """Box fixed at each spawn, same name reused, must exit 1 (Skeptic repro)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        i = 0
        for seg, (x0, y0) in enumerate([(0.0, 0.0), (500.0, 200.0)]):
            spawn_ctr = (x0, y0 + 0.3, 0.9)
            for k in range(8):
                pos = (x0, y0 + k * 4.0, 0.5)
                row = {
                    "i": i,
                    "t": i * 0.5,
                    "traffic": [_car(
                        "gvd_traffic_0",
                        pos,
                        {
                            "ctr": [spawn_ctr[0], spawn_ctr[1], spawn_ctr[2]],
                            "half": [0.95, 2.3, 0.75],
                            "ax": [list(a) for a in AX],
                        },
                    )],
                }
                if seg == 1 and k == 0:
                    row["veh_meta"] = {"model": "etk800"}
                rows.append(row)
                i += 1
        _write_rows(root, rows)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert stats["n_spawns"] == 2, stats
        assert stats["spawn_stuck"] and summary["status"] == "SPAWN-STUCK"
        assert stats["max_ctr_pos_m"] is not None and stats["max_ctr_pos_m"] > 20.0
        # Teleport distance is not counted as motion.
        assert stats["moved_m"] < 100.0, stats
    print("two-spawn stuck box fails")


def check_two_spawn_live_box_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        i = 0
        for seg, (x0, y0) in enumerate([(0.0, 0.0), (500.0, 200.0)]):
            for k in range(LIVE_N):
                pos = (x0, y0 + k * 5.0, 0.5)
                row = {
                    "i": i,
                    "t": i * 0.5,
                    "traffic": [_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))],
                }
                if seg == 1 and k == 0:
                    row["veh_meta"] = {"model": "etk800"}
                rows.append(row)
                i += 1
        _write_rows(root, _add_axes_companion_rows(rows))
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and stats["proven_live"] and stats["n_spawns"] == 2
        _assert_mixed_axes_caveat(ctb.format_report(summary, "synthetic"))
    print("two-spawn live box passes")


def check_mixed_live_and_stuck_fails() -> None:
    """One live companion does not hide a spawn-stuck one."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos0, pos1 in zip(_path(), _path(start=(100.0, 0.0, 0.5))):
            ticks.append([
                _car("gvd_traffic_0", pos0, _box(_live_ctr(pos0))),
                _car("gvd_traffic_1", pos1, _box((100.0, 0.0, 0.8))),
            ])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        assert summary["status"] == "SPAWN-STUCK"
        assert summary["stuck"] == ["gvd_traffic_1"]
        assert summary["cars"]["gvd_traffic_0"]["proven_live"]
    print("mixed live/stuck fails")


def check_parked_ignored_unproven_without_mover() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = [
            _car("gvd_traffic_0", (0.0, i * 0.4, 0.5), _box((0.0, 0.0, 0.8)))
            for i in range(6)
        ]
        _write(root, [[c] for c in ticks])
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        assert not summary["ok"] and summary["status"] == "UNPROVEN"
        assert not summary["moving"] and not summary["stuck"]
        assert summary["cars"]["gvd_traffic_0"]["moved_m"] < ctb.MOVE_M
        text = ctb.format_report(summary, "synthetic")
        assert "PARKED" in text and "unproven" in text
    print("parked only is unproven")


def check_moving_without_box_unproven() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = [[_car("gvd_traffic_0", pos, None)] for pos in _path()]
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        assert summary["status"] == "UNPROVEN"
        assert "gvd_traffic_0" in summary["unproven"]
        assert summary["cars"]["gvd_traffic_0"]["box_nonnull"] == 0.0
        assert summary["cars"]["gvd_traffic_0"]["max_ctr_pos_m"] is None
    print("moving without box is unproven")


def check_too_few_moving_boxed_is_unproven() -> None:
    """A live box with fewer than MIN_MOVING_BOXED samples must not pass."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 0,5,10,15,20,25: only 3 samples with displacement > 10 m.
        ticks = [[_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))] for pos in _path(n=6)]
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert stats["moving"] and not stats["proven_live"] and stats["unproven"]
        assert stats["n_moving_boxed"] < ctb.MIN_MOVING_BOXED
        assert summary["status"] == "UNPROVEN"
    print("too few moving boxed is unproven")


def check_ignores_non_companion_and_accepts_state_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos in _path():
            ticks.append([
                _car("gvd_traffic_0", pos, _box(_live_ctr(pos))),
                _car("parked_bus", (0.0, 0.0, 0.5), _box((0.0, 0.0, 0.8))),
            ])
        state = _write(root, _add_axes_companion(ticks))
        assert ctb.main([str(state)]) == 0
        summary = ctb.summarize(ctb.load_state(state))
        assert "parked_bus" not in summary["cars"]
        assert summary["ok"]
        _assert_mixed_axes_caveat(ctb.format_report(summary, "synthetic"))
    print("non-companion ignored; state.jsonl path ok")


def check_missing_and_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert ctb.main([str(root)]) == 2
        empty = root / "state.jsonl"
        empty.write_text("", encoding="utf-8")
        assert ctb.main([str(root)]) == 2
    print("missing/empty exit 2")


def check_pos_jump_and_veh_meta_split() -> None:
    """A teleport is not motion. Live boxes at two short stints stay unproven."""

    def _stint(x0: float, y0: float, t0: int, veh_meta: bool) -> list[dict]:
        rows = []
        for k in range(6):
            pos = (x0, y0 + k * 0.4, 0.5)
            row = {
                "i": t0 + k,
                "t": (t0 + k) * 0.5,
                "traffic": [_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))],
            }
            if veh_meta and k == 0:
                row["veh_meta"] = {"model": "etk800"}
            rows.append(row)
        return rows

    # Pos jump, no veh_meta: 2 m + 500 m teleport + 2 m must not look like motion.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = _stint(0.0, 0.0, 0, False) + _stint(500.0, 0.0, 6, False)
        _write_rows(root, rows)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert stats["n_spawns"] == 2, stats
        assert stats["moved_m"] < 5.0, stats
        assert not stats["moving"] and not stats["proven_live"]
        assert summary["status"] == "UNPROVEN"

    # veh_meta with a 20 m gap (under JUMP_M) must still split.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = _stint(0.0, 0.0, 0, False) + _stint(0.0, 20.0, 6, True)
        _write_rows(root, rows)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert stats["n_spawns"] == 2, stats
        assert stats["moved_m"] < 5.0, stats
        assert not stats["proven_live"] and summary["status"] == "UNPROVEN"
    print("pos jump and veh_meta split")


def check_turning_car_axes() -> None:
    """Live rotating axes pass; axes frozen at spawn while the car turns fail."""
    # Drive +Y, then on a 50° heading (not a 90° axis flip: a frozen
    # identity box would still have an axis along +X after a right-angle turn).
    poses = []
    y = 0.0
    while y <= 40.0 + 1e-9:
        poses.append(((0.0, y, 0.5), 0.0))
        y += 5.0
    yaw = math.radians(50.0)
    x, y = 0.0, 40.0
    for _ in range(8):
        x += 5.0 * math.sin(yaw)
        y += 5.0 * math.cos(yaw)
        poses.append(((x, y, 0.5), yaw))

    def _run(rotate_axes: bool) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ticks = []
            for pos, yaw in poses:
                direction, up, ax = _orient(yaw)
                if not rotate_axes:
                    ax = [list(a) for a in AX]
                    direction, up, _ = _orient(yaw)
                ticks.append([_car(
                    "gvd_traffic_0", pos, _box(_live_ctr(pos), ax=ax),
                    direction=direction, up=up,
                )])
            _write(root, ticks)
            rc = ctb.main([str(root)])
            summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
            summary["_rc"] = rc
            return summary

    live = _run(True)
    assert live["_rc"] == 0 and live["status"] == "LIVE", live
    assert live["cars"]["gvd_traffic_0"]["proven_live"]
    assert live["cars"]["gvd_traffic_0"]["axes_yaw_proven"]
    assert live["cars"]["gvd_traffic_0"]["yaw_span_deg"] >= ctb.MIN_YAW_SPAN_DEG

    frozen = _run(False)
    assert frozen["_rc"] == 1 and frozen["status"] == "AXES-STUCK", frozen
    stats = frozen["cars"]["gvd_traffic_0"]
    assert stats["axes_stuck"] and not stats["spawn_stuck"]
    text = ctb.format_report(frozen, "synthetic")
    assert "AXES-STUCK" in text and "dir/up" in text
    print("turning-car axes: live passes, frozen fails")



def check_rolled_box_up_misaligned_fails() -> None:
    """Car stays level; box rolled 30° about dir must exit 1 (locks the up check)."""
    roll = math.radians(30.0)
    c, s = math.cos(roll), math.sin(roll)
    # Right-handed roll about +Y (dir): right and up tilt, dir stays +Y.
    ax = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = [
            [_car("gvd_traffic_0", pos, _box(_live_ctr(pos), ax=ax))]
            for pos in _path()
        ]
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["status"] == "AXES-STUCK" and not summary["ok"], summary
        assert stats["axes_stuck"] and not stats["spawn_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
    print("rolled 30° box (level car) fails")


def check_frozen_axes_after_exact_90deg_turn_fails() -> None:
    """Frozen axes after an exact 90° turn must not pass as LIVE."""
    poses: list[tuple[tuple[float, float, float], float]] = []
    y = 0.0
    while y <= 40.0 + 1e-9:
        poses.append(((0.0, y, 0.5), 0.0))
        y += 5.0
    # Turn exactly 90° to +X; identity axes then have ax0 along the new dir.
    x, y = 0.0, 40.0
    yaw = math.pi / 2.0
    for _ in range(8):
        x += 5.0 * math.sin(yaw)
        y += 5.0 * math.cos(yaw)
        poses.append(((x, y, 0.5), yaw))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos, yaw in poses:
            direction, up, _ = _orient(yaw)
            ticks.append([_car(
                "gvd_traffic_0", pos, _box(_live_ctr(pos), ax=AX),
                direction=direction, up=up,
            )])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["status"] == "AXES-STUCK" and not summary["ok"], summary
        assert stats["axes_stuck"] and not stats["spawn_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
    print("frozen axes after exact 90° turn fail")


def check_box_length_across_car_fails() -> None:
    """A box whose long half lies across the car (not along dir) must exit 1."""
    # Swap width/length: ax0 (right) carries 2.2 m, dir only 1.0 m.
    across = [HALF[1], HALF[0], HALF[2]]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = [
            [_car("gvd_traffic_0", pos, _box(_live_ctr(pos), half=across))]
            for pos in _path()
        ]
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["status"] == "AXES-STUCK" and not summary["ok"], summary
        assert stats["axes_stuck"] and not stats["spawn_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
    print("box length across the car fails")


def check_ctr_within_axis_slack_passes() -> None:
    """An offset in (half_i, half_i+slack] along one axis is still LIVE."""
    # Lateral (right / ax0): half_0=1.0, so 1.5 m is on the inclusive slack edge.
    off = HALF[0] + ctb.CTR_POS_SLACK_M
    assert HALF[0] < off <= HALF[0] + ctb.CTR_POS_SLACK_M
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos in _path():
            ctr = (pos[0] + off, pos[1], pos[2])
            ticks.append([_car("gvd_traffic_0", pos, _box(ctr))])
        _write(root, _add_axes_companion(ticks))
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and summary["status"] == "LIVE", summary
        assert stats["proven_live"] and not stats["spawn_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
        assert stats["max_ctr_pos_m"] is not None
        assert abs(stats["max_ctr_pos_m"] - off) < 1e-9
        _assert_mixed_axes_caveat(ctb.format_report(summary, "synthetic"))
    print("ctr within per-axis slack passes")


def check_ctr_2m_lateral_fails() -> None:
    """A 2 m lateral offset is outside half_right+slack and must be SPAWN-STUCK."""
    off = 2.0
    assert off > HALF[0] + ctb.CTR_POS_SLACK_M
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos in _path():
            ctr = (pos[0] + off, pos[1], pos[2])
            ticks.append([_car("gvd_traffic_0", pos, _box(ctr))])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["status"] == "SPAWN-STUCK" and not summary["ok"], summary
        assert stats["spawn_stuck"] and not stats["axes_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
        assert stats["max_ctr_pos_m"] is not None
        assert abs(stats["max_ctr_pos_m"] - off) < 1e-9
    print("2 m lateral ctr offset fails")


def check_exactly_6_moving_boxed_passes() -> None:
    """Exactly MIN_MOVING_BOXED moving boxed samples is enough for LIVE."""
    # 0,5,...,40 m: six samples with displacement > MOVE_M (10 m).
    poses = _path(n=9, step=5.0)
    moving = [p for p in poses if (p[1] - poses[0][1]) > ctb.MOVE_M]
    assert len(moving) == ctb.MIN_MOVING_BOXED
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = [[_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))] for pos in poses]
        _write(root, _add_axes_companion(ticks))
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and summary["status"] == "LIVE", summary
        assert stats["n_moving_boxed"] == ctb.MIN_MOVING_BOXED
        assert stats["proven_live"]
        _assert_mixed_axes_caveat(ctb.format_report(summary, "synthetic"))
    print("exactly 6 moving boxed passes")


def check_proven_segment_plus_short_tail_passes() -> None:
    """A proven segment plus a later short respawn tail must still be LIVE."""
    rows: list[dict] = []
    i = 0
    for pos in _path():
        rows.append({
            "i": i,
            "t": i * 0.5,
            "traffic": [_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))],
        })
        i += 1
    # Respawn under JUMP_M (20 m gap) with a short tail that never reaches
    # MIN_MOVING_BOXED. No violating sample; this is a typical smoke-run tail.
    tail_start = (0.0, 20.0, 0.5)
    for k in range(4):
        pos = (tail_start[0], tail_start[1] + k * 4.0, tail_start[2])
        row = {
            "i": i,
            "t": i * 0.5,
            "traffic": [_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))],
        }
        if k == 0:
            row["veh_meta"] = {"model": "etk800"}
        rows.append(row)
        i += 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_rows(root, _add_axes_companion_rows(rows))
        assert ctb.main([str(root)]) == 0
        loaded = ctb.load_state(root / "state.jsonl")
        summary = ctb.summarize(loaded)
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and summary["status"] == "LIVE", summary
        assert stats["n_spawns"] == 2 and stats["proven_live"]
        _assert_mixed_axes_caveat(ctb.format_report(summary, "synthetic"))
        by_name, _, _ = ctb.split_segments(loaded)
        segs = [ctb._segment_stats(s) for s in by_name["gvd_traffic_0"]]
        assert len(segs) == 2
        assert segs[0]["proven_live"]
        assert not segs[1]["proven_live"]
        assert not segs[1]["spawn_stuck"] and not segs[1]["axes_stuck"]
        assert segs[1]["n_moving_boxed"] < ctb.MIN_MOVING_BOXED
    print("proven segment plus short tail passes")


def check_ctr_1_3m_above_car_fails() -> None:
    """A box centre ~1.3 m above the car must be SPAWN-STUCK (locks the vertical axis)."""
    off = 1.3
    assert off > HALF[2] + ctb.CTR_POS_SLACK_M
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos in _path():
            ctr = (pos[0], pos[1], pos[2] + off)
            ticks.append([_car("gvd_traffic_0", pos, _box(ctr))])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["status"] == "SPAWN-STUCK" and not summary["ok"], summary
        assert stats["spawn_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
        assert stats["max_ctr_pos_m"] is not None
        assert abs(stats["max_ctr_pos_m"] - off) < 1e-9
    print("1.3 m above-car ctr fails")


def check_straight_only_axes_unproven() -> None:
    """A7: one straight live car must exit 3 with status AXES-UNPROVEN."""
    ticks = [[_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))] for pos in _path()]
    rc, summary, text = _run_ticks(ticks)
    stats = summary["cars"]["gvd_traffic_0"]
    assert rc == 3 and not summary["ok"] and summary["status"] == "AXES-UNPROVEN", summary
    assert stats["proven_live"] and not stats["axes_yaw_proven"]
    assert stats["yaw_span_deg"] < ctb.MIN_YAW_SPAN_DEG
    assert "yaw_span_deg=" in text
    assert "LIVE, axes UNPROVEN" in text
    assert "status AXES-UNPROVEN" in text
    assert "Do not claim those axes are live" in text
    assert "Do not start the long run" in text
    assert "Re-smoke on a route where a gvd_traffic_* car turns >= 30° while boxed and more than 10 m from spawn" in text
    assert text.index("Re-smoke on a route") < text.index("keep only half")
    assert "PASS:" not in text
    assert "Long run may proceed" not in text
    print("straight-only axes UNPROVEN exits 3")


def check_two_straight_cars_axes_unproven() -> None:
    """C10: two straight live cars must exit 3 with status AXES-UNPROVEN."""
    ticks = []
    for pos0, pos1 in zip(_path(), _path(start=(20.0, 0.0, 0.5))):
        ticks.append([
            _car("gvd_traffic_0", pos0, _box(_live_ctr(pos0))),
            _car("gvd_traffic_1", pos1, _box(_live_ctr(pos1))),
        ])
    rc, summary, text = _run_ticks(ticks)
    assert rc == 3 and not summary["ok"] and summary["status"] == "AXES-UNPROVEN", summary
    for name in ("gvd_traffic_0", "gvd_traffic_1"):
        stats = summary["cars"][name]
        assert stats["proven_live"] and not stats["axes_yaw_proven"]
        assert stats["yaw_span_deg"] < ctb.MIN_YAW_SPAN_DEG
    assert "status AXES-UNPROVEN" in text
    assert "Do not start the long run" in text
    assert "PASS:" not in text
    print("two straight cars axes UNPROVEN exits 3")


def check_turn_only_inside_first_10m_axes_unproven() -> None:
    """Y12: a heading change only inside the first MOVE_M must not prove axes."""
    poses: list[tuple[tuple[float, float, float], float]] = []
    x, y, z = 0.0, 0.0, 0.5
    yaws = [0.0] + [math.radians(40.0)] * (LIVE_N - 1)
    for yaw in yaws:
        poses.append(((x, y, z), yaw))
        x += 5.0 * math.sin(yaw)
        y += 5.0 * math.cos(yaw)
    # The first 40° sample is still inside MOVE_M; first moving sample is at 40°.
    first_turn = poses[1][0]
    assert _dist_from(poses[0][0], first_turn) < ctb.MOVE_M
    rc, summary, text = _run_ticks(_ticks_from_poses(poses))
    stats = summary["cars"]["gvd_traffic_0"]
    assert rc == 3 and summary["status"] == "AXES-UNPROVEN", summary
    assert stats["proven_live"] and not stats["axes_yaw_proven"]
    assert stats["yaw_span_deg"] < ctb.MIN_YAW_SPAN_DEG
    assert "AXES-UNPROVEN" in text
    print("turn only inside first 10 m is axes UNPROVEN")


def check_short_turning_tail_does_not_prove_axes() -> None:
    """Y7: yaw span on a non-proven respawn tail must not prove axes."""
    rows: list[dict] = []
    i = 0
    for pos in _path():
        rows.append({
            "i": i,
            "t": i * 0.5,
            "traffic": [_car("gvd_traffic_0", pos, _box(_live_ctr(pos)))],
        })
        i += 1
    # Short tail: 8 samples, 5 moving boxed, headings 0° then 35°.
    # n_moving_boxed < MIN_MOVING_BOXED, but yaw span of those samples is 35°.
    start = (0.0, 20.0, 0.5)
    step = 5.0
    turn_from = 5
    x, y, z = start
    for k in range(8):
        yaw = 0.0 if k < turn_from else math.radians(35.0)
        pos = (x, y, z)
        direction, up, ax = _orient(yaw)
        row = {
            "i": i,
            "t": i * 0.5,
            "traffic": [_car(
                "gvd_traffic_0", pos, _box(_live_ctr(pos), ax=ax),
                direction=direction, up=up,
            )],
        }
        if k == 0:
            row["veh_meta"] = {"model": "etk800"}
        rows.append(row)
        yw = 0.0 if k < turn_from else math.radians(35.0)
        x += step * math.sin(yw)
        y += step * math.cos(yw)
        i += 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_rows(root, rows)
        rc = ctb.main([str(root)])
        loaded = ctb.load_state(root / "state.jsonl")
        summary = ctb.summarize(loaded)
        stats = summary["cars"]["gvd_traffic_0"]
        by_name, _, _ = ctb.split_segments(loaded)
        segs = [ctb._segment_stats(s) for s in by_name["gvd_traffic_0"]]
        assert len(segs) == 2
        assert segs[0]["proven_live"] and segs[0]["yaw_span_deg"] < ctb.MIN_YAW_SPAN_DEG
        assert not segs[1]["proven_live"]
        assert segs[1]["n_moving_boxed"] < ctb.MIN_MOVING_BOXED
        assert segs[1]["yaw_span_deg"] >= ctb.MIN_YAW_SPAN_DEG
        assert rc == 3 and summary["status"] == "AXES-UNPROVEN", summary
        assert stats["proven_live"] and not stats["axes_yaw_proven"]
    print("short turning tail does not prove axes")


def check_unboxed_turn_does_not_prove_axes() -> None:
    """Y11: yaw from unboxed moving samples must not prove axes."""
    poses = _straight_then_turn_poses(turn_deg=40.0)
    boxed = [yaw == 0.0 for (_pos, yaw) in poses]
    assert any(boxed) and not all(boxed)
    assert all(boxed[i] for i, (_p, yaw) in enumerate(poses) if yaw == 0.0)
    assert not any(boxed[i] for i, (_p, yaw) in enumerate(poses) if yaw != 0.0)
    rc, summary, text = _run_ticks(_ticks_from_poses(poses, boxed=boxed))
    stats = summary["cars"]["gvd_traffic_0"]
    assert rc == 3 and summary["status"] == "AXES-UNPROVEN", summary
    assert stats["proven_live"] and not stats["axes_yaw_proven"]
    assert stats["yaw_span_deg"] < ctb.MIN_YAW_SPAN_DEG
    assert "AXES-UNPROVEN" in text
    print("unboxed turn does not prove axes")


def check_flipped_axes_live_turning_passes() -> None:
    """A5/A5b: a live turning box with axes opposite dir/up is still LIVE."""
    poses = _straight_then_turn_poses(turn_deg=40.0)
    rc, summary, text = _run_ticks(_ticks_from_poses(poses, flip_ax=True))
    stats = summary["cars"]["gvd_traffic_0"]
    assert rc == 0 and summary["ok"] and summary["status"] == "LIVE", summary
    assert stats["proven_live"] and stats["axes_yaw_proven"]
    assert stats["yaw_span_deg"] >= ctb.MIN_YAW_SPAN_DEG
    assert "AXES-STUCK" not in text
    print("flipped live turning box is LIVE")


def check_flipped_crosswise_box_is_axes_stuck() -> None:
    """A5c: flipped axes with the long half across the car are AXES-STUCK."""
    poses = _straight_then_turn_poses(turn_deg=40.0)
    across = [HALF[1], HALF[0], HALF[2]]
    rc, summary, text = _run_ticks(
        _ticks_from_poses(poses, flip_ax=True, half=across)
    )
    stats = summary["cars"]["gvd_traffic_0"]
    assert rc == 1 and not summary["ok"] and summary["status"] == "AXES-STUCK", summary
    assert stats["axes_stuck"] and not stats["spawn_stuck"]
    assert "AXES-STUCK" in text
    print("flipped crosswise box is AXES-STUCK")


def check_turn_29_5_deg_axes_unproven() -> None:
    """A 29.5° heading change must exit 3 (G9: threshold 30→29 would pass)."""
    rc, summary, text = _run_ticks(
        _ticks_from_poses(_straight_then_turn_poses(turn_deg=29.5))
    )
    stats = summary["cars"]["gvd_traffic_0"]
    span = stats["yaw_span_deg"]
    assert 29.0 < span < 30.0, span
    assert rc == 3 and not summary["ok"] and summary["status"] == "AXES-UNPROVEN", summary
    assert stats["proven_live"] and not stats["axes_yaw_proven"]
    assert "AXES-UNPROVEN" in text
    print("29.5° turn is axes UNPROVEN")


def check_exact_30_deg_span_is_proven() -> None:
    """Exactly 30.0° on a proven segment counts as axes proven (Y3: >= → >)."""
    seg = {
        "ticks": 14,
        "moved_m": 65.0,
        "box_nonnull": 1.0,
        "max_ctr_pos_m": 0.7,
        "ctr_span_m": 65.0,
        "n_box": 14,
        "n_moving_boxed": 11,
        "moving": True,
        "spawn_stuck": False,
        "axes_stuck": False,
        "unproven": False,
        "yaw_span_deg": 30.0,
        "proven_live": True,
    }
    merged = ctb._merge_stats([seg])
    assert merged["yaw_span_deg"] == 30.0
    assert repr(merged["yaw_span_deg"]) == "30.0"
    assert merged["axes_yaw_proven"] is True
    just_under = dict(seg, yaw_span_deg=29.5)
    assert ctb._merge_stats([just_under])["axes_yaw_proven"] is False
    print("exact 30.0° span is axes proven")


def check_turning_car_axes_proven() -> None:
    """A live box on a >=30° turn must report axes proven."""
    poses: list[tuple[tuple[float, float, float], float]] = []
    y = 0.0
    while y <= 40.0 + 1e-9:
        poses.append(((0.0, y, 0.5), 0.0))
        y += 5.0
    yaw = math.radians(30.0)
    x, y = 0.0, 40.0
    for _ in range(8):
        x += 5.0 * math.sin(yaw)
        y += 5.0 * math.cos(yaw)
        poses.append(((x, y, 0.5), yaw))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos, yaw in poses:
            direction, up, ax = _orient(yaw)
            ticks.append([_car(
                "gvd_traffic_0", pos, _box(_live_ctr(pos), ax=ax),
                direction=direction, up=up,
            )])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and summary["status"] == "LIVE", summary
        assert stats["proven_live"] and stats["axes_yaw_proven"]
        assert stats["yaw_span_deg"] >= ctb.MIN_YAW_SPAN_DEG
        text = ctb.format_report(summary, "synthetic")
        assert "LIVE, axes UNPROVEN" not in text
        assert "Axes are UNPROVEN" not in text
        assert "axes are unproven" not in text
        assert "Long run may proceed on this evidence." in text
        assert f"yaw span >= {ctb.MIN_YAW_SPAN_DEG:g}°" in text
        assert "yaw_span_deg=" in text
    print("turning car (>=30°) axes proven")


def check_unusable_half_on_moving_sample_is_spawn_stuck() -> None:
    """Unusable half on a moving boxed sample must be SPAWN-STUCK (not AXES-STUCK)."""
    poses = _path()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos in poses:
            if (pos[1] - poses[0][1]) > ctb.MOVE_M:
                box = _box(_live_ctr(pos), half=[None, HALF[1], HALF[2]])  # type: ignore[list-item]
            else:
                box = _box(_live_ctr(pos))
            ticks.append([_car("gvd_traffic_0", pos, box)])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["status"] == "SPAWN-STUCK" and not summary["ok"], summary
        assert stats["spawn_stuck"]
        assert stats["n_moving_boxed"] >= ctb.MIN_MOVING_BOXED
        assert "AXES-STUCK" not in summary["status"]
        text = ctb.format_report(summary, "synthetic")
        assert "SPAWN-STUCK" in text
    print("unusable half on moving sample is SPAWN-STUCK")


def check_predicates_locked() -> None:
    """Mutations of the |ctr-pos| limit, axis check, or spawn split must fail tests."""
    src = (ROOT / "scripts" / "check_traffic_box.py").read_text(encoding="utf-8")
    assert "SPAWN_STUCK_M" not in src
    assert "box.ctr follows pos" not in src
    assert "abs(_dot(delta, ax)) > half[i] + CTR_POS_SLACK_M" in src
    assert "if ctr_exceeds_half(box, pos, ctr):" in src
    assert "abs(_dot(a, dir_u)) >= AXIS_DOT_MIN" in src
    assert "abs(_dot(a, up_u)) >= AXIS_DOT_MIN" in src
    assert "return has_dir and has_up" in src
    assert "longest half among axes not aligned with up" in src
    assert "up-aligned axis the vertical half" not in src
    assert "Argument order is (ref, axis)" not in src
    assert "for i, ax in enumerate(units):" in src
    assert "for i, ax in enumerate(units[:2]):" not in src
    assert "MIN_YAW_SPAN_DEG = 30.0" in src
    assert 's["yaw_span_deg"] >= MIN_YAW_SPAN_DEG' in src
    assert "LIVE, axes UNPROVEN" in src
    assert 'status = "AXES-UNPROVEN"' in src
    assert 'return 3 if summary["status"] == "AXES-UNPROVEN" else 1' in src
    assert "if paired is None:\n        return True" in src
    assert '_dist(pos, last_pos[name]) > JUMP_M' in src
    assert 'if "veh_meta" in row:' in src
    assert "n_moving_boxed >= MIN_MOVING_BOXED" in src
    # Behaviour: a far ctr on a moving boxed sample is stuck even with 0.3 m wobble.
    samples = [
        _car("gvd_traffic_0", (0.0, y, 0.5), _box((0.15 * math.sin(i), 0.15 * math.cos(i), 0.8)))
        for i, y in enumerate((0.0, 5.0, 12.0, 18.0, 24.0, 30.0, 36.0))
    ]
    # One segment, via summarize.
    rows = [{"t": i, "traffic": [s]} for i, s in enumerate(samples)]
    summary = ctb.summarize(rows)
    assert summary["cars"]["gvd_traffic_0"]["spawn_stuck"]
    # Just at MOVE_M is not a moving boxed sample.
    short = [
        _car("gvd_traffic_0", (0.0, y, 0.5), _box((0.0, 0.0, 0.8)))
        for y in (0.0, ctb.MOVE_M)
    ]
    short_stats = ctb._segment_stats(short)
    assert not short_stats["moving"] and not short_stats["spawn_stuck"]
    print("predicates locked")


def main() -> None:
    check_live_box_passes()
    check_spawn_stuck_flagged()
    check_wobbling_stuck_box_fails()
    check_two_spawn_stuck_box_fails()
    check_two_spawn_live_box_passes()
    check_mixed_live_and_stuck_fails()
    check_parked_ignored_unproven_without_mover()
    check_moving_without_box_unproven()
    check_too_few_moving_boxed_is_unproven()
    check_ignores_non_companion_and_accepts_state_file()
    check_missing_and_empty()
    check_pos_jump_and_veh_meta_split()
    check_turning_car_axes()
    check_rolled_box_up_misaligned_fails()
    check_frozen_axes_after_exact_90deg_turn_fails()
    check_box_length_across_car_fails()
    check_ctr_within_axis_slack_passes()
    check_ctr_2m_lateral_fails()
    check_exactly_6_moving_boxed_passes()
    check_proven_segment_plus_short_tail_passes()
    check_ctr_1_3m_above_car_fails()
    check_straight_only_axes_unproven()
    check_two_straight_cars_axes_unproven()
    check_turn_only_inside_first_10m_axes_unproven()
    check_short_turning_tail_does_not_prove_axes()
    check_unboxed_turn_does_not_prove_axes()
    check_flipped_axes_live_turning_passes()
    check_flipped_crosswise_box_is_axes_stuck()
    check_turn_29_5_deg_axes_unproven()
    check_exact_30_deg_span_is_proven()
    check_turning_car_axes_proven()
    check_unusable_half_on_moving_sample_is_spawn_stuck()
    check_predicates_locked()
    print("check_traffic_box ok")


if __name__ == "__main__":
    main()
