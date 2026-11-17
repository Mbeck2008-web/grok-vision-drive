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


HALF = [2.2, 1.0, 0.7]
AX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# A small, stable origin-to-centre offset (live box, not spawn).
CTR_OFFSET = (0.3, -0.4, 0.5)
# Enough points that a 5 m step yields >= MIN_MOVING_BOXED samples past MOVE_M.
LIVE_N = 14


def _box(ctr: tuple[float, float, float], ax: list | None = None) -> dict:
    return {"ctr": [ctr[0], ctr[1], ctr[2]], "half": list(HALF), "ax": [list(a) for a in (ax or AX)]}


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
        _write(root, ticks)
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
        assert "box.ctr follows pos" not in ctb.format_report(summary, "synthetic")
        assert "IQU45 smoke check" in ctb.format_report(summary, "synthetic")
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
        assert "|box.ctr - pos|" in text
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
        _write_rows(root, rows)
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and stats["proven_live"] and stats["n_spawns"] == 2
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
        state = _write(root, ticks)
        assert ctb.main([str(state)]) == 0
        summary = ctb.summarize(ctb.load_state(state))
        assert "parked_bus" not in summary["cars"]
        assert summary["ok"]
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

    frozen = _run(False)
    assert frozen["_rc"] == 1 and frozen["status"] == "AXES-STUCK", frozen
    stats = frozen["cars"]["gvd_traffic_0"]
    assert stats["axes_stuck"] and not stats["spawn_stuck"]
    text = ctb.format_report(frozen, "synthetic")
    assert "AXES-STUCK" in text and "dir/up" in text
    print("turning-car axes: live passes, frozen fails")


def check_predicates_locked() -> None:
    """Mutations of the |ctr-pos| limit, axis check, or spawn split must fail tests."""
    src = (ROOT / "scripts" / "check_traffic_box.py").read_text(encoding="utf-8")
    assert "SPAWN_STUCK_M" not in src
    assert "box.ctr follows pos" not in src
    assert "ctr_pos > limit" in src
    assert "return norm + CTR_POS_SLACK_M" in src
    assert "abs(_dot(a, dir_u)) >= AXIS_DOT_MIN" in src
    assert "abs(_dot(a, up_u)) >= AXIS_DOT_MIN" in src
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
    check_predicates_locked()
    print("check_traffic_box ok")


if __name__ == "__main__":
    main()
