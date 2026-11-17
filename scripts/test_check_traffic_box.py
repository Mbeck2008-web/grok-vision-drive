#!/usr/bin/env python3
"""Offline checks for scripts/check_traffic_box.py (synthetic state.jsonl)."""
from __future__ import annotations

import json
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


def _box(ctr: tuple[float, float, float]) -> dict:
    return {"ctr": [ctr[0], ctr[1], ctr[2]], "half": list(HALF), "ax": [list(a) for a in AX]}


def _car(name: str, pos: tuple[float, float, float], box: dict | None) -> dict:
    return {
        "name": name,
        "pos": [pos[0], pos[1], pos[2]],
        "vel": [0.0, 8.0, 0.0],
        "dir": [0.0, 1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "box": box,
    }


def _write(root: Path, cars_per_tick: list[list[dict | None]]) -> Path:
    """Write state.jsonl. A tick may be traffic=null (the 2 Hz gaps)."""
    path = root / "state.jsonl"
    lines = []
    for i, cars in enumerate(cars_per_tick):
        row: dict = {"t": float(i) * 0.1, "i": i}
        row["traffic"] = None if cars is None else cars
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _path(n: int, start: tuple[float, float, float] = (0.0, 0.0, 0.5), step: float = 5.0) -> list[tuple[float, float, float]]:
    return [(start[0], start[1] + i * step, start[2]) for i in range(n)]


def check_live_box_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = _path(8)  # 35 m of travel
        ticks: list[list[dict | None]] = []
        for i, pos in enumerate(path):
            if i % 2 == 1:
                ticks.append(None)  # traffic is only on some ticks
                continue
            ctr = (pos[0] + CTR_OFFSET[0], pos[1] + CTR_OFFSET[1], pos[2] + CTR_OFFSET[2])
            ticks.append([_car("gvd_traffic_0", pos, _box(ctr))])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 0
        summary = ctb.summarize(ctb.load_state(ctb.resolve_state(root)))
        stats = summary["cars"]["gvd_traffic_0"]
        assert summary["ok"] and summary["status"] == "LIVE"
        assert stats["moving"] and not stats["spawn_stuck"] and not stats["unproven"]
        assert stats["box_nonnull"] == 1.0
        # |ctr - pos| is the constant offset, not the travel distance.
        expected = (CTR_OFFSET[0] ** 2 + CTR_OFFSET[1] ** 2 + CTR_OFFSET[2] ** 2) ** 0.5
        assert stats["max_ctr_pos_m"] is not None
        assert abs(stats["max_ctr_pos_m"] - expected) < 1e-9, stats
        assert stats["moved_m"] > ctb.MOVE_M
        assert stats["ctr_span_m"] is not None and stats["ctr_span_m"] > ctb.MOVE_M
    print("live box passes")


def check_spawn_stuck_flagged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spawn_ctr = (10.0, 20.0, 0.8)
        ticks = []
        for pos in _path(8, start=(10.0, 20.0, 0.5)):
            ticks.append([_car("gvd_traffic_0", pos, _box(spawn_ctr))])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        stats = summary["cars"]["gvd_traffic_0"]
        assert not summary["ok"] and summary["status"] == "SPAWN-STUCK"
        assert summary["stuck"] == ["gvd_traffic_0"]
        assert stats["spawn_stuck"] and stats["moving"]
        assert stats["box_nonnull"] == 1.0
        assert stats["ctr_span_m"] == 0.0
        # max |ctr-pos| grows with travel (last pos is 35 m ahead, z differs).
        assert stats["max_ctr_pos_m"] is not None and stats["max_ctr_pos_m"] > ctb.MOVE_M
        text = ctb.format_report(summary, "synthetic")
        assert "SPAWN-STUCK" in text and "Do not start the long run" in text
        assert "Keep only half" in text
    print("spawn-stuck ctr flagged")


def check_mixed_live_and_stuck_fails() -> None:
    """One live companion does not hide a spawn-stuck one."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos0, pos1 in zip(_path(8), _path(8, start=(100.0, 0.0, 0.5))):
            live_ctr = (pos0[0] + CTR_OFFSET[0], pos0[1] + CTR_OFFSET[1], pos0[2] + CTR_OFFSET[2])
            ticks.append([
                _car("gvd_traffic_0", pos0, _box(live_ctr)),
                _car("gvd_traffic_1", pos1, _box((100.0, 0.0, 0.8))),
            ])
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        assert summary["status"] == "SPAWN-STUCK"
        assert summary["stuck"] == ["gvd_traffic_1"]
        assert not summary["cars"]["gvd_traffic_0"]["spawn_stuck"]
        assert summary["cars"]["gvd_traffic_0"]["moving"]
    print("mixed live/stuck fails")


def check_parked_ignored_unproven_without_mover() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 2 m of wander, under MOVE_M, even if ctr is fixed.
        ticks = [_car("gvd_traffic_0", (0.0, i * 0.4, 0.5), _box((0.0, 0.0, 0.8)))
                 for i in range(6)]
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
        ticks = [[_car("gvd_traffic_0", pos, None)] for pos in _path(8)]
        _write(root, ticks)
        assert ctb.main([str(root)]) == 1
        summary = ctb.summarize(ctb.load_state(root / "state.jsonl"))
        assert summary["status"] == "UNPROVEN" and summary["unproven"] == ["gvd_traffic_0"]
        assert summary["cars"]["gvd_traffic_0"]["box_nonnull"] == 0.0
        assert summary["cars"]["gvd_traffic_0"]["max_ctr_pos_m"] is None
    print("moving without box is unproven")


def check_ignores_non_companion_and_accepts_state_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ticks = []
        for pos in _path(8):
            ctr = (pos[0] + CTR_OFFSET[0], pos[1] + CTR_OFFSET[1], pos[2] + CTR_OFFSET[2])
            ticks.append([
                _car("gvd_traffic_0", pos, _box(ctr)),
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


def check_spawn_stuck_predicate_is_live() -> None:
    """Mutation target: the spawn-stuck flag must use MOVE_M / SPAWN_STUCK_M.

    If the comparison is removed or forced false, a car that travelled past
    MOVE_M with a fixed ctr would pass, and this check fails.
    """
    src = (ROOT / "scripts" / "check_traffic_box.py").read_text(encoding="utf-8")
    # The flag is the conjunction the smoke run depends on; both constants
    # must appear in the spawn-stuck predicate, not only in print strings.
    needle = "moving\n        and first_ctr is not None\n        and ctr_span < SPAWN_STUCK_M"
    assert needle in src, "spawn-stuck predicate missing or rewritten"
    assert "pos_span > MOVE_M" in src
    samples = [
        _car("gvd_traffic_0", (0.0, y, 0.5), _box((0.0, 0.0, 0.8)))
        for y in (0.0, 5.0, 12.0)
    ]
    stats = ctb._car_stats(samples)
    assert stats["moving"] and stats["spawn_stuck"]
    # Just under MOVE_M must not flag.
    short = [
        _car("gvd_traffic_0", (0.0, y, 0.5), _box((0.0, 0.0, 0.8)))
        for y in (0.0, ctb.MOVE_M)
    ]
    # moved_m == MOVE_M is not ">" so not moving.
    short_stats = ctb._car_stats(short)
    assert not short_stats["moving"] and not short_stats["spawn_stuck"]
    # Ctr that also travels past SPAWN_STUCK_M is live, not stuck.
    live = [
        _car("gvd_traffic_0", (0.0, y, 0.5), _box((0.0, y, 0.8)))
        for y in (0.0, 5.0, 12.0)
    ]
    live_stats = ctb._car_stats(live)
    assert live_stats["moving"] and not live_stats["spawn_stuck"]
    print("spawn-stuck predicate locked")


def main() -> None:
    check_live_box_passes()
    check_spawn_stuck_flagged()
    check_mixed_live_and_stuck_fails()
    check_parked_ignored_unproven_without_mover()
    check_moving_without_box_unproven()
    check_ignores_non_companion_and_accepts_state_file()
    check_missing_and_empty()
    check_spawn_stuck_predicate_is_live()
    print("check_traffic_box ok")


if __name__ == "__main__":
    main()
