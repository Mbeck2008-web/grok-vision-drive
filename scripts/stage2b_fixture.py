"""A recording that stage 2b can be trained and scored on (tests only): a car drives past stop signs and red lights.

48x64 ``raw/main`` frames, ``state.jsonl`` (``ego.sim_t`` / ``ego.speed_mps`` / AI pedals from a scripted driver on a
known longitudinal model, ``veh_meta``), ``gt.jsonl`` (``lane_gt``) and ``signs.jsonl`` (``id`` from the world position,
ego-frame ``x`` / ``y``, the lamp ``state``). The driver cruises, brakes to a stop 1 m before each intersection's line
(0.7 pedal), waits 2.4 s and drives on. Which rows end up in the holdout is the repo's own block rule.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DT = 0.2  # 5 Hz keeps the recording (and the CPU training on it) short; the fit takes steps up to 0.25 s
SEG = "west_coast_usa#0"
SEGMENT_GAP_S = 60.0
SEG_B = "west_coast_usa#1"  # the last of two short segments is the holdout (the repo's whole-segment rule)
K_THR, K_BRK, K_DRAG = 2.0, 4.5, 0.05  # the car's true map (what fit_motion should find on the train rows)
CRUISE = 10.0


def script(n_rows: int, intersections: list[tuple[str, float, str | None]]):
    """``[(cls, world_y, state)]`` -> per-tick (pos_y, v, throttle, brake) for ``n_rows`` ticks."""
    pos, v, out = 0.0, CRUISE, []
    hold_until, passed = -1, set()
    for k in range(n_rows):
        thr = float(np.clip(K_DRAG * v / K_THR + 0.1 * (CRUISE - v), 0.0, 1.0))
        brk = 0.0
        nxt = [(y, c) for c, y, _s in intersections if y - pos > -1.0 and (c, y) not in passed]
        if nxt:
            dist = min(nxt)[0] - pos
            if k < hold_until:
                thr, brk = 0.0, 0.3
            elif dist < 17.0 and v > 0.2 and hold_until < 0:
                thr, brk = 0.0, 0.7
            elif v <= 0.2 and dist < 17.0 and hold_until < 0:
                hold_until = k + 12
                thr, brk = 0.0, 0.3
            if hold_until > 0 and k >= hold_until:
                passed.add((min(nxt)[1], min(nxt)[0]))
                hold_until = -1
                thr, brk = 1.0, 0.0
        out.append((pos, v, thr, brk))
        v = max(0.0, v + (K_THR * thr - K_BRK * brk - K_DRAG * v) * DT)
        pos += v * DT
    return out


def build(root: Path, n_rows: int, intersections: list[tuple[str, float, str | None]], *, with_signs: bool = True,
          with_aux: bool = False) -> Path:
    import cv2

    from python.collect.signal_labels import sign_id
    from python.debug.lane_targets import LANE_GT_DIM

    cam = root / "raw" / "main"
    cam.mkdir(parents=True)
    rng = np.random.default_rng(0)
    run = script(n_rows, intersections)
    state_lines, gt_lines, sign_lines = [], [], []
    for index, (pos, v, thr, brk) in enumerate(run):
        frame = (rng.random((48, 64, 3)) * 160 + 40).astype(np.uint8)
        col = int(32 + 20 * math.sin(index / 9.0))
        frame[:, max(0, col - 2): col + 2, :] = 250
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        (cam / f"{index:06d}.png").write_bytes(buf.tobytes())
        t = index * DT + (SEGMENT_GAP_S if index >= SPLIT else 0.0)  # a real second segment starts later; the split manifest purges train rows near the holdout in time
        state = {
            "i": index, "t": t, "dt_s": None if index == 0 else DT, "segment": SEG if index < SPLIT else SEG_B, "spawn_type": "city",
            "ego": {"sim_t": 100.0 + t, "speed_mps": v},
            "cams": {"main": {"file": f"raw/main/{index:06d}.png", "t": t - 0.02}},
            "wheel": {"steer": 0.1, "steer_ai": 0.2 * math.sin(index / 20.0)},
            "pedals": {"throttle": thr, "brake": brk, "throttle_ai": thr, "brake_ai": brk, "clutch_ai": None, "parkingbrake_ai": 0.0},
            "veh": {"reverse": 0},
        }
        if index == 0:
            state["veh_meta"] = {"model": "test", "steer_lock_deg": 450.0, "wheelbase_m": 2.7}
        state_lines.append(state)
        center = 0.6 * math.sin(index / 9.0)
        lane = []
        for _ in range(4):
            lane += [center, center - 1.8, center + 1.8]
        gt_lines.append({"i": index, "t": t, "tech": {"lane_gt": lane + [0.0, 1.0], "lane_gt_valid": [1.0] * LANE_GT_DIM,
                                                       "speed_mps": v}})
        signs = []
        for cls, wy, lamp in intersections:
            y = wy - pos
            if -5.0 <= y <= 80.0:
                signs.append({"id": sign_id(cls, (0.5, wy)), "cls": cls, "x": 0.5, "y": y,
                              "state": lamp if cls == "traffic_light" else "unknown", "src": "traffic_signals"})
        sign_lines.append({"t": t, "signs": sorted(signs, key=lambda s: (s["y"], s["x"], s["cls"]))})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_lines) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt_lines) + "\n", encoding="utf-8")
    if with_aux:  # the train-only planes stage 1 needs (u16 cm depth, u8 sem ids), same shape as test_trainer2_recipe._fixture
        (root / "raw" / "main_depth").mkdir()
        (root / "raw" / "main_sem").mkdir()
        for index in range(n_rows):
            cm = np.full((48, 64), 1250 + 20 * (index % 7), np.uint16)
            cm[:12] = 30000
            cm[-4:, -8:] = 0
            ids = np.ones((48, 64), np.uint8)
            ids[:, 32:] = 2 + (index % 2)
            ids[:10] = 3
            ids[-3:, :6] = 0
            for sub, arr in (("main_depth", cm), ("main_sem", ids)):
                ok, buf = cv2.imencode(".png", arr)
                assert ok
                (root / "raw" / sub / f"{index:06d}.png").write_bytes(buf.tobytes())
        (root / "sem_classes.json").write_text(json.dumps({"kind": "gvd_sem_classes", "train_only": True,
                                                           "classes": {"0": "unknown", "1": "road", "2": "car", "3": "sky"}}),
                                               encoding="utf-8")
    if with_signs:
        (root / "signs.jsonl").write_text("\n".join(json.dumps(r) for r in sign_lines) + "\n", encoding="utf-8")
    return root


N_ROWS = 230
SPLIT = 80  # rows [0, SPLIT) are segment #0 (train: one stop sign, the fit's braking rows), the rest segment #1 (the holdout)
TRAIN_PART = [("stop_sign", 45.0, None)]


def intersections(n_rows: int = N_ROWS) -> list[tuple[str, float, str | None]]:
    """One train intersection, then a stop sign and a red light whose 40 m approach starts in segment #1."""
    head = script(n_rows, TRAIN_PART)
    sign_y = head[SPLIT + 4][0] + 45.0
    return TRAIN_PART + [("stop_sign", sign_y, None), ("traffic_light", sign_y + 90.0, "red")]
