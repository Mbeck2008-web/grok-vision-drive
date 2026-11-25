#!/usr/bin/env python3
"""Offline tests for automatic Recorder v2 clip QC and training target masks."""
from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gvd_test_env  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )


def _write_png(path: Path, color: tuple[int, int, int] = (40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


def _yaw_quat(yaw: float) -> list[float]:
    # BeamNG ego forward = (-sin(yaw), cos(yaw), 0)
    half = 0.5 * yaw
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _make_row(
    i: int,
    t: float,
    x: float,
    y: float,
    *,
    yaw: float = 0.0,
    speed: float = 10.0,
    yaw_rate: float = 0.0,
    steer: float = 0.0,
    throttle: float = 0.35,
    brake: float = 0.0,
    turnsignal: int = 0,
    settle: bool = False,
    stall: bool = False,
    recover: bool = False,
    map_name: str = "west_coast_usa",
    spawn_type: str = "road",
    damage: float = 0.0,
    cam_path: str | None = None,
    dt_s: float | None = None,
    dir_z: float = 0.0,
) -> dict:
    vx = -math.sin(yaw) * speed
    vy = math.cos(yaw) * speed
    row: dict = {
        "i": i,
        "t": round(t, 4),
        "sim_t": round(t, 4),
        "segment": 0,
        "map": map_name,
        "spawn_type": spawn_type,
        "settle": settle,
        "stall": stall,
        "recover": recover,
        "ego": {
            "pos": [round(x, 4), round(y, 4), 0.0],
            "vel": [round(vx, 4), round(vy, 4), 0.0],
            "dir": [round(-math.sin(yaw), 6), round(math.cos(yaw), 6), round(dir_z, 6)],
            "quat": _yaw_quat(yaw),
            "speed_mps": round(speed, 4),
            "yaw_rate": round(yaw_rate, 4),
            "sim_t": round(t, 4),
            "damage": damage,
        },
        "veh": {
            "steer_ai": round(steer, 4),
            "throttle_ai": round(throttle, 4),
            "brake_ai": round(brake, 4),
            "turnsignal": turnsignal,
            "gear": "D",
        },
        "wheel": {"steer": round(steer, 4), "steer_ai": round(steer, 4)},
        "pedals": {
            "throttle": round(throttle, 4),
            "throttle_ai": round(throttle, 4),
            "brake": round(brake, 4),
            "brake_ai": round(brake, 4),
        },
        "tech": {
            "lanes": [],
            "curbs": [],
            "objects": [],
            "signs": [],
            "lane_gt": [0.0] * 13 + [1.0],
        },
    }
    if dt_s is not None:
        row["dt_s"] = dt_s
    if cam_path is not None:
        row["cams"] = {"main": {"file": cam_path, "t": round(t, 4)}}
    return row


def test_settle_5s_and_map_warmup_20s() -> None:
    from python.train.clip_qc import analyze_clip, analyze_rows

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_warmup"
        clip.mkdir()
        rows = []
        # 5 Hz for 52 seconds: t = 100.0 .. 151.8
        # Map change at t=132.0 -> warmup [100,120) and [132,152)
        # Mid-clip teleport settle window at t=123.0 .. 125.0 -> settle_5s [123, 128)
        for idx in range(260):
            t = 100.0 + idx * 0.2
            m = "italy" if t < 132.0 else "east_coast_usa"
            settle = (123.0 <= t < 126.0)
            rows.append(_make_row(idx, t, 0.0, idx * 2.0, map_name=m, settle=settle))
        _write_jsonl(clip / "state.jsonl", rows)
        (clip / "collect.json").write_text(
            json.dumps(
                {
                    "settle_windows": [
                        {"seq": 0, "source": "spawn", "t0": 98.0, "t1": 103.0, "wait_s": 5.0, "i0": 0, "i1": 15},
                        {"seq": 1, "source": "segment", "t0": 123.0, "t1": 126.0, "wait_s": 3.0, "i0": 115, "i1": 130},
                    ]
                }
            ),
            encoding="utf-8",
        )
        res = analyze_clip(clip, check_images=False)
        by_i = {r["i"]: r for r in res["rows"]}
        # Initial spawn settle_5s anchored to first row t=100.0 -> [100.0, 105.0)
        assert "settle_5s" in by_i[0]["reasons"]
        assert "settle_5s" in by_i[24]["reasons"]  # t=104.8
        assert "settle_5s" not in by_i[25]["reasons"]  # t=105.0
        # Initial 20 s map warmup [100.0, 120.0)
        assert "map_warmup" in by_i[0]["reasons"]
        assert "map_warmup" in by_i[99]["reasons"]  # t=119.8
        assert "map_warmup" not in by_i[100]["reasons"]  # t=120.0
        # Clean driving window [120.0, 123.0)
        assert by_i[105]["masked"] is False
        assert by_i[105]["reasons"] == []
        # Mid-clip teleport 5 s settle [123.0, 128.0)
        assert "settle_5s" in by_i[115]["reasons"]  # t=123.0
        assert "settle_5s" in by_i[139]["reasons"]  # t=127.8
        assert "settle_5s" not in by_i[140]["reasons"]  # t=128.0
        # Clean window [128.0, 132.0)
        assert by_i[145]["masked"] is False
        # Map change at t=132.0 triggers fresh 20 s warmup
        assert "map_warmup" in by_i[160]["reasons"]  # t=132.0
        assert "map_warmup" in by_i[259]["reasons"]  # t=151.8


def test_near_signal_stoplight_stop_yield_and_dz() -> None:
    from python.train.clip_qc import analyze_clip, analyze_rows

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_signals"
        clip.mkdir()
        # Drive along +Y at 10 m/s (2 m per 0.2 s) from y=0 to y=198
        rows = [_make_row(i, 100.0 + i * 0.2, 0.0, i * 2.0) for i in range(100)]
        _write_jsonl(clip / "state.jsonl", rows)
        # signals.jsonl:
        # - traffic light at y=60, z=0 -> masks y in [35, 85]
        # - stop sign on overpass at y=120, z=10.0 -> |dz|=10 > 3 -> must NOT mask!
        # - yield sign at y=160, z=0 -> masks y in [135, 185]
        sig_rows = []
        for r in rows:
            sig_rows.append(
                {
                    "t": r["t"],
                    "signals": {
                        "n1": {
                            "n2": [
                                {"action": 2.0, "instance": "trafficLight 1", "pos": {"x": 0.0, "y": 60.0, "z": 0.0}, "state": "redTrafficLight"},
                                {"action": 3.0, "instance": "stop 99", "pos": {"x": 0.0, "y": 120.0, "z": 10.0}, "state": "basicStop"},
                                {"action": 3.0, "instance": "yield 7", "pos": {"x": 0.0, "y": 160.0, "z": 0.0}, "state": "basicYield"},
                                {"action": 3.0, "instance": "speed_limit 1", "pos": {"x": 0.0, "y": 10.0, "z": 0.0}},
                            ]
                        }
                    },
                }
            )
        _write_jsonl(clip / "signals.jsonl", sig_rows)
        # stoplines.jsonl: stop_line at y_m=12 on row i=48 (y=96) -> masks i=48
        _write_jsonl(clip / "stoplines.jsonl", [{"i": 48, "stop_line_target": {"y_m": 12.0, "valid": True}}])
        res = analyze_clip(clip, check_images=False, warmup_s=0.0, settle_s=0.0)
        by_i = {r["i"]: r for r in res["rows"]}
        assert "near_signal" not in by_i[5]["reasons"]   # y=10 speed_limit only
        assert "near_signal" not in by_i[15]["reasons"]  # y=30
        assert "near_signal" in by_i[20]["reasons"]      # y=40 (dist 20m to traffic light)
        assert "near_signal" in by_i[30]["reasons"]      # y=60
        assert "near_signal" in by_i[48]["reasons"]      # stop_line_target y_m=12
        assert "near_signal" not in by_i[60]["reasons"]  # y=120 but z=10m overpass!
        assert "near_signal" in by_i[80]["reasons"]      # y=160 yield sign


def test_blinkers_excluded_from_targets() -> None:
    from python.train.clip_qc import EXCLUDED_TARGET_FIELDS, analyze_clip
    from python.train.ctl_targets import CTL_FIELDS
    from python.train.scene_net import HEAD_NAMES, OUTPUT_FIELDS, labels_for_row

    for bad in EXCLUDED_TARGET_FIELDS:
        assert bad not in CTL_FIELDS
        assert bad not in OUTPUT_FIELDS
        assert bad not in HEAD_NAMES

    row = _make_row(0, 130.0, 0.0, 0.0, turnsignal=1)
    row["blinker"] = 1
    labels = labels_for_row(row)
    for bad in EXCLUDED_TARGET_FIELDS:
        assert bad not in labels

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_blinkers"
        clip.mkdir()
        rows = [_make_row(i, 100.0 + i * 0.2, 0.0, i * 2.0, turnsignal=(1 if i >= 120 else 0)) for i in range(150)]
        _write_jsonl(clip / "state.jsonl", rows)
        res = analyze_clip(clip, check_images=False)
        by_i = {r["i"]: r for r in res["rows"]}
        # Nonzero turnsignal does NOT cause masking or reject
        assert by_i[130]["masked"] is False
        assert by_i[130]["reasons"] == []


def test_maneuvers_uturn_abrupt_stuck_and_overline_corner() -> None:
    from python.train.clip_qc import analyze_clip, analyze_rows

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_maneuvers"
        clip.mkdir()
        rows: list[dict] = []
        gt_rows: list[dict] = []

        def _edge_strip(pts_normals, width=8.0):
            edges = []
            half = width * 0.5
            for (mx, my), (rx, ry) in pts_normals:
                edges.append({
                    "left": [mx - rx * half, my - ry * half, 0.0],
                    "middle": [mx, my, 0.0],
                    "right": [mx + rx * half, my + ry * half, 0.0],
                })
            return edges

        straight_edges = _edge_strip([((0.0, float(k * 5)), (1.0, 0.0)) for k in range(41)], 8.0)
        corner_pts = []
        for k in range(21):
            ang = math.pi - (math.pi / 2.0) * (k / 20.0)
            rx = 25.0 + 25.0 * math.cos(ang)
            ry = 200.0 + 25.0 * math.sin(ang)
            # Tangent along decreasing ang: (sin(ang), -cos(ang)), right normal = (ty, -tx) = (-cos(ang), -sin(ang))
            corner_pts.append(((rx, ry), (-math.cos(ang), -math.sin(ang))))
        corner_edges = _edge_strip(corner_pts, 8.0)
        road_dump = {
            "road_network": {
                "straight": {"lanesLeft": 1, "lanesRight": 1, "oneway": False, "drivability": 1.0, "edges": straight_edges},
                "corner": {"lanesLeft": 1, "lanesRight": 1, "oneway": False, "drivability": 1.0, "edges": corner_edges},
            }
        }
        road_path = Path(td) / "test_map.json.gz"
        with gzip.open(road_path, "wt", encoding="utf-8") as fh:
            json.dump(road_dump, fh)

        # Segment 1 (i=0..19): normal steady driving on right lane (x=2.0, y=0..38)
        for i in range(20):
            rows.append(_make_row(i, 100.0 + i * 0.2, 2.0, i * 2.0, yaw=0.0, speed=10.0))

        # Segment 2 (i=20..39): abrupt turn spike at i=25..28 (yaw_rate=1.6, lat_acc=16.0)
        for i in range(20, 40):
            yr = 1.6 if 25 <= i <= 28 else 0.05
            st = 0.8 if 25 <= i <= 28 else 0.02
            rows.append(_make_row(i, 100.0 + i * 0.2, 2.0, 40.0 + (i - 20) * 2.0, yaw=0.0, speed=10.0, yaw_rate=yr, steer=st))

        # Segment 3 (i=40..59): stuck with throttle while speed=0.1 (i=42..56 -> 3.0s)
        for i in range(40, 60):
            stk = 42 <= i <= 56
            rows.append(
                _make_row(
                    i,
                    100.0 + i * 0.2,
                    2.0,
                    80.0,
                    yaw=0.0,
                    speed=(0.1 if stk else 8.0),
                    throttle=(0.6 if stk else 0.3),
                    brake=0.0,
                )
            )

        # Segment 4 (i=60..89): U-turn! Drives +Y at (2, 100), loops 180 deg back to (-2, 100)
        for k in range(30):
            i = 60 + k
            ang = (k / 29.0) * math.pi  # 0 -> pi (180 deg)
            # Small 6m radius turnaround loop
            x = 2.0 - 6.0 * (1.0 - math.cos(ang))
            y = 100.0 + 6.0 * math.sin(ang)
            yaw = -ang  # turns left from +Y (0) toward -Y (-pi)
            rows.append(_make_row(i, 100.0 + i * 0.2, x, y, yaw=yaw, speed=6.0, yaw_rate=0.52))

        # Segment 5 (i=90..109): Tight corner (radius 25m -> |yaw_rate|/speed = 0.04 >= 0.02)
        # First 10 rows stay inside right lane (+2.0m); rows 100..109 cut across centerline (offset 0.0m) -> over_line_corner!
        for k in range(20):
            i = 90 + k
            ang = math.pi - (math.pi / 2.0) * (k / 19.0)
            rad = 23.0 if k < 10 else 25.0  # 23.0 is +2m right-lane center (inner side of right turn); 25.0 is yellow centerline!
            rx = 25.0 + rad * math.cos(ang)
            ry = 200.0 + rad * math.sin(ang)
            # Tangent along decreasing ang: (-sin(ang), cos(ang)) -> wait tangent of (cos, sin) as ang decreases is (sin(ang), -cos(ang))
            tx, ty = math.sin(ang), -math.cos(ang)
            yaw = math.atan2(-tx, ty)
            rows.append(_make_row(i, 100.0 + i * 0.2, rx, ry, yaw=yaw, speed=10.0, yaw_rate=-0.4))
            gt_rows.append({"i": i, "off_road": False})

        _write_jsonl(clip / "state.jsonl", rows)
        _write_jsonl(clip / "gt.jsonl", gt_rows)

        res = analyze_clip(
            clip,
            road_dump=str(road_path),
            check_images=False,
            warmup_s=0.0,
            settle_s=0.0,
        )
        by_i = {r["i"]: r for r in res["rows"]}
        assert by_i[10]["reasons"] == []
        assert "abrupt_turn" in by_i[26]["reasons"]
        assert "stuck" in by_i[50]["reasons"]
        assert any("uturn" in by_i[i]["reasons"] for i in range(60, 90))
        assert "over_line_corner" not in by_i[95]["reasons"]
        assert "over_line_corner" in by_i[105]["reasons"]

        # High-speed lateral acceleration (> 6 m/s^2) even when |yaw_rate| <= 1.0 and steer smooth
        lat_rows = [_make_row(k, 100.0 + k * 0.2, 0.0, k * 5.0, speed=25.0, yaw_rate=0.32, steer=0.05) for k in range(5)]
        lat_res = analyze_rows(lat_rows, warmup_s=0.0, settle_s=0.0)
        assert "abrupt_turn" in lat_res["rows"][2]["reasons"]

        # Waiting at a red light (stationary with slight creep throttle) is near_signal, NOT stuck
        wait_rows = [_make_row(k, 100.0 + k * 0.2, 0.0, 60.0, speed=0.2, throttle=0.2, brake=0.0) for k in range(15)]
        wait_res = analyze_rows(
            wait_rows,
            world_signals=[("traffic_light", 0.0, 60.0, 0.0)],
            has_signal_files=True,
            warmup_s=0.0,
            settle_s=0.0,
        )
        assert "near_signal" in wait_res["rows"][5]["reasons"]
        assert "stuck" not in wait_res["rows"][5]["reasons"]


def test_clip_classification_train_pretrain_reject() -> None:
    from python.train.clip_qc import analyze_clip, analyze_rows

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # 1) Clean train clip: 50s @ 10 Hz (500 rows), valid PNGs, 25s clean driving after 20s warmup
        c_train = root / "clip_train"
        c_train.mkdir()
        rows_train = []
        for i in range(500):
            p = f"raw/main/{i:06d}.png"
            _write_png(c_train / p)
            rows_train.append(_make_row(i, 100.0 + i * 0.1, 0.0, i * 1.0, speed=10.0, cam_path=p))
        _write_jsonl(c_train / "state.jsonl", rows_train)
        (c_train / "clip_tags.json").write_text(json.dumps({"duration_s": 50.0, "edge_case_observed": True}), encoding="utf-8")
        res_train = analyze_clip(c_train, check_images=True)
        assert res_train["clip_class"] == "train"
        assert res_train["reason_codes"] == []

        # 2) Pretrain-only clip: valid PNGs, 40s span, but ego damaged + impact
        c_pre = root / "clip_pretrain"
        c_pre.mkdir()
        rows_pre = []
        for i in range(200):
            p = f"raw/main/{i:06d}.png"
            _write_png(c_pre / p)
            rows_pre.append(_make_row(i, 100.0 + i * 0.2, 0.0, i * 1.0, speed=6.0, damage=(10.0 if i > 150 else 0.0), cam_path=p))
        _write_jsonl(c_pre / "state.jsonl", rows_pre)
        _write_jsonl(c_pre / "impacts.jsonl", [{"i": 150, "t": 130.0}])
        res_pre = analyze_clip(c_pre, check_images=True)
        assert res_pre["clip_class"] == "pretrain_only"
        assert "ego_damaged_or_impact" in res_pre["reason_codes"]

        # 3) Reject clip: only 4 seconds of rows (< 10s min span)
        c_rej = root / "clip_reject"
        c_rej.mkdir()
        rows_rej = [_make_row(i, 100.0 + i * 0.2, 0.0, i * 2.0) for i in range(20)]
        _write_jsonl(c_rej / "state.jsonl", rows_rej)
        res_rej = analyze_clip(c_rej, check_images=False)
        assert res_rej["clip_class"] == "reject"
        assert "span_too_short" in res_rej["reason_codes"]


def test_highway_lane_position_spread_report() -> None:
    from python.train.clip_qc import analyze_clip, analyze_rows, format_highway_spread, merge_highway_spreads

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_hwy"
        clip.mkdir()
        # 2+2 twoway highway along +Y at x=0, width=14.0m (lane width = 3.5m; right lanes at x=+1.75 (idx 0), x=+5.25 (idx 1))
        road_dump = {
            "road_network": {
                "hwy1": {
                    "lanesLeft": 2,
                    "lanesRight": 2,
                    "oneway": False,
                    "drivability": 1.0,
                    "edges": [
                        {"left": [-7.0, float(y), 0.0], "middle": [0.0, float(y), 0.0], "right": [7.0, float(y), 0.0]}
                        for y in range(0, 500, 10)
                    ],
                }
            }
        }
        road_path = Path(td) / "hwy.json.gz"
        with gzip.open(road_path, "wt", encoding="utf-8") as fh:
            json.dump(road_dump, fh)

        rows = []
        # 20 highway rows: 4 in left-of-right-half lane (x=1.75 -> idx 0), 16 in rightmost lane (x=5.25 -> idx 1)
        for i in range(20):
            x = 1.75 if i < 4 else 5.25
            rows.append(_make_row(i, 100.0 + i * 0.2, x, 20.0 + i * 5.0, yaw=0.0, speed=25.0, spawn_type="highway"))
        _write_jsonl(clip / "state.jsonl", rows)
        res = analyze_clip(clip, road_dump=str(road_path), check_images=False)
        sp = res["highway_lane_spread"]
        assert sp["highway_rows"] == 20
        assert sp["classified_rows"] == 20
        assert sp["lane_index_counts"] == {"0": 4, "1": 16}
        assert abs(sp["rightmost_lane_share"] - 0.8) < 1e-6
        assert abs(sp["leftmost_lane_share"] - 0.2) < 1e-6
        assert abs(sp["norm_lane_pos_mean"] - 0.8) < 1e-6
        txt = format_highway_spread(sp)
        assert "rightmost=80.0%" in txt
        merged = merge_highway_spreads([sp, sp])
        assert merged["highway_rows"] == 40
        assert abs(merged["rightmost_lane_share"] - 0.8) < 1e-6
        assert "rightmost=80.0%" in format_highway_spread(merged)


def test_training_integration_ctl_and_scene_net() -> None:
    from python.train.clip_qc import apply_clip_qc_masks
    from python.train.ctl_targets import build_ctl
    from python.train.scene_net import labels_for_row

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_train_int"
        clip.mkdir()
        # 60 rows @ 5 Hz (0..11.8s), clean except signals at i=30..35 (y=60..70)
        rows = [_make_row(i, 100.0 + i * 0.2, 0.0, i * 2.0, steer=0.1, throttle=0.4, brake=0.0) for i in range(60)]
        _write_jsonl(clip / "state.jsonl", rows)
        sig_rows = [
            {
                "t": r["t"],
                "signals": {
                    "n1": {"n2": [{"action": 2.0, "instance": "trafficLight 1", "pos": {"x": 0.0, "y": 65.0, "z": 0.0}, "state": "redTrafficLight"}]}
                },
            }
            for r in rows
        ]
        _write_jsonl(clip / "signals.jsonl", sig_rows)

        qc = apply_clip_qc_masks(rows, rows, root=clip, warmup_s=0.0, settle_s=0.0, signal_radius_m=10.0)
        assert qc["masked_rows"] > 0
        # i=32 (y=64) is masked; i=2 (y=4) is clean (horizons 0..5s end at t=105.4 < signal at t=105.6)
        assert rows[32]["clip_qc_masked"] is True
        assert not rows[2].get("clip_qc_masked")

        targets, _counts = build_ctl(rows, rows, plan=None)
        # Masked row 32 has zero ctl_valid
        assert float(targets[32][1].sum()) == 0.0
        # Clean non-crossing row i=2: all future horizons stay valid (kills M7 if dirty ignores clip_qc)
        assert float(targets[2][1][:, 1:].min()) == 1.0
        # Row 25 (t=105.0) is clean now: h0 valid, h1 (t=106.0, i=30, inside signal) masked,
        # h3 (t=108.0, past the signal) still zeroed because the trajectory crossed the masked window.
        assert float(targets[25][1][:, 0].min()) == 1.0
        assert float(targets[25][1][:, 1].max()) == 0.0
        assert float(targets[25][1][:, 3].max()) == 0.0

        # scene_net driving targets masked, perception heads kept
        masked_labels = labels_for_row(rows[32])
        assert float(masked_labels["lane_gt_valid"].sum()) == 0.0
        assert float(masked_labels["driving_target_valid"]) == 0.0

        # torch scene_loss zero check if torch available
        try:
            import torch
            from python.train.scene_net import SceneNet, _lane_gt_parts
        except Exception:
            gvd_test_env.skip_check("torch unavailable for scene_net loss check")
            return
        net = SceneNet()
        x = torch.zeros(1, 3, net.canvas_h, net.canvas_w)
        pred, _ = net(x, torch.tensor([0.2]), net.initial_state())
        tgt = {k: torch.as_tensor(v) for k, v in masked_labels.items()}
        bce, reg = _lane_gt_parts(torch.nn.functional, pred, tgt, torch.device("cpu"), torch.float32)
        assert float(bce.detach()) == 0.0
        assert float(reg.detach()) == 0.0





def test_liveness_tipped_stationary_damage_and_settle_i0_none() -> None:
    from python.train.clip_qc import analyze_clip, analyze_rows

    # (a) tipped car (dir_z > 0.3) with non-zero speed, non-0-based i, and a damage spike
    tipped = []
    for k, i in enumerate(range(10, 20)):
        tipped.append(
            _make_row(
                i,
                100.0 + k * 0.2,
                0.0,
                k * 2.0,
                speed=5.0,
                dir_z=0.9,
                damage=(500.0 if i == 15 else 0.0),
                dt_s=0.2,
            )
        )
    tip_res = analyze_rows(tipped, warmup_s=0.0, settle_s=0.0)
    by_i = {r["i"]: r for r in tip_res["rows"]}
    assert "stuck" in by_i[10]["reasons"]
    assert "stuck" in by_i[19]["reasons"]
    assert "impact" in by_i[15]["reasons"]
    assert tip_res["reason_counts"]["stuck"] >= 9
    assert tip_res["reason_counts"]["impact"] >= 1

    # (b) >10 s stationary wait at a red light with dt_s is near_signal, NOT stuck
    wait = [
        _make_row(i, 100.0 + i * 0.2, 0.0, 60.0, speed=0.2, throttle=0.0, brake=0.2, dt_s=0.2)
        for i in range(65)
    ]
    wait_res = analyze_rows(
        wait,
        world_signals=[("traffic_light", 0.0, 60.0, 0.0)],
        has_signal_files=True,
        warmup_s=0.0,
        settle_s=0.0,
    )
    assert wait_res["reason_counts"]["near_signal"] == 65
    assert wait_res["reason_counts"]["stuck"] == 0
    assert "stuck" not in wait_res["rows"][40]["reasons"]
    assert "near_signal" in wait_res["rows"][40]["reasons"]

    # (c) the same >10 s stationary wait on an open road IS stuck
    open_road = [
        _make_row(i, 100.0 + i * 0.2, 0.0, i * 0.01, speed=0.2, throttle=0.0, brake=0.0, dt_s=0.2)
        for i in range(65)
    ]
    open_res = analyze_rows(open_road, warmup_s=0.0, settle_s=0.0)
    assert open_res["reason_counts"]["stuck"] == 65
    assert "stuck" in open_res["rows"][40]["reasons"]

    # (d) settle_windows with i0=None and t0 < first_t anchors to first_t (25 rows / 5.0 s)
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip_settle_none"
        clip.mkdir()
        rows = [_make_row(i, 100.0 + i * 0.2, 0.0, i * 2.0, dt_s=0.2) for i in range(40)]
        _write_jsonl(clip / "state.jsonl", rows)
        (clip / "collect.json").write_text(
            json.dumps({"settle_windows": [{"seq": 0, "source": "spawn", "t0": 98.0, "i0": None, "wait_s": 5.0}]}),
            encoding="utf-8",
        )
        res = analyze_clip(clip, check_images=False, warmup_s=0.0, settle_s=5.0)
        settle_rows = [r for r in res["rows"] if "settle_5s" in r["reasons"]]
        assert len(settle_rows) == 25, len(settle_rows)
        by_i = {r["i"]: r for r in res["rows"]}
        assert "settle_5s" in by_i[0]["reasons"]
        assert "settle_5s" in by_i[24]["reasons"]  # t=104.8
        assert "settle_5s" not in by_i[25]["reasons"]  # t=105.0


def test_train_directory_mask_clip_qc() -> None:
    from python.train.scene_net import torch_ready

    if not torch_ready():
        gvd_test_env.skip_check("torch unavailable for train_directory mask_clip_qc")
        return
    import contextlib
    import io

    from python.train.train_scene import train_directory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "scene"
        cam = root / "raw" / "main"
        cam.mkdir(parents=True)
        rows = []
        # 30 s @ 5 Hz so map_warmup covers [0, 20) and later rows stay eligible.
        # A stoplight at y=130 masks a mid-clip band after warmup.
        for i in range(150):
            t = 100.0 + i * 0.2
            p = f"raw/main/{i:06d}.png"
            _write_png(root / p)
            row = _make_row(i, t, 0.0, i * 2.0, speed=10.0, steer=0.1, throttle=0.3, brake=0.0, cam_path=p, dt_s=0.2)
            row["segment"] = "west_coast_usa#0"
            row["tech"]["lane_gt"] = [0.25, -1.5, 2.0] * 4 + [0.0, 1.0]
            row["tech"]["lane_gt_valid"] = [1.0] * 14
            rows.append(row)
        _write_jsonl(root / "state.jsonl", rows)
        _write_jsonl(
            root / "signals.jsonl",
            [
                {
                    "t": r["t"],
                    "signals": {
                        "n1": {
                            "n2": [
                                {
                                    "action": 2.0,
                                    "instance": "trafficLight 1",
                                    "pos": {"x": 0.0, "y": 130.0, "z": 0.0},
                                    "state": "redTrafficLight",
                                }
                            ]
                        }
                    },
                }
                for r in rows
            ],
        )
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            stats = train_directory(
                root,
                epochs=1,
                lr=1e-3,
                capability=None,
                export_path=None,
                free_vram_bytes=0,
                free_ram_bytes=8_000_000_000,
                resume=False,
                dry_run=True,
                open_panel=False,
                autostart=True,
                cams=("main",),
                heads=("lane_gt", "ctl"),
                mask_clip_qc=True,
                augment=False,
                workers=0,
                status_root=root / "_status",
            )
        text = log.getvalue()
        assert "[GVD] clip-qc:" in text, text[-3000:]
        assert "masked=" in text
        assert int(stats.get("dry_run") or 0) or stats.get("dry_run") is True


def test_cli_and_camera_only_runtime() -> None:
    from python.train.train_scene import build_parser

    args = build_parser().parse_args(["--strips", ".", "--mask-clip-qc"])
    assert args.mask_clip_qc is True
    assert args.clip_qc_drop is False

    # Runtime modules must stay camera-only (no import of clip_qc)
    for rel in ("python/run_vision.py",):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "clip_qc" not in src

    with tempfile.TemporaryDirectory() as td:
        sess = Path(td)
        c1 = sess / "clip_000_clear_west_coast_usa"
        c1.mkdir()
        rows = [_make_row(i, 100.0 + i * 0.2, 0.0, i * 2.0) for i in range(180)]
        _write_jsonl(c1 / "state.jsonl", rows)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/qc_clips.py"), str(sess), "--skip-images", "--write"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert (c1 / "clip_qc.json").is_file()
        assert (c1 / "clip_qc_mask.jsonl").is_file()
        assert "highway_spread:" in proc.stdout


def main() -> int:
    tests = [
        test_settle_5s_and_map_warmup_20s,
        test_near_signal_stoplight_stop_yield_and_dz,
        test_blinkers_excluded_from_targets,
        test_maneuvers_uturn_abrupt_stuck_and_overline_corner,
        test_clip_classification_train_pretrain_reject,
        test_highway_lane_position_spread_report,
        test_training_integration_ctl_and_scene_net,
        test_liveness_tipped_stationary_damage_and_settle_i0_none,
        test_train_directory_mask_clip_qc,
        test_cli_and_camera_only_runtime,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"[test_clip_qc] {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
