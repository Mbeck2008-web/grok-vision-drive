#!/usr/bin/env python3
"""Optional control head (``--heads lane_gt,ctl``), offline only. Spec T1-T7.

* T1 targets: the future label is picked by ``ego.sim_t`` (nearest row within
  0.15 s), never by index; a 0.3 s hole at t+3 voids only h=3; a 1.2 s step at
  t+4 voids h=4 and h=5 and keeps h<=3; a segment change voids the window.
* T2 row validity: null / range / reverse / parking brake / warm-up, each
  counted once; a null clutch is ignored; the ``[GVD] ctl mask:`` line.
* T3 window and split: a settle row at t+2 voids h>=2 only; an impact window
  voids the horizons that reach it; a train row 4.5 s before a holdout slice
  has h=5 invalid (piece end); an invalid h never voids a shorter one.
* T4 loss: masked mean smooth-L1 (beta 0.1) per channel, summed; masked slots
  do not pull; ``ctl_weight`` scales it; a default loss has ``ctl`` = 0 and the
  same total; the lane-only branch still holds with ``ctl`` in the heads.
* T5 eval: lag stats, F1 with no predicted positive, n<300 and <30 brake
  events UNPROVEN, B2 skip past 0.25 s, horizon and run verdicts.
* T6 contract: arch tag, ONNX outputs and metadata only when on, default net
  and export unchanged, CLI refusals and create-only exits (code 2).
* T7 a tiny CPU run with 8 cameras, ``--cam-age`` and the ctl head writes
  ckpt_ep0, exports [1,6] ctl outputs, the offline ONNX eval matches the
  logged ctl val within 1e-3, and the run verdict line prints.
"""
# gvd-ci: requires torch
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

SEG = "west_coast_usa#0"


def _steer(sim_t: float) -> float:
    return float(0.5 * math.sin(sim_t * 1.3))


def _row(i: int, sim_t: float, *, t: float | None = None, seg: str = SEG, steer: float | None = None,
         thr: float | None = 0.3, brk: float | None = 0.0, **tags) -> dict:
    t = float(sim_t) if t is None else float(t)
    row = {
        "i": int(i),
        "t": t,
        "dt_s": None,
        "segment": seg,
        "ego": {"sim_t": float(sim_t), "speed_mps": 10.0},
        "wheel": {"steer_ai": _steer(sim_t) if steer is None else steer, "steer": 0.0},
        "pedals": {"throttle_ai": thr, "brake_ai": brk, "clutch_ai": None, "parkingbrake_ai": 0.0},
        "veh": {"reverse": 0},
        "cams": {"main": {"file": f"raw/main/{i:06d}.png", "t": t}},
    }
    row.update(tags)
    return row


def _irregular(n: int, start: float = 0.0, seed: int = 3) -> list[float]:
    rng = np.random.default_rng(seed)
    out, now = [], float(start)
    for _ in range(n):
        out.append(round(now, 4))
        now += float(rng.uniform(0.06, 0.14))
    return out


def _targets(full, kept=None, plan=None, impacts=()):
    from python.train.ctl_targets import build_ctl

    return build_ctl(full, full if kept is None else kept, plan=plan, impacts=impacts)


def check_t1_targets() -> None:
    from python.train.ctl_targets import CTL_HORIZONS_S, K

    assert CTL_HORIZONS_S == (0, 1, 2, 3, 4, 5) and K == 6
    times = _irregular(140)
    full = [_row(i, s) for i, s in enumerate(times)]
    targets, _counts = _targets(full)
    ctl, valid = targets[0]
    assert ctl.shape == (3, 6) and valid.shape == (3, 6), (ctl.shape, valid.shape)
    by_index_differs = False
    for h in range(6):
        want_j = min(range(len(times)), key=lambda j: abs(times[j] - h))
        assert abs(times[want_j] - h) <= 0.15
        assert valid[0, h] == 1.0, (h, valid[:, h])
        assert abs(ctl[0, h] - _steer(times[want_j])) < 1e-6, (h, ctl[0, h], _steer(times[want_j]))
        if h and abs(_steer(times[min(10 * h, len(times) - 1)]) - ctl[0, h]) > 1e-4:
            by_index_differs = True
        assert abs(ctl[1, h] - 0.3) < 1e-6 and ctl[2, h] == 0.0
    assert by_index_differs, "the fixture must make index picks differ from sim_t picks"
    # A 0.3 s hole at t+3: nothing within 0.15 s of t+3, so only h=3 is void.
    holed = [row for row in full if not (2.85 < row["ego"]["sim_t"] < 3.15)]
    targets, _ = _targets(holed)
    v = targets[0][1][0]
    assert list(v) == [1, 1, 1, 0, 1, 1], list(v)
    # A 1.2 s step at t+4 voids h=4 and h=5 and keeps h<=3.
    stepped = [row for row in full if not (3.5 < row["ego"]["sim_t"] < 4.7)]
    assert any(b["ego"]["sim_t"] - a["ego"]["sim_t"] > 1.0 for a, b in zip(stepped, stepped[1:]))
    targets, _ = _targets(stepped)
    v = targets[0][1][0]
    assert list(v) == [1, 1, 1, 1, 0, 0], list(v)
    # A segment change at t+2.5 voids h>=3; the h=0..2 labels stay.
    seg = [dict(row, segment=SEG if row["ego"]["sim_t"] < 2.5 else "west_coast_usa#1") for row in full]
    targets, _ = _targets(seg)
    v = targets[0][1][0]
    assert list(v) == [1, 1, 1, 0, 0, 0], list(v)
    # The segment end: the last rows have no h=5 partner.
    last = targets[len(seg) - 1][1][0]
    assert last[0] == 1 and not last[1:].any(), list(last)
    print("T1 targets: sim_t pick (not index), 0.3 s hole voids h=3 only, 1.2 s step voids h=4,5, segment end OK")


def check_t2_validity() -> None:
    from python.train.ctl_targets import mask_line

    times = [round(0.1 * k, 4) for k in range(80)]
    full = [_row(i, s) for i, s in enumerate(times)]
    full[5]["pedals"]["throttle_ai"] = None
    full[6]["wheel"]["steer_ai"] = 1.4
    full[7]["veh"]["reverse"] = 1
    full[8]["pedals"]["parkingbrake_ai"] = 0.9
    full[9]["pedals"]["brake_ai"] = float("nan")
    full[10]["pedals"]["clutch_ai"] = None  # never a target
    full[11]["pedals"]["throttle_ai"] = -0.01
    targets, counts = _targets(full)
    for i in (5, 6, 7, 8, 9, 11):
        assert not targets[i][1].any(), (i, targets[i][1])
    assert targets[10][1][:, 0].all(), targets[10][1]
    assert counts["null"] == 2 and counts["range"] == 2 and counts["reverse"] == 1 and counts["pbrake"] == 1, counts
    # Window: a future reverse / parking-brake row voids the horizons that reach it.
    assert list(targets[4][1][0][:2]) == [1, 0], targets[4][1][0]
    # Warm-up: a kept-row gap > 1.0 s (row t, not stale dt_s) clears the gap row and the next 15.
    kept = [row for row in full if not (30 <= row["i"] < 45)]
    for row in kept:
        row["dt_s"] = 0.1  # stale, as after mask_event_rows
    targets, counts = _targets(full, kept)
    gap_at = next(k for k, row in enumerate(kept) if row["i"] == 45)
    warm = [kept[gap_at + n]["i"] for n in range(16)]
    for i in warm:
        assert not targets[i][1].any(), i
    assert targets[kept[gap_at + 16]["i"]][1][:, 0].all()
    assert counts["warmup"] == 16, counts
    assert counts["rows"] == len(kept), counts
    line = mask_line(counts)
    assert re.fullmatch(
        r"\[GVD\] ctl mask: rows=\d+ valid=\d+ null=\d+ range=\d+ reverse=\d+ pbrake=\d+ warmup=\d+ "
        r"fut_h3=\d+ fut_h4=\d+ fut_h5=\d+",
        line,
    ), line
    print(f"T2 validity: null/range/reverse/pbrake, clutch ignored, warm-up 16 from kept t OK | {line}")


def check_t3_window_and_split() -> None:
    times = [round(0.1 * k, 4) for k in range(300)]
    full = [_row(i, s) for i, s in enumerate(times)]
    full[20 + 20]["settle"] = True  # t=4.0
    targets, _ = _targets(full)
    assert list(targets[20][1][0]) == [1, 1, 0, 0, 0, 0], targets[20][1][0]
    # recover and stall rows behave the same.
    rec = [dict(row) for row in full]
    rec[40] = dict(rec[40], settle=False, recover=True)
    targets, _ = _targets(rec)
    assert list(targets[20][1][0]) == [1, 1, 0, 0, 0, 0]
    # Impact at t=20 s: its window starts 3 s before (17.0); a row at 14.5 keeps h<=2.
    clean = [_row(i, s) for i, s in enumerate(times)]
    impacts = [{"kind": "impact", "t": 20.0, "segment": SEG}, {"kind": "impact_end", "t": 21.0, "segment": SEG}]
    targets, _ = _targets(clean, impacts=impacts)
    assert list(targets[145][1][0]) == [1, 1, 1, 0, 0, 0], targets[145][1][0]
    # Split: train piece [0, 15.0], holdout slice [19.0, 27.0]. A train row 4.5 s before the slice
    # (t=14.5) reaches past its piece end at h>=1; the slice keeps its own rows.
    train = [row for row in clean if row["t"] <= 15.0]
    hold = [row for row in clean if 19.0 <= row["t"] <= 27.0]
    kept = train + hold
    plan = [{"kind": "train", "rows": train[k:k + 16]} for k in range(0, len(train), 8)]
    plan.append({"kind": "holdout", "rows": hold})
    targets, counts = _targets(clean, kept, plan=plan)
    v = targets[145][1][0]
    assert v[0] == 1 and v[5] == 0, list(v)
    assert list(targets[100][1][0]) == [1, 1, 1, 1, 1, 1], targets[100][1][0]
    # The slice opens after a 4 s kept-row gap: its first 16 rows are warm-up.
    assert not targets[190][1].any() and not targets[205][1].any()
    hv = targets[210][1][0]
    assert list(hv) == [1, 1, 1, 1, 1, 1], list(hv)
    assert list(targets[250][1][0]) == [1, 1, 1, 0, 0, 0], targets[250][1][0]
    assert counts["fut_h5"] > 0
    # Rows outside every plan item are never trained or scored: no targets.
    assert not targets.get(170, (None, np.zeros((3, 6))))[1].any()
    print("T3 window/split: settle/recover/impact void reaching horizons only; piece end blocks leakage OK")


def check_t4_loss() -> None:
    import torch

    from python.train.scene_net import SceneNet, _ctl_loss, scene_loss

    torch.manual_seed(0)
    pred = torch.rand(4, 3, 6) * 0.8
    goal = torch.rand(4, 3, 6) * 0.8
    valid = (torch.rand(4, 3, 6) > 0.4).float()
    valid[:, 2, :] = 0.0  # brake all void: contributes 0
    got = float(_ctl_loss(pred, goal, valid))
    want = 0.0
    for c in range(3):
        m = valid[:, c].numpy() > 0.5
        if not m.any():
            continue
        d = np.abs(pred[:, c].numpy() - goal[:, c].numpy())[m]
        sl = np.where(d < 0.1, 0.5 * d * d / 0.1, d - 0.05)
        want += float(sl.mean())
    assert abs(got - want) < 1e-6, (got, want)
    moved = goal.clone()
    moved[valid < 0.5] = 99.0
    assert abs(float(_ctl_loss(pred, moved, valid)) - got) < 1e-6, "masked slots must not pull"
    nanned = goal.clone()
    nanned[valid < 0.5] = float("nan")
    assert abs(float(_ctl_loss(pred, nanned, valid)) - got) < 1e-6
    assert float(_ctl_loss(pred, goal, torch.zeros_like(valid))) == 0.0

    net = SceneNet(cams=("main",), ctl=True).eval()
    base = SceneNet(cams=("main",)).eval()
    state = net.initial_state()
    strip = torch.rand(1, 3, net.canvas_h, net.canvas_w)
    with torch.no_grad():
        p_ctl, _ = net.forward(strip, 0.1, state)
        p_base, _ = base.forward(strip, 0.1, base.initial_state())
    for name in ("ctl_steer", "ctl_throttle", "ctl_brake"):
        assert tuple(getattr(p_ctl, name).shape) == (6,), (name, getattr(p_ctl, name).shape)
        assert getattr(p_base, name) is None
    assert float(p_ctl.ctl_steer.abs().max()) <= 1.0 and float(p_ctl.ctl_brake.min()) >= 0.0
    from python.train.scene_net import labels_for_row

    row = _row(0, 0.0)
    row["tech"] = {}
    plain = labels_for_row(row)
    assert "ctl" not in plain and "ctl_valid" not in plain
    row["ctl"] = np.full((3, 6), 0.2, np.float32)
    row["ctl_valid"] = np.ones((3, 6), np.float32)
    labels = labels_for_row(row)
    target = {k: torch.as_tensor(v) for k, v in labels.items()}
    for key in ("object_class", "sign_class", "sign_state"):
        target[key] = target[key].long()
    off = scene_loss(p_ctl, target, ("lane_gt",))
    on1 = scene_loss(p_ctl, target, ("lane_gt", "ctl"))
    on3 = scene_loss(p_ctl, target, ("lane_gt", "ctl"), ctl_weight=3.0)
    assert float(off["ctl"]) == 0.0 and abs(float(off["total"]) - float(off["lane_gt"])) < 1e-6, off
    assert float(on1["ctl"]) > 0.0
    assert abs(float(on1["total"]) - float(on1["lane_gt"]) - float(on1["ctl"])) < 1e-5
    assert abs(float(on3["total"]) - float(on3["lane_gt"]) - 3.0 * float(on3["ctl"])) < 1e-5
    # Lane-only branch still taken with ctl on (scene_net fix): the polyline terms stay detached zeros.
    for key in ("points", "class", "lamps", "valid"):
        assert float(on1[key]) == 0.0 and not on1[key].requires_grad, key
    # Default run: no ctl target, "ctl" present and 0, the total is the old sum.
    default = scene_loss(p_base, target)
    assert float(default["ctl"]) == 0.0
    old = sum(float(default[k]) for k in ("points", "class", "lamps", "valid", "lane_gt"))
    assert abs(float(default["total"]) - old) < 1e-5
    print("T4 loss: masked-mean smooth-L1 per channel, masked slots inert, ctl_weight, default total unchanged OK")


def check_t5_eval() -> None:
    from python.train.ctl_targets import (
        b2_pairs,
        f1_score,
        lag_stats,
        lane_verdict,
        run_verdict,
        score_ctl,
        slice_verdict,
    )

    rows = [_row(i, 0.1 * i) for i in range(400)]
    for k, row in enumerate(rows):
        row["cams"]["main"]["t"] = row["t"] - 0.03
    lag = lag_stats(rows)
    assert lag["ok"] and abs(lag["median_ms"] - 30.0) < 1e-6, lag
    rows[3]["cams"]["main"]["t"] = rows[3]["t"] - 0.5
    rows[4]["cams"]["main"]["t"] = rows[4]["t"] - 0.5
    rows[5]["cams"]["main"]["t"] = rows[5]["t"] - 0.5
    rows[6]["cams"]["main"]["t"] = rows[6]["t"] - 0.5
    rows[7]["cams"]["main"]["t"] = rows[7]["t"] - 0.5
    assert not lag_stats(rows)["ok"], "p99 >= 100 ms must fail"

    assert f1_score(np.array([0.0, 0.0]), np.array([0.5, 0.0])) == 0.0
    assert abs(f1_score(np.array([0.5, 0.0, 0.5]), np.array([0.5, 0.0, 0.0])) - 2 / 3) < 1e-9

    # B2: previous kept row in the same segment within 0.25 s, else skipped.
    seq = [_row(0, 0.0), _row(1, 0.1), _row(2, 0.5), _row(3, 0.6, seg="other#0")]
    assert b2_pairs(seq) == [None, 0, None, None], b2_pairs(seq)

    good = {"n": 300, "steer_mae": 0.08, "b1med_steer": 0.1, "thr_mae": 0.08, "b1med_thr": 0.1,
            "brk_mae": 0.05, "b1med_brk": 0.06, "brk_f1": 0.6, "brk_events": 30,
            "b4_steer": 0.1, "b4_thr": 0.1, "b4_brk_f1": 0.5}
    assert slice_verdict(good, 3) == "PASS"
    assert slice_verdict(dict(good, n=299), 3) == "UNPROVEN"
    assert slice_verdict(dict(good, brk_events=29), 3) == "UNPROVEN"
    assert slice_verdict(dict(good, brk_events=29, steer_mae=0.09), 3) == "FAIL"
    assert slice_verdict(dict(good, steer_mae=0.086), 0) == "FAIL"  # > 0.85 x B1
    assert slice_verdict(dict(good, b4_steer=0.083), 3) == "FAIL"  # > 0.95 x B4 at h=3
    assert slice_verdict(dict(good, b4_steer=0.083), 0) == "PASS"  # no B4 bar at h=0
    assert slice_verdict(dict(good, brk_f1=0.49), 3) == "FAIL"
    assert slice_verdict(dict(good, brk_mae=0.06), 3) == "FAIL"
    assert slice_verdict(dict(good, brk_f1=0.5, b4_brk_f1=0.5), 3) == "FAIL"  # must beat B4
    assert run_verdict({0: "PASS", 3: "PASS", 4: "PASS", 5: "PASS"}, "UNPROVEN") == "PASS"
    assert run_verdict({0: "PASS", 3: "PASS", 4: "UNPROVEN", 5: "PASS"}, "PASS") == "UNPROVEN"
    assert run_verdict({0: "PASS", 3: "PASS", 4: "PASS", 5: "PASS"}, "FAIL") == "FAIL"
    assert run_verdict({0: "FAIL", 3: "UNPROVEN", 4: "UNPROVEN", 5: "UNPROVEN"}, "PASS") == "FAIL"
    assert lane_verdict(1.05, 1.0) == "PASS" and lane_verdict(1.06, 1.0) == "FAIL" and lane_verdict(1.0, None) == "UNPROVEN"

    # score_ctl on a small set: every slice is short, so UNPROVEN, and the line has every field.
    from python.train.ctl_targets import val_lines

    recs = []
    for k in range(40):
        row = _row(k, 0.1 * k, spawn_type="city" if k % 2 else "highway")
        row["ctl"] = np.tile(np.array([[0.1], [0.3], [0.2 if k % 3 == 0 else 0.0]], np.float32), (1, 6))
        row["ctl_valid"] = np.ones((3, 6), np.float32)
        recs.append((row, row["ctl"] + 0.01))
    b1 = {"median": np.zeros((3, 6), np.float32), "mean": np.zeros((3, 6), np.float32)}
    result = score_ctl(recs, b1, lock_deg=450.0)
    lines = val_lines(result)
    assert len(lines) == 7, lines
    for h, line in zip((0, 1, 2, 3, 4, 5), lines):
        assert line.startswith(f"[GVD] val ctl h={h} n=40 steer_mae="), line
        for key in ("steer_r", "thr_mae", "brk_mae", "brk_f1", "b1med_steer", "b1mean_steer", "b0_brk_f1",
                    "b2_steer", "b3_steer", "b3_brk_f1", "b4_steer", "b4_thr", "b4_brk_f1", "city_steer",
                    "hwy_steer", "ramp_steer"):
            assert f" {key}=" in line, (key, line)
        assert line.endswith("verdict=UNPROVEN"), line
    assert lines[-1] == "[GVD] val ctl run verdict=UNPROVEN h0=UNPROVEN h3=UNPROVEN h4=UNPROVEN h5=UNPROVEN lane=UNPROVEN", lines[-1]
    assert abs(result["h"][0]["steer_mae_deg"] - 0.01 * 450.0) < 1e-3, result["h"][0]
    # B4 fixture: every horizon column repeats the h=0 output. It clears the B1 bars but not the
    # B4 bars at h=3,4,5, so those horizons and the run FAIL; h=0 has no B4 bar and PASSes.
    rng = np.random.default_rng(7)
    recs = []
    for k in range(320):
        row = _row(k, 0.1 * k, spawn_type="city")
        x = rng.uniform(-0.8, 0.8)
        thr = rng.uniform(0.0, 1.0)
        brk = 0.6 if k % 5 == 0 else 0.0
        row["ctl"] = np.tile(np.array([[x], [thr], [brk]], np.float32), (1, 6))
        row["ctl_valid"] = np.ones((3, 6), np.float32)
        recs.append((row, row["ctl"].copy()))
    for k in range(3):  # ramp / highway slices exist in the record but are short (UNPROVEN)
        recs[k][0]["spawn_type"] = "highway"
    b1 = {"median": np.tile(np.array([[0.0], [0.5], [0.0]], np.float32), (1, 6)),
          "mean": np.tile(np.array([[0.0], [0.5], [0.1]], np.float32), (1, 6))}
    result = score_ctl(recs, b1)
    assert result["h"][0]["verdict_all"] == "PASS", result["h"][0]
    for h in (3, 4, 5):
        assert result["h"][h]["verdict"] == "FAIL", (h, result["h"][h]["verdict"])
    assert result["run"] == "FAIL"
    # B0 (all-zero brake) is rejected by the brake bar.
    assert result["h"][0]["b0_brk_f1"] == 0.0 < 0.5
    print("T5 eval: lag, F1=0 without positives, n/brake-event floors, B2 skip, verdicts, val lines OK")


def check_t6_contract() -> None:
    import torch

    from python.train.cam_strip import metadata_dict, parse_heads
    from python.train.scene_net import (
        CTL_FIELDS,
        CTL_HORIZONS_S,
        CTL_OUTPUT_FIELDS,
        HEAD_NAMES,
        OUTPUT_FIELDS,
        SceneNet,
        checkpoint_arch,
        export_scene_onnx,
    )

    assert CTL_FIELDS == ("steer_ai", "throttle_ai", "brake_ai")
    assert CTL_HORIZONS_S == (0, 1, 2, 3, 4, 5)
    assert CTL_OUTPUT_FIELDS == ("ctl_steer", "ctl_throttle", "ctl_brake")
    assert "ctl" not in HEAD_NAMES and "ctl_steer" not in OUTPUT_FIELDS
    assert parse_heads("lane_gt,ctl") == ("lane_gt", "ctl")
    assert parse_heads("") == HEAD_NAMES
    cams = ("main",)
    assert not hasattr(SceneNet(cams=cams), "pause_arch") or SceneNet(cams=cams).pause_arch == checkpoint_arch(cams)
    assert SceneNet(cams=cams, ctl=True).pause_arch == checkpoint_arch(cams) + ":ctl=3x6"
    assert SceneNet(cams=cams, ctl=True, cam_age=True).pause_arch == checkpoint_arch(cams) + ":cam_age=8:ctl=3x6"
    assert SceneNet(cams=cams, cam_age=True).pause_arch == checkpoint_arch(cams) + ":cam_age=8"
    plain = SceneNet(cams=cams)
    with_ctl = SceneNet(cams=cams, ctl=True)
    extra = sum(p.numel() for p in with_ctl.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 512 * 18 + 18, extra
    assert not any(name.startswith("ctl") for name in plain.state_dict())

    default_meta = metadata_dict(cams, ("lane_gt",), height=48, width=64)
    assert not any(key.startswith("gvd_ctl") for key in default_meta), default_meta
    meta = metadata_dict(cams, ("lane_gt", "ctl"), height=48, width=64)
    want = {
        "gvd_ctl": "1",
        "gvd_ctl_fields": "steer_ai,throttle_ai,brake_ai",
        "gvd_ctl_outputs": "ctl_steer,ctl_throttle,ctl_brake",
        "gvd_ctl_horizons_s": "0,1,2,3,4,5",
        "gvd_ctl_time_base": "sim_t",
        "gvd_ctl_units": "norm_-1_1,0_1,0_1",
    }
    for key, value in want.items():
        assert meta[key] == value, (key, meta.get(key))
    for key in default_meta:
        if key not in ("gvd_heads", "gvd_heads_trained"):
            assert meta[key] == default_meta[key], key

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        plain_path = export_scene_onnx(Path(tmp) / "plain.onnx", plain, heads=("lane_gt",))
        ctl_path = export_scene_onnx(Path(tmp) / "ctl.onnx", with_ctl, heads=("lane_gt", "ctl"))
        s_plain = ort.InferenceSession(str(plain_path), providers=["CPUExecutionProvider"])
        s_ctl = ort.InferenceSession(str(ctl_path), providers=["CPUExecutionProvider"])
        plain_out = [o.name for o in s_plain.get_outputs()]
        assert plain_out[: len(OUTPUT_FIELDS)] == list(OUTPUT_FIELDS) and not any("ctl" in n for n in plain_out), plain_out
        ctl_out = {o.name: o.shape for o in s_ctl.get_outputs()}
        for name in CTL_OUTPUT_FIELDS:
            assert list(ctl_out[name]) == [1, 6], (name, ctl_out.get(name))
        assert dict(s_ctl.get_modelmeta().custom_metadata_map)["gvd_ctl"] == "1"
        assert "gvd_ctl" not in dict(s_plain.get_modelmeta().custom_metadata_map)
        before = ctl_path.read_bytes()
        try:
            export_scene_onnx(ctl_path, with_ctl, heads=("lane_gt", "ctl"))
        except FileExistsError:
            pass
        else:
            raise AssertionError("a ctl export replaced an existing file")
        assert ctl_path.read_bytes() == before

    # Causality: outputs at t do not change when the frames after t change.
    torch.manual_seed(1)
    frames = torch.rand(30, 1, 3, with_ctl.canvas_h, with_ctl.canvas_w)
    with_ctl.eval()

    def run(seq: torch.Tensor) -> list[torch.Tensor]:
        state = with_ctl.initial_state()
        outs = []
        with torch.no_grad():
            for frame in seq:
                pred, state = with_ctl.forward(frame, 0.1, state)
                outs.append(torch.stack([pred.ctl_steer, pred.ctl_throttle, pred.ctl_brake]))
        return outs

    first = run(frames)
    changed = frames.clone()
    changed[16:] = torch.rand_like(changed[16:])
    second = run(changed)
    for t in range(16):
        assert torch.equal(first[t], second[t]), t
    assert not torch.equal(first[20], second[20])

    _check_cli_refusals()
    print("T6 contract: arch tag, outputs/metadata only when on, default unchanged, refusals exit 2 OK")


def _main_code(argv: list[str]) -> tuple[int | None, str]:
    from python.train import train_scene

    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            train_scene.main(argv)
            code = 0
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue()


def _check_cli_refusals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        strips = tmp / "rec"
        strips.mkdir()
        base = ["--strips", str(strips), "--no-panel", "--autostart", "--dry-run"]
        cases = [
            (["--heads", "ctl"], "ctl needs lane_gt"),
            (["--heads", "lane_gt,ctl", "--ctl-weight", "0"], "--ctl-weight"),
            (["--heads", "lane_gt,ctl", "--ctl-weight", "-1"], "--ctl-weight"),
            (["--heads", "lane_gt", "--ctl-weight", "2"], "--ctl-weight"),
            (["--heads", "lane_gt,ctl", "--backbone", "imagenet-resnet18"], "--backbone"),
            (["--heads", "lane_gt,ctl", "--export", str(tmp / "x.onnx")], "--export"),
        ]
        full = tmp / "ckfull"
        full.mkdir()
        (full / "old.npz").write_bytes(b"x")
        pfull = tmp / "pfull"
        pfull.mkdir()
        (pfull / "scene.gvd-pause.npz").write_bytes(b"x")
        cases.append((["--heads", "lane_gt,ctl", "--ckpt-dir", str(full)], "create-only"))
        cases.append((["--heads", "lane_gt,ctl", "--pause-dir", str(pfull)], "create-only"))
        for extra, needle in cases:
            code, text = _main_code(base + extra)
            assert code == 2, (extra, code, text[-2000:])
            assert needle in text, (needle, text[-2000:])
            assert not (tmp / "x.onnx").exists()
        cache = tmp / "cache"
        cache.mkdir()
        (cache / "keep.txt").write_text("keep", encoding="utf-8")
        code, text = _main_code(["--strips", str(strips), "--build-cache", str(cache)])
        assert code == 2 and "create-only" in text, (code, text[-2000:])
        assert (cache / "keep.txt").read_text(encoding="utf-8") == "keep"


N_ROWS = 160
from python.data.strip_writer import STRIP_ORDER as ALL8  # noqa: E402


def _lane(index: int) -> list[float]:
    center = 0.6 * math.sin(index / 9.0)
    values = []
    for _ in range(4):
        values += [center, center - 1.8, center + 1.8]
    return values + [0.0, 1.0]


def _fixture(root: Path) -> None:
    import cv2

    from python.debug.lane_targets import LANE_GT_DIM

    for cam in ALL8:
        (root / "raw" / cam).mkdir(parents=True)
    rng = np.random.default_rng(0)
    state_lines, gt_lines = [], []
    for index in range(N_ROWS):
        t = float(index) * 0.1
        sim = 100.0 + float(index) * 0.1 + float(rng.uniform(-0.02, 0.02))
        frame = (rng.random((24, 32, 3)) * 160 + 40).astype(np.uint8)
        col = int(16 + 10 * math.sin(index / 9.0))
        frame[:, max(0, col - 1): col + 1, :] = 250
        cams = {}
        for k, cam in enumerate(ALL8):
            if cam != "main" and (index + k) % 3:
                continue  # side cameras render every third tick: hold-last matters
            ok, buf = cv2.imencode(".png", frame if cam == "main" else np.ascontiguousarray(frame[:, ::-1]))
            assert ok
            (root / "raw" / cam / f"{index:06d}.png").write_bytes(buf.tobytes())
            cams[cam] = {"file": f"raw/{cam}/{index:06d}.png", "t": t - 0.02}
        state = {
            "i": index,
            "t": t,
            "dt_s": None if index == 0 else 0.1,
            "segment": SEG,
            "spawn_type": "city" if (index // 40) % 2 == 0 else "highway",
            "ego": {"sim_t": sim, "speed_mps": 10.0},
            "cams": cams,
            "wheel": {"steer": 0.1, "steer_ai": 0.4 * math.sin(sim / 2.0)},
            "pedals": {"throttle": 0.2, "brake": 0.0, "throttle_ai": 0.3 + 0.2 * math.cos(sim / 3.0),
                       "brake_ai": 0.4 if index % 25 < 4 else 0.0, "clutch_ai": None, "parkingbrake_ai": 0.0},
            "veh": {"reverse": 0},
        }
        if index == 0:
            state["veh_meta"] = {"model": "test", "steer_lock_deg": 450.0, "wheelbase_m": 2.7}
        state_lines.append(state)
        tech = {"lane_gt": _lane(index), "lane_gt_valid": [1.0] * LANE_GT_DIM, "speed_mps": 10.0}
        gt_lines.append({"i": index, "t": t, "tech": tech})
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_lines) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt_lines) + "\n", encoding="utf-8")


def check_t7_tiny_run() -> None:
    import onnxruntime as ort
    import torch

    from eval_onnx_holdout import evaluate
    from python.train.epoch_ckpt import export_epoch_onnx, load_epoch_ckpt
    from python.train.train_scene import train_directory

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        _fixture(root)
        ckpts = Path(tmp) / "ckpts"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            train_directory(
                root,
                epochs=1,
                lr=1e-3,
                capability=None,
                export_path=None,
                free_vram_bytes=0,
                free_ram_bytes=8_000_000_000,
                resume=False,
                open_panel=False,
                autostart=True,
                cams=ALL8,
                heads=("lane_gt", "ctl"),
                augment=False,
                workers=0,
                min_valid=0.0,
                holdout_blocks=1,
                cam_age=True,
                ckpt_dir=ckpts,
                status_root=Path(tmp) / "_status",
            )
        text = log.getvalue()
        mask = [line for line in text.splitlines() if line.startswith("[GVD] ctl mask:")]
        assert len(mask) == 1, text[-3000:]
        assert "[GVD] ctl lag: median_ms=" in text, text[-3000:]
        assert re.search(r"\[GVD\] ctl/lane_gt epoch=0 step=0 ", text), text[-3000:]
        for h in (0, 1, 2, 3, 4, 5):
            assert re.search(rf"^\[GVD\] val ctl h={h} n=\d+ steer_mae=", text, re.M), (h, text[-3000:])
        run_lines = re.findall(r"^\[GVD\] val ctl run verdict=\S+ h0=\S+ h3=\S+ h4=\S+ h5=\S+ lane=\S+$", text, re.M)
        assert run_lines, text[-3000:]
        print(mask[0])
        print(run_lines[-1])
        ep0 = sorted(ckpts.glob("*.ckpt_ep0_step*.npz"))
        assert len(ep0) == 1, sorted(p.name for p in ckpts.glob("*"))
        meta = load_epoch_ckpt(ep0[0])["meta"]
        assert meta["ctl"] is True and "ctl" in meta["heads"], meta
        status = Path(tmp) / "_status" / "metrics.jsonl"
        vals = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines() if line.strip()]
        vals = [rec for rec in vals if rec.get("kind") == "val"]
        rec = next(r for r in vals if r["epoch"] == 0)
        assert rec.get("ctl") and rec["ctl"]["h"]["0"]["n"] > 0, rec.get("ctl")
        out = Path(tmp) / "ep0.onnx"
        export_epoch_onnx(ep0[0], out)
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        shapes = {o.name: list(o.shape) for o in session.get_outputs()}
        for name in ("ctl_steer", "ctl_throttle", "ctl_brake"):
            assert shapes.get(name) == [1, 6], (name, shapes)
        assert dict(session.get_modelmeta().custom_metadata_map)["gvd_ctl"] == "1"
        result = evaluate(root, out, holdout_blocks=1, eval_chunk=16)
        assert result.get("ctl"), sorted(result)
        for h in ("0", "3", "4", "5"):
            got, want = result["ctl"]["h"][h], rec["ctl"]["h"][h]
            assert got["n"] == want["n"], (h, got["n"], want["n"])
            for key in ("steer_mae", "thr_mae", "brk_mae", "b4_steer"):
                if want[key] is None:
                    assert got[key] is None, (h, key)
                    continue
                assert abs(float(got[key]) - float(want[key])) < 1e-3, (h, key, got[key], want[key])
        assert abs(float(result["rms"]) - float(rec["rms"])) < 1e-3
        print(
            f"T7 tiny run: ckpt_ep0 + ONNX ctl [1,6]; onnx h0 steer_mae={result['ctl']['h']['0']['steer_mae']:.5f} "
            f"logged={rec['ctl']['h']['0']['steer_mae']:.5f} n={rec['ctl']['h']['0']['n']}"
        )


def main() -> None:
    check_t1_targets()
    check_t2_validity()
    check_t3_window_and_split()
    check_t4_loss()
    check_t5_eval()
    check_t6_contract()
    check_t7_tiny_run()
    print("ctl head: T1-T7 OK")


if __name__ == "__main__":
    main()
