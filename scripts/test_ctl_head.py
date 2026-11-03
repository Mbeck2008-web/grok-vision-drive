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
import hashlib
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
        "tech": {"speed_mps": 10.0},
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
    # Backward with --heads lane_gt,ctl: the polyline / curb / object / sign heads get no gradient
    # (None or all zero); lane_gt and ctl do. The same pass with every head shows the check can fail.
    def grads(heads):
        net.zero_grad(set_to_none=True)
        net.train()
        pred, _ = net.forward(strip, 0.1, net.initial_state())
        scene_loss(pred, target, heads)["total"].backward()
        net.eval()
        return {name: p.grad for name, p in net.named_parameters()}

    frozen = ("lane_head", "lane_valid_head", "curb_head", "curb_valid_head", "object_head", "object_class_head",
              "object_valid_head", "sign_head", "sign_class_head", "sign_state_head", "sign_valid_head")
    got_g = grads(("lane_gt", "ctl"))
    for name, g in got_g.items():
        if name.split(".")[0] in frozen:
            assert g is None or float(g.abs().max()) == 0.0, name
    assert any(g is not None and float(g.abs().max()) > 0 for n, g in got_g.items() if n.startswith("ctl_head."))
    all_g = grads(None)
    assert any(g is not None and float(g.abs().max()) > 0 for n, g in all_g.items() if n.split(".")[0] in frozen)
    net.zero_grad(set_to_none=True)
    # Default run: no ctl target, "ctl" present and 0, the total is the old sum.
    default = scene_loss(p_base, target)
    assert float(default["ctl"]) == 0.0
    old = sum(float(default[k]) for k in ("points", "class", "lamps", "valid", "lane_gt"))
    assert abs(float(default["total"]) - old) < 1e-5
    print("T4 loss: masked-mean smooth-L1 per channel, masked slots inert, ctl_weight, lane_gt,ctl backward leaves "
          "polyline/object/sign grads None or 0, default total unchanged OK")


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
    # PASS needs the lane bar PASS too: no paired lane run (UNPROVEN) caps the run at UNPROVEN.
    assert run_verdict({0: "PASS", 3: "PASS", 4: "PASS", 5: "PASS"}, "PASS") == "PASS"
    assert run_verdict({0: "PASS", 3: "PASS", 4: "PASS", 5: "PASS"}, "UNPROVEN") == "UNPROVEN"
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
    # B4 fixture: the labels move with h, and every horizon column repeats the model's h=0 output.
    # It clears the B1 bars but not the B4 bars at h=3,4,5, so those horizons and the run FAIL;
    # h=0 has no B4 bar and PASSes. The steer and throttle 0.95 x B4 bars trip on their own.
    def hold_fixture(brake: bool) -> list:
        rng = np.random.default_rng(7)
        out = []
        for k in range(320):
            row = _row(k, 0.1 * k, spawn_type="city")
            x = rng.uniform(-0.6, 0.6)
            thr = rng.uniform(0.2, 0.8)
            brk = 0.6 if brake and k % 5 == 0 else 0.0
            drift = rng.uniform(-0.15, 0.15, size=(2, 6))
            drift[:, 0] = 0.0
            row["ctl"] = np.stack([x + drift[0], thr + drift[1], np.full(6, brk)]).astype(np.float32)
            row["ctl_valid"] = np.ones((3, 6), np.float32)
            out.append((row, np.repeat(row["ctl"][:, :1], 6, axis=1)))
        for k in range(3):  # ramp / highway slices exist in the record but are short (UNPROVEN)
            out[k][0]["spawn_type"] = "highway"
        return out

    b1 = {"median": np.tile(np.array([[0.0], [0.5], [0.0]], np.float32), (1, 6)),
          "mean": np.tile(np.array([[0.0], [0.5], [0.1]], np.float32), (1, 6))}
    result = score_ctl(hold_fixture(True), b1, lane="PASS")
    assert result["h"][0]["verdict_all"] == "PASS", result["h"][0]
    for h in (3, 4, 5):
        m = result["h"][h]
        assert m["verdict"] == "FAIL", (h, m["verdict"])
        assert m["steer_mae"] <= 0.85 * m["b1med_steer"] and m["thr_mae"] <= 0.85 * m["b1med_thr"], m
        assert m["steer_mae"] > 0.95 * m["b4_steer"] and m["thr_mae"] > 0.95 * m["b4_thr"], m
        # Each B4 bar alone: brake neutral (fewer than 30 events), the other channel well clear.
        quiet = dict(m, brk_events=0)
        assert slice_verdict(dict(quiet, thr_mae=0.0), h) == "FAIL", "steer 0.95 x B4 alone"
        assert slice_verdict(dict(quiet, steer_mae=0.0), h) == "FAIL", "throttle 0.95 x B4 alone"
        assert slice_verdict(dict(quiet, steer_mae=0.0, thr_mae=0.0), h) == "UNPROVEN"
    assert result["run"] == "FAIL"
    # No brake events at all: brake is open (UNPROVEN), yet h=3,4,5 still FAIL on steer/throttle vs B4.
    quiet_result = score_ctl(hold_fixture(False), b1, lane="PASS")
    assert quiet_result["h"][0]["verdict_all"] == "UNPROVEN", quiet_result["h"][0]["verdict_all"]
    for h in (3, 4, 5):
        assert quiet_result["h"][h]["brk_events"] == 0
        assert quiet_result["h"][h]["verdict_all"] == "FAIL", (h, quiet_result["h"][h]["verdict_all"])
    assert quiet_result["run"] == "FAIL"
    # B0 (all-zero brake) is rejected by the brake bar.
    assert result["h"][0]["b0_brk_f1"] == 0.0 < 0.5

    # loss_mask: a row the trainer gives no loss (bad wheel/pedals, no tech) is not scored, and
    # B1 ignores it too. Wild predictions on those rows change nothing.
    from python.train.ctl_targets import b1_constants

    base = hold_fixture(True)
    dirty = [(dict(row), pred) for row, pred in base]
    for k in range(0, 320, 7):
        row, pred = dirty[k]
        if k % 2:
            row["wheel"] = dict(row["wheel"], steer=3.0)
        else:
            row["tech"] = {}
        dirty[k] = (row, pred + 0.7)
    clean = [(row, pred) for k, (row, pred) in enumerate(base) if k % 7]
    got_d, got_c = score_ctl(dirty, b1, lane="PASS"), score_ctl(clean, b1, lane="PASS")
    for h in range(6):
        for key in ("n", "steer_mae", "thr_mae", "brk_mae", "brk_f1", "b2_steer", "b3_steer", "b4_steer", "b2_n"):
            assert got_d["h"][h][key] == got_c["h"][h][key], (h, key, got_d["h"][h][key], got_c["h"][h][key])
    assert got_d["h"][0]["n"] == 320 - len(range(0, 320, 7))
    train_rows = [row for row, _p in dirty]
    for row in train_rows[::7]:
        row["ctl"] = np.full((3, 6), 0.9, np.float32)
    want_b1 = b1_constants([row for k, (row, _p) in enumerate(dirty) if k % 7])
    got_b1 = b1_constants(train_rows)
    assert np.array_equal(got_b1["median"], want_b1["median"]) and np.array_equal(got_b1["mean"], want_b1["mean"])

    # Lane bar from run A's eval JSON: rms <= 1.05 x A; anything missing or unreadable is UNPROVEN.
    from python.train.ctl_targets import lane_bar

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def ref(name: str, body) -> Path:
            path = tmp / name
            path.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
            return path

        from python.train.ctl_targets import holdout_rows_sha

        rows = [{"segment": "west_coast_usa#0", "i": 7}, {"segment": "west_coast_usa#0", "i": 8}, {"segment": "b", "i": 0}]
        want_sha = hashlib.sha256(b"west_coast_usa#0:7\nwest_coast_usa#0:8\nb:0\n").hexdigest()
        assert holdout_rows_sha(rows) == want_sha, holdout_rows_sha(rows)
        assert holdout_rows_sha(rows[::-1]) != want_sha  # ordered
        sha = want_sha
        a = {"rms": 0.40, "n": 500, "holdout_rows": 600, "epoch": 3, "rows_sha": sha}
        mine = {"rms": 0.40 * 1.04, "n": 500, "holdout_rows": 600, "epoch": 3, "rows_sha": sha}
        assert lane_bar(mine, ref("a.json", a))["verdict"] == "PASS"  # 1.04x
        assert lane_bar(dict(mine, rms=0.40 * 1.05), ref("a0.json", a))["verdict"] == "PASS"  # at the bar
        assert lane_bar(dict(mine, rms=0.40 * 1.06), ref("a1.json", a))["verdict"] == "FAIL"  # 1.06x
        bar = lane_bar(dict(mine, rms=0.43), ref("a2.json", a))
        assert bar["verdict"] == "FAIL" and abs(bar["ratio"] - 0.43 / 0.40) < 1e-9, bar
        # rows_sha decides the holdout match: n / holdout_rows alone are not compared.
        assert lane_bar(dict(mine, n=499), ref("a4.json", a))["verdict"] == "PASS"
        no_sha = {k: v for k, v in a.items() if k != "rows_sha"}
        no_epoch = {k: v for k, v in a.items() if k != "epoch"}
        for label, b_side in (("rows_sha", {k: v for k, v in mine.items() if k != "rows_sha"}),
                              ("epoch", {k: v for k, v in mine.items() if k != "epoch"}),
                              ("epoch", dict(mine, epoch=None))):
            got = lane_bar(b_side, ref(f"b_{label}.json", a))
            assert got["verdict"] == "UNPROVEN" and label in got["reason"], (label, got)
        for body, why in (
            (None, "no --lane-ref"),
            ("missing", "missing"),
            ("{not json", "unreadable"),
            ({"n": 500}, "no rms"),
            (dict(a, rms=None), "no rms"),
            (dict(a, rms="nan"), "no rms"),
            (dict(a, rows_sha="0" * 64), "rows_sha"),
            (no_sha, "rows_sha"),
            (dict(a, epoch=2), "epoch"),
            (no_epoch, "epoch"),
            (dict(a, epoch_ok=False), "expect-epoch"),
            (dict(a, ctl={"run": "PASS"}), "lane_gt-only"),
        ):
            if body is None:
                path = None
            elif body == "missing":
                path = tmp / "nope.json"
            else:
                path = ref(f"r{abs(hash(str(body)))}.json", body)
            got = lane_bar(mine, path)
            assert got["verdict"] == "UNPROVEN" and why in got["reason"], (body, got)
        assert lane_bar(dict(mine, rms=None), ref("a3.json", a))["verdict"] == "UNPROVEN"
    print("T5 eval: lag, F1=0 without positives, n/brake-event floors, B2 skip, verdicts (PASS needs lane PASS), "
          "h0-hold trips the steer/throttle B4 bars alone, loss_mask on scores and B1, lane bar vs run A OK")


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

    # Causality on a --cam-age ctl net (8 cameras): replace the frames, the rows (dt) and the ages
    # after t; the ctl outputs at every step <= t are bit-identical, in torch and in ONNX.
    from python.data.cam_hold import CAM_AGE_N

    torch.manual_seed(1)
    net8 = SceneNet(cams=ALL8, ctl=True, cam_age=True)
    net8.eval()
    steps, t_cut = 24, 12
    frames = torch.rand(steps, 1, 3, net8.canvas_h, net8.canvas_w)
    dts = (0.05 + 0.1 * torch.rand(steps)).tolist()
    ages = torch.rand(steps, 1, CAM_AGE_N) * 0.3
    changed_frames = frames.clone()
    changed_frames[t_cut + 1:] = torch.rand_like(changed_frames[t_cut + 1:])
    changed_dts = dts[: t_cut + 1] + [0.3] * (steps - t_cut - 1)
    changed_ages = ages.clone()
    changed_ages[t_cut + 1:] = 0.9

    def run_torch(seq: torch.Tensor, dt_seq: list, age_seq: torch.Tensor) -> list[torch.Tensor]:
        state = net8.initial_state()
        outs = []
        with torch.no_grad():
            for frame, dt, age in zip(seq, dt_seq, age_seq):
                pred, state = net8.forward(frame, dt, state, age)
                outs.append(torch.stack([pred.ctl_steer, pred.ctl_throttle, pred.ctl_brake]))
        return outs

    first = run_torch(frames, dts, ages)
    second = run_torch(changed_frames, changed_dts, changed_ages)
    for t in range(t_cut + 1):
        assert torch.equal(first[t], second[t]), t
    assert not torch.equal(first[t_cut + 4], second[t_cut + 4])

    from python.train.scene_lane_runtime import _base_name
    from python.train.scene_net import RESIDUAL_S

    with tempfile.TemporaryDirectory() as tmp:
        path8 = export_scene_onnx(Path(tmp) / "age_ctl.onnx", net8, heads=("lane_gt", "ctl"))
        sess = ort.InferenceSession(str(path8), providers=["CPUExecutionProvider"])
    ins = {_base_name(item.name): item for item in sess.get_inputs()}
    outs_names = [_base_name(item.name) for item in sess.get_outputs()]
    assert "ages" in ins, sorted(ins)
    keys = [key for key in ("h", "tokens", "mask") if key in ins]

    def run_onnx(seq: torch.Tensor, dt_seq: list, age_seq: torch.Tensor) -> list[np.ndarray]:
        state = {key: np.full([int(d) for d in ins[key].shape], float(RESIDUAL_S) + 1.0 if key == "mask" else 0.0,
                              dtype=np.float32) for key in keys}
        got = []
        for frame, dt, age in zip(seq, dt_seq, age_seq):
            feeds = {"strip": frame.numpy(), "dt": np.asarray([dt], np.float32), "ages": age.numpy(), **state}
            out = dict(zip(outs_names, sess.run(None, {ins[k].name: v for k, v in feeds.items()})))
            state = {key: out[key] for key in keys}
            got.append(np.stack([out[name].reshape(-1) for name in CTL_OUTPUT_FIELDS]))
        return got

    o_first = run_onnx(frames, dts, ages)
    o_second = run_onnx(changed_frames, changed_dts, changed_ages)
    for t in range(t_cut + 1):
        assert np.array_equal(o_first[t], o_second[t]), t
    assert not np.array_equal(o_first[t_cut + 4], o_second[t_cut + 4])
    # Same net, same inputs: ONNX tracks torch (not bit-equal across runtimes).
    assert max(float(np.abs(o_first[t] - first[t].numpy().reshape(3, -1)).max()) for t in range(steps)) < 1e-3

    _check_cli_refusals()
    print("T6 contract: arch tag, outputs/metadata only when on, default unchanged, cam-age causality bit-identical "
          "in torch and ONNX, refusals exit 2 and write nothing OK")


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


def _listing(root: Path) -> dict[str, tuple[int, str]]:
    """Every dir and file under ``root``: relative path -> (size, sha256); dirs are (-1, "")."""
    import hashlib

    out = {}
    for path in sorted(Path(root).rglob("*")):
        rel = path.relative_to(root).as_posix()
        out[rel] = (-1, "") if path.is_dir() else (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return out


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
        cache = tmp / "cache"
        cache.mkdir()
        (cache / "keep.txt").write_text("keep", encoding="utf-8")
        cases.append((["--build-cache", str(cache)], "create-only"))
        for extra, needle in cases:
            before = _listing(tmp)
            argv = ["--strips", str(strips)] + extra if extra[0] == "--build-cache" else base + extra
            code, text = _main_code(argv)
            assert code == 2, (extra, code, text[-2000:])
            assert needle in text, (needle, text[-2000:])
            after = _listing(tmp)
            assert after == before, (extra, sorted(set(after) ^ set(before)))
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


def _first_val_block(text: str) -> tuple[dict, dict]:
    """The epoch-0 holdout lines from stdout: the ``val step=`` line and its ``val ctl`` lines.

    Under ``main(argv)`` there is no status dir on a box without the data disk, so stdout is the record.
    """
    lines = text.splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith("[GVD] val step="))
    val = {key: value for key, value in re.findall(r"(\w+)=(\S+)", lines[start])}
    ctl: dict = {}
    for line in lines[start + 1:]:
        got = re.match(r"^\[GVD\] val ctl h=(\d) (.*)$", line)
        if got:
            ctl[got.group(1)] = dict(re.findall(r"(\w+)=(\S+)", got.group(2)))
            continue
        if line.startswith("[GVD] val ctl run "):
            ctl["run"] = dict(re.findall(r"(\w+)=(\S+)", line))
            break
    return val, ctl


def check_t7_tiny_run() -> None:
    import onnxruntime as ort
    import torch

    import export_epoch_onnx
    from eval_onnx_holdout import evaluate
    from eval_onnx_holdout import main as eval_main
    from python.train.epoch_ckpt import load_epoch_ckpt

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "data" / "rec"
        _fixture(root)
        run = tmp / "run"
        ckpts, pause = run / "ckpts", run / "pause"
        assert not run.exists()
        # The real entry point, clean dirs. --exact-epochs 1: epoch 0 only, then stop
        # (plain --epochs is logged, not a cap).
        code, text = _main_code([
            "--strips", str(root), "--cams", ",".join(ALL8), "--cam-age", "--heads", "lane_gt,ctl",
            "--exact-epochs", "1", "--no-panel", "--autostart", "--workers", "0", "--min-valid", "0",
            "--holdout-blocks", "1", "--ckpt-dir", str(ckpts), "--pause-dir", str(pause),
        ])
        assert code == 0, (code, text[-3000:])
        assert "[GVD] exact epochs: stopped after epoch=0 (early stop off)" in text, text[-3000:]
        assert "[GVD] holdout stopped" not in text
        mask = [line for line in text.splitlines() if line.startswith("[GVD] ctl mask:")]
        assert len(mask) == 1, text[-3000:]
        assert "[GVD] ctl lag: median_ms=" in text, text[-3000:]
        assert re.search(r"\[GVD\] ctl/lane_gt epoch=0 step=0 ", text), text[-3000:]
        run_lines = re.findall(r"^\[GVD\] val ctl run verdict=\S+ h0=\S+ h3=\S+ h4=\S+ h5=\S+ lane=\S+$", text, re.M)
        assert run_lines and all(line.endswith("lane=UNPROVEN") for line in run_lines), run_lines
        assert not any("verdict=PASS" in line for line in run_lines), run_lines
        print(mask[0])
        print(run_lines[0])
        val, logged = _first_val_block(text)
        assert {"0", "1", "2", "3", "4", "5"} <= set(logged), sorted(logged)
        assert int(logged["0"]["n"]) > 0, logged.get("0")  # h4/h5 can be n=0 on this short holdout
        ep0 = sorted(ckpts.glob("*.ckpt_ep0_step*.npz"))
        assert len(ep0) == 1 and len(list(ckpts.glob("*.npz"))) == 1, sorted(p.name for p in ckpts.glob("*"))
        meta = load_epoch_ckpt(ep0[0])["meta"]
        assert meta["ctl"] is True and "ctl" in meta["heads"] and meta.get("cam_age") is True, meta
        out = tmp / "ep0.onnx"
        # Missing epoch: export refuses --epoch 1 (only ckpt_ep0 exists), nothing written.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                missing = export_epoch_onnx.main(["--ckpt-dir", str(ckpts), "--epoch", "1", "--out", str(tmp / "ep1.onnx")])
            except SystemExit as exc:
                missing = exc.code
        assert missing not in (0, None) and not (tmp / "ep1.onnx").exists(), missing
        assert export_epoch_onnx.main(["--ckpt-dir", str(ckpts), "--epoch", "0", "--out", str(out)]) == 0
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        shapes = {o.name: list(o.shape) for o in session.get_outputs()}
        for name in ("ctl_steer", "ctl_throttle", "ctl_brake"):
            assert shapes.get(name) == [1, 6], (name, shapes)
        meta_map = dict(session.get_modelmeta().custom_metadata_map)
        assert meta_map["gvd_ctl"] == "1" and meta_map.get("gvd_cam_age") == "1", meta_map

        # Lane bar, no usable run-A file: UNPROVEN, so the run is at best UNPROVEN.
        result = evaluate(root, out, holdout_blocks=1, eval_chunk=16, lane_ref=tmp / "no_run_a.json", expect_epoch=0)
        assert result.get("ctl"), sorted(result)
        assert re.fullmatch(r"[0-9a-f]{64}", str(result.get("rows_sha"))), result.get("rows_sha")
        assert result["epoch"] == 0 and result["epoch_ok"] is True, (result["epoch"], result.get("epoch_ok"))
        assert result["lane_bar"]["verdict"] == "UNPROVEN" and result["ctl"]["lane"] == "UNPROVEN", result["lane_bar"]
        assert result["ctl"]["run"] != "PASS"
        for h in ("0", "3", "4", "5"):
            got, want = result["ctl"]["h"][h], logged[h]
            assert got["n"] == int(want["n"]), (h, got["n"], want["n"])
            for key in ("steer_mae", "thr_mae", "brk_mae", "b4_steer"):
                if want[key] == "na":
                    assert got[key] is None, (h, key)
                    continue
                assert abs(float(got[key]) - float(want[key])) < 1e-3, (h, key, got[key], want[key])
        assert abs(float(result["rms"]) - float(val["rms"])) < 1e-3, (result["rms"], val["rms"])
        assert result["n"] == int(val["n"])

        # Run A's eval JSON (lane_gt-only: no ctl block) on the same holdout and epoch.
        run_a = {key: result[key] for key in ("rms", "n", "holdout_rows", "epoch", "step", "rows_sha")}
        a_pass = tmp / "a_pass.json"
        a_pass.write_text(json.dumps(run_a), encoding="utf-8")
        b_json = tmp / "b_eval.json"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            assert eval_main(["--strips", str(root), "--onnx", str(out), "--holdout-blocks", "1",
                              "--expect-epoch", "0", "--lane-ref", str(a_pass), "--json", str(b_json)]) == 0
        printed = log.getvalue()
        assert re.search(r"^\[GVD\] val \(onnx\) lane bar verdict=PASS ", printed, re.M), printed[-2000:]
        saved = json.loads(b_json.read_text(encoding="utf-8"))
        assert saved["rows_sha"] == result["rows_sha"] and saved["epoch_ok"] is True, saved.get("rows_sha")
        assert saved["lane_bar"]["verdict"] == "PASS" and saved["ctl"]["lane"] == "PASS", saved["lane_bar"]
        assert re.search(r"^\[GVD\] val \(onnx\) ctl run verdict=\S+ .* lane=PASS$", printed, re.M), printed[-2000:]
        # A twice as good as B: lane FAIL, and the run FAILs with it.
        a_fail = tmp / "a_fail.json"
        a_fail.write_text(json.dumps(dict(run_a, rms=float(result["rms"]) / 2.0)), encoding="utf-8")
        failed = evaluate(root, out, holdout_blocks=1, eval_chunk=16, lane_ref=a_fail)
        assert failed["lane_bar"]["verdict"] == "FAIL" and failed["ctl"]["run"] == "FAIL", failed["lane_bar"]
        # --expect-epoch: a different epoch, or an ONNX with no gvd_ckpt_epoch, makes the
        # ctl run and the lane bar UNPROVEN (even against a lane ref that would FAIL or PASS).
        import onnx

        for lane_file in (a_pass, a_fail):
            wrong = evaluate(root, out, holdout_blocks=1, eval_chunk=16, lane_ref=lane_file, expect_epoch=3)
            assert wrong["epoch_ok"] is False, wrong.get("epoch_ok")
            assert wrong["lane_bar"]["verdict"] == "UNPROVEN" and wrong["ctl"]["run"] == "UNPROVEN", wrong["lane_bar"]
            assert "expect-epoch" in wrong["lane_bar"]["reason"], wrong["lane_bar"]
        model = onnx.load(str(out))
        kept = [p for p in model.metadata_props if p.key != "gvd_ckpt_epoch"]
        del model.metadata_props[:]
        model.metadata_props.extend(kept)
        no_epoch = tmp / "ep0_no_epoch.onnx"
        onnx.save(model, str(no_epoch))
        bare = evaluate(root, no_epoch, holdout_blocks=1, eval_chunk=16, lane_ref=a_pass, expect_epoch=0)
        assert bare["epoch"] is None and bare["epoch_ok"] is False, (bare["epoch"], bare.get("epoch_ok"))
        assert bare["lane_bar"]["verdict"] == "UNPROVEN" and bare["ctl"]["run"] == "UNPROVEN", bare["lane_bar"]
        # Without --expect-epoch that ONNX still has no epoch: the lane bar is UNPROVEN (epoch missing).
        loose = evaluate(root, no_epoch, holdout_blocks=1, eval_chunk=16, lane_ref=a_pass)
        assert loose["epoch_ok"] is None and loose["lane_bar"]["verdict"] == "UNPROVEN", loose["lane_bar"]
        print(
            f"T7 tiny run (main argv, 8 cams, --cam-age): ckpt_ep0 + ONNX ctl [1,6]; "
            f"onnx h0 steer_mae={result['ctl']['h']['0']['steer_mae']:.5f} logged={logged['0']['steer_mae']} "
            f"n={logged['0']['n']}; lane bar missing->UNPROVEN, equal->PASS, A/2->FAIL; "
            f"--expect-epoch 3 / no gvd_ckpt_epoch -> UNPROVEN; export --epoch 1 refused"
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
