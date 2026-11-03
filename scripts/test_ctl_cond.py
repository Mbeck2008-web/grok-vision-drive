#!/usr/bin/env python3
"""B'1 control-head cond input and the Rev 5 ctl bars. Offline only.

* C1 cond: 5 floats from the t-dt row (``ego.speed_mps / 30``, ``steer_ai``,
  ``throttle_ai``, ``brake_ai``, ``ego.yaw_rate``), never the h=0 label row:
  perturbing row t's own controls leaves its cond unchanged. ``yaw_rate`` keeps
  the ``ego.yaw_rate`` sign (``imu.yaw_rate`` is never read). A row with no
  usable t-dt row (segment start, segment change, sim_t step over 0.25 s,
  a null or out-of-range field) gets zeros and ``ctl_cond_ok`` 0.
* C2 attach: ``attach_ctl(cond=True)`` writes ``ctl_cond`` / ``ctl_cond_ok``
  and voids every ctl target of a row without cond (``nocond=`` on the mask
  line); the default call writes no cond key and the same targets.
* C3 Rev 5 bars: ref = min(B3, B1 median); h3 steer and throttle <= 0.85 x ref;
  h4-h5 steer <= 0.95 x ref, throttle <= 0.85 x ref; B4 (hold own h0) and h0
  report-only; brake graded only with >= 30 events. Rev 4 stays the default.
* C4 net / ONNX / runtime (torch): the cond input moves the ctl outputs and
  only them; a cond net refuses a forward without cond; the ONNX has a
  ``cond`` [1, 5] input and ``gvd_ctl_cond`` metadata and tracks torch; the
  live runtime refuses a cond model; ``build_net`` rebuilds it from the meta.
* C5 CLI (torch): ``--ctl-cond`` is off by default, refused without the ctl
  head, and a tiny run with it writes a cond checkpoint whose ONNX eval matches
  the logged ctl val and prints the Rev 5 run line.
"""
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

from gvd_test_env import skip_check  # noqa: E402

SEG = "west_coast_usa#0"


def _row(i: int, sim_t: float, *, seg: str = SEG, steer: float = 0.1, thr: float = 0.3, brk: float = 0.0,
         speed: float = 12.0, yaw: float = 0.05, imu_yaw: float | None = None) -> dict:
    row = {
        "i": int(i),
        "t": float(sim_t),
        "dt_s": None if i == 0 else 0.1,
        "segment": seg,
        "ego": {"sim_t": float(sim_t), "speed_mps": speed, "yaw_rate": yaw},
        "wheel": {"steer_ai": steer, "steer": 0.0},
        "pedals": {"throttle_ai": thr, "brake_ai": brk, "clutch_ai": None, "parkingbrake_ai": 0.0},
        "veh": {"reverse": 0},
        "cams": {"main": {"file": f"raw/main/{i:06d}.png", "t": float(sim_t)}},
        "tech": {"speed_mps": speed},
    }
    if imu_yaw is not None:
        row["imu"] = {"yaw_rate": imu_yaw}
    return row


def _seq(n: int = 60) -> list[dict]:
    return [_row(i, round(0.1 * i, 4), steer=0.3 * math.sin(i / 5.0), thr=0.2 + 0.01 * (i % 30),
                 brk=0.5 if i % 11 == 0 else 0.0, speed=5.0 + 0.2 * i, yaw=0.02 * math.cos(i / 4.0))
            for i in range(n)]


def check_c1_cond() -> None:
    from python.train.ctl_targets import (
        COND_DIM,
        COND_FIELDS,
        COND_MAX_DT_S,
        COND_SPEED_NORM_MPS,
        build_cond,
        cond_from_row,
    )

    assert COND_DIM == 5 and len(COND_FIELDS) == 5, COND_FIELDS
    assert COND_SPEED_NORM_MPS == 30.0 and COND_MAX_DT_S == 0.25
    full = _seq()
    cond = build_cond(full, full)
    for k in range(1, len(full)):
        vec, ok = cond[k]
        prev = full[k - 1]
        want = [prev["ego"]["speed_mps"] / 30.0, prev["wheel"]["steer_ai"], prev["pedals"]["throttle_ai"],
                prev["pedals"]["brake_ai"], prev["ego"]["yaw_rate"]]
        assert ok == 1.0 and vec.shape == (5,) and vec.dtype == np.float32, (k, ok, vec)
        assert np.allclose(vec, want, atol=1e-6), (k, vec, want)
    # First row of the segment: no t-dt row, zeros and not ok.
    vec0, ok0 = cond[0]
    assert ok0 == 0.0 and not vec0.any(), (vec0, ok0)
    # The h=0 label row is never read: perturb row t's own controls, speed and yaw; its cond is unchanged.
    t = 30
    moved = [dict(row) for row in full]
    moved[t] = _row(t, full[t]["ego"]["sim_t"], steer=-0.9, thr=0.95, brk=0.9, speed=40.0, yaw=-1.0)
    got = build_cond(moved, moved)
    assert np.array_equal(got[t][0], cond[t][0]) and got[t][1] == 1.0, (got[t], cond[t])
    # ...while perturbing the t-dt row does change it (the check can fail).
    moved_prev = [dict(row) for row in full]
    moved_prev[t - 1] = _row(t - 1, full[t - 1]["ego"]["sim_t"], steer=-0.9, thr=0.95, brk=0.9, speed=40.0, yaw=-1.0)
    got_prev = build_cond(moved_prev, moved_prev)
    assert not np.array_equal(got_prev[t][0], cond[t][0])
    assert np.allclose(got_prev[t][0], [40.0 / 30.0, -0.9, 0.95, 0.9, -1.0], atol=1e-6), got_prev[t][0]
    # yaw_rate keeps the ego.yaw_rate sign; imu.yaw_rate (opposite sign on row 0 in the spec) is never read.
    signed = [_row(0, 0.0, yaw=0.3, imu_yaw=-0.3), _row(1, 0.1, yaw=-0.2, imu_yaw=0.2), _row(2, 0.2)]
    got = build_cond(signed, signed)
    assert abs(got[1][0][4] - 0.3) < 1e-6 and abs(got[2][0][4] + 0.2) < 1e-6, (got[1], got[2])
    assert cond_from_row(_row(0, 0.0, yaw=0.3, imu_yaw=-0.3))[4] > 0.0
    no_ego_yaw = _row(0, 0.0, imu_yaw=0.3)
    del no_ego_yaw["ego"]["yaw_rate"]
    assert cond_from_row(no_ego_yaw) is None, "no ego.yaw_rate: no cond (no fallback to imu)"
    # t-dt is the previous full-recording row: a row dropped from training still supplies it.
    kept = [row for row in full if row["i"] != 29]
    got = build_cond(full, kept)
    assert 29 not in got and np.array_equal(got[30][0], cond[30][0]) and got[30][1] == 1.0
    # Segment change, a step over 0.25 s, a null or out-of-range field: zeros and not ok.
    seg = [dict(row, segment=SEG if row["i"] < 20 else "west_coast_usa#1") for row in full]
    got = build_cond(seg, seg)
    assert got[20][1] == 0.0 and not got[20][0].any() and got[21][1] == 1.0
    gap = [row for row in full if row["i"] not in (40, 41)]
    got = build_cond(gap, gap)
    assert got[42][1] == 0.0 and not got[42][0].any(), got[42]
    near = [row for row in full if row["i"] != 40]  # 0.2 s step: within 0.25 s
    assert build_cond(near, near)[41][1] == 1.0
    for group, key, bad in (("wheel", "steer_ai", None), ("pedals", "throttle_ai", 1.5), ("ego", "speed_mps", float("nan")),
                            ("ego", "yaw_rate", None), ("pedals", "brake_ai", -0.2)):
        broken = [dict(row) for row in full]
        broken[9] = _row(9, full[9]["ego"]["sim_t"])
        broken[9][group][key] = bad
        got = build_cond(broken, broken)
        assert got[10][1] == 0.0 and not got[10][0].any(), (group, key, got[10])
    print("C1 cond: t-dt row only (h0 perturbation inert), speed/30, ego.yaw_rate sign, "
          "segment start/change, >0.25 s step and bad fields -> zeros + ok=0 OK")


def check_c2_attach() -> None:
    from python.train.ctl_targets import attach_ctl, mask_line

    full = _seq(80)
    plain = [dict(row) for row in full]
    with_cond = [dict(row) for row in full]
    counts_plain = attach_ctl(full, plain)
    counts_cond = attach_ctl(full, with_cond, cond=True)
    assert not any("ctl_cond" in row or "ctl_cond_ok" in row for row in plain)
    assert "nocond" not in counts_plain and "nocond=" not in mask_line(counts_plain)
    assert counts_cond["nocond"] == 1, counts_cond  # row 0 only
    assert mask_line(counts_cond).endswith(" nocond=1"), mask_line(counts_cond)
    assert mask_line(counts_plain) == mask_line({k: v for k, v in counts_cond.items() if k != "nocond"})
    for a, b in zip(plain, with_cond):
        assert np.asarray(b["ctl_cond"]).shape == (5,)
        if b["i"] == 0:
            assert b["ctl_cond_ok"] == 0.0 and not np.asarray(b["ctl_valid"]).any()
            continue
        assert b["ctl_cond_ok"] == 1.0
        assert np.array_equal(a["ctl"], b["ctl"]) and np.array_equal(a["ctl_valid"], b["ctl_valid"]), a["i"]
    print(f"C2 attach: cond keys only with cond=True, no-cond rows voided and counted | {mask_line(counts_cond)}")


def _good() -> dict:
    return {"n": 300, "steer_mae": 0.08, "b1med_steer": 0.1, "b3_steer": 0.2, "thr_mae": 0.08, "b1med_thr": 0.1,
            "b3_thr": 0.2, "brk_mae": 0.05, "b1med_brk": 0.06, "brk_f1": 0.6, "brk_events": 30,
            "b4_steer": 0.01, "b4_thr": 0.01, "b4_brk_f1": 0.99}


def check_c3_rev5_bars() -> None:
    from python.train.ctl_targets import (
        GRADED,
        GRADED_REV5,
        REV4,
        REV5,
        run_verdict,
        score_ctl,
        slice_verdict,
        val_lines,
    )

    assert GRADED == (0, 3, 4, 5) and GRADED_REV5 == (3, 4, 5)
    good = _good()
    # Rev 4 stays the default: B4 at 0.01 trips it, so Run B's FAIL reproduces.
    assert slice_verdict(good, 3) == "FAIL" and slice_verdict(good, 3, bar=REV4) == "FAIL"
    # Rev 5: B4 is report-only; ref = min(B3, B1 med) = B1 med here.
    for h in (3, 4, 5):
        assert slice_verdict(good, h, bar=REV5) == "PASS", h
    assert slice_verdict(good, 0, bar=REV5) == "REPORT"
    assert slice_verdict(dict(good, n=299), 3, bar=REV5) == "UNPROVEN"
    # ref picks the smaller baseline: B3 below B1 med tightens the bar.
    tight = dict(good, b3_steer=0.09)  # ref 0.09: 0.85 x = 0.0765 < 0.08
    assert slice_verdict(tight, 3, bar=REV5) == "FAIL"
    assert slice_verdict(tight, 4, bar=REV5) == "PASS"  # 0.95 x 0.09 = 0.0855 >= 0.08
    assert slice_verdict(dict(good, b3_thr=0.09), 4, bar=REV5) == "FAIL"  # throttle 0.85 at h4 too
    # h3 steer 0.85, h4/h5 steer 0.95, throttle 0.85 everywhere (at and just past each bar).
    assert slice_verdict(dict(good, steer_mae=0.085), 3, bar=REV5) == "PASS"
    assert slice_verdict(dict(good, steer_mae=0.0851), 3, bar=REV5) == "FAIL"
    for h in (4, 5):
        assert slice_verdict(dict(good, steer_mae=0.095), h, bar=REV5) == "PASS", h
        assert slice_verdict(dict(good, steer_mae=0.0951), h, bar=REV5) == "FAIL", h
        assert slice_verdict(dict(good, thr_mae=0.0851), h, bar=REV5) == "FAIL", h
    assert slice_verdict(dict(good, thr_mae=0.0851), 3, bar=REV5) == "FAIL"
    # Brake: graded at >= 30 events (F1 >= 0.5 and brk_mae < B1 med, no B4 term); below 30 ungraded.
    assert slice_verdict(dict(good, brk_f1=0.49), 3, bar=REV5) == "FAIL"
    assert slice_verdict(dict(good, brk_mae=0.06), 3, bar=REV5) == "FAIL"
    assert slice_verdict(dict(good, brk_f1=0.5, b4_brk_f1=0.99), 3, bar=REV5) == "PASS"
    assert slice_verdict(dict(good, brk_events=29, brk_f1=0.0, brk_mae=1.0), 3, bar=REV5) == "PASS"
    assert slice_verdict(dict(good, brk_events=29, steer_mae=0.09), 3, bar=REV5) == "FAIL"
    # Run: h3-h5 only; h0 is report-only.
    assert run_verdict({0: "FAIL", 3: "PASS", 4: "PASS", 5: "PASS"}, "PASS", bar=REV5) == "PASS"
    assert run_verdict({0: "PASS", 3: "PASS", 4: "PASS", 5: "PASS"}, "PASS") == "PASS"
    assert run_verdict({0: "FAIL", 3: "PASS", 4: "PASS", 5: "PASS"}, "PASS") == "FAIL"
    assert run_verdict({3: "PASS", 4: "UNPROVEN", 5: "PASS"}, "PASS", bar=REV5) == "UNPROVEN"
    assert run_verdict({3: "PASS", 4: "PASS", 5: "PASS"}, "UNPROVEN", bar=REV5) == "UNPROVEN"
    assert run_verdict({3: "PASS", 4: "PASS", 5: "FAIL"}, "PASS", bar=REV5) == "FAIL"

    # score_ctl: a model that holds the GT h0 label equals B3, so it FAILs Rev 5 at h3-h5;
    # one that is clearly better than both baselines PASSes, with brake ungraded (no events).
    rng = np.random.default_rng(73)

    def fixture(better: bool, brake: bool) -> list:
        out = []
        for k in range(320):
            row = _row(k, 0.1 * k)
            row["spawn_type"] = "city"
            x, thr = rng.uniform(-0.6, 0.6), rng.uniform(0.2, 0.8)
            drift = rng.uniform(-0.15, 0.15, size=(2, 6))
            drift[:, 0] = 0.0
            brk = np.full(6, 0.6 if brake and k % 5 == 0 else 0.0)
            row["ctl"] = np.stack([x + drift[0], thr + drift[1], brk]).astype(np.float32)
            row["ctl_valid"] = np.ones((3, 6), np.float32)
            pred = row["ctl"] + 0.002 if better else np.repeat(row["ctl"][:, :1], 6, axis=1)
            out.append((row, pred))
        out[0][0]["spawn_type"] = "highway"
        return out

    b1 = {"median": np.tile(np.array([[0.0], [0.5], [0.0]], np.float32), (1, 6)),
          "mean": np.tile(np.array([[0.0], [0.5], [0.1]], np.float32), (1, 6))}
    hold = score_ctl(fixture(False, False), b1, lane="PASS", bar=REV5)
    assert hold["bar"] == REV5
    for h in (3, 4, 5):
        m = hold["h"][h]
        assert abs(m["steer_mae"] - m["b3_steer"]) < 1e-12 and m["verdict"] == "FAIL", (h, m["verdict"])
        assert m["ref_steer"] == min(m["b3_steer"], m["b1med_steer"]), m
    assert hold["h"][0]["verdict"] == "REPORT" and hold["run"] == "FAIL"
    good_run = score_ctl(fixture(True, False), b1, lane="PASS", bar=REV5)
    for h in (3, 4, 5):
        assert good_run["h"][h]["brk_events"] == 0 and good_run["h"][h]["verdict_all"] == "PASS", h
    # The highway slice has 1 row: UNPROVEN caps the horizon, so the run is UNPROVEN, not PASS.
    assert good_run["run"] == "UNPROVEN", good_run["run"]
    lines = val_lines(good_run)
    assert lines[-1] == ("[GVD] val ctl run verdict=UNPROVEN bar=rev5 h0=REPORT h3=UNPROVEN h4=UNPROVEN "
                         "h5=UNPROVEN lane=PASS"), lines[-1]
    assert " ref_steer=" in lines[3] and " ref_thr=" in lines[3] and " brk_graded=0 " in lines[3], lines[3]
    from python.train.ctl_targets import ctl_record

    assert val_lines(ctl_record(good_run)) == lines
    # Rev 4 default: the same records give the Rev 4 lines, byte for byte without the Rev 5 fields.
    rev4 = val_lines(score_ctl(fixture(True, False), b1, lane="PASS"))
    assert re.fullmatch(r"\[GVD\] val ctl run verdict=\S+ h0=\S+ h3=\S+ h4=\S+ h5=\S+ lane=PASS", rev4[-1]), rev4[-1]
    assert " ref_steer=" not in rev4[3]
    print("C3 Rev 5 bars: ref=min(B3,B1med), h3 0.85/0.85, h4-h5 0.95/0.85, B4 and h0 report-only, "
          "brake ungraded under 30 events, GT-h0 hold FAILs, Rev 4 default unchanged OK")


def check_c4_net_onnx_runtime() -> None:
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        skip_check("c4_net_onnx_runtime", "torch / onnxruntime not installed")
        return
    from python.train.ctl_targets import COND_DIM, CTL_OUTPUT_FIELDS
    from python.train.epoch_ckpt import build_net
    from python.train.scene_lane_runtime import SceneLaneRuntime, _base_name
    from python.train.scene_net import RESIDUAL_S, SceneNet, checkpoint_arch, export_scene_onnx

    cams = ("main",)
    torch.manual_seed(73)
    net = SceneNet(cams=cams, ctl=True, cond_dim=COND_DIM).eval()
    assert net.pause_arch == checkpoint_arch(cams) + ":ctl=3x6:cond=5", net.pause_arch
    plain = SceneNet(cams=cams, ctl=True).eval()
    extra = sum(p.numel() for p in net.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 5 * 18, extra
    strip = torch.rand(1, 3, net.canvas_h, net.canvas_w)
    try:
        net.forward(strip, 0.1, net.initial_state())
    except ValueError as exc:
        assert "cond" in str(exc), exc
    else:
        raise AssertionError("a cond net ran without cond")
    with torch.no_grad():
        net.ctl_head.weight[:, 512:].normal_(0.0, 0.5)
        a, _ = net.forward(strip, 0.1, net.initial_state(), cond=torch.zeros(1, 5))
        b, _ = net.forward(strip, 0.1, net.initial_state(), cond=torch.tensor([[0.5, 0.3, 0.4, 0.2, 0.1]]))
    assert not torch.equal(a.ctl_steer, b.ctl_steer)
    assert torch.equal(a.lane_gt, b.lane_gt), "cond feeds the ctl head only"
    try:
        plain.forward(strip, 0.1, plain.initial_state(), cond=torch.zeros(1, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("a net without cond took a cond input")

    with tempfile.TemporaryDirectory() as tmp:
        path = export_scene_onnx(Path(tmp) / "cond.onnx", net, heads=("lane_gt", "ctl"))
        plain_path = export_scene_onnx(Path(tmp) / "plain.onnx", plain, heads=("lane_gt", "ctl"))
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        s_plain = ort.InferenceSession(str(plain_path), providers=["CPUExecutionProvider"])
        ins = {_base_name(i.name): i for i in sess.get_inputs()}
        assert list(ins["cond"].shape) == [1, 5], ins["cond"].shape
        assert "cond" not in {_base_name(i.name) for i in s_plain.get_inputs()}
        meta = dict(sess.get_modelmeta().custom_metadata_map)
        assert meta["gvd_ctl_cond"] == "5", meta
        assert meta["gvd_ctl_cond_fields"] == "speed_mps/30,steer_ai,throttle_ai,brake_ai,yaw_rate", meta
        assert meta["gvd_ctl_cond_source"] == "t-dt", meta
        assert "gvd_ctl_cond" not in dict(s_plain.get_modelmeta().custom_metadata_map)
        outs = [_base_name(o.name) for o in sess.get_outputs()]
        keys = [k for k in ("h", "tokens", "mask") if k in ins]
        state = {k: np.full([int(d) for d in ins[k].shape], float(RESIDUAL_S) + 1.0 if k == "mask" else 0.0,
                            dtype=np.float32) for k in keys}
        cond = np.asarray([[0.5, 0.3, 0.4, 0.2, 0.1]], np.float32)
        feeds = {"strip": strip.numpy(), "dt": np.asarray([0.1], np.float32), "cond": cond, **state}
        got = dict(zip(outs, sess.run(None, {ins[k].name: v for k, v in feeds.items()})))
        for name in CTL_OUTPUT_FIELDS:
            assert np.abs(got[name].reshape(-1) - getattr(b, name).numpy().reshape(-1)).max() < 1e-4, name
        # Live: the runtime does not feed cond, so a cond model is refused, never run with zeros.
        rt = SceneLaneRuntime(path)
        assert not rt.ready and rt.session is None and "cond" in rt.reason, rt.reason
        assert SceneLaneRuntime(plain_path).ready
    rebuilt = build_net({"cams": list(cams), "ctl": True, "ctl_cond_dim": 5})
    assert rebuilt.cond_dim == 5 and rebuilt.pause_arch == net.pause_arch
    assert build_net({"cams": list(cams), "ctl": True}).cond_dim == 0
    print("C4 net/ONNX/runtime: arch :cond=5, cond moves ctl only, forward needs cond, ONNX cond [1,5] + "
          "gvd_ctl_cond metadata tracks torch, live runtime refuses a cond model, build_net from meta OK")


def check_c5_cli_and_tiny_run() -> None:
    try:
        import onnxruntime  # noqa: F401
        import torch
    except ImportError:
        skip_check("c5_cli_and_tiny_run", "torch / onnxruntime not installed")
        return
    import export_epoch_onnx
    from eval_onnx_holdout import evaluate
    from python.train import train_scene
    from python.train.epoch_ckpt import load_epoch_ckpt
    from test_ctl_head import _first_val_block, _fixture, _listing, _main_code

    args = train_scene.build_parser().parse_args(["--strips", "x"])
    assert args.ctl_cond is False, "--ctl-cond is off by default"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        strips = tmp / "rec"
        strips.mkdir()
        before = _listing(tmp)
        code, text = _main_code(["--strips", str(strips), "--no-panel", "--autostart", "--dry-run",
                                 "--heads", "lane_gt", "--ctl-cond"])
        assert code == 2 and "--ctl-cond needs --heads lane_gt,ctl" in text, (code, text[-2000:])
        assert _listing(tmp) == before

    torch.manual_seed(73)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "data" / "rec"
        _fixture(root)
        lines = (root / "state.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        for k, row in enumerate(rows):
            row["ego"]["yaw_rate"] = 0.05 * math.sin(k / 7.0)
        (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        ckpts = tmp / "run" / "ckpts"
        code, text = _main_code([
            "--strips", str(root), "--cams", "main", "--heads", "lane_gt,ctl", "--ctl-cond", "--seed", "73",
            "--exact-epochs", "1", "--no-panel", "--autostart", "--workers", "0", "--min-valid", "0",
            "--holdout-blocks", "1", "--ckpt-dir", str(ckpts), "--pause-dir", str(tmp / "run" / "pause"),
        ])
        assert code == 0, (code, text[-3000:])
        mask = [line for line in text.splitlines() if line.startswith("[GVD] ctl mask:")]
        assert len(mask) == 1 and re.search(r" nocond=\d+$", mask[0]), mask
        assert "[GVD] ctl cond: dim=5 fields=speed_mps/30,steer_ai,throttle_ai,brake_ai,yaw_rate source=t-dt" in text, \
            text[-3000:]
        run_lines = re.findall(r"^\[GVD\] val ctl run verdict=\S+ bar=rev5 h0=REPORT h3=\S+ h4=\S+ h5=\S+ lane=\S+$",
                               text, re.M)
        assert run_lines, text[-3000:]
        _val, logged = _first_val_block(text)
        ep0 = sorted(ckpts.glob("*.ckpt_ep0_step*.npz"))
        assert len(ep0) == 1
        meta = load_epoch_ckpt(ep0[0])["meta"]
        assert meta["ctl"] is True and meta["ctl_cond_dim"] == 5 and meta["arch"].endswith(":ctl=3x6:cond=5"), meta
        out = tmp / "ep0.onnx"
        assert export_epoch_onnx.main(["--ckpt-dir", str(ckpts), "--epoch", "0", "--out", str(out)]) == 0
        result = evaluate(root, out, holdout_blocks=1, eval_chunk=16, expect_epoch=0)
        assert result["ctl"]["bar"] == "rev5", result["ctl"].get("bar")
        assert result["ctl"]["h"]["0"]["verdict"] == "REPORT"
        for h in ("0", "3"):
            got, want = result["ctl"]["h"][h], logged[h]
            assert got["n"] == int(want["n"]), (h, got["n"], want["n"])
            for key in ("steer_mae", "thr_mae", "brk_mae"):
                if want[key] == "na":
                    assert got[key] is None, (h, key)
                    continue
                assert abs(float(got[key]) - float(want[key])) < 1e-3, (h, key, got[key], want[key])
        print(mask[0])
        print(run_lines[0])
        print(f"C5 CLI/tiny run: --ctl-cond default off, refused without ctl, cond ckpt + ONNX eval "
              f"h0 steer_mae={result['ctl']['h']['0']['steer_mae']:.5f} logged={logged['0']['steer_mae']} OK")


def main() -> None:
    check_c1_cond()
    check_c2_attach()
    check_c3_rev5_bars()
    check_c4_net_onnx_runtime()
    check_c5_cli_and_tiny_run()
    print("ctl cond / Rev 5: C1-C5 OK")


if __name__ == "__main__":
    main()
