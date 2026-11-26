#!/usr/bin/env python3
"""Fail-first checks for gating the wide camera to near range (gvd/wide-near-range).

The wide camera stays in the rig, byte-identical. Its lane points feed the
planner only inside WIDE_NEAR_MAX_Y_M ahead. AEB on the planner path must not
miss any brake that origin/main 98af813 makes on the 85 recorded strips.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

BASELINE = ROOT / "tests" / "fixtures" / "aeb_baseline" / "main_98af813_255.json"
# sha256 of every camera dict on origin/main 98af813 (wide included, unchanged).
CFG_SHA256 = {
    "narrow": "96d34ff2ffecd1d529b6a7576a39efc2399ef49c6c373ed2e962cfb444e1cca6",
    "main": "11071b95f2ebffb24a40a41afe070a05b3a0bd759fad42e8ef26f81b82b42830",
    "wide": "7b3c813102e3ff22b93a984437834a0e5013103a75f4f611217c0c5c1fa7a530",
    "pillarL": "f66b4496e73685069a1fb791f138537eb366ca785f265b9604b7674a98aea2ac",
    "pillarR": "b92dc478e794fbcfbcd1be10ab3737ad032401af50e113838a5657827308e4c6",
    "repeatL": "f246aae8906ef9a4cccbdad08ebb4f6daff08090aa36895cb01cfd88ea6e1411",
    "repeatR": "5a8fc1098ff11a895e6ee8062cc6cb07ca36bc226517ce6ab4b9078491ceb879",
    "rear": "9602f3a60908ccca03194a17e9965fbf4ea196fd613b4599d6c8b2c68c6b0527",
}
# 22-case harness outcomes on origin/main (with wide), corrected main crop 504:664.
# replay_gt_run6_37 / run7_39 are labelled expect-brake but main itself gives "off".
MAIN_HARNESS_OFF_DESPITE_LABEL = ("replay_gt_run6_37_stop_sign", "replay_gt_run7_39_stop_sign")


def check_camera_config_unchanged() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cfg["cameras"]}
    assert list(by_id) == list(CFG_SHA256), list(by_id)
    for cid, want in CFG_SHA256.items():
        got = hashlib.sha256(json.dumps(by_id[cid], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert got == want, f"{cid} camera config changed"


def check_wide_points_near_range_only() -> None:
    from python.perception import lanes

    limit = float(lanes.WIDE_NEAR_MAX_Y_M)
    assert 5.0 <= limit <= 10.0, limit
    # 96 px wide sector rows, pinned as fixed distances (not WIDE_NEAR_MAX_Y_M):
    # row 49 is 8.4165 m and must be KEPT; row 48 is 9.4152 m and must be DROPPED.
    # Mutating the gate to 8.0 drops the kept row; mutating it to 10.0 keeps the
    # dropped row. Both must fail this check.
    KEEP_WIDE_ROW_Y_M = 8.4165
    DROP_WIDE_ROW_Y_M = 9.4152
    ROW_TOL_M = 0.001
    pose = lanes.camera_pose("wide")
    keep_hit = lanes.project_pose_pixel(pose, 84.0, 49.0, width=168, height=96)
    drop_hit = lanes.project_pose_pixel(pose, 84.0, 48.0, width=168, height=96)
    assert keep_hit is not None and abs(float(keep_hit["y"]) - KEEP_WIDE_ROW_Y_M) < ROW_TOL_M, keep_hit
    assert drop_hit is not None and abs(float(drop_hit["y"]) - DROP_WIDE_ROW_Y_M) < ROW_TOL_M, drop_hit
    seen_far_before = 0
    gated_keep = gated_drop = ungated_keep = ungated_drop = 0
    for p in sorted((ROOT / "tests" / "fixtures").rglob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (96, 1336) or "esbr" in p.name:
            continue
        img = img.copy()
        img[:28, :420] = (30, 30, 32)
        for poly in lanes.lanes_from_view(None, img).lanes_bev:
            for pt in poly:
                if pt.get("src") == "wide":
                    y = float(pt["y"])
                    assert y <= limit, (p.name, pt)
                    assert abs(float(pt["x"])) <= lanes.WIDE_NEAR_MAX_X_M, (p.name, pt)
                    if abs(y - KEEP_WIDE_ROW_Y_M) <= ROW_TOL_M:
                        gated_keep += 1
                    if abs(y - DROP_WIDE_ROW_Y_M) <= ROW_TOL_M:
                        gated_drop += 1
        # ungated fit on the same strip: far wide points do exist (the gate is doing work)
        for poly in lanes._estimate_span_lanes(
            np.ascontiguousarray(img[:, 336:840]), lanes.STITCH_LANE_FAR_M,
            ((0, 160), (168, 328), (336, 496)), gate_wide=False,
        ).lanes_bev:
            for pt in poly:
                if pt.get("src") != "wide":
                    continue
                y = float(pt["y"])
                if y > limit:
                    seen_far_before += 1
                if abs(y - KEEP_WIDE_ROW_Y_M) <= ROW_TOL_M:
                    ungated_keep += 1
                if abs(y - DROP_WIDE_ROW_Y_M) <= ROW_TOL_M:
                    ungated_drop += 1
    assert seen_far_before > 0, "fixture set has no far wide points; the gate is untested"
    assert ungated_keep > 0 and ungated_drop > 0, (
        f"fixtures must include the {KEEP_WIDE_ROW_Y_M} m and {DROP_WIDE_ROW_Y_M} m wide rows "
        f"(ungated keep={ungated_keep}, drop={ungated_drop})"
    )
    assert gated_keep > 0, f"{KEEP_WIDE_ROW_Y_M} m wide row must be KEPT (gated count {gated_keep})"
    assert gated_drop == 0, f"{DROP_WIDE_ROW_Y_M} m wide row must be DROPPED (gated count {gated_drop})"
    print(
        f"wide gate: 0 wide points past {limit} m (ungated fit had {seen_far_before}); "
        f"kept {KEEP_WIDE_ROW_Y_M} m x{gated_keep}, dropped {DROP_WIDE_ROW_Y_M} m"
    )


def _render_projected_road(cam_id: str, curve: float = 0.0, w: int = 256, h: int = 192) -> np.ndarray:
    from python.perception.lanes import camera_pose, pose_pixel_of_ground

    pose = camera_pose(cam_id)
    assert pose is not None, cam_id
    img = np.full((h, w, 3), 85, dtype=np.uint8)
    for base_x, col in ((-1.75, (0, 220, 240)), (1.75, (245, 245, 245))):
        pts = []
        for y in range(4, 46, 2):
            x = base_x + curve * float(y) * float(y)
            uv = pose_pixel_of_ground(pose, x, float(y), width=w, height=h)
            if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
                pts.append((int(round(uv[0])), int(round(uv[1]))))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 3)
    return img



def check_direct_cipv_and_fuse_byte_identical() -> None:
    from python.perception.cipv import CipvResult
    from python.perception.detect import Detection
    from python.perception.pipeline import ModularPerception, fuse_planner_cipv
    from python.perception.stitch360 import stitch_frames

    cases = [
        ("off", None, 10.0, "none", CipvResult({"id": 1}, 0.8, "brake"), ("brake", 0.8, 0.0)),
        ("off", None, 6.0, "vehicle", CipvResult({"id": 1}, 0.9, "brake"), ("brake", 0.9, 0.0)),
        ("brake", 0.4, 0.0, "vehicle", CipvResult({"id": 1}, 1.0, "brake"), ("brake", 0.4, 0.0)),
        ("warn", 2.0, 5.0, "vehicle", CipvResult({"id": 1}, 1.8, "warn"), ("warn", 2.0, 5.0)),
        ("off", None, 10.0, "none", CipvResult({"id": 1}, 1.9, "warn"), ("off", 1.9, 10.0)),
        ("off", None, 10.0, "none", CipvResult(None, None, "off"), ("off", None, 10.0)),
        (None, None, None, "none", CipvResult(None, None, None), (None, None, None)),
    ]
    for plan_aeb, plan_ttc, plan_tv, stop_r, cipv, want in cases:
        got = fuse_planner_cipv(
            plan_aeb=plan_aeb,
            plan_ttc=plan_ttc,
            plan_target_v=plan_tv,
            plan_stop_reason=stop_r,
            cipv=cipv,
        )
        assert got == want, (got, want)

    class _ClosingDet:
        name = "closing"

        def __init__(self) -> None:
            self.i = 0

        def detect(self, bgr):
            y = 14.0 - 1.0 * self.i
            self.i += 1
            return [Detection("vehicle", 0.95, (100, 100, 140, 140), x=0.0, y=y)]

    import python.perception.pipeline as pipe_mod

    blank = np.zeros((192, 256, 3), dtype=np.uint8)
    st = stitch_frames({"main": blank, "narrow": blank})

    def _run_with_fixed_time(stitch_bgr):
        orig_time = pipe_mod.time.time
        step = [1000.0]
        def _fake_time():
            step[0] += 0.01
            return step[0]
        pipe_mod.time.time = _fake_time
        try:
            p = ModularPerception(allow_synthetic=False, detector_id="empty")
            p.set_detector(_ClosingDet(), [])
            out = None
            for _ in range(12):
                out = p.tick(blank, stitch_bgr=stitch_bgr, ego_speed_mps=10.0, dt_s=0.1)
            return out
        finally:
            pipe_mod.time.time = orig_time

    o1 = _run_with_fixed_time(None)
    o2 = _run_with_fixed_time(st.bgr)
    assert o1 is not None and o2 is not None
    assert o1.planner["aeb"] == "brake" and o2.planner["aeb"] == "brake"
    assert json.dumps(o1.planner, sort_keys=True) == json.dumps(o2.planner, sort_keys=True)


def _load_main_from_gt_jpg(path: Path) -> np.ndarray:
    img = cv2.imread(str(path)).copy()
    assert img is not None, path
    img[:28, :420] = (30, 30, 32)
    return cv2.resize(img[:, 504:664], (256, 192))


def run_aeb_regression_suite(*, include_wide: bool = False) -> list[tuple[str, bool, str, str]]:
    from python.perception.cipv import select_cipv
    from python.perception.detect import Detection
    from python.perception.pipeline import ModularPerception, fuse_planner_cipv
    from python.perception.stitch360 import stitch_frames
    from python.planning.path_predictor import predict_path

    class _SeqDet:
        name = "seq"

        def __init__(self, seq):
            self._seq = list(seq)
            self._i = 0

        def detect(self, bgr):
            d = self._seq[min(self._i, len(self._seq) - 1)]
            self._i += 1
            return list(d)

    def _run_perc(main_img, wide_img, narrow_img, seq_dets, ego):
        p = ModularPerception(allow_synthetic=False, detector_id="empty")
        p.set_detector(_SeqDet(seq_dets), [])
        out = None
        for _ in range(len(seq_dets)):
            frames = {"main": main_img, "narrow": narrow_img}
            if include_wide and wide_img is not None:
                frames["wide"] = wide_img
            st = stitch_frames(frames)
            out = p.tick(main_img, stitch_bgr=st.bgr, ego_speed_mps=ego, dt_s=0.1)
        assert out is not None
        return str(out.planner["aeb"]), f"stop={out.planner['stop_reason']},src={out.lane_source}"

    straight_m = _render_projected_road("main", curve=0.0)
    straight_n = _render_projected_road("narrow", curve=0.0)
    curve_m = _render_projected_road("main", curve=-0.004)
    curve_n = _render_projected_road("narrow", curve=-0.004)
    straight_w = straight_m if include_wide else None
    curve_w = curve_m if include_wide else None
    blank = np.zeros((192, 256, 3), dtype=np.uint8)
    results: list[tuple[str, bool, str, str]] = []

    replay_cases = [
        ("replay_gt_run5_40_stop_sign", "tests/fixtures/gt_run5/frame_000040.jpg", [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.5, y=4.0)]] * 3, 8.0),
        ("replay_gt_run6_19_cipv", "tests/fixtures/gt_run6/frame_000019.jpg", [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=15.0 - 1.0 * i)] for i in range(12)], 10.0),
        ("replay_gt_run6_37_stop_sign", "tests/fixtures/gt_run6/frame_000037.jpg", [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.5, y=3.5)]] * 3, 8.0),
        ("replay_gt_run6_75_y_lt_4", "tests/fixtures/gt_run6/frame_000075.jpg", [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=28.0 - 1.0 * i)] for i in range(26)], 10.0),
        ("replay_gt_run7_39_stop_sign", "tests/fixtures/gt_run7/depart_000039.jpg", [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.5, y=4.0)]] * 3, 8.0),
        ("replay_gt_run8_31_cipv", "tests/fixtures/gt_run8/depart_000031.jpg", [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=14.0 - 1.0 * i)] for i in range(10)], 10.0),
        ("replay_gt_run9_107_stop_sign", "tests/fixtures/gt_run9/frame_000107.jpg", [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.5, y=4.0)]] * 3, 8.0),
        ("replay_gt_run3_cipv", "tests/fixtures/gt_run3_frame.jpg", [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=14.0 - 1.0 * i)] for i in range(10)], 10.0),
    ]
    for label, rel, dets, ego in replay_cases:
        m = _load_main_from_gt_jpg(ROOT / rel)
        aeb, info = _run_perc(m, m if include_wide else None, m, dets, ego)
        results.append((label, True, aeb, info))

    syn_cases = [
        ("syn_cipv_closing", straight_m, straight_w, straight_n, [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=15.0 - 1.0 * i)] for i in range(12)], 10.0, True),
        ("syn_y_lt_4_stopped_lead", straight_m, straight_w, straight_n, [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=28.0 - 1.0 * i)] for i in range(26)], 10.0, True),
        ("syn_stop_sign", straight_m, straight_w, straight_n, [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.4, y=4.0)]] * 3, 8.0, True),
        ("syn_curve_lead_perc", curve_m, curve_w, curve_n, [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=-0.004 * (15.0 - 1.0 * i) ** 2, y=15.0 - 1.0 * i)] for i in range(12)], 10.0, True),
        ("syn_blank_lanes_cipv_wins", blank, blank if include_wide else None, blank, [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=15.0 - 1.0 * i)] for i in range(12)], 10.0, True),
        ("neg_clear_road", straight_m, straight_w, straight_n, [[]] * 3, 10.0, False),
        ("neg_matched_rel_0_y8", straight_m, straight_w, straight_n, [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=8.0)]] * 3, 8.0, False),
        ("neg_adjacent_lane", straight_m, straight_w, straight_n, [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=5.5, y=8.0 - 1.0 * i)] for i in range(5)], 10.0, False),
        ("neg_far_stop_sign", straight_m, straight_w, straight_n, [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.5, y=30.0)]] * 3, 8.0, False),
    ]
    for label, m, w, n, dets, ego, expect_brake in syn_cases:
        aeb, info = _run_perc(m, w, n, dets, ego)
        results.append((label, expect_brake, aeb, info))

    curved_lanes = [
        [{"x": -1.75 - 0.008 * y * y, "y": float(y), "z": 0.0} for y in range(0, 42, 2)],
        [{"x": 1.75 - 0.008 * y * y, "y": float(y), "z": 0.0} for y in range(0, 42, 2)],
    ]
    straight_lanes = [
        [{"x": -1.75, "y": float(y), "z": 0.0} for y in range(0, 42, 2)],
        [{"x": 1.75, "y": float(y), "z": 0.0} for y in range(0, 42, 2)],
    ]
    for label, lanes_bev, tracks, signs, ego, expect_brake in [
        ("plan_lead_on_curve", curved_lanes, [{"id": 1, "class": "vehicle", "x": -0.008 * 64.0, "y": 8.0, "yaw": 1.57, "speed_mps": -8.0}], [], 10.0, True),
        ("plan_y_lt_4_stopped", straight_lanes, [{"id": 1, "class": "vehicle", "x": 0.0, "y": 3.0, "yaw": 1.57, "speed_mps": -8.0}], [], 8.0, True),
        ("plan_stop_sign", straight_lanes, [], [{"cls": "stop_sign", "x": 1.5, "y": 4.0}], 8.0, True),
        ("plan_stopped_lead_world_v0", straight_lanes, [{"id": 1, "class": "vehicle", "x": 0.0, "y": 5.0, "yaw": 1.57, "speed_mps": -10.0}], [], 10.0, True),
        ("neg_plan_rel_0_matched", straight_lanes, [{"id": 1, "class": "vehicle", "x": 0.0, "y": 8.0, "yaw": 1.57, "speed_mps": 0.0}], [], 8.0, False),
    ]:
        plan = predict_path(lanes_bev=lanes_bev, lane_conf=0.9, ego_speed_mps=ego, tracks=tracks, signs=signs)
        cipv = select_cipv(tracks, ego_speed_mps=ego)
        aeb, _ttc, _tv = fuse_planner_cipv(
            plan_aeb=plan.aeb,
            plan_ttc=plan.ttc_lead,
            plan_target_v=plan.target_v,
            plan_stop_reason=plan.stop_reason,
            cipv=cipv,
        )
        results.append((label, expect_brake, str(aeb), f"plan={plan.aeb},cipv={cipv.aeb},stop={plan.stop_reason}"))
    return results




def run_sweep() -> dict:
    from python.perception.detect import Detection
    from python.perception.pipeline import ModularPerception

    class D:
        name = "seq"

        def __init__(self, s):
            self.s, self.i = s, 0

        def detect(self, _b):
            d = self.s[min(self.i, len(self.s) - 1)]
            self.i += 1
            return list(d)

    out = {}
    for p in sorted((ROOT / "tests" / "fixtures").rglob("*.jpg")):
        if "esbr" in p.name:
            continue
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (96, 1336):
            continue
        img = img.copy()
        img[:28, :420] = (30, 30, 32)
        m = cv2.resize(img[:, 504:664], (256, 192))
        res = {}
        for kind, dets, ego in [
            ("stop_sign", [[Detection("stop_sign", 0.9, (100, 100, 140, 140), x=1.5, y=4.0)]] * 3, 8.0),
            ("cipv_close", [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=0.0, y=15.0 - 1.0 * i)] for i in range(12)], 10.0),
            ("curve_lead", [[Detection("vehicle", 0.9, (100, 100, 140, 140), x=-0.5, y=12.0 - 0.8 * i)] for i in range(10)], 10.0),
        ]:
            perc = ModularPerception(allow_synthetic=False, detector_id="empty")
            perc.set_detector(D(dets), [])
            for _ in dets:
                t = perc.tick(m, stitch_bgr=img, ego_speed_mps=ego, dt_s=0.1)
            res[kind] = [t.planner["aeb"], t.planner["stop_reason"], t.lane_source]
        out[str(p.relative_to(ROOT))] = res
    return out


def check_sweep_zero_missed_vs_main() -> None:
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    got = run_sweep()
    assert set(got) == set(base) and len(base) == 85, (len(got), len(base))
    missed, new = [], []
    mb = bb = 0
    for f in sorted(base):
        for k in base[f]:
            a, b = base[f][k][0], got[f][k][0]
            mb += a == "brake"
            bb += b == "brake"
            if a == "brake" and b != "brake":
                missed.append((f, k))
            if a != "brake" and b == "brake":
                new.append((f, k))
    print(f"sweep 255: main brakes {mb}, branch brakes {bb}, missed {len(missed)}, new {len(new)} {new}")
    assert not missed, f"MISSED BRAKES vs main: {missed}"
    assert new == [], f"NEW BRAKES vs main: {new}"
    record_diffs = []
    for f in sorted(base):
        for k in base[f]:
            ba, bs, bl = base[f][k]
            ga, gs, gl = got[f][k]
            if (ba, bs, bl) != (ga, gs, gl):
                record_diffs.append((f, k, (ba, bs, bl), (ga, gs, gl)))
    assert record_diffs == [], f"full-record diffs (aeb, stop_reason, lane_source): {record_diffs}"


def check_harness_vs_main() -> None:
    rows = run_aeb_regression_suite(include_wide=True)
    missed = [n for n, exp, aeb, _ in rows if exp and aeb != "brake" and n not in MAIN_HARNESS_OFF_DESPITE_LABEL]
    false_b = [n for n, exp, aeb, _ in rows if not exp and aeb == "brake"]
    label_gap = [n for n, exp, aeb, _ in rows if exp and aeb != "brake" and n in MAIN_HARNESS_OFF_DESPITE_LABEL]
    assert len(rows) == 22, len(rows)
    assert not missed, f"MISSED BRAKES: {missed}"
    assert not false_b, f"FALSE BRAKES: {false_b}"
    print(f"harness 22: 0 missed, 0 false brakes; label-vs-main gap (off on main too, label not edited): {label_gap}")


def main() -> None:
    for fn in (
        check_camera_config_unchanged,
        check_wide_points_near_range_only,
        check_direct_cipv_and_fuse_byte_identical,
        check_harness_vs_main,
        check_sweep_zero_missed_vs_main,
    ):
        fn()
        print(f"PASS: {fn.__name__}")


if __name__ == "__main__":
    main()
