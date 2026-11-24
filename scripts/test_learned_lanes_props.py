#!/usr/bin/env python3
"""Learned lane head in the live lane path, Hough as fallback: property + fuzz tests.

Invariants (python/perception/lane_select.py):

I1  No paint, no lines. Blank, uniform, sky, per-pixel noise and edge-only
    frames give zero painted lanes in the FINAL output, learned or Hough
    fallback, even when the learned head hallucinates confident lanes.
I2  Edges are not paint. Learned curbs never enter lanes_bev (they ride in
    curbs_bev), and an edge-only scene (grass, dirt, bright sidewalk, curb
    bands wider than PAINT_MAX_WIDTH_PX) yields no lanes.
I3  Never silent. Missing model, refused model, low confidence, NaN/inf
    output, bad shape, untrained head, runtime error, dropped frame: each
    falls back with an explicit reason in the telemetry.
I4  Finite and bounded. Every emitted point is finite and inside the ego
    box; conf in [0, 1]; curvature finite and clipped; any size of frame
    (empty, 1 px, 4K) is handled.
I5  Deterministic for the same inputs and model state.

Counterexamples on main (3e4a577), before this change (hypothesis, shrunk):
  Hough noise:      lanes_from_view(noise(h=96, w=128, seed=0, lo=0)) -> 2 lanes, conf 0.95
  Hough edge-only:  edge_only(h=96, w=128, side='grass', curb_v=185, curb_px=12, horizon=0.5,
                    half_bottom=0.5, half_top=0.125, shift=0.125) -> 1 lane, conf 0.4
  naive learned wiring (scene_to_predict_kwargs): curb_valid=[0.0, 1.0] -> 1 curb in lanes_bev;
                    lanes[0, 3, 0]=nan with lane_valid 0.9 -> {'x': nan, ...} in lanes_bev

hypothesis is a dev-only dependency (CI installs it). Without it the property
checks are skipped visibly; the example-based checks still run.

    py -3 scripts/test_learned_lanes_props.py
"""
from __future__ import annotations

import io
import os
import json
import math
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402
from lane_fuzz_frames import edge_only, noise, painted, sky_over_asphalt, uniform  # noqa: E402
from python.perception import lane_select as ls  # noqa: E402
from python.perception.lanes import LaneResult  # noqa: E402

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError:  # dev-only dependency
    given = None

NO_PAINT_FAMILIES = ("uniform", "noise", "sky", "edge")
REASONS = {
    ls.REASON_BAD_SHAPE, ls.REASON_NONFINITE, ls.REASON_OUT_OF_RANGE, ls.REASON_LOW_CONF,
}


# ---------------------------------------------------------------- fakes


def lane_poly(x: float, n: int = ls.LANE_POINTS, y0: float = 2.0, y1: float = 40.0) -> np.ndarray:
    ys = np.linspace(y0, y1, n)
    return np.stack([np.full(n, float(x)), ys], axis=-1)


def head_outputs(lane_xs=(-1.75, 1.75), lane_p=0.95, curb_xs=(), curb_p=0.95) -> dict:
    lanes = np.zeros((ls.N_LANES, ls.LANE_POINTS, 2), dtype=np.float32)
    valid = np.zeros(ls.N_LANES, dtype=np.float32)
    for i, x in enumerate(lane_xs):
        lanes[i] = lane_poly(x)
        valid[i] = lane_p
    curbs = np.zeros((ls.N_CURBS, ls.LANE_POINTS, 2), dtype=np.float32)
    cvalid = np.zeros(ls.N_CURBS, dtype=np.float32)
    for i, x in enumerate(curb_xs):
        curbs[i] = lane_poly(x)
        cvalid[i] = curb_p
    return {"lanes": lanes, "lane_valid": valid, "curbs": curbs, "curb_valid": cvalid}


class FakeRuntime:
    """Duck-typed SceneLaneRuntime: fixed head outputs, no ONNX."""

    def __init__(self, outputs=None, *, head_conf=0.9, lane_x=0.0, raise_on_observe=None, untrained=()):
        self.ready = True
        self.reason = "ok"
        self.path = "fake.onnx"
        self.outputs = head_outputs() if outputs is None else outputs
        self.head_conf = head_conf
        self.lane_x = lane_x
        self.raise_on_observe = raise_on_observe
        self.untrained = set(untrained)
        self.calls = 0

    def observe(self, strip, dt_s):
        self.calls += 1
        if self.raise_on_observe is not None:
            raise self.raise_on_observe
        return self.lane_x, self.head_conf

    def output(self, name):
        if name in self.untrained:
            from python.train.cam_strip import UntrainedOutputError

            raise UntrainedOutputError(f"{name} untrained")
        return self.outputs.get(name)


def selector_with(rt=None, **kw) -> ls.LearnedLaneSelector:
    rt = FakeRuntime(**kw) if rt is None else rt
    return ls.LearnedLaneSelector("fake.onnx", runtime_factory=lambda _p: rt)


def missing_selector() -> ls.LearnedLaneSelector:
    """The real SceneLaneRuntime on a path that does not exist."""
    return ls.LearnedLaneSelector(str(Path(tempfile.gettempdir()) / "gvd_no_such_scene.onnx"))


STRIP = np.zeros((8, 8, 3), dtype=np.uint8)  # the fake runtime never reads it

PAINTED = dict(horizon=0.45, half_bottom=0.45, half_top=0.06, shift=0.0, stripe_px=4, yellow_left=True)


def painted_frame(h=240, w=320) -> np.ndarray:
    return painted(h, w, **PAINTED)


def assert_finite_bounded(res: LaneResult) -> None:
    assert 0.0 <= float(res.conf) <= 1.0, res.conf
    assert math.isfinite(float(res.curvature)) and abs(res.curvature) <= ls.CURVATURE_MAX, res.curvature
    for poly in res.lanes_bev:
        assert poly, "empty polyline"
        for p in poly:
            x, y = float(p["x"]), float(p["y"])
            assert math.isfinite(x) and math.isfinite(y), p
            assert abs(x) <= ls.LAT_MAX_M and ls.Y_MIN_M <= y <= ls.Y_MAX_M, p


def assert_no_lanes(sel: ls.LaneSelection, what: str) -> None:
    assert not sel.result.lanes_bev, (what, len(sel.result.lanes_bev), sel.telemetry())
    assert sel.result.conf == 0.0, (what, sel.result.conf)


# ---------------------------------------------------------------- example checks


def check_constants_match_scene_net() -> None:
    from python.train import scene_net
    from python.train.scene_lane_runtime import SCENE_LANE_MIN_CONF

    assert (ls.N_LANES, ls.LANE_POINTS, ls.N_CURBS) == (scene_net.N_LANES, scene_net.LANE_POINTS, scene_net.N_CURBS)
    assert ls.LEARNED_LANE_MIN_CONF == SCENE_LANE_MIN_CONF
    print("lane head shapes and conf floor match scene_net / scene_lane_runtime OK")


def check_flag_default_off() -> None:
    assert ls.LEARNED_LANES_DEFAULT is False
    assert ls.learned_lanes_enabled(None, env={}) is False
    assert ls.learned_lanes_enabled(None, env={"GVD_LEARNED_LANES": ""}) is False
    assert ls.learned_lanes_enabled(None, env={"GVD_LEARNED_LANES": "0"}) is False
    assert ls.learned_lanes_enabled(None, env={"GVD_LEARNED_LANES": "1"}) is True
    assert ls.learned_lanes_enabled(False, env={"GVD_LEARNED_LANES": "1"}) is False
    assert ls.learned_lanes_enabled(True, env={}) is True
    from python.run_vision import build_parser

    assert build_parser().parse_args([]).learned_lanes is None
    assert build_parser().parse_args(["--learned-lanes"]).learned_lanes is True
    assert build_parser().parse_args(["--no-learned-lanes"]).learned_lanes is False
    print("--learned-lanes defaults OFF (CLI > GVD_LEARNED_LANES > off) OK")


def check_build_selector_wiring() -> None:
    lines: list[str] = []

    def out(*a, **_k):
        lines.append(" ".join(str(x) for x in a))

    off = ls.build_lane_selector(SimpleNamespace(learned_lanes=False, lane_source="hough"), print_fn=out)
    assert off is None and "learned lanes OFF" in lines[-1], lines
    ign = ls.build_lane_selector(SimpleNamespace(learned_lanes=True, lane_source="scene"), print_fn=out)
    assert ign is None and "ignored" in lines[-1], lines
    on = ls.build_lane_selector(
        SimpleNamespace(learned_lanes=True, lane_source="hough", scene_onnx="/nonexistent/x.onnx"), print_fn=out
    )
    assert on is not None and not on.ready and "model_missing" in lines[-1], lines
    # The live strip is only composed when the learned head can use it.
    assert ls.lane_strip_for(SimpleNamespace(lane_selector=on), SimpleNamespace(frames={"main": STRIP})) is None
    ok = selector_with()
    strip = ls.lane_strip_for(SimpleNamespace(lane_selector=ok), SimpleNamespace(frames={"main": painted_frame()}))
    assert strip is not None and strip.ndim == 3
    assert ls.lane_strip_for(SimpleNamespace(lane_selector=ok), SimpleNamespace(frames=None)) is None
    seen = None
    lines.clear()
    for head in ({"source": "learned", "reason": "none"},) * 3 + ({"source": "hough_fallback", "reason": "low_conf"},):
        seen = ls.log_lane_head_change(seen, head, print_fn=out)
    assert len(lines) == 2 and "low_conf" in lines[-1], lines
    print("build_lane_selector / lane_strip_for / change log OK")


def check_fallback_missing_model() -> None:
    sel = missing_selector()
    assert not sel.ready and sel.load_reason.startswith(ls.REASON_MODEL_MISSING), sel.load_reason
    out = sel.select(painted_frame(), strip=STRIP)
    assert out.source == ls.SOURCE_FALLBACK and out.fallback, out.telemetry()
    assert out.fallback_reason.startswith(ls.REASON_MODEL_MISSING), out.fallback_reason
    assert out.telemetry()["reason"].startswith("model_missing")
    assert out.result.lanes_bev, "Hough fallback must still find the painted lines"
    print(f"missing model -> hough_fallback ({out.fallback_reason[:40]}...), Hough lanes kept OK")


def check_fallback_refused_model() -> None:
    from python.train.scene_lane_runtime import SceneInputShapeError

    def boom(_p):
        raise SceneInputShapeError("strip input is [1,3,96,512], the live canvas is [1,3,96,768]")

    sel = ls.LearnedLaneSelector("x.onnx", runtime_factory=boom)
    out = sel.select(painted_frame(), strip=STRIP)
    assert out.source == ls.SOURCE_FALLBACK and out.fallback_reason.startswith(ls.REASON_MODEL_REFUSED), out.telemetry()
    not_ready = SimpleNamespace(ready=False, reason="onnx has no lane_gt output; retrain and export")
    sel = ls.LearnedLaneSelector("x.onnx", runtime_factory=lambda _p: not_ready)
    assert sel.select(painted_frame(), strip=STRIP).fallback_reason.startswith(ls.REASON_MODEL_REFUSED)
    print("refused model -> hough_fallback (model_refused) OK")


def check_fallback_low_conf() -> None:
    for kw, label in (
        ({"outputs": head_outputs(lane_p=0.3)}, "lane_valid 0.3"),
        ({"outputs": head_outputs(lane_xs=())}, "no lane slot"),
        ({"head_conf": 0.2}, "lane_gt conf 0.2"),
        ({"head_conf": ls.LEARNED_LANE_MIN_CONF - 1e-6}, "just under the floor"),
    ):
        out = selector_with(**kw).select(painted_frame(), strip=STRIP)
        assert out.source == ls.SOURCE_FALLBACK and out.fallback_reason == ls.REASON_LOW_CONF, (label, out.telemetry())
        assert out.result.lanes_bev, (label, "Hough fallback lost the painted lines")
    at = selector_with(head_conf=ls.LEARNED_LANE_MIN_CONF).select(painted_frame(), strip=STRIP)
    assert at.source == ls.SOURCE_LEARNED, at.telemetry()
    print("low confidence -> hough_fallback (low_conf); floor itself is learned OK")


def check_fallback_nonfinite() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        for key, idx in (("lanes", (0, 3, 0)), ("lane_valid", (1,)), ("curbs", (1, 0, 1)), ("curb_valid", (0,))):
            outs = head_outputs(curb_xs=(-3.0,))
            outs[key] = np.array(outs[key], dtype=np.float32)
            outs[key][idx] = bad
            out = selector_with(outputs=outs).select(painted_frame(), strip=STRIP)
            assert out.source == ls.SOURCE_FALLBACK and out.fallback_reason == ls.REASON_NONFINITE, (bad, key, out.telemetry())
            assert_finite_bounded(out.result)
        out = selector_with(head_conf=bad).select(painted_frame(), strip=STRIP)
        assert out.fallback_reason == ls.REASON_NONFINITE, (bad, out.telemetry())
    print("NaN / +inf / -inf in any head -> hough_fallback (nonfinite_output) OK")


def check_fallback_other_reasons() -> None:
    from python.train.scene_lane_runtime import SceneInputShapeError

    big = head_outputs(lane_xs=(-1.75, 1e6))
    assert selector_with(outputs=big).select(painted_frame(), strip=STRIP).fallback_reason == ls.REASON_OUT_OF_RANGE
    over = head_outputs(lane_p=1.5)
    assert selector_with(outputs=over).select(painted_frame(), strip=STRIP).fallback_reason == ls.REASON_OUT_OF_RANGE
    short = head_outputs()
    short["lanes"] = short["lanes"][:3]
    assert selector_with(outputs=short).select(painted_frame(), strip=STRIP).fallback_reason == ls.REASON_BAD_SHAPE
    gone = head_outputs()
    gone["curbs"] = None
    assert selector_with(outputs=gone).select(painted_frame(), strip=STRIP).fallback_reason == ls.REASON_BAD_SHAPE
    unt = selector_with(untrained=("lanes",)).select(painted_frame(), strip=STRIP)
    assert unt.fallback_reason.startswith(ls.REASON_UNTRAINED), unt.telemetry()
    err = selector_with(raise_on_observe=RuntimeError("ort")).select(painted_frame(), strip=STRIP)
    assert err.fallback_reason.startswith(ls.REASON_RUNTIME_ERROR), err.telemetry()
    none_x = selector_with(lane_x=None).select(painted_frame(), strip=STRIP)
    assert none_x.fallback_reason == ls.REASON_RUNTIME_ERROR, none_x.telemetry()
    rt = FakeRuntime(raise_on_observe=SceneInputShapeError("live strip size"))
    sel = selector_with(rt)
    first = sel.select(painted_frame(), strip=STRIP)
    rt.raise_on_observe = None
    again = sel.select(painted_frame(), strip=STRIP)
    assert first.fallback_reason.startswith(ls.REASON_DISABLED) and again.fallback_reason == first.fallback_reason
    assert rt.calls == 1, "a shape error latches: the runtime is not asked again"
    print("out_of_range / bad_shape / lanes_untrained / runtime_error / latched shape error OK")


def check_dropped_frames_and_sizes() -> None:
    sel = selector_with()
    out = sel.select(None, None, strip=None)
    assert out.fallback_reason == ls.REASON_NO_FRAME and not out.result.lanes_bev, out.telemetry()
    out = sel.select(painted_frame(), strip=None)
    assert out.fallback_reason == ls.REASON_NO_FRAME, out.telemetry()
    for shape in ((0, 0, 3), (0, 64, 3), (64, 0, 3), (1, 1, 3), (3, 3, 3), (2160, 3840, 3)):
        img = np.zeros(shape, dtype=np.uint8)
        for s in (sel, missing_selector()):
            res = s.select(img, strip=STRIP)
            assert_finite_bounded(res.result)
            assert not res.result.lanes_bev, (shape, res.telemetry())
    assert ls.paint_evidence_px(np.zeros((2160, 3840, 3), np.uint8)) == 0
    assert ls.paint_evidence_px(np.zeros((5, 5), np.uint8)) == 0
    assert ls.paint_evidence_px(np.full((40, 40, 3), np.nan, np.float32)) == 0
    print("dropped frames (no_frame) and empty / 1 px / 4K frames OK")


def check_learned_drives_on_paint() -> None:
    out = selector_with(outputs=head_outputs(curb_xs=(-3.5, 3.5))).select(painted_frame(), strip=STRIP)
    assert out.source == ls.SOURCE_LEARNED and not out.fallback, out.telemetry()
    assert len(out.result.lanes_bev) == 2 and len(out.curbs_bev) == 2
    xs = sorted(round(p["x"], 2) for poly in out.result.lanes_bev for p in poly[:1])
    assert xs == [-1.75, 1.75], xs
    assert out.telemetry()["reason"] == "none" and out.telemetry()["mode"] == "learned"
    print("confident learned lanes on paint -> source learned, curbs separate OK")


def check_hallucinating_head_gated() -> None:
    sel = selector_with()
    for img in (uniform(240, 320, (70, 72, 74)), noise(240, 320, 1), sky_over_asphalt(240, 320, 0.45, 210)):
        out = sel.select(img, strip=STRIP)
        assert_no_lanes(out, "learned hallucination on no-paint")
        assert out.paint_gated and out.source == ls.SOURCE_NONE, out.telemetry()
    print("confident learned lanes on a no-paint frame are vetoed by the paint gate OK")


def check_pipeline_wiring() -> None:
    from python.perception.pipeline import ModularPerception

    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    pout = perc.tick(painted_frame(), ego_speed_mps=5.0)
    assert pout.lane_head == ls.HOUGH_ONLY_TELEMETRY, pout.lane_head
    hough_lanes = pout.lanes_bev
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.lane_selector = selector_with()
    pout = perc.tick(painted_frame(), ego_speed_mps=5.0, lane_strip=STRIP)
    assert pout.lane_head["source"] == "learned" and pout.lane_head["mode"] == "learned", pout.lane_head
    assert len(pout.lanes_bev) == 2 and pout.lanes_bev != hough_lanes
    held = perc.tick(painted_frame(), ego_speed_mps=5.0, lane_strip=STRIP)  # same pixels: stale tick
    assert held.lane_head == pout.lane_head
    perc = ModularPerception(allow_synthetic=False, detector_id="empty")
    perc.lane_selector = missing_selector()
    pout = perc.tick(noise(240, 320, 5), ego_speed_mps=5.0, lane_strip=STRIP)
    assert pout.lane_head["fallback"] and pout.lane_head["reason"].startswith("model_missing"), pout.lane_head
    assert not pout.lanes_bev and "lanes" in pout.missing
    print("pipeline: flag off is Hough as before; learned / fallback reach lane_head OK")


def check_eval_harness() -> None:
    from python.debug.lane_head_eval import MIN_BOUNDARIES, MIN_NO_PAINT, ego_boundaries, score, verdict, x_at
    from python.debug.lane_targets import LANE_GT_DIM, LOOKAHEADS_M, SLOT_LEFT, SLOT_RIGHT, lookahead_slot

    assert x_at([{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 10.0}], 5.0) == 1.0
    assert x_at([{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 10.0}], 11.0) is None
    assert ego_boundaries([[{"x": -1.7, "y": 0}, {"x": -1.7, "y": 40}], [{"x": 1.8, "y": 0}, {"x": 1.8, "y": 40}]], 8.0) == (-1.7, 1.8)

    def gt_vec(left, right):
        v = np.full(LANE_GT_DIM, np.nan)
        for k in range(len(LOOKAHEADS_M)):
            v[lookahead_slot(k, SLOT_LEFT)] = left
            v[lookahead_slot(k, SLOT_RIGHT)] = right
        return v

    samples = [{"has_paint": True, "lane_gt": gt_vec(-1.75, 1.75)} for _ in range(30)]
    samples += [{"has_paint": False, "lane_gt": gt_vec(np.nan, np.nan)} for _ in range(10)]
    perfect = LaneResult(conf=1.0, lanes_bev=[[dict(x=float(x), y=float(y)) for y in (0, 40)] for x in (-1.75, 1.75)], curvature=0.0)
    off = LaneResult(conf=1.0, lanes_bev=[[dict(x=float(x), y=float(y)) for y in (0, 40)] for x in (-1.0, 2.5)], curvature=0.0)
    m_good = score(samples, lambda s: perfect if s["has_paint"] else None)
    m_bad = score(samples, lambda s: off)
    assert m_good["mae_m"] == 0.0 and m_good["recall"] == 1.0 and m_good["fp_rate"] == 0.0, m_good
    assert m_bad["recall"] == 0.0 and abs(m_bad["mae_m"] - 0.75) < 1e-9 and m_bad["fp_rate"] == 1.0, m_bad
    small = verdict(m_good, m_bad)
    assert not small["learned_beats_hough"] and "insufficient data" in small["why_not"][0], small
    big = dict(m_good, n_boundaries=MIN_BOUNDARIES, n_no_paint=MIN_NO_PAINT)
    assert verdict(big, m_bad)["learned_beats_hough"] is True
    assert verdict(dict(m_bad, n_boundaries=MIN_BOUNDARIES, n_no_paint=MIN_NO_PAINT), m_good)["learned_beats_hough"] is False
    # The CLI on synthetic .npz fixtures: no model here, so learned is unmeasured, never "better".
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_learned_lanes as cli

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(4):
            np.savez(Path(tmp) / f"p{i}.npz", main=painted_frame(), lane_gt=gt_vec(-1.75, 1.75), has_paint=np.array([True]))
            np.savez(Path(tmp) / f"n{i}.npz", main=noise(240, 320, i), lane_gt=gt_vec(np.nan, np.nan), has_paint=np.array([False]))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["--npz-dir", tmp, "--onnx", str(Path(tmp) / "missing.onnx"), "--json", str(Path(tmp) / "r.json")])
        rep = json.loads((Path(tmp) / "r.json").read_text())
    assert rc == 0 and rep["n_samples"] == 8, rep
    assert rep["learned"]["n_matched"] == 0 and rep["learned_beats_hough"] is False, rep
    assert rep["verdict_vs_hough"]["learned_beats_hough"] is False, rep
    assert rep["hough"]["n_no_paint"] == 4 and rep["hough"]["n_boundaries"] == 32, rep
    assert rep["hough_gated"]["fp_rate"] == 0.0, rep["hough_gated"]
    print(
        f"eval harness on synthetic fixtures OK (noise fp_rate: hough={rep['hough']['fp_rate']} "
        f"hough_gated={rep['hough_gated']['fp_rate']})"
    )


def check_hough_output_sanitized() -> None:
    """I4 on the fallback path: a broken Hough result never leaks NaN, far points or conf > 1."""
    good = [{"x": -1.75, "y": float(y), "z": 0.0} for y in range(0, 40, 4)]
    edge = [{"x": ls.LAT_MAX_M, "y": ls.Y_MIN_M}, {"x": -ls.LAT_MAX_M, "y": ls.Y_MAX_M}]
    bad = [
        [{"x": float("nan"), "y": 5.0}], [{"x": 0.0, "y": float("inf")}], [{"y": 3.0}], [{"x": 1.0}],
        [{"x": ls.LAT_MAX_M + 1e-6, "y": 5.0}], [{"x": -ls.LAT_MAX_M - 1e-6, "y": 5.0}],
        [{"x": 0.0, "y": ls.Y_MIN_M - 1e-6}], [{"x": 0.0, "y": ls.Y_MAX_M + 1e-6}], [],
    ]
    for conf, curv, want_conf, want_curv in (
        (0.95, 0.01, 0.95, 0.01), (7.0, 9.0, 1.0, ls.CURVATURE_MAX), (-3.0, -9.0, 0.0, -ls.CURVATURE_MAX),
        (float("nan"), float("nan"), 0.0, 0.0), (float("inf"), float("-inf"), 0.0, 0.0),
    ):
        res = ls.sanitize_result(LaneResult(conf=conf, lanes_bev=[good, edge, *bad], curvature=curv))
        assert res.lanes_bev == [good, edge], res.lanes_bev
        assert res.conf == want_conf and res.curvature == want_curv, (conf, curv, res)
    only_bad = ls.sanitize_result(LaneResult(conf=0.9, lanes_bev=bad, curvature=0.0))
    assert only_bad.lanes_bev == [] and only_bad.conf == 0.0, only_bad
    sel = ls.LearnedLaneSelector(
        None, runtime_factory=lambda _p: None,
        hough=lambda m, s: LaneResult(conf=float("nan"), lanes_bev=[good, *bad], curvature=float("inf")),
    )
    out = sel.select(painted_frame(), strip=STRIP)
    assert out.result.lanes_bev == [good] and out.result.conf == 0.0 and out.result.curvature == 0.0, out.result
    print("broken Hough output is sanitized (NaN / far / missing keys dropped, conf and curvature clamped) OK")


def check_parse_boundaries() -> None:
    """Exact edges of the parse rules (each one a mutant before this check)."""
    at = head_outputs(lane_xs=(-ls.LAT_MAX_M, ls.LAT_MAX_M), lane_p=ls.LANE_VALID_MIN, curb_xs=(ls.LAT_MAX_M,), curb_p=1.0)
    at["lanes"][0, 0, 1] = ls.Y_MIN_M
    at["lanes"][1, -1, 1] = ls.Y_MAX_M
    at["lane_valid"][2] = 1.0
    at["lanes"][2] = lane_poly(0.0)
    parsed = ls.parse_learned_lanes(at, 1.0)
    assert parsed.reason == "" and len(parsed.result.lanes_bev) == 3 and parsed.conf == 1.0, parsed
    assert len(parsed.curbs_bev) == 1
    capped = ls.parse_learned_lanes(head_outputs(lane_p=0.9), 3.0)
    assert abs(capped.conf - 0.9) < 1e-6 and capped.result.conf == capped.conf, capped
    low = ls.parse_learned_lanes(head_outputs(lane_p=0.45))
    assert low.reason == ls.REASON_LOW_CONF and low.conf == 0.0 and low.result is None, low
    lowh = ls.parse_learned_lanes(head_outputs(lane_p=0.8), 0.3)
    assert lowh.reason == ls.REASON_LOW_CONF and abs(lowh.conf - 0.3) < 1e-9, lowh
    out_sel = selector_with(head_conf=0.3).select(painted_frame(), strip=STRIP)
    assert abs(out_sel.telemetry()["learned_conf"] - 0.3) < 1e-9, out_sel.telemetry()
    for mutate, reason in (
        (lambda o: o.__setitem__("lanes", o["lanes"][:5]), ls.REASON_BAD_SHAPE),
        (lambda o: o.__setitem__("lane_valid", o["lane_valid"][:5]), ls.REASON_BAD_SHAPE),
        (lambda o: o.__setitem__("curbs", o["curbs"][:1]), ls.REASON_BAD_SHAPE),
        (lambda o: o.__setitem__("curb_valid", o["curb_valid"][:1]), ls.REASON_BAD_SHAPE),
        (lambda o: o.__setitem__("lanes", "not a tensor"), ls.REASON_BAD_SHAPE),
        (lambda o: o["curb_valid"].__setitem__(0, -0.01), ls.REASON_OUT_OF_RANGE),
        (lambda o: o["curb_valid"].__setitem__(1, 1.01), ls.REASON_OUT_OF_RANGE),
        (lambda o: o["lane_valid"].__setitem__(5, -0.01), ls.REASON_OUT_OF_RANGE),
        (lambda o: o["curbs"].__setitem__((0, 4, 1), ls.Y_MAX_M + 1), ls.REASON_OUT_OF_RANGE),
        (lambda o: o["lanes"].__setitem__((1, 4, 1), ls.Y_MIN_M - 1), ls.REASON_OUT_OF_RANGE),
    ):
        outs = head_outputs(curb_xs=(-3.0, 3.0))
        outs = {k: np.array(v, dtype=np.float32) for k, v in outs.items()}
        mutate(outs)
        got = ls.parse_learned_lanes(outs)
        assert got.result is None and got.reason == reason, (reason, got)
    assert ls.parse_learned_lanes(head_outputs(), "x").reason == ls.REASON_BAD_SHAPE
    assert ls.parse_learned_lanes(head_outputs(), float("nan")).reason == ls.REASON_NONFINITE
    # A slot below LANE_VALID_MIN is out of range only when it is kept.
    far = head_outputs(lane_xs=(-1.75, 1.75, 1e6), lane_p=0.9)
    far["lane_valid"][2] = 0.49
    assert ls.parse_learned_lanes(far).reason == "" and len(ls.parse_learned_lanes(far).result.lanes_bev) == 2
    print("parse boundaries: valid floor, conf cap, bounds, per-head shape and range reasons OK")


def check_selector_plumbing() -> None:
    """dt, strip and stitch reach the right callee; reason strings name the error."""
    seen = {}

    class Rec(FakeRuntime):
        def observe(self, strip, dt_s):
            seen["strip"], seen["dt"] = strip, dt_s
            return super().observe(strip, dt_s)

    rt = Rec()
    selector_with(rt).select(painted_frame(), strip=STRIP, dt_s=0.125)
    assert seen["strip"] is STRIP and seen["dt"] == 0.125, seen
    calls = []

    def hough(m, s):
        calls.append((m is not None, s is not None))
        return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)

    from python.perception.stitch360 import GAP_PX, SECTOR_H, SECTOR_W, STITCH_ORDER, forward_lane_view

    n = len(STITCH_ORDER)
    stitch = np.zeros((SECTOR_H, n * SECTOR_W + (n - 1) * GAP_PX, 3), dtype=np.uint8)
    band = forward_lane_view(stitch)
    band[:] = painted(band.shape[0], band.shape[1], **PAINTED)
    i0 = STITCH_ORDER.index("wide") * (SECTOR_W + GAP_PX)
    stitch[:, i0:i0 + band.shape[1]] = band
    fwd = ls.forward_image(None, stitch)
    assert fwd.shape == band.shape and np.array_equal(fwd, band), fwd.shape
    assert ls.forward_image(painted_frame(), None).shape == painted_frame().shape
    assert ls.forward_image(None, painted_frame()).shape == painted_frame().shape
    assert ls.forward_image(None, None) is None
    assert ls.forward_image(np.zeros((4, 4), np.uint8)) is None
    assert ls.forward_image(np.zeros((1, 4, 3), np.uint8)) is None
    assert ls.forward_image(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4, 3)
    f32 = ls.forward_image(np.full((4, 4, 3), 300.0, np.float32))
    assert f32.dtype == np.uint8 and int(f32.max()) == 255
    blank_main = np.zeros((240, 320, 3), np.uint8)
    sel = ls.LearnedLaneSelector(None, runtime_factory=lambda _p: None, hough=hough)
    out = sel.select(blank_main, stitch, strip=None)
    assert calls[-1] == (True, True), calls
    assert out.paint_px >= ls.PAINT_MIN_PX, "paint evidence must come from the stitch band, not main"
    assert ls.paint_evidence_px(painted_frame().astype(np.float32)) == ls.paint_evidence_px(painted_frame())
    sky_only = painted_frame()
    sky_only[int(ls.PAINT_ROI_TOP * sky_only.shape[0]):] = 70
    assert ls.paint_evidence_px(sky_only) == 0, "paint above the ROI does not count"
    err = selector_with(raise_on_observe=RuntimeError("ort")).select(painted_frame(), strip=STRIP)
    assert err.fallback_reason == "runtime_error: RuntimeError", err.fallback_reason
    from python.train.scene_lane_runtime import SceneInputShapeError

    dis = selector_with(raise_on_observe=SceneInputShapeError("x")).select(painted_frame(), strip=STRIP)
    assert dis.fallback_reason == "learned_disabled: SceneInputShapeError", dis.fallback_reason

    class KeyErr(FakeRuntime):
        def output(self, name):
            raise KeyError(name)

    ke = selector_with(KeyErr()).select(painted_frame(), strip=STRIP)
    assert ke.fallback_reason == "runtime_error: KeyError", ke.fallback_reason
    assert selector_with(untrained=("curb_valid",)).select(painted_frame(), strip=STRIP).fallback_reason == "lanes_untrained: curb_valid"
    # Paint exactly at the floor passes; one under is gated.
    real = ls.paint_evidence_px
    try:
        ls.paint_evidence_px = lambda _img: ls.PAINT_MIN_PX
        at = selector_with().select(painted_frame(), strip=STRIP)
        ls.paint_evidence_px = lambda _img: ls.PAINT_MIN_PX - 1
        under = selector_with().select(painted_frame(), strip=STRIP)
    finally:
        ls.paint_evidence_px = real
    assert not at.paint_gated and at.result.lanes_bev and at.telemetry()["paint_px"] == ls.PAINT_MIN_PX
    assert under.paint_gated and not under.result.lanes_bev
    tel = selector_with().select(painted_frame(), strip=STRIP).telemetry()
    assert tel == {
        "mode": "learned", "source": "learned", "fallback": False, "reason": "none", "learned_conf": 0.9,
        "paint_px": ls.paint_evidence_px(painted_frame()), "paint_gated": False, "curbs_n": 0,
    }, tel
    assert "model ready (fake.onnx)" in selector_with().describe()
    assert "model_missing" in missing_selector().describe() and "Hough fallback" in missing_selector().describe()
    print("selector plumbing: dt / strip / stitch band, error names, paint floor, telemetry OK")


def check_curvature() -> None:
    ys = np.linspace(2.0, 40.0, ls.LANE_POINTS)
    bend = np.stack([-1.75 + 0.01 * ys ** 2, ys], -1)
    straight = np.stack([np.full_like(ys, 1.75), ys], -1)
    assert abs(ls._curvature([bend]) - 0.02) < 1e-9
    assert abs(ls._curvature([bend, straight]) - 0.01) < 1e-9
    assert ls._curvature([np.stack([np.full_like(ys, 1.0), np.full_like(ys, 5.0)], -1)]) == 0.0
    assert ls._curvature([np.stack([-1.75 + 1.0 * ys ** 2, ys], -1)]) == ls.CURVATURE_MAX
    assert ls._curvature([np.stack([-1.75 - 1.0 * ys ** 2, ys], -1)]) == -ls.CURVATURE_MAX
    assert ls._curvature([]) == 0.0
    print("learned curvature: quadratic fit, clipped, flat polylines skipped OK")


def check_env_and_load_details() -> None:
    for raw, want in (("1", True), ("true", True), (" TRUE ", True), ("yes", True), ("On", True),
                      ("0", False), ("false", False), ("off", False), ("  ", False), ("2", False)):
        assert ls.learned_lanes_enabled(None, env={"GVD_LEARNED_LANES": raw}) is want, raw
    saved = os.environ.get("GVD_LEARNED_LANES")
    try:
        os.environ["GVD_LEARNED_LANES"] = "1"
        assert ls.learned_lanes_enabled(None) is True
        os.environ.pop("GVD_LEARNED_LANES")
        assert ls.learned_lanes_enabled(None) is False
    finally:
        if saved is not None:
            os.environ["GVD_LEARNED_LANES"] = saved
    got = []
    ls.LearnedLaneSelector("/models/x.onnx", runtime_factory=lambda p: got.append(p) or FakeRuntime())
    assert got == ["/models/x.onnx"], got
    bare = ls.LearnedLaneSelector(None, runtime_factory=lambda _p: object())
    assert bare.runtime is None and bare.load_reason == "model_refused: not ready", bare.load_reason
    blank = ls.LearnedLaneSelector(None, runtime_factory=lambda _p: SimpleNamespace(ready=False, reason=""))
    assert blank.load_reason == "model_refused: not ready", blank.load_reason
    ok = selector_with()
    assert ok.load_reason == "" and ok.disabled_reason == "" and ok.ready

    def boom(_p):
        raise ValueError("bad graph")

    assert ls.LearnedLaneSelector(None, runtime_factory=boom).load_reason == "model_refused: ValueError: bad graph"
    # Paint above PAINT_ROI_TOP never counts; the same stripe below it does.
    top = np.full((240, 320, 3), 70, np.uint8)
    rows = int(ls.PAINT_ROI_TOP * 240) - 2
    top[:rows, 150:154] = 235
    low = np.full((240, 320, 3), 70, np.uint8)
    low[240 - rows:, 150:154] = 235
    assert ls.paint_evidence_px(top) == 0 and ls.paint_evidence_px(low) >= ls.PAINT_MIN_PX, (
        ls.paint_evidence_px(top), ls.paint_evidence_px(low))
    bgra = np.concatenate([painted_frame(), np.full((240, 320, 1), 255, np.uint8)], axis=2)
    assert ls.paint_evidence_px(bgra) == ls.paint_evidence_px(painted_frame())
    yel = np.full((240, 320, 3), 70, np.uint8)
    yel[100:, 150:154] = (40, 190, 230)
    assert ls.paint_evidence_px(yel) >= ls.PAINT_MIN_PX
    assert ls.paint_evidence_px(yel.astype(np.float32)) == ls.paint_evidence_px(yel)
    print("env parsing, factory path, load reasons, paint ROI / BGRA / yellow / float OK")


# ---------------------------------------------------------------- properties


class Frame:
    """A generated frame that shrinks and prints as its recipe, not its pixels."""

    def __init__(self, fn, *args, **kwargs):
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.img = fn(*args, **kwargs)

    def __repr__(self) -> str:
        parts = [repr(a) for a in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{self.fn.__name__}({', '.join(parts)})"

if given is not None:
    PROP = settings(
        max_examples=int(os.environ.get("GVD_PROP_EXAMPLES", "60")), deadline=None, derandomize=True, database=None,
        suppress_health_check=list(HealthCheck), print_blob=False,
    )
    dims = st.tuples(st.integers(96, 480), st.integers(128, 640))
    geom = st.fixed_dictionaries({
        "horizon": st.floats(0.3, 0.55), "half_bottom": st.floats(0.25, 0.6),
        "half_top": st.floats(0.02, 0.2), "shift": st.floats(-0.15, 0.15),
    })

    @st.composite
    def no_paint_frames(draw):
        h, w = draw(dims)
        fam = draw(st.sampled_from(NO_PAINT_FAMILIES))
        if fam == "uniform":
            return Frame(uniform, h, w, tuple(draw(st.integers(0, 255)) for _ in range(3)))
        if fam == "noise":
            return Frame(noise, h, w, draw(st.integers(0, 2**16)), lo=draw(st.integers(0, 230)))
        if fam == "sky":
            return Frame(sky_over_asphalt, h, w, draw(st.floats(0.0, 1.0)), draw(st.integers(120, 255)))
        g = draw(geom)
        curb = draw(st.one_of(st.just(0), st.integers(ls.PAINT_MAX_WIDTH_PX + 2, 64)))
        return Frame(edge_only, h, w, **g, side=draw(st.sampled_from(["grass", "dirt", "concrete"])),
                     curb_v=draw(st.integers(185, 255)), curb_px=curb)

    @st.composite
    def edge_only_frames(draw):
        h, w = draw(dims)
        return Frame(edge_only, h, w, **draw(geom), side=draw(st.sampled_from(["grass", "dirt", "concrete"])),
                     curb_v=draw(st.integers(185, 255)),
                     curb_px=draw(st.integers(ls.PAINT_MAX_WIDTH_PX + 2, 64)))

    @st.composite
    def painted_frames(draw):
        # Camera-like aspect (w <= 2h): the paint gate's row-continuity check
        # assumes a lane stripe moves at most ~4 px per image row.
        h = draw(st.integers(160, 480))
        w = draw(st.integers(200, min(640, 2 * h)))
        return Frame(painted, h, w, horizon=draw(st.floats(0.35, 0.5)), half_bottom=draw(st.floats(0.35, 0.5)),
                     half_top=draw(st.floats(0.04, 0.12)), shift=draw(st.floats(-0.08, 0.08)),
                     stripe_px=draw(st.integers(3, 8)), yellow_left=draw(st.booleans()))

    finite_or_not = st.one_of(
        st.floats(-50, 50), st.floats(allow_nan=True, allow_infinity=True), st.sampled_from([0.0, 1.0, 1e9, -1e9]),
    )

    @st.composite
    def raw_heads(draw):
        def arr(n):
            return np.asarray(draw(st.lists(finite_or_not, min_size=n, max_size=n)), dtype=np.float64)

        lanes = np.stack([lane_poly(draw(st.floats(-6, 6))) for _ in range(ls.N_LANES)])
        if draw(st.booleans()):
            lanes = lanes.reshape(-1)
            idx = draw(st.lists(st.integers(0, lanes.size - 1), max_size=4))
            lanes[idx] = arr(len(idx))
            lanes = lanes.reshape(ls.N_LANES, ls.LANE_POINTS, 2)
        curbs = np.stack([lane_poly(draw(st.floats(-8, 8))) for _ in range(ls.N_CURBS)])
        valid = st.lists(st.one_of(st.floats(0, 1), finite_or_not), min_size=1, max_size=1)
        lane_valid = np.asarray([draw(valid)[0] for _ in range(ls.N_LANES)])
        curb_valid = np.asarray([draw(valid)[0] for _ in range(ls.N_CURBS)])
        return {"lanes": lanes, "lane_valid": lane_valid, "curbs": curbs, "curb_valid": curb_valid}

    @PROP
    @given(img=no_paint_frames(), hallucinate=st.booleans())
    def prop_no_paint_no_lines(img, hallucinate):
        """I1: final output has zero painted lanes, learned (hallucinating) or Hough fallback."""
        sel = selector_with() if hallucinate else missing_selector()
        out = sel.select(img.img, strip=STRIP)
        assert_no_lanes(out, "no-paint frame")
        assert out.paint_px < ls.PAINT_MIN_PX, out.paint_px

    @PROP
    @given(img=edge_only_frames(), curb_p=st.floats(0.5, 1.0), lane_p=st.floats(0.0, 0.49))
    def prop_edges_are_not_paint(img, curb_p, lane_p):
        """I2: a head that sees only curbs, on an edge-only scene, gives no painted lane."""
        outs = head_outputs(lane_p=lane_p, curb_xs=(-3.5, 3.5), curb_p=curb_p)
        out = selector_with(outputs=outs).select(img.img, strip=STRIP)
        assert_no_lanes(out, "edge-only")
        assert out.fallback_reason == ls.REASON_LOW_CONF, out.telemetry()
        assert len(out.curbs_bev) == 2, "curbs are reported, apart from lanes"

    @PROP
    @given(heads=raw_heads(), head_conf=st.one_of(st.none(), finite_or_not))
    def prop_parse_finite_bounded_separate(heads, head_conf):
        """I2 + I4 on the raw head: never a curb point in lanes_bev; all output finite and bounded."""
        parsed = ls.parse_learned_lanes(heads, head_conf)
        if parsed.result is None:
            assert parsed.reason in REASONS, parsed.reason
            return
        assert parsed.reason == ""
        assert_finite_bounded(parsed.result)
        assert parsed.result.conf >= ls.LEARNED_LANE_MIN_CONF
        lane_set = {tuple(np.round(heads["lanes"][i].reshape(-1), 6)) for i in range(ls.N_LANES)}
        kept = int((np.asarray(heads["lane_valid"]) >= ls.LANE_VALID_MIN).sum())
        assert len(parsed.result.lanes_bev) == kept
        for poly in parsed.result.lanes_bev:
            flat = tuple(np.round(np.asarray([[p["x"], p["y"]] for p in poly]).reshape(-1), 6))
            assert flat in lane_set, "a lanes_bev polyline that is not a lanes slot (curb leak)"

    @PROP
    @given(img=st.one_of(no_paint_frames(), painted_frames()), heads=raw_heads())
    def prop_select_finite_and_deterministic(img, heads):
        """I3 + I4 + I5: any head output on any frame: explicit source, bounded output, same twice."""
        a = selector_with(outputs=heads).select(img.img, strip=STRIP)
        b = selector_with(outputs=heads).select(img.img.copy(), strip=STRIP)
        assert_finite_bounded(a.result)
        assert a.source in (ls.SOURCE_LEARNED, ls.SOURCE_FALLBACK, ls.SOURCE_NONE)
        if a.source == ls.SOURCE_LEARNED:
            assert a.fallback_reason == "" and not a.paint_gated, a.telemetry()
        if a.source == ls.SOURCE_FALLBACK:
            assert a.fallback and a.telemetry()["reason"] not in ("", "none"), a.telemetry()
        if a.source == ls.SOURCE_NONE:
            # Lines were vetoed by the paint gate; says so, and nothing is emitted.
            assert a.paint_gated and not a.result.lanes_bev, a.telemetry()
        assert a.telemetry() == b.telemetry()
        assert a.result.lanes_bev == b.result.lanes_bev and a.result.conf == b.result.conf

    @PROP
    @given(img=painted_frames())
    def prop_paint_gate_is_live(img):
        """The gate must not veto real paint: a confident head on a painted frame drives."""
        assert ls.paint_evidence_px(ls.forward_image(img.img)) >= ls.PAINT_MIN_PX
        out = selector_with().select(img.img, strip=STRIP)
        assert out.source == ls.SOURCE_LEARNED and len(out.result.lanes_bev) == 2, out.telemetry()

    PROPS = [
        prop_no_paint_no_lines, prop_edges_are_not_paint, prop_parse_finite_bounded_separate,
        prop_select_finite_and_deterministic, prop_paint_gate_is_live,
    ]
else:
    PROPS = []


def main() -> None:
    check_constants_match_scene_net()
    check_flag_default_off()
    check_build_selector_wiring()
    check_fallback_missing_model()
    check_fallback_refused_model()
    check_fallback_low_conf()
    check_fallback_nonfinite()
    check_fallback_other_reasons()
    check_dropped_frames_and_sizes()
    check_learned_drives_on_paint()
    check_hough_output_sanitized()
    check_parse_boundaries()
    check_selector_plumbing()
    check_curvature()
    check_env_and_load_details()
    check_hallucinating_head_gated()
    check_pipeline_wiring()
    check_eval_harness()
    if given is None:
        skip_check("hypothesis_properties", "hypothesis not installed (dev-only dep; CI installs it)")
    for prop in PROPS:
        prop()
        print(f"{prop.__name__}: holds OK")
    print("test_learned_lanes_props: OK")


if __name__ == "__main__":
    main()
