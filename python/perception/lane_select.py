"""Live lane selection: the learned SceneNet lane head first, Hough as the fallback.

Off by default (``--learned-lanes`` / ``GVD_LEARNED_LANES=1`` turns it on). No
trained lane weights and no held-out learned-vs-Hough eval exist yet, so the
default stays Hough until ``scripts/eval_learned_lanes.py`` shows the learned
head winning on real held-out data.

Invariants (``scripts/test_learned_lanes_props.py`` checks each one):

I1  No paint, no lines. On a frame with no painted stripe (blank, uniform,
    sky, noise, an edge-only road), the selected output has zero painted
    lanes, whichever source produced it (learned or Hough fallback).
I2  Edges are not paint. Learned ``curbs`` never enter ``lanes_bev``; they
    ride separately in ``curbs_bev``. An edge-only scene yields no lanes.
I3  Never silent. Every tick reports ``source`` and, when Hough is used
    under the learned flag, ``fallback_reason`` (missing model, refused
    model, low confidence, non-finite output, bad shape, untrained head,
    no frame, runtime error, latched shape error).
I4  Finite and bounded. Every emitted point is finite, ``|x| <= LAT_MAX_M``
    and ``Y_MIN_M <= y <= Y_MAX_M``; ``conf`` is in [0, 1]; curvature is
    finite and clipped. Any non-finite or out-of-range learned output is
    rejected whole and falls back.
I5  Deterministic. The same inputs (and model state) give the same output.

The paint gate (``paint_evidence_px``) is a pixel check: thin bright or
saturated stripes in the lower part of the forward image, in connected
components long enough to be a stripe. It vetoes both sources. A worn or
hazy stripe that it misses gives no lanes (a stop, not a guess).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from python.perception.lanes import LaneResult, lanes_from_view

LEARNED_LANES_ENV = "GVD_LEARNED_LANES"
LEARNED_LANES_DEFAULT = False

# Shapes of the SceneNet lane / curb heads (python.train.scene_net N_LANES,
# LANE_POINTS, N_CURBS; a test pins them equal).
N_LANES = 6
LANE_POINTS = 16
N_CURBS = 2

# Per-slot sigmoid floor for a lane slot to count as present.
LANE_VALID_MIN = 0.5
# Same floor as the scene lane runtime (SCENE_LANE_MIN_CONF; a test pins it).
LEARNED_LANE_MIN_CONF = 0.40
# Ego-frame bounds for an emitted point (metres; x right, y ahead).
LAT_MAX_M = 15.0
Y_MIN_M = -10.0
Y_MAX_M = 120.0
CURVATURE_MAX = 0.05

# Paint gate. Stripe pixels in components at least this long (px) count.
PAINT_MIN_RUN_PX = 12
PAINT_MIN_PX = 16
PAINT_ROI_TOP = 0.35
# Ridge half-widths (px). A bright run up to 2 * max(PAINT_TAUS) wide is a stripe.
PAINT_TAUS = (2, 4, 6, 8, 10, 12)
PAINT_MAX_WIDTH_PX = 2 * max(PAINT_TAUS)
PAINT_MIN_ROWS = 5

LEARNED_OUTPUTS = ("lanes", "lane_valid", "curbs", "curb_valid")

SOURCE_LEARNED = "learned"
SOURCE_FALLBACK = "hough_fallback"
SOURCE_NONE = "none"

REASON_MODEL_MISSING = "model_missing"
REASON_MODEL_REFUSED = "model_refused"
REASON_NO_FRAME = "no_frame"
REASON_RUNTIME_ERROR = "runtime_error"
REASON_UNTRAINED = "lanes_untrained"
REASON_BAD_SHAPE = "bad_shape"
REASON_NONFINITE = "nonfinite_output"
REASON_OUT_OF_RANGE = "out_of_range"
REASON_LOW_CONF = "low_conf"
REASON_DISABLED = "learned_disabled"


def learned_lanes_enabled(value: Any = None, env: Mapping[str, str] | None = None) -> bool:
    """CLI value wins; else ``GVD_LEARNED_LANES``; else the default (off)."""
    if value is not None:
        return bool(value)
    import os

    raw = (env if env is not None else os.environ).get(LEARNED_LANES_ENV)
    if raw is None or str(raw).strip() == "":
        return LEARNED_LANES_DEFAULT
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _empty() -> LaneResult:
    return LaneResult(conf=0.0, lanes_bev=[], curvature=0.0)


def forward_image(main_bgr: Any, stitch_bgr: Any = None) -> np.ndarray | None:
    """The image the forward lane fit reads: the windshield band of a full
    stitch, else the stitch, else ``main``. None for no frame, a frame under
    2x2, or under 3 channels. Float input is clipped to 0..255 (NaN is 0)."""
    from python.perception.lanes import _full_stitch

    img = stitch_bgr if stitch_bgr is not None else main_bgr
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[2] < 3:
        return None
    if stitch_bgr is not None and _full_stitch(arr):
        from python.perception.stitch360 import forward_lane_view

        arr = np.asarray(forward_lane_view(arr))
    if arr.dtype != np.uint8:
        arr = np.clip(np.nan_to_num(arr.astype(np.float64)), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def _ridge(chan: np.ndarray) -> np.ndarray:
    """Max over tau of min(c(x) - c(x - tau), c(x) - c(x + tau)), along rows.

    High only on a horizontal run brighter (or more saturated) than BOTH
    sides within ``tau``: a thin stripe. The edge of a band wider than
    ``2 * max(PAINT_TAUS)`` has the band itself on one side, so it is not a
    ridge. Out-of-frame neighbours count as equal (no ridge at the border).
    """
    best = np.full(chan.shape, -32768, dtype=np.int16)
    w = chan.shape[1]
    for tau in PAINT_TAUS:
        if tau >= w:
            continue
        left = np.empty_like(chan)
        right = np.empty_like(chan)
        left[:, tau:] = chan[:, :-tau]
        left[:, :tau] = chan[:, :tau]
        right[:, :-tau] = chan[:, tau:]
        right[:, -tau:] = chan[:, -tau:]
        best = np.maximum(best, np.minimum(chan - left, chan - right))
    return best


def paint_evidence_px(bgr: Any) -> int:
    """Pixels of thin painted stripe in the lower image (0 means no paint).

    A stripe pixel is white (S <= 18, V >= 185) and a value ridge of at
    least 18, or yellow (H 8-42, S >= 28, V >= 80) and a saturation ridge of
    at least 12 (``_ridge``: brighter than both sides, so the edge of a wide
    bright sidewalk or curb band is not paint). Only components whose
    bounding box is at least ``PAINT_MIN_RUN_PX`` long count, so per-pixel
    noise does not. Limit: a bright band thinner than ``PAINT_MAX_WIDTH_PX``
    in the image looks exactly like paint in pixels and does count.

    The count is the stripe area after the +-4 px join: a W px stripe over R
    rows clear of the ROI edges gives (W + 8) * (R - PAINT_MIN_ROWS + 1) when
    W_ridge = W, which holds only up to W = 12 px (wider stripes are capped
    by the ridge). No frame, under 3 channels, or under 4 rows below
    ``PAINT_ROI_TOP`` gives 0. Float input is clipped to 0..255 first, like a
    camera.
    """
    if bgr is None:
        return 0
    arr = np.asarray(bgr)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return 0
    img = np.ascontiguousarray(arr[:, :, :3])
    if img.dtype != np.uint8:
        img = np.clip(np.nan_to_num(img.astype(np.float64)), 0, 255).astype(np.uint8)
    h = img.shape[0]
    top = int(PAINT_ROI_TOP * h)
    img = img[top:]
    if img.shape[0] < 4:
        return 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    # Paint colour in a horizontal run of at least 3 px: single noisy pixels are not paint.
    run3 = np.ones((1, 3), np.uint8)
    yellow = cv2.morphologyEx(cv2.inRange(hsv, (8, 28, 80), (42, 255, 255)), cv2.MORPH_OPEN, run3) > 0
    white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 185), (180, 18, 255)), cv2.MORPH_OPEN, run3) > 0
    mask = np.zeros(yellow.shape, dtype=np.uint8)
    mask[(yellow & (_ridge(sat) >= 12)) | (white & (_ridge(val) >= 18))] = 255
    # A lane stripe is continuous down the image: keep ridge pixels that have
    # ridge neighbours within +-4 px on each of PAINT_MIN_ROWS consecutive rows.
    # Noise is not continuous. Limit: a stripe survives only while it moves at
    # most s <= (W_ridge + 8) / 5 px per row, where W_ridge = W only up to
    # 12 px (4 px: 2 px/row; 6 px: 2 px/row, 0 at 3; 12 px: 4 px/row).
    # A flatter (more sideways) thin stripe gives 0 here: a stop, not a guess.
    joined = cv2.dilate(mask, np.ones((1, 9), np.uint8))
    mask = cv2.erode(joined, np.ones((PAINT_MIN_ROWS, 1), np.uint8))
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = 0
    for i in range(1, int(n)):
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) >= PAINT_MIN_RUN_PX:
            total += int(stats[i, cv2.CC_STAT_AREA])
    return int(total)


def _poly_ok(poly: Any) -> bool:
    for p in poly:
        x = float(p.get("x", float("nan")))
        y = float(p.get("y", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if abs(x) > LAT_MAX_M or y < Y_MIN_M or y > Y_MAX_M:
            return False
    return True


def sanitize_result(res: LaneResult) -> LaneResult:
    """Drop non-finite / out-of-bounds polylines; clamp conf and curvature."""
    polys = [list(p) for p in (res.lanes_bev or []) if p and _poly_ok(p)]
    conf = float(res.conf) if math.isfinite(float(res.conf)) else 0.0
    conf = min(1.0, max(0.0, conf)) if polys else 0.0
    curv = float(res.curvature) if math.isfinite(float(res.curvature)) else 0.0
    curv = float(np.clip(curv, -CURVATURE_MAX, CURVATURE_MAX))
    return LaneResult(conf=conf, lanes_bev=polys, curvature=curv)


def _poly_dicts(points: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y), "z": 0.0} for x, y in points]


def _curvature(polys: list[np.ndarray]) -> float:
    """Mean 2a of x = a*y^2 + b*y + c over lanes spanning at least 1 m in y, clipped."""
    curvs = []
    for pts in polys:
        ys = pts[:, 1]
        if float(np.ptp(ys)) < 1.0:
            continue
        a = np.polyfit(ys, pts[:, 0], 2)[0]
        if math.isfinite(float(a)):
            curvs.append(2.0 * float(a))
    if not curvs:
        return 0.0
    return float(np.clip(np.mean(curvs), -CURVATURE_MAX, CURVATURE_MAX))


@dataclass
class LearnedParse:
    result: LaneResult | None
    curbs_bev: list = field(default_factory=list)
    conf: float = 0.0
    reason: str = ""


def parse_learned_lanes(outputs: Mapping[str, Any], head_conf: Any = None) -> LearnedParse:
    """SceneNet ``lanes/lane_valid/curbs/curb_valid`` → painted lanes + curbs.

    Pure. Painted lanes come only from ``lanes``; ``curbs`` go to
    ``curbs_bev`` and never into ``lanes_bev`` (I2). ``result`` is None with
    a reason when the output must not be used.
    """
    try:
        lanes = np.asarray(outputs.get("lanes"), dtype=np.float64)
        lane_valid = np.asarray(outputs.get("lane_valid"), dtype=np.float64)
        curbs = np.asarray(outputs.get("curbs"), dtype=np.float64)
        curb_valid = np.asarray(outputs.get("curb_valid"), dtype=np.float64)
    except (TypeError, ValueError):
        return LearnedParse(None, reason=REASON_BAD_SHAPE)
    if (
        lanes.size != N_LANES * LANE_POINTS * 2
        or lane_valid.size != N_LANES
        or curbs.size != N_CURBS * LANE_POINTS * 2
        or curb_valid.size != N_CURBS
    ):
        return LearnedParse(None, reason=REASON_BAD_SHAPE)
    lanes = lanes.reshape(N_LANES, LANE_POINTS, 2)
    lane_valid = lane_valid.reshape(N_LANES)
    curbs = curbs.reshape(N_CURBS, LANE_POINTS, 2)
    curb_valid = curb_valid.reshape(N_CURBS)
    if not (np.isfinite(lanes).all() and np.isfinite(lane_valid).all()
            and np.isfinite(curbs).all() and np.isfinite(curb_valid).all()):
        return LearnedParse(None, reason=REASON_NONFINITE)
    if head_conf is not None:
        try:
            hc = float(head_conf)
        except (TypeError, ValueError):
            return LearnedParse(None, reason=REASON_BAD_SHAPE)
        if not math.isfinite(hc):
            return LearnedParse(None, reason=REASON_NONFINITE)
    else:
        hc = None
    if (lane_valid < 0.0).any() or (lane_valid > 1.0).any() or (curb_valid < 0.0).any() or (curb_valid > 1.0).any():
        return LearnedParse(None, reason=REASON_OUT_OF_RANGE)
    keep = [i for i in range(N_LANES) if lane_valid[i] >= LANE_VALID_MIN]
    keep_c = [i for i in range(N_CURBS) if curb_valid[i] >= LANE_VALID_MIN]

    def inside(pts: np.ndarray) -> bool:
        return bool((np.abs(pts[:, 0]) <= LAT_MAX_M).all()
                    and (pts[:, 1] >= Y_MIN_M).all() and (pts[:, 1] <= Y_MAX_M).all())

    if any(not inside(lanes[i]) for i in keep) or any(not inside(curbs[i]) for i in keep_c):
        return LearnedParse(None, reason=REASON_OUT_OF_RANGE)
    curbs_bev = [_poly_dicts(curbs[i]) for i in keep_c]
    conf = float(max(lane_valid[i] for i in keep)) if keep else 0.0
    if hc is not None:
        conf = min(conf, float(min(1.0, max(0.0, hc))))
    if not keep or conf < LEARNED_LANE_MIN_CONF:
        return LearnedParse(None, curbs_bev=curbs_bev, conf=conf, reason=REASON_LOW_CONF)
    polys = [lanes[i] for i in keep]
    res = LaneResult(conf=conf, lanes_bev=[_poly_dicts(p) for p in polys], curvature=_curvature(polys))
    return LearnedParse(res, curbs_bev=curbs_bev, conf=conf, reason="")


@dataclass
class LaneSelection:
    result: LaneResult
    source: str
    fallback_reason: str = ""
    learned_conf: float = 0.0
    curbs_bev: list = field(default_factory=list)
    paint_px: int = 0
    paint_gated: bool = False

    @property
    def fallback(self) -> bool:
        return bool(self.fallback_reason)

    def telemetry(self) -> dict[str, Any]:
        return {
            "mode": "learned",
            "source": self.source,
            "fallback": self.fallback,
            "reason": self.fallback_reason or "none",
            "learned_conf": round(float(self.learned_conf), 3),
            "paint_px": int(self.paint_px),
            "paint_gated": bool(self.paint_gated),
            "curbs_n": len(self.curbs_bev),
        }


HOUGH_ONLY_TELEMETRY = {"mode": "hough", "source": "hough", "fallback": False, "reason": "flag_off"}


def _default_runtime_factory(path: Any) -> Any:
    from python.train.scene_lane_runtime import SceneLaneRuntime

    return SceneLaneRuntime(path)


class LearnedLaneSelector:
    """One tick: learned lanes when usable, else Hough; then the paint gate.

    ``runtime_factory(path)`` builds an object with ``ready``, ``reason``,
    ``observe(strip, dt) -> (lane_x, conf)`` and ``output(name)``, i.e. a
    :class:`python.train.scene_lane_runtime.SceneLaneRuntime`.
    """

    def __init__(
        self,
        onnx_path: Any = None,
        *,
        runtime_factory: Callable[[Any], Any] | None = None,
        hough: Callable[..., LaneResult] = lanes_from_view,
    ) -> None:
        self.hough = hough
        self.runtime: Any = None
        self.load_reason = ""
        self.disabled_reason = ""
        factory = runtime_factory or _default_runtime_factory
        try:
            rt = factory(onnx_path)
        except Exception as exc:  # SceneInputShapeError, SceneCamAgeError, anything else
            self.load_reason = f"{REASON_MODEL_REFUSED}: {type(exc).__name__}: {exc}"
            return
        if rt is None or not bool(getattr(rt, "ready", False)):
            why = str(getattr(rt, "reason", "") or "not ready")
            if why.startswith("missing"):
                self.load_reason = f"{REASON_MODEL_MISSING}: {why}"
            else:
                self.load_reason = f"{REASON_MODEL_REFUSED}: {why}"
            return
        self.runtime = rt

    @property
    def ready(self) -> bool:
        return self.runtime is not None and not self.disabled_reason

    def describe(self) -> str:
        if self.runtime is None:
            return f"learned lanes ON, model not usable ({self.load_reason}); Hough fallback every tick"
        return f"learned lanes ON, model ready ({getattr(self.runtime, 'path', '?')})"

    def _learned(self, main_bgr: Any, strip: Any, dt_s: Any) -> LearnedParse:
        if self.runtime is None:
            return LearnedParse(None, reason=self.load_reason or REASON_MODEL_MISSING)
        if self.disabled_reason:
            return LearnedParse(None, reason=self.disabled_reason)
        if main_bgr is None or strip is None:
            return LearnedParse(None, reason=REASON_NO_FRAME)
        try:
            _lane_x, head_conf = self.runtime.observe(strip, dt_s)
        except Exception as exc:
            from python.train.scene_lane_runtime import SceneInputShapeError

            if isinstance(exc, SceneInputShapeError):
                # Latched for the session, like the scene guard's HOLD.
                self.disabled_reason = f"{REASON_DISABLED}: {type(exc).__name__}"
                return LearnedParse(None, reason=self.disabled_reason)
            return LearnedParse(None, reason=f"{REASON_RUNTIME_ERROR}: {type(exc).__name__}")
        if _lane_x is None:
            return LearnedParse(None, reason=REASON_RUNTIME_ERROR)
        outs: dict[str, Any] = {}
        for name in LEARNED_OUTPUTS:
            try:
                outs[name] = self.runtime.output(name)
            except Exception as exc:
                if type(exc).__name__ == "UntrainedOutputError":
                    return LearnedParse(None, reason=f"{REASON_UNTRAINED}: {name}")
                return LearnedParse(None, reason=f"{REASON_RUNTIME_ERROR}: {type(exc).__name__}")
            if outs[name] is None:
                return LearnedParse(None, reason=REASON_BAD_SHAPE)
        return parse_learned_lanes(outs, head_conf)

    def select(self, main_bgr: Any, stitch_bgr: Any = None, *, strip: Any = None, dt_s: Any = None) -> LaneSelection:
        parsed = self._learned(main_bgr, strip, dt_s)
        if parsed.result is not None:
            source = SOURCE_LEARNED
            result = parsed.result
        else:
            source = SOURCE_FALLBACK
            result = self.hough(main_bgr, stitch_bgr)
        result = sanitize_result(result)
        paint = paint_evidence_px(forward_image(main_bgr, stitch_bgr))
        gated = paint < PAINT_MIN_PX
        if gated and result.lanes_bev:
            # I1: no paint, no lines, from either source.
            result = _empty()
            source = SOURCE_NONE
        return LaneSelection(
            result=result,
            source=source,
            fallback_reason=parsed.reason,
            learned_conf=float(parsed.conf),
            curbs_bev=list(parsed.curbs_bev),
            paint_px=int(paint),
            paint_gated=bool(gated),
        )


# --- run_vision wiring -------------------------------------------------------


def build_lane_selector(args: Any, *, factory: Callable[[Any], Any] | None = None, print_fn: Callable[..., Any] = print) -> Any:
    """``LearnedLaneSelector`` for ``--learned-lanes`` with ``--lane-source hough``, else None.

    The start line always says which lane source is live and why.
    """
    on = learned_lanes_enabled(getattr(args, "learned_lanes", None))
    source = str(getattr(args, "lane_source", "hough") or "hough")
    if not on:
        print_fn("[GVD] lane head: hough (learned lanes OFF; --learned-lanes or GVD_LEARNED_LANES=1)", flush=True)
        return None
    if source != "hough":
        print_fn(f"[GVD] lane head: --learned-lanes ignored, --lane-source {source} drives lanes", flush=True)
        return None
    path = getattr(args, "scene_onnx", "") or None
    selector = LearnedLaneSelector(path, runtime_factory=factory)
    print_fn(f"[GVD] lane head: {selector.describe()}", flush=True)
    return selector


def lane_strip_for(perc: Any, bundle: Any) -> Any:
    """The 8-camera model strip, only when the learned selector is live."""
    selector = getattr(perc, "lane_selector", None)
    if selector is None or not getattr(selector, "ready", False):
        return None
    frames = getattr(bundle, "frames", None)
    if not isinstance(frames, dict) or not frames:
        return None
    from python.data.strip_writer import compose_strip

    return compose_strip(frames)


def log_lane_head_change(prev: Any, head: Any, *, print_fn: Callable[..., Any] = print) -> Any:
    """Print one line whenever the lane source or fallback reason changes."""
    if not isinstance(head, dict):
        return prev
    key = (head.get("source"), head.get("reason"), bool(head.get("paint_gated", False)))
    if key != prev:
        print_fn(
            f"[GVD] lane head: source={key[0]} reason={key[1]} paint_gated={int(key[2])} "
            f"learned_conf={head.get('learned_conf', 'na')}",
            flush=True,
        )
    return key
