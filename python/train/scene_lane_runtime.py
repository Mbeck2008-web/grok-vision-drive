"""Lane center from the scene-net ONNX graph.

The supervisor asks this for ``lane_x`` when ``--lane-source scene`` is set.
A missing file, a missing runtime, or a confidence under the floor is a stop,
not a return to the pixel lane finder. The step time is the real seconds
since the previous strip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from python.debug.lane_targets import (
    LANE_GT_DIM,
    confidence_from_vector,
    lane_x_from_vector,
)
from python.train.scene_net import (
    GRU_HIDDEN,
    RESIDUAL_S,
    RESIDUAL_TOKENS,
    SECTOR_FEAT,
    TOKEN_DIM,
    default_onnx_path,
)

SCENE_LANE_MIN_CONF = 0.40


class SceneInputShapeError(ValueError):
    """The ONNX graph's strip input and the live canvas disagree in size.

    The exported graph has fixed input dims, and the ``--backbone`` net's
    pooling bins are traced for the training canvas, so a different size can
    never be fed or silently resized.
    """


class SceneCameraContractError(SceneInputShapeError):
    """The model's trained camera list (``gvd_cams``) is absent, or names a
    camera the live supervisor does not feed. Refused at load like a shape
    mismatch: the net would otherwise see a black sector it never trained on.
    """


class SceneHeadsContractError(SceneInputShapeError):
    """The model's trained-heads list (``gvd_heads``) is absent on a live load,
    unreadable, or does not hold ``lane_gt``. Refused at load like ``gvd_cams``.
    """


# ONNX outputs per head. Only the outputs of a head on the model's gvd_heads
# list are ever read; the others are dropped the moment the graph returns.
HEAD_OUTPUTS: dict[str, tuple[str, ...]] = {
    "lanes": ("lanes", "lane_valid"),
    "curbs": ("curbs", "curb_valid"),
    "objects": ("objects", "object_class", "object_valid"),
    "signs": ("signs", "sign_class", "sign_state", "sign_valid"),
    "lane_gt": ("lane_gt",),
}
STATE_OUTPUTS = ("h", "tokens", "mask")


def check_heads_contract(meta: Any, *, required: bool = True) -> tuple[str, ...]:
    """The model's trained heads (``gvd_heads``).

    Live (``required``), a model without the list is refused, the same as a
    model without ``gvd_cams``. Off the live path a model without it is
    treated as lane-only. A list without ``lane_gt`` is always refused: the
    scene lane source drives on nothing else.
    """
    from python.train.cam_strip import parse_heads

    raw = str((meta or {}).get("gvd_heads") or "").strip() if isinstance(meta, dict) else ""
    if not raw:
        if required:
            raise SceneHeadsContractError(
                "model has no gvd_heads metadata (trained heads); re-export it with this trainer"
            )
        return ("lane_gt",)
    try:
        heads = parse_heads(raw)
    except ValueError as exc:
        raise SceneHeadsContractError(f"model gvd_heads={raw!r} is not a head list: {exc}") from exc
    if "lane_gt" not in heads:
        raise SceneHeadsContractError(f"model gvd_heads={raw!r} has no lane_gt; the scene lane source needs it")
    return heads


def trained_outputs(by_name: Any, heads: Any) -> dict[str, Any]:
    """Only the outputs of trained heads (plus the recurrent state). An
    untrained head's output (e.g. objects/signs/curbs trained on no labels)
    is never returned, so nothing downstream can draw or plan on it."""
    keep = set(STATE_OUTPUTS)
    for head in heads or ():
        keep.update(HEAD_OUTPUTS.get(str(head), ()))
    return {name: value for name, value in dict(by_name or {}).items() if name in keep}


def live_cams_from_config(cfg: Any) -> tuple[str, ...]:
    """Camera ids the live supervisor attaches, from ``config/cameras.yaml``."""
    out: list[str] = []
    items = cfg.get("cameras") if isinstance(cfg, dict) else None
    for item in items or []:
        cam = str(item.get("id") or "") if isinstance(item, dict) else ""
        if cam and cam not in out:
            out.append(cam)
    return tuple(out)


def check_camera_contract(meta: Any, live_cams: Any) -> tuple[str, ...]:
    """The model's ``gvd_cams``; refuse a missing list or a camera not fed live.

    Live feeds may include more cameras than the model (the runtime cuts the
    model's sectors out of the live strip); every model camera must be fed.
    """
    from python.train.cam_strip import parse_cams

    raw = str((meta or {}).get("gvd_cams") or "").strip() if isinstance(meta, dict) else ""
    if not raw:
        raise SceneCameraContractError(
            "model has no gvd_cams metadata (trained camera list); re-export it with this trainer"
        )
    try:
        model_cams = parse_cams(raw)
    except ValueError as exc:
        raise SceneCameraContractError(f"model gvd_cams={raw!r} is not a camera list: {exc}") from exc
    live = tuple(str(cam) for cam in (live_cams or ()))
    absent = [cam for cam in model_cams if cam not in live]
    if absent:
        raise SceneCameraContractError(
            f"model trained on cams={','.join(model_cams)} but the live feed has {','.join(live) or 'none'} "
            f"(missing {','.join(absent)}); refusing to drive on black sectors"
        )
    return model_cams


def _base_name(name: str) -> str:
    """``h.1`` -> ``h``: the tracer renames pass-through state inputs."""
    return str(name).split(".", 1)[0]


def _fixed_dims(shape: Any) -> tuple[int, ...] | None:
    dims = tuple(shape or ())
    if not dims or not all(isinstance(d, int) and d > 0 for d in dims):
        return None
    return tuple(int(d) for d in dims)


def scene_lane_decision(
    lane_x: float | None,
    confidence: float | None,
    *,
    min_conf: float = SCENE_LANE_MIN_CONF,
) -> str:
    """``lane`` when the net is sure, otherwise ``stop``."""
    if lane_x is None or confidence is None:
        return "stop"
    try:
        conf = float(confidence)
        x = float(lane_x)
    except (TypeError, ValueError):
        return "stop"
    if conf != conf or x != x or conf < float(min_conf):
        return "stop"
    return "lane"


def straight_lane_path(lane_x: float, *, length_m: float = 36.0, step_m: float = 1.0) -> list[dict[str, float]]:
    """Ego polyline the wheel already knows how to follow. x is constant."""
    path: list[dict[str, float]] = []
    y = 0.0
    guard = 0
    while y <= float(length_m) + 1e-6 and guard < 80:
        path.append({"x": float(lane_x), "y": float(y), "z": 0.0})
        y += float(step_m)
        guard += 1
    return path


SCENE_SHAPE_HOLD = "scene_hold:input_shape"


class SceneLaneGuard:
    """The supervisor's handler for :class:`SceneInputShapeError`.

    A load-time error refuses Engage: ``refused`` holds the message and
    ``load`` returns no runtime. A live error is a HOLD: ``observe`` returns
    ``(None, 0.0)``, ``hold`` holds the message, and the caller sends
    ``stop_command(reason=SCENE_SHAPE_HOLD)``. The HOLD stays latched (the
    runtime is not asked again) until the car is disengaged, so a strip that
    comes back the right size cannot resume steering on its own. The
    exception never leaves this class.
    """

    def __init__(self) -> None:
        self.refused: str | None = None
        self.hold: str | None = None

    def load(self, factory: Any, path: Any) -> Any:
        try:
            return factory(path)
        except SceneInputShapeError as exc:
            self.refused = str(exc)
            return None

    def observe(self, runtime: Any, strip_bgr: Any, dt_s: float | None, *, engaged: bool) -> tuple[float | None, float]:
        if not engaged:
            self.hold = None
        if self.hold is not None:
            return None, 0.0
        try:
            return runtime.observe(strip_bgr, dt_s)
        except SceneInputShapeError as exc:
            self.hold = str(exc)
            return None, 0.0

    def status(self) -> dict[str, Any]:
        """``scene_lane`` block for gvd_state.json."""
        if self.refused is not None:
            return {"state": "refused", "reason": SCENE_SHAPE_HOLD, "detail": self.refused}
        if self.hold is not None:
            return {"state": "hold", "reason": SCENE_SHAPE_HOLD, "detail": self.hold}
        return {"state": "ok", "reason": "none", "detail": ""}


class SceneLaneRuntime:
    """One ONNX session and its GRU state. Not tied to a fixed frame rate."""

    def __init__(self, path: Path | str | None = None, *, live_cams: Any = None) -> None:
        """``live_cams``: cameras the supervisor feeds. Set, the model's
        ``gvd_cams`` must be present and fed (``check_camera_contract``)."""
        self.path = Path(path) if path is not None else default_onnx_path()
        self.live_cams = None if live_cams is None else tuple(str(cam) for cam in live_cams)
        self.session: Any = None
        self.ready = False
        self.reason = "not loaded"
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self.cams: tuple[str, ...] = ()
        self.heads: tuple[str, ...] = ()
        # Last outputs of trained heads only (see trained_outputs).
        self.last_outputs: dict[str, Any] = {}
        self.canvas_w = 0
        self.token_dim = TOKEN_DIM
        self.h = np.zeros((1, 1, GRU_HIDDEN), dtype=np.float32)
        self.tokens = np.zeros((1, RESIDUAL_TOKENS, TOKEN_DIM), dtype=np.float32)
        self.mask = np.full((1, RESIDUAL_TOKENS), float(RESIDUAL_S) + 1.0, dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            self.reason = f"missing {self.path}"
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            self.reason = f"onnxruntime missing ({type(exc).__name__})"
            return
        try:
            self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            self._input_names = [item.name for item in self.session.get_inputs()]
            self._output_names = [item.name for item in self.session.get_outputs()]
        except Exception as exc:
            self.session = None
            self.reason = f"onnx load failed ({type(exc).__name__})"
            return
        if "lane_gt" not in self._output_names:
            self.session = None
            self.reason = "onnx has no lane_gt output; retrain and export"
            return
        from python.data.strip_writer import CANVAS_H, CANVAS_W
        from python.train.cam_strip import cams_from_metadata

        meta: dict[str, str] = {}
        try:
            meta = dict(self.session.get_modelmeta().custom_metadata_map)
        except Exception:
            meta = {}
        if self.live_cams is not None:
            try:
                check_camera_contract(meta, self.live_cams)
            except SceneCameraContractError as exc:
                self.session = None
                raise SceneCameraContractError(f"{self.path}: {exc}") from exc
        try:
            self.heads = check_heads_contract(meta, required=self.live_cams is not None)
        except SceneHeadsContractError as exc:
            self.session = None
            raise SceneHeadsContractError(f"{self.path}: {exc}") from exc
        self.cams = cams_from_metadata(meta)
        try:
            self.canvas_w = int(meta.get("gvd_canvas_w") or CANVAS_W)
        except (TypeError, ValueError):
            self.canvas_w = CANVAS_W
        self.token_dim = len(self.cams) * SECTOR_FEAT
        self.tokens = np.zeros((1, RESIDUAL_TOKENS, self.token_dim), dtype=np.float32)
        inputs = {_base_name(item.name): item for item in self.session.get_inputs()}
        strip_in = inputs.get("strip")
        dims = None if strip_in is None else _fixed_dims(strip_in.shape)
        want = (1, 3, int(CANVAS_H), int(self.canvas_w))
        if dims is None or dims != want:
            self.session = None
            raise SceneInputShapeError(
                f"{self.path}: strip input is {None if strip_in is None else list(strip_in.shape)}, "
                f"the live canvas is {list(want)} (cams={','.join(self.cams)}, gvd_canvas_w={self.canvas_w}). "
                "Re-export at the live canvas size; the graph has fixed dims and is never resized."
            )
        # State tensors take the graph's own shapes (the --backbone net carries a 1-wide token).
        for key in ("h", "tokens", "mask"):
            item = inputs.get(key)
            shape = None if item is None else _fixed_dims(item.shape)
            if shape is not None and shape != tuple(getattr(self, key).shape):
                fill = float(RESIDUAL_S) + 1.0 if key == "mask" else 0.0
                setattr(self, key, np.full(shape, fill, dtype=np.float32))
        if "tokens" in inputs:
            self.token_dim = int(self.tokens.shape[-1])
        self.ready = True
        self.reason = "ok"

    def observe(self, strip_bgr: Any, dt_s: float | None) -> tuple[float | None, float]:
        """Return ``(lane_x at 8 m, confidence)``. A cold runtime is confidence 0."""
        if not self.ready or self.session is None or strip_bgr is None:
            return None, 0.0
        from python.train.cam_strip import prepare_model_strip

        prepared = prepare_model_strip(np.asarray(strip_bgr), self.cams, self.canvas_w)
        image = None if prepared is None else _strip_nchw(prepared, width=self.canvas_w)
        if image is None:
            from python.data.strip_writer import CANVAS_H

            raise SceneInputShapeError(
                f"live strip {tuple(np.asarray(strip_bgr).shape)} does not give the model canvas "
                f"{CANVAS_H}x{self.canvas_w}x3 (cams={','.join(self.cams)}); it is not resized to fit"
            )
        dt = np.asarray([0.0 if dt_s is None else float(dt_s)], dtype=np.float32)
        feeds = {
            "strip": image,
            "dt": dt,
            "h": self.h,
            "tokens": self.tokens,
            "mask": self.mask,
        }
        feeds = {name: feeds[_base_name(name)] for name in self._input_names if _base_name(name) in feeds}
        try:
            outputs = self.session.run(None, feeds)
        except Exception:
            return None, 0.0
        by_name = trained_outputs(
            {_base_name(name): value for name, value in zip(self._output_names, outputs)}, self.heads
        )
        self.last_outputs = {name: value for name, value in by_name.items() if name not in STATE_OUTPUTS}
        vector = np.asarray(by_name.get("lane_gt"), dtype=np.float32).reshape(-1)
        if vector.size < LANE_GT_DIM:
            return None, 0.0
        if "h" in by_name:
            self.h = np.asarray(by_name["h"], dtype=np.float32)
        if "tokens" in by_name:
            self.tokens = np.asarray(by_name["tokens"], dtype=np.float32)
        if "mask" in by_name:
            self.mask = np.asarray(by_name["mask"], dtype=np.float32)
        return lane_x_from_vector(vector), confidence_from_vector(vector)


def _strip_nchw(strip_bgr: Any, width: int | None = None) -> np.ndarray | None:
    from python.data.strip_writer import CANVAS_H, CANVAS_W

    arr = np.asarray(strip_bgr)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    expect_w = CANVAS_W if width is None else int(width)
    if arr.shape[0] != CANVAS_H or arr.shape[1] != expect_w:
        return None
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if float(arr.max()) > 1.5:
        arr = arr / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...])
