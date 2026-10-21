"""Camera subset for the scene net.

The default canvas is still the eight-camera strip. ``--cams main`` builds
a canvas that contains only those sectors, in strip order, and the ONNX
file records that list. A horizontal pixel shift is not applied: ``lane_gt``
is in metres, and a crop does not move those metres by a known amount.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from python.data.strip_writer import (
    CANVAS_H,
    CANVAS_W,
    GAP_PX,
    SECTORS,
    STRIP_ORDER,
    WIDE_CAMS,
    SectorGeom,
    sector_width,
)

HEAD_NAMES = ("lanes", "curbs", "objects", "signs", "lane_gt")
# ONNX outputs each head writes (scene_net.OUTPUT_FIELDS, split by head). An
# export carries every output; the ones whose head was not in --heads are
# random-init and are listed in ``gvd_outputs_untrained``.
HEAD_OUTPUTS = {
    "lanes": ("lanes", "lane_valid"),
    "curbs": ("curbs", "curb_valid"),
    "objects": ("objects", "object_class", "object_valid"),
    "signs": ("signs", "sign_class", "sign_state", "sign_valid"),
    "lane_gt": ("lane_gt",),
}
_HEAD_ALIASES = {
    "lane": "lanes",
    "lane_gt": "lane_gt",
    "lane-gt": "lane_gt",
    "points": "lanes",
    "curb": "curbs",
    "object": "objects",
    "car": "objects",
    "cars": "objects",
    "sign": "signs",
    "lamps": "signs",
    "lights": "signs",
}


def parse_cams(text: str | None) -> tuple[str, ...]:
    """Strip order, restricted to the names in ``text``. Empty keeps all eight."""
    wanted = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if not wanted:
        return tuple(STRIP_ORDER)
    unknown = [name for name in wanted if name not in STRIP_ORDER]
    if unknown:
        raise ValueError(f"unknown camera {unknown[0]}")
    chosen = tuple(name for name in STRIP_ORDER if name in set(wanted))
    if not chosen:
        raise ValueError("camera list is empty")
    return chosen


def parse_heads(text: str | None) -> tuple[str, ...]:
    """Head names. Empty trains every head."""
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if not parts:
        return HEAD_NAMES
    names: list[str] = []
    for part in parts:
        key = _HEAD_ALIASES.get(part.lower(), part.lower())
        if key not in HEAD_NAMES:
            raise ValueError(f"unknown head {part}")
        if key not in names:
            names.append(key)
    return tuple(names)


def sectors_for(cams: tuple[str, ...] | list[str]) -> tuple[SectorGeom, ...]:
    """Sectors for ``cams`` in strip order, packed from the left with the usual gaps."""
    names = tuple(cams) if cams else tuple(STRIP_ORDER)
    x = 0
    out: list[SectorGeom] = []
    for index, cam_id in enumerate(names):
        width = sector_width(cam_id)
        aspect = "16:9" if cam_id in WIDE_CAMS else "4:3"
        out.append(SectorGeom(cam_id, x, x + width, width, aspect))
        x += width
        if index + 1 != len(names):
            x += GAP_PX
    return tuple(out)


def scaled_layout(
    cams: tuple[str, ...] | list[str],
    scale: float = 1.0,
) -> tuple[tuple[SectorGeom, ...], int, int]:
    """Sectors, height, and width after an optional downscale.

    Scale 1 keeps the recorded canvas, including the 2272-wide eight-camera strip.
    """
    base = sectors_for(cams)
    factor = float(scale)
    if factor >= 0.999:
        height = CANVAS_H
        width = base[-1].x1 if base else 0
        return base, height, int(width)
    gap = max(0, int(round(GAP_PX * factor)))
    packed: list[SectorGeom] = []
    x = 0
    for index, sec in enumerate(base):
        width = max(1, int(round(sec.width * factor)))
        packed.append(SectorGeom(sec.cam_id, x, x + width, width, sec.aspect))
        x += width
        if index + 1 != len(base):
            x += gap
    height = max(1, int(round(CANVAS_H * factor)))
    return tuple(packed), height, x


def scale_canvas(image: np.ndarray, scale: float, cams: tuple[str, ...] | list[str] | None = None) -> np.ndarray:
    """Resize a packed canvas to the scaled layout. Scale 1 returns the same array."""
    factor = float(scale)
    if factor >= 0.999:
        return image
    import cv2

    names = tuple(cams) if cams else tuple(STRIP_ORDER)
    _sectors, height, width = scaled_layout(names, factor)
    return cv2.resize(np.asarray(image), (int(width), int(height)), interpolation=cv2.INTER_AREA)


def canvas_width(cams: tuple[str, ...] | list[str] | None = None) -> int:
    sectors = sectors_for(tuple(cams) if cams else tuple(STRIP_ORDER))
    if not sectors:
        return 0
    return int(sectors[-1].x1)


def full_sector(cam_id: str) -> SectorGeom | None:
    for sec in SECTORS:
        if sec.cam_id == cam_id:
            return sec
    return None


def extract_cams(full_bgr: np.ndarray, cams: tuple[str, ...]) -> np.ndarray:
    """Repack sectors out of a full 2272-wide strip. A full-width request is a copy."""
    if tuple(cams) == tuple(STRIP_ORDER):
        return np.ascontiguousarray(full_bgr)
    sectors = sectors_for(cams)
    canvas = np.zeros((CANVAS_H, canvas_width(cams), 3), dtype=np.uint8)
    src = np.asarray(full_bgr)
    for sec in sectors:
        origin = full_sector(sec.cam_id)
        if origin is None or src.ndim != 3 or int(src.shape[1]) < origin.x1:
            continue
        canvas[:, sec.x0 : sec.x1] = src[:, origin.x0 : origin.x1]
    return canvas


def prepare_model_strip(image: np.ndarray, cams: tuple[str, ...], canvas_w: int) -> np.ndarray | None:
    """Return the canvas the exported net expects.

    A strip that is already that width is used as-is. A full eight-camera
    strip is cut down to ``cams``.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3 or int(arr.shape[0]) != CANVAS_H:
        return None
    if int(arr.shape[1]) == int(canvas_w):
        return np.ascontiguousarray(arr)
    if int(arr.shape[1]) == CANVAS_W:
        return extract_cams(arr, cams)
    return None


def photometric_jitter(image: np.ndarray, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Brightness, contrast, and gamma. The lane labels stay in metres.

    A sideways crop would need a pixel-to-metre scale at every lookahead.
    That scale is not in the label, so the pixels move and the metres do not.
    """
    rng = rng or np.random.RandomState()
    src = np.asarray(image)
    scale = 255.0 if float(src.max()) > 1.5 else 1.0
    values = src.astype(np.float32) / scale
    values = values + float(rng.uniform(-0.08, 0.08))
    contrast = float(rng.uniform(0.85, 1.15))
    values = (values - 0.5) * contrast + 0.5
    gamma = float(rng.uniform(0.8, 1.2))
    values = np.clip(values, 0.0, 1.0) ** gamma
    out = np.clip(values * scale, 0.0, scale)
    if src.dtype == np.uint8:
        return np.ascontiguousarray(out.astype(np.uint8))
    return np.ascontiguousarray(out.astype(src.dtype))


def metadata_dict(
    cams: tuple[str, ...],
    heads: tuple[str, ...] | None,
    *,
    height: int,
    width: int,
    cam_age: bool = False,
) -> dict[str, str]:
    """String metadata stored on the ONNX model.

    ``gvd_cam_age`` is ``1`` when the graph takes the ``ages`` input (8
    per-camera hold ages, ``cam_hold.CAM_AGE_ORDER``), else ``0``. A model
    without the key is an older export: no ages (``cam_age_flag``).

    ``gvd_heads_trained`` is the heads the run trained (``--heads``; empty =
    all). ``gvd_outputs_untrained`` lists every exported output of a head that
    was not trained: those values are random-init and no reader may use them
    (``untrained_outputs`` / ``require_trained``).
    """
    trained = tuple(heads) if heads else HEAD_NAMES
    untrained = [out for head in HEAD_NAMES if head not in trained for out in HEAD_OUTPUTS[head]]
    from python.data.cam_hold import (
        CAM_AGE_CAP_META,
        CAM_AGE_HZ_META,
        CAM_AGE_META,
        CAM_AGE_ORDER,
        CAM_AGE_ORDER_META,
        HOLD_CAP_S,
        HOLD_TICK_HZ,
    )

    props = {
        "gvd_cams": ",".join(cams),
        "gvd_heads": ",".join(trained),
        "gvd_heads_trained": ",".join(trained),
        "gvd_outputs_untrained": ",".join(untrained),
        "gvd_canvas_h": str(int(height)),
        "gvd_canvas_w": str(int(width)),
        CAM_AGE_META: "1" if cam_age else "0",
    }
    if cam_age:
        props[CAM_AGE_ORDER_META] = ",".join(CAM_AGE_ORDER)
        props[CAM_AGE_CAP_META] = repr(float(HOLD_CAP_S))
        props[CAM_AGE_HZ_META] = str(int(HOLD_TICK_HZ))
    return props


class CamAgeFlagError(ValueError):
    """The model's ``gvd_cam_age`` metadata cannot be read or disagrees with its inputs."""


def cam_age_flag(meta: Mapping[str, str] | None) -> bool:
    """True when the model takes ``ages``. Absent is an older export (False).

    Anything other than ``0`` / ``1``, or a ``1`` whose order, cap or tick is
    not this build's, raises ``CamAgeFlagError``.
    """
    from python.data.cam_hold import (
        CAM_AGE_CAP_META,
        CAM_AGE_HZ_META,
        CAM_AGE_META,
        CAM_AGE_ORDER,
        CAM_AGE_ORDER_META,
        HOLD_CAP_S,
        HOLD_TICK_HZ,
    )

    meta = dict(meta or {})
    raw = meta.get(CAM_AGE_META)
    if raw is None or str(raw).strip() == "0":
        return False
    if str(raw).strip() != "1":
        raise CamAgeFlagError(f"{CAM_AGE_META}={raw!r} is not 0 or 1")
    order = str(meta.get(CAM_AGE_ORDER_META) or "")
    if order != ",".join(CAM_AGE_ORDER):
        raise CamAgeFlagError(f"{CAM_AGE_ORDER_META}={order!r}, this build feeds {','.join(CAM_AGE_ORDER)}")
    try:
        cap = float(meta.get(CAM_AGE_CAP_META))
        hz = int(meta.get(CAM_AGE_HZ_META))
    except (TypeError, ValueError) as exc:
        raise CamAgeFlagError(f"{CAM_AGE_CAP_META}/{CAM_AGE_HZ_META} missing or unreadable") from exc
    if cap != float(HOLD_CAP_S) or hz != int(HOLD_TICK_HZ):
        raise CamAgeFlagError(
            f"model ages are cap={cap} s at {hz} Hz; this build feeds cap={HOLD_CAP_S} s at {HOLD_TICK_HZ} Hz"
        )
    return True


class UntrainedOutputError(RuntimeError):
    """A reader asked for an ONNX output whose head was not trained."""


def untrained_outputs(meta: Mapping[str, str] | None) -> frozenset[str]:
    """Outputs the model marks untrained.

    ``gvd_outputs_untrained`` when present; else derived from
    ``gvd_heads_trained`` / ``gvd_heads``. A model without any of them (an
    export older than this metadata) marks nothing.
    """
    meta = dict(meta or {})
    if "gvd_outputs_untrained" in meta:
        return frozenset(p.strip() for p in str(meta["gvd_outputs_untrained"]).split(",") if p.strip())
    raw = meta.get("gvd_heads_trained", meta.get("gvd_heads"))
    if raw is None:
        return frozenset()
    trained = {p.strip() for p in str(raw).split(",") if p.strip()}
    return frozenset(out for head in HEAD_NAMES if head not in trained for out in HEAD_OUTPUTS[head])


def require_trained(meta: Mapping[str, str] | None, outputs: Iterable[str]) -> None:
    """Raise ``UntrainedOutputError`` when any of ``outputs`` is marked untrained."""
    bad = sorted(set(outputs) & untrained_outputs(meta))
    if bad:
        raise UntrainedOutputError(
            f"ONNX outputs {','.join(bad)} come from heads this model did not train "
            f"(gvd_heads_trained={dict(meta or {}).get('gvd_heads_trained', dict(meta or {}).get('gvd_heads', '?'))})"
        )


def cams_from_metadata(meta: Mapping[str, str] | None) -> tuple[str, ...]:
    """Cameras recorded on the model. Missing metadata is the full strip."""
    if not meta:
        return tuple(STRIP_ORDER)
    try:
        return parse_cams(str(meta.get("gvd_cams") or ""))
    except ValueError:
        return tuple(STRIP_ORDER)


def stamp_onnx_metadata(path: str, props: Mapping[str, str]) -> bool:
    """Write ``props`` onto an ONNX file. False when the onnx package is absent."""
    try:
        import onnx  # type: ignore
    except ImportError:
        return False
    try:
        model = onnx.load(path)
    except Exception:
        return False
    kept = [item for item in model.metadata_props if item.key not in props]
    del model.metadata_props[:]
    for item in kept:
        entry = model.metadata_props.add()
        entry.key = item.key
        entry.value = item.value
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        entry.value = str(value)
    onnx.save(model, path)
    return True
