"""Camera subset for the scene net.

The default canvas is still the eight-camera strip. ``--cams main`` builds
a canvas that contains only those sectors, in strip order, and the ONNX
file records that list. A horizontal pixel shift is not applied: ``lane_gt``
is in metres, and a crop does not move those metres by a known amount.
"""

from __future__ import annotations

from typing import Mapping

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
) -> dict[str, str]:
    """String metadata stored on the ONNX model."""
    return {
        "gvd_cams": ",".join(cams),
        "gvd_heads": ",".join(heads) if heads else ",".join(HEAD_NAMES),
        "gvd_canvas_h": str(int(height)),
        "gvd_canvas_w": str(int(width)),
    }


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
    model = onnx.load(path)
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
