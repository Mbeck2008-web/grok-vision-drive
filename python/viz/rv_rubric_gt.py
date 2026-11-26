"""Reality View step 8(1): the C-from-GT quality rubric (scorer only).

Spec: docs/specs/reality_view_3dgs.md, "8. Kill or graduate (quality rubric)"
item (1) and "Step 8 quality rubric (proposed, unmeasured)".

GT-NAMED MODULE. This scores a render against sim GT, so it is offline-only:
no live path may import it (``check_no_sim_gt_on_live_path``: a GT fixture is
only consumed by a test named GT; see ``find_non_gt_importers``). It contains
no renderer, no unprojection source and no safety-path imports; numpy only.

Conventions (my choice where the spec is silent, labeled in the PR):
* Camera frame is OpenCV (x right, y down, z forward); depth is z-depth in
  metres; ``t_ego_cam`` is camera -> ego; ego is x forward, y left, z up.
* A rendered-depth pixel is a *hole* when it is non-finite or <= 0.
* A pixel is *covered* (has a splat / billboard) when opacity >= 0.1 AND its
  rendered depth is valid. ``opacity=None`` means "every valid-depth pixel is
  opaque", because ``RvOutputSpec`` carries no opacity channel.
* AbsRel and delta < 1.25 are computed over pixels that are covered AND have
  valid GT depth in 0.5-80 m. Holes are not scored there; the coverage gate
  is what bounds them. Zero scored pixels fails; it never passes vacuously.
* The depth buffer is REQUIRED. A render without a usable one fails AbsRel
  and delta < 1.25 (never skips, never raises); see ``score_metrics``.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from python.viz import rv_contract as _rvc

# ---- spec thresholds (all "proposed, unmeasured" in the spec) ----------------
ABS_REL_MAX = 0.20                      # AbsRel <= 0.20, GT 0.5-80 m
DELTA_125_MIN = 0.70                    # fraction with max(d/d*, d*/d) < 1.25 >= 0.70
COVERAGE_MIN = 0.50                     # >= 50% of GT-valid 0.5-40 m chase pixels
COVERAGE_OPACITY_MIN = 0.1              # splat / billboard opacity >= 0.1
PSNR_MIN_DB = 16.0
SSIM_MIN = 0.45
GT_DEPTH_RANGE_M = (0.5, 80.0)          # AbsRel / delta
GT_COVERAGE_RANGE_M = (0.5, 40.0)       # coverage
VEHICLE_CENTER_ERR_MAX_M = 3.0          # checklist: known vehicle centre error <= 3 m
STALE_AGE_MS = 200.0                    # checklist: stale indicator when age > 200 ms

# ---- my choices where the spec names no number (see module docstring) --------
SSIM_WINDOW = 7                          # uniform window, per channel, valid region
UNOBSERVED_OPAQUE_MAX_FRAC = 0.01        # "not a fake wall": <= 1% of unobserved px opaque
GHOST_MIN_FRAC_OF_VEHICLE = 0.25         # spurious elevated surface >= 25% of the vehicle's px
GHOST_HEIGHT_MIN_M = 0.5                 # above the GT road plane
GHOST_DEPTH_TOL_M = 1.0                  # off-surface if |d - d*| > max(1 m, 20% d*)
GHOST_EDGE_DILATE_PX = 2                 # ignore edge bleed around the vehicle mask
GHOST_OPEN_PX = 1                        # morphological opening: 1-2 px silhouette bleed is not a vehicle
UNOBSERVED_EDGE_ERODE_PX = 2             # ignore billboard bleed at the unobserved region's border
ROAD_MIN_COVERAGE = COVERAGE_MIN         # road is a surface if >= 50% of road px covered ...
ROAD_ABS_REL_MAX = ABS_REL_MAX           # ... at <= 0.20 AbsRel (not a depth-less spray)

CHECKLIST_ITEMS: tuple[str, ...] = (
    "road_surface",
    "ego_pose_lines_up",
    "vehicle_placed_once",
    "unobserved_not_wall",
    "no_sim_gt_on_live_path",
    "stale_indicator",
)
METRIC_ITEMS: tuple[str, ...] = ("abs_rel", "delta_125", "coverage", "psnr_db", "ssim")


@dataclass(frozen=True)
class GtFrame:
    """Sim GT at the chase (or held-out) pose. Offline only."""

    depth: np.ndarray               # float (H, W), z-depth m, non-finite = no GT
    rgb: np.ndarray                 # uint8 (H, W, 3), that camera's GT RGB
    intrinsics: np.ndarray          # (3, 3)
    t_ego_cam: np.ndarray           # (4, 4) camera -> ego
    road_mask: np.ndarray           # bool (H, W)
    vehicle_mask: np.ndarray        # bool (H, W), the known vehicle
    unobserved_mask: np.ndarray     # bool (H, W), no camera coverage of this surface
    vehicle_center_ego: np.ndarray  # (3,) known vehicle centre, ego frame


@dataclass(frozen=True)
class RenderedView:
    out: _rvc.RvOutputSpec
    opacity: np.ndarray | None = None  # float (H, W) in [0, 1]; None = valid depth is opaque


@dataclass(frozen=True)
class Harness:
    """Scripted inputs for the checklist items that are not pixels."""

    render_ego_poses: np.ndarray    # (N, 4, 4) ego pose the render used, per tick
    mesh_ego_poses: np.ndarray      # (N, 4, 4) pose of the existing ego mesh / ribbon
    ego_footprint_m: tuple[float, float] = (4.5, 1.9)  # (length, width)


@dataclass
class RubricResult:
    passed: bool
    items: dict[str, bool]
    values: dict[str, Any]
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _usable_depth(out: _rvc.RvOutputSpec) -> tuple[np.ndarray | None, str | None]:
    """The rendered-depth buffer if it satisfies the contract, else (None, why)."""
    try:
        _rvc.validate_rv_output(out)
    except _rvc.RvSchemaError as exc:
        return None, f"rendered depth buffer missing or invalid: {exc}"
    return np.asarray(out.depth, dtype=np.float64), None


def _covered(depth: np.ndarray, opacity: np.ndarray | None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if opacity is None:
        return valid
    op = np.asarray(opacity, dtype=np.float64)
    if op.shape != depth.shape:
        return np.zeros(depth.shape, dtype=bool)
    return valid & (op >= COVERAGE_OPACITY_MIN)


def _gt_in_range(gt_depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    d = np.asarray(gt_depth, dtype=np.float64)
    return np.isfinite(d) & (d >= lo) & (d <= hi)


def _box_mean(a: np.ndarray, win: int) -> np.ndarray:
    """Valid-region uniform filter of a 2D float64 array via an integral image."""
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    s = c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]
    return s / float(win * win)


def psnr_db(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse)


def psnr_passes(value: float) -> bool:
    """Inclusive published gate: PSNR >= 16 dB. Exact 16.0 must pass."""
    return float(value) >= PSNR_MIN_DB


def ssim_passes(value: float) -> bool:
    """Inclusive published gate: SSIM >= 0.45. Exact 0.45 must pass."""
    return float(value) >= SSIM_MIN


def ssim(a: np.ndarray, b: np.ndarray, win: int = SSIM_WINDOW) -> float:
    """Mean SSIM over per-channel uniform windows (K1=0.01, K2=0.03, L=255)."""
    if a.shape != b.shape or a.ndim != 3 or min(a.shape[0], a.shape[1]) < win:
        raise ValueError(f"ssim needs equal (H, W, C) images at least {win}px: {a.shape} vs {b.shape}")
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    vals = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        mx, my = _box_mean(x, win), _box_mean(y, win)
        vx = _box_mean(x * x, win) - mx * mx
        vy = _box_mean(y * y, win) - my * my
        cxy = _box_mean(x * y, win) - mx * my
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        vals.append(float(np.mean(s)))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# metrics (Step 8 table)
# ---------------------------------------------------------------------------
def score_metrics(view: RenderedView, gt: GtFrame) -> RubricResult:
    """Spec Step 8 metric table on one render vs sim GT. All five must pass."""
    items = {k: False for k in METRIC_ITEMS}
    values: dict[str, Any] = {k: None for k in METRIC_ITEMS}
    failures: list[str] = []

    depth, why = _usable_depth(view.out)
    if depth is None:
        # Spec: "A run that does not emit that buffer fails these two metrics
        # (it does not skip them)."
        failures.append(f"abs_rel/delta_125: {why}")
    else:
        cov_px = _covered(depth, view.opacity)
        gt_d = np.asarray(gt.depth, dtype=np.float64)
        # coverage
        near = _gt_in_range(gt_d, *GT_COVERAGE_RANGE_M)
        n_near = int(near.sum())
        if n_near == 0:
            failures.append("coverage: no GT-valid pixels in 0.5-40 m")
        else:
            values["coverage"] = float((cov_px & near).sum()) / n_near
            items["coverage"] = values["coverage"] >= COVERAGE_MIN
        # AbsRel and delta < 1.25
        scored = cov_px & _gt_in_range(gt_d, *GT_DEPTH_RANGE_M)
        if not scored.any():
            failures.append("abs_rel/delta_125: no covered pixel with valid GT depth in 0.5-80 m")
        else:
            d, g = depth[scored], gt_d[scored]
            values["abs_rel"] = float(np.mean(np.abs(d - g) / g))
            values["delta_125"] = float(np.mean(np.maximum(d / g, g / d) < 1.25))
            items["abs_rel"] = values["abs_rel"] <= ABS_REL_MAX
            items["delta_125"] = values["delta_125"] >= DELTA_125_MIN

    rgb = np.asarray(view.out.rgb) if view.out.rgb is not None else None
    if rgb is None or rgb.shape != gt.rgb.shape:
        failures.append("psnr_db/ssim: rendered RGB missing or not the GT camera's shape")
    else:
        values["psnr_db"] = psnr_db(rgb, gt.rgb)
        values["ssim"] = ssim(rgb, gt.rgb)
        items["psnr_db"] = psnr_passes(values["psnr_db"])
        items["ssim"] = ssim_passes(values["ssim"])

    failures.extend(f"{k} failed: {values[k]}" for k in METRIC_ITEMS if not items[k] and values[k] is not None)
    return RubricResult(all(items.values()), items, values, failures)


def score_slot_metrics(slot: bytes | bytearray | memoryview, opacity: np.ndarray | None, gt: GtFrame) -> RubricResult:
    """Score a render the way the viz process publishes it: through the SHM slot.

    A slot the contract rejects (no RV_FLAG_HAS_DEPTH, truncated depth payload,
    bad magic) is a render without a usable depth buffer: the metrics fail.
    """
    try:
        _, out = _rvc.unpack_output_slot(slot)
    except _rvc.RvSchemaError as exc:
        items = {k: False for k in METRIC_ITEMS}
        values = {k: None for k in METRIC_ITEMS}
        return RubricResult(False, items, values, [f"abs_rel/delta_125: rendered depth buffer missing or invalid: {exc}"])
    return score_metrics(RenderedView(out=out, opacity=opacity), gt)


# ---------------------------------------------------------------------------
# checklist (six yes/no items)
# ---------------------------------------------------------------------------
def _ground_height(gt: GtFrame) -> float:
    pts = _unproject_px(np.asarray(gt.depth, dtype=np.float64), gt, gt.road_mask & np.isfinite(gt.depth))
    return float(np.median(pts[:, 2])) if len(pts) else 0.0


def _unproject_px(depth: np.ndarray, gt: GtFrame, mask: np.ndarray) -> np.ndarray:
    """Ego-frame points for the masked pixels of a depth image in the GT camera."""
    v, u = np.nonzero(mask)
    z = depth[v, u]
    k = np.asarray(gt.intrinsics, dtype=np.float64)
    pc = np.stack([(u + 0.5 - k[0, 2]) / k[0, 0] * z, (v + 0.5 - k[1, 2]) / k[1, 1] * z, z], axis=-1)
    t = np.asarray(gt.t_ego_cam, dtype=np.float64)
    return pc @ t[:3, :3].T + t[:3, 3]


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square-window dilation, no wrap-around at the image border."""
    h, w = mask.shape
    pad = np.pad(mask, r, constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= pad[dy : dy + h, dx : dx + w]
    return out


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    return ~_dilate(~mask, r)


def _item_road_surface(depth: np.ndarray, view: RenderedView, gt: GtFrame, failures: list[str]) -> bool:
    road = np.asarray(gt.road_mask, dtype=bool) & _gt_in_range(gt.depth, *GT_COVERAGE_RANGE_M)
    if not road.any():
        failures.append("road_surface: no GT road pixels in 0.5-40 m")
        return False
    cov = _covered(depth, view.opacity)
    frac = float((cov & road).sum()) / float(road.sum())
    both = cov & road
    err = float(np.mean(np.abs(depth[both] - gt.depth[both]) / gt.depth[both])) if both.any() else math.inf
    ok = frac >= ROAD_MIN_COVERAGE and err <= ROAD_ABS_REL_MAX
    if not ok:
        failures.append(f"road_surface: road coverage {frac:.3f}, road AbsRel {err:.3f}")
    return ok


def _footprint_overlap(rel: np.ndarray, length: float, width: float) -> float:
    """Fraction of the render footprint (sampled grid) inside the mesh footprint."""
    xs = np.linspace(-length / 2, length / 2, 21)
    ys = np.linspace(-width / 2, width / 2, 9)
    gx, gy = np.meshgrid(xs, ys)
    p = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], axis=0)
    q = rel @ p
    inside = (np.abs(q[0]) <= length / 2) & (np.abs(q[1]) <= width / 2)
    return float(inside.mean())


def _item_ego_pose(h: Harness, failures: list[str]) -> bool:
    rp = np.asarray(h.render_ego_poses, dtype=np.float64)
    mp = np.asarray(h.mesh_ego_poses, dtype=np.float64)
    if rp.ndim != 3 or rp.shape != mp.shape or rp.shape[1:] != (4, 4) or rp.shape[0] == 0 or not np.all(np.isfinite(rp)) or not np.all(np.isfinite(mp)):
        failures.append("ego_pose_lines_up: harness poses must be finite matching (N, 4, 4)")
        return False
    length, width = h.ego_footprint_m
    # Spec: "no >= 5 m lateral jump (scripted: ego footprint overlap with the mesh
    # projection)". Overlap is the scripted test, and it is stricter than the 5 m
    # figure: a footprint of width w only overlaps its mesh while the lateral
    # offset is < w (so two overlapping ticks can never be 5 m apart laterally).
    for i, (r_, m_) in enumerate(zip(rp, mp)):
        rel = np.linalg.inv(m_) @ r_
        if _footprint_overlap(rel, length, width) <= 0.0:
            failures.append(f"ego_pose_lines_up: tick {i} render footprint does not overlap the ego mesh (lateral {rel[1, 3]:.2f} m)")
            return False
    return True


def _item_vehicle(depth: np.ndarray, view: RenderedView, gt: GtFrame, values: dict[str, Any], failures: list[str]) -> bool:
    cov = _covered(depth, view.opacity)
    vm = np.asarray(gt.vehicle_mask, dtype=bool)
    n_veh = int(vm.sum())
    pts = _unproject_px(depth, gt, vm & cov)
    if n_veh == 0 or len(pts) == 0:
        failures.append("vehicle_placed_once: the known vehicle is not rendered")
        values["vehicle_center_err_m"] = None
        return False
    est = np.median(pts[:, :2], axis=0)
    err = float(np.linalg.norm(est - np.asarray(gt.vehicle_center_ego, dtype=np.float64)[:2]))
    values["vehicle_center_err_m"] = err
    if err > VEHICLE_CENTER_ERR_MAX_M:
        failures.append(f"vehicle_placed_once: centre error {err:.2f} m > {VEHICLE_CENTER_ERR_MAX_M} m")
        return False
    # drawn twice: elevated rendered surface that is off the GT surface, away from the vehicle
    gt_d = np.asarray(gt.depth, dtype=np.float64)
    tol = np.maximum(GHOST_DEPTH_TOL_M, 0.2 * np.where(np.isfinite(gt_d), gt_d, 0.0))
    with np.errstate(invalid="ignore"):
        off = ~np.isfinite(gt_d) | (np.abs(depth - gt_d) > tol)
    cand = cov & off & ~_dilate(vm, GHOST_EDGE_DILATE_PX)
    cand = _dilate(_erode(cand, GHOST_OPEN_PX), GHOST_OPEN_PX)  # opening: drop thin silhouette bleed
    ghost_pts = _unproject_px(depth, gt, cand)
    ground = _ground_height(gt)
    n_ghost = int((ghost_pts[:, 2] >= ground + GHOST_HEIGHT_MIN_M).sum()) if len(ghost_pts) else 0
    values["vehicle_ghost_px"] = n_ghost
    if n_ghost >= GHOST_MIN_FRAC_OF_VEHICLE * n_veh:
        failures.append(f"vehicle_placed_once: {n_ghost} off-surface elevated px vs {n_veh} vehicle px (drawn twice)")
        return False
    return True


def _item_unobserved(depth: np.ndarray, view: RenderedView, gt: GtFrame, values: dict[str, Any], failures: list[str]) -> bool:
    un = _erode(np.asarray(gt.unobserved_mask, dtype=bool), UNOBSERVED_EDGE_ERODE_PX)
    if not un.any():
        failures.append("unobserved_not_wall: clip has no unobserved pixels to check")
        return False
    frac = float((_covered(depth, view.opacity) & un).sum()) / float(un.sum())
    values["unobserved_opaque_frac"] = frac
    if frac > UNOBSERVED_OPAQUE_MAX_FRAC:
        failures.append(f"unobserved_not_wall: {frac:.3f} of unobserved px are opaque surface")
        return False
    return True


def find_non_gt_importers(repo_root: Path) -> list[str]:
    """Files under python/ and scripts/ that import this module without a GT-named path."""
    offenders: list[str] = []
    for sub in ("python", "scripts"):
        for p in sorted((Path(repo_root) / sub).rglob("*.py")):
            if "gt" in p.stem.split("_"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
                if "rv_rubric_gt" not in text:  # cheap pre-filter; the AST decides
                    continue
                tree = ast.parse(text)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                if any(n.split(".")[-1] == "rv_rubric_gt" for n in names):
                    offenders.append(str(p.relative_to(repo_root)).replace("\\", "/"))
                    break
    return offenders


def _live_input_rejects_gt_keys() -> bool:
    rgb = np.zeros((8, 8, 3), np.uint8)
    cam = _rvc.RvCameraInput("main", rgb, np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32))
    for key in ("depth_gt", "gt_depth", "sim_depth", "aux_depth"):
        spec = _rvc.RvInputSpec(
            _rvc.RV_SCHEMA_VERSION, 0, 0.0, np.eye(4, dtype=np.float32), {"main": cam}, {key: np.zeros((8, 8), np.float32)}
        )
        try:
            _rvc.validate_rv_input(spec)
        except _rvc.RvSchemaError:
            continue
        return False
    return True


def _item_no_gt_on_live_path(repo_root: Path, failures: list[str]) -> bool:
    ok = True
    if not _live_input_rejects_gt_keys():
        failures.append("no_sim_gt_on_live_path: the live input contract accepted a GT depth key")
        ok = False
    bad = find_non_gt_importers(repo_root)
    if bad:
        failures.append(f"no_sim_gt_on_live_path: non-GT-named files import the GT rubric: {bad}")
        ok = False
    return ok


def _item_stale(out: _rvc.RvOutputSpec, pack: Callable[..., bytes], failures: list[str]) -> bool:
    fresh = _rvc.RvOutputSpec(**{**out.__dict__, "stale": False})
    try:
        for age, want in ((0.0, False), (199.9, False), (STALE_AGE_MS, False), (STALE_AGE_MS + 0.1, True), (250.0, True), (5000.0, True)):
            hdr, spec = _rvc.unpack_output_slot(pack(fresh, age))
            got = bool(hdr.flags & _rvc.RV_FLAG_STALE) and spec.stale
            if got != want:
                failures.append(f"stale_indicator: age {age} ms -> stale={got}, expected {want}")
                return False
    except _rvc.RvSchemaError as exc:
        failures.append(f"stale_indicator: {exc}")
        return False
    return True


def _default_pack(out: _rvc.RvOutputSpec, age_ms: float) -> bytes:
    return _rvc.pack_output_slot(out, age_ms=age_ms)


def score_checklist(
    view: RenderedView,
    gt: GtFrame,
    harness: Harness,
    *,
    repo_root: Path,
    pack: Callable[[_rvc.RvOutputSpec, float], bytes] = _default_pack,
) -> RubricResult:
    """The six yes/no items of the spec's checklist. All must be yes."""
    failures: list[str] = []
    values: dict[str, Any] = {}
    depth, why = _usable_depth(view.out)
    items = {k: False for k in CHECKLIST_ITEMS}
    if depth is None:
        failures.append(f"road_surface/vehicle_placed_once/unobserved_not_wall: {why}")
    else:
        items["road_surface"] = _item_road_surface(depth, view, gt, failures)
        items["vehicle_placed_once"] = _item_vehicle(depth, view, gt, values, failures)
        items["unobserved_not_wall"] = _item_unobserved(depth, view, gt, values, failures)
    items["ego_pose_lines_up"] = _item_ego_pose(harness, failures)
    items["no_sim_gt_on_live_path"] = _item_no_gt_on_live_path(repo_root, failures)
    items["stale_indicator"] = _item_stale(view.out, pack, failures) if depth is not None else False
    return RubricResult(all(items.values()), items, values, failures)
