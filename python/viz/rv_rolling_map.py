"""Reality View step 6: bounded static rolling map (voxel-hash EMA), offline CPU.

Spec: docs/specs/reality_view_3dgs.md (Step 6, ``check_no_ghost_rolling_map``).
Pure CPU numpy: no GL, no accelerator code or config, no wall clock, no sampling,
no sim GT, no safety-path imports. The caller owns time (``t_s``) and ego pose.

Design: one record per occupied ``voxel_m`` cell, kept sorted by packed cell key
(deterministic). A frame's points are binned to cells; each hit cell moves its
centre and colour toward the frame mean by an EMA (``alpha``) and its opacity
``w`` toward 1 (``w += beta * o * (1 - w)``, so w is in (0, 1] and never falls
on re-observation). A new cell starts at ``W_NEW * o`` (< 0.1): a splat is only
"visible" (opacity >= 0.1) after a second hit, so a one-frame ghost is
invisible. Points inside a dynamic box (inflated by ``dyn_margin_m``) never
enter the map, and cells already inside such a box are purged. Cells are evicted
when older than ``max_age_s`` or outside the +/- ``radius_m`` window around the
ego; when over ``n_max`` the oldest go first, ties farthest-first, then by key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

N_MAX_DEFAULT = 300_000  # spec step 6 working cap (== rv_contract.RV_MAX_SPLATS)
RADIUS_M_DEFAULT = 64.0  # spec step 6 window (+/- m around ego)
VOXEL_M_DEFAULT = 0.25  # spec step 6 cell size
MAX_AGE_S_DEFAULT = 30.0
ALPHA_DEFAULT = 0.3
BETA_DEFAULT = 0.35
W_NEW = 0.05
VISIBLE_OPACITY = 0.1
DYN_MARGIN_M_DEFAULT = 0.5
BACKWARD_TOL_S_DEFAULT = 0.25
MAX_MERGE_POINTS = 4_000_000
MAX_DYNAMIC_BOXES = 1024
_KEY_BITS = 21
_KEY_OFF = 1 << (_KEY_BITS - 1)

STATUS_OK = "ok"
STATUS_CLOCK_BACKWARDS = "rejected_clock_backwards"


@dataclass(frozen=True)
class DynamicBox:
    """Oriented box of a dynamic object in world coordinates (yaw about +z)."""

    cx: float
    cy: float
    cz: float
    length: float  # along the box x axis (before yaw)
    width: float  # along the box y axis (before yaw)
    height: float  # along z
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class MergeStats:
    status: str
    n_in: int
    n_nonfinite: int
    n_out_of_window: int
    n_dynamic: int
    n_new: int
    n_updated: int
    n_purged_dynamic: int
    n_evicted_age: int
    n_evicted_window: int
    n_evicted_cap: int
    count: int


@dataclass(frozen=True)
class MapSnapshot:
    xyz: np.ndarray  # (n, 3) float64 world metres
    rgb: np.ndarray  # (n, 3) uint8
    opacity: np.ndarray  # (n,) float32 in (0, 1]
    last_t_s: np.ndarray  # (n,) float64
    hits: np.ndarray  # (n,) int64 >= 1


def _finite_scalar(name: str, v: object) -> float:
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number")
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite")
    return f


def _vec3(name: str, v: object) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (3,) or not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be 3 finite floats")
    return a


class RollingMap:
    """Bounded static voxel-hash EMA map. See module docstring."""

    def __init__(
        self,
        *,
        voxel_m: float = VOXEL_M_DEFAULT,
        radius_m: float = RADIUS_M_DEFAULT,
        n_max: int = N_MAX_DEFAULT,
        max_age_s: float = MAX_AGE_S_DEFAULT,
        alpha: float = ALPHA_DEFAULT,
        beta: float = BETA_DEFAULT,
        dyn_margin_m: float = DYN_MARGIN_M_DEFAULT,
        backward_tol_s: float = BACKWARD_TOL_S_DEFAULT,
    ) -> None:
        self.voxel_m = _finite_scalar("voxel_m", voxel_m)
        self.radius_m = _finite_scalar("radius_m", radius_m)
        self.max_age_s = _finite_scalar("max_age_s", max_age_s)
        self.alpha = _finite_scalar("alpha", alpha)
        self.beta = _finite_scalar("beta", beta)
        self.dyn_margin_m = _finite_scalar("dyn_margin_m", dyn_margin_m)
        self.backward_tol_s = _finite_scalar("backward_tol_s", backward_tol_s)
        if isinstance(n_max, (bool, np.bool_)) or not isinstance(n_max, (int, np.integer)):
            raise ValueError("n_max must be an int")
        self.n_max = int(n_max)
        if not 0.05 <= self.voxel_m <= 8.0:
            raise ValueError("voxel_m must be in [0.05, 8]")
        if not self.voxel_m <= self.radius_m <= RADIUS_M_DEFAULT:
            raise ValueError("radius_m must be in [voxel_m, 64]")
        if not 1 <= self.n_max <= N_MAX_DEFAULT:
            raise ValueError("n_max must be in [1, 300000]")
        if not self.max_age_s > 0.0:
            raise ValueError("max_age_s must be > 0")
        if not (0.0 < self.alpha <= 1.0 and 0.0 < self.beta <= 1.0):
            raise ValueError("alpha and beta must be in (0, 1]")
        if self.dyn_margin_m < 0.0 or self.backward_tol_s < 0.0:
            raise ValueError("dyn_margin_m and backward_tol_s must be >= 0")
        self._world_limit_m = (_KEY_OFF - 2) * self.voxel_m - self.radius_m
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self._keys = np.empty((0,), dtype=np.int64)
        self._pos = np.empty((0, 3), dtype=np.float64)
        self._col = np.empty((0, 3), dtype=np.float64)
        self._w = np.empty((0,), dtype=np.float64)
        self._last_t = np.empty((0,), dtype=np.float64)
        self._hits = np.empty((0,), dtype=np.int64)
        self._t = -math.inf

    def __len__(self) -> int:
        return int(self._keys.shape[0])

    @property
    def t_s(self) -> float:
        """Newest accepted timestamp (-inf before the first accepted merge)."""
        return self._t

    def snapshot(self) -> MapSnapshot:
        return MapSnapshot(
            xyz=self._pos.copy(),
            rgb=np.clip(np.rint(self._col), 0, 255).astype(np.uint8),
            opacity=self._w.astype(np.float32),
            last_t_s=self._last_t.copy(),
            hits=self._hits.copy(),
        )

    # -- helpers -----------------------------------------------------------
    def _cell_keys(self, p: np.ndarray) -> np.ndarray:
        ijk = np.floor(p / self.voxel_m).astype(np.int64) + _KEY_OFF
        return (ijk[:, 0] << (2 * _KEY_BITS)) | (ijk[:, 1] << _KEY_BITS) | ijk[:, 2]

    def _check_boxes(self, boxes: Iterable[DynamicBox]) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        out: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for b in boxes:
            if not isinstance(b, DynamicBox):
                raise ValueError("dynamic_boxes must be DynamicBox instances")
            vals = [_finite_scalar(n, getattr(b, n)) for n in ("cx", "cy", "cz", "length", "width", "height", "yaw_rad")]
            if not (vals[3] > 0.0 and vals[4] > 0.0 and vals[5] > 0.0):
                raise ValueError("dynamic box dimensions must be > 0")
            if len(out) >= MAX_DYNAMIC_BOXES:
                raise ValueError("too many dynamic boxes")
            half = np.array([vals[3], vals[4], vals[5]], dtype=np.float64) * 0.5 + self.dyn_margin_m
            out.append((np.array(vals[:3], dtype=np.float64), half, math.cos(vals[6]), math.sin(vals[6])))
        return out

    @staticmethod
    def _in_boxes(p: np.ndarray, boxes: Sequence[tuple[np.ndarray, np.ndarray, float, float]]) -> np.ndarray:
        inside = np.zeros((p.shape[0],), dtype=bool)
        for centre, half, c, s in boxes:
            d = p - centre
            lx = c * d[:, 0] + s * d[:, 1]
            ly = -s * d[:, 0] + c * d[:, 1]
            inside |= (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1]) & (np.abs(d[:, 2]) <= half[2])
        return inside

    # -- merge -------------------------------------------------------------
    def merge(
        self,
        points_xyz: object,
        rgb: object,
        *,
        t_s: float,
        ego_xyz: object,
        dynamic_boxes: Iterable[DynamicBox] = (),
        opacity: object = None,
    ) -> MergeStats:
        """Merge one frame (or one camera of a frame) of world-frame points.

        Contract errors (shape, dtype, non-finite scalars, bad boxes) raise
        ``ValueError`` before any state change. Per-point garbage (non-finite
        xyz, opacity not in (0, 1]) is dropped and counted. A timestamp more than
        ``backward_tol_s`` behind the newest accepted one is refused with
        ``STATUS_CLOCK_BACKWARDS`` and no state change; a smaller step back is
        clamped to the newest timestamp.
        """
        t = _finite_scalar("t_s", t_s)
        ego = _vec3("ego_xyz", ego_xyz)
        if float(np.max(np.abs(ego))) > self._world_limit_m:
            raise ValueError("ego_xyz outside the supported world range")
        boxes = self._check_boxes(dynamic_boxes)
        pts = np.asarray(points_xyz)
        col = np.asarray(rgb)
        if pts.dtype.kind not in "fiu" or pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_xyz must be an (N, 3) real array")
        if col.dtype != np.uint8 or col.shape != pts.shape:
            raise ValueError("rgb must be a uint8 array shaped like points_xyz")
        n_in = int(pts.shape[0])
        if n_in > MAX_MERGE_POINTS:
            raise ValueError("too many points in one merge")
        if opacity is None:
            op = np.ones((n_in,), dtype=np.float64)
        else:
            op = np.asarray(opacity)
            if op.dtype.kind not in "fiu" or op.shape != (n_in,):
                raise ValueError("opacity must be an (N,) real array")
            op = op.astype(np.float64)
            if bool(np.any(op[np.isfinite(op)] > 1.0)):
                raise ValueError("opacity must not exceed 1")
        pts = pts.astype(np.float64)

        if t < self._t - self.backward_tol_s:
            return MergeStats(STATUS_CLOCK_BACKWARDS, n_in, 0, 0, 0, 0, 0, 0, 0, 0, 0, len(self))
        t_eff = max(t, self._t)

        # Existing cells: age, window, dynamic purge (new ego / boxes / time).
        age = t_eff - self._last_t
        old_age = age > self.max_age_s
        old_win = np.max(np.abs(self._pos - ego), axis=1) > self.radius_m
        old_dyn = self._in_boxes(self._pos, boxes)
        drop_age = old_age
        drop_win = old_win & ~old_age
        drop_dyn = old_dyn & ~old_age & ~old_win
        keep = ~(old_age | old_win | old_dyn)
        keys, pos, colr = self._keys[keep], self._pos[keep], self._col[keep]
        w, last_t, hits = self._w[keep], self._last_t[keep], self._hits[keep]

        # New points: drop garbage, out-of-window, dynamic.
        finite = np.all(np.isfinite(pts), axis=1) & np.isfinite(op) & (op > 0.0)
        n_nonfinite = int(n_in - np.count_nonzero(finite))
        in_win = np.zeros((n_in,), dtype=bool)
        in_win[finite] = np.max(np.abs(pts[finite] - ego), axis=1) <= self.radius_m
        n_out = int(np.count_nonzero(finite & ~in_win))
        dyn = np.zeros((n_in,), dtype=bool)
        dyn[in_win] = self._in_boxes(pts[in_win], boxes)
        n_dyn = int(np.count_nonzero(dyn))
        use = in_win & ~dyn
        p_new, c_new, o_new = pts[use], col[use].astype(np.float64), op[use]

        n_new = n_upd = 0
        if p_new.shape[0] > 0:
            k_new = self._cell_keys(p_new)
            ukeys, inv, cnt = np.unique(k_new, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            cntf = cnt.astype(np.float64)
            mp = np.stack([np.bincount(inv, weights=p_new[:, i], minlength=ukeys.shape[0]) for i in range(3)], axis=1) / cntf[:, None]
            mc = np.stack([np.bincount(inv, weights=c_new[:, i], minlength=ukeys.shape[0]) for i in range(3)], axis=1) / cntf[:, None]
            mo = np.bincount(inv, weights=o_new, minlength=ukeys.shape[0]) / cntf
            idx = np.searchsorted(keys, ukeys)
            idx_c = np.minimum(idx, max(keys.shape[0] - 1, 0))
            exists = (idx < keys.shape[0]) & (keys[idx_c] == ukeys) if keys.shape[0] else np.zeros(ukeys.shape, dtype=bool)
            ei = idx[exists]
            pos[ei] = pos[ei] + self.alpha * (mp[exists] - pos[ei])
            colr[ei] = colr[ei] + self.alpha * (mc[exists] - colr[ei])
            w[ei] = np.minimum(1.0, w[ei] + self.beta * mo[exists] * (1.0 - w[ei]))
            last_t[ei] = t_eff
            hits[ei] += 1
            n_upd = int(ei.shape[0])
            fresh = ~exists
            n_new = int(np.count_nonzero(fresh))
            keys = np.concatenate([keys, ukeys[fresh]])
            pos = np.concatenate([pos, mp[fresh]])
            colr = np.concatenate([colr, mc[fresh]])
            w = np.concatenate([w, W_NEW * mo[fresh]])
            last_t = np.concatenate([last_t, np.full((n_new,), t_eff)])
            hits = np.concatenate([hits, np.ones((n_new,), dtype=np.int64)])

        # An EMA step can in principle land a centre inside a box (voxel larger than the box): purge again.
        late_dyn = self._in_boxes(pos, boxes)
        n_late_dyn = int(np.count_nonzero(late_dyn))
        if n_late_dyn:
            keep2 = ~late_dyn
            keys, pos, colr = keys[keep2], pos[keep2], colr[keep2]
            w, last_t, hits = w[keep2], last_t[keep2], hits[keep2]

        # Cap: oldest first, ties farthest first, ties by key.
        n_cap = 0
        if keys.shape[0] > self.n_max:
            excess = keys.shape[0] - self.n_max
            d = pos - ego
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            order = np.lexsort((keys, -dist, -(t_eff - last_t)))
            sel = np.ones((keys.shape[0],), dtype=bool)
            sel[order[:excess]] = False
            keys, pos, colr, w, last_t, hits = keys[sel], pos[sel], colr[sel], w[sel], last_t[sel], hits[sel]
            n_cap = int(excess)

        srt = np.argsort(keys, kind="stable")
        self._keys, self._pos, self._col = keys[srt], pos[srt], colr[srt]
        self._w, self._last_t, self._hits = w[srt], last_t[srt], hits[srt]
        self._t = t_eff
        return MergeStats(
            STATUS_OK, n_in, n_nonfinite, n_out, n_dyn, n_new, n_upd,
            int(np.count_nonzero(drop_dyn)) + n_late_dyn, int(np.count_nonzero(drop_age)),
            int(np.count_nonzero(drop_win)), n_cap, len(self),
        )
