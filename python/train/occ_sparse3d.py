"""Fully sparse multi-camera occupancy core (AI-Day layout, no dense volume anywhere).

Path per tick, all on a sparse voxel set (0.5 m ids of ``occ_sparse.GRID_SHAPE``):

1. Lift: every camera's shared-stem map ``(C, H', W')`` -> per-pixel feature
   and depth softmax (``DEPTH_M``). Each (cam, row, col, depth) ray sample has
   a fixed voxel id from ``config/cameras.yaml`` (position, yaw, pitch, fov).
2. Fuse: samples of ALL cameras are index_added into the ACTIVE voxel set
   only (the unique ids rays reach, precomputed). The full grid is never allocated.
   A camera older than ``stale_s`` writes nothing this tick.
3. Prune: a score head keeps the top ``keep`` voxels -> ``SparseSet`` (ids, feat, score).
4. Temporal memory: last tick's set is moved into this ego frame by an exact
   coordinate transform (dy, dyaw) on voxel centres, re-quantised, gated and
   merged with the current set by voxel id, then pruned again. Ego motion
   defaults to a camera-only VO head on the two sets. Vehicle speed/yaw rate
   are used only with ``ego_from_vehicle`` (default off). Sim GT pose is never an input.
5. Read: ``gather`` returns a feature at any query voxel by looking up its
   3x3x3 neighbours in the set (sorted ids + searchsorted).

Training-only for now (not in ONNX). The torch-free stub keeps imports cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from python.train.occ_sparse import GRID_MIN, GRID_SHAPE, VOXEL_M

FEAT_C = 32
TOPK_D = 2  # depth bins kept per ray
KEEP = 4096
STALE_S = 0.5
DEPTH_M = tuple(float(d) for d in np.linspace(2.0, 46.0, 12))
FULL_GRID = int(np.prod(GRID_SHAPE))


def voxel_id(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Flat id (ix, iy, iz row-major like ``occ_sparse``), -1 outside the grid."""
    ix = np.floor((x - GRID_MIN[0]) / VOXEL_M).astype(np.int64)
    iy = np.floor((y - GRID_MIN[1]) / VOXEL_M).astype(np.int64)
    iz = np.floor((z - GRID_MIN[2]) / VOXEL_M).astype(np.int64)
    ok = (ix >= 0) & (ix < GRID_SHAPE[0]) & (iy >= 0) & (iy < GRID_SHAPE[1]) & (iz >= 0) & (iz < GRID_SHAPE[2])
    return np.where(ok, (ix * GRID_SHAPE[1] + iy) * GRID_SHAPE[2] + iz, -1)


def lift_voxels(cams: Sequence[str], rows: int, cols: int, poses: Mapping[str, Any] | None = None) -> np.ndarray:
    """``(n_cams, rows, cols, len(DEPTH_M))`` voxel id of each ray sample, -1 outside the grid or no pose."""
    if poses is None:
        from python.perception.lanes import camera_poses

        poses = camera_poses()
    out = np.full((len(cams), int(rows), int(cols), len(DEPTH_M)), -1, np.int64)
    u = (np.arange(int(cols)) + 0.5) / float(cols) * 2.0 - 1.0  # left -1 .. right +1
    v = (np.arange(int(rows)) + 0.5) / float(rows) * 2.0 - 1.0  # top -1 .. bottom +1
    d = np.asarray(DEPTH_M)
    for i, cid in enumerate(cams):
        p = poses.get(str(cid))
        if p is None:
            continue
        tv = math.tan(math.radians(float(p.fov_v_deg)) / 2.0)
        th = tv * float(p.width) / float(p.height)
        az = math.radians(float(p.yaw_deg)) + np.arctan(u * th)  # (cols,)
        el = math.radians(float(p.pitch_deg)) - np.arctan(v * tv)  # (rows,)
        ce, se = np.cos(el)[:, None, None], np.sin(el)[:, None, None]
        sa, ca = np.sin(az)[None, :, None], np.cos(az)[None, :, None]
        x = float(p.x) + d * ce * sa
        y = float(p.y) + d * ce * ca
        z = float(p.z) + d * se * np.ones_like(sa)
        out[i] = voxel_id(x, y, z)
    return out


@dataclass
class SparseSet:
    """Top-``keep`` voxels of one tick, padded: ``ids`` ``(B, N)`` (-1 = pad), ``feat`` ``(B, N, C)``, ``score`` ``(B, N)``."""

    ids: Any
    feat: Any
    score: Any


@dataclass
class SparseState:
    """Last tick's set (its ego frame) and the camera-only ego estimate ``vo`` ``(B, 3)`` =
    (dx m right, dy m FORWARD, dyaw rad left +) since the previous tick."""

    last: SparseSet | None = None
    vo: Any = None
    carried: Any = None  # (B, keep) last tick's ids warped into this frame (-1 = dropped); viz/debug only


try:
    import torch
    from torch import nn

    def _centers(ids: Any) -> Any:
        ix = torch.div(ids, (GRID_SHAPE[1] * GRID_SHAPE[2]), rounding_mode="floor")
        iy = torch.div(ids % (GRID_SHAPE[1] * GRID_SHAPE[2]), GRID_SHAPE[2], rounding_mode="floor")
        iz = ids % GRID_SHAPE[2]
        lo = torch.tensor(GRID_MIN, dtype=torch.float32, device=ids.device)
        return lo + (torch.stack([ix, iy, iz], -1).float() + 0.5) * VOXEL_M

    def _ids_of(xyz: Any) -> Any:
        lo = torch.tensor(GRID_MIN, dtype=xyz.dtype, device=xyz.device)
        ijk = torch.floor((xyz - lo) / VOXEL_M).long()
        shape = torch.tensor(GRID_SHAPE, device=xyz.device)
        ok = ((ijk >= 0) & (ijk < shape)).all(-1)
        flat = (ijk[..., 0] * GRID_SHAPE[1] + ijk[..., 1]) * GRID_SHAPE[2] + ijk[..., 2]
        return torch.where(ok, flat, torch.full_like(flat, -1))

    def warp_ids(ids: Any, dy_m: Any, dyaw_rad: Any, dx_m: Any = None) -> Any:
        """Last tick's voxel ids in this tick's ego frame. Ego frame: x right, y FORWARD, z up; yaw + = left.

        The ego moved ``dx_m`` right and ``dy_m`` forward, then turned ``dyaw_rad``.
        Exact on coordinates: p_new = R(-dyaw) @ (x - dx, y - dy). Pads (-1) and cells leaving the grid -> -1.
        """
        xyz = _centers(ids.clamp(min=0))
        c = torch.cos(dyaw_rad).reshape(-1, 1)
        s = torch.sin(dyaw_rad).reshape(-1, 1)
        dx = torch.zeros_like(dy_m) if dx_m is None else dx_m
        x, y = xyz[..., 0] - dx.reshape(-1, 1), xyz[..., 1] - dy_m.reshape(-1, 1)
        moved = torch.stack([c * x + s * y, -s * x + c * y, xyz[..., 2]], -1)
        out = _ids_of(moved)
        return torch.where(ids >= 0, out, torch.full_like(out, -1))

    _NB = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])

    def gather(sset: SparseSet, q: Any) -> Any:
        """Feature at query voxels ``q`` ``(B, K)``: mean of the set's features over the 3x3x3 neighbourhood.

        Returns ``(B, K, C + 1)``; the last channel is the hit fraction (0 = nothing stored nearby).
        """
        b, k = q.shape
        ids, feat = sset.ids, sset.feat
        order = torch.argsort(ids, dim=1)
        sids = torch.gather(ids, 1, order)
        sfeat = torch.gather(feat, 1, order.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))
        ix = torch.div(q, (GRID_SHAPE[1] * GRID_SHAPE[2]), rounding_mode="floor")
        iy = torch.div(q % (GRID_SHAPE[1] * GRID_SHAPE[2]), GRID_SHAPE[2], rounding_mode="floor")
        iz = q % GRID_SHAPE[2]
        acc = feat.new_zeros(b, k, feat.shape[-1])
        hits = feat.new_zeros(b, k)
        n = sids.shape[1]
        for off in _NB.tolist():
            jx, jy, jz = ix + off[0], iy + off[1], iz + off[2]
            ok = (jx >= 0) & (jx < GRID_SHAPE[0]) & (jy >= 0) & (jy < GRID_SHAPE[1]) & (jz >= 0) & (jz < GRID_SHAPE[2])
            nid = (jx * GRID_SHAPE[1] + jy) * GRID_SHAPE[2] + jz
            pos = torch.searchsorted(sids, nid.contiguous()).clamp(max=n - 1)
            found = ok & (torch.gather(sids, 1, pos) == nid)
            f = torch.gather(sfeat, 1, pos.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))
            acc = acc + f * found.unsqueeze(-1).to(f.dtype)
            hits = hits + found.to(f.dtype)
        out = acc / hits.clamp(min=1.0).unsqueeze(-1)
        return torch.cat([out, (hits / float(len(_NB))).unsqueeze(-1)], dim=-1)

    def _merge(ids: Any, feat: Any, keep: int, score_fn: Any) -> SparseSet:
        """Per sample: mean features of duplicate ids, score, keep top ``keep`` (pad -1)."""
        b = ids.shape[0]
        out_ids, out_f, out_s = [], [], []
        for i in range(b):
            m = ids[i] >= 0
            uid, inv = torch.unique(ids[i][m], return_inverse=True)
            f = feat[i][m]
            s = f.new_zeros(len(uid), f.shape[-1]).index_add(0, inv, f)
            cnt = f.new_zeros(len(uid)).index_add(0, inv, torch.ones_like(inv, dtype=f.dtype))
            f = s / cnt.clamp(min=1.0).unsqueeze(-1)
            sc = score_fn(f).squeeze(-1)
            if len(uid) > keep:
                top = torch.topk(sc, keep).indices
                uid, f, sc = uid[top], f[top], sc[top]
            pad = keep - len(uid)
            out_ids.append(torch.cat([uid, uid.new_full((pad,), -1)]))
            out_f.append(torch.cat([f, f.new_zeros(pad, f.shape[-1])]))
            out_s.append(torch.cat([sc, sc.new_full((pad,), -1e4)]))
        return SparseSet(torch.stack(out_ids), torch.stack(out_f), torch.stack(out_s))

    class SparseOcc3D(nn.Module):
        """All cameras' stem maps -> one fused sparse voxel set with ego-aligned temporal memory."""

        def __init__(self, in_ch: int, cams: Sequence[str], rows: int, cols: int,
                     poses: Mapping[str, Any] | None = None, keep: int = KEEP) -> None:
            super().__init__()
            self.cams = tuple(str(c) for c in cams)
            self.keep = int(keep)
            self.stale_s = STALE_S
            self.ego_from_vehicle = False
            self.lift = nn.Conv2d(int(in_ch), FEAT_C + len(DEPTH_M), kernel_size=1)
            vox = torch.as_tensor(lift_voxels(self.cams, int(rows), int(cols), poses))  # (n, R, W, D)
            self.register_buffer("vox", vox, persistent=False)
            # Every voxel some ray bin could reach. Reported only; never allocated per tick.
            self.register_buffer("frustum", torch.unique(vox[vox >= 0]), persistent=False)
            self.samples = len(self.cams) * int(rows) * int(cols) * TOPK_D  # hard cap on per-tick candidates
            self.post = nn.Sequential(nn.Linear(FEAT_C + 3, FEAT_C), nn.ReLU(), nn.Linear(FEAT_C, FEAT_C))
            self.score = nn.Linear(FEAT_C, 1)
            self.mem_gate = nn.Linear(2 * FEAT_C, 1)
            self.mem_keep = nn.Linear(FEAT_C, FEAT_C)
            self.vo = nn.Sequential(nn.Linear(4 * FEAT_C, 64), nn.ReLU(), nn.Linear(64, 3))

        def ages_for(self, ages8: Any) -> Any:
            from python.data.cam_hold import CAM_AGE_ORDER

            return torch.as_tensor(ages8)[..., [list(CAM_AGE_ORDER).index(c) for c in self.cams]]

        def splat(self, maps: Any, ages: Any = None) -> tuple[Any, Any]:
            """``(B, n, C, H', W')`` -> ``(B, S)`` voxel ids and ``(B, S, FEAT_C)`` features, ``S = n*H'*W'*TOPK_D``.

            Each ray keeps only its TOPK_D most likely depth bins (feature x bin prob). Out-of-grid bins and
            stale cams give id -1. Nothing frustum- or grid-sized is ever built.
            """
            b, n, ch, h, w = maps.shape
            if n != len(self.cams) or (h, w) != tuple(self.vox.shape[1:3]):
                raise ValueError(f"sparse occ wants {len(self.cams)} cams x {tuple(self.vox.shape[1:3])}, got {n} x {(h, w)}")
            out = self.lift(maps.reshape(b * n, ch, h, w))
            feat = out[:, :FEAT_C].permute(0, 2, 3, 1)  # (b*n, H, W, C)
            p, di = out[:, FEAT_C:].softmax(dim=1).topk(TOPK_D, dim=1)  # (b*n, K, H, W)
            vox = self.vox.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(b * n, h, w, -1)
            ids = torch.gather(vox, -1, di.permute(0, 2, 3, 1))  # (b*n, H, W, K)
            f = feat.unsqueeze(3) * p.permute(0, 2, 3, 1).unsqueeze(-1)  # (b*n, H, W, K, C)
            ids = ids.reshape(b, n, -1)
            if ages is not None:
                stale = torch.as_tensor(ages, device=ids.device).reshape(b, n, 1) > float(self.stale_s)
                ids = torch.where(stale, torch.full_like(ids, -1), ids)
            return ids.reshape(b, -1), f.reshape(b, -1, FEAT_C)

        def cam_samples(self, maps: Any, ages: Any = None) -> dict[str, Any]:
            """Per-ray top-k depth samples for camera overlays: ``(B, n, H', W', K)`` depth m, prob, voxel id."""
            b, n, ch, h, w = maps.shape
            with torch.no_grad():
                out = self.lift(maps.reshape(b * n, ch, h, w))
                p, di = out[:, FEAT_C:].softmax(dim=1).topk(TOPK_D, dim=1)
                ids, _ = self.splat(maps, ages)
            depth = torch.as_tensor(DEPTH_M, dtype=p.dtype, device=p.device)[di]
            to = (b, n, h, w, TOPK_D)
            return {"depth_m": depth.permute(0, 2, 3, 1).reshape(to), "prob": p.permute(0, 2, 3, 1).reshape(to),
                    "ids": ids.reshape(to)}

        def fuse(self, maps: Any, ages: Any = None) -> SparseSet:
            ids, f = self.splat(maps, ages)
            xyz = _centers(ids.clamp(min=0)).to(f.dtype)
            f = self.post(torch.cat([f, xyz / 20.0], -1))
            return _merge(ids, f, self.keep, self.score)

        def _pool(self, s: SparseSet) -> Any:
            m = (s.ids >= 0).to(s.feat.dtype).unsqueeze(-1)
            mean = (s.feat * m).sum(1) / m.sum(1).clamp(min=1.0)
            xyz = _centers(s.ids.clamp(min=0)).to(s.feat.dtype)
            wy = ((s.feat * m) * xyz[..., 1:2]).sum(1) / m.sum(1).clamp(min=1.0) / 20.0
            return torch.cat([mean, wy], -1)

        def step(self, maps: Any, dt: Any, state: SparseState | None = None, ego: Any = None,
                 ages: Any = None, cur: SparseSet | None = None) -> tuple[SparseSet, SparseState]:
            cur = self.fuse(maps, ages) if cur is None else cur
            if state is None or state.last is None:
                return cur, SparseState(cur, None)
            b = cur.ids.shape[0]
            last = state.last
            vo = self.vo(torch.cat([self._pool(cur), self._pool(last)], -1))
            if self.ego_from_vehicle:
                if ego is None:
                    raise ValueError("ego_from_vehicle is on but no (speed, yaw_rate) was given")
                dt = torch.as_tensor(dt, dtype=cur.feat.dtype, device=cur.feat.device).reshape(-1).expand(b)
                ego = torch.as_tensor(ego, dtype=cur.feat.dtype, device=cur.feat.device).reshape(b, 2)
                # wheel speed is along the body's forward axis (y); no lateral slip term from vehicle signals
                move = torch.stack([torch.zeros_like(dt), ego[:, 0] * dt, ego[:, 1] * dt], 1)
            elif ego is not None:
                raise ValueError("vehicle ego given with ego_from_vehicle off: runtime is camera-only (use vo)")
            else:
                move = vo
            prev_ids = warp_ids(last.ids, move[:, 1], move[:, 2], move[:, 0])
            # Memory gate per carried voxel: what is stored there now vs what was carried.
            here = gather(SparseSet(cur.ids, cur.feat, cur.score), prev_ids.clamp(min=0))[..., :FEAT_C]
            g = torch.sigmoid(self.mem_gate(torch.cat([last.feat, here], -1)))
            carried = g * self.mem_keep(last.feat)
            merged = _merge(torch.cat([cur.ids, prev_ids], 1), torch.cat([cur.feat, carried], 1), self.keep, self.score)
            return merged, SparseState(merged, vo, prev_ids)

        def window(self, maps: Any, dts: Any, ego: Any = None, state: SparseState | None = None,
                   ages: Any = None) -> tuple[list[SparseSet], SparseState]:
            outs = []
            for k in range(int(maps.shape[1])):
                s, state = self.step(maps[:, k], dts[:, k] if torch.is_tensor(dts) else dts, state,
                                     None if ego is None else ego[:, k], None if ages is None else ages[:, k])
                outs.append(s)
            return outs, state  # type: ignore[return-value]

except ImportError:  # pragma: no cover - torch-free hosts
    class SparseOcc3D:  # type: ignore[no-redef]
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("PyTorch is required for the sparse occupancy core")
