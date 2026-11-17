"""Train the sparse occupancy net from binary shards (docs/specs/occ_sparse.md).

Two steps, both offline:

    py -3 -m python.train.train_occ build --recording <rec> --out <nvme>\\occ_shards
    py -3 -m python.train.train_occ train --shards <nvme>\\occ_shards --epochs 20 --export <dir>\\occ.onnx

``build`` reads one recording (``state.jsonl``, ``raw/<cam>/``,
``raw/<cam>_depth``, ``raw/<cam>_sem``, ``sem_classes.json``) once and
writes tar shards: recorded image bytes plus sparse labels. ``train`` reads
only the shards; it never opens a recording folder or a loose PNG.

The sampler draws recoveries at ``--recovery-frac`` (0.20..0.30, default
0.25). ``--amp auto`` uses fp16 autocast + GradScaler on CUDA, fp32 on CPU.
Note: AGENTS.md says Pascal stays fp32 for the scene net; fp16 here is for
memory, and Pascal fp16 math is slow, so ``--amp off`` is the fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from python.data.occ_shards import (
    DEFAULT_SHARD_BYTES, ShardIndex, ShardWriter, is_recovery_row, pack_labels, recovery_mix_weights, unpack_labels,
)
from python.train import occ_sparse as occ
from python.train.occ_sparse import OccGrid

DEFAULT_LONG_SIDE = 640


def _boxes_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    from python.train.vehicle_labels import _boxes, _traffic_field

    traffic = _traffic_field(row)
    if not isinstance(traffic, list):
        return []
    try:
        return [b for b in _boxes(row, traffic) if isinstance(b, dict)]
    except Exception:  # noqa: BLE001 — a bad traffic row scores nothing
        return []


def _ego_xyd(row: dict[str, Any]) -> list[float] | None:
    """World ego pose (x, y, dir_x, dir_y) for temporal memory warps, or None."""
    from python.train.vehicle_labels import _ego_pose

    pos, d = _ego_pose(row)
    try:
        out = [float(pos[0]), float(pos[1]), float(d[0]), float(d[1])]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return out if all(np.isfinite(out)) and (out[2] or out[3]) else None


def build_shards(recording: Path | str, out_dir: Path | str, *, cams: Sequence[str] | None = None,
                 grid: OccGrid | None = None, stride: int = 4, max_bytes: int = DEFAULT_SHARD_BYTES,
                 poses: dict[str, Any] | None = None, prefix: str | None = None) -> dict[str, Any]:
    """One recording -> shards. Returns counts (ticks, written, recoveries, no-depth skips)."""
    from python.data.raw_frames import load_raw_camera, load_scene_rows, raw_frame_candidates
    from python.train.occ_labels import build_occ_labels
    from python.train.sensor_aux import read_sem_classes

    rec = Path(recording)
    grid = grid or OccGrid()
    if poses is None:
        from python.perception.lanes import camera_poses

        poses = dict(camera_poses())
    cam_ids = list(cams) if cams else list(poses)
    sem_names = (read_sem_classes(rec).get("classes") or {}) if (rec / "sem_classes.json").is_file() else {}
    writer = ShardWriter(out_dir, prefix or rec.name.replace(".", "_"), max_bytes=max_bytes)
    stats = {"ticks": 0, "written": 0, "recoveries": 0, "no_depth": 0, "no_image": 0}
    raw = rec / "raw"
    try:
        for row in load_scene_rows(rec):
            stats["ticks"] += 1
            try:
                i = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            members: dict[str, bytes] = {}
            for cam in cam_ids:
                for path in raw_frame_candidates(raw, cam, i, ".png"):
                    if path.is_file():
                        members[f"{cam}.png"] = path.read_bytes()
                        break
            if not members:
                stats["no_image"] += 1
                continue
            depth, sem = {}, {}
            for cam in poses:
                d = load_raw_camera(rec, f"{cam}_depth", i)
                if d is not None and d.ndim == 2:
                    depth[cam] = d.astype(np.uint16)
                    s = load_raw_camera(rec, f"{cam}_sem", i)
                    if s is not None and s.ndim == 2:
                        sem[cam] = s
            if not depth:
                stats["no_depth"] += 1
                continue
            lab = build_occ_labels(depth, poses, grid, stride=stride, sem_by_cam=sem, sem_names=sem_names,
                                   boxes=_boxes_for_row(row))
            recov = is_recovery_row(row)
            stats["recoveries"] += int(recov)
            members["occ.npz"] = pack_labels(lab)
            members["json"] = json.dumps({"rec": rec.name, "i": i, "recovery": recov, "ego": _ego_xyd(row),
                                          "cams": sorted(k[:-4] for k in members if k.endswith(".png"))}).encode()
            writer.write(f"{rec.name.replace('.', '_')}-{i:08d}", members)
            stats["written"] += 1
    finally:
        stats["shards"] = [str(p) for p in writer.close()]
    return stats


class OccShardDataset:
    """Samples from shard files only. Missing cameras are black and listed in ``cam_valid``."""

    def __init__(self, index: ShardIndex, cam_ids: Sequence[str], grid: OccGrid, *, long_side: int = DEFAULT_LONG_SIDE,
                 aspect: float = 4.0 / 3.0, n_occ: int = 4096, n_free: int = 4096, seed: int = 0,
                 mem_len: int = 2) -> None:
        self.index = index
        self.cam_ids = list(cam_ids)
        self.grid = grid
        self.w = int(long_side)
        self.h = int(round(long_side / aspect))
        self.n_occ, self.n_free = n_occ, n_free
        self.seed = seed
        metas = [index.meta(i) for i in range(len(index))]
        self.recovery = np.array([bool(m.get("recovery")) for m in metas])
        self.ego = [m.get("ego") for m in metas]
        # temporal memory: the mem_len previous ticks of the same recording (newest first)
        self.mem_len = int(mem_len)
        by_rec: dict[str, list[tuple[int, int]]] = {}
        for k, m in enumerate(metas):
            by_rec.setdefault(str(m.get("rec")), []).append((int(m.get("i", 0)), k))
        self.prev: list[list[int]] = [[] for _ in metas]
        for rows in by_rec.values():
            rows.sort()
            for j, (_, k) in enumerate(rows):
                self.prev[k] = [rows[j - d][1] for d in range(1, self.mem_len + 1) if j - d >= 0]

    def __len__(self) -> int:
        return len(self.index)

    def _images(self, blobs: dict[str, bytes]) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        imgs = np.zeros((len(self.cam_ids), 3, self.h, self.w), np.float32)
        valid = np.zeros(len(self.cam_ids), np.float32)
        for k, cam in enumerate(self.cam_ids):
            data = blobs.get(f"{cam}.png")
            if not data:
                continue
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)[..., ::-1]
            imgs[k] = rgb.transpose(2, 0, 1) / 255.0
            valid[k] = 1.0
        return imgs, valid

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        from python.train.occ_labels import sample_queries
        from python.train.occ_sparse import relative_xf

        blobs = self.index.read(i)
        imgs, valid = self._images(blobs)
        k_mem = max(1, self.mem_len)
        past = np.zeros((k_mem,) + imgs.shape, np.float32)
        xf = np.zeros((k_mem, 2, 3), np.float32)
        mvalid = np.zeros(k_mem, np.float32)
        for k, j in enumerate(self.prev[i][:self.mem_len]):
            if self.ego[i] is None or self.ego[j] is None:
                continue
            past[k] = self._images(self.index.read(j, exts=[f"{c}.png" for c in self.cam_ids]))[0]
            xf[k] = relative_xf(self.ego[i], self.ego[j])
            mvalid[k] = 1.0
        lab = unpack_labels(blobs["occ.npz"])
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        batch = sample_queries(lab, self.grid, n_occ=self.n_occ, n_free=self.n_free, rng=rng)
        batch["images"] = torch.from_numpy(imgs)
        batch["cam_valid"] = torch.from_numpy(valid)
        batch["past_images"] = torch.from_numpy(past)
        batch["mem_xf"] = torch.from_numpy(xf)
        batch["mem_valid"] = torch.from_numpy(mvalid)
        return batch


def forward_with_memory(net: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Training forward: past frames go through the SAME backbone under no_grad (cached at runtime)."""
    import torch

    past = batch["past_images"]
    b, k = past.shape[:2]
    with torch.no_grad():
        mf = net.encode(past.reshape((b * k,) + tuple(past.shape[2:])))
    mf = mf.reshape((b, k) + tuple(mf.shape[1:]))
    feat = net.encode(batch["images"])
    out = net.decode(feat, batch["queries"], mf, batch["mem_xf"], batch["mem_valid"])
    # camera-only VO head, supervised by the recorded pose delta (training label only, never a runtime input)
    out["vo"] = net.vo(feat, mf[:, 0])
    out["vo_target"] = occ.delta_from_xf(batch["mem_xf"][:, 0])
    out["vo_valid"] = batch["mem_valid"][:, 0]
    return out


def vo_loss(out: dict[str, Any]) -> Any:
    """Smooth-L1 on (dx, dy, dyaw*10) where the previous frame exists."""
    import torch

    if "vo" not in out:
        return 0.0
    scale = out["vo"].new_tensor([1.0, 1.0, 10.0])
    err = torch.nn.functional.smooth_l1_loss(out["vo"] * scale, out["vo_target"] * scale, reduction="none").sum(-1)
    v = out["vo_valid"]
    return (err * v).sum() / torch.clamp(v.sum(), min=1.0)


def make_sampler(dataset: OccShardDataset, target: float, num_samples: int | None = None, seed: int = 0):
    """Weighted sampler enforcing the recovery fraction. Prints the actual input fraction."""
    import torch

    w = recovery_mix_weights(dataset.recovery, target)
    frac_in = float(dataset.recovery.mean()) if len(dataset) else 0.0
    eff = float(w[dataset.recovery].sum() / w.sum()) if len(dataset) else 0.0
    print(f"occ mix: recoveries {frac_in:.3f} of shards, drawn at {eff:.3f} (target {target:.2f})", flush=True)
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(w.tolist(), num_samples or len(dataset), replacement=True,
                                                  generator=gen)


def train(args: argparse.Namespace) -> int:
    import torch

    from python.train.occ_labels import occ_loss
    from python.train.occ_sparse import SparseOccNet, export_onnx, rig_poses

    shards = sorted(Path(args.shards).glob("*.tar"))
    index = ShardIndex(shards)
    poses = rig_poses(args.cams.split(",") if args.cams else None)
    grid = OccGrid()
    ds = OccShardDataset(index, [p.cam_id for p in poses], grid, long_side=args.long_side,
                         n_occ=args.queries // 2, n_free=args.queries // 2)
    sampler = make_sampler(ds, args.recovery_frac, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                                         pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SparseOccNet(poses, grid).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    amp = dev.type == "cuda" and args.amp != "off"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for epoch in range(args.epochs):
        t0, tot, n = time.time(), 0.0, 0
        for batch in loader:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = forward_with_memory(net, batch)
            loss = occ_loss(out, batch) + vo_loss(out)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            n += 1
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
        print(f"occ epoch {epoch + 1}/{args.epochs} loss {tot / max(n, 1):.4f} "
              f"{time.time() - t0:.1f}s peak_vram_gb {peak:.2f}", flush=True)
    if args.export:
        export_onnx(net.cpu(), args.export, image_hw=(ds.h, ds.w))
        print(f"occ export {args.export}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="recording -> tar shards")
    b.add_argument("--recording", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--stride", type=int, default=4)
    b.add_argument("--max-shard-gb", type=float, default=1.0)
    t = sub.add_parser("train", help="train from shards")
    t.add_argument("--shards", required=True)
    t.add_argument("--cams", default="")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--queries", type=int, default=8192)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--long-side", type=int, default=DEFAULT_LONG_SIDE)
    t.add_argument("--recovery-frac", type=float, default=0.25)
    t.add_argument("--amp", choices=("auto", "off"), default="auto")
    t.add_argument("--workers", type=int, default=4 if sys.platform == "win32" else 0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--export", default="")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        stats = build_shards(args.recording, args.out, stride=args.stride,
                             max_bytes=int(args.max_shard_gb * 2**30))
        print(json.dumps(stats), flush=True)
        return 0 if stats["written"] else 1
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
