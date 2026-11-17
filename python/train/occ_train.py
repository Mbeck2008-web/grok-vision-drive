"""Train the sparse occupancy net from tar shards (training only).

Queries are sparse per sample: ``n_occ`` jittered occupied-cell points,
``n_free`` recorded free-space points, ``n_coarse`` coarse-cell centres. The
loss never touches a dense grid. Batches come from ``RecoveryMixSampler``
(20-30% recovery ticks). Every step logs data wait vs compute time.

``--amp auto`` follows the repo rule (``scene_net.amp_enabled``: fp16 autocast
at compute 7.0+, fp32 on Pascal 6.1). ``--amp fp16`` forces fp16 autocast +
GradScaler on CUDA (Pascal runs fp16 math at a low rate; it saves memory, not
time). CPU runs fp32.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from python.perception.occ_geometry import DEFAULT_GRID, OccGrid, PinholeCam
from python.perception.occ_sparse import LEVELS

N_OCC = 1024
N_FREE = 1024
N_COARSE = 512
JITTER = 0.45


def sample_queries(
    keys: np.ndarray,
    cls: np.ndarray,
    free: np.ndarray,
    rng: np.random.Generator,
    *,
    grid: OccGrid = DEFAULT_GRID,
    n_occ: int = N_OCC,
    n_free: int = N_FREE,
    n_coarse: int = N_COARSE,
    levels: Sequence[int] = LEVELS,
) -> dict[str, np.ndarray]:
    """Fixed-size sparse query set + targets for one labelled tick.

    ``n_coarse`` points are split over the coarse levels (cell centres of
    observed cells: holding an occupied fine cell or crossed by a free ray).
    ``any[:, l]`` is 1 when the level-``l`` cell holding the point holds an
    occupied fine cell.
    """
    keys = np.asarray(keys, dtype=np.int64)
    coarse_grids = [grid.coarse(f) for f in levels]
    pts: list[np.ndarray] = []
    occ: list[np.ndarray] = []
    occ_w: list[np.ndarray] = []
    tcls: list[np.ndarray] = []
    if keys.size and n_occ:
        pick = rng.integers(0, keys.size, size=int(n_occ))
        p = grid.centers_of_keys(keys[pick]) + rng.uniform(-JITTER, JITTER, size=(int(n_occ), 3)) * grid.voxel_m
        pts.append(p); occ.append(np.ones(n_occ)); occ_w.append(np.ones(n_occ)); tcls.append(cls[pick].astype(np.int64))
    if free.shape[0] and n_free:
        pick = rng.integers(0, free.shape[0], size=int(n_free))
        pts.append(free[pick].astype(np.float64)); occ.append(np.zeros(n_free)); occ_w.append(np.ones(n_free)); tcls.append(np.full(n_free, -1))
    centers = grid.centers_of_keys(keys) if keys.size else np.zeros((0, 3))
    occ_levels = [np.unique(cg.keys_of(centers)) for cg in coarse_grids]
    per = [int(n_coarse) // len(levels)] * len(levels)
    per[0] += int(n_coarse) - sum(per)
    for cg, occ_l, n_l in zip(coarse_grids, occ_levels, per):
        free_l = np.unique(cg.keys_of(free)) if free.shape[0] else np.zeros((0,), np.int64)
        seen = np.union1d(occ_l, free_l[free_l >= 0])
        if seen.size and n_l:
            pick = seen[rng.integers(0, seen.size, size=n_l)]
            pts.append(cg.centers_of_keys(pick)); occ.append(np.zeros(n_l)); occ_w.append(np.zeros(n_l)); tcls.append(np.full(n_l, -1))
    if not pts:
        raise ValueError("tick has neither occupied cells nor free samples")
    P = np.concatenate(pts)
    want = int(n_occ + n_free + n_coarse)
    O, W, C = np.concatenate(occ), np.concatenate(occ_w), np.concatenate(tcls)
    if P.shape[0] < want:  # pad by repeating (a tick with no hits or no free rays)
        rep = rng.integers(0, P.shape[0], size=want - P.shape[0])
        P, O, W, C = np.concatenate([P, P[rep]]), np.concatenate([O, O[rep]]), np.concatenate([W, W[rep]]), np.concatenate([C, C[rep]])
    anyt = np.stack([np.isin(cg.keys_of(P), occ_l) for cg, occ_l in zip(coarse_grids, occ_levels)], axis=-1)
    return {"pts": P.astype(np.float32), "occ": O.astype(np.float32), "occ_w": W.astype(np.float32),
            "any": anyt.astype(np.float32), "cls": C.astype(np.int64)}


def prep_image(img_bgr: np.ndarray, cam: PinholeCam) -> np.ndarray:
    """BGR uint8 -> RGB float32 CHW in [0,1] at the net's camera resolution."""
    import cv2

    if (img_bgr.shape[1], img_bgr.shape[0]) != (cam.width, cam.height):
        img_bgr = cv2.resize(img_bgr, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img_bgr[..., ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


class QueryDataset:
    """Shard sample -> (per-camera images, sparse queries)."""

    def __init__(self, shards: Any, cams: Sequence[PinholeCam], *, seed: int = 0, grid: OccGrid = DEFAULT_GRID, **q: int) -> None:
        self.shards = shards
        self.cams = list(cams)
        self.seed = int(seed)
        self.grid = grid
        self.q = q
        self.calls = 0

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.shards[i]
        self.calls += 1
        rng = np.random.default_rng((self.seed, int(i), self.calls))
        out = sample_queries(s["keys"], s["cls"], s["free"], rng, grid=self.grid, **self.q)
        out["images"] = [prep_image(s["images"][c.cam_id], c) for c in self.cams]
        out["recovery"] = bool(s["meta"].get("recovery"))
        return out


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    out = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in ("pts", "occ", "occ_w", "any", "cls")}
    out["images"] = [torch.from_numpy(np.stack([b["images"][k] for b in batch])) for k in range(len(batch[0]["images"]))]
    out["recovery"] = torch.tensor([b["recovery"] for b in batch])
    return out


def occ_loss(out: Mapping[str, Any], batch: Mapping[str, Any]) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    occ = out["occ"].float()
    w = batch["occ_w"]
    l_occ = (F.binary_cross_entropy_with_logits(occ, batch["occ"], reduction="none") * w).sum() / w.sum().clamp(min=1.0)
    l_any = F.binary_cross_entropy_with_logits(out["any"].float(), batch["any"])
    cls = batch["cls"]
    if bool((cls >= 0).any()):
        l_cls = F.cross_entropy(out["cls"].float().reshape(-1, out["cls"].shape[-1]), cls.reshape(-1), ignore_index=-1)
    else:
        l_cls = occ.sum() * 0.0
    total = l_occ + l_any + 0.5 * l_cls
    with torch.no_grad():
        m = w > 0
        pred = occ[m] > 0
        tgt = batch["occ"][m] > 0.5
        inter = float((pred & tgt).sum())
        union = float((pred | tgt).sum())
    return total, {"loss": float(total.detach()), "occ": float(l_occ.detach()), "any": float(l_any.detach()),
                   "cls": float(l_cls.detach()), "iou": inter / union if union else 0.0}


def resolve_amp(mode: str, device: Any) -> bool:
    import torch

    if mode == "off" or getattr(device, "type", str(device)) != "cuda":
        if mode == "fp16" and getattr(device, "type", str(device)) != "cuda":
            raise ValueError("--amp fp16 needs CUDA")
        return False
    if mode == "fp16":
        return True
    from python.train.scene_net import amp_enabled

    return bool(amp_enabled(torch.cuda.get_device_capability(device)))


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r) / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)


def train(
    shards: Any,
    cams: Sequence[PinholeCam],
    *,
    steps: int,
    batch_size: int = 4,
    lr: float = 1e-3,
    amp: str = "auto",
    device: str = "cpu",
    workers: int = 0,
    ratio: float = 0.25,
    seed: int = 0,
    log: Any = None,
    net: Any = None,
    **q: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from python.data.occ_shards import OccShardDataset
    from python.perception.occ_sparse import SparseOccNet
    from python.train.occ_sampler import RecoveryMixSampler

    torch.manual_seed(seed)
    ds = shards if isinstance(shards, OccShardDataset) else OccShardDataset(shards)
    dev = torch.device(device)
    net = (net if net is not None else SparseOccNet(cams)).to(dev)
    sampler = RecoveryMixSampler(ds.recovery_flags(), batch_size, ratio=ratio, seed=seed, batches=steps)
    dl = DataLoader(QueryDataset(ds, cams, seed=seed, **q), batch_sampler=sampler, collate_fn=collate,
                    num_workers=int(workers), persistent_workers=False)
    use_amp = resolve_amp(amp, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    hist: list[dict[str, float]] = []
    t_prev = time.perf_counter()
    for step, batch in enumerate(dl):
        t_data = time.perf_counter()
        batch = {k: ([t.to(dev, non_blocking=True) for t in v] if k == "images" else v.to(dev)) for k, v in batch.items()}
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            out = net(batch["images"], batch["pts"])
        loss, stats = occ_loss(out, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_done = time.perf_counter()
        stats.update({"step": step, "wait_ms": (t_data - t_prev) * 1e3, "compute_ms": (t_done - t_data) * 1e3,
                      "recovery_frac": float(batch["recovery"].float().mean())})
        hist.append(stats)
        if log is not None:
            log(json.dumps(stats))
        t_prev = time.perf_counter()
    res: dict[str, Any] = {"history": hist, "amp": use_amp, "peak_rss_mb": peak_rss_mb(), "net": net,
                           "params": int(sum(p.numel() for p in net.parameters()))}
    if dev.type == "cuda":
        res["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    return res


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    from python.perception.occ_sparse import live_cams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shards", nargs="+", help=".tar shards or a folder of them")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", choices=("auto", "fp16", "off"), default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--input-long", type=int, default=640)
    ap.add_argument("--cams", default="", help="comma list; default all 8 rig cams")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import torch

    cfg = yaml.safe_load(Path("config/cameras.yaml").read_text(encoding="utf-8"))
    cams = live_cams(cfg, a.input_long)
    if a.cams:
        keep = set(a.cams.split(","))
        cams = [c for c in cams if c.cam_id in keep]
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "occ_train.jsonl", "x", encoding="utf-8") as fh:
        res = train(a.shards, cams, steps=a.steps, batch_size=a.batch, lr=a.lr, amp=a.amp, device=a.device,
                    workers=a.workers, log=lambda s: (fh.write(s + "\n"), fh.flush()))
    torch.save({"state_dict": res["net"].state_dict(), "cams": [c.cam_id for c in cams], "input_long": a.input_long},
               a.out / "occ_sparse.pt")
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "net")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
