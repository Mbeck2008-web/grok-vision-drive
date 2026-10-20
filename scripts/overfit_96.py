"""Overfit acceptance: can the scene net memorise 96 real front frames?

CPU only. Fixed LR 1e-3, clip 1.0, AdamW with the trainer's weight decay,
the trainer's lane_gt loss with ``--conf-weight`` applied. Frames are main
camera canvases from the frame cache with a non-black main area and valid
5 m / 10 m centers (the same rows fix (a) keeps). Each step is a batch of
single-frame windows from a fresh GRU state, the training protocol.

Pass: in-sample 8 m r >= 0.9 AND pred_std >= 0.5 x label_std within
``--steps`` (default 1500). Logs r, pred_std/label_std and bias every 50.

``--mode nogru`` feeds the heads the per-frame residual (skip) features only,
``--mode noskip`` the GRU output only, to locate where information dies.
``--diag`` prints per-layer grad norms and activation stats.

    set CUDA_VISIBLE_DEVICES=-1
    set GVD_GT_FILE=gt_painted_v2.jsonl
    py -3 scripts/overfit_96.py --root E:\\GVD_data\\scene_lanes --cache C:\\GVD_cache_main
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def pick_rows(root: Path, cache, n: int, seed: int):
    from python.data.raw_frames import load_scene_rows
    from python.debug.lane_targets import lane_center_is_valid, lane_gt_from_record
    from python.train.train_scene import _frame_row

    rows = load_scene_rows(root)
    cand = []
    for row in rows:
        tech = row.get("tech") if isinstance(row.get("tech"), dict) else None
        if tech is None or not cache.has(int(row["i"])):
            continue
        if lane_center_is_valid(lane_gt_from_record(tech)[1]):
            cand.append(row)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(cand))
    chosen, frames = [], []
    for k in order:
        got = _frame_row(root, cand[k], cams=("main",), cache=cache, augment=False)
        if got is None:  # fix (a): a black main area is skipped
            continue
        chosen.append(cand[k])
        frames.append(got[0])
        if len(chosen) >= n:
            break
    return chosen, np.stack(frames)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\GVD_data\scene_lanes")
    ap.add_argument("--cache", default=r"C:\GVD_cache_main")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--conf-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("full", "nogru", "noskip"), default="full")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--backbone", choices=("scratch", "imagenet-resnet18", "mobilenetv3"), default="scratch",
                    help="Single-frame ImageNet baseline instead of the GRU scene net (--mode must stay full)")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--backbone-lr-mult", type=float, default=1.0, help="Backbone LR = lr x this (default 1.0)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    import torch

    if torch.cuda.is_available():
        print("refusing: CUDA is visible; set CUDA_VISIBLE_DEVICES=-1", flush=True)
        return 2
    torch.set_num_threads(int(args.threads))
    from python.debug.lane_targets import lane_x_from_vector, score_centers
    from python.train.continual import WEIGHT_DECAY
    from python.train.frame_cache import FrameCache
    from python.train.scene_net import SceneNet, _heads_batch, _prediction_at, labels_for_row, scene_loss
    from python.train.train_scene import _as_target

    cache = FrameCache(args.cache)
    rows, frames = pick_rows(Path(args.root), cache, int(args.n), int(args.seed))
    labels = [labels_for_row(r) for r in rows]
    x8 = np.array([lane_x_from_vector(y["lane_gt"]) for y in labels])
    print(
        f"[overfit] rows={len(rows)} frames={frames.shape} label x8 mean={x8.mean():+.3f} std={x8.std():.3f} "
        f"mode={args.mode} lr={args.lr} clip={args.clip} conf_weight={args.conf_weight} wd={WEIGHT_DECAY}",
        flush=True,
    )
    torch.manual_seed(int(args.seed))
    if args.backbone != "scratch":
        if args.mode != "full":
            raise SystemExit("--mode applies to the scratch net only")
        from python.train.pretrained_net import build_scene_net

        net = build_scene_net(("main",), 1.0, args.backbone, freeze=bool(args.freeze_backbone))
        print(f"[overfit] backbone={args.backbone} freeze={bool(args.freeze_backbone)} single-frame, no GRU", flush=True)
    else:
        net = SceneNet(cams=("main",))
    if args.mode != "full":
        import types

        def _fwd(self, image, dt, state):
            from python.train.scene_net import SceneState

            token = self._token(image)
            residual = token[:, 0]
            if self.skip is not None:
                residual = self.skip(residual)
            if args.mode == "nogru":
                feat = residual
                new_state = state
            else:
                from python.train.scene_net import _dt_column, _state_on

                state = _state_on(state, image)
                dt_col = _dt_column(dt, image)
                gru_out, h_new = self.gru(torch.cat([self.gru_in_drop(token), dt_col], dim=-1), state.h)
                feat = gru_out[:, 0, :]
                new_state = SceneState(h=h_new, tokens=state.tokens, mask=state.mask)
            return _heads_batch(self, feat), new_state

        net._forward_batched = types.MethodType(_fwd, net)
    net.train()
    if args.backbone != "scratch" and float(args.backbone_lr_mult) != 1.0:
        opt = torch.optim.AdamW(net.param_groups(float(args.lr), float(args.backbone_lr_mult)), lr=float(args.lr), weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=WEIGHT_DECAY)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_all = sum(p.numel() for p in net.parameters())
    n_grad = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(
        f"[overfit] optimizer groups={len(opt.param_groups)} lr={[g['lr'] for g in opt.param_groups]} "
        f"wd={[g['weight_decay'] for g in opt.param_groups]} params_in_opt={n_opt} params={n_all} requires_grad={n_grad}",
        flush=True,
    )
    device = torch.device("cpu")
    targets = [_as_target(y, device) for y in labels]
    rng = np.random.RandomState(int(args.seed) + 1)
    xt = torch.from_numpy(frames)

    def evaluate():
        net.eval()
        with torch.no_grad():
            preds = []
            for k in range(0, len(frames), 32):
                part = xt[k : k + 32]
                p, _ = net.forward_batch(part, torch.full((len(part),), 0.1), net.initial_state(batch=len(part)))
                preds.append(p.lane_gt.numpy())
        net.train()
        vec = np.concatenate(preds)
        px = np.array([lane_x_from_vector(v) for v in vec])
        s = score_centers(px, x8)
        return s, float(px.std())

    passed = False
    started = time.time()
    for step in range(int(args.steps) + 1):
        if step % 50 == 0:
            s, sd = evaluate()
            ratio = sd / max(1e-9, float(x8.std()))
            r = s["r"] if s["r"] is not None else float("nan")
            print(
                f"[overfit] step={step} r={r:+.3f} pred_std/label_std={ratio:.3f} bias={s['bias']:+.3f} "
                f"rms={s['rms']:.3f}/{s['rms_mean']:.3f} t={time.time() - started:.0f}s",
                flush=True,
            )
            if r >= 0.9 and ratio >= 0.5:
                passed = True
                print(f"[overfit] PASS at step {step}", flush=True)
                break
        if step == int(args.steps):
            break
        idx = rng.choice(len(frames), int(args.batch), replace=False)
        pred, _ = net.forward_batch(xt[idx], torch.full((len(idx),), 0.1), net.initial_state(batch=len(idx)))
        total = None
        for b, k in enumerate(idx):
            term = scene_loss(
                _prediction_at(pred, b), targets[k], ("lane_gt",), lane_loss="huber", conf_weight=float(args.conf_weight)
            )["total"]
            total = term if total is None else total + term
        loss = total / len(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.diag and step % 100 == 0:
            for name, p in net.named_parameters():
                if p.grad is not None and ("stem" in name or "gru" in name or "skip" in name or "lane_gt_head" in name):
                    print(f"   grad {name:40s} |g|={p.grad.norm():.2e} |w|={p.norm():.2e}", flush=True)
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float(args.clip)))
        opt.step()
        if args.diag and step % 100 == 0:
            print(f"   step {step} loss={loss.item():.4f} grad_norm_preclip={gn:.3f}", flush=True)
    if not passed:
        print(f"[overfit] FAIL after {args.steps} steps", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
