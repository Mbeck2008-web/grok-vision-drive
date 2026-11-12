#!/usr/bin/env python3
"""Score a scene-net .npz checkpoint on the trainer's own holdout. No training.

Loads the checkpoint with ``epoch_ckpt.build_net`` (preset and ``route_cond``
come from the file; an older file with neither is small and has no route
conditioning, same as #206). Rebuilds the holdout the way ``train_directory``
does after #187 (``--holdout-blocks``, train-only extra ``--strips`` roots,
``--mask-events``, and the #211 default wrong-way / road-clip filters) and
scores it with ``train_scene._eval_sequence`` + ``val_summary``.

Prints the usual holdout log lines, then one JSON object with ``n``, ``r``,
``rms``, and the constant-predictor ``rms_mean``. The checkpoint is never
written. Inference is fp32 (including Pascal).

An ep11 re-score on this path is the *current* holdout (#187 time-block
split plus #211 default wrong-way / road-clip filters), not comparable
like-for-like to the original 0.6338 m number: that checkpoint predates
#187, so today's holdout rows may have been training frames then, and
#211 drops frames the old val set may have included.

    py -3 scripts/eval_scene_ckpt.py --ckpt <file.npz> --strips <prior> --strips <r2> ^
        --holdout-source <prior> --holdout-blocks 5 --mask-events --seed 73
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval-only scene checkpoint score on the current trainer holdout (#187 + #211; not like-for-like to pre-#187 numbers such as 0.6338)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Scene-net epoch checkpoint (.npz). Read only.")
    parser.add_argument("--strips", action="append", required=True, help="Recording folder. Repeat; extras are train-only.")
    parser.add_argument("--holdout-source", default="", help="Which --strips root is the holdout. Default: the first.")
    parser.add_argument("--holdout-blocks", type=int, default=None, help="Same as train_scene (default 1 or GVD_HOLDOUT_BLOCKS).")
    parser.add_argument("--mask-events", action="store_true", help="Same as train_scene --mask-events.")
    parser.add_argument("--mask-files", default="", help="Same as train_scene --mask-files.")
    parser.add_argument("--cams", default="", help="Comma list. Default: the checkpoint's cams.")
    parser.add_argument("--cache", default="", help="Frame cache folder, as train_scene --cache.")
    parser.add_argument("--eval-chunk", type=int, default=None, help="GRU reset period. Default: the checkpoint's eval_chunk, else 16.")
    parser.add_argument("--seed", type=int, default=0, help="Logged only (eval does not train or augment).")
    ww = parser.add_mutually_exclusive_group()
    ww.add_argument(
        "--drop-wrong-way",
        action="store_true",
        help="Same as train_scene: default drop already on; this fails if the road dump is missing.",
    )
    ww.add_argument(
        "--keep-wrong-way",
        action="store_true",
        help="Keep wrong-way frames (opt out of the default drop).",
    )
    parser.add_argument("--road-dump", default="", help="Same as train_scene --road-dump.")
    from python.train.frame_quality import DEFAULT_MAX_ROAD_CLIP

    parser.add_argument(
        "--max-road-clip",
        type=float,
        default=DEFAULT_MAX_ROAD_CLIP,
        help=f"Same as train_scene --max-road-clip (default {DEFAULT_MAX_ROAD_CLIP}; -1 disables).",
    )
    return parser


def _refuse(message: str) -> int:
    print(f"eval: {message}", file=sys.stderr)
    return 2


def load_eval_net(path: Path) -> tuple[Any, dict[str, Any]]:
    """CPU net + meta from a scene-net epoch checkpoint. Honors preset and route_cond."""
    from python.train.epoch_ckpt import build_net, load_epoch_ckpt, load_into_net

    payload = load_epoch_ckpt(path)
    meta = dict(payload["meta"] or {})
    weights = payload["weights"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{path.name}: checkpoint has no weights")
    net = build_net(meta, weights)
    load_into_net(net, weights)
    net.eval()
    return net, meta


def prepare_holdout(
    strip_roots: tuple[Path, ...],
    holdout_source: Path,
    *,
    cams: tuple[str, ...] | None,
    mask_files: str,
    mask_events: bool,
    drop_wrong_way: bool,
    require_road_dump: bool,
    max_road_clip: float,
    road_dump: str,
    holdout_blocks: int,
    cache: Any,
    cam_age: bool,
    route_cond: bool,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], str, Path]:
    """The trainer's selected rows and pass plan, in the same order as train_directory."""
    from python.data.raw_frames import refuse_lossy
    from python.train.continual import CHUNK_MAX, pass_plan
    from python.train.event_mask import drop_masked, mask_file_sources, selection_line
    from python.train.scene_loader import select_training_rows
    from python.train.strip_sources import combined_pass_plan, load_multi_source_rows

    for one in strip_roots:
        refuse_lossy(one)
    state_rows, source_meta = load_multi_source_rows(strip_roots, holdout_source=holdout_source)
    root = Path(holdout_source).resolve()
    if not source_meta.get("multi"):
        mask_sources = mask_file_sources(root, state_rows, mask_files)
    else:
        mask_sources = []
    dropped: dict[str, int] = {}
    if len(strip_roots) == 1:
        rows, _missing, _excluded = select_training_rows(root, state_rows, cams=cams, dropped=dropped)
    else:
        rows = []
        for one in strip_roots:
            subset = [r for r in state_rows if Path(str(r.get("source_root") or one)).resolve() == one.resolve()]
            kept, _miss, _excl = select_training_rows(one, subset, cams=cams, dropped=dropped)
            rows.extend(kept)
    if mask_events:
        from python.train.event_mask import describe, mask_event_rows

        before = len(rows)
        rows, event_counts = mask_event_rows(root, rows)
        print(f"[GVD] mask-events: {describe(event_counts, before)}", flush=True)
    selection_before = rows
    if source_meta.get("multi"):
        from python.train.event_mask import apply_mask_files_multi

        rows, mask_sources = apply_mask_files_multi(strip_roots, rows, mask_files)
    elif mask_sources:
        rows = drop_masked(rows, set().union(*(source["ids"] for source in mask_sources)))
    if drop_wrong_way or float(max_road_clip) >= 0.0:
        from python.train.frame_quality import (
            ROAD_CLIP_SIDECAR,
            FrameQualityError,
            apply_frame_quality,
            describe_quality,
        )
        from python.train.scene_loader import recompute_dt

        before_q = len(rows)
        sidecar = None
        cache_root = getattr(cache, "root", None) if cache is not None else None
        if cache_root:
            sidecar = Path(cache_root) / ROAD_CLIP_SIDECAR
        try:
            rows, quality_stats = apply_frame_quality(
                rows,
                drop_wrong_way=bool(drop_wrong_way),
                max_road_clip=float(max_road_clip),
                road_dump=road_dump or None,
                require_road_dump=bool(require_road_dump),
                cams=cams,
                cache=cache,
                clip_sidecar=sidecar,
            )
        except FrameQualityError as exc:
            raise SystemExit(str(exc)) from exc
        rows = recompute_dt(rows)
        print(f"[GVD] frame-quality: {describe_quality(quality_stats, before_q)}", flush=True)
    selection_text, _meta = selection_line(state_rows, selection_before, rows, mask_sources)
    print(selection_text, flush=True)
    if int(holdout_blocks) > 1:
        print(f"[GVD] holdout blocks: {int(holdout_blocks)}", flush=True)
    if cam_age:
        from python.data.cam_hold import annotate_hold

        if source_meta.get("multi"):
            plan_for_hold = combined_pass_plan(
                rows,
                length=CHUNK_MAX,
                stride=CHUNK_MAX // 2,
                blocks=int(holdout_blocks),
                holdout_key=source_meta.get("holdout_key"),
            )
        else:
            plan_for_hold = pass_plan(rows, CHUNK_MAX, CHUNK_MAX // 2, blocks=int(holdout_blocks))
        annotate_hold(state_rows, rows, plan_for_hold)
    else:
        if source_meta.get("multi"):
            plan_for_hold = combined_pass_plan(
                rows,
                length=CHUNK_MAX,
                stride=CHUNK_MAX // 2,
                blocks=int(holdout_blocks),
                holdout_key=source_meta.get("holdout_key"),
            )
        else:
            plan_for_hold = pass_plan(rows, CHUNK_MAX, CHUNK_MAX // 2, blocks=int(holdout_blocks))
    if route_cond:
        from python.train.route_cond import attach_route_cmds, route_counts_line

        print(route_counts_line(attach_route_cmds(rows, missing="mask")), flush=True)
    from python.train.continual import HOLDOUT_FAST_MPS, holdout_mix

    mix = holdout_mix(plan_for_hold)
    print(
        f"[GVD] holdout: rows={mix['rows']} slices={mix['slices']} fast={mix['fast']} slow={mix['slow']} "
        f"no_speed={mix['no_speed']} (fast is row speed >= {HOLDOUT_FAST_MPS} m/s)",
        flush=True,
    )
    return rows, plan_for_hold, selection_text, root


def evaluate_checkpoint(
    ckpt: Path,
    strip_roots: tuple[Path, ...],
    *,
    holdout_source: Path,
    holdout_blocks: int,
    mask_events: bool,
    mask_files: str,
    cams: tuple[str, ...] | None,
    cache: Any,
    eval_chunk: int | None,
    drop_wrong_way: bool,
    require_road_dump: bool,
    max_road_clip: float,
    road_dump: str,
    seed: int,
) -> dict[str, Any]:
    import torch

    from python.debug.remote_status import json_num
    from python.train.cam_strip import canonical_cams
    from python.train.continual import CHUNK_MAX
    from python.train.model_size import checkpoint_preset
    from python.train.train_scene import _eval_sequence, val_summary

    net, meta = load_eval_net(ckpt)
    used_cams = canonical_cams(cams) if cams else (tuple(meta.get("cams") or ()) or None)
    if used_cams:
        used_cams = canonical_cams(used_cams)
    scale = float(meta.get("scale") or 1.0)
    cam_age = bool(meta.get("cam_age"))
    route_cond = bool(getattr(net, "route_cond", False))
    preset = checkpoint_preset(meta)
    heads = tuple(meta.get("heads") or ()) or None
    chunk = int(eval_chunk) if eval_chunk is not None else int(meta.get("eval_chunk") or CHUNK_MAX)
    print(
        f"[GVD] eval-only ckpt={ckpt.name} preset={preset} route_cond={int(route_cond)} "
        f"cam_age={int(cam_age)} scale={scale} eval_chunk={chunk}",
        flush=True,
    )
    print(f"[GVD] seed={int(seed)}", flush=True)

    rows, plan, _selection, root = prepare_holdout(
        strip_roots,
        holdout_source,
        cams=used_cams,
        mask_files=mask_files,
        mask_events=bool(mask_events),
        drop_wrong_way=bool(drop_wrong_way),
        require_road_dump=bool(require_road_dump),
        max_road_clip=float(max_road_clip),
        road_dump=road_dump,
        holdout_blocks=int(holdout_blocks),
        cache=cache,
        cam_age=cam_age,
        route_cond=route_cond,
    )
    del rows  # plan already references the selected rows
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        major, minor = torch.cuda.get_device_capability(0)
        print(
            f"[GVD] eval-only device=cuda capability={major}.{minor} amp=False precision=fp32",
            flush=True,
        )
    else:
        print("[GVD] eval-only device=cpu amp=False precision=fp32", flush=True)
    net = net.to(device)
    val_pairs: list[tuple[float, float]] = []
    val_pairs_long: list[tuple[float, float]] = []
    val_parts: dict[str, float] = {"lane_reg": 0.0, "conf_bce": 0.0, "n": 0}
    slices = 0
    holdout_rows = 0
    for item in plan:
        if item.get("kind") != "holdout":
            continue
        slices += 1
        slice_rows = list(item["rows"])
        holdout_rows += len(slice_rows)
        # Tagged segments and --holdout-blocks K use incoming=reset. Untagged K=1
        # can say incoming=end (GRU from the last train chunk); without a train
        # forward that state is the same fallback the trainer uses when end_state
        # is missing: a fresh GRU.
        state = net.initial_state(device=device, batch=1)
        _eval_sequence(
            net,
            root,
            slice_rows,
            state,
            device=device,
            amp=False,
            view=None,
            pairs=val_pairs,
            cams=used_cams,
            cache=cache,
            heads=heads,
            scale=scale,
            reset_every=int(chunk),
            pairs_long=val_pairs_long,
            parts=val_parts,
        )
    scored = val_summary(val_pairs, val_pairs_long, val_parts) if val_pairs else {}
    payload = {
        "n": 0 if not val_pairs else scored.get("n"),
        "r": json_num(scored.get("r")),
        "rms": json_num(scored.get("rms")),
        "rms_mean": json_num(scored.get("rms_mean")),
    }
    print(
        f"[GVD] val n={payload['n']} r={payload['r']} rms={payload['rms']} "
        f"rms_mean={payload['rms_mean']} slices={slices} holdout_rows={holdout_rows} "
        f"eval_chunk={int(chunk)}",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from python.train.cam_strip import parse_cams
    from python.train.frame_quality import FrameQualityError
    from python.train.train_scene import frame_quality_from_args, resolve_holdout_blocks
    from python.train.strip_sources import parse_strips_arg, resolve_holdout_source

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        return _refuse(f"checkpoint not found: {ckpt}")
    try:
        strip_roots = parse_strips_arg(args.strips)
        holdout_root = resolve_holdout_source(strip_roots, args.holdout_source or None)
        holdout_blocks = resolve_holdout_blocks(args.holdout_blocks)
        cams = parse_cams(args.cams) if args.cams else None
        drop_wrong_way, require_road_dump, max_road_clip, road_dump = frame_quality_from_args(args)
    except (ValueError, SystemExit) as exc:
        message = str(exc)
        if message in {"0", "1", "2"}:
            message = "invalid arguments"
        return _refuse(message)

    cache = None
    if args.cache:
        from python.train.frame_cache import FrameCache

        cache = FrameCache(args.cache)
        if getattr(cache, "stale", False):
            cache.close()
            return _refuse(f"cache refused: {cache.reason}")
    try:
        result = evaluate_checkpoint(
            ckpt,
            strip_roots,
            holdout_source=holdout_root,
            holdout_blocks=holdout_blocks,
            mask_events=bool(args.mask_events),
            mask_files=str(args.mask_files or ""),
            cams=cams,
            cache=cache,
            eval_chunk=args.eval_chunk,
            drop_wrong_way=drop_wrong_way,
            require_road_dump=require_road_dump,
            max_road_clip=max_road_clip,
            road_dump=road_dump,
            seed=int(args.seed),
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except FileNotFoundError as exc:
        return _refuse(f"checkpoint not found: {exc}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, FrameQualityError) as exc:
        return _refuse(f"invalid checkpoint {ckpt}: {exc}")
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        message = str(exc)
        if message in {"1", "2"}:
            message = "refused"
        return _refuse(message)
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    sys.exit(main())
