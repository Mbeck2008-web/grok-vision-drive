#!/usr/bin/env python3
"""Export the ONNX of one epoch checkpoint (never the in-memory best weights).

    py -3 scripts/export_epoch_onnx.py --ckpt <file>.ckpt_ep3_step12332.npz --out <file.onnx>
    py -3 scripts/export_epoch_onnx.py --ckpt-dir <folder> --epoch 3 [--run 20261021T040702Z] --out <file.onnx>

``--out`` is create-only. The ONNX carries the checkpoint name, sha256,
epoch and step in its metadata (``gvd_ckpt_*``); ``scripts/eval_onnx_holdout.py``
prints them on its val line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one epoch checkpoint to ONNX")
    parser.add_argument("--ckpt", type=Path, default=None, help="One .ckpt_ep<N>_step<S>.npz file.")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Folder to search with --epoch.")
    parser.add_argument("--epoch", "--export-epoch", dest="epoch", type=int, default=None)
    parser.add_argument("--run", default=None, help="Run id (the run<...> part of the name) when a folder has several runs.")
    parser.add_argument("--out", type=Path, required=True, help="ONNX path (create-only).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from python.train.epoch_ckpt import export_epoch_onnx, find_epoch_ckpt

    if args.ckpt is not None:
        ckpt = args.ckpt
    elif args.ckpt_dir is not None and args.epoch is not None:
        try:
            ckpt = find_epoch_ckpt(args.ckpt_dir, args.epoch, args.run)
        except (FileNotFoundError, ValueError) as exc:
            print(f"export: {exc}", file=sys.stderr)
            return 2
    else:
        print("export: pass --ckpt, or --ckpt-dir with --epoch", file=sys.stderr)
        return 2
    try:
        info = export_epoch_onnx(ckpt, args.out)
    except FileExistsError as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 2
    print(
        f"[GVD] exported epoch={info['epoch']} step={info['step']} onnx={info['onnx']} "
        f"onnx_sha256={info['onnx_sha256']} ckpt_sha256={info['ckpt_sha256']} val={json.dumps(info['val'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
