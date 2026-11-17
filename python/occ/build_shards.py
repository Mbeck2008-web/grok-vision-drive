"""Recording -> occupancy shards (training only).

    python -m python.occ.build_shards --rec <data dir>/scene_lanes/<rec> --out <nvme>/occ_shards

Per ``state.jsonl`` row ``i``: RGB of ``--cams`` (runtime input), labels from
``raw/<cam>_depth`` + ``raw/<cam>_sem`` of the ``train_gt`` cameras and the
row's traffic boxes, and ``meta.recover`` from the row tags. Rows without
depth for any train_gt camera are skipped and counted, never filled in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from python.occ.camera import CameraModel, cameras_from_config
from python.occ.grid import DEFAULT_GRID, VoxelGrid
from python.occ.labels import depth_to_m, CamFrame, boxes_from_row, vo_target, voxel_labels
from python.occ.shards import ShardWriter, is_recovery_row


def build_from_rows(
    rows: Sequence[Mapping[str, Any]],
    read: Callable[[str, int], np.ndarray | None],
    out_dir: Path | str,
    *,
    rgb_cams: Sequence[str],
    gt_cams: Mapping[str, CameraModel],
    sem_classes: Mapping[str, Any] | None,
    grid: VoxelGrid = DEFAULT_GRID,
    prefix: str = "occ",
    max_bytes: int | None = None,
) -> dict[str, Any]:
    kw = {"max_bytes": max_bytes} if max_bytes else {}
    stats = {"written": 0, "skipped_no_depth": 0, "skipped_no_rgb": 0, "recover": 0}
    prev_row = None
    with ShardWriter(out_dir, prefix, **kw) as w:
        for row in rows:
            try:
                i = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            rgb, cam_t = {}, {}
            stamps = row.get("cam_t") if isinstance(row.get("cam_t"), Mapping) else {}
            for cam in rgb_cams:
                img = read(cam, i)
                if img is None or img.ndim != 3 or img.shape[2] < 3:
                    continue  # a camera that did not render this tick is missing, not filled in
                rgb[cam] = np.ascontiguousarray(img[..., 2::-1])  # BGR(A) -> RGB
                cam_t[cam] = stamps.get(cam, row.get("t"))
            if not rgb:
                stats["skipped_no_rgb"] += 1
                prev_row = row
                continue
            frames = []
            for cid, cam in gt_cams.items():
                d = read(f"{cid}_depth", i)
                if d is None:
                    continue
                s = read(f"{cid}_sem", i)
                frames.append(CamFrame(cam.scaled(d.shape[1], d.shape[0]), d, s))
            if not frames:
                stats["skipped_no_depth"] += 1
                prev_row = row
                continue
            lab = voxel_labels(frames, sem_classes, boxes_from_row(row), grid=grid, seed=i)
            lab["vo_t"] = vo_target(prev_row, row)
            for f in frames:  # depth-bin head targets (training only), 1/4 res float16 metres; NaN = invalid
                dm = depth_to_m(f.depth)
                lab[f"depth_{f.cam.cam_id}"] = np.where(np.isfinite(dm), dm, np.nan)[::4, ::4].astype(np.float16)
            prev_row = row
            rec = is_recovery_row(row)
            meta = {"i": i, "t": row.get("t"), "recover": rec, "cams": sorted(rgb), "cam_t": cam_t, "label_cams": [f.cam.cam_id for f in frames]}
            w.write(f"s{i:08d}", meta=meta, rgb=rgb, arrays=lab)
            stats["written"] += 1
            stats["recover"] += int(rec)
    stats["shards"] = [str(p) for p in w.paths]
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    from python.data.raw_frames import load_raw_camera
    from python.data.strip_writer import read_state_jsonl
    from python.train.sensor_aux import read_sem_classes

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cams", default="narrow,main,wide,pillarL,pillarR,repeatL,repeatR,rear")
    ap.add_argument("--gt-cams", default="main,wide")
    a = ap.parse_args(argv)
    root = Path(a.rec)
    rows = read_state_jsonl(root / "state.jsonl")
    gt_ids = tuple(c for c in a.gt_cams.split(",") if c)
    stats = build_from_rows(
        rows,
        lambda cam, i: load_raw_camera(root, cam, i),
        a.out,
        rgb_cams=[c for c in a.cams.split(",") if c],
        gt_cams=cameras_from_config(ids=gt_ids),
        sem_classes=read_sem_classes(root),
        prefix=root.name.replace(".", "_") or "occ",
    )
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
