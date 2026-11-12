#!/usr/bin/env python3
"""Write an obstacles sidecar from depth and road geometry. Never touches gt.jsonl.

Train/eval only. The planner and the visualizer must not import this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.debug.gt_geom import RoadPoly, parse_road_edges  # noqa: E402
from python.debug.lane_targets import lane_gt_from_record  # noqa: E402
from python.train.obstacles import label_obstacle_frame, proven_blocked_names  # noqa: E402


class _Pose:
    def __init__(self, spec: dict[str, Any], width: int | None, height: int | None, origin: list[Any]) -> None:
        pos = list(spec.get("pos_m") or [0.0, 1.2, 1.2])
        self.x = float(pos[0]) + float(origin[0] if len(origin) > 0 else 0.0)
        self.y = float(pos[1]) + float(origin[1] if len(origin) > 1 else 0.0)
        self.z = float(pos[2]) + float(origin[2] if len(origin) > 2 else 0.0)
        self.yaw_deg = float(spec.get("yaw_deg") or 0.0)
        self.pitch_deg = float(spec.get("pitch_deg") or 0.0)
        self.fov_v_deg = float(spec.get("fov_v") or 70.0)
        res = spec.get("live_res") or [640, 480]
        self.width = int(width or res[0])
        self.height = int(height or res[1])
        self.cam_id = str(spec.get("id") or "")


def _pose(cam: str, width: int | None, height: int | None) -> _Pose | None:
    from python.sensors.cameras import load_camera_config

    cfg = load_camera_config()
    origin = list(cfg.get("origin_offset") or [0.0, 0.0, 0.0])
    for spec in cfg.get("cameras") or []:
        if isinstance(spec, dict) and str(spec.get("id") or "") == cam:
            return _Pose(spec, width, height, origin)
    return None


def _roads(path: Path) -> list[RoadPoly]:
    if path.name.endswith(".gz"):
        from python.collect.road_dump import load_road_dump

        return list(load_road_dump(path)[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("roads") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    found: list[RoadPoly] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        poly = parse_road_edges(str(item.get("id") or item.get("road_id") or "road"), item.get("edges"))
        if poly is not None:
            found.append(poly)
    return found


def _pick_road(roads: list[RoadPoly], row: dict[str, Any]) -> RoadPoly | None:
    if not roads:
        return None
    tags = row.get("tags") if isinstance(row.get("tags"), dict) else {}
    road_tag = tags.get("road") if isinstance(tags.get("road"), dict) else {}
    wanted = str(road_tag.get("road_id") or row.get("road_id") or "")
    if wanted:
        for poly in roads:
            if poly.road_id == wanted:
                return poly
    return roads[0]


def _ego(row: dict[str, Any], key: str) -> Any:
    ego = row.get("ego") if isinstance(row.get("ego"), dict) else {}
    if key in ego:
        return ego.get(key)
    return row.get(key)


def label_recording(
    root: Path,
    *,
    cam: str,
    roads: list[RoadPoly],
    blocked_names: list[str] | None,
    pose: _Pose,
    out_name: str,
) -> int:
    from python.data.raw_frames import load_raw_camera
    from python.train.sensor_aux import read_sem_classes

    state = root / "state.jsonl"
    out = root / out_name
    classes: dict[str, Any] = {}
    if (root / "sem_classes.json").is_file():
        classes = read_sem_classes(root)
    proven = proven_blocked_names(classes, blocked_names)
    count = 0
    with state.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as dest:
        for line_index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            index = line_index if row.get("i") is None else int(row["i"])
            tech = row.get("tech") if isinstance(row.get("tech"), dict) else row
            values, valid = lane_gt_from_record(tech if isinstance(tech, dict) else None)
            road = _pick_road(roads, row)
            depth = load_raw_camera(root, f"{cam}_depth", index)
            sem = load_raw_camera(root, f"{cam}_sem", index)
            record = tech if isinstance(tech, dict) else {}
            labeled = label_obstacle_frame(
                depth,
                pose=pose,
                lane_gt=values,
                lane_gt_valid=valid,
                road_width_m=record.get("road_width_m", record.get("width_m")),
                road_offset_m=record.get("road_offset_m", record.get("offset_m")),
                road_heading_err_rad=record.get("heading_err_rad", record.get("road_heading_err_rad")),
                road=road,
                ego_pos=_ego(row, "pos"),
                ego_dir=_ego(row, "dir"),
                ego_up=_ego(row, "up"),
                sem=sem,
                sem_classes=classes,
                blocked_names=proven,
                cam=cam,
                t=row.get("t"),
            )
            labeled["i"] = index
            dest.write(json.dumps(labeled, sort_keys=True) + "\n")
            count += 1
    src = "sem" if proven else "geom"
    names = ",".join(sorted(proven)) if proven else "-"
    print(f"wrote {out} rows={count} mask_src={src} blocked={names}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label obstacles from depth and road geometry (sidecar only)")
    ap.add_argument("root", help="Recording folder with state.jsonl")
    ap.add_argument("--cam", default="main")
    ap.add_argument("--roads", help="Road dump (.json or .json.gz) with middle/left/right xyz")
    ap.add_argument("--blocked-names", default="", help="Comma-separated class names copied from sem_classes.json")
    ap.add_argument("--pose-width", type=int, default=0)
    ap.add_argument("--pose-height", type=int, default=0)
    ap.add_argument("--out", default="obstacles.jsonl")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = root / args.out
    if out.name == "gt.jsonl" or out.exists():
        print(f"refusing to overwrite {out}", flush=True)
        return 2
    if not (root / "state.jsonl").is_file():
        print(f"missing {root / 'state.jsonl'}", flush=True)
        return 2
    pose = _pose(str(args.cam), args.pose_width or None, args.pose_height or None)
    if pose is None:
        print(f"no camera pose for {args.cam}", flush=True)
        return 2
    roads = _roads(Path(args.roads)) if args.roads else []
    names = [part.strip() for part in str(args.blocked_names).split(",") if part.strip()]
    return label_recording(
        root,
        cam=str(args.cam),
        roads=roads,
        blocked_names=names,
        pose=pose,
        out_name=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
