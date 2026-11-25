#!/usr/bin/env python3
"""Automatic offline QC & highway lane-spread report for Recorder v2 clips.

Read-only by default; pass ``--write`` to write ``clip_qc.json`` and
``clip_qc_mask.jsonl`` inside each clip directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.train.clip_qc import (  # noqa: E402
    analyze_clip,
    format_highway_spread,
    merge_highway_spreads,
    write_clip_qc,
)
from python.train.frame_quality import load_quality_roads  # noqa: E402


def _find_clips(target: Path) -> list[Path]:
    if (target / "state.jsonl").is_file() or (target / "collect.json").is_file():
        return [target]
    clips = sorted(p for p in target.iterdir() if p.is_dir() and ((p / "state.jsonl").is_file() or p.name.startswith("clip_")))
    return clips


def _roads_for_dir(roads_dir: Path | None) -> dict[str, list]:
    cache: dict[str, list] = {}
    if roads_dir is None or not roads_dir.is_dir():
        return cache
    for f in sorted(roads_dir.iterdir()):
        if f.name.endswith(".json.gz") or f.suffix == ".json":
            stem = f.name[:-8] if f.name.endswith(".json.gz") else f.stem
            try:
                cache[stem] = load_quality_roads(f)
            except Exception:
                continue
    return cache


def _clip_map_name(clip_dir: Path) -> str | None:
    tags_p = clip_dir / "clip_tags.json"
    if tags_p.is_file():
        try:
            m = json.loads(tags_p.read_text(encoding="utf-8")).get("map")
            if m:
                return str(m)
        except Exception:
            pass
    st = clip_dir / "state.jsonl"
    if st.is_file():
        for line in st.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
            try:
                m = json.loads(line).get("map")
                if m:
                    return str(m)
            except Exception:
                continue
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline Recorder v2 clip QC and target mask generator.")
    ap.add_argument("path", help="Path to one clip directory or a session directory containing clip_* folders.")
    ap.add_argument("--roads-dir", default="", help="Optional folder with <map>.json.gz road dumps.")
    ap.add_argument("--road-dump", default="", help="Optional single <map>.json.gz road dump.")
    ap.add_argument("--skip-images", action="store_true", help="Skip raw PNG readability check.")
    ap.add_argument("--write", action="store_true", help="Write clip_qc.json and clip_qc_mask.jsonl per clip.")
    args = ap.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        print(f"error: path not found: {target}", file=sys.stderr)
        return 2
    clips = _find_clips(target)
    if not clips:
        print(f"error: no clips found under {target}", file=sys.stderr)
        return 2

    single_roads = load_quality_roads(args.road_dump) if args.road_dump else None
    roads_by_map = _roads_for_dir(Path(args.roads_dir)) if args.roads_dir else {}

    counts = {"train": 0, "pretrain_only": 0, "reject": 0}
    spreads = []
    results = []
    for clip in clips:
        m = _clip_map_name(clip)
        roads = single_roads if single_roads is not None else (roads_by_map.get(m) if m else None)
        res = analyze_clip(clip, roads=roads, check_images=not args.skip_images)
        if args.write:
            write_clip_qc(clip, res)
        counts[res["clip_class"]] = counts.get(res["clip_class"], 0) + 1
        spreads.append(res["highway_lane_spread"])
        results.append({k: v for k, v in res.items() if k != "rows"})
        rc = ",".join(res["reason_codes"]) or "ok"
        masks = ",".join(f"{k}={v}" for k, v in res["reason_counts"].items() if v) or "none"
        print(
            f"{res['clip']}: class={res['clip_class']} rows={res['total_rows']} "
            f"usable={res['usable_train_rows']} ({res['usable_train_s']:.1f}s) "
            f"reasons=[{rc}] masks=[{masks}]"
        )

    session_spread = merge_highway_spreads(spreads)
    print(f"summary: train={counts['train']} pretrain_only={counts['pretrain_only']} reject={counts['reject']}")
    print(f"highway_spread: {format_highway_spread(session_spread)}")
    if args.write and len(clips) > 1:
        (target / "session_qc.json").write_text(
            json.dumps(
                {
                    "kind": "gvd_session_qc",
                    "counts": counts,
                    "highway_lane_spread": session_spread,
                    "clips": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
