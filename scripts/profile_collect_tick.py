#!/usr/bin/env python3
"""Profile one collector camera tick offline with stub cameras (item 0, 2026-11-24).

Stages per tick, in milliseconds:

* per camera: ``read`` (the stream_raw call; a stub sleep of ``--read-ms`` stands in
  for BeamNG's shared-memory copy), ``colour_decode`` (colour_to_bgr) and
  ``gt_decode`` (train_gt_from_raw: depth cm + semantic ids),
* ``fetch_total``: the whole ``_poll_cameras`` call, serial and ``--cam-fetch parallel``,
* ``submit``: handing the tick to the background writer (the only disk-side cost
  the capture loop pays),
* writer thread: ``encode_ms`` (lossless PNG for every plane) and ``commit_ms``
  (files + state line), measured synchronously on the same tick.

Cameras use config/cameras.yaml ids and ``live_res`` (times ``--scale``), with
depth/sem where ``train_gt`` is set. This is a stub: it shows where CPU time goes
in GVD's own code, not BeamNG render time, and it is not a live Hz number.

    python scripts/profile_collect_tick.py --ticks 5 --json /tmp/profile.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


class StubCam:
    def __init__(self, cid: str, w: int, h: int, gt: bool, read_s: float, seed: int) -> None:
        self.name = f"gvd_collect_{cid}"
        self.resolution = (w, h)
        self.is_render_depth = gt
        self.is_render_annotations = gt
        self.near_far_planes = (0.05, 300.0)
        self.gvd_sem_palette = {"road": (128, 64, 128), "car": (0, 0, 142)}
        self.read_s = read_s
        rng = np.random.default_rng(seed)
        # BeamNGpy hands back RGBA bytes; depth float32; annotation RGBA.
        self._colour = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8).tobytes()
        self._depth = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32).tobytes()
        ann = np.zeros((h, w, 4), dtype=np.uint8)
        ann[..., :3] = (128, 64, 128)
        self._ann = ann.tobytes()

    def stream_raw(self) -> dict[str, Any]:
        time.sleep(self.read_s)
        raw: dict[str, Any] = {"colour": bytes(self._colour)}
        if self.is_render_depth:
            raw["depth"] = bytes(self._depth)
        if self.is_render_annotations:
            raw["annotation"] = bytes(self._ann)
        return raw


def build_rig(scale: float, read_s: float) -> dict[str, StubCam]:
    from python.sensors.cameras import CAM_IDS, load_camera_config

    cfg = load_camera_config()
    cams: dict[str, StubCam] = {}
    for i, spec in enumerate(cfg.get("cameras") or []):
        cid = str(spec.get("id") or "")
        if cid not in CAM_IDS:
            continue
        res = spec.get("live_res") or [640, 480]
        w, h = max(8, int(int(res[0]) * scale)), max(8, int(int(res[1]) * scale))
        cams[cid] = StubCam(cid, w, h, bool(spec.get("train_gt")), read_s, i)
    return cams


def per_camera(cams: dict[str, StubCam]) -> dict[str, dict[str, float]]:
    import scripts.collect_scene_lanes as collect
    from python.data import raw_frames
    from python.sensors.cameras import colour_to_bgr

    out: dict[str, dict[str, float]] = {}
    for cid, cam in cams.items():
        t0 = time.perf_counter()
        raw = cam.stream_raw()
        t1 = time.perf_counter()
        image = colour_to_bgr(collect._colour(raw), cam.resolution, tone=False)
        t2 = time.perf_counter()
        gt = raw_frames.train_gt_from_raw(cid, cam, raw)
        t3 = time.perf_counter()
        assert image is not None, cid
        out[cid] = {
            "read": (t1 - t0) * 1000.0,
            "colour_decode": (t2 - t1) * 1000.0,
            "gt_decode": (t3 - t2) * 1000.0,
            "res": list(cam.resolution),
            "gt_planes": len(gt),
        }
    return out


def profile(ticks: int, scale: float, read_ms: float) -> dict[str, Any]:
    import scripts.collect_scene_lanes as collect
    from python.data.raw_frames import RawFrameWriter, encode_cameras
    from python.data.raw_queue import RawWriteQueue

    cams = build_rig(scale, read_ms / 1000.0)
    result: dict[str, Any] = {
        "note": "stub cameras (sleep stands in for the shared-memory read); GVD CPU time only, not a live Hz claim",
        "ticks": ticks,
        "scale": scale,
        "read_ms": read_ms,
    }
    for mode in ("serial", "parallel"):
        per: dict[str, list[dict[str, float]]] = {cid: [] for cid in cams}
        fetch, submit, total = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            queue = RawWriteQueue(RawFrameWriter(Path(tmp), [{"id": c} for c in cams]), max_depth=64)
            try:
                for i in range(ticks):
                    for cid, row in per_camera(cams).items():
                        per[cid].append(row)
                    t0 = time.perf_counter()
                    frames, stamps, _reasons = collect._poll_cameras(cams, parallel=(mode == "parallel"))
                    frames.update(collect.take_train_gt())
                    t1 = time.perf_counter()
                    queue.submit(frames, t=0.1 * i, timestamps=stamps, tags={"gt_stride": 1})
                    t2 = time.perf_counter()
                    fetch.append((t1 - t0) * 1000.0)
                    submit.append((t2 - t1) * 1000.0)
                    total.append((t2 - t0) * 1000.0)
            finally:
                queue.close()
        result[mode] = {
            "stages_ms": {
                "fetch_total": statistics.median(fetch),
                "submit": statistics.median(submit),
                "tick_total": statistics.median(total),
            },
            "per_cam_ms": {
                cid: {
                    "read": statistics.median(r["read"] for r in rows),
                    "colour_decode": statistics.median(r["colour_decode"] for r in rows),
                    "gt_decode": statistics.median(r["gt_decode"] for r in rows),
                    "res": rows[0]["res"],
                    "gt_planes": rows[0]["gt_planes"],
                }
                for cid, rows in per.items()
            },
        }
    # Writer-thread cost of one full tick, measured synchronously.
    frames, _st, _r = collect._poll_cameras(cams, parallel=False)
    frames.update(collect.take_train_gt())
    with tempfile.TemporaryDirectory() as tmp:
        writer = RawFrameWriter(Path(tmp), [{"id": c} for c in cams])
        t0 = time.perf_counter()
        payload = encode_cameras(frames)
        t1 = time.perf_counter()
        writer.write_encoded(payload, t=0.0, tags={"gt_stride": 1})
        t2 = time.perf_counter()
        writer.close()
    result["writer"] = {"encode_ms": (t1 - t0) * 1000.0, "commit_ms": (t2 - t1) * 1000.0, "planes": len(frames)}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply each camera's live_res (1.0 = real size).")
    ap.add_argument("--read-ms", type=float, default=10.0, help="Stub shared-memory read time per camera.")
    ap.add_argument("--json", default="", help="Write the result here too.")
    args = ap.parse_args(argv)
    result = profile(max(1, int(args.ticks)), float(args.scale), float(args.read_ms))
    for mode in ("serial", "parallel"):
        st = result[mode]["stages_ms"]
        print(f"profile {mode}: fetch {st['fetch_total']:.1f} ms, submit {st['submit']:.2f} ms, tick {st['tick_total']:.1f} ms")
    for cid, row in result["serial"]["per_cam_ms"].items():
        print(f"profile cam {cid} {row['res'][0]}x{row['res'][1]} gt={row['gt_planes']}: read {row['read']:.1f} "
              f"colour {row['colour_decode']:.1f} gt {row['gt_decode']:.1f} ms")
    w = result["writer"]
    print(f"profile writer thread: encode {w['encode_ms']:.1f} ms, commit {w['commit_ms']:.1f} ms for {w['planes']} planes")
    print(f"profile note: {result['note']}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
