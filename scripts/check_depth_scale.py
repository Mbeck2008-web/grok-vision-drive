#!/usr/bin/env python3
"""Offline depth-plane check on a recording: planar z or ray length, and the implied mount.

For each sampled frame of each ``<cam>_depth`` folder that has a matching
``<cam>_sem`` plane, the road pixels (STREET/ASPHALT) are back-projected
under both readings and a plane is fitted (``python/data/depth_check.py``).
The flatter reading wins by a factor of 3 in rms or the frame is
``unclear``. It also prints the camera height above the road and the pitch
against the road, next to the mount in ``cameras.json`` (or the yaml), so a
scale or mount error shows up as a height mismatch, not a bent plane.

Reads PNGs only. Never writes into the recording.

    py -3 scripts\\check_depth_scale.py E:\\GVD_data\\clips_v2\\<session>\\clip_000_... --cams main,wide --every 50

Exit 0 when at least 80% of fitted frames read ``planar_z``; 3 otherwise;
2 when nothing could be fitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS_SHARE = 0.8


def _cameras(rec: Path) -> dict[str, dict]:
    path = rec / "cameras.json"
    if path.is_file():
        cams = json.loads(path.read_text(encoding="utf-8")).get("cameras") or []
        return {str(c["id"]): c for c in cams if isinstance(c, dict) and c.get("id")}
    from python.data.raw_frames import camera_sidecar
    from python.sensors.cameras import load_camera_config

    return {c["id"]: c for c in camera_sidecar(load_camera_config())}


def main(argv: list[str] | None = None) -> int:
    import cv2

    from python.data.depth_check import depth_kind_verdict
    from python.data.raw_frames import camera_intrinsics

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--cams", default="", help="Comma list of base cameras. Empty: every <cam>_depth folder.")
    ap.add_argument("--frames", default="", help="Comma list of frame ids (file stems). Empty: every --every-th file.")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=40, help="Per camera.")
    args = ap.parse_args(argv)
    rec = args.recording
    raw = rec / "raw"
    classes_path = rec / "sem_classes.json"
    if not classes_path.is_file():
        print(f"depth check: no sem_classes.json in {rec}")
        return 2
    classes = json.loads(classes_path.read_text(encoding="utf-8")).get("classes") or {}
    cams = _cameras(rec)
    want = [c.strip() for c in args.cams.split(",") if c.strip()]
    bases = want or sorted(p.name[: -len("_depth")] for p in raw.glob("*_depth") if p.is_dir())
    stems = [s.strip() for s in args.frames.split(",") if s.strip()]
    verdicts: list[str] = []
    for base in bases:
        cam = cams.get(base)
        if cam is None:
            print(f"depth check {base}: no camera record, skipped")
            continue
        intr = cam.get("intrinsics") or camera_intrinsics(cam["fov_v"], cam["resolution"])
        files = sorted((raw / f"{base}_depth").rglob("*.png"))
        if stems:
            files = [f for f in files if f.stem in stems]
        else:
            files = files[:: max(1, int(args.every))][: max(1, int(args.max_frames))]
        for dfile in files:
            sfiles = list((raw / f"{base}_sem").rglob(dfile.name))
            if not sfiles:
                continue
            depth = cv2.imread(str(dfile), cv2.IMREAD_UNCHANGED)
            sem = cv2.imread(str(sfiles[0]), cv2.IMREAD_UNCHANGED)
            if depth is None or sem is None:
                print(f"depth check {base} {dfile.stem}: unreadable, skipped")
                continue
            got = depth_kind_verdict(depth, sem, classes, intr)
            z, ray = got["planar_z"], got["ray"]
            if z["rms_m"] is None:
                print(f"depth check {base} {dfile.stem}: {got['ground_px']} road px, too few to fit")
                continue
            verdicts.append(got["verdict"])
            print(
                f"depth check {base} {dfile.stem}: verdict {got['verdict']} "
                f"planar_z rms={z['rms_m'] * 100:.2f}cm inl={z['inlier_frac']:.3f} h={z['height_m']:.3f}m "
                f"pitch={z['pitch_deg']:+.2f}deg | ray rms={ray['rms_m'] * 100:.2f}cm inl={ray['inlier_frac']:.3f} "
                f"h={ray['height_m']:.3f}m | mount z={cam['pos_m'][2]:.3f}m pitch={-float(cam['pitch_deg']):+.2f}deg"
            )
    if not verdicts:
        print("depth check: nothing fitted")
        return 2
    share = verdicts.count("planar_z") / len(verdicts)
    final = "planar_z" if share >= PASS_SHARE else "mixed"
    print(f"depth check summary: {len(verdicts)} frames, planar_z share {share:.2f} -> verdict {final}")
    return 0 if final == "planar_z" else 3


if __name__ == "__main__":
    raise SystemExit(main())
