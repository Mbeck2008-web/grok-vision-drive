#!/usr/bin/env python3
"""Relabel a recording with the v103 painted-lane labels (body-centreline origin).

Same pipeline as the v102 labels (each main frame labelled at its own
``cams.main.t``, pose interpolated between the bracketing state rows, one
``LaneTracker`` per drive), with two changes:

* lateral x is measured from the body centreline, not the ref node. The ref
  node's offset is read from the vehicle's jbeam
  (``python.debug.vehicle_origin``), per row: a row's ``vehicle`` /
  ``vehicle_model`` field when the recording has one, else ``--vehicle``.
  rec5c rows do not record the model; the collector spawns ``etk800``
  (``python/collect/level_session.py``), the default.
* only lane-line materials count as paint (``painted_lanes.LANE_LINE_MATERIALS``).

Hop mask, applied at selection. ``--hop-mask`` (default
``mask_v102_hop.jsonl`` in the recording folder; one ``{"i": ...}`` per line)
lists rows whose walk hops between the two stripes of a double line. Those rows
are still walked, so the tracker state of the rows after them is the same as
without the mask, but they are written with an empty ``tech`` and
``hop_masked``: ``scene_net.loss_mask`` then leaves them out of training. A
missing mask file refuses to run unless ``--no-hop-mask`` says so; a mask id
that is not a state row refuses too.

Every labelled row declares its frame (``tech.gt_origin``,
``tech.gt_ref_right_of_origin_m``, ``tech.gt_labeler``); ``<out-stem>.meta.json``
is the header. Writes only those two files, both create-only; reads jsonl
with ``open``. Rows without a main frame are ``{"i", "t", "no_frame": true}``.

    python scripts/relabel_v103.py RECORDING --zip LEVELS/west_coast_usa.zip --vehicle-zip VEHICLES/etk800.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.debug.lane_targets import ORIGIN_CENTRELINE, lane_center_is_valid, tech_from_painted  # noqa: E402
from python.debug.painted_lanes import (  # noqa: E402
    LANE_LINE_MATERIALS,
    LaneTracker,
    PaintedLanes,
    decals_from_level_zip,
    is_paint,
)
from python.debug.vehicle_origin import ref_right_of_centre_m  # noqa: E402

LABELER = "v103"
TELEPORT_M = 2.0
HOP_MASK_DEFAULT = "mask_v102_hop.jsonl"


class MaskError(ValueError):
    """A hop-mask file that cannot be applied."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_hop_mask(path: Path, known: set[int]) -> tuple[set[int], dict[str, Any]]:
    """Row ids in a hop-mask jsonl and a description of it (file sha256, its meta header if present).

    Every line must be a JSON object with an integer ``i`` that is a state row.
    """
    ids: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                index = json.loads(line)["i"]
            except (ValueError, KeyError, TypeError) as exc:
                raise MaskError(f"{path.name} line {number}: not a JSON object with an i ({exc!r})") from None
            if isinstance(index, bool) or not isinstance(index, int):
                raise MaskError(f"{path.name} line {number}: i is not an integer")
            ids.add(index)
    unknown = sorted(ids - known)
    if unknown:
        raise MaskError(f"{path.name}: {len(unknown)} row id(s) not in state.jsonl ({unknown[:5]})")
    info: dict[str, Any] = {"file": path.name, "sha256": _sha256(path), "rows": len(ids)}
    meta_p = path.with_name(path.stem + ".meta.json")
    if meta_p.is_file():
        with open(meta_p, "r", encoding="utf-8") as f:
            head = json.load(f)
        info["meta"] = {k: head.get(k) for k in ("rule", "spec_sha256", "T_m", "count_masked", "source_labels_sha256") if k in head}
    return ids, info


def _yaw(d: Any) -> float:
    return math.atan2(float(d[1]), float(d[0]))


def _wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def frame_pose(rows: list[dict[str, Any]], k: int, tf: float) -> tuple[list[float], list[float], str, float | None]:
    """Ref-node pose at the frame time: interpolated in the bracket, else back-extrapolated (v102 rule)."""
    r = rows[k]
    e = r["ego"]
    if k > 0:
        p = rows[k - 1]
        ep = p["ego"]
        h = r["t"] - p["t"]
        dist = math.dist(ep["pos"][:2], e["pos"][:2])
        vmax = max(float(ep.get("speed_mps") or 0), float(e.get("speed_mps") or 0))
        same = p.get("segment") == r.get("segment")
        if same and h > 0 and dist <= vmax * h + TELEPORT_M and p["t"] <= tf <= r["t"]:
            al = (tf - p["t"]) / h
            pos = [ep["pos"][j] + al * (e["pos"][j] - ep["pos"][j]) for j in range(3)]
            y0, y1 = _yaw(ep["dir"]), _yaw(e["dir"])
            yw = y0 + al * _wrap(y1 - y0)
            dz = ep["dir"][2] + al * (e["dir"][2] - ep["dir"][2])
            return pos, [math.cos(yw), math.sin(yw), dz], "interp", al
    dt = tf - r["t"]
    v = float(e.get("speed_mps") or 0)
    yr = float(e.get("yaw_rate") or 0)
    yw = _yaw(e["dir"])
    pos = [e["pos"][0] + v * dt * math.cos(yw), e["pos"][1] + v * dt * math.sin(yw), e["pos"][2]]
    return pos, [math.cos(yw + yr * dt), math.sin(yw + yr * dt), e["dir"][2]], "extrap", None


def row_vehicle(row: dict[str, Any], default: str) -> str:
    for key in ("vehicle_model", "vehicle"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict) and v.get("model"):
            return str(v["model"])
    return default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--zip", required=True, help="the recording's level zip (content/levels/<map>.zip)")
    ap.add_argument("--vehicle", default="etk800", help="model for rows that do not record one")
    ap.add_argument("--vehicle-zip", default=None, help="content/vehicles/<model>.zip; default $BNG_HOME/content/vehicles")
    ap.add_argument("--out", default="gt_v103.jsonl")
    ap.add_argument("--hop-mask", default=HOP_MASK_DEFAULT, help="hop-mask jsonl (relative to the recording folder)")
    ap.add_argument("--no-hop-mask", action="store_true", help="relabel without a hop mask (diagnostics only)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = root / a.out
    meta_p = root / (Path(a.out).stem + ".meta.json")
    if out.name in ("gt.jsonl", "state.jsonl") or out.exists() or meta_p.exists():
        print(f"refusing to overwrite {out.name}", flush=True)
        return 2

    def rd(name: str) -> list[dict[str, Any]]:
        p = root / name
        if not p.is_file():
            return []
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    home = os.environ.get("BNG_HOME") or os.environ.get("BEAMNG_HOME")

    def vehicle_zip(model: str) -> Path | None:
        if a.vehicle_zip:
            return Path(a.vehicle_zip)
        if home:
            return Path(home) / "content" / "vehicles" / f"{model}.zip"
        return None

    t0 = time.time()
    rows = rd("state.jsonl")
    hop: set[int] = set()
    hop_info: dict[str, Any] | None = None
    if not a.no_hop_mask:
        mask_p = Path(a.hop_mask) if Path(a.hop_mask).is_absolute() else root / a.hop_mask
        if not mask_p.is_file():
            print(f"refusing: hop mask {mask_p.name} not found (pass --no-hop-mask to relabel without one)", flush=True)
            return 2
        try:
            hop, hop_info = read_hop_mask(mask_p, {int(r["i"]) for r in rows})
        except MaskError as exc:
            print(f"refusing: {exc}", flush=True)
            return 2
        print(f"hop mask {mask_p.name}: {len(hop)} rows, sha256 {hop_info['sha256']}", flush=True)
    cuts = rd("cuts.jsonl")
    reset_at = {int(c["i"]) for c in cuts if c.get("action", "") != "keep" and c.get("i") is not None}
    decals = [d for d in decals_from_level_zip(a.zip) if is_paint(d.get("material"))]
    lanes = PaintedLanes(decals)
    offsets: dict[str, tuple[float, str]] = {}

    def offset_for(model: str) -> float:
        if model not in offsets:
            offsets[model] = ref_right_of_centre_m(model, vehicle_zip(model))
            print(f"vehicle {model}: ref node {offsets[model][0]:+.3f} m right of the centreline ({offsets[model][1]})", flush=True)
        return offsets[model][0]

    trk = LaneTracker()
    n = n_main = n_valid = resets = n_hop = 0
    modes: dict[str, int] = {}
    last_k: int | None = None
    with open(out, "x", encoding="utf-8") as dst:
        for k, r in enumerate(rows):
            n += 1
            cam = (r.get("cams") or {}).get("main")
            if not cam:
                dst.write(json.dumps({"i": r["i"], "t": r["t"], "no_frame": True}) + "\n")
                continue
            n_main += 1
            tf = float(cam["t"])
            pos, d, mode, al = frame_pose(rows, k, tf)
            modes[mode] = modes.get(mode, 0) + 1
            reset = last_k is None or mode == "extrap"
            if last_k is not None:
                p = rows[last_k]
                if p.get("segment") != r.get("segment") or tf - p["cams"]["main"]["t"] > 1.0:
                    reset = True
                if any(j in reset_at for j in range(last_k + 1, k + 1)):
                    reset = True
            if reset:
                trk.reset()
                resets += 1
            model = row_vehicle(r, a.vehicle)
            ref_right = offset_for(model)
            got = lanes.lane_at(pos, d, tracker=trk, origin_right_m=-ref_right)
            tech = tech_from_painted(got, origin=ORIGIN_CENTRELINE, ref_right_of_origin_m=ref_right)
            tech.update(gt_map=r.get("map"), gt_time="cams.main.t", gt_pose=mode, gt_labeler=LABELER, gt_vehicle=model)
            rec = {"i": r["i"], "t": r["t"], "frame_t": tf, "frame": cam["file"],
                   "pose": {"mode": mode, "alpha": al, "pos": pos, "dir": d}, "tech": tech}
            if int(r["i"]) in hop:
                # Walked above (tracker continuity), but not a label: empty tech, so loss_mask is False.
                rec["tech"] = {}
                rec["hop_masked"] = hop_info["file"] if hop_info else True
                n_hop += 1
            else:
                n_valid += int(lane_center_is_valid(tech["lane_gt_valid"]))
            dst.write(json.dumps(rec) + "\n")
            last_k = k

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    meta = {
        "labeler": LABELER,
        "labeler_code_sha": git("rev-parse", "HEAD"),
        "labeler_tree_dirty": bool(git("status", "--porcelain")),
        "origin": {
            "x": ORIGIN_CENTRELINE,
            "y": "state pos ref node",
            "note": "x right, y forward; x is measured from the body centreline, the ref node sits at +ref_right m",
        },
        "vehicles": {m: {"ref_right_of_centre_m": v, "source": s} for m, (v, s) in offsets.items()},
        "paint_materials": sorted(LANE_LINE_MATERIALS),
        "paint_source": Path(a.zip).name,
        "paint_decals": len(decals),
        "label_time": "cams.main.t (pose interpolated between bracketing state rows)",
        "no_frame_rows": "written as {i,t,no_frame:true} with no tech",
        "tracker_resets": resets,
        "pose_modes": modes,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows_state": len(rows),
        "rows_written": n,
        "rows_main_labelled": n_main,
        "rows_no_frame": n - n_main,
        "valid_centre_5_10m": n_valid,
        "hop_mask": hop_info,
        "rows_hop_masked": n_hop,
        "hop_mask_selection": "hop-masked rows are walked but written with tech {} and hop_masked, so scene_net.loss_mask leaves them out",
        "recording_code_sha": None,
    }
    try:
        with open(root / "collect.json", "r", encoding="utf-8") as f:
            meta["recording_code_sha"] = json.load(f).get("code_sha")
    except (OSError, ValueError):
        pass
    with open(meta_p, "x", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
