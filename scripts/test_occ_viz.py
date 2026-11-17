#!/usr/bin/env python3
"""Occupancy outputs -> viz: derived polylines, multi-frame motion, v1 message.

Schema: ``gvd_scene`` v1 as frozen in docs/specs/viz_cpp.md @ f3555bf
(PR #232). ``validate_v1`` below checks the v1 required fields and shapes
and ignores any other key, as the C++ parser does.

* road edges derived from drivable cells sit within 0.45 m of the synthetic
  road's real edges (asymmetric road, all 8 cameras, front and rear);
* a vehicle cluster sits on the box face the cameras see; a full footprint
  gives the box's centre, size and heading;
* motion state needs 3 frames (frame 1 and 2 are ``unknown``) and, with
  ego-motion compensation (straight and turning), names same_way /
  oncoming / crossing / alongside / parked correctly;
* the synthetic scene's outputs round-trip into a valid v1 message under
  60000 bytes; unknown / GT input keys change nothing; the validator
  ignores unknown message keys;
* red appears only on road edges and curbs; lanes are white or yellow by
  paint; parked is orange.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception.occ_geometry import DEFAULT_GRID, rig_cams  # noqa: E402
from python.perception.occ_outputs import OccTracker, objects, occ_outputs, road_edges  # noqa: E402
from python.perception.occ_sparse import OccPrediction  # noqa: E402
from python.train.occ_labels import build_occ_labels  # noqa: E402
from python.viz.occ_viz import MAX_DATAGRAM, MOTION_COLORS, V1_KEYS, encode, occ_to_viz  # noqa: E402
from synthetic_occ import SEM_CLASSES, render  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
CAMS = {c.cam_id: c.scaled(160) for c in rig_cams(CFG)}
ROAD = (-3.0, 4.2)


def validate_v1(msg) -> None:
    """v1 required fields (docs/specs/viz_cpp.md @ f3555bf). Unknown keys are ignored."""
    assert isinstance(msg, dict)
    for k in V1_KEYS:
        assert k in msg, f"v1 field {k} missing"
    assert msg["v"] == 1 and msg["type"] == "gvd_scene"
    assert isinstance(msg["seq"], int) and isinstance(msg["t"], (int, float))
    assert isinstance(msg["ego"], dict) and isinstance(msg["ego"]["speed_mps"], (int, float))
    assert isinstance(msg["ego"]["engaged"], bool)
    assert isinstance(msg["aeb"], str) and isinstance(msg["path_conf"], (int, float))

    def pts_ok(pts):
        assert isinstance(pts, list) and len(pts) <= 120
        for p in pts:
            assert isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)

    assert isinstance(msg["path"], dict)
    pts_ok(msg["path"]["pts"])
    assert len(msg["lanes"]) <= 16 and len(msg["edges"]) <= 8 and len(msg["tracks"]) <= 64 and len(msg["signs"]) <= 16
    for ln in msg["lanes"]:
        pts_ok(ln["pts"]); assert len(ln["pts"]) >= 2 and ln["paint"] in ("white", "yellow")
        assert ln.get("kind", "detected") in ("detected", "predicted")
    for ed in msg["edges"]:
        pts_ok(ed["pts"]); assert len(ed["pts"]) >= 2 and ed.get("kind", "detected") in ("detected", "predicted")
    for tr in msg["tracks"]:
        assert isinstance(tr["cls"], str) and all(isinstance(tr[k], (int, float)) for k in ("x", "y"))
        for k in ("yaw", "l", "w", "h", "v"):
            assert k not in tr or isinstance(tr[k], (int, float)), k
        assert "id" not in tr or isinstance(tr["id"], int)
    for s in msg["signs"]:
        assert isinstance(s["cls"], str) and s["state"] in ("red", "yellow", "green", "unknown")


def _scene_pred(vehicle_box=True) -> OccPrediction:
    planes = {cid: render(cam, box=vehicle_box, road=ROAD)[:2] for cid, cam in CAMS.items()}
    lab = build_occ_labels(planes, CAMS, sem_classes=SEM_CLASSES, stride=1)
    return OccPrediction(lab.keys, np.ones(lab.keys.size, np.float32), lab.cls, 0, 0)


def check_edges() -> None:
    pred = _scene_pred()
    edges = {e["side"]: np.asarray(e["pts"]) for e in road_edges(pred.keys, pred.cls)}
    assert set(edges) == {"left", "right"}, set(edges)
    for side, want in (("left", ROAD[0]), ("right", ROAD[1])):
        a = edges[side]
        err = np.abs(a[:, 0] - want)
        assert np.mean(err <= 0.45) >= 0.9, (side, float(np.median(err)), float(err.max()))
        assert a[:, 1].max() > 20.0 and a[:, 1].min() < -3.0, (side, a[:, 1].min(), a[:, 1].max())
        assert np.all(np.diff(a[:, 1]) > 0), "edge points ordered by y"
    print(f"edges: left median |err| {np.median(np.abs(edges['left'][:, 0] - ROAD[0])):.2f} m, "
          f"right {np.median(np.abs(edges['right'][:, 0] - ROAD[1])):.2f} m, y {edges['left'][:, 1].min():.1f}..{edges['left'][:, 1].max():.1f} OK")


def check_objects() -> None:
    # Cameras see the box's near face only: the cluster is that face.
    pred = _scene_pred()
    objs = objects(pred.keys, pred.cls)
    assert len(objs) == 1, objs
    o = objs[0]
    assert abs(o["x"]) < 0.3 and abs(o["y"] - 10.2) < 0.5 and 1.6 <= o["length"] <= 2.8, o
    assert o["yaw_src"] == "pca" and abs(o["yaw"]) < 0.2, o
    assert 1.2 <= o["height"] <= 2.0, o
    # A full footprint (a sized box in the labels) gives the box's size and heading.
    planes = {cid: render(cam, box=False, road=ROAD)[:2] for cid, cam in CAMS.items()}
    lab = build_occ_labels(planes, CAMS, sem_classes=SEM_CLASSES, stride=2,
                           vehicles=[{"center": np.array([6.0, 20.0, 0.2]), "size": (4.4, 1.8, 1.5), "yaw": 0.0}])
    full = objects(lab.keys, lab.cls)
    assert len(full) == 1, full
    f = full[0]
    assert abs(f["x"] - 6.0) < 0.3 and abs(f["y"] - 20.0) < 0.3, f
    assert abs(abs(f["yaw"]) - math.pi / 2) < 0.1 and abs(f["length"] - 4.4) <= 0.8 and abs(f["width"] - 1.8) <= 0.8, f
    print(f"objects: near-face cluster at ({o['x']:.2f}, {o['y']:.2f}); full box at ({f['x']:.2f}, {f['y']:.2f}) "
          f"yaw {f['yaw']:.2f} l {f['length']:.1f} w {f['width']:.1f} OK")


def _drive(world_tracks, *, ego_v=10.0, ego_w=0.0, frames=6, dt=0.1):
    """World-frame tracks (x0, y0, vx, vy) seen from a moving ego -> tracker reports per frame."""
    tr = OccTracker()
    ex, ey, eh = 0.0, 0.0, math.pi / 2  # ego world pose, heading +y
    reports = []
    for f in range(frames):
        t = f * dt
        if f:
            ex += ego_v * dt * math.cos(eh); ey += ego_v * dt * math.sin(eh)
            eh += ego_w * dt
        objs = []
        for (x0, y0, vx, vy) in world_tracks:
            wx, wy = x0 + vx * t - ex, y0 + vy * t - ey
            fwd = (math.cos(eh), math.sin(eh)); right = (math.sin(eh), -math.cos(eh))
            objs.append({"cls": "vehicle", "x": wx * right[0] + wy * right[1], "y": wx * fwd[0] + wy * fwd[1], "yaw": 0.0})
        reports.append(tr.update(objs, t, ego_speed_mps=ego_v, ego_yaw_rate=ego_w))
    return reports


def check_motion() -> None:
    cases = {"same_way": (0.0, 20.0, 0.0, 15.0), "oncoming": (-3.5, 40.0, 0.0, -12.0), "crossing": (-12.0, 25.0, 8.0, 0.0),
             "alongside": (3.6, 1.0, 0.0, 10.0), "parked": (6.0, 15.0, 0.0, 0.0)}
    for ego_w in (0.0, 0.3):
        reps = _drive(list(cases.values()), ego_w=ego_w)
        for f in (0, 1):
            assert all(r["motion"] == "unknown" for r in reps[f]), (f, [r["motion"] for r in reps[f]])
        got = {r["id"]: r for r in reps[-1]}
        assert len(got) == len(cases), got
        for (name, _), tid in zip(cases.items(), sorted(got)):
            assert got[tid]["motion"] == name, (ego_w, name, got[tid]["motion"], got[tid].get("vx"), got[tid].get("vy"))
        sw = got[sorted(got)[0]]
        # a car heading north, seen after the ego turned left by ego_w * 0.5 s
        assert abs(sw["speed_mps"] - 15.0) < 0.5 and abs(sw["yaw"] - (math.pi / 2 - ego_w * 0.5)) < 0.05, sw
    one = OccTracker().update([{"x": 0.0, "y": 10.0, "cls": "vehicle"}], 0.0, ego_speed_mps=10.0)
    assert one[0]["motion"] == "unknown" and "vx" not in one[0]
    print("motion: unknown for frames 1-2; same_way/oncoming/crossing/alongside/parked after 3 (straight and turning ego) OK")


def _outputs():
    pred = _scene_pred()
    tracker = OccTracker()
    for i in range(3):
        out = occ_outputs(pred, tracker, 0.1 * i, ego_speed_mps=0.0)
    out["lanes"] = [{"points": [[-1.8, 0.0], [-1.8, 30.0]], "material": "line_yellow"},
                    {"points": [{"x": 1.8, "y": 0.0}, {"x": 1.8, "y": 30.0}]}]
    out["crosswalks"] = [{"pts": [[-3.0, 30.0], [4.2, 30.0]]}]
    out["stop_lines"] = [{"pts": [[0.0, 28.0], [4.0, 28.0]]}]
    out["ego"]["engaged"] = True
    return out


def check_message() -> None:
    out = _outputs()
    msg = occ_to_viz(out, seq=7)
    validate_v1(msg)
    data = encode(msg)
    assert len(data) <= MAX_DATAGRAM
    back = json.loads(data)
    validate_v1(back)
    assert back == json.loads(json.dumps(msg)), "datagram round trip changed the message"
    assert len(back["edges"]) == 2 and len(back["tracks"]) == 1 and back["tracks"][0]["motion"] == "parked"
    assert [ln["paint"] for ln in back["lanes"]] == ["yellow", "white"]
    occ = back["occ"]
    assert len(occ["cells"]) == min(2500, out["cells"]["keys"].size) and occ["voxel"] == 0.4
    ix, iy, iz, k = occ["cells"][0]
    key = DEFAULT_GRID.key_of_cell(np.array([ix, iy, iz]))
    assert key in set(out["cells"]["keys"].tolist())
    # GT / unknown input keys are not read
    noisy = dict(out, gt_boxes=[{"x": 1, "y": 2}], depth=np.zeros((4, 4)), tracks_gt=[1], foo="bar")
    assert occ_to_viz(noisy, seq=7) == msg
    # the validator ignores unknown message keys; v1 alone still validates
    validate_v1(dict(msg, future_field={"a": 1}))
    validate_v1({k: msg[k] for k in V1_KEYS})
    # an oversize occupancy block is cut, never the v1 fields
    big = encode(msg, max_bytes=20000)
    small = json.loads(big)
    validate_v1(small)
    assert len(big) <= 20000 and len(small["occ"]["cells"]) < len(occ["cells"]) and small["edges"] == back["edges"]
    print(f"message: v1 valid, {len(data)} bytes, {len(occ['cells'])} occ cells, GT/unknown inputs ignored OK")


def check_colors() -> None:
    msg = occ_to_viz(_outputs())
    reds = []
    for group, recs in msg.items():
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, dict) and r.get("color") == "red":
                    reds.append((group, r.get("class")))
    assert reds and all(g == "edges" and c in ("road_edge", "curb") for g, c in reds), reds
    assert all(ln["color"] == ln["paint"] and ln["color"] in ("white", "yellow") for ln in msg["lanes"])
    assert "red" not in MOTION_COLORS.values() and MOTION_COLORS["parked"] == "orange"
    assert all(m["color"] == "white" for g in ("crosswalks", "stop_lines") for m in msg[g])
    print("colors: red only on edges/curbs; lanes white/yellow by paint; parked orange OK")


def check_runtime_net() -> None:
    try:
        import torch
    except ImportError:
        from gvd_test_env import skip_check

        skip_check("occ_viz runtime net", "torch missing")
        return
    from python.perception.occ_sparse import SparseOccNet

    cams = [c.scaled(64) for c in rig_cams(CFG)]
    net = SparseOccNet(cams).eval()
    pred = net.predict([torch.rand(1, 3, c.height, c.width) for c in cams], max_points=4096)
    msg = occ_to_viz(occ_outputs(pred, OccTracker(), 0.0))
    validate_v1(json.loads(encode(msg)))
    print(f"runtime: untrained SparseOccNet.predict -> outputs -> v1 message ({len(msg['occ']['cells'])} cells) OK")


def main() -> int:
    check_edges()
    check_objects()
    check_motion()
    check_message()
    check_colors()
    check_runtime_net()
    print("test_occ_viz: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
