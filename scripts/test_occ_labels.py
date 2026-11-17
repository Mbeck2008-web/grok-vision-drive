#!/usr/bin/env python3
"""Sparse occupancy labels from depth + semantics of all 8 rig cameras.

* voxelization: a ground plane at z=0.2 lands in z cell 5 only, as drivable;
  a box lands in its own cells as vehicle; back-projection inverts projection.
* sparse: stored cells are a small fraction of the dense grid.
* free samples sit in front of the hit and never in an occupied cell.
* depth 0 is no label; far-plane (sky) depth is a free ray with no cell.
* all eight cameras contribute (rear sees ground behind, pillars the sides).
* traffic rows: no size -> relabel observed cells only; size -> fill the box.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.perception.occ_geometry import DEFAULT_GRID, backproject, project, rig_cams  # noqa: E402
from python.train.occ_labels import (  # noqa: E402
    OCC_DRIVABLE, OCC_OTHER, OCC_VEHICLE, build_occ_labels, class_lut, sparse_ratio, vehicles_in_ego, world_to_ego,
)
from synthetic_occ import BOX_HI, BOX_LO, SEM_CLASSES, render  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
CAMS = {c.cam_id: c.scaled(160) for c in rig_cams(CFG)}
G = DEFAULT_GRID


def _planes(ids, box=True):
    out = {}
    for cid in ids:
        d, s, _ = render(CAMS[cid], box=box)
        out[cid] = (d, s)
    return out


def check_roundtrip() -> None:
    cam = CAMS["pillarL"]
    pts = np.array([[-6.0, 3.0, 0.5], [-10.0, 0.0, 1.0], [-4.0, 1.0, 2.0]])
    uv, z, ok = project(cam, pts)
    assert ok.all(), ok
    back = backproject(cam, z, uv[:, 0] - 0.5, uv[:, 1] - 0.5)
    assert np.allclose(back, pts, atol=1e-6), (back, pts)
    print("geometry: project -> backproject is the identity OK")


def check_ground_voxels() -> None:
    lab = build_occ_labels(_planes(["main"], box=False), CAMS, sem_classes=SEM_CLASSES, stride=1)
    assert lab.keys.size > 100, lab.keys.size
    iz = G.cell_of_key(lab.keys)[:, 2]
    assert set(iz.tolist()) == {5}, sorted(set(iz.tolist()))
    assert (lab.cls == OCC_DRIVABLE).all(), np.bincount(lab.cls)
    assert np.all(np.diff(lab.keys) > 0), "keys must be sorted unique"
    # every ground cell centre really is in front of main within the grid
    ctr = G.centers_of_keys(lab.keys)
    assert np.all(ctr[:, 1] > 1.0), ctr[:, 1].min()
    # the free points are above the ground, in front of the hit
    assert lab.free.shape[0] > 500, lab.free.shape
    assert np.all(lab.free[:, 2] > 0.2 + 1e-3), lab.free[:, 2].min()
    assert not np.isin(G.keys_of(lab.free), lab.keys).any()
    print(f"ground: {lab.keys.size} cells all in z cell 5, drivable; {lab.free.shape[0]} free points above it OK")


def check_box_and_sparse() -> None:
    lab = build_occ_labels(_planes(["main", "wide", "narrow"]), CAMS, sem_classes=SEM_CLASSES, stride=1)
    ctr = G.centers_of_keys(lab.keys)
    inbox = np.all((ctr >= np.array(BOX_LO) - 0.4) & (ctr <= np.array(BOX_HI) + 0.4), axis=1)
    veh = lab.cls == OCC_VEHICLE
    assert veh.sum() >= 20, veh.sum()
    assert inbox[veh].all(), ctr[veh & ~inbox][:5]
    # the face of the box toward the cameras is labelled (y = 10.0 row)
    face = veh & (np.abs(ctr[:, 1] - 10.2) < 0.21)
    assert face.sum() >= 10, face.sum()
    # nothing labelled occupied behind the box at box height (occluded)
    hidden = (ctr[:, 1] > BOX_HI[1] + 0.5) & (np.abs(ctr[:, 0]) < 0.6) & (ctr[:, 2] > 0.6) & (ctr[:, 2] < 1.6)
    assert not hidden.any(), ctr[hidden][:5]
    r = sparse_ratio(lab)
    assert 0 < r < 0.05, r
    # free points never reach into the box
    fin = np.all((lab.free > np.array(BOX_LO) + 0.05) & (lab.free < np.array(BOX_HI) - 0.05), axis=1)
    assert not fin.any(), lab.free[fin][:3]
    print(f"box: {int(veh.sum())} vehicle cells inside the box, none behind it; stored/dense = {r:.4f} OK")


def check_invalid_depth() -> None:
    d, s, _ = render(CAMS["main"], box=False)
    zero = np.zeros_like(d)
    lab = build_occ_labels({"main": (zero, s)}, CAMS, sem_classes=SEM_CLASSES)
    assert lab.keys.size == 0 and lab.free.shape[0] == 0, (lab.keys.size, lab.free.shape)
    sky = np.full_like(d, int(CAMS["main"].far_m * 100))
    lab = build_occ_labels({"main": (sky, s)}, CAMS, sem_classes=SEM_CLASSES)
    assert lab.keys.size == 0, lab.keys.size
    assert lab.free.shape[0] > 100, lab.free.shape
    print("depth 0 -> no label; far-plane depth -> free ray, no cell OK")


def check_all_eight() -> None:
    planes = _planes(list(CAMS))
    lab = build_occ_labels(planes, CAMS, sem_classes=SEM_CLASSES)
    assert set(lab.cams) == set(CAMS) and len(lab.cams) == 8, lab.cams
    ctr = G.centers_of_keys(lab.keys)
    assert (ctr[:, 1] < -3.0).sum() > 20, "rear camera adds ground behind the car"
    assert (ctr[:, 0] < -3.0).sum() > 20 and (ctr[:, 0] > 3.0).sum() > 20, "side cameras add ground left and right"
    only_main = build_occ_labels({"main": planes["main"]}, CAMS, sem_classes=SEM_CLASSES)
    assert lab.keys.size > 2 * only_main.keys.size, (lab.keys.size, only_main.keys.size)
    try:
        build_occ_labels({"bogus": planes["main"]}, CAMS)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown camera must be refused")
    print(f"8 cams: {lab.keys.size} cells (main alone {only_main.keys.size}); behind and both sides labelled OK")


def check_vehicle_rows() -> None:
    ego = {"pos": [100.0, 200.0, 5.0], "dir": [-1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}
    c = world_to_ego(np.array([[90.0, 200.0, 5.0], [100.0, 210.0, 5.0]]), ego["pos"], ego["dir"], ego["up"])
    assert np.allclose(c[0], [0.0, 10.0, 0.0]), c[0]
    assert np.allclose(c[1], [10.0, 0.0, 0.0]), c[1]  # facing west, north is on the right (+X)
    rows = vehicles_in_ego([{"pos": [88.0, 200.0, 5.2], "dir": [-1.0, 0.0, 0.0]}, {"pos": None}], ego)
    assert len(rows) == 1 and "size" not in rows[0] and abs(rows[0]["yaw"]) < 1e-9, rows
    # unsized row on the box centre: relabels observed cells only
    planes = _planes(["main"])
    no_sem = {k: (v[0], None) for k, v in planes.items()}
    base = build_occ_labels(no_sem, CAMS)
    assert (base.cls == OCC_OTHER).all()
    near = build_occ_labels(no_sem, CAMS, vehicles=[{"center": np.array([0.0, 12.2, 0.2])}])
    assert np.array_equal(near.keys, base.keys), "an unsized row must not add cells"
    assert (near.cls == OCC_VEHICLE).sum() >= 10
    # sized row away from anything observed: fills its box
    sized = build_occ_labels(no_sem, CAMS, vehicles=[{"center": np.array([-8.0, 30.0, 0.0]), "size": (4.0, 2.0, 1.6), "yaw": 0.0}])
    extra = np.setdiff1d(sized.keys, base.keys)
    assert extra.size >= 5 * 10 * 4 * 0.8, extra.size
    ectr = G.centers_of_keys(extra)
    assert np.all(np.abs(ectr[:, 0] + 8.0) <= 1.0 + 1e-6) and np.all(np.abs(ectr[:, 1] - 30.0) <= 2.0 + 1e-6), ectr[:3]
    print("vehicle rows: world->ego OK; unsized relabels only; sized fills its box OK")


def check_class_lut() -> None:
    lut = class_lut({"classes": {"0": "unknown", "1": "ASPHALT", "2": "CAR", "3": "STREET", "4": "TRUCK", "5": "car"}})
    assert lut[1] == OCC_DRIVABLE and lut[2] == OCC_VEHICLE and lut[4] == OCC_VEHICLE
    assert lut[3] == OCC_OTHER and lut[5] == OCC_OTHER and lut[0] == OCC_OTHER
    print("class allowlists: exact names only OK")


def main() -> int:
    check_roundtrip()
    check_ground_voxels()
    check_box_and_sparse()
    check_invalid_depth()
    check_all_eight()
    check_vehicle_rows()
    check_class_lut()
    print("test_occ_labels: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
