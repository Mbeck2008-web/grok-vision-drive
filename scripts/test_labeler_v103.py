#!/usr/bin/env python3
"""v103 painted-lane labels: lane-line materials only, body-centreline origin.

Fixtures (no data paths):
* ``tests/fixtures/painted/west_coast_usa_v103_rows.json.gz``: west_coast_usa
  line decals within 60 m of 100 rec5c frame poses (state pos = ref node):
  ``straight60`` (straight rows: |yaw rate| < 0.02 rad/s, 10-30 m centre bend
  < 0.15 m) and ``near_parking`` (rows with a parking decal within 12 m).
* ``tests/fixtures/vehicles/etk800_refnodes.jbeam``: the etk800 refNodes
  block and the nodes it names, verbatim from the body jbeam.

The geometry check measures the lane centre with its own code (linear decal
segments, crossings pooled over 0-12.2 m, nearest lines either side of the
origin), not with ``painted_lanes``. Plain asserts so
``run_offline_suite.py`` runs it; functions are named ``test_*`` for pytest.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import python.debug.painted_lanes as PL  # noqa: E402
from python.debug.lane_targets import ORIGIN_CENTRELINE, tech_from_painted  # noqa: E402
from python.debug.painted_lanes import LANE_LINE_MATERIALS, PaintedLanes, is_paint  # noqa: E402
from python.debug.vehicle_origin import (  # noqa: E402
    centreline_origin_right_m,
    parse_ref_offset,
    ref_offset_from_jbeam_file,
    ref_right_of_centre_m,
)

ROWS = ROOT / "tests" / "fixtures" / "painted" / "west_coast_usa_v103_rows.json.gz"
JBEAM = ROOT / "tests" / "fixtures" / "vehicles" / "etk800_refnodes.jbeam"
GEOMETRY_TOL_M = 0.10


def _fixture() -> dict:
    with gzip.open(ROWS, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _old_is_paint(material: object) -> bool:
    """The v102 rule: every ``line*`` / ``m_line*`` material."""
    m = str(material or "").lower()
    return m.startswith("line") or m.startswith("m_line")


def _lanes(decals: list[dict], rule=None) -> PaintedLanes:
    if rule is None:
        return PaintedLanes(decals)
    keep = PL.is_paint
    try:
        PL.is_paint = rule
        return PaintedLanes(decals)
    finally:
        PL.is_paint = keep


# ---------------------------------------------------------------- materials


def test_is_paint_allowlist() -> None:
    for m in ("line_parking1", "line_parking2", "LINE_PARKING1"):
        assert not is_paint(m), m
    for m in ("line_yellowblack", "line_white", "line_yellow", "line_dashed_short", "line_dashed_short_yellow",
              "line_dashed_long", "line_reflectors", "m_line_white", "m_line_yellow", "Line_YellowBlack"):
        assert is_paint(m), m
    for m in ("", None, "road_asphalt", "ChaulkLine", "checkered_line", "road_tramline"):
        assert not is_paint(m), m
    # Every level line material is either in the allowlist or one of the two parking decals.
    level_lines = {"line_white", "line_yellow", "line_dashed_short", "line_parking1", "line_yellowblack",
                   "line_dashed_short_yellow", "line_parking2", "line_dashed_long", "line_reflectors",
                   "m_line_yellow", "m_line_white"}
    assert level_lines - LANE_LINE_MATERIALS == {"line_parking1", "line_parking2"}


def test_paint_rule_alone_leaves_rec5c_labels_unchanged() -> None:
    data = _fixture()
    decals = data["decals"]
    assert any(d["material"] == "line_parking1" for d in decals) and any(d["material"] == "line_parking2" for d in decals)
    old, new = _lanes(decals, _old_is_paint), _lanes(decals)
    assert len(old.segs) > len(new.segs), "the old rule must see the parking decals"
    poses = data["poses"]["near_parking"] + data["poses"]["straight60"]
    assert len(data["poses"]["near_parking"]) >= 30
    labelled = 0
    for p in poses:
        a, b = old.lane_at(p["pos"], p["dir"]), new.lane_at(p["pos"], p["dir"])
        assert a == b, (p["i"], a, b)
        labelled += any(g is not None for g in a)
    assert labelled >= 60, labelled


# ------------------------------------------------------------------- origin


def test_vehicle_origin_from_jbeam() -> None:
    got = ref_offset_from_jbeam_file(JBEAM)
    assert (got.ref_node, got.left_node) == ("f2r", "f2l")
    assert got.ref_right_of_centre_m > 0.0, "jbeam +x is left; f2r (x<0) is right of the centreline"
    assert got.extremes_ref_right_of_centre_m == got.ref_right_of_centre_m
    off, src = ref_right_of_centre_m("etk800", jbeam=JBEAM)
    assert off == got.ref_right_of_centre_m and "f2r" in src
    assert centreline_origin_right_m("etk800", jbeam=JBEAM) == -off
    # A synthetic car: ref at jbeam x=0.2, its left: partner at 1.2 -> centre 0.7 (left), ref 0.5 m right of it.
    text = '{"c":{"refNodes":[\n["ref:","back:","left:","up:"],\n["a","b","c","d"]\n],"nodes":[\n["a", 0.2, 0, 0],\n["c", 1.2, 0, 0],\n["z", -0.7, 0, 0],\n]}}'
    syn = parse_ref_offset(text)
    assert syn is not None and abs(syn.ref_right_of_centre_m - 0.5) < 1e-9, syn
    try:
        ref_right_of_centre_m("etk800")
        raise AssertionError("no jbeam must raise, never default")
    except ValueError:
        pass


def test_lane_at_origin_offset() -> None:
    o = centreline_origin_right_m("etk800", jbeam=JBEAM)
    paint = PaintedLanes([{"material": "line_white", "nodes": [[x, y, 0.0, 0.2] for y in range(-60, 121, 10)]} for x in (-1.75, 1.75, 5.25)])
    north = (0.0, 1.0, 0.0)
    ref = paint.lane_at((0.0, 0.0, 0.0), north)
    cl = paint.lane_at((0.0, 0.0, 0.0), north, origin_right_m=o)
    for a, b in zip(ref, cl):
        assert abs(b[0] - (a[0] - o)) < 1e-9 and abs(b[1] - (a[1] - o)) < 1e-9
    # Heading east, world +x forward: the car's right is world -y.
    east = (1.0, 0.0, 0.0)
    paint_e = PaintedLanes([{"material": "line_white", "nodes": [[x, y, 0.0, 0.2] for x in range(-60, 121, 10)]} for y in (1.75, -1.75)])
    a, b = paint_e.lane_at((0.0, 0.0, 0.0), east)[0], paint_e.lane_at((0.0, 0.0, 0.0), east, origin_right_m=o)[0]
    assert abs(a[0] + 1.75) < 1e-9 and abs(b[0] - (-1.75 - o)) < 1e-9
    # A ref node 0.1 m right of a line, with the centreline left of it: the centreline is in the left lane.
    near = paint.lane_at((1.85, 0.0, 0.0), north)[0]
    assert near is not None and abs(near[0] + 0.1) < 1e-9
    near_cl = paint.lane_at((1.85, 0.0, 0.0), north, origin_right_m=o)[0]
    assert near_cl is not None and abs(near_cl[1] - (-0.1 - o)) < 1e-9
    tech = tech_from_painted(cl, origin=ORIGIN_CENTRELINE, ref_right_of_origin_m=-o)
    assert tech["gt_origin"] == ORIGIN_CENTRELINE and tech["gt_ref_right_of_origin_m"] == -o


# ----------------------------------------------------------------- geometry


def _segments(decals: list[dict]) -> np.ndarray:
    out = []
    for d in decals:
        if not is_paint(d["material"]):
            continue
        p = np.asarray([n[:3] for n in d["nodes"]], dtype=float)
        if len(p) >= 2:
            out.append(np.hstack([p[:-1], p[1:]]))
    return np.vstack(out)


def _crossings(seg: np.ndarray, origin: np.ndarray, z: float, f: np.ndarray, y: float) -> list[float]:
    r = np.array([f[1], -f[0]])
    o = origin + y * f
    a, b = seg[:, 0:2] - o, seg[:, 3:5] - o
    ay, by = a @ f, b @ f
    ax, bx = a @ r, b @ r
    d = seg[:, 3:5] - seg[:, 0:2]
    dl = np.hypot(d[:, 0], d[:, 1])
    keep = (ay * by <= 0) & (np.abs(by - ay) > 1e-6) & (np.abs(0.5 * (seg[:, 2] + seg[:, 5]) - z) <= 3.0)
    keep &= np.abs(d @ f) / np.maximum(dl, 1e-9) >= math.cos(math.radians(45.0))
    t = -ay[keep] / (by[keep] - ay[keep])
    return (ax[keep] + t * (bx[keep] - ax[keep])).tolist()


def independent_centre_5m(seg: np.ndarray, pos: list[float], direction: list[float], origin_right_m: float) -> float | None:
    """Lane centre near 5 m from the raw decals: lines pooled over 0-12.2 m (dash gaps), nearest either side of the origin."""
    f = np.asarray(direction[:2], dtype=float)
    f /= np.hypot(*f)
    r = np.array([f[1], -f[0]])
    origin = np.asarray(pos[:2], dtype=float) + origin_right_m * r
    xs = sorted(x for y in np.arange(0.0, 12.2 + 1e-9, 0.5) for x in _crossings(seg, origin, float(pos[2]), f, float(y)))
    if not xs:
        return None
    clusters, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= 0.6:
            cur.append(x)
        else:
            clusters.append(float(np.median(cur)))
            cur = [x]
    clusters.append(float(np.median(cur)))
    left = [x for x in clusters if x <= 0.0]
    right = [x for x in clusters if x > 0.0]
    if not left or not right:
        return None
    return 0.5 * (left[-1] + right[0])


def geometry_mean_abs_diff(label_origin_right_m: float) -> tuple[float, int]:
    """Mean |label centre at 5 m - independent centreline-frame centre| on the straight rows."""
    data = _fixture()
    seg = _segments(data["decals"])
    lanes = PaintedLanes(data["decals"])
    o_cl = centreline_origin_right_m("etk800", jbeam=JBEAM)
    diffs = []
    for p in data["poses"]["straight60"]:
        lab = lanes.lane_at(p["pos"], p["dir"], origin_right_m=label_origin_right_m)[0]
        geo = independent_centre_5m(seg, p["pos"], p["dir"], o_cl)
        if lab is None or geo is None:
            continue
        diffs.append(abs(0.5 * (lab[0] + lab[1]) - geo))
    return float(np.mean(diffs)), len(diffs)


def test_geometry_centreline_frame() -> None:
    mean, n = geometry_mean_abs_diff(centreline_origin_right_m("etk800", jbeam=JBEAM))
    print(f"geometry: centreline labels vs independent centreline geometry, mean |diff| {mean:.3f} m, n={n}")
    assert n >= 50, n
    assert mean <= GEOMETRY_TOL_M, mean


def test_geometry_negative_control() -> None:
    """Revert the origin (ref-node labels): the same geometry check must fail."""
    mean, n = geometry_mean_abs_diff(0.0)
    print(f"negative control: ref-node labels vs centreline geometry, mean |diff| {mean:.3f} m, n={n}")
    assert n >= 50, n
    assert mean > GEOMETRY_TOL_M, f"the geometry check does not see the origin: {mean}"


def _toy_recording(tmp: Path, n_rows: int = 6) -> tuple[Path, Path, Path]:
    """A straight road along +x: line_white at y = +2.0 (left) and y = -1.5 (right); the car drives along y = 0."""
    import zipfile

    rec = tmp / "rec"
    rec.mkdir()
    with open(rec / "state.jsonl", "w", encoding="utf-8") as f:
        for i in range(n_rows):
            t = 0.2 * i
            f.write(json.dumps({"i": i, "t": t, "segment": "toy#0", "map": "toy", "wheel": {}, "pedals": {},
                                "ego": {"pos": [2.0 * i, 0.0, 0.0], "dir": [1.0, 0.0, 0.0], "speed_mps": 10.0, "yaw_rate": 0.0},
                                "cams": {"main": {"t": t, "file": f"raw/main/000/{i:06d}.png"}}}) + "\n")
    level = tmp / "toy_level.zip"
    with zipfile.ZipFile(level, "w") as z:
        lines = []
        for pid, yl in (("left", 2.0), ("right", -1.5)):
            nodes = [[float(x), yl, 0.0, 0.2] for x in range(-20, 121, 10)]
            lines.append(json.dumps({"class": "DecalRoad", "material": "line_white", "nodes": nodes, "persistentId": pid}))
        z.writestr("levels/toy/main/items.level.json", "\n".join(lines) + "\n")
    vehicle = tmp / "etk800.zip"
    with zipfile.ZipFile(vehicle, "w") as z:
        z.writestr("vehicles/etk800/etk800_body_sedan.jbeam", JBEAM.read_text(encoding="utf-8"))
    return rec, level, vehicle


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_relabel_applies_hop_mask() -> None:
    """relabel_v103 drops hop-masked rows at selection (empty tech -> loss_mask False), keeps the tracker
    continuity of the rows after them, and refuses a missing or foreign mask."""
    import os
    import tempfile

    sys.path.insert(0, str(ROOT / "scripts"))
    import relabel_v103 as RV
    from python.data.raw_frames import load_scene_rows
    from python.train.scene_net import loss_mask

    with tempfile.TemporaryDirectory() as td:
        rec, level, vehicle = _toy_recording(Path(td))
        base = [str(rec), "--zip", str(level), "--vehicle-zip", str(vehicle)]
        # No mask file and no --no-hop-mask: refuse, write nothing.
        assert RV.main(base) == 2
        assert not (rec / "gt_v103.jsonl").exists() and not (rec / "gt_v103.meta.json").exists()
        # A mask id that is not a state row: refuse.
        (rec / "bad_mask.jsonl").write_text(json.dumps({"i": 99}) + "\n", encoding="utf-8")
        assert RV.main(base + ["--hop-mask", "bad_mask.jsonl"]) == 2
        assert not (rec / "gt_v103.jsonl").exists()
        # The mask, with its meta header.
        (rec / RV.HOP_MASK_DEFAULT).write_text(json.dumps({"i": 2, "mask": "hop"}) + "\n", encoding="utf-8")
        (rec / "mask_v102_hop.meta.json").write_text(json.dumps({"rule": "hop", "spec_sha256": "0" * 64, "T_m": 0.12}), encoding="utf-8")
        assert RV.main(base) == 0
        assert RV.main(base + ["--no-hop-mask", "--out", "gt_v103_nomask.jsonl"]) == 0
        got = _read_jsonl(rec / "gt_v103.jsonl")
        ref = _read_jsonl(rec / "gt_v103_nomask.jsonl")
        meta = json.loads((rec / "gt_v103.meta.json").read_text(encoding="utf-8"))
        assert meta["rows_hop_masked"] == 1 and meta["hop_mask"]["rows"] == 1, meta
        assert meta["hop_mask"]["meta"]["spec_sha256"] == "0" * 64 and meta["hop_mask"]["meta"]["T_m"] == 0.12
        assert json.loads((rec / "gt_v103_nomask.meta.json").read_text(encoding="utf-8"))["hop_mask"] is None
        assert len(got) == len(ref) == 6
        for a, b in zip(got, ref):
            if a["i"] == 2:
                assert a["tech"] == {} and a["hop_masked"] == RV.HOP_MASK_DEFAULT, a
                assert b["tech"]["gt_origin"] == ORIGIN_CENTRELINE
            else:
                assert a["tech"] == b["tech"], (a["i"], a["tech"], b["tech"])
                assert "hop_masked" not in a
        # Centreline frame: origin 0.33 m left of the ref node, lines at y +2.0 / -1.5 -> x -1.67 / +1.83 at y = 5 m.
        tech0 = got[0]["tech"]
        assert tech0["lane_gt_valid"][1] == 1.0 and tech0["lane_gt_valid"][2] == 1.0, tech0
        assert abs(tech0["lane_gt"][1] - (-1.67)) < 0.02 and abs(tech0["lane_gt"][2] - 1.83) < 0.02, tech0["lane_gt"][:3]
        # Selection: the trainer's join + loss_mask leave the masked row out, keep the others.
        old = os.environ.get("GVD_GT_FILE")
        os.environ["GVD_GT_FILE"] = "gt_v103.jsonl"
        try:
            joined = {int(r["i"]): r for r in load_scene_rows(rec)}
        finally:
            if old is None:
                os.environ.pop("GVD_GT_FILE", None)
            else:
                os.environ["GVD_GT_FILE"] = old
        assert not loss_mask(joined[2])
        assert all(loss_mask(joined[i]) for i in (0, 1, 3, 4, 5))
    print("relabel hop mask: masked row out of loss_mask, other rows identical to the unmasked relabel, refusals ok")


def main() -> None:
    test_is_paint_allowlist()
    test_paint_rule_alone_leaves_rec5c_labels_unchanged()
    test_vehicle_origin_from_jbeam()
    test_lane_at_origin_offset()
    test_geometry_centreline_frame()
    test_geometry_negative_control()
    test_relabel_applies_hop_mask()
    print("test_labeler_v103: OK")


if __name__ == "__main__":
    main()
