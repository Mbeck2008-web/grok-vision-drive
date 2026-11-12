#!/usr/bin/env python3
"""Recorder road-edge heights for labels v4.

The map file keeps left/right world z from the DecalRoad edges. A frame
points at that file. Old dumps with no edge height still load as None.
The polyline is not copied onto every tick (state.jsonl has a 3.5 GiB cap).
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.control_log import TRAFFIC_CHUNK, parse_traffic, road_tag  # noqa: E402
from python.collect.road_dump import (  # noqa: E402
    load_road_dump,
    network_from_polys,
    persist_recording_roads,
    road_gt_for_row,
    write_road_dump,
)
from python.collect.road_loader import reset_shared  # noqa: E402
from python.debug.gt_geom import RoadPoly  # noqa: E402


class _Hit:
    def __init__(self) -> None:
        self.road_id = "kept"
        self.offset_m = 0.25
        self.heading_err_rad = -0.04
        self.lane_width_m = 3.5


def _spec() -> dict:
    edges = []
    for k in range(4):
        y = 4.0 * k
        edges.append(
            {
                "middle": [0.0, y, 6.0],
                "left": [-1.75, y, 5.0],
                "right": [1.75, y, 7.0],
            }
        )
    return {"kept": {"drivability": 1, "edges": edges}}


class _Bng:
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.calls = 0

    def get_road_network(self, include_edges: bool = True, drivable_only: bool = True):
        self.calls += 1
        assert include_edges is True and drivable_only is True
        return self.spec


def check_recording_keeps_edge_z_and_old_dumps_stay_none() -> None:
    reset_shared()
    spec = _spec()
    bng = _Bng(spec)
    # A poly without heights must not replace the game payload.
    bare = RoadPoly("other", [(0.0, 0.0), (0.0, 12.0)], 4.0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = persist_recording_roads(root, "west_coast_usa", bng, [bare])
        assert path is not None and path.name == "west_coast_usa.json.gz"
        assert bng.calls == 1
        rows = [
            {
                "i": index,
                "map": "west_coast_usa",
                "road": {
                    "road_id": "kept",
                    "edge_z_file": "roads/west_coast_usa.json.gz",
                },
            }
            for index in range(3)
        ]
        for row in rows:
            got = road_gt_for_row(root, row)
            assert got is not None
            assert len(got.left_z) == len(got.xy) == len(got.right_z)
            assert got.left_z == [5.0, 5.0, 5.0, 5.0]
            assert got.right_z == [7.0, 7.0, 7.0, 7.0]
            assert got.z == [6.0, 6.0, 6.0, 6.0]

        old = RoadPoly("old", [(0.0, 0.0), (0.0, 10.0)], 7.0)
        old_path = persist_recording_roads(root, "legacy", None, [old])
        assert old_path is not None
        loaded = road_gt_for_row(root, {"map": "legacy", "road": {"road_id": "old"}})
        assert loaded is not None and loaded.left_z is None and loaded.right_z is None

        # A hand-written dump whose points are xy only is the same None.
        xy_only = {
            "bare": {
                "edges": [
                    {"middle": [0.0, 0.0], "left": [-1.0, 0.0], "right": [1.0, 0.0]},
                    {"middle": [0.0, 8.0], "left": [-1.0, 8.0], "right": [1.0, 8.0]},
                ]
            }
        }
        raw = root / "roads" / "xy.json.gz"
        write_road_dump(raw, level="xy", spawn=None, spawn_source="", road_network=xy_only)
        polys, _payload = load_road_dump(raw)
        assert polys and polys[0].left_z is None and polys[0].right_z is None
    print("  recording keeps left_z/right_z per edge point; old dumps stay None")


def check_frame_does_not_copy_the_polyline() -> None:
    poly = RoadPoly(
        "kept",
        [(0.0, float(i)) for i in range(40)],
        7.0,
        z=[1.0] * 40,
        left_z=[1.1] * 40,
        right_z=[0.9] * 40,
    )
    tag = road_tag(_Hit(), 2, poly)
    assert tag is not None
    assert tag["edge_z"] is True
    assert tag["road_width_m"] == 7.0
    assert tag["road_offset_m"] == 0.25
    assert tag["heading_err_rad"] == -0.04
    assert "left_z" not in tag and "right_z" not in tag
    blob = json.dumps(tag)
    assert len(blob) < 400, len(blob)
    built = network_from_polys([poly])
    assert built["kept"]["edges"][3]["left"][2] == 1.1
    assert built["kept"]["edges"][3]["right"][2] == 0.9
    print(f"  road tag is {len(blob)} bytes and does not repeat the edge polyline")


def check_traffic_model_is_recorded_without_a_class_guess() -> None:
    assert "jbeam" in TRAFFIC_CHUNK
    for banned in ("truck", "bus", "pedestrian", "bike"):
        assert banned not in TRAFFIC_CHUNK, banned
    car = {
        "name": "parked",
        "model": "etk800",
        "pos": [1, 2, 3],
        "vel": [0, 0, 0],
        "dir": [1, 0, 0],
    }
    parsed = parse_traffic(json.dumps([car]))
    assert parsed is not None and parsed[0]["model"] == "etk800"
    nameless = dict(car)
    del nameless["model"]
    plain = parse_traffic(json.dumps([nameless]))
    assert plain is not None and "model" not in plain[0]
    empty = parse_traffic(json.dumps([{**car, "model": ""}]))
    assert empty is not None and "model" not in empty[0]
    print("  traffic model is the jbeam string; a missing model is omitted")


def check_disk_delta_against_the_two_hour_baseline() -> None:
    """Per-frame JSON growth, plus one gzip per map. Images stay the 96 GB / 2 h baseline."""
    old = {"road_id": "kept", "lanes_here": 2, "lane_width_m": 3.5}
    new = {
        **old,
        "road_width_m": 7.0,
        "road_offset_m": 0.25,
        "heading_err_rad": -0.04,
        "edge_z": True,
        "edge_z_file": "roads/west_coast_usa.json.gz",
    }
    delta = len(json.dumps(new, sort_keys=True)) - len(json.dumps(old, sort_keys=True))
    assert 0 < delta < 400, delta
    frames = 8 * 3600 * 10  # main is 10 Hz, and a row is written on that tick
    json_bytes = delta * frames
    roads = 0
    for path in sorted((ROOT / "tests" / "fixtures" / "roads").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        blob = json.dumps(payload.get("road_network")).encode("utf-8")
        packed = gzip.compress(blob)
        roads += len(packed)
    baseline_gb = 96.0 / 2.0 * 8.0
    extra_gb = (json_bytes + roads) / 1e9
    print(
        f"  disk per_frame={delta} B json_8h={json_bytes / 1e6:.1f} MB "
        f"roads_once={roads / 1e6:.1f} MB extra={extra_gb:.4f} GB "
        f"8h={baseline_gb + extra_gb:.2f} GB (baseline {baseline_gb:.0f} GB)"
    )


def main() -> None:
    check_recording_keeps_edge_z_and_old_dumps_stay_none()
    check_frame_does_not_copy_the_polyline()
    check_traffic_model_is_recorded_without_a_class_guess()
    check_disk_delta_against_the_two_hour_baseline()
    print("test_labels_v4_edge_z: OK")


if __name__ == "__main__":
    main()
