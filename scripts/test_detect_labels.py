#!/usr/bin/env python3
"""Training-label exporter. Ground truth stays out of the drawable contract."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_parse_heads_stay_opt_in() -> None:
    from python.train.cam_strip import DETECT_HEADS, HEAD_NAMES, parse_heads

    assert parse_heads("") == HEAD_NAMES
    assert parse_heads(None) == HEAD_NAMES
    for name in DETECT_HEADS:
        assert name not in parse_heads("")
    assert parse_heads("lane_gt,edge,obj,brakelight") == ("lane_gt", "edge", "obj", "brakelight")
    assert parse_heads("object") == ("objects",)
    try:
        parse_heads("nope")
    except ValueError as exc:
        assert "unknown head" in str(exc)
    else:
        raise AssertionError("unknown head was accepted")
    print("parse_heads: edge/obj/brakelight opt-in OK")


def check_edge_is_not_paint() -> None:
    from python.train.detect_targets import label_record

    row = {
        "i": 3,
        "lane_gt": [1.0, 2.0, 3.0],
        "road": {"lane_width_m": 3.5, "lanes_here": 2},
    }
    before = copy.deepcopy(row)
    rec = label_record(row)
    assert row == before
    assert "lane_gt" not in rec
    assert rec["label_src"] == "beamng_gt"
    assert rec["edge"]["method"] == "absent"
    assert sum(rec["edge"]["valid"]) == 0.0
    wide = label_record({"road": {"road_width_m": 8.0, "road_offset_m": 0.0, "heading_err_rad": 0.0}})
    assert wide["edge"]["method"] == "road_width"
    assert abs(wide["edge"]["values"][0] + 4.0) < 1e-5
    assert abs(wide["edge"]["values"][1] - 4.0) < 1e-5
    # A short left polyline does not invent the right side or the far ranges.
    partial = label_record(
        {
            "edge_left": [[-3.0, 0.0], [-3.0, 12.0]],
            "road": {"road_width_m": 8.0, "road_offset_m": 0.0},
        }
    )
    assert partial["edge"]["method"] == "polyline"
    assert abs(partial["edge"]["values"][0] + 3.0) < 1e-5
    assert partial["edge"]["valid"][1] == 0.0
    assert partial["edge"]["valid"][4] == 0.0
    print("labels: edge is not paint, one side does not invent the other OK")


def check_boxes_and_brake() -> None:
    from python.train.detect_targets import brake_label, object_label

    ego = {"pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}
    row = {
        "ego": ego,
        "veh": {"brakelights": 1.0},
        "traffic": [
            {"pos": [2.0, 10.0, 0.0], "dir": [0.0, 1.0, 0.0]},
            {"x": 0.0, "y": -4.0, "brakelights": 1.0},
        ],
    }
    slots = object_label(row)["slots"]
    filled = [s for s in slots if s["valid"][0] > 0.5]
    assert len(filled) == 2
    lead = next(s for s in filled if abs(s["box"][1] - 10.0) < 1e-5)
    assert abs(lead["box"][0] - 2.0) < 1e-5
    assert abs(lead["box"][3]) < 1e-5
    assert lead["valid"][4] == 0.0
    assert brake_label(row)["valid"] == 0.0

    lamps = {
        "ego": ego,
        "traffic": [
            {"x": 0.0, "y": 30.0, "brakelights": 0.0},
            {"x": 0.2, "y": 12.0, "brakelights": 1.0},
        ],
    }
    brake = brake_label(lamps)
    assert brake["valid"] == 1.0 and brake["on"] == 1.0

    cones = object_label(
        {
            "static_objects": [
                {
                    "cls": "cone",
                    "x": 1.0,
                    "y": 4.0,
                    "z": 0.2,
                    "yaw": 0.1,
                    "length_m": 0.3,
                    "width_m": 0.3,
                    "height_m": 0.7,
                }
            ]
        }
    )
    cone = cones["slots"][0]
    assert cone["cls"] == "cone"
    assert cone["valid"][4] == 1.0 and abs(cone["box"][4] - 0.3) < 1e-6
    print("labels: pose without invented size, lead brake only, cone box OK")


def check_exporter_and_contract() -> None:
    from python.train.detect_targets import clear_detect_cache, export_root, training_arrays
    from python.train.viz_contract import VizContractError, renderable_state, require_model_drawable

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        row = {"i": 1, "t": 0.1, "road": {"lane_width_m": 3.5}}
        (folder / "state.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert export_root(folder) == 1
        saved = json.loads((folder / "detect.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert saved["label_src"] == "beamng_gt"
        assert renderable_state(saved) == {}
        try:
            require_model_drawable(saved["edge"])
        except VizContractError:
            pass
        else:
            raise AssertionError("a training edge was treated as drawable")
        clear_detect_cache()
        # Sidecar wins over the raw row.
        saved["edge"]["values"] = [-1.0] * 8
        saved["edge"]["valid"] = [1.0] * 8
        (folder / "detect.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")
        clear_detect_cache()
        arrays = training_arrays({"i": 1, "road": {"lane_width_m": 3.5}}, root=folder)
        assert abs(float(arrays["edge"][0]) + 1.0) < 1e-6
        assert "lane_gt" not in arrays
    print("exporter: detect.jsonl is training-only and not drawable OK")


def main() -> None:
    check_parse_heads_stay_opt_in()
    check_edge_is_not_paint()
    check_boxes_and_brake()
    check_exporter_and_contract()
    print("test_detect_labels: all OK")


if __name__ == "__main__":
    main()
