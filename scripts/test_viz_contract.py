#!/usr/bin/env python3
"""The visualizer may draw only a live model's own outputs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _model(head: str, conf: float, **extra):
    body = {
        "src": "model",
        "head": head,
        "ckpt": "e2e_scene.ckpt_ep3.npz",
        "confidence": conf,
    }
    body.update(extra)
    return body


def check_rejects_foreign_and_low_conf() -> None:
    from python.train.viz_contract import (
        VizContractError,
        is_model_drawable,
        package_detect,
        package_lane,
        renderable_state,
        require_model_drawable,
    )

    good = _model("edge", 0.91, points=[{"side": "left", "x": -1.5, "y": 5.0}])
    assert require_model_drawable(good)["head"] == "edge"
    foreign = [
        {"src": "beamng_gt", "head": "edge", "ckpt": "a.npz", "confidence": 0.99, "points": [1]},
        {"src": "map", "head": "edge", "ckpt": "a.npz", "confidence": 0.99, "points": [1]},
        {"src": "ai", "head": "lane_gt", "ckpt": "a.npz", "confidence": 0.99, "lane_x": 0.2},
        {"src": "game", "head": "obj", "ckpt": "a.npz", "confidence": 0.99},
        {"src": "debug", "head": "edge", "ckpt": "a.npz", "confidence": 0.99},
        {"src": "sensor", "head": "brakelight", "ckpt": "a.npz", "confidence": 0.99},
        {"src": "road", "head": "edge", "ckpt": "a.npz", "confidence": 0.99},
        _model("edge", 0.91, label_src="beamng_gt", points=[1]),
        _model("edge", 0.2, points=[{"x": 1.0, "y": 5.0}]),
        {"src": "model", "ckpt": "a.npz", "confidence": 0.9, "points": [1]},
        {"src": "model", "head": "edge", "confidence": 0.9, "points": [1]},
    ]
    for field in foreign:
        assert not is_model_drawable(field), field
        try:
            require_model_drawable(field)
        except VizContractError:
            pass
        else:
            raise AssertionError(f"accepted a non-model field: {field}")

    state = {
        "lane_x": 0.4,
        "lane": good and _model("lane_gt", 0.88, lane_x=0.4),
        "detect": {
            "edge": good,
            "road": {"src": "map", "head": "edge", "ckpt": "a.npz", "confidence": 1.0, "points": [1]},
            "path": {"src": "ai_path", "points": [{"x": 0.0, "y": 10.0}]},
            "objects": [
                _model("obj", 0.8, cls="cone", x=1.0, y=6.0),
                _model("obj", 0.1, cls="vehicle", x=0.0, y=12.0),
            ],
        },
    }
    kept = renderable_state(state)
    assert "lane_x" not in kept
    assert set(kept["detect"]) == {"edge", "objects"}
    assert len(kept["detect"]["objects"]) == 1
    assert kept["detect"]["objects"][0]["cls"] == "cone"
    assert kept["lane"]["head"] == "lane_gt"

    low = package_detect(ckpt="run.npz", edge=[1.0] * 8, edge_conf=-8.0)
    assert low["edge"]["points"] == []
    assert renderable_state({"detect": low}) == {}
    high = package_detect(ckpt="run.npz", edge=[-1.5, 1.5] * 4, edge_conf=8.0)
    assert high["edge"]["points"]
    assert high["edge"]["src"] == "model" and high["edge"]["ckpt"] == "run.npz"
    assert renderable_state({"detect": high})["detect"]["edge"]["head"] == "edge"
    lane = package_lane(ckpt="run.npz", lane_x=0.3, confidence=0.1)
    assert lane["lane_x"] is None
    assert renderable_state({"lane": lane}) == {}
    print("viz contract: foreign, untagged, and low-confidence fields draw nothing OK")


def main() -> None:
    check_rejects_foreign_and_low_conf()
    print("test_viz_contract: all OK")


if __name__ == "__main__":
    main()
