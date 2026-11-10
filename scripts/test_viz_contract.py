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
    nested = _model(
        "edge",
        0.91,
        points=[{"side": "left", "x": -1.5, "y": 5.0}],
        gt_poly={"src": "gt", "points": [{"x": 0.0, "y": 5.0}]},
        path_world=[{"x": 1.0, "y": 10.0}],
    )
    cleaned = renderable_state({"edge": nested})["edge"]
    assert cleaned["points"][0]["x"] == -1.5
    assert "gt_poly" not in cleaned
    assert "path_world" not in cleaned
    print("viz contract: foreign, untagged, and low-confidence fields draw nothing OK")


def _sources(node) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key in ("src", "label_src"):
            if isinstance(node.get(key), str):
                found.append(node[key])
        for value in node.values():
            found.extend(_sources(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_sources(value))
    return found


def check_as_state_live_output() -> None:
    """The dict as_state returns. No second renderable_state call."""
    import numpy as np

    from python.debug.lane_targets import LANE_GT_DIM, LOOKAHEADS_M
    from python.train.scene_ckpt_runtime import SceneCkptPred
    from python.viz.scene_ribbon import scene_ckpt_ribbon_polyline

    values = np.full((LANE_GT_DIM,), np.nan, dtype=np.float32)
    for i, x in enumerate((0.0, 0.4, 1.2, 2.0)):
        if i >= len(LOOKAHEADS_M):
            break
        values[i * 3] = np.float32(x)
    ckpt = "m.ckpt_ep1.npz"
    pred = SceneCkptPred(lane_gt=values, lane_x=0.4, confidence=0.9, ckpt=ckpt)
    pred.detect = {
        "edge": {
            "src": "model",
            "head": "edge",
            "ckpt": ckpt,
            "confidence": 0.91,
            "points": [{"side": "left", "x": -1.5, "y": 5.0}],
            "gt_poly": {"src": "gt", "points": [{"x": 9.0, "y": 5.0}]},
            "path_world": [{"x": 1.0, "y": 10.0}],
        },
        "spoof": {
            "src": "model",
            "label_src": "beamng_gt",
            "head": "edge",
            "ckpt": ckpt,
            "confidence": 0.95,
            "points": [{"side": "left", "x": 4.0, "y": 5.0}],
        },
        "dim": {
            "src": "model",
            "head": "edge",
            "ckpt": ckpt,
            "confidence": 0.20,
            "points": [{"side": "left", "x": 3.0, "y": 5.0}],
        },
        "map_road": {
            "src": "map",
            "head": "edge",
            "ckpt": ckpt,
            "confidence": 0.99,
            "points": [{"x": 0.0, "y": 5.0}],
        },
    }
    state = pred.as_state()
    assert state["confidence"] == 0.9
    assert state["shadow"] is True and state["source"] == "scene_ckpt"
    got_xy = [(c["x"], c["y"]) for c in state["centers"]]
    want_xy = [(0.0, 5.0), (0.4, 10.0), (1.2, 20.0), (2.0, 30.0)]
    assert len(got_xy) == len(want_xy)
    for got, want in zip(got_xy, want_xy):
        assert abs(got[0] - want[0]) < 1e-5 and abs(got[1] - want[1]) < 1e-5, (got, want)
    assert all(c["src"] == "model" and c["head"] == "lane_gt" and c["ckpt"] == ckpt for c in state["centers"])
    assert state["lane_x"]["lane_x"] == 0.4
    assert state["lane_x"]["src"] == "model"
    assert "gt_poly" not in state["detect"]["edge"]
    assert "path_world" not in state["detect"]["edge"]
    assert "spoof" not in state["detect"]
    assert "dim" not in state["detect"]
    assert "map_road" not in state["detect"]
    assert "map_road" not in state
    assert "beamng_gt" not in _sources(state)
    assert "map" not in _sources(state)
    assert "gt" not in _sources(state)
    poly = scene_ckpt_ribbon_polyline(state["centers"], state["confidence"])
    assert len(poly) >= 2

    low = SceneCkptPred(lane_gt=values, lane_x=0.4, confidence=0.20, ckpt=ckpt)
    low.detect = {
        "edge": {
            "src": "model",
            "head": "edge",
            "ckpt": ckpt,
            "confidence": 0.20,
            "points": [{"side": "right", "x": 1.0, "y": 5.0}],
        }
    }
    low_state = low.as_state()
    assert low_state["caption"] == "LOW"
    assert low_state["confidence"] == 0.20
    assert not low_state.get("centers")
    assert low_state.get("lane_x") is None
    assert "lane" not in low_state
    assert not low_state.get("detect")
    print("as_state live: gt and map dropped, low conf emits no centers or lane_x, ribbon still fed OK")


def main() -> None:
    check_rejects_foreign_and_low_conf()
    check_as_state_live_output()
    print("test_viz_contract: all OK")


if __name__ == "__main__":
    main()
