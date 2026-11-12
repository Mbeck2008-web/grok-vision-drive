#!/usr/bin/env python3
"""Traffic rows become ego-frame object labels. No BeamNG and no camera.

Size is not a label: a missing length/width stays 0, and those channels do
not enter the object loss. Brake lights are stored from the traffic row and
stay masked until the IQU45 check (the head flag defaults off).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gvd_test_env import skip_check  # noqa: E402

from python.collect.signal_labels import world_to_ego  # noqa: E402
from python.debug.gt_geom import wrap_rad  # noqa: E402
from python.train.scene_net import (  # noqa: E402
    HEAD_NAMES,
    LOSS_TERMS,
    OBJECT_CLASSES,
    OBJECT_FEAT,
    OUTPUT_FIELDS,
    labels_for_row,
    labels_from_tech,
)


def _row(ego_pos, ego_dir, traffic) -> dict:
    return {
        "ego": {"pos": list(ego_pos), "dir": list(ego_dir)},
        "veh": {"brakelights": 1},
        "traffic": traffic,
        "tech": {
            "tracks": [
                {"cls": "truck", "x": 1.0, "y": 8.0, "yaw": 0.0, "length": 4.5, "width": 1.8}
            ]
        },
    }


def check_heading_wrap() -> None:
    """A dir that crosses ±pi stays near the aligned yaw, not the raw difference."""
    ego_h = math.pi
    traf_h = -math.pi + 0.02
    raw = traf_h - ego_h
    assert raw < -math.pi
    ego_dir = [math.cos(ego_h), math.sin(ego_h), 0.0]
    traf_dir = [math.cos(traf_h), math.sin(traf_h), 0.0]
    # 15 m ahead along ego forward (+x is world -x when heading is pi).
    pos = [-15.0, 0.0, 0.2]
    row = _row(
        [0.0, 0.0, 0.0],
        ego_dir,
        [
            {
                "name": "gvd_traffic_0",
                "pos": pos,
                "vel": [0.0, 5.0, 40.0],
                "dir": traf_dir,
                "brake_lights": 1,
            }
        ],
    )
    labels = labels_for_row(row)
    assert int(labels["object_valid"].sum()) == 1
    assert int(labels["object_class"][0]) == OBJECT_CLASSES.index("car")
    # Horizontal speed only. The 40 m/s vertical component is not the label.
    assert abs(float(labels["objects"][0, 5]) - 5.0) < 1e-4
    xy = world_to_ego((pos[0], pos[1]), row["ego"]["pos"], row["ego"]["dir"])
    assert xy is not None
    got = labels["objects"][0]
    assert math.hypot(float(got[0]) - xy[0], float(got[1]) - xy[1]) < 0.5
    yaw = float(got[2])
    # Aligned-with-ego is pi/2 in this frame (x right, y forward). The 0.02 rad
    # past the branch cut is a small step off that, not raw ≈ -2π.
    assert abs(yaw) <= math.pi + 1e-6
    assert abs(wrap_rad(yaw - (math.pi / 2.0 + 0.02))) < 1e-3, yaw
    assert abs(yaw - raw) > 1.0
    print("  heading wrap: yaw near pi/2, not the unwrapped dir difference")


def check_missing_brake_lights_masked() -> None:
    """A missing brake_lights field is masked. The ego lamp is not copied across."""
    ego_dir = [0.0, 1.0, 0.0]
    cars = [
        {
            "name": "gvd_traffic_0",
            "model": "etk800",
            "pos": [0.0, 12.0, 0.0],
            "vel": [0.0, 8.0, 0.0],
            "dir": [0.0, 1.0, 0.0],
            "brake_lights": 1,
        },
        {
            "name": "ambient_2",
            "model": "citybus",
            "pos": [4.0, 20.0, 0.0],
            "vel": [0.0, 3.0, 0.0],
            "dir": [0.0, 1.0, 0.0],
            "brakelights": 0,
        },
        {
            "name": "ambient_3",
            "pos": [-3.0, 9.0, 0.0],
            "vel": [1.0, 0.0, 0.0],
            "dir": [1.0, 0.0, 0.0],
        },
        {
            # Lua spawn: scenetree object, Electrics attach failed. Not "lights off".
            "name": "gvd_traffic_1",
            "model": "etk800",
            "pos": [0.0, 30.0, 0.0],
            "vel": [0.0, 1.0, 0.0],
            "dir": [0.0, 1.0, 0.0],
            "brake_lights": None,
        },
    ]
    row = _row([0.0, 0.0, 0.0], ego_dir, cars)
    off = labels_for_row(row)
    assert "brake_lights" in off and "brake_lights_valid" in off
    assert np.allclose(off["brake_lights"][:3], [1.0, 0.0, 0.0])
    assert math.isnan(float(off["brake_lights"][3]))
    assert np.all(off["brake_lights_valid"] == 0.0)
    opened = labels_for_row(row, brake_light_head=True)
    assert np.allclose(opened["brake_lights_valid"][:3], [1.0, 1.0, 0.0])
    # Null stays masked after the flag. It is not a supervised 0.
    assert float(opened["brake_lights_valid"][3]) == 0.0
    assert math.isnan(float(opened["brake_lights"][3]))
    assert float(opened["object_valid"][3]) == 1.0
    assert float(opened["brake_lights"][1]) == 0.0
    pull = np.nan_to_num(np.ones(4) - opened["brake_lights"][:4], nan=99.0)
    pull = pull * opened["brake_lights_valid"][:4]
    assert float(pull[3]) == 0.0
    assert opened["brake_lights_valid"][4:].sum() == 0.0
    # citybus is a model string, not a class guess of bus. No model → vehicle.
    assert int(opened["object_class"][0]) == OBJECT_CLASSES.index("car")
    assert int(opened["object_class"][1]) == OBJECT_CLASSES.index("vehicle")
    assert int(opened["object_class"][2]) == OBJECT_CLASSES.index("vehicle")
    assert "brake_lights" not in OUTPUT_FIELDS
    assert "brake_lights" not in HEAD_NAMES
    assert "brake_lights" not in LOSS_TERMS
    assert OBJECT_FEAT == 8
    plain = labels_for_row({"tech": {}})
    assert "brake_lights" not in plain
    print("  brake lights: missing field masked, head flag off by default")


def check_objects_behind_ego() -> None:
    """A companion behind the ego keeps a negative y and still scores."""
    ego_pos = [100.0, 250.0, 5.0]
    ego_dir = [1.0, 0.0, 0.0]
    behind = [80.0, 250.0, 1.0]
    ahead = [110.0, 240.0, 1.0]
    row = _row(
        ego_pos,
        ego_dir,
        [
            {
                "name": "gvd_traffic_1",
                "model": "etk800",
                "pos": ahead,
                "vel": [3.0, 4.0, 12.0],
                "dir": [1.0, 0.0, 0.0],
            },
            {
                "name": "gvd_traffic_0",
                "model": "etk800",
                "pos": behind,
                "vel": [6.0, 0.0, 0.0],
                "dir": [0.0, 1.0, 0.0],
            },
            {"name": "gvd_traffic_2", "pos": None, "vel": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0]},
        ],
    )
    labels = labels_for_row(row)
    assert int(labels["object_valid"].sum()) == 2
    assert float(labels["object_valid"][2]) == 0.0
    # Order is the traffic list. The rear car is not dropped or sorted away.
    assert abs(float(labels["objects"][1, 0])) < 1e-3
    assert abs(float(labels["objects"][1, 1]) - (-20.0)) < 1e-3
    assert abs(float(labels["objects"][0, 0]) - 10.0) < 1e-3
    assert abs(float(labels["objects"][0, 1]) - 10.0) < 1e-3
    rear = world_to_ego((behind[0], behind[1]), ego_pos, ego_dir)
    assert rear is not None and rear[1] < 0.0
    assert math.hypot(float(labels["objects"][1, 0]) - rear[0], float(labels["objects"][1, 1]) - rear[1]) < 0.5
    # Same world heading as the ego is pi/2. The rear car points along +Y, ego along +X.
    assert abs(wrap_rad(float(labels["objects"][1, 2]) - math.pi)) < 1e-3
    assert abs(float(labels["objects"][0, 5]) - 5.0) < 1e-4
    # No measured size, so the 4.5 x 1.8 fill must not come back.
    assert float(labels["objects"][0, 3]) == 0.0
    assert float(labels["objects"][0, 4]) == 0.0
    assert float(labels["objects"][1, 3]) == 0.0
    # A null probe does not keep the tech track and does not invent the last car.
    null = labels_for_row(_row(ego_pos, ego_dir, None))
    assert int(null["object_valid"].sum()) == 0
    assert not np.any(np.abs(null["objects"][:, :2]) > 0.0)
    painted = _row(ego_pos, ego_dir, row["traffic"])
    painted["tech"]["lanes"] = [[{"x": -1.5, "y": 0.0}, {"x": -1.5, "y": 30.0}]]
    kept = labels_for_row(painted)
    assert int(kept["lane_valid"].sum()) == 1
    assert int(kept["object_valid"].sum()) == 2
    print("  behind ego: y=-20 scores, null traffic scores nothing")


def check_runtime_stays_camera_only() -> None:
    import ast

    banned = ("vehicle_labels",)
    targets = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "planning" / "path_predictor.py",
    ]
    targets.extend(sorted((ROOT / "python" / "perception").glob("*.py")))
    targets.extend(sorted((ROOT / "python" / "viz").glob("*.py")))
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            for name in names:
                assert not any(part in name for part in banned), f"{path.name} imports {name}"
    print(f"  runtime: {len(targets)} files do not import vehicle labels")


def check_left_yaw_stays_inside_pi() -> None:
    """A left-pointing dir is +pi, not -pi, including after the float32 cast."""
    cap = float(np.nextafter(np.float32(math.pi), np.float32(0.0)))
    assert cap <= math.pi
    for ego_h in (math.pi, 1.0):
        ego_dir = [math.cos(ego_h), math.sin(ego_h), 0.0]
        # Ego left is -right. right = (sin h, -cos h), so left = (-sin h, cos h).
        left = [-math.sin(ego_h), math.cos(ego_h), 0.0]
        row = {
            "ego": {"pos": [0.0, 0.0, 0.0], "dir": ego_dir},
            "traffic": [
                {
                    "name": "gvd_traffic_0",
                    "pos": [12.0 * ego_dir[0], 12.0 * ego_dir[1], 0.0],
                    "vel": [0.0, 0.0, 0.0],
                    "dir": left,
                }
            ],
        }
        yaw = float(labels_for_row(row)["objects"][0, 2])
        # Forcing the branch cut to -pi fails this. float32(pi) is past pi and fails it too.
        assert -math.pi < yaw <= math.pi, (ego_h, yaw)
        assert yaw > 0.0, (ego_h, yaw)
        assert yaw == cap, (ego_h, yaw, cap)
    print("  yaw cut: left-pointing dir is +pi in (-pi, pi] after float32")


def check_brake_only_zero_or_one() -> None:
    """0 and 1 are lamps. 0.5, 2, -1, and NaN are masked, not a dim light."""
    def packed(value: object):
        row = _row(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [
                {
                    "name": "gvd_traffic_0",
                    "pos": [0.0, 10.0, 0.0],
                    "vel": [0.0, 1.0, 0.0],
                    "dir": [0.0, 1.0, 0.0],
                    "brake_lights": value,
                }
            ],
        )
        return labels_for_row(row, brake_light_head=True)

    for bad in (0.5, 2.0, -1.0, float("nan")):
        got = packed(bad)
        assert math.isnan(float(got["brake_lights"][0])), (bad, got["brake_lights"][0])
        assert float(got["brake_lights_valid"][0]) == 0.0, bad
        assert float(got["object_valid"][0]) == 1.0, bad
    off = packed(0)
    on = packed(1)
    assert float(off["brake_lights"][0]) == 0.0 and float(off["brake_lights_valid"][0]) == 1.0
    assert float(on["brake_lights"][0]) == 1.0 and float(on["brake_lights_valid"][0]) == 1.0
    assert float(packed(False)["brake_lights"][0]) == 0.0
    assert float(packed(True)["brake_lights_valid"][0]) == 1.0
    print("  brake lights: only exact 0 or 1 is a label")


def check_size_default_dropped() -> None:
    from python.train.scene_net import object_feature_keep

    missing = labels_from_tech({"objects": [{"cls": "car", "x": 0.2, "y": 6.0, "yaw": 0.4}]})
    assert float(missing["objects"][0, 3]) == 0.0
    assert float(missing["objects"][0, 4]) == 0.0
    explicit = labels_from_tech(
        {"objects": [{"cls": "car", "x": 0.2, "y": 6.0, "yaw": 0.4, "length": 4.5, "width": 1.8}]}
    )
    assert abs(float(explicit["objects"][0, 3]) - 4.5) < 1e-6
    assert abs(float(explicit["objects"][0, 4]) - 1.8) < 1e-6
    keep = object_feature_keep()
    assert keep.shape == (OBJECT_FEAT,)
    assert keep[3] == 0.0 and keep[4] == 0.0
    assert keep[0] == keep[1] == keep[2] == keep[5] == 1.0
    print("  size: missing length/width is 0, and those columns are out of the keep mask")


def check_size_channels_out_of_loss() -> None:
    try:
        import torch
    except ImportError:
        skip_check("size_loss", "torch missing")
        return
    from python.train.scene_net import ScenePrediction, scene_loss

    def prediction(objects: "torch.Tensor") -> ScenePrediction:
        z6 = torch.zeros(6, 16, 2)
        return ScenePrediction(
            lanes=z6,
            lane_valid=torch.zeros(6),
            curbs=torch.zeros(2, 16, 2),
            curb_valid=torch.zeros(2),
            objects=objects,
            object_class=torch.zeros(16, len(OBJECT_CLASSES)),
            object_valid=torch.full((16,), 0.5),
            signs=torch.zeros(8, 4),
            sign_class=torch.zeros(8, 4),
            sign_state=torch.zeros(8, 4),
            sign_valid=torch.zeros(8),
        )

    base = torch.zeros(16, OBJECT_FEAT)
    base[0] = torch.tensor([0.0, 10.0, math.pi / 2.0, 9.0, 9.0, 3.0, 0.0, 0.0])
    pred_t = base.clone().requires_grad_(True)

    def target(length: float, width: float) -> dict:
        gt = base.detach().clone()
        gt[0, 0] = 2.0
        gt[0, 3] = length
        gt[0, 4] = width
        valid = torch.zeros(16)
        valid[0] = 1.0
        classes = torch.zeros(16, dtype=torch.long)
        classes[0] = OBJECT_CLASSES.index("car")
        return {"objects": gt, "object_class": classes, "object_valid": valid}

    wide = scene_loss(prediction(pred_t), target(4.5, 1.8), ("objects",))
    flat = scene_loss(prediction(pred_t), target(0.0, 0.0), ("objects",))
    assert abs(float(wide["points"].detach()) - float(flat["points"].detach())) < 1e-6
    wide["points"].backward()
    assert pred_t.grad is not None
    assert torch.count_nonzero(pred_t.grad[:, 3:5]) == 0
    assert float(pred_t.grad[0, 0].abs()) > 0.0
    print("  size loss: length/width do not change the point loss or its grad")


def check_traffic_partial_unseen_grad_is_zero() -> None:
    """Traffic slots supervise x, y, yaw, speed. partial and unseen_s do not."""
    try:
        import torch
    except ImportError:
        skip_check("traffic_partial_grad", "torch missing")
        return
    from python.train.scene_net import ScenePrediction, scene_loss

    row = _row(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [
            {
                "name": "gvd_traffic_0",
                "model": "etk800",
                "pos": [0.0, 12.0, 0.0],
                "vel": [0.0, 4.0, 0.0],
                "dir": [0.0, 1.0, 0.0],
            }
        ],
    )
    labels = labels_for_row(row)
    gt = torch.tensor(labels["objects"])
    pred_t = gt.clone()
    pred_t[0, 0] = 5.0
    pred_t[0, 6] = 4.0
    pred_t[0, 7] = 7.0
    pred_t.requires_grad_(True)
    target = {
        "objects": gt,
        "object_class": torch.tensor(labels["object_class"]),
        "object_valid": torch.tensor(labels["object_valid"]),
        "object_feat_keep": torch.tensor(labels["object_feat_keep"]),
    }
    pred = ScenePrediction(
        lanes=torch.zeros(6, 16, 2),
        lane_valid=torch.zeros(6),
        curbs=torch.zeros(2, 16, 2),
        curb_valid=torch.zeros(2),
        objects=pred_t,
        object_class=torch.zeros(16, len(OBJECT_CLASSES)),
        object_valid=torch.full((16,), 0.5),
        signs=torch.zeros(8, 4),
        sign_class=torch.zeros(8, 4),
        sign_state=torch.zeros(8, 4),
        sign_valid=torch.zeros(8),
    )
    scene_loss(pred, target, ("objects",))["points"].backward()
    assert pred_t.grad is not None
    assert torch.count_nonzero(pred_t.grad[:, 6:8]) == 0
    assert float(pred_t.grad[0, 0].abs()) > 0.0
    # A hand-built track with no traffic key still trains partial.
    hand = torch.zeros(16, OBJECT_FEAT)
    hand[0] = torch.tensor([0.0, 10.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    hand_pred = hand.clone()
    hand_pred[0, 6] = 2.0
    hand_pred.requires_grad_(True)
    hand_tgt = {
        "objects": hand,
        "object_class": torch.zeros(16, dtype=torch.long),
        "object_valid": torch.zeros(16),
    }
    hand_tgt["object_valid"][0] = 1.0
    scene_loss(
        ScenePrediction(
            lanes=torch.zeros(6, 16, 2),
            lane_valid=torch.zeros(6),
            curbs=torch.zeros(2, 16, 2),
            curb_valid=torch.zeros(2),
            objects=hand_pred,
            object_class=torch.zeros(16, len(OBJECT_CLASSES)),
            object_valid=torch.full((16,), 0.5),
            signs=torch.zeros(8, 4),
            sign_class=torch.zeros(8, 4),
            sign_state=torch.zeros(8, 4),
            sign_valid=torch.zeros(8),
        ),
        hand_tgt,
        ("objects",),
    )["points"].backward()
    assert float(hand_pred.grad[0, 6].abs()) > 0.0
    print("  traffic loss: partial and unseen_s gradients are 0")


def main() -> None:
    check_heading_wrap()
    check_left_yaw_stays_inside_pi()
    check_missing_brake_lights_masked()
    check_brake_only_zero_or_one()
    check_objects_behind_ego()
    check_size_default_dropped()
    check_size_channels_out_of_loss()
    check_traffic_partial_unseen_grad_is_zero()
    check_runtime_stays_camera_only()
    print("test_vehicle_labels: OK")


if __name__ == "__main__":
    main()
