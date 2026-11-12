#!/usr/bin/env python3
"""Labels v4 intersections: intent entry/exit, per-tick signs, stop-line scalar.

Offline. Synthetic junction and signal rows only. No BeamNG, no planner.
Torch checks skip when PyTorch is not installed.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from gvd_test_env import skip_check  # noqa: E402
from synthetic_roads import load_graph, track  # noqa: E402


def _rows_from_poses(poses: list[tuple[float, float, float]], *, dt: float = 0.1) -> list[dict]:
    rows = []
    for i, (x, y, heading) in enumerate(poses):
        rows.append(
            {
                "t": round(i * dt, 4),
                "pos": [float(x), float(y), 0.0],
                "dir": [math.cos(heading), math.sin(heading), 0.0],
            }
        )
    return rows


def _graph():
    from python.sensors.route_intent import Graph

    return Graph.from_fixture(load_graph(), "synth_a")


def _labels_for_path(path: list[str]):
    from python.collect.route_hindsight import label_sequence

    data = load_graph()
    poses = track(data, "synth_a", path, speed_mps=10.0, dt_s=0.1)
    return label_sequence(_graph(), _rows_from_poses(poses))


def check_isolation() -> None:
    banned = ("intersection_labels", "label_signs", "stop_line_target")
    targets = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "control" / "actuate.py",
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
                low = name.lower()
                assert not any(b in low for b in banned), f"{path.name} imports {name}"
    live = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "signs.jsonl" not in live
    assert "stoplines.jsonl" not in live
    assert "stop_line_head" not in live
    print(f"  isolation: {len(targets)} runtime files import no intersection labeler")


def check_intent_writes_entry_exit() -> None:
    from python.collect.route_hindsight import branches_are_junction_neighbors

    graph = _graph()
    labels = _labels_for_path(["S", "T", "X"])
    turned = [row for row in labels if row["cmd"] == "right"]
    assert turned, {row["cmd"] for row in labels}
    for row in turned:
        assert row["entry"] == "S", row
        assert row["exit"] == "X", row
        assert branches_are_junction_neighbors(graph, row["entry"], row["exit"])
        assert 0.0 <= float(row["next_jct_m"]) <= 60.0
    far = [row for row in labels if row["cmd"] == "follow"]
    assert far
    assert all("entry" not in row and "exit" not in row for row in far)
    assert not branches_are_junction_neighbors(graph, "S", "YL")
    print("  S->T->X writes entry S and exit X; far frames stay follow without a branch")


def check_masked_turns_stay_masked() -> None:
    from python.collect.route_hindsight import branches_are_junction_neighbors, label_sequence
    from python.sensors.route_intent import Graph

    graph = _graph()
    uturn = _labels_for_path(["S", "T", "S"])
    masked = [row for row in uturn if row.get("reason") == "uturn"]
    assert masked, {(row["cmd"], row.get("reason")) for row in uturn}
    for row in masked:
        assert row["cmd"] == "mask"
        assert branches_are_junction_neighbors(graph, row["entry"], row["exit"])

    ang = math.radians(18.0)
    amb_graph = Graph(
        nodes={
            "A": (0.0, 0.0),
            "J": (50.0, 0.0),
            "B": (50.0 + 50.0 * math.cos(ang), 50.0 * math.sin(ang)),
            "C": (50.0, 50.0),
        },
        edges=(("A", "J", False), ("J", "B", False), ("J", "C", False)),
    )
    poses = [(i * 2.5, 0.0, 0.0) for i in range(40)]
    for i in range(1, 25):
        poses.append((50.0 + i * 2.5 * math.cos(ang), i * 2.5 * math.sin(ang), ang))
    ambiguous = label_sequence(amb_graph, _rows_from_poses(poses))
    amb_rows = [row for row in ambiguous if row.get("reason") == "ambiguous"]
    assert amb_rows, {(row["cmd"], row.get("reason")) for row in ambiguous}
    for row in amb_rows:
        assert row["cmd"] == "mask"
        assert branches_are_junction_neighbors(amb_graph, row["entry"], row["exit"])

    short = _labels_for_path(["S", "T"])
    no_future = [row for row in short if row.get("reason") == "no_future"]
    assert no_future
    assert all(row["cmd"] == "mask" for row in no_future)
    assert all(row["cmd"] not in ("left", "right", "straight") for row in uturn + ambiguous + short if row["cmd"] == "mask")
    print("  uturn and ambiguous stay mask and keep neighbor ids; no_future stays mask")


def check_intent_cli_and_route_loss_mask() -> None:
    import label_route_intent
    from python.train.route_cond import ROUTE_COND_DIM, attach_route_cmds, encode_route

    assert ROUTE_COND_DIM == 5
    assert encode_route("left", 20.0).shape == (5,)
    data = load_graph()
    poses = track(data, "synth_a", ["S", "T", "X"], speed_mps=10.0, dt_s=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state = root / "state.jsonl"
        out = root / "intent.jsonl"
        graph_path = root / "graph.json"
        graph_path.write_text(
            (ROOT / "tests/fixtures/junctions_synthetic.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with state.open("w", encoding="utf-8") as fh:
            for row in _rows_from_poses(poses):
                fh.write(json.dumps(row) + "\n")
        code = label_route_intent.main(
            ["--state", str(state), "--graph", str(graph_path), "--level", "synth_a", "--out", str(out)]
        )
        assert code == 0, code
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert any(row["cmd"] == "right" and row["entry"] == "S" and row["exit"] == "X" for row in lines)
        refused = label_route_intent.main(
            ["--state", str(state), "--graph", str(graph_path), "--level", "synth_a", "--out", str(root / "gt.jsonl")]
        )
        assert refused == 2
        assert not (root / "gt.jsonl").exists()

        (root / "intent.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"t": 1.0, "cmd": "left", "next_jct_m": 20.0, "entry": "S", "exit": "W", "src": "hindsight"}),
                    json.dumps({"t": 3.0, "cmd": "follow", "next_jct_m": 90.0, "src": "hindsight"}),
                    json.dumps({"t": 4.0, "cmd": "mask", "next_jct_m": 12.0, "reason": "uturn", "entry": "S", "exit": "S", "src": "hindsight"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows = [
            {"t": 1.0, "source_root": str(root)},
            {"t": 2.0, "source_root": str(root)},
            {"t": 3.0, "source_root": str(root)},
            {"t": 4.0, "source_root": str(root)},
        ]
        counts = attach_route_cmds(rows, missing="mask")
        assert rows[0]["route_cmd"] == "left" and rows[0]["route_supervised"] == 1.0
        assert rows[1]["route_cmd"] == "mask" and rows[1]["route_cmd"] != "follow"
        assert rows[1]["route_supervised"] == 0.0
        assert counts["missing"] == 1
        assert rows[2]["route_cmd"] == "follow" and rows[2]["route_supervised"] == 0.0
        assert rows[3]["route_cmd"] == "mask" and rows[3]["route_supervised"] == 0.0
    trainer = (ROOT / "python/train/train_scene.py").read_text(encoding="utf-8")
    assert 'attach_route_cmds(rows, missing="mask")' in trainer
    print("  intent CLI writes entry/exit; a missing sidecar stays mask, not follow")


def _signal_fixture() -> tuple[list[dict], list[dict]]:
    state = [
        {"t": 1.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 2.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 3.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 4.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 5.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 6.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 7.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        {"t": 8.0},
    ]
    signals = [
        {"t": 1.0, "signals": [{"cls": "traffic_light", "pos": [2.0, 10.0, 0.0], "state": "red"}]},
        {"t": 2.0, "err": "missing get_map_node_signals"},
        {"t": 3.0, "signals": [{"cls": "traffic_light", "pos": [1.0, 8.0, 0.0], "state": "purple"}]},
        {"t": 5.0, "signals": [{"cls": "traffic_light", "pos": [0.0, 30.0, 0.0], "state": "green"}]},
        {"t": 6.0, "signals": "not-a-dump"},
        {"t": 7.0, "signals": []},
        {"t": 8.0, "signals": [{"cls": "traffic_light", "pos": [0.0, 12.0, 0.0], "state": "red"}]},
    ]
    return state, signals


def check_signs_join_masks_missing_and_unknown() -> None:
    from python.collect.signal_labels import label_signs
    from python.train.intersection_labels import attach_sign_labels
    from python.train.scene_net import labels_from_tech

    state, signals = _signal_fixture()
    rows = label_signs(state, signals)
    by_t = {row["t"]: row for row in rows}
    red = by_t[1.0]["signs"][0]
    assert red["cls"] == "traffic_light" and red["state"] == "red" and red["src"] == "traffic_signals"
    assert abs(float(red["y"]) - 10.0) < 0.2 and abs(float(red["x"]) - 2.0) < 0.2
    assert "signs" not in by_t[2.0] and by_t[2.0]["err"] == "missing get_map_node_signals"
    unknown = by_t[3.0]["signs"][0]
    assert unknown["state"] == "unknown"
    assert unknown["state"] not in ("red", "yellow", "green")
    assert "signs" not in by_t[4.0] and "err" in by_t[4.0]
    green = by_t[5.0]["signs"][0]
    assert green["state"] == "green" and abs(float(green["y"]) - 30.0) < 0.2
    assert abs(float(green["y"]) - float(red["y"])) > 5.0
    assert "signs" not in by_t[6.0] and "err" in by_t[6.0]
    assert by_t[7.0]["signs"] == []
    assert "signs" not in by_t[8.0] and "err" in by_t[8.0]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with (root / "signs.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        stale = [{"cls": "stop_sign", "x": 0.0, "y": 1.0, "state": "red"}]
        joined = []
        for state_row in state:
            item = {"t": state_row["t"], "source_root": str(root), "tech": {"signs": list(stale)}}
            joined.append(item)
        counts = attach_sign_labels(joined)
        assert counts["parsed"] >= 2 and counts["masked"] >= 1
        assert joined[0]["tech"]["signs"][0]["state"] == "red"
        assert "signs" not in joined[1]["tech"]
        assert joined[2]["tech"]["signs"][0]["state"] == "unknown"
        unknown_labels = labels_from_tech(joined[2]["tech"])
        assert int(unknown_labels["sign_valid"].sum()) == 1
        assert int(unknown_labels["sign_state"][0]) == 0
        err_labels = labels_from_tech(joined[1]["tech"])
        assert int(err_labels["sign_valid"].sum()) == 0
        # No sidecar at all: do not invent signs, and do not wipe an unrelated row.
        other = Path(tmp) / "empty"
        other.mkdir()
        untouched = {"t": 1.0, "source_root": str(other), "tech": {"signs": list(stale)}}
        attach_sign_labels([untouched])
        assert untouched["tech"]["signs"][0]["cls"] == "stop_sign"
    print("  signs.jsonl joins on t; {t, err}, unknown lamp, and a missing tick stay masked")


def check_signs_cli_reads_per_tick_dump() -> None:
    import export_signal_labels as ex
    from python.collect.signal_labels import label_signs

    state, signals = _signal_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with (root / "state.jsonl").open("w", encoding="utf-8") as fh:
            for row in state:
                fh.write(json.dumps(row) + "\n")
        dump = root / "signals.jsonl"
        with dump.open("w", encoding="utf-8") as fh:
            for row in signals:
                fh.write(json.dumps(row) + "\n")
        code = ex.main([str(root), "--signals", str(dump), "--out", "signs.jsonl"])
        assert code == 0, code
        written = [json.loads(line) for line in (root / "signs.jsonl").read_text(encoding="utf-8").splitlines()]
        expect = label_signs(state, signals)
        assert [row.get("t") for row in written] == [row.get("t") for row in expect]
        assert written[1].get("err") == "missing get_map_node_signals" and "signs" not in written[1]
        assert written[2]["signs"][0]["state"] == "unknown"
        assert written[0]["signs"][0]["state"] == "red"
        assert written[4]["signs"][0]["state"] == "green"
        refused = ex.main([str(root), "--signals", str(dump), "--out", "gt.jsonl"])
        assert refused == 2
        assert not (root / "gt.jsonl").exists()
    print("  export CLI reads per-tick signals.jsonl and refuses gt.jsonl")


def check_stop_line_scalar_masks_invalid() -> None:
    import label_stoplines
    from python.debug.painted_lanes import PaintedLanes
    from python.train.cam_strip import HEAD_NAMES, parse_heads
    from python.train.intersection_labels import attach_stop_lines, stop_line_target
    from python.train.scene_net import OUTPUT_FIELDS, labels_for_row

    assert "stop_line" not in OUTPUT_FIELDS
    assert "stop_line" not in HEAD_NAMES
    assert "stop_line" not in parse_heads("")
    assert parse_heads("lane_gt,stop_line") == ("lane_gt", "stop_line")

    decals = [
        {"material": "line_white", "nodes": [[-1.75, y, 0.0, 0.15] for y in range(-10, 41, 2)]},
        {"material": "line_white", "nodes": [[1.75, y, 0.0, 0.15] for y in range(-10, 41, 2)]},
        {"material": "line_white", "nodes": [[-2.0, 15.0, 0.0, 0.15], [2.0, 15.0, 0.0, 0.15]]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "state.jsonl").write_text(
            json.dumps({"t": 1.0, "map": "synth", "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}) + "\n",
            encoding="utf-8",
        )
        decal_path = root / "decals.json"
        decal_path.write_text(json.dumps(decals), encoding="utf-8")
        code = label_stoplines.main([str(root), "--decals", str(decal_path), "--out", "stoplines.jsonl"])
        assert code == 0, code
        assert not (root / "gt.jsonl").exists()
        sidecar = [json.loads(line) for line in (root / "stoplines.jsonl").read_text(encoding="utf-8").splitlines()]
        values, mask = stop_line_target(sidecar[0]["bars"])
        assert float(mask.sum()) == 2.0
        assert abs(float(values[0]) - 15.0) < 0.3
        assert float(values[1]) == 1.0
        paint = PaintedLanes(decals)
        lanes = paint.lane_at((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert lanes[0] is not None

        masked, masked_m = stop_line_target(
            [{"src": "mask", "y_m": 12.0, "x0": -2.0, "x1": 2.0, "valid": False}]
        )
        assert float(masked_m.sum()) == 0.0
        nearest, near_m = stop_line_target(
            [
                {"src": "mask", "y_m": 4.0, "x0": -1.0, "x1": 1.0, "valid": False},
                {"src": "stop_line", "y_m": 18.0, "x0": -2.0, "x1": 2.0, "valid": True},
                {"src": "stop_line", "y_m": 30.0, "x0": -2.0, "x1": 2.0, "valid": True},
            ]
        )
        assert abs(float(nearest[0]) - 18.0) < 1e-3 and float(near_m[0]) == 1.0

        row = {
            "t": 1.0,
            "source_root": str(root),
            "lane_gt": [0.25] * 14,
            "lane_gt_valid": [1.0] * 14,
        }
        bare = labels_for_row(dict(row))
        assert "stop_line" not in bare
        attach_stop_lines([row])
        supervised = labels_for_row(row, stop_line=True)
        alone = labels_for_row(
            {"lane_gt": [0.25] * 14, "lane_gt_valid": [1.0] * 14},
            stop_line=True,
        )
        import numpy as np

        assert np.allclose(supervised["lane_gt"], alone["lane_gt"])
        assert abs(float(supervised["stop_line"][0]) - 15.0) < 0.3
        assert float(supervised["stop_line_valid"].sum()) == 2.0
        missing = {"t": 9.0, "source_root": str(root)}
        attach_stop_lines([missing])
        assert float(missing["stop_line_valid"].sum()) == 0.0
    trainer = (ROOT / "python/train/train_scene.py").read_text(encoding="utf-8")
    assert "stop_line=use_stop_line" in trainer
    assert "attach_stop_lines" in trainer
    assert "attach_sign_labels" in trainer
    ckpt = (ROOT / "python/train/epoch_ckpt.py").read_text(encoding="utf-8")
    assert "stop_line_head." in ckpt
    print("  stop-line scalar trains the nearest valid bar; a mask bar and a missing tick do not")


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _blank_prediction(torch, signs, sign_class, sign_state, sign_valid):
    from python.train.scene_net import (
        LANE_POINTS,
        N_CURBS,
        N_LANES,
        N_OBJECTS,
        OBJECT_CLASSES,
        OBJECT_FEAT,
        ScenePrediction,
    )
    from python.debug.lane_targets import LANE_GT_DIM

    return ScenePrediction(
        lanes=torch.zeros(N_LANES, LANE_POINTS, 2),
        lane_valid=torch.zeros(N_LANES),
        curbs=torch.zeros(N_CURBS, LANE_POINTS, 2),
        curb_valid=torch.zeros(N_CURBS),
        objects=torch.zeros(N_OBJECTS, OBJECT_FEAT),
        object_class=torch.zeros(N_OBJECTS, len(OBJECT_CLASSES)),
        object_valid=torch.zeros(N_OBJECTS),
        signs=signs,
        sign_class=sign_class,
        sign_state=sign_state,
        sign_valid=sign_valid,
        lane_gt=torch.zeros(LANE_GT_DIM),
    )


def _sign_pred(torch, slots: dict[int, tuple[float, float, list[float]]]):
    from python.train.scene_net import N_SIGNS, SIGN_CLASSES, SIGN_FEAT

    signs = torch.zeros(N_SIGNS, SIGN_FEAT)
    sign_class = torch.zeros(N_SIGNS, len(SIGN_CLASSES))
    sign_state = torch.zeros(N_SIGNS, 4)
    sign_valid = torch.zeros(N_SIGNS)
    for index in range(N_SIGNS):
        signs[index, 1] = -500.0
    for index, (x, y, logits) in slots.items():
        signs[index, 0] = x
        signs[index, 1] = y
        sign_state[index] = torch.tensor(logits, dtype=torch.float32)
        sign_valid[index] = 1.0
    return signs, sign_class, sign_state, sign_valid


def _target_from_signs(torch, signs: list[dict]):
    from python.train.scene_net import labels_from_tech

    labels = labels_from_tech({"signs": signs})
    return {key: torch.as_tensor(value) for key, value in labels.items()}


def check_lamp_loss_skips_unknown() -> None:
    src = (ROOT / "python/train/scene_net.py").read_text(encoding="utf-8")
    assert "gt_state[gt_i] > 0" in src
    assert '"stop_line"' in src.split("wanted -")[1].split("frozenset")[0]
    torch = _torch()
    if torch is None:
        skip_check("lamp_loss_skips_unknown", "PyTorch is not installed")
        print("  lamp CE source skips unknown (index 0); numeric check needs PyTorch")
        return
    from python.train.scene_net import scene_loss

    unknown_only = _target_from_signs(
        torch, [{"cls": "traffic_light", "x": 0.0, "y": 30.0, "state": "unknown"}]
    )
    signs, sign_class, sign_state, sign_valid = _sign_pred(torch, {0: (0.0, 0.0, [0.0, 0.0, 0.0, 10.0])})
    sign_state = sign_state.detach().requires_grad_(True)
    pred = _blank_prediction(torch, signs, sign_class, sign_state, sign_valid)
    loss = scene_loss(pred, unknown_only, heads=("signs",))
    assert float(loss["points"]) > 0.0
    assert float(loss["lamps"]) == 0.0
    assert not bool(loss["lamps"].requires_grad)
    assert int(unknown_only["sign_valid"].sum()) == 1

    red_logits = [0.0, 10.0, 0.0, 0.0]
    green_logits = [0.0, 0.0, 0.0, 10.0]
    red_only = _target_from_signs(torch, [{"cls": "traffic_light", "x": 4.0, "y": 20.0, "state": "red"}])
    both = _target_from_signs(
        torch,
        [
            {"cls": "traffic_light", "x": 0.0, "y": 10.0, "state": "unknown"},
            {"cls": "traffic_light", "x": 4.0, "y": 20.0, "state": "red"},
        ],
    )
    stop_and_red = _target_from_signs(
        torch,
        [
            {"cls": "stop_sign", "x": 0.0, "y": 10.0, "state": "unknown"},
            {"cls": "traffic_light", "x": 4.0, "y": 20.0, "state": "red"},
        ],
    )
    pred_both = _blank_prediction(torch, *_sign_pred(torch, {0: (0.0, 10.0, green_logits), 1: (4.0, 20.0, red_logits)}))
    pred_red = _blank_prediction(torch, *_sign_pred(torch, {1: (4.0, 20.0, red_logits)}))
    lamps_both = float(scene_loss(pred_both, both, heads=("signs",))["lamps"])
    lamps_red = float(scene_loss(pred_red, red_only, heads=("signs",))["lamps"])
    lamps_stop = float(scene_loss(pred_both, stop_and_red, heads=("signs",))["lamps"])
    assert abs(lamps_both - lamps_red) < 1e-5, (lamps_both, lamps_red)
    assert abs(lamps_stop - lamps_red) < 1e-5, (lamps_stop, lamps_red)

    wrong = _blank_prediction(torch, *_sign_pred(torch, {1: (4.0, 20.0, green_logits)}))
    wrong_state = wrong.sign_state.detach().requires_grad_(True)
    wrong = _blank_prediction(torch, wrong.signs, wrong.sign_class, wrong_state, wrong.sign_valid)
    red_loss = scene_loss(wrong, red_only, heads=("signs",))
    assert float(red_loss["lamps"].detach()) > 0.0
    assert bool(red_loss["lamps"].requires_grad)
    print("  lamp CE skips unknown and non-lights; a red lamp still trains; position loss stays")


def check_stop_line_head_is_training_only() -> None:
    torch = _torch()
    if torch is None:
        skip_check("stop_line_head", "PyTorch is not installed")
        return
    from python.train.scene_net import OUTPUT_FIELDS, SceneNet, scene_loss

    plain = SceneNet(cams=("main",))
    headed = SceneNet(cams=("main",), stop_line=True)
    assert getattr(plain, "stop_line", False) is False
    assert not any(name.startswith("stop_line_head") for name in plain.state_dict())
    assert any(name.startswith("stop_line_head") for name in headed.state_dict())
    assert ":stop_line=2" in (getattr(headed, "pause_arch", "") or "")
    missing, unexpected = headed.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert any(name.startswith("stop_line_head") for name in missing)
    torch.nn.init.zeros_(headed.stop_line_head.weight)
    torch.nn.init.zeros_(headed.stop_line_head.bias)
    h, w = int(headed.canvas_h), int(headed.canvas_w)
    strip = torch.zeros(1, 3, h, w)
    pred, _ = headed.forward(strip, 0.1, headed.initial_state())
    assert tuple(pred.stop_line.shape) == (2,)
    assert "stop_line" not in pred.as_dict()
    assert list(pred.as_dict()) == list(OUTPUT_FIELDS)
    base_pred, _ = plain.forward(strip, 0.1, plain.initial_state())
    assert getattr(base_pred, "stop_line", None) is None

    goal = torch.tensor([15.0, 1.0])
    on = scene_loss(pred, {"stop_line": goal, "stop_line_valid": torch.ones(2)}, heads=("stop_line",))
    off = scene_loss(pred, {"stop_line": goal, "stop_line_valid": torch.zeros(2)}, heads=("stop_line",))
    assert float(on["stop_line"].detach()) > 0.0
    assert float(off["stop_line"].detach()) == 0.0
    # A lane-only run that also names the scalar must not turn the lamp loss back on.
    sign_target = _target_from_signs(
        torch, [{"cls": "traffic_light", "x": 0.0, "y": 30.0, "state": "red"}]
    )
    sign_target["lane_gt"] = torch.zeros_like(pred.lane_gt)
    sign_target["lane_gt_valid"] = torch.zeros_like(pred.lane_gt)
    sign_target["stop_line"] = goal
    sign_target["stop_line_valid"] = torch.ones(2)
    sibling = scene_loss(pred, sign_target, heads=("lane_gt", "stop_line"))
    assert float(sibling["lamps"]) == 0.0
    print("  stop_line head is opt-in, absent from ONNX fields, and masked when valid is 0")


def main() -> int:
    check_isolation()
    check_intent_writes_entry_exit()
    check_masked_turns_stay_masked()
    check_intent_cli_and_route_loss_mask()
    check_signs_join_masks_missing_and_unknown()
    check_signs_cli_reads_per_tick_dump()
    check_stop_line_scalar_masks_invalid()
    check_lamp_loss_skips_unknown()
    check_stop_line_head_is_training_only()
    print("test_intersection_labels: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
