#!/usr/bin/env python3
"""Train-only sign/lamp GT from traffic_signals. No planner, no viz, no GPU."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def check_isolation() -> None:
    banned = ("signal_labels", "export_signal_labels", "traffic_signals")
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
                assert not any(b in low for b in banned), f"{path.relative_to(ROOT)} imports {name}"
    print(f"  isolation: {len(targets)} runtime files import no signal_labels or traffic_signals")


def check_missing_api_is_masked() -> None:
    from python.collect.signal_labels import read_map_node_signals, sidecar_row

    raw, err = read_map_node_signals(object())
    assert raw is None and err
    row = sidecar_row(t=1.5, raw=raw, err=err, ego_pos=None, ego_dir=None)
    assert row == {"err": err, "t": 1.5}
    print("  missing traffic_signals -> err row, no invented lamps")


def check_labels_from_fake_signals() -> None:
    from python.collect.signal_labels import read_map_node_signals, sidecar_row

    class _API:
        @staticmethod
        def get_map_node_signals():
            return [
                {"cls": "traffic_light", "pos": [2.0, 10.0, 0.0], "state": "red"},
                {"name": "stop_sign", "pos": [0.0, 20.0, 0.0]},
                {"cls": "speed_limit", "pos": [-3.0, 8.0, 0.0], "state": "green"},
            ]

    class _Bng:
        traffic_signals = _API()

    raw, err = read_map_node_signals(_Bng())
    assert err is None and raw
    row = sidecar_row(
        t=2.0,
        raw=raw,
        err=None,
        ego_pos=(0.0, 0.0, 0.0),
        ego_dir=(0.0, 1.0, 0.0),
    )
    assert row["t"] == 2.0 and "err" not in row
    signs = {s["cls"]: s for s in row["signs"]}
    assert signs["traffic_light"]["state"] == "red"
    assert signs["traffic_light"]["src"] == "traffic_signals"
    assert abs(float(signs["traffic_light"]["y"]) - 10.0) < 0.2
    assert abs(float(signs["traffic_light"]["x"]) - 2.0) < 0.2
    assert signs["stop_sign"]["state"] == "unknown"
    assert signs["speed_limit"]["state"] == "unknown"  # speed_limit has no lamp
    print("  fake session: red light, unknown stop, speed_limit lamp stays unknown")


def check_empty_network() -> None:
    from python.collect.signal_labels import sidecar_row

    row = sidecar_row(t=0.0, raw=[], err=None, ego_pos=(0, 0, 0), ego_dir=(0, 1, 0))
    assert row == {"t": 0.0, "signs": []}
    print("  empty network -> empty signs, no exception")


DOCUMENTED_MAP_NODE_SIGNALS = {
    "node_a": {
        "node_b": [
            {
                "instance": "trafficLight 1",
                "pos": {"x": 2.0, "y": 10.0, "z": 0.0},
                "state": "red",
                "action": "stop",
            },
            {
                "instance": "stop 2",
                "pos": [0.0, 20.0, 0.0],
            },
        ]
    }
}


def check_documented_map_node_signals_shape() -> None:
    from python.collect.signal_labels import labels_from_signals, sidecar_row

    row = sidecar_row(
        t=1.0,
        raw=DOCUMENTED_MAP_NODE_SIGNALS,
        err=None,
        ego_pos=(0.0, 0.0, 0.0),
        ego_dir=(0.0, 1.0, 0.0),
    )
    assert "err" not in row, row
    signs = {s["cls"]: s for s in row["signs"]}
    light = signs["traffic_light"]
    assert light["state"] == "red"
    assert light["src"] == "traffic_signals"
    assert abs(float(light["x"]) - 2.0) < 0.2, light
    assert abs(float(light["y"]) - 10.0) < 0.2, light
    assert signs["stop_sign"]["state"] == "unknown"
    compact = {
        "n1": {
            "n2": [
                {
                    "instance": "trafficLight1",
                    "pos": {"x": 2.0, "y": 10.0, "z": 0.0},
                    "state": "red",
                }
            ]
        }
    }
    labels = labels_from_signals(compact, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert labels and labels[0]["cls"] == "traffic_light" and labels[0]["state"] == "red", labels
    print("  documented get_map_node_signals shape: red light at x~2 y~10, stop state unknown")


def check_list_instances_trafficLight_1() -> None:
    from python.collect.signal_labels import labels_from_signals

    raw = [
        {"name": "trafficLight 1", "pos": {"x": 2.0, "y": 10.0, "z": 0.0}, "state": "red"},
    ]
    labels = labels_from_signals(raw, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert labels and labels[0]["cls"] == "traffic_light", labels
    assert labels[0]["state"] == "red" and abs(float(labels[0]["x"]) - 2.0) < 0.2
    print("  list_instances-like trafficLight 1 is classified as a red light")


def check_unrecognized_payload_is_err_row() -> None:
    from python.collect.signal_labels import sidecar_row

    for raw in ("not-a-dump", 42):
        row = sidecar_row(
            t=1.0, raw=raw, err=None, ego_pos=(0.0, 0.0, 0.0), ego_dir=(0.0, 1.0, 0.0)
        )
        assert "err" in row and row["t"] == 1.0, (raw, row)
        assert "signs" not in row, row
    print("  unrecognized payload -> err row, not a silent empty signs list")


def check_export_cli() -> None:
    import export_signal_labels as ex

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "state.jsonl").write_text(
            json.dumps({"t": 1.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]}) + "\n",
            encoding="utf-8",
        )
        dump = root / "signals.json"
        dump.write_text(
            json.dumps([{"cls": "yield", "pos": [1.0, 5.0, 0.0], "state": "unknown"}]),
            encoding="utf-8",
        )
        out = root / "signs.jsonl"
        code = ex.main([str(root), "--signals", str(dump), "--out", out.name])
        assert code == 0, code
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["signs"][0]["cls"] == "yield"
        assert rows[0]["signs"][0]["state"] == "unknown"
        assert not (root / "gt.jsonl").exists()
        code = ex.main([str(root), "--signals", str(dump), "--out", "gt.jsonl"])
        assert code == 2
    print("  export CLI writes signs.jsonl and refuses gt.jsonl")


def main() -> None:
    check_isolation()
    check_missing_api_is_masked()
    check_labels_from_fake_signals()
    check_empty_network()
    check_documented_map_node_signals_shape()
    check_list_instances_trafficLight_1()
    check_unrecognized_payload_is_err_row()
    check_export_cli()
    print("test_signal_labels: OK")


if __name__ == "__main__":
    main()
