#!/usr/bin/env python3
"""Train-only stop/yield labels from the BeamNG scenetree. No live path, no GPU."""
from __future__ import annotations

import ast
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EGO_POS = (0.0, 0.0, 0.0)
EGO_DIR = (0.0, 1.0, 0.0)


def _live_files() -> list[Path]:
    targets = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "control" / "actuate.py",
        ROOT / "python" / "planning" / "path_predictor.py",
    ]
    targets.extend(sorted((ROOT / "python" / "perception").glob("*.py")))
    targets.extend(sorted((ROOT / "python" / "viz").glob("*.py")))
    return targets


def check_live_path_never_reads_static_signs() -> None:
    """The supervisor and the visualizer do not import or name the scenetree source."""
    banned = "static_signs"
    for path in _live_files():
        text = path.read_text(encoding="utf-8")
        assert banned not in text, f"{path.relative_to(ROOT)} mentions {banned}"
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            for name in names:
                assert banned not in name.lower(), f"{path.relative_to(ROOT)} imports {name}"
    trainer = (ROOT / "python" / "train" / "train_scene.py").read_text(encoding="utf-8")
    assert "static_signs" not in trainer
    assert "record_static_signs" not in (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    print(f"  live path: {len(_live_files())} files do not import or read static_signs")


def check_scenetree_fixture_is_ego_frame() -> None:
    from python.collect.static_signs import labels_from_scenetree

    objects = [
        {
            "class": "TSStatic",
            "name": "corner_a",
            "shapeName": "/art/shapes/objects/s_sign_stop.dae",
            "pos": [2.0, 15.0, 1.2],
        },
        {
            "class": "TSStatic",
            "name": "merge_yield",
            "shapeName": "/levels/west_coast_usa/art/shapes/signs/yield_sign.dae",
            "position": [4.0, 30.0, 0.0],
        },
        {
            "class": "TSStatic",
            "name": "drift",
            "shapeName": "/art/shapes/objects/s_sign_drift.dae",
            "pos": [1.0, 8.0, 0.0],
        },
        {
            "class": "TSStatic",
            "name": "lamp",
            "shapeName": "/art/shapes/signals/stop_light.dae",
            "pos": [0.0, 12.0, 5.0],
        },
        {
            "class": "PointLight",
            "name": "stop_sign",
            "shapeName": "/art/shapes/objects/s_sign_stop.dae",
            "pos": [0.0, 3.0, 0.0],
        },
    ]
    labels = labels_from_scenetree(objects, EGO_POS, EGO_DIR)
    by_cls = {row["cls"]: row for row in labels}
    assert set(by_cls) == {"stop_sign", "yield"}, labels
    stop = by_cls["stop_sign"]
    assert stop["state"] == "unknown" and stop["src"] == "scenetree"
    assert abs(float(stop["x"]) - 2.0) < 1e-6 and abs(float(stop["y"]) - 15.0) < 1e-6
    assert by_cls["yield"]["state"] == "unknown"
    assert abs(float(by_cls["yield"]["x"]) - 4.0) < 1e-6
    assert abs(float(by_cls["yield"]["y"]) - 30.0) < 1e-6
    # Facing +X, the same stop is ahead, not to the right.
    turned = labels_from_scenetree(objects[:1], EGO_POS, (1.0, 0.0, 0.0))
    assert abs(float(turned[0]["x"]) - (-15.0)) < 1e-6, turned
    assert abs(float(turned[0]["y"]) - 2.0) < 1e-6, turned
    print("  scenetree fixture: stop and yield in ego frame; drift, light, and non-TSStatic dropped")


def check_dedupe_against_signals_api() -> None:
    from python.collect.signal_labels import labels_from_signals
    from python.collect.static_signs import DEDUPE_M, labels_from_scenetree, merge_sign_labels

    signals = labels_from_signals(
        [
            {"name": "stop_sign", "pos": [0.0, 20.0, 0.0]},
            {"cls": "traffic_light", "pos": [2.0, 10.0, 0.0], "state": "red"},
        ],
        EGO_POS,
        EGO_DIR,
    )
    static = labels_from_scenetree(
        [
            {"class": "TSStatic", "shapeName": "s_sign_stop.dae", "pos": [0.4, 21.0, 0.0]},
            {"class": "TSStatic", "shapeName": "s_sign_stop.dae", "pos": [0.0, 20.0 + DEDUPE_M + 0.5, 0.0]},
            {"class": "TSStatic", "shapeName": "yield_sign.dae", "pos": [0.0, 20.0, 0.0]},
        ],
        EGO_POS,
        EGO_DIR,
    )
    merged = merge_sign_labels(signals, static)
    stops = [row for row in merged if row["cls"] == "stop_sign"]
    assert len(stops) == 2, merged
    assert any(row.get("src") == "traffic_signals" for row in stops)
    assert any(abs(float(row["y"]) - 22.5) < 1e-6 for row in stops)
    assert not any(abs(float(row["y"]) - 21.0) < 1e-6 and row.get("src") == "scenetree" for row in stops)
    yields = [row for row in merged if row["cls"] == "yield"]
    assert len(yields) == 1 and yields[0]["src"] == "scenetree"
    lights = [row for row in merged if row["cls"] == "traffic_light"]
    assert len(lights) == 1 and lights[0]["state"] == "red" and lights[0]["src"] == "traffic_signals"
    print("  dedupe: a signals stop within 2 m drops the scenetree twin; a farther stop and a yield stay")


def check_zero_match_writes_no_signs() -> None:
    from python.collect.static_signs import record_static_signs

    class _Empty:
        def queue_lua_command(self, command: str, response: bool = False):  # noqa: ANN201
            assert "TSStatic" in command and "s_sign_stop" in command
            return json.dumps({"ok": True, "objects": []})

    buf = io.StringIO()
    with redirect_stdout(buf):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "static_signs.json"
            catalog = record_static_signs(_Empty(), "gridmap_v2", path)
            written = json.loads(path.read_text(encoding="utf-8"))
            from python.collect.signal_labels import label_signs
            from python.collect.static_signs import merge_static_into_rows

            state = [{"t": 1.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0], "tags": {"map": "gridmap_v2"}}]
            signals = [{"t": 1.0, "signals": [{"cls": "traffic_light", "pos": [2.0, 10.0, 0.0], "state": "red"}]}]
            merged = merge_static_into_rows(label_signs(state, signals), state, catalog)
    text = buf.getvalue()
    assert "[GVD] stop-sign source: 0 matches on gridmap_v2" in text, text
    entry = written["maps"]["gridmap_v2"]
    assert entry["matches"] == 0 and entry["signs"] == []
    assert "err" not in entry and "absent" not in entry
    signs = merged[0]["signs"]
    assert [row["cls"] for row in signs] == ["traffic_light"]
    assert all(row.get("src") != "scenetree" for row in signs)
    assert "static_signs" not in merged[0]
    print("  zero-match map logs the empty line and writes no scenetree signs")


def check_enumeration_failure_is_a_marker() -> None:
    from python.collect.static_signs import merge_static_into_rows, record_static_signs

    class _Dead:
        def queue_lua_command(self, command: str, response: bool = False):  # noqa: ANN201
            del command, response
            raise ConnectionError("socket closed")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "static_signs.json"
        catalog = record_static_signs(_Dead(), "italy", path)
        entry = catalog["maps"]["italy"]
        assert entry["absent"] is True and entry["signs"] == [] and entry["err"]
        state = [{"t": 1.0, "pos": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0], "tags": {"map": "italy"}}]
        from python.collect.signal_labels import label_signs

        base = label_signs(
            state,
            [{"t": 1.0, "signals": [{"cls": "traffic_light", "pos": [1.0, 8.0, 0.0], "state": "green"}]}],
        )
        merged = merge_static_into_rows(base, state, catalog)
        assert merged[0]["static_signs"]["absent"] is True
        assert merged[0]["static_signs"]["err"]
        assert [row["cls"] for row in merged[0]["signs"]] == ["traffic_light"]
        assert all(row.get("src") != "scenetree" for row in merged[0]["signs"])
    print("  enumeration failure writes an absent marker and no invented signs")


def check_sign_class_counts_show_zero_stops() -> None:
    import tempfile

    from python.train.intersection_labels import attach_sign_labels, sign_counts_line

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            {
                "t": 1.0,
                "signs": [
                    {"cls": "traffic_light", "x": 2.0, "y": 10.0, "state": "red"},
                    {"cls": "stop_sign", "x": 0.0, "y": 20.0, "state": "unknown"},
                    {"cls": "stop_sign", "x": 1.0, "y": 22.0, "state": "unknown"},
                ],
            },
            {"t": 2.0, "signs": [{"cls": "yield", "x": 0.0, "y": 5.0, "state": "unknown"}]},
            {"t": 3.0, "err": "missing signals"},
        ]
        with (root / "signs.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        joined = [{"t": 1.0, "source_root": str(root)}, {"t": 2.0, "source_root": str(root)}, {"t": 3.0, "source_root": str(root)}]
        counts = attach_sign_labels(joined)
        line = sign_counts_line(counts)
        fields = dict(part.split("=", 1) for part in line.removeprefix("[GVD] signs: ").split())
        assert fields["parsed"] == "2" and fields["masked"] == "1"
        assert fields["stop_sign"] == "2" and fields["yield"] == "1"
        assert fields["traffic_light"] == "1" and fields["speed_limit"] == "0"
        lights_only = root / "lights"
        lights_only.mkdir()
        (lights_only / "signs.jsonl").write_text(
            json.dumps({"t": 1.0, "signs": [{"cls": "traffic_light", "x": 0.0, "y": 4.0, "state": "red"}]}) + "\n",
            encoding="utf-8",
        )
        zero = sign_counts_line(attach_sign_labels([{"t": 1.0, "source_root": str(lights_only)}]))
        zfields = dict(part.split("=", 1) for part in zero.removeprefix("[GVD] signs: ").split())
        assert zfields["stop_sign"] == "0" and zfields["yield"] == "0" and zfields["speed_limit"] == "0"
        assert zfields["traffic_light"] == "1"
    print(f"  trainer line names every sign class ({line})")


def check_export_merges_static_catalog() -> None:
    import tempfile

    import export_signal_labels as ex

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state = {
            "t": 1.0,
            "pos": [0.0, 0.0, 0.0],
            "dir": [0.0, 1.0, 0.0],
            "tags": {"map": "west_coast_usa"},
        }
        (root / "state.jsonl").write_text(json.dumps(state) + "\n", encoding="utf-8")
        signals = {
            "t": 1.0,
            "signals": [{"cls": "traffic_light", "pos": [2.0, 10.0, 0.0], "state": "red"}],
        }
        (root / "signals.jsonl").write_text(json.dumps(signals) + "\n", encoding="utf-8")
        catalog = {
            "src": "scenetree",
            "maps": {
                "west_coast_usa": {
                    "matches": 1,
                    "signs": [
                        {
                            "cls": "stop_sign",
                            "name": "corner",
                            "pos": [0.0, 20.0, 1.0],
                            "shapeName": "/art/shapes/objects/s_sign_stop.dae",
                        }
                    ],
                }
            },
        }
        (root / "static_signs.json").write_text(json.dumps(catalog), encoding="utf-8")
        code = ex.main([str(root), "--signals", str(root / "signals.jsonl"), "--out", "signs.jsonl"])
        assert code == 0, code
        row = json.loads((root / "signs.jsonl").read_text(encoding="utf-8").splitlines()[0])
        by_cls = {sign["cls"]: sign for sign in row["signs"]}
        assert by_cls["traffic_light"]["state"] == "red"
        assert by_cls["stop_sign"]["src"] == "scenetree"
        assert by_cls["stop_sign"]["state"] == "unknown"
        assert abs(float(by_cls["stop_sign"]["y"]) - 20.0) < 1e-6
    print("  export writes the scenetree stop into signs.jsonl beside the signal light")


def check_recorder_is_wired() -> None:
    text = (ROOT / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8")
    assert "record_static_signs" in text
    print("  collector calls record_static_signs")


def main() -> None:
    check_live_path_never_reads_static_signs()
    check_scenetree_fixture_is_ego_frame()
    check_dedupe_against_signals_api()
    check_zero_match_writes_no_signs()
    check_enumeration_failure_is_a_marker()
    check_sign_class_counts_show_zero_stops()
    check_export_merges_static_catalog()
    check_recorder_is_wired()
    print("test_static_signs: OK")


if __name__ == "__main__":
    main()
