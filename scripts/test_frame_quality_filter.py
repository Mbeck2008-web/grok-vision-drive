#!/usr/bin/env python3
"""Training-time frame quality filter: wrong-way drop and road-clip drop.

(1) ``--drop-wrong-way`` and ``--max-road-clip`` default off / -1.
(2) A car pointing against a one-way stored tangent is ``wrong_way``;
    the same pose pointing with the road is ``ok``.
(3) On a two-way road, the opposing lane is ``wrong_way`` and the correct
    (right-hand) side is ``ok``. The centerline and off-network poses are
    ``unknown`` and are not dropped.
(4) ``road_clip_fraction`` uses the lower half only (any RGB channel >= 250).
(5) ``apply_frame_quality`` with both filters off keeps every row; with the
    flags on it drops only wrong-way and over-clipped frames.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import collect_root  # noqa: E402


def _write_dump(path: Path, roads: dict) -> Path:
    payload = {"map": "test", "road_network": roads}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _edge(x: float, y: float, half: float = 3.5, z: float = 0.0) -> dict:
    return {
        "middle": [x, y, z],
        "left": [x, y + half, z],
        "right": [x, y - half, z],
    }


def _along_x(
    road_id: str,
    *,
    oneway: bool,
    half: float = 3.5,
    lanes_left: int = 1,
    lanes_right: int = 1,
    flip: bool = False,
    y: float = 0.0,
    z: float = 0.0,
    reverse_nodes: bool = False,
) -> dict:
    left, right = _edge(0.0, y, half, z), _edge(40.0, y, half, z)
    if reverse_nodes:
        left, right = right, left
    return {
        road_id: {
            "oneWay": "1" if oneway else "0",
            "lanesLeft": lanes_left,
            "lanesRight": lanes_right,
            "drivability": 1,
            "flipDirection": "1" if flip else "0",
            "edges": [left, right],
        }
    }


def check_parser_defaults() -> None:
    from python.train.train_scene import build_parser

    args = build_parser().parse_args([])
    assert args.drop_wrong_way is False
    assert args.max_road_clip == -1.0
    assert args.road_dump == ""
    args = build_parser().parse_args(["--drop-wrong-way", "--max-road-clip", "0.20", "--road-dump", "x.json"])
    assert args.drop_wrong_way is True
    assert args.max_road_clip == 0.20
    assert args.road_dump == "x.json"


def check_wrong_way_oneway() -> None:
    from python.train.frame_quality import classify_wrong_way, load_quality_roads

    with collect_root("gvd-fq-") as tmp:
        dump = _write_dump(tmp / "roads.json", _along_x("ow", oneway=True, half=2.0, lanes_left=0, lanes_right=1))
        roads = load_quality_roads(dump)
        # Stored tangent is +X. Against traffic (heading pi) on the carriageway.
        against = classify_wrong_way(20.0, -0.8, math.pi, roads)
        assert against["verdict"] == "wrong_way", against
        assert against["reason"] == "oneway_against", against
        along = classify_wrong_way(20.0, -0.8, 0.0, roads)
        assert along["verdict"] == "ok", along
        assert along["reason"] == "oneway_with", along


def check_wrong_way_oneway_flip_direction() -> None:
    """Legal travel is opposite node order when flipDirection is true.

    Edges still run +X. Heading 0 (with the stored nodes) is against legal
    traffic; heading pi is with it. Current code that ignores flipDirection
    calls heading 0 ``ok`` and must fail this test.
    """
    from python.train.frame_quality import classify_wrong_way, load_quality_roads

    with collect_root("gvd-fq-") as tmp:
        dump = _write_dump(
            tmp / "roads.json",
            _along_x("ow_flip", oneway=True, half=2.0, lanes_left=0, lanes_right=1, flip=True),
        )
        roads = load_quality_roads(dump)
        against_legal = classify_wrong_way(20.0, -0.8, 0.0, roads)
        assert against_legal["verdict"] == "wrong_way", against_legal
        assert against_legal["reason"] == "oneway_against", against_legal
        with_legal = classify_wrong_way(20.0, -0.8, math.pi, roads)
        assert with_legal["verdict"] == "ok", with_legal
        assert with_legal["reason"] == "oneway_with", with_legal


def check_overpass_does_not_steal_lower_road() -> None:
    """A higher one-way with opposite legal heading must not snap in 2D."""
    from python.train.frame_quality import classify_wrong_way, load_quality_roads

    with collect_root("gvd-fq-") as tmp:
        roads_spec = {}
        roads_spec.update(_along_x("lower", oneway=True, half=2.0, lanes_left=0, lanes_right=1, y=0.0, z=0.0))
        # Closer in XY than the lower road, 8 m above, stored tangent -X.
        roads_spec.update(
            _along_x(
                "upper",
                oneway=True,
                half=2.0,
                lanes_left=0,
                lanes_right=1,
                y=-0.2,
                z=8.0,
                reverse_nodes=True,
            )
        )
        dump = _write_dump(tmp / "roads.json", roads_spec)
        roads = load_quality_roads(dump)
        got = classify_wrong_way(20.0, -0.3, 0.0, roads, z=0.0)
        assert got["verdict"] == "ok", got
        assert got["reason"] == "oneway_with", got
        assert got["road_id"] == "lower", got


def check_apply_frame_quality_defaults() -> None:
    import inspect

    from python.train.frame_quality import apply_frame_quality

    params = inspect.signature(apply_frame_quality).parameters
    assert params["drop_wrong_way"].default is False
    assert params["max_road_clip"].default == -1.0
    assert params["road_dump"].default is None
    assert params["roads"].default is None
    assert params["image_loader"].default is None
    rows = [{"i": 1}, {"i": 2}]
    kept, stats = apply_frame_quality(rows)
    assert [row["i"] for row in kept] == [1, 2]
    assert stats["wrong_way"] == 0
    assert stats["overexposed"] == 0


def check_wrong_way_twoway() -> None:
    from python.train.frame_quality import classify_wrong_way, drop_wrong_way_rows, load_quality_roads

    with collect_root("gvd-fq-") as tmp:
        dump = _write_dump(tmp / "roads.json", _along_x("tw", oneway=False, half=3.5, lanes_left=1, lanes_right=1))
        roads = load_quality_roads(dump)
        correct = classify_wrong_way(20.0, -1.8, 0.0, roads)
        assert correct["verdict"] == "ok", correct
        opposing = classify_wrong_way(20.0, 1.8, 0.0, roads)
        assert opposing["verdict"] == "wrong_way", opposing
        assert opposing["reason"] == "twoway_opposing_lane", opposing
        center = classify_wrong_way(20.0, 0.1, 0.0, roads)
        assert center["verdict"] == "unknown", center
        assert center["reason"] == "twoway_centerline", center
        missing = classify_wrong_way(20.0, 80.0, 0.0, roads)
        assert missing["verdict"] == "unknown", missing
        assert missing["reason"] == "off_network", missing

        rows = [
            {"i": 0, "ego": {"pos": [20.0, -1.8, 0.0], "dir": [1.0, 0.0, 0.0]}},
            {"i": 1, "ego": {"pos": [20.0, 1.8, 0.0], "dir": [1.0, 0.0, 0.0]}},
            {"i": 2, "ego": {"pos": [20.0, 80.0, 0.0], "dir": [1.0, 0.0, 0.0]}},
        ]
        kept, stats = drop_wrong_way_rows(rows, roads)
        assert [r["i"] for r in kept] == [0, 2], (kept, stats)
        assert stats["wrong_way"] == 1
        assert stats["unknown"] == 1
        assert stats["ok"] == 1


def check_road_clip_fraction() -> None:
    from python.train.frame_quality import apply_frame_quality, road_clip_fraction

    clean = np.full((40, 20, 3), 40, dtype=np.uint8)
    assert road_clip_fraction(clean) == 0.0
    upper = clean.copy()
    upper[:20] = 255
    assert road_clip_fraction(upper) == 0.0
    lower = clean.copy()
    lower[20:] = 255
    frac = road_clip_fraction(lower)
    assert frac == 1.0, frac
    assert frac > 0.20

    rows = [
        {"i": 0, "source_root": "a"},
        {"i": 1, "source_root": "a"},
        {"i": 2, "source_root": "a"},
    ]
    frames = {0: clean, 1: upper, 2: lower}

    def loader(root, index):
        return frames[int(index)]

    kept, stats = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=0.20,
        image_loader=loader,
        roads=[],
    )
    assert [r["i"] for r in kept] == [0, 1], (kept, stats)
    assert stats["overexposed"] == 1

    same, off = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=-1.0,
        image_loader=loader,
        roads=[],
    )
    assert [r["i"] for r in same] == [0, 1, 2]
    assert off["overexposed"] == 0
    assert off["wrong_way"] == 0


def check_combined_drop() -> None:
    from python.train.frame_quality import apply_frame_quality, load_quality_roads

    with collect_root("gvd-fq-") as tmp:
        dump = _write_dump(tmp / "roads.json", _along_x("ow", oneway=True, half=2.0, lanes_left=0, lanes_right=1))
        roads = load_quality_roads(dump)
        clean = np.full((10, 10, 3), 40, dtype=np.uint8)
        blown = np.full((10, 10, 3), 255, dtype=np.uint8)
        rows = [
            {"i": 10, "ego": {"pos": [20.0, -0.8, 0.0], "dir": [1.0, 0.0, 0.0]}},   # ok, clean
            {"i": 11, "ego": {"pos": [20.0, -0.8, 0.0], "dir": [-1.0, 0.0, 0.0]}},  # ww, clean
            {"i": 12, "ego": {"pos": [20.0, -0.8, 0.0], "dir": [1.0, 0.0, 0.0]}},   # ok, blown
        ]
        frames = {10: clean, 11: clean, 12: blown}
        kept, stats = apply_frame_quality(
            rows,
            drop_wrong_way=True,
            max_road_clip=0.20,
            image_loader=lambda root, index: frames[int(index)],
            roads=roads,
        )
        assert [r["i"] for r in kept] == [10], (kept, stats)
        assert stats["wrong_way"] == 1
        assert stats["overexposed"] == 1


def main() -> None:
    check_parser_defaults()
    check_apply_frame_quality_defaults()
    check_wrong_way_oneway()
    check_wrong_way_oneway_flip_direction()
    check_overpass_does_not_steal_lower_road()
    check_wrong_way_twoway()
    check_road_clip_fraction()
    check_combined_drop()
    print("test_frame_quality_filter: OK")


if __name__ == "__main__":
    main()
