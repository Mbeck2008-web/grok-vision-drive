#!/usr/bin/env python3
"""Training-time frame quality filter: wrong-way drop and road-clip drop.

(1) Parser defaults turn both filters on (wrong-way drop, ``--max-road-clip``
    0.20). ``--keep-wrong-way`` and ``--max-road-clip -1`` opt out.
    ``--drop-wrong-way`` stays accepted as a compatibility alias.
(2) A car pointing against a one-way stored tangent is ``wrong_way``;
    the same pose pointing with the road is ``ok``.
(3) On a two-way road, the opposing lane is ``wrong_way`` and the correct
    (right-hand) side is ``ok``. The centerline and off-network poses are
    ``unknown`` and are not dropped.
(4) ``road_clip_fraction`` uses the lower half only (any RGB channel >= 250).
(5) ``apply_frame_quality`` with both filters off keeps every row; with the
    flags on it drops only wrong-way and over-clipped frames.
(6) A missing road dump warns once and keeps wrong-way frames unless the
    user passed ``--drop-wrong-way`` (``require_road_dump``), which fails
    loudly. The warning does not claim every frame was kept when clip still
    runs at the default T=0.20.
(7) Clip is the max of per-cam lower-half fractions on the selected training
    cameras. A fully washed main sector is dropped at T=0.20 even when the
    other cameras are black or absent (the old full-canvas fraction cannot
    trip T). A second build with a sidecar does not re-decode.
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


def _flag(parser, option: str):
    for action in parser._actions:
        if option in action.option_strings:
            return action
    raise AssertionError(f"missing flag {option}")


def check_parser_defaults() -> None:
    from python.train.frame_quality import DEFAULT_MAX_ROAD_CLIP
    from python.train.train_scene import build_parser, frame_quality_from_args

    parser = build_parser()
    # Mutation of these defaults must die: filters are on unless opted out.
    assert DEFAULT_MAX_ROAD_CLIP == 0.20
    assert _flag(parser, "--max-road-clip").default == DEFAULT_MAX_ROAD_CLIP
    assert _flag(parser, "--keep-wrong-way").default is False
    assert _flag(parser, "--drop-wrong-way").default is False

    args = parser.parse_args([])
    assert args.keep_wrong_way is False
    assert args.drop_wrong_way is False
    assert args.max_road_clip == DEFAULT_MAX_ROAD_CLIP
    assert args.road_dump == ""
    drop, require, clip, dump = frame_quality_from_args(args)
    assert drop is True, "wrong-way drop must be on by default"
    assert require is False
    assert clip == 0.20
    assert dump == ""

    args = parser.parse_args(["--keep-wrong-way", "--max-road-clip", "-1"])
    drop, require, clip, dump = frame_quality_from_args(args)
    assert args.keep_wrong_way is True
    assert drop is False
    assert require is False
    assert clip == -1.0

    args = parser.parse_args(["--drop-wrong-way", "--road-dump", "x.json"])
    drop, require, clip, dump = frame_quality_from_args(args)
    assert args.drop_wrong_way is True
    assert drop is True
    assert require is True
    assert clip == 0.20
    assert dump == "x.json"

    import io
    from contextlib import redirect_stderr

    with redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--drop-wrong-way", "--keep-wrong-way"])
        except SystemExit:
            pass
        else:
            raise AssertionError("expected --drop-wrong-way and --keep-wrong-way to conflict")


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
    assert params["cams"].default is None
    assert params["cache"].default is None
    assert params["cam_loader"].default is None
    assert params["clip_sidecar"].default is None
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
        {"i": 0, "source_root": "a", "source": "s"},
        {"i": 1, "source_root": "a", "source": "s"},
        {"i": 2, "source_root": "a", "source": "s"},
    ]
    frames = {0: clean, 1: upper, 2: lower}

    def cam_loader(root, index, cam):
        assert cam == "main"
        return frames[int(index)]

    kept, stats = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=0.20,
        cams=("main",),
        cam_loader=cam_loader,
        roads=[],
    )
    assert [r["i"] for r in kept] == [0, 1], (kept, stats)
    assert stats["overexposed"] == 1

    same, off = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=-1.0,
        cams=("main",),
        cam_loader=cam_loader,
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
            cams=("main",),
            cam_loader=lambda root, index, cam: frames[int(index)],
            roads=roads,
        )
        assert [r["i"] for r in kept] == [10], (kept, stats)
        assert stats["wrong_way"] == 1
        assert stats["overexposed"] == 1


def check_washed_main_drops_when_other_cams_black() -> None:
    """A fully washed main sector trips T=0.20 even when other cams are black.

    This test must die on the old compose_strip canvas path: a 100% clipped
    main sector is only 256/2272 ≈ 0.113 of that canvas, below T=0.20.
    """
    from python.data.strip_writer import CANVAS_W, SECTOR_W_169, STRIP_ORDER, compose_strip
    from python.train.frame_quality import DEFAULT_MAX_ROAD_CLIP, apply_frame_quality, road_clip_fraction

    main = np.full((192, 256, 3), 255, dtype=np.uint8)
    black_43 = np.zeros((192, 256, 3), dtype=np.uint8)
    black_169 = np.zeros((192, SECTOR_W_169, 3), dtype=np.uint8)
    frames = {}
    for cam in STRIP_ORDER:
        if cam == "main":
            frames[cam] = main
        elif cam in ("wide", "rear"):
            frames[cam] = black_169
        else:
            frames[cam] = black_43
    canvas = compose_strip(frames)
    assert canvas.shape[1] == CANVAS_W
    canvas_frac = road_clip_fraction(canvas)
    assert canvas_frac < 0.20, canvas_frac  # old filter would keep
    assert road_clip_fraction(main) > 0.20

    def cam_loader(root, index, cam):
        return frames[cam]

    rows = [{"i": 0, "source_root": "a", "source": "s"}]
    kept, stats = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=DEFAULT_MAX_ROAD_CLIP,
        cams=("main",),
        cam_loader=cam_loader,
    )
    assert [r["i"] for r in kept] == [], (kept, stats)
    assert stats["overexposed"] == 1

    # All eight selected, others black (frac 0): max still follows the washed main.
    kept_all, stats_all = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=DEFAULT_MAX_ROAD_CLIP,
        cams=None,
        cam_loader=cam_loader,
    )
    assert [r["i"] for r in kept_all] == []
    assert stats_all["overexposed"] == 1


def check_multicam_drop_uses_max() -> None:
    """Multi-cam selections drop on the max per-cam fraction, not main-only."""
    from python.train.frame_quality import CLIP_AGGREGATE, apply_frame_quality

    assert CLIP_AGGREGATE == "max"
    clean = np.full((192, 256, 3), 40, dtype=np.uint8)
    washed_wide = np.full((192, 340, 3), 255, dtype=np.uint8)

    def cam_loader(root, index, cam):
        if cam == "wide":
            return washed_wide
        return clean

    rows = [{"i": 0, "source_root": "a", "source": "s"}]
    keep_main, stats_main = apply_frame_quality(
        rows, drop_wrong_way=False, max_road_clip=0.20, cams=("main",), cam_loader=cam_loader
    )
    assert [r["i"] for r in keep_main] == [0], stats_main
    assert stats_main["overexposed"] == 0

    drop_both, stats_both = apply_frame_quality(
        rows,
        drop_wrong_way=False,
        max_road_clip=0.20,
        cams=("main", "wide"),
        cam_loader=cam_loader,
    )
    assert [r["i"] for r in drop_both] == [], stats_both
    assert stats_both["overexposed"] == 1


def check_clip_skips_depth_sem() -> None:
    from python.data.strip_writer import STRIP_ORDER
    from python.train.frame_quality import apply_frame_quality

    seen: list[str] = []
    clean = np.full((192, 256, 3), 40, dtype=np.uint8)

    def cam_loader(root, index, cam):
        seen.append(cam)
        return clean

    apply_frame_quality(
        [{"i": 0, "source_root": "a", "source": "s"}],
        drop_wrong_way=False,
        max_road_clip=0.20,
        cams=None,
        cam_loader=cam_loader,
    )
    assert "main_depth" not in seen and "main_sem" not in seen, seen
    assert set(seen) == set(STRIP_ORDER), seen


def check_clip_sidecar_skips_second_decode() -> None:
    from python.train.frame_quality import apply_frame_quality

    with collect_root("gvd-fq-clip-") as tmp:
        calls = {"n": 0}
        clean = np.full((192, 256, 3), 40, dtype=np.uint8)

        def cam_loader(root, index, cam):
            calls["n"] += 1
            return clean

        sidecar = tmp / "road_clip_v1.jsonl"
        rows = [{"i": 7, "source_root": str(tmp), "source": "rec"}]
        first, stats1 = apply_frame_quality(
            rows,
            drop_wrong_way=False,
            max_road_clip=0.20,
            cams=("main",),
            cam_loader=cam_loader,
            clip_sidecar=sidecar,
        )
        assert [r["i"] for r in first] == [7]
        assert stats1["clip_decoded"] == 1
        assert calls["n"] == 1
        assert sidecar.is_file()

        second, stats2 = apply_frame_quality(
            rows,
            drop_wrong_way=False,
            max_road_clip=0.20,
            cams=("main",),
            cam_loader=cam_loader,
            clip_sidecar=sidecar,
        )
        assert [r["i"] for r in second] == [7]
        assert stats2["clip_decoded"] == 0
        assert stats2["clip_sidecar_hits"] == 1
        assert calls["n"] == 1, f"second build re-decoded: {calls['n']}"


def check_clip_uses_cache_not_raw() -> None:
    from python.train.frame_quality import apply_frame_quality

    class FakeCache:
        def __init__(self) -> None:
            self.hits = 0
            self.root = Path("/tmp/fake-cache")

        def sector(self, cam, index, source=None):
            self.hits += 1
            assert cam == "main"
            assert int(index) == 3
            assert source == "rec"
            return np.full((192, 256, 3), 40, dtype=np.uint8)

    def cam_loader(root, index, cam):
        raise AssertionError("must not decode raw when the cache has the sector")

    cache = FakeCache()
    kept, stats = apply_frame_quality(
        [{"i": 3, "source_root": "a", "source": "rec"}],
        drop_wrong_way=False,
        max_road_clip=0.20,
        cams=("main",),
        cache=cache,
        cam_loader=cam_loader,
    )
    assert [r["i"] for r in kept] == [3]
    assert cache.hits == 1
    assert stats["clip_cache_hits"] == 1
    assert stats["clip_decoded"] == 0


def check_missing_dump_warn_and_keep() -> None:
    """Missing dump keeps wrong-way frames; default clip still runs."""
    from python.train.frame_quality import DEFAULT_MAX_ROAD_CLIP, apply_frame_quality, describe_quality

    clean = np.full((10, 10, 3), 40, dtype=np.uint8)
    blown = np.full((10, 10, 3), 255, dtype=np.uint8)
    frames = {0: clean, 1: blown}
    rows = [
        {"i": 0, "map": "no_such_map", "ego": {"pos": [20.0, -0.8, 0.0], "dir": [1.0, 0.0, 0.0]}},
        {"i": 1, "map": "no_such_map", "ego": {"pos": [20.0, -0.8, 0.0], "dir": [-1.0, 0.0, 0.0]}},
    ]
    warnings: list[str] = []
    kept, stats = apply_frame_quality(
        rows,
        drop_wrong_way=True,
        require_road_dump=False,
        road_dump="/no/such/road_dump.json",
        max_road_clip=DEFAULT_MAX_ROAD_CLIP,
        cams=("main",),
        cam_loader=lambda root, index, cam: frames[int(index)],
        warn=warnings.append,
    )
    assert [r["i"] for r in kept] == [0], (kept, stats)
    assert stats["wrong_way"] == 0
    assert stats["overexposed"] == 1
    assert stats["dump_missing"] == 1
    assert stats["unknown"] == 2
    assert len(warnings) == 1, warnings
    assert "keeping all" not in warnings[0]
    assert "keeping all 2 frames" not in warnings[0]
    assert "wrong-way" in warnings[0] and "unknown" in warnings[0]
    assert "road-clip filter still runs" in warnings[0]
    assert "T=0.2" in warnings[0]
    assert "Pass --drop-wrong-way to fail" in warnings[0]
    line = describe_quality(stats, 2)
    assert "wrong_way=0" in line
    assert "overexposed=1" in line
    assert "dump_missing=1" in line
    assert "kept=1" in line


def check_missing_dump_explicit_flag_fails() -> None:
    """Explicit --drop-wrong-way (require_road_dump) fails loudly when the dump is missing."""
    from python.train.frame_quality import FrameQualityError, apply_frame_quality

    rows = [{"i": 0, "map": "no_such_map", "ego": {"pos": [20.0, -0.8, 0.0], "dir": [1.0, 0.0, 0.0]}}]
    warnings: list[str] = []
    try:
        apply_frame_quality(
            rows,
            drop_wrong_way=True,
            require_road_dump=True,
            road_dump="/no/such/road_dump.json",
            max_road_clip=-1.0,
            warn=warnings.append,
        )
    except FrameQualityError as exc:
        assert "not found" in str(exc).lower() or "road dump" in str(exc).lower(), exc
    else:
        raise AssertionError("expected FrameQualityError when --drop-wrong-way and the dump is missing")
    assert warnings == []


def main() -> None:
    check_parser_defaults()
    check_apply_frame_quality_defaults()
    check_wrong_way_oneway()
    check_wrong_way_oneway_flip_direction()
    check_overpass_does_not_steal_lower_road()
    check_wrong_way_twoway()
    check_road_clip_fraction()
    check_combined_drop()
    check_washed_main_drops_when_other_cams_black()
    check_multicam_drop_uses_max()
    check_clip_skips_depth_sem()
    check_clip_sidecar_skips_second_decode()
    check_clip_uses_cache_not_raw()
    check_missing_dump_warn_and_keep()
    check_missing_dump_explicit_flag_fails()
    print("test_frame_quality_filter: OK")


if __name__ == "__main__":
    main()
