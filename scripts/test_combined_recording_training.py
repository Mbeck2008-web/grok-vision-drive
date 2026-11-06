"""Combined multi-recording training: strip list, namespaced ids, holdout, cache."""

from __future__ import annotations

import sys
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def _png_bytes(colour=(12, 24, 36)) -> bytes:
    import cv2

    img = np.zeros((32, 40, 3), dtype=np.uint8)
    img[:, :] = colour
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _write_recording(root: Path, *, n: int, session: str, t0: float = 0.0, colour=(12, 24, 36)) -> None:
    cam = root / "raw" / "main"
    cam.mkdir(parents=True)
    blob = _png_bytes(colour)
    lines = []
    for i in range(n):
        (cam / f"{i:06d}.png").write_bytes(blob)
        lines.append(
            {
                "i": i,
                "t": float(t0 + i),
                "dt_s": None if i == 0 else 1.0,
                "session": session,
                "cams": {"main": {"file": f"raw/main/{i:06d}.png"}},
            }
        )
    (root / "state.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")


def check_parse_strips_arg() -> None:
    from python.train.strip_sources import parse_strips_arg

    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a" / "ctl_same"
        b = Path(tmp) / "b" / "ctl_same"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        (a / "state.jsonl").write_text("{}\n", encoding="utf-8")
        (b / "state.jsonl").write_text("{}\n", encoding="utf-8")
        one = parse_strips_arg([str(a)])
        assert len(one) == 1 and one[0].resolve() == a.resolve()
        two = parse_strips_arg([str(a), str(b)])
        assert len(two) == 2
        joined = parse_strips_arg([f"{a},{b}"])
        assert len(joined) == 2
        try:
            parse_strips_arg([str(a), str(a)])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate resolved paths must raise")
    print("parse_strips_arg")


def check_argparse_two_strips_tokens() -> None:
    from python.train.train_scene import build_parser

    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "prior" / "ctl_20261105_p01"
        b = Path(tmp) / "new" / "ctl_20261105_p01"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        ns = parser.parse_args(["--strips", str(a), "--strips", str(b), "--holdout-blocks", "5", "--dry-run", "--no-panel"])
        assert ns.strips == [str(a), str(b)]
    # Comma form still one token.
    ns2 = parser.parse_args(["--strips", "X,Y", "--dry-run", "--no-panel"])
    assert ns2.strips == ["X,Y"]
    print("argparse_two_strips_tokens")


def check_namespaced_rows_no_collision() -> None:
    from python.train.strip_sources import load_multi_source_rows, source_key_for

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "sgt" / "ctl_20261105_p01"
        new = Path(tmp) / "sgt_r2" / "ctl_20261105_p01"
        _write_recording(prior, n=4, session="same-session-id", colour=(10, 0, 0))
        _write_recording(new, n=4, session="same-session-id", colour=(0, 10, 0))
        rows, meta = load_multi_source_rows((prior, new), holdout_source=prior)
        assert meta["holdout_key"] == source_key_for(prior)
        assert meta["train_only_keys"] == (source_key_for(new),)
        keys = {r["source"] for r in rows}
        assert len(keys) == 2
        # Same i appears twice under different sources.
        assert sum(1 for r in rows if int(r["i"]) == 0) == 2
        sessions = {r["session"] for r in rows}
        assert len(sessions) == 2
        for r in rows:
            assert Path(r["source_root"]).resolve() in {prior.resolve(), new.resolve()}
            assert r["session"].startswith(r["source"] + ":")
    print("namespaced_rows_no_collision")


def check_holdout_baseline_identical() -> None:
    """Prior-only + holdout-blocks 5 → same holdout i set as single-root pass_plan."""
    from python.train.continual import pass_plan
    from python.train.strip_sources import combined_pass_plan, load_multi_source_rows
    from python.data.raw_frames import load_scene_rows

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "prior" / "ctl_20261105_p01"
        # Long enough for block split to matter (timestamps 0..N-1).
        _write_recording(prior, n=200, session="sess-a", t0=1000.0)
        single = load_scene_rows(prior)
        baseline = pass_plan(single, 16, 8, blocks=5)
        base_hold = [int(r["i"]) for item in baseline if item["kind"] == "holdout" for r in item["rows"]]
        assert base_hold, "fixture must produce a non-empty holdout"
        rows, _meta = load_multi_source_rows((prior,), holdout_source=prior)
        # Single-root multi loader must not rewrite session (preserve baseline).
        assert all(":" not in str(r.get("session", "")) or False for r in rows) or all(
            r.get("session") == "sess-a" for r in rows
        )
        combined = combined_pass_plan(rows, length=16, stride=8, blocks=5, holdout_key=None)
        # When only one source, holdout_key None means treat as single-root plan.
        comb_hold = [int(r["i"]) for item in combined if item["kind"] == "holdout" for r in item["rows"]]
        assert comb_hold == base_hold, (comb_hold, base_hold)
    print("holdout_baseline_identical")


def check_no_leakage_new_train_only() -> None:
    from python.train.strip_sources import combined_pass_plan, load_multi_source_rows, source_key_for

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "p" / "ctl_20261105_p01"
        new = Path(tmp) / "n" / "ctl_20261105_p01"
        _write_recording(prior, n=200, session="s", t0=0.0)
        _write_recording(new, n=200, session="s", t0=0.0)
        rows, meta = load_multi_source_rows((prior, new), holdout_source=prior)
        plan = combined_pass_plan(
            rows, length=16, stride=8, blocks=5, holdout_key=meta["holdout_key"]
        )
        hold_sources = {r["source"] for item in plan if item["kind"] == "holdout" for r in item["rows"]}
        assert hold_sources, "combined plan must keep a prior holdout"
        assert hold_sources == {meta["holdout_key"]}
        hold_ids = {(r["source"], int(r["i"])) for item in plan if item["kind"] == "holdout" for r in item["rows"]}
        train_ids = {(r["source"], int(r["i"])) for item in plan if item["kind"] == "train" for r in item["rows"]}
        assert hold_ids.isdisjoint(train_ids)
        assert any(r["source"] == source_key_for(new) for item in plan if item["kind"] == "train" for r in item["rows"])
        assert all(
            r["source"] != source_key_for(new)
            for item in plan
            if item["kind"] == "holdout"
            for r in item["rows"]
        )
    print("no_leakage_new_train_only")


def check_cache_multi_source_and_size_guard() -> None:
    from python.train.frame_cache import CACHE_FORMAT, CACHE_FORMAT_MULTI, FrameCache, build_frame_cache, build_multi_frame_cache

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "p" / "ctl_20261105_p01"
        new = Path(tmp) / "n" / "ctl_20261105_p01"
        _write_recording(prior, n=3, session="s", colour=(20, 0, 0))
        _write_recording(new, n=3, session="s", colour=(0, 20, 0))
        # single-root still format-2
        dest1 = Path(tmp) / "cache1"
        m1 = build_frame_cache(prior, dest1, ("main",))
        assert m1["format"] == CACHE_FORMAT
        dest2 = Path(tmp) / "cache2"
        m2 = build_multi_frame_cache((prior, new), dest2, ("main",))
        assert m2["format"] == CACHE_FORMAT_MULTI
        assert len(m2["sources"]) == 2
        for chunk in (dest2 / c["file"] for c in m2["chunks"]):
            assert chunk.stat().st_size < 4 * 1024**3
        from python.train.strip_sources import source_key_for

        pk, nk = source_key_for(prior), source_key_for(new)
        with FrameCache(dest2) as cache:
            assert not cache.stale
            assert cache.has(0, source=pk) and cache.has(0, source=nk)
            a = cache.canvas(0, source=pk)
            b = cache.canvas(0, source=nk)
            assert a is not None and b is not None
            # Different colours → different cached pixels (not colliding on i alone).
            assert int(a[a.shape[0] // 2, a.shape[1] // 2, 0]) != int(b[b.shape[0] // 2, b.shape[1] // 2, 0])
        # Stale when a listed root no longer matches (rename away).
        stale_dir = Path(tmp) / "cache_stale"
        build_multi_frame_cache((prior, new), stale_dir, ("main",))
        # Rewrite manifest root to a missing path.
        man = json.loads((stale_dir / "manifest.json").read_text(encoding="utf-8"))
        man["sources"][0]["root"] = str(Path(tmp) / "missing_prior")
        data = (json.dumps(man, indent=2) + "\n").encode("utf-8")
        (stale_dir / "manifest.json").write_bytes(data)
        import hashlib

        (stale_dir / "manifest.sha256").write_text(f"{hashlib.sha256(data).hexdigest()}  manifest.json\n", encoding="utf-8")
        with FrameCache(stale_dir) as cache:
            # Path mismatch → stale (or refuse has).
            assert cache.stale or not cache.has(0, source=man["sources"][0]["key"])
    print("cache_multi_source_and_size_guard")


def check_single_root_session_untouched() -> None:
    from python.train.strip_sources import load_multi_source_rows

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "only" / "ctl_20261105_p01"
        _write_recording(root, n=5, session="keep-me")
        rows, meta = load_multi_source_rows((root,), holdout_source=root)
        assert meta["multi"] is False
        assert all(r.get("session") == "keep-me" for r in rows)
        assert all("source" not in r for r in rows)
    print("single_root_session_untouched")


def main() -> None:
    check_parse_strips_arg()
    check_argparse_two_strips_tokens()
    check_namespaced_rows_no_collision()
    check_single_root_session_untouched()
    check_holdout_baseline_identical()
    check_no_leakage_new_train_only()
    check_cache_multi_source_and_size_guard()
    print("test_combined_recording_training: OK")


if __name__ == "__main__":
    main()
