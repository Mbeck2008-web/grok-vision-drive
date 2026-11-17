#!/usr/bin/env python3
"""Occupancy shards: WebDataset-format tar, read by seek, never loose PNGs.

* round trip: lossless RGB, labels and meta byte-equal; index sidecar offsets
  equal a fresh scan of the tar; a shard without its sidecar still reads.
* the loader refuses a recording folder / a PNG path / an empty folder, and
  reads every sample while any open() / np.fromfile / cv2.imread of a .png
  path raises (no loose-PNG fallback).
* shards roll over at ``max_bytes``; an existing shard name is never reused.
* ``recording_to_shards`` (offline, the one loose-frame reader) turns a tiny
  8-camera recording into shards with recovery flags and labels.
"""
from __future__ import annotations

import builtins
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from python.data import occ_shards  # noqa: E402
from python.data.occ_shards import INDEX_SUFFIX, OccShardDataset, ShardOnlyError, ShardWriter, _scan_index, recording_to_shards  # noqa: E402
from python.perception.occ_geometry import rig_cams  # noqa: E402
from synthetic_occ import SEM_CLASSES, render  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))


def _sample(rng, i):
    imgs = {"main": rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), "rear": rng.integers(0, 255, (18, 32, 3), dtype=np.uint8)}
    lab = {"keys": np.sort(rng.choice(10 ** 6, 50, replace=False)).astype(np.int64), "cls": rng.integers(0, 3, 50).astype(np.uint8),
           "free": rng.normal(size=(40, 3)).astype(np.float32)}
    return imgs, lab, {"t": float(i), "recovery": i % 4 == 0}


def check_roundtrip(tmp: Path) -> None:
    rng = np.random.default_rng(0)
    out = tmp / "sh"
    samples = [_sample(rng, i) for i in range(6)]
    with ShardWriter(out) as w:
        for i, (im, lab, meta) in enumerate(samples):
            w.write(f"k{i:03d}", im, lab, meta)
    paths = w.paths
    assert len(paths) == 1 and paths[0].name == "occ-000000.tar", paths
    ds = OccShardDataset(out)
    assert len(ds) == 6
    assert ds.recovery_flags().tolist() == [True, False, False, False, True, False]
    for i, (im, lab, meta) in enumerate(samples):
        got = ds[i]
        for cam in im:
            assert np.array_equal(got["images"][cam], im[cam]), cam
        assert np.array_equal(got["keys"], lab["keys"]) and got["keys"].dtype == np.int64
        assert np.array_equal(got["cls"], lab["cls"])
        assert np.allclose(got["free"], lab["free"].astype(np.float16).astype(np.float32))
        assert got["meta"]["t"] == meta["t"] and got["meta"]["cams"] == ["main", "rear"]
    idx = json.loads(Path(str(paths[0]) + INDEX_SUFFIX).read_text())["samples"]
    assert idx == _scan_index(paths[0]), "sidecar offsets differ from the tar"
    Path(str(paths[0]) + INDEX_SUFFIX).unlink()
    ds2 = OccShardDataset([paths[0]])
    assert np.array_equal(ds2[3]["keys"], samples[3][1]["keys"]) and ds2.recovery_flags().tolist() == ds.recovery_flags().tolist()
    ds.close(); ds2.close()
    print("shards: lossless round trip; sidecar offsets == tar scan; no-sidecar read OK")


def check_no_loose_png(tmp: Path) -> None:
    rec = tmp / "rec"
    (rec / "raw" / "main").mkdir(parents=True)
    cv2.imwrite(str(rec / "raw" / "main" / "000000.png"), np.zeros((4, 4, 3), np.uint8))
    (tmp / "empty").mkdir()
    for bad in (rec, rec / "raw" / "main" / "000000.png", tmp / "empty", tmp / "missing.tar"):
        try:
            OccShardDataset(bad)
        except ShardOnlyError:
            continue
        raise AssertionError(f"loader accepted {bad}")
    rng = np.random.default_rng(1)
    with ShardWriter(tmp / "sh2") as w:
        for i in range(3):
            w.write(f"k{i}", *_sample(rng, i))
    real_open, real_fromfile, real_imread = builtins.open, np.fromfile, cv2.imread
    calls = []

    def guard(fn):
        def inner(path, *a, **k):
            if str(path).lower().endswith(".png"):
                calls.append(str(path))
                raise AssertionError(f"loose PNG read: {path}")
            return fn(path, *a, **k)
        return inner

    builtins.open, np.fromfile, cv2.imread = guard(real_open), guard(real_fromfile), guard(real_imread)
    try:
        ds = OccShardDataset(tmp / "sh2")
        for i in range(len(ds)):
            assert ds[i]["images"]["main"].shape == (24, 32, 3)
        ds.close()
    finally:
        builtins.open, np.fromfile, cv2.imread = real_open, real_fromfile, real_imread
    assert not calls, calls
    print("shards: folders/PNG paths refused; reads succeed with every .png open blocked OK")


def check_rollover(tmp: Path) -> None:
    rng = np.random.default_rng(2)
    out = tmp / "sh3"
    out.mkdir()
    (out / "occ-000000.tar").write_bytes(b"keep")  # must not be overwritten
    with ShardWriter(out, max_bytes=12_000) as w:
        for i in range(8):
            w.write(f"k{i}", *_sample(rng, i))
    assert (out / "occ-000000.tar").read_bytes() == b"keep"
    assert len(w.paths) >= 3 and all(p.name != "occ-000000.tar" for p in w.paths), w.paths
    ds = OccShardDataset(w.paths)
    assert len(ds) == 8 and [ds[i]["meta"]["t"] for i in range(8)] == [float(i) for i in range(8)]
    ds.close()
    try:
        ShardWriter(tmp / "sh4").write("bad.key", *_sample(rng, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("a '.' in the key must be refused")
    print(f"shards: {len(w.paths)} shards at max_bytes, existing name kept; dotted key refused OK")


def check_recording(tmp: Path) -> None:
    cams = {c.cam_id: c.scaled(64) for c in rig_cams(CFG)}
    rec = tmp / "rec8"
    rows = []
    for i in range(4):
        files = {}
        for cid, cam in cams.items():
            d, s, bgr = render(cam)
            for kind, arr in (("", bgr), ("_depth", d), ("_sem", s)):
                rel = f"raw/{cid}{kind}/{i:06d}.png"
                (rec / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rec / rel), arr)
                files[f"{cid}{kind}"] = {"file": rel}
        tags = {"excursion": {"phase": "recover" if i == 2 else "cruise"}}
        rows.append({"i": i, "t": 0.1 * i, "cams": files, "tags": tags, "ego": {"pos": [0, 0, 0], "dir": [0, 1, 0]}})
    (rec / "state.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (rec / "sem_classes.json").write_text(json.dumps(SEM_CLASSES))
    cfg = dict(CFG)
    cfg["cameras"] = [dict(c, live_res=[64, 48] if c.get("live_res", [0, 0])[1] == 480 else [64, 36]) for c in CFG["cameras"]]
    paths = recording_to_shards(rec, tmp / "rec8_sh", cfg)
    ds = OccShardDataset(paths)
    assert len(ds) == 4 and ds.recovery_flags().tolist() == [False, False, True, False]
    s = ds[0]
    assert sorted(s["images"]) == sorted(cams) and sorted(s["meta"]["label_cams"]) == sorted(cams)
    assert s["keys"].size > 50 and s["free"].shape[0] > 50
    ds.close()
    (tmp / "cams.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert occ_shards.main([str(rec), str(tmp / "rec8_cli"), "--cameras", str(tmp / "cams.yaml"), "--limit", "2"]) == 0
    cli = OccShardDataset(tmp / "rec8_cli")
    assert len(cli) == 2 and np.array_equal(cli[0]["keys"], s["keys"])
    cli.close()
    print(f"recording -> shards: 4 ticks x 8 cams, {s['keys'].size} cells, recovery flag kept, CLI same labels OK")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        check_roundtrip(tmp)
        check_no_loose_png(tmp)
        check_rollover(tmp)
        check_recording(tmp)
    print("test_occ_shards: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
