#!/usr/bin/env python3
"""Junction map and junction labels, offline. No BeamNG, no wall clock in any assert.

Spec tests (``python/collect/junction_map.py``):
 1 synthetic fixture types, one-way arm, no junction on U1/U2, gated arm -> minor
 2 west_coast_usa golden freeze: counts, type histogram, sha256, two builds byte-identical
 3 labels on a scripted track: start inside a circle, monotone ``next_jct_m``, ``in_jct``,
   31 m jump resets, dead end is null, ``jct_exit.jsonl`` exits match the scripted turns
 6 the relabel never reads ``state.jsonl`` while a collector runs (fake clock, patched os.stat)
 7 create-only ``jct.jsonl`` (offline half; the live half is in test_jct_live_parity.py)
 8 isolation (AST): runtime files never import junction_map; labels reach only the sidecar
 9 cost, report only: ``step()`` p50 / p99 over 20k fixture poses (no assert on time)
11 navgraph node classification (major / minor / unmapped, 2.9 m maps, 3.1 m does not)
12 type threshold boundaries
13 overpass: decks more than JOIN_DZ_M apart give no junction
Test 10 is ``scripts/test_junction_grade.py`` (its own suite script); test 4 and 5 are in
``scripts/test_jct_live_parity.py``.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from python.collect import junction_map as jm  # noqa: E402
from synthetic_roads import load_graph, network_spec, track  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
WCU_DUMP = FIXTURES / "roads" / "west_coast_usa.json.gz"
GOLDEN = FIXTURES / "junctions" / "west_coast_usa.golden.json"
COMMITTED_MAP = ROOT / "junctions" / "west_coast_usa.json.gz"


def _synth_map(level: str = "synth_a", mutate=None) -> dict:
    roads = network_spec(load_graph(), level)
    if mutate is not None:
        mutate(roads)
    return jm.build_junction_map(roads, level, "synthetic")


def _at(doc: dict, xy) -> dict:
    hits = [j for j in doc["junctions"] if math.hypot(j["c"][0] - xy[0], j["c"][1] - xy[1]) < 1.0]
    assert len(hits) == 1, (xy, [j["c"] for j in doc["junctions"]])
    return hits[0]


def _rows(level: str, poses, t0: float = 0.0, z: float = 0.5) -> list[dict]:
    return [
        {"t": round(t0 + 0.1 * i, 6), "map": level,
         "ego": {"pos": [x, y, z], "dir": [round(math.cos(h), 9), round(math.sin(h), 9), 0.0]}}
        for i, (x, y, h) in enumerate(poses)
    ]


# ---------------------------------------------------------------- 1


def check_synthetic_types() -> None:
    data = load_graph()
    lvl = data["levels"]["synth_a"]
    doc = _synth_map()
    assert len(doc["junctions"]) == 3, [j["type"] for j in doc["junctions"]]
    want = {"T": "T", "X": "4way", "Y": "Y"}
    for node, kind in want.items():
        j = _at(doc, lvl["nodes"][node])
        assert j["type"] == kind and j["minor"] is False, (node, j)
    xj = _at(doc, lvl["nodes"]["X"])
    ow = {a["road_id"]: a["oneway"] for a in xj["arms"]}
    assert ow == {"T-X": False, "X-XN": True, "XS-X": False, "X-Y": False}, ow
    assert all(a["lanes"] == 2 for a in xj["arms"]), xj["arms"]
    u1 = lvl["nodes"][lvl["unreachable"]]
    assert all(math.hypot(j["c"][0] - u1[0], j["c"][1] - u1[1]) > 100 for j in doc["junctions"])
    # r_m: one node, so the largest arm half-width (2 x 3.5 m lanes -> 7 m road, 3.5 m).
    assert abs(xj["r_m"] - 3.5) < 1e-6, xj["r_m"]
    ids = [j["id"] for j in doc["junctions"]]
    assert ids == sorted(ids) and [j["c"][0] for j in doc["junctions"]] == sorted(j["c"][0] for j in doc["junctions"])

    def gate(roads):
        roads["S-T"]["gatedRoad"] = "1"

    gated = _synth_map(mutate=gate)
    tj = _at(gated, lvl["nodes"]["T"])
    assert tj["minor"] is True and [a["minor"] for a in tj["arms"] if a["road_id"] == "S-T"] == [True], tj
    assert _at(gated, lvl["nodes"]["X"])["minor"] is False

    def low(roads):
        roads["S-T"]["drivability"] = 0.2
        roads["W-T"]["hiddenInNavi"] = "1"  # dropped: T keeps 2 arms and is no junction

    lowd = _synth_map(mutate=low)
    assert all(math.hypot(j["c"][0], j["c"][1]) > 50 for j in lowd["junctions"]), lowd["junctions"]
    print(f"  1 synthetic: T/X/Y typed, one-way arm X-XN, U1-U2 none, gated S-T makes T minor")


# ---------------------------------------------------------------- 2

_WCU: dict = {}


def _wcu_build() -> tuple[dict, bytes]:
    if not _WCU:
        raw = WCU_DUMP.read_bytes()
        data = json.loads(gzip.decompress(raw))
        doc = jm.build_junction_map(data["road_network"], "west_coast_usa", hashlib.sha256(raw).hexdigest())
        _WCU["doc"], _WCU["bytes"] = doc, jm.canonical_bytes(doc)
    return _WCU["doc"], _WCU["bytes"]


def check_golden_freeze() -> None:
    doc, blob = _wcu_build()
    raw = WCU_DUMP.read_bytes()
    again = jm.canonical_bytes(
        jm.build_junction_map(json.loads(gzip.decompress(raw))["road_network"], "west_coast_usa",
                              hashlib.sha256(raw).hexdigest()))
    assert again == blob, "two builds differ"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    summary = jm.summary(doc)
    for key in ("junctions", "major", "minor", "types", "types_major"):
        assert summary[key] == golden["counts"][key], (key, summary[key], golden["counts"][key])
    assert hashlib.sha256(blob).hexdigest() == golden["sha256"], (hashlib.sha256(blob).hexdigest(), golden["sha256"])
    assert golden["src_sha256"] == doc["src_sha256"] == hashlib.sha256(raw).hexdigest()
    for key, value in jm.PARAMS.items():
        assert golden["params"][key] == value, (key, golden["params"][key], value)
    assert golden["params"]["MATCH_M"] == 25.0 and golden["params"]["ARM_MAP_M"] == 3.0
    assert golden["params"]["UNMAPPED_MAX"] == 0.05
    assert golden["bars"] == {"major_recall_min": 0.95, "major_precision_min": 0.90}, golden["bars"]
    # The committed map is the same build, and dump_navgraph's freeze check agrees.
    committed = jm.load_map(COMMITTED_MAP)
    assert jm.canonical_bytes(committed) == blob, "junctions/west_coast_usa.json.gz is not the frozen build"
    assert gzip.decompress(COMMITTED_MAP.read_bytes()) == blob
    import dump_navgraph

    ok, why = dump_navgraph.golden_matches("west_coast_usa", build=lambda: blob)
    assert ok, why
    ok, why = dump_navgraph.golden_matches("west_coast_usa", build=lambda: blob + b" ")
    assert not ok and "sha" in why, why
    print(f"  2 golden: {summary['junctions']} junctions ({summary['major']} major, {summary['minor']} minor) "
          f"types {summary['types']} sha {golden['sha256'][:12]}; two builds identical")


# ---------------------------------------------------------------- 3


def _label(doc: dict, rows: list[dict]) -> tuple[list[dict], "jm.JunctionLabeler"]:
    lab = jm.JunctionLabeler(doc)
    return [lab.step(r) for r in rows], lab


def _approaches(labels: list[dict]) -> list[tuple[list[float], bool]]:
    """``next_jct_m`` runs between circles (rows outside any circle with a junction ahead),
    each with whether the run ends by entering a circle (a cut corner can miss one)."""
    runs, cur = [], []
    for out in labels:
        if out["in_jct"] or out["next_jct_m"] is None:
            if cur:
                runs.append((cur, bool(out["in_jct"])))
            cur = []
        else:
            cur.append(out["next_jct_m"])
    if cur:
        runs.append((cur, False))
    return runs


def check_labels_on_track() -> None:
    data = load_graph()
    lvl = data["levels"]["synth_a"]
    doc = _synth_map()
    ids = {n: _at(doc, lvl["nodes"][n])["id"] for n in ("T", "X", "Y")}
    by_id = {j["id"]: j for j in doc["junctions"]}
    rows = _rows("synth_a", track(data, "synth_a", ["S", "T", "X", "Y", "YL"]))
    labels, lab = _label(doc, rows)
    seen = []
    for row, out in zip(rows, labels):
        assert set(out) == {"t", "in_jct", "jct_id", "next_jct_m", "jct_type", "minor", "src"}, out
        assert out["t"] == row["t"] and out["src"] == jm.map_sha256(doc)
        x, y = row["ego"]["pos"][:2]
        inside = [j for j in doc["junctions"] if math.hypot(x - j["c"][0], y - j["c"][1]) <= j["r_m"]]
        assert out["in_jct"] == bool(inside), (row, out)
        if out["in_jct"]:
            assert out["next_jct_m"] == 0.0 and out["jct_id"] == inside[0]["id"], out
        if out["jct_id"] is not None:
            assert out["jct_type"] == by_id[out["jct_id"]]["type"] and out["minor"] == by_id[out["jct_id"]]["minor"]
            if not seen or seen[-1] != out["jct_id"]:
                seen.append(out["jct_id"])
    assert seen == [ids["T"], ids["X"], ids["Y"]], seen
    runs = _approaches(labels)
    assert len(runs) >= 3, runs
    for run, entered in runs[:3]:
        assert all(b <= a + 1e-9 for a, b in zip(run, run[1:])), run
        if entered:  # 1 m steps: the last row before a circle is under 1.5 m from it
            assert run[-1] < 1.5, run[-1]
    # The right turn at T cuts inside the circle edge; the X and Y entries are driven.
    assert sum(1 for _run, entered in runs if entered) >= 2, [(r[-1], e) for r, e in runs]
    # Capped at 150 m: S is 200 m from T, so the first rows have nothing ahead yet.
    assert labels[0]["next_jct_m"] is None and labels[0]["jct_id"] is None
    assert max(v for run, _e in runs for v in run) <= 150.0
    # Past Y toward the dead end YL: nothing ahead.
    assert labels[-1]["next_jct_m"] is None and labels[-1]["jct_id"] is None and labels[-1]["in_jct"] is False
    assert lab.resets == 0

    # Start inside the T circle: 0 on every in-circle row, then the junction ahead (X), not T.
    start = _rows("synth_a", track(data, "synth_a", ["T", "X", "Y"]))
    lab_s, _ = _label(doc, start)
    assert lab_s[0]["in_jct"] is True and lab_s[0]["jct_id"] == ids["T"]
    k = next(i for i, o in enumerate(lab_s) if not o["in_jct"])
    assert all(o["next_jct_m"] == 0.0 and o["jct_id"] == ids["T"] for o in lab_s[:k])
    nxt = next(o for o in lab_s[k:] if o["jct_id"] is not None)
    assert nxt["jct_id"] == ids["X"], nxt
    assert all(o["jct_id"] != ids["T"] for o in lab_s[k:]), "pointed back at the junction just left"

    # A 31 m jump resets; 29 m does not.
    base = _rows("synth_a", track(data, "synth_a", ["T", "X"]))
    for gap, want in ((31.0, 1), (29.0, 0)):
        jumped = [copy.deepcopy(r) for r in base[:60]]
        tail = copy.deepcopy(base[60])
        tail["ego"]["pos"][0] = round(base[59]["ego"]["pos"][0] + gap, 4)
        tail["t"] = round(base[59]["t"] + 0.1, 6)
        jumped.append(tail)
        _, lab_j = _label(doc, jumped)
        assert lab_j.resets == want, (gap, lab_j.resets)

    # Dead end: leaving T toward W has no junction ahead.
    dead = _rows("synth_a", track(data, "synth_a", ["T", "W"]))
    lab_d, _ = _label(doc, dead)
    after = [o for o in lab_d if not o["in_jct"]]
    assert after and all(o["next_jct_m"] is None and o["jct_id"] is None for o in after), after[:3]

    # No pose: every field but t and src is null.
    blank = jm.JunctionLabeler(doc).step({"t": 1.0, "map": "synth_a", "ego": {"pos": None, "dir": None}})
    assert blank == {"t": 1.0, "in_jct": None, "jct_id": None, "next_jct_m": None, "jct_type": None,
                     "minor": None, "src": jm.map_sha256(doc)}, blank

    # Exits (offline only) match the scripted turns. The exit reads the heading change
    # across the circle alone (r_m 3.5 m here), so these tracks hug the lane through the
    # corner (light smoothing); the default track cuts inside small circles.
    checked = 0
    for case in data["crossings"]:
        rows_c = _rows("synth_a", track(data, "synth_a", case["path"], smooth_half=2))
        labs, _ = _label(doc, rows_c)
        exits = jm.junction_exits(rows_c, labs)
        hit = [e for e in exits if e["jct_id"] == ids[case["node"]]]
        assert len(hit) == 1 and hit[0]["exit"] == case["expect"], (case, exits)
        assert set(hit[0]) == {"jct_id", "t_in", "t_out", "exit"} and hit[0]["t_in"] < hit[0]["t_out"]
        checked += 1
    print(f"  3 track: T->X->Y ahead/in_jct ok, start-in-circle ok, 31 m reset, dead end null, {checked} exits match")


# ---------------------------------------------------------------- 6 / 7


def _write_run(folder: Path, doc: dict, maps_dir: Path) -> list[dict]:
    data = load_graph()
    rows = _rows("synth_a", track(data, "synth_a", ["S", "T", "X"]))
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "state.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    maps_dir.mkdir(parents=True, exist_ok=True)
    jm.write_map(doc, maps_dir)
    return rows


@contextlib.contextmanager
def _opens():
    seen: list[str] = []
    real_open, real_io_open = builtins.open, io.open

    def spy(file, *a, **k):
        seen.append(str(file))
        return real_open(file, *a, **k)

    builtins.open = spy  # type: ignore[assignment]
    io.open = spy  # type: ignore[assignment]
    try:
        yield seen
    finally:
        builtins.open = real_open  # type: ignore[assignment]
        io.open = real_io_open  # type: ignore[assignment]


def _relabel(argv: list[str], now: float) -> tuple[int, str]:
    import label_junctions

    buf = io.StringIO()
    old_now = label_junctions.now_s
    label_junctions.now_s = lambda: now  # type: ignore[assignment]
    try:
        with contextlib.redirect_stdout(buf):
            code = label_junctions.main(argv)
    finally:
        label_junctions.now_s = old_now  # type: ignore[assignment]
    return code, buf.getvalue()


def check_never_read_while_running() -> None:
    doc = _synth_map()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        run, maps, status = tmp_p / "run", tmp_p / "maps", tmp_p / "status"
        _write_run(run, doc, maps)
        status.mkdir()
        now = 1_900_000_000.0
        state = run / "state.jsonl"
        real_stat = os.stat
        argv = [str(run), "--junctions", str(maps), "--status-dir", str(status)]

        def fake_stat(path, *a, **k):
            st = real_stat(path, *a, **k)
            if Path(path).name == "state.jsonl":
                vals = list(st)
                vals[8] = int(now - fake_stat.age)  # st_mtime
                return os.stat_result(vals)
            return st

        os.stat = fake_stat  # type: ignore[assignment]
        try:
            # Fresh running heartbeat (59 s old): exit 3, no state file opened.
            fake_stat.age = 10_000.0
            hb = status / "collect_heartbeat.json"
            hb.write_text(json.dumps({"kind": "collect_heartbeat", "state": "running", "t": now - 59.0}), encoding="utf-8")
            with _opens() as seen:
                code, text = _relabel(argv, now)
            assert code == 3 and not any(p.endswith("state.jsonl") for p in seen), (code, seen, text)
            assert not (run / "jct.jsonl").exists()
            # Fresh state.jsonl mtime (119 s) with an old heartbeat: exit 3 too.
            hb.write_text(json.dumps({"kind": "collect_heartbeat", "state": "running", "t": now - 61.0}), encoding="utf-8")
            fake_stat.age = 119.0
            with _opens() as seen:
                code, text = _relabel(argv, now)
            assert code == 3 and not any(p.endswith("state.jsonl") for p in seen), (code, seen, text)
            # A stopped heartbeat and a 121 s old file: labels are written.
            hb.write_text(json.dumps({"kind": "collect_heartbeat", "state": "stopped", "t": now - 1.0}), encoding="utf-8")
            fake_stat.age = 121.0
            code, text = _relabel(argv, now)
            assert code == 0, (code, text)
        finally:
            os.stat = real_stat  # type: ignore[assignment]
        lines = (run / "jct.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(state.read_text(encoding="utf-8").splitlines())
        assert (run / "jct_exit.jsonl").is_file()
    print("  6 relabel: fresh heartbeat or fresh state.jsonl -> exit 3, no state file opened; stale -> labels")


def check_create_only_offline() -> None:
    doc = _synth_map()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        run, maps = tmp_p / "run", tmp_p / "maps"
        _write_run(run, doc, maps)
        (run / "jct.jsonl").write_bytes(b"keep\n")
        code, text = _relabel([str(run), "--junctions", str(maps), "--status-dir", str(tmp_p / "none")], 2e9)
        assert code == 0 and "exists" in text, text
        assert (run / "jct.jsonl").read_bytes() == b"keep\n"
        # The sidecar itself refuses an existing file and never writes.
        side = jm.JctSidecar(run / "jct.jsonl", maps, log=lambda s: None)
        side.prepare("synth_a")
        side.commit(side.step({"t": 0.0, "map": "synth_a", "ego": {"pos": [0, 0, 0], "dir": [1, 0, 0]}}))
        side.close()
        assert (run / "jct.jsonl").read_bytes() == b"keep\n"
        # Map writer: create-only too.
        try:
            jm.write_map(doc, maps)
        except FileExistsError:
            pass
        else:
            raise AssertionError("write_map overwrote a junction map")
    print("  7 create-only (offline): an existing jct.jsonl and junction map are kept")


# ---------------------------------------------------------------- 8


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [node.module or ""] + [alias.name for alias in node.names]
    return names


def check_isolation() -> None:
    targets = [
        ROOT / "python" / "run_vision.py",
        ROOT / "python" / "control" / "actuate.py",
        ROOT / "python" / "planning" / "path_predictor.py",
        ROOT / "python" / "train" / "scene_lane_runtime.py",
        ROOT / "python" / "collect" / "excursion.py",
        ROOT / "python" / "collect" / "spawn_mix.py",
    ]
    for path in targets:
        assert path.is_file(), f"{path.relative_to(ROOT)} moved; update the isolation scan"
    perception = sorted((ROOT / "python" / "perception").glob("*.py"))
    assert perception
    targets += perception
    for path in targets:
        for name in _imports(path):
            assert "junction_map" not in name.lower(), f"{path.relative_to(ROOT)} imports {name}"
    # The collector: the labeler handle and its lines are only used through the sidecar.
    src = ROOT / "scripts" / "collect_scene_lanes.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    uses = {"jct": 0, "jct_line": 0}
    allowed = {"prepare", "step", "commit", "undo", "flush", "close"}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id in uses and isinstance(node.ctx, ast.Load)):
            continue
        uses[node.id] += 1
        parent = parents[node]
        if node.id == "jct":
            ok = (isinstance(parent, ast.Attribute) and parent.attr in allowed) or (
                isinstance(parent, ast.Compare))  # "jct is not None"
            assert ok, f"collect_scene_lanes.py:{node.lineno} uses the labeler outside the sidecar"
        else:
            call = parent
            ok = (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                  and call.func.attr == "commit" and isinstance(call.func.value, ast.Name) and call.func.value.id == "jct")
            assert ok, f"collect_scene_lanes.py:{node.lineno} passes a label line somewhere other than jct.commit"
    assert uses["jct"] >= 4 and uses["jct_line"] >= 1, uses
    assert any("junction_map" in n for n in _imports(src))
    print(f"  8 isolation: {len(targets)} runtime files import no junction_map; collector labels go only to the sidecar")


# ---------------------------------------------------------------- 9


def check_cost_report() -> None:
    doc, _ = _wcu_build()
    graph = doc["graph"]
    pts = []
    for e in graph["edges"]:
        xy = e["xy"]
        for p, q in zip(xy, xy[1:]):
            pts.append((p[0], p[1], math.atan2(q[1] - p[1], q[0] - p[0])))
    step = max(1, len(pts) // 20000)
    poses = (pts[::step] * 2)[:20000]
    rows = _rows("west_coast_usa", poses)
    lab = jm.JunctionLabeler(doc)
    took = []
    for r in rows:
        t0 = time.perf_counter()
        lab.step(r)
        took.append(time.perf_counter() - t0)
    took.sort()
    p50, p99 = took[len(took) // 2] * 1000.0, took[int(len(took) * 0.99)] * 1000.0
    print(f"  9 cost (report only, no assert): step() over {len(rows)} fixture poses p50 {p50:.3f} ms p99 {p99:.3f} ms")


# ---------------------------------------------------------------- 11


def check_navgraph_classes() -> None:
    fx = json.loads((FIXTURES / "navgraph_synthetic.json").read_text(encoding="utf-8"))
    got = jm.classify_navgraph_nodes(fx["navgraph"], fx["road_network"])
    by = {n["id"]: kind for kind in ("major", "minor", "unmapped") for n in got[kind]}
    assert by == fx["expect"], (by, fx["expect"])
    assert got["candidates"] == 4
    assert all(len(n["pos"]) >= 2 for kind in ("major", "minor", "unmapped") for n in got[kind])
    # Site A's north road is 2.9 m off its link (maps). Moved to 3.1 m it no longer maps.
    moved = copy.deepcopy(fx["road_network"])
    for pt in moved[fx["offset_road"]]["edges"]:
        for k in ("left", "middle", "right"):
            pt[k][0] = round(pt[k][0] + 0.2, 4)
    got2 = jm.classify_navgraph_nodes(fx["navgraph"], moved)
    assert [n["id"] for n in got2["unmapped"]] == ["A0", "D0"], got2
    print("  11 navgraph: major / gated minor / drivability-0.2 minor / unmapped; 2.9 m maps, 3.1 m does not")


# ---------------------------------------------------------------- 12 / 13


def check_type_boundaries() -> None:
    data = load_graph()
    cases = {k: v for k, v in data["boundaries"].items() if k != "about"}
    for name, case in cases.items():
        graph = {"lane_width_m": data["lane_width_m"], "sample_m": data["sample_m"], "levels": {name: case}}
        doc = jm.build_junction_map(network_spec(graph, name), name, "synthetic")
        assert len(doc["junctions"]) == 1, (name, doc["junctions"])
        assert doc["junctions"][0]["type"] == case["expect"], (name, doc["junctions"][0]["type"], case["expect"])
    print(f"  12 type boundaries: {len(cases)} cases (156/204 T, 154/206 Y, one-way 29 merge, 31 not, two-way 29 not)")


def check_overpass() -> None:
    fx = json.loads((FIXTURES / "junctions_overpass.json").read_text(encoding="utf-8"))
    for name, case in fx["cases"].items():
        doc = jm.build_junction_map(case["roads"], name, "synthetic")
        assert len(doc["junctions"]) == case["expect"], (name, doc["junctions"])
    print("  13 overpass: dz 1.6 m and 6 m give 0 junctions, dz 1.4 m gives 1")


def main() -> None:
    check_synthetic_types()
    check_type_boundaries()
    check_overpass()
    check_navgraph_classes()
    check_labels_on_track()
    check_never_read_while_running()
    check_create_only_offline()
    check_isolation()
    check_golden_freeze()
    check_cost_report()
    print("test_junction_map: OK")


if __name__ == "__main__":
    main()
