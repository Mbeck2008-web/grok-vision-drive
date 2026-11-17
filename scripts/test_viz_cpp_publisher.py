#!/usr/bin/env python3
"""C++ viz publisher (python/viz/cpp_publisher.py, docs/specs/viz_cpp.md).

* schema v2: only the pinned top-level keys, JSON, one datagram;
* perception only: the builder reads a fixed whitelist of state keys, and
  GT keys (path_world, debug_gt, *_gt, gt_*) never change the message;
* paint: yellow only when the record carries yellow paint, else white;
* non-blocking: a missing listener or a receiver that never reads costs
  well under a millisecond per publish and never raises;
* fail-safe: a missing gvd_viz, junk state, or a socket that throws leave
  the publisher (and the caller) running; --viz-cpp is off by default.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.viz import cpp_publisher as cp  # noqa: E402

READ_TOP = {"engaged", "ego", "planner", "path_ego", "path_conf", "lanes_ext", "road_edges", "tracks", "signs", "dots", "occ_out"}
GT_KEYS = {
    "path_world": [{"x": 100.0, "y": 200.0}, {"x": 101.0, "y": 240.0}],
    "debug_gt": True,
    "gt_lanes": [{"points": [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 40.0}], "paint": "yellow"}],
    "lanes_gt": [{"points": [{"x": 3.0, "y": 0.0}, {"x": 3.0, "y": 40.0}]}],
    "vehicles_gt": [{"x": 0.0, "y": 10.0}],
    "scene_lane": {"gt": True},
    "labels": {"lanes": []},
}


class Recording(dict):
    """A dict that remembers every key read through get / [] / in."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read: set[str] = set()

    def get(self, key, default=None):  # noqa: D401
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def rich_state(n_tracks: int = 6, n_pts: int = 40) -> dict:
    lanes = []
    for i, x in enumerate((-5.4, -1.8, 1.8, 5.4)):
        rec = {
            "points": [{"x": x, "y": float(y)} for y in range(-10, -10 + 3 * n_pts, 3)],
            "kind": "detected",
            "side": "left" if x < 0 else "right",
            "index": i - 2,
            "style": "unknown",
        }
        if i == 0:
            rec["paint"] = "yellow"
        lanes.append(rec)
    return {
        "engaged": True,
        "ego": {"speed_mps": 13.4, "steer_deg": 2.0},
        "planner": {"cipv_id": 1, "target_v": 15.0, "aeb": "off"},
        "path_ego": [{"x": 0.0, "y": float(y)} for y in range(2, 40, 2)],
        "path_conf": 0.9,
        "lanes_ext": lanes,
        "road_edges": [{"points": [{"x": -7.0, "y": -10.0}, {"x": -7.0, "y": 90.0}], "kind": "predicted", "side": "left"}],
        "tracks": [
            {"id": k, "class": "vehicle", "x": 3.6 * (k % 3 - 1), "y": 8.0 + 6 * k, "yaw": 1.57, "speed_mps": 10.0}
            for k in range(1, n_tracks + 1)
        ],
        "signs": [{"cls": "traffic_light", "x": 0.0, "y": 45.0, "state": "red", "conf": 0.8}],
        "loop_hz": 15.0,
    }



def check_dots() -> None:
    # explicit dots are forwarded 1:1; a missing vertex is not invented
    st = {
        "engaged": False,
        "ego": {"speed_mps": 1.0},
        "dots": [
            {"id": 1, "x": 0.0, "y": 10.0, "cls": "white", "prob": 0.8, "observed": True},
            {"id": 2, "x": 1.0, "y": 10.0, "cls": "curb", "prob": 0.4, "observed": False},
        ],
        "lanes_ext": [{"points": [{"x": 9.0, "y": 0.0}, {"x": 9.0, "y": 40.0}], "paint": "yellow"}],
    }
    msg = cp.build_scene_message(st, seq=1, t=0.0)
    assert len(msg["dots"]) == 2, msg["dots"]
    assert msg["dots"][0]["id"] == 1 and msg["dots"][0]["prob"] == 0.8
    assert msg["dots"][1]["cls"] == "curb" and msg["dots"][1]["observed"] is False
    # polyline vertices become dots when no dots/occ_out given, no mid-segment samples
    st2 = {"lanes_ext": [{"points": [{"x": -1.8, "y": 0.0}, {"x": -1.8, "y": 10.0}], "paint": "white"}]}
    d2 = cp.build_scene_message(st2, seq=1, t=0.0)["dots"]
    assert len(d2) == 2 and all(abs(d["x"] + 1.8) < 1e-6 for d in d2), d2
    assert {round(d["y"], 1) for d in d2} == {0.0, 10.0}
    # occ_out v2: edges[] are red, lanes[] carry paint. Cells are not paint.
    from python.viz.occ_out_v2_contract import make_occ_out_v2
    st3 = {"occ_out": make_occ_out_v2(
        cells={"ijk": [[64, 40, 4], [70, 40, 4]], "cls": [1, 2], "prob": [0.9, 0.8], "observed": [True, False]},
        edges=[{"side": "left", "kind": "curb", "pts": [[-7.0, 0.0], [-7.0, 10.0]],
                "conf": [0.9, 0.8], "observed": [True, False]},
               {"side": "right", "kind": "road_edge", "pts": [[7.0, 0.0], [7.0, 10.0]],
                "conf": [0.7, 0.6], "observed": [True, True]}],
        lanes=[{"paint": "yellow", "pts": [[-1.8, 0.0], [-1.8, 4.0]],
                "conf": [0.6, 0.5], "observed": [True, False]}],
    )}
    d3 = cp.build_scene_message(st3, seq=1, t=0.0)["dots"]
    cls = {(round(d["x"], 1), d["cls"], d["observed"]) for d in d3}
    assert (-7.0, "curb", True) in cls and (-7.0, "curb", False) in cls, d3
    assert (7.0, "edge", True) in cls
    assert (-1.8, "yellow", True) in cls and any(d["x"] == -1.8 and d["y"] == 4.0 and d["observed"] is False for d in d3)
    # occupancy cells (cls 1=drivable, 2=vehicle) must not become beads
    assert not any(d["cls"] in {"white", "yellow"} and abs(d["x"] - 0.0) < 1e-6 for d in d3)
    # invented schema / wrong version yields no dots
    assert cp.build_scene_message({"occ_out": {"v": 99, "type": "occ_out", "edges": []}}, seq=1, t=0.0)["dots"] == []
    assert cp.build_scene_message({"occ_out": {"v": 1, "type": "occ_out"}}, seq=1, t=0.0)["dots"] == []
    fake = {"occ": {"xyz": [[0.0, 5.0], [2.0, 5.0]], "cls": [5, 2], "prob": [0.9, 0.7], "observed": [True, False]}}
    assert cp.build_scene_message({"occ_out": fake}, seq=1, t=0.0)["dots"] == []
    # over-cap: drop lowest-prob / farthest, never interpolate
    many = [{"id": i, "x": float(i), "y": 80.0, "cls": "white", "prob": 0.1 if i else 1.0, "observed": True}
            for i in range(cp.MAX_DOTS + 50)]
    d4 = cp.build_scene_message({"dots": many}, seq=1, t=0.0)["dots"]
    assert len(d4) == cp.MAX_DOTS
    assert any(d.get("id") == 0 and d["prob"] == 1.0 for d in d4)
    print(f"dots: explicit {len(msg['dots'])}, vertices {len(d2)}, occ_out {len(d3)}, cap {len(d4)} OK")


def check_schema() -> None:
    msg = cp.build_scene_message(rich_state(), seq=7, t=1.5)
    assert set(msg) <= cp.MESSAGE_KEYS, set(msg) - cp.MESSAGE_KEYS
    for key in ("v", "type", "seq", "t", "ego", "lanes", "edges", "tracks", "signs", "path"):
        assert key in msg, key
    assert msg["v"] == cp.SCHEMA_VERSION == 2 and msg["type"] == "gvd_scene" and msg["seq"] == 7
    assert msg["lead_id"] == 1 and msg["ego"]["set_mps"] == 15.0 and msg["ego"]["engaged"] is True
    assert len(msg["lanes"]) == 4 and len(msg["tracks"]) == 6 and len(msg["signs"]) == 1
    assert msg["edges"][0]["kind"] == "predicted"
    assert all(len(p) == 2 for ln in msg["lanes"] for p in ln["pts"])
    tr = msg["tracks"][0]
    assert set(tr) <= {"id", "cls", "x", "y", "yaw", "l", "w", "h", "v", "lead"}, tr
    data = cp.encode_message(msg)
    assert data is not None and len(data) <= cp.MAX_DATAGRAM
    assert json.loads(data) == json.loads(json.dumps(msg))
    print(f"schema v2: keys {sorted(msg)} , {len(data)} bytes OK")


def check_perception_only() -> None:
    rec = Recording(rich_state())
    rec.update(GT_KEYS)
    msg_gt = cp.build_scene_message(rec, seq=1, t=0.0)
    extra = rec.read - READ_TOP
    assert not extra, f"builder read non-whitelisted keys: {sorted(extra)}"
    msg_clean = cp.build_scene_message(rich_state(), seq=1, t=0.0)
    assert msg_gt == msg_clean, "GT keys changed the message"
    blob = cp.encode_message(msg_gt).decode()
    for bad in ("path_world", "debug_gt", "gt_lanes", "lanes_gt", "vehicles_gt", "100.0", "200.0"):
        assert bad not in blob, bad
    print(f"perception only: read {sorted(rec.read)}; GT keys ignored OK")


def check_paint() -> None:
    assert cp.lane_paint({"paint": "yellow"}) == "yellow"
    assert cp.lane_paint({"paint": "white"}) == "white"
    # exact key and exact value only: aliases and substring matches are rejected
    for rec in (
        {},
        {"index": -1, "side": "left"},
        {"paint": "white"},
        {"kind": "predicted", "index": 2},
        None,
        {"paint": "not_yellow"},
        {"paint": "Yellow"},
        {"paint_color": "yellow"},
        {"colour": "yellow"},
        {"line_color": "yellow"},
        {"color": "yellow"},
        {"material": "line_yellow"},
        {"yellow": True},
    ):
        assert cp.lane_paint(rec) == "white", rec
    msg = cp.build_scene_message(rich_state())
    assert [ln["paint"] for ln in msg["lanes"]] == ["yellow", "white", "white", "white"]
    # a lane with an invalid paint value is dropped, not emitted as yellow
    st = rich_state()
    st["lanes_ext"] = [
        {"points": [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 10.0}], "paint": "not_yellow"},
        {"points": [{"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 10.0}], "paint": "yellow"},
    ]
    msg2 = cp.build_scene_message(st)
    assert [ln["paint"] for ln in msg2["lanes"]] == ["yellow"]
    print("paint: exact paint=yellow only; aliases and substrings rejected OK")


def check_nonblocking_no_listener() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens on this port now
    pub = cp.CppVizPublisher(port=port, min_interval_s=0.0)
    assert pub._sock is not None and pub._sock.getblocking() is False
    st = rich_state(n_tracks=64, n_pts=120)
    times = []
    for _ in range(400):
        pub.publish(st)
        times.append(pub.last_ms)
    pub.close()
    med = statistics.median(times)
    assert pub.errors == 0, pub.last_error
    assert pub.sent + pub.dropped + pub.too_big == 400, pub.status()
    assert med < 3.0, f"median publish {med:.3f} ms"
    assert max(times) < 50.0, f"max publish {max(times):.1f} ms"
    print(f"no listener: 400 heavy publishes, median {med:.3f} ms, max {max(times):.2f} ms, "
          f"sent={pub.sent} dropped={pub.dropped} OK")


def check_receiver_never_reads() -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    rx.bind(("127.0.0.1", 0))
    pub = cp.CppVizPublisher(port=rx.getsockname()[1], min_interval_s=0.0)
    st = rich_state()
    t0 = time.perf_counter()
    for _ in range(3000):
        pub.publish(st)
    wall = time.perf_counter() - t0
    rx.close()
    pub.close()
    assert pub.errors == 0, pub.last_error
    assert wall < 5.0, f"3000 publishes into a full buffer took {wall:.2f} s"
    print(f"full receiver buffer: 3000 publishes in {wall*1000:.0f} ms, sent={pub.sent} dropped={pub.dropped} OK")


def check_roundtrip_and_rate() -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    pub = cp.CppVizPublisher(port=rx.getsockname()[1])
    assert pub.publish(rich_state(), now=100.0) is True
    got = json.loads(rx.recv(65536))
    assert got["type"] == "gvd_scene" and got["seq"] == 1 and len(got["lanes"]) == 4
    # rate cap: a second publish inside 1/60 s is skipped, after it goes out
    assert pub.publish(rich_state(), now=100.005) is False
    assert pub.publish(rich_state(), now=100.02) is True
    assert json.loads(rx.recv(65536))["seq"] == 2  # skipped ticks use no seq
    rx.close()
    pub.close()
    print("roundtrip: datagram decodes to the built message; 60 Hz cap OK")


def check_oversize() -> None:
    st = rich_state(n_tracks=200, n_pts=400)
    for ln in st["lanes_ext"]:
        ln["points"] = [{"x": 1.23456 + i * 1e-3, "y": float(i) * 0.25} for i in range(2000)]
    st["lanes_ext"] = st["lanes_ext"] * 4
    st["road_edges"] = st["road_edges"] * 8
    msg = cp.build_scene_message(st)
    assert len(msg["tracks"]) == cp.MAX_TRACKS and len(msg["lanes"]) <= cp.MAX_LANES
    assert all(len(ln["pts"]) <= cp.MAX_POINTS for ln in msg["lanes"])
    data = cp.encode_message(msg)
    assert data is not None
    # Pin against the UDP payload limit, not MAX_DATAGRAM (a 1e6 mutant would
    # still pass a comparison to itself).
    assert cp.MAX_DATAGRAM <= cp.UDP_MAX_PAYLOAD == 65507
    assert len(data) <= 65507, len(data)
    assert len(data) <= cp.MAX_DATAGRAM, len(data)
    # even a caller who asks for more than the UDP limit is clamped
    assert len(cp.encode_message(msg, max_bytes=10**6) or b"") <= 65507
    tiny = cp.encode_message(msg, max_bytes=9000)
    assert tiny is None or len(tiny) <= 9000
    print(f"oversize: {len(data)} bytes <= {cp.MAX_DATAGRAM} (<= 65507) OK")


def check_fail_safe() -> None:
    pub = cp.CppVizPublisher(port=9, min_interval_s=0.0)
    assert pub.launch("/nonexistent/dir/gvd_viz") is False
    assert "not found" in pub.note, pub.note
    for junk in ({"tracks": "x", "lanes_ext": [None, 5, {"points": "no"}], "signs": [1], "ego": 3},
                 {"tracks": [{"x": float("nan"), "y": 1}], "path_ego": [[1, 2, 3], "x"]}, {}, None):
        pub.publish(junk)  # must not raise
    assert pub.errors == 0, pub.last_error

    class Boom:
        def sendto(self, *_a):
            raise RuntimeError("boom")

        def close(self):
            pass

    pub._sock = Boom()
    assert pub.publish(rich_state()) is False and pub.errors == 1
    pub.close()
    print("fail-safe: missing exe, junk state, throwing socket -> no raise OK")


def check_flag_default_off() -> None:
    try:
        from python.run_vision import build_parser, start_cpp_viz
    except Exception as exc:  # noqa: BLE001
        from gvd_test_env import skip_check

        skip_check("flag_default_off", f"run_vision import failed here: {exc!r}")
        return
    args = build_parser().parse_args([])
    assert args.viz_cpp is False and args.viz_cpp_port == 47800
    assert start_cpp_viz(args) is None
    args = build_parser().parse_args(["--viz-cpp", "--viz-cpp-exe", "/nonexistent/gvd_viz", "--viz-cpp-port", "47999"])
    pub = start_cpp_viz(args)
    assert pub is not None and pub.addr == ("127.0.0.1", 47999) and "not found" in pub.note
    assert pub.publish(rich_state()) in (True, False) and pub.errors == 0
    pub.close()
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "cpp_viz.publish(st)" in src and "cpp_viz.close()" in src
    print("--viz-cpp: off by default; missing gvd_viz leaves the publisher harmless OK")


def check_schema_strict() -> None:
    # aeb: only off|warn|brake; anything else becomes off (not passed through)
    for raw, want in (("off", "off"), ("warn", "warn"), ("brake", "brake"),
                      ("maybe", "off"), ("WARN", "off"), (True, "off"), (1, "off")):
        st = rich_state()
        st["planner"]["aeb"] = raw
        assert cp.build_scene_message(st)["aeb"] == want, (raw, want)
    # path_conf: default 1.0, clamped to [0, 1]
    st = rich_state()
    del st["path_conf"]
    assert cp.build_scene_message(st)["path_conf"] == 1.0
    st = rich_state(); st["path_conf"] = 1.8
    assert cp.build_scene_message(st)["path_conf"] == 1.0
    st = rich_state(); st["path_conf"] = -0.2
    assert cp.build_scene_message(st)["path_conf"] == 0.0
    st = rich_state(); st["path_conf"] = 0.45
    assert cp.build_scene_message(st)["path_conf"] == 0.45
    # ids: floats and strings are rejected (no truncation)
    st = rich_state()
    st["planner"]["cipv_id"] = 4.9
    assert "lead_id" not in cp.build_scene_message(st)
    st["planner"]["cipv_id"] = "4"
    assert "lead_id" not in cp.build_scene_message(st)
    st["planner"]["cipv_id"] = 4
    assert cp.build_scene_message(st)["lead_id"] == 4
    st = rich_state()
    st["tracks"] = [{"id": 1.9, "class": "vehicle", "x": 0.0, "y": 8.0}]
    assert "id" not in cp.build_scene_message(st)["tracks"][0]
    # kind enum: invalid kind drops the lane / edge
    st = rich_state()
    st["lanes_ext"] = [{"points": [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 10.0}], "kind": "guessed"}]
    assert cp.build_scene_message(st)["lanes"] == []
    st["road_edges"] = [{"points": [{"x": 5.0, "y": 0.0}, {"x": 5.0, "y": 10.0}], "kind": "guessed"}]
    assert cp.build_scene_message(st)["edges"] == []
    # unknown track / sign class dropped; invalid sign state dropped
    st = rich_state()
    st["tracks"] = [{"id": 1, "class": "spaceship", "x": 0.0, "y": 8.0},
                    {"id": 2, "class": "vehicle", "x": 1.0, "y": 8.0}]
    msg = cp.build_scene_message(st)
    assert [t["id"] for t in msg["tracks"]] == [2]
    st = rich_state()
    st["signs"] = [{"cls": "billboard", "x": 0.0, "y": 20.0},
                   {"cls": "traffic_light", "x": 1.0, "y": 20.0, "state": "purple"}]
    assert cp.build_scene_message(st)["signs"] == []
    st["signs"] = [{"cls": "traffic_light", "x": 1.0, "y": 20.0, "state": "red"}]
    assert cp.build_scene_message(st)["signs"][0]["state"] == "red"
    # live publisher does not invent track.lead, but passes it through when set
    st = rich_state()
    assert all("lead" not in t for t in cp.build_scene_message(st)["tracks"])
    st["tracks"][0]["lead"] = True
    assert cp.build_scene_message(st)["tracks"][0].get("lead") is True
    # on-wire keys are exact: no class/points/length aliases in the datagram
    st = rich_state()
    msg = cp.build_scene_message(st)
    blob = json.dumps(msg)
    assert "paint_color" not in blob and "line_color" not in blob and '"points"' not in blob
    for tr in msg["tracks"]:
        assert "class" not in tr and "length" not in tr and "width" not in tr
        assert "height" not in tr and "speed_mps" not in tr and "pts" not in tr
    for ln in msg["lanes"]:
        assert "points" not in ln and "pts" in ln
    # caps on the publisher side (tracks, signs, edges)
    st = rich_state(n_tracks=80)
    assert len(cp.build_scene_message(st)["tracks"]) == cp.MAX_TRACKS == 64
    st = rich_state()
    st["signs"] = [{"cls": "traffic_light", "x": float(i), "y": 20.0, "state": "red"} for i in range(20)]
    signs = cp.build_scene_message(st)["signs"]
    assert len(signs) == cp.MAX_SIGNS == 16
    assert signs[0]["x"] == 0.0 and signs[-1]["x"] == 15.0
    st = rich_state()
    st["road_edges"] = [
        {"points": [{"x": float(i), "y": 0.0}, {"x": float(i), "y": 10.0}]} for i in range(12)
    ]
    edges = cp.build_scene_message(st)["edges"]
    assert len(edges) == cp.MAX_EDGES == 8
    assert edges[0]["pts"][0][0] == 0.0 and edges[-1]["pts"][0][0] == 7.0
    print("schema strict: enums, ints, caps, path_conf, no aliases, lead pass-through OK")



def check_seq_increments_on_attempt() -> None:
    """seq increments when a send is attempted, including drops; rate-limit skips do not."""
    pub = cp.CppVizPublisher(port=9, min_interval_s=0.0)

    class DropSock:
        def __init__(self) -> None:
            self.n = 0
            self.payloads: list[bytes] = []

        def sendto(self, data, _addr):
            self.n += 1
            if self.n <= 3:
                raise BlockingIOError
            self.payloads.append(data)

        def close(self) -> None:
            pass

    sock = DropSock()
    pub._sock = sock
    st = rich_state()
    for _ in range(3):
        assert pub.publish(st) is False
    assert pub.dropped == 3 and pub.sent == 0
    assert pub.publish(st) is True
    assert json.loads(sock.payloads[0])["seq"] == 4, json.loads(sock.payloads[0])["seq"]
    pub.close()
    print("seq: three drops then a send -> first sent seq is 4 OK")


def check_close_terminates_and_reaps() -> None:
    pub = cp.CppVizPublisher(port=9, min_interval_s=0.0)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pub._proc = proc
    t0 = time.monotonic()
    pub.close()
    elapsed = time.monotonic() - t0
    assert pub._proc is None
    assert proc.poll() is not None, "dummy exe still running after close()"
    assert elapsed < 1.0, f"close() took {elapsed:.2f}s; terminate should be prompt (kill fallback is 2s)"
    if sys.platform != "win32":
        assert proc.returncode == -signal.SIGTERM, proc.returncode
    print(f"close(): terminated dummy pid, reaped rc={proc.returncode} in {elapsed*1000:.0f} ms OK")



def check_close_kill_fallback() -> None:
    """A child that ignores SIGTERM must be kill()ed after the wait timeout."""
    if sys.platform == "win32":
        print("close() kill fallback: skipped on Windows (TerminateProcess cannot be ignored)")
        return
    pub = cp.CppVizPublisher(port=9, min_interval_s=0.0)
    ready_r, ready_w = os.pipe()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "os.write(%d, b'x'); time.sleep(60)" % ready_w,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(ready_w,),
    )
    os.close(ready_w)
    got = os.read(ready_r, 1)
    os.close(ready_r)
    assert got == b"x", got
    pub._proc = proc
    t0 = time.monotonic()
    pub.close()
    elapsed = time.monotonic() - t0
    assert pub._proc is None
    assert proc.poll() is not None, "SIGTERM-immune child still running after close(); kill() fallback missing"
    assert proc.returncode == -signal.SIGKILL, proc.returncode
    assert 1.5 <= elapsed < 6.0, f"close() took {elapsed:.2f}s; expected ~2s terminate timeout then kill"
    print(f"close() kill fallback: SIGTERM ignored, SIGKILL reaped rc={proc.returncode} in {elapsed:.2f}s OK")


def check_sendto_oserror() -> None:
    pub = cp.CppVizPublisher(port=9, min_interval_s=0.0)

    class BoomSock:
        def sendto(self, *_a):
            raise OSError("refused")

        def close(self):
            pass

    pub._sock = BoomSock()
    assert pub.publish(rich_state()) is False
    assert pub.errors == 0 and pub.dropped >= 1, pub.status()
    pub.close()
    print("sendto OSError: counted as drop, no raise OK")



def check_occ_out_contract_freeze() -> None:
    """Frozen field set matches b939818, and the live module when available."""
    import ast
    import os
    from python.viz.occ_out_v2_contract import (
        CAM_IDS, CELL_KEYS, EDGE_KEYS, EGO_KEYS, GRID_KEYS, LANE_KEYS, LANE_PAINTS,
        OCC_CLASS_NAMES, OCC_OUT_CONTRACT_SHA, OCC_OUT_VERSION, TOP_KEYS,
        frozen_field_set, make_occ_out_v2,
    )
    frozen = frozen_field_set()
    assert OCC_OUT_VERSION == 2 and OCC_OUT_CONTRACT_SHA == "b939818"
    assert frozen["CELL_KEYS"] == ("ijk", "cls", "prob", "observed")
    assert frozen["EDGE_KEYS"] == ("side", "kind", "pts", "conf", "observed")
    assert frozen["LANE_KEYS"] == ("paint", "pts", "conf", "observed")
    assert frozen["OCC_CLASS_NAMES"] == ("other", "drivable", "vehicle")
    assert frozen["LANE_PAINTS"] == ("white", "yellow")
    assert frozen["GRID_KEYS"] == ("voxel_m", "origin", "shape")
    assert "observed" in CELL_KEYS and "paint" in LANE_KEYS and "v" in TOP_KEYS and "narrow" in CAM_IDS
    msg = make_occ_out_v2(
        edges=[{"side": "left", "kind": "curb", "pts": [[-7.0, 0.0], [-7.0, 4.0]],
                "conf": [1.0, 1.0], "observed": [True, True]}],
    )
    assert set(msg) == set(TOP_KEYS) and msg["v"] == 2 and set(msg["grid"]) == set(GRID_KEYS)
    assert set(msg["ego"]) == set(EGO_KEYS)
    # Optional: compare against the real contract file (read-only). Do not import
    # it — it pulls occ_geometry / numpy and lives in another worktree.
    candidates = []
    env = os.environ.get("GVD_OCC_OUT_CONTRACT")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/workspace/swarm/swarm-2/grok-vision-drive/python/perception/occ_out.py"))
    compared = False
    for path in candidates:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        ns: dict = {}
        want = set(frozen)
        keep = want | {n + "_V1" for n in want} | {"CELL_KEYS_V1", "EDGE_KEYS_V1", "LANE_KEYS_V1"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name not in keep and not name.endswith("_KEYS") and name not in want:
                    if name not in ("OCC_CLASS_NAMES", "LANE_PAINTS", "EDGE_KINDS", "CAM_IDS"):
                        continue
                try:
                    ns[name] = eval(compile(ast.Expression(node.value), str(path), "eval"), {"__builtins__": {}}, ns)
                except Exception:
                    continue
        got = {k: tuple(ns[k]) for k in want if k in ns}
        assert got == frozen, (path, {k: (got.get(k), frozen[k]) for k in sorted(want) if got.get(k) != frozen[k]})
        compared = True
        print(f"occ_out v2 freeze matches {path} OK")
        break
    if not compared:
        print("occ_out v2 freeze: live module not present, constants self-checked OK")


def check_occ_out_adapter_strict() -> None:
    from python.viz.occ_out_v2_contract import make_occ_out_v2
    # string "false" is not a bool — that point is dropped
    msg = make_occ_out_v2(edges=[{
        "side": "left", "kind": "curb",
        "pts": [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        "conf": [0.9, 0.8, 0.7],
        "observed": [True, "false", False],
    }])
    dots = cp.collect_dots({"occ_out": msg})
    assert len(dots) == 2 and all(isinstance(d["observed"], bool) for d in dots), dots
    assert {round(d["y"], 1) for d in dots} == {1.0, 3.0}
    # missing conf / observed drops the point, not full brightness
    msg2 = make_occ_out_v2(lanes=[{
        "paint": "white", "pts": [[1.0, 0.0], [1.0, 2.0], [1.0, 4.0]],
        "conf": [0.5],  # only first point
        "observed": [True, True, True],
    }])
    d2 = cp.collect_dots({"occ_out": msg2})
    assert len(d2) == 1 and d2[0]["y"] == 0.0 and d2[0]["prob"] == 0.5, d2
    # numpy-safe: arrays for pts/conf/observed, and bool_ scalars
    np = None
    try:
        import numpy as np
    except ImportError:
        print("occ_out adapter: numpy not installed, array path skipped")
        return
    msg3 = make_occ_out_v2(edges=[{
        "side": "right", "kind": "road_edge",
        "pts": np.asarray([[3.0, 0.0], [3.0, 5.0]], dtype=float),
        "conf": np.asarray([0.4, 0.3], dtype=float),
        "observed": np.asarray([True, False]),
    }])
    d3 = cp.collect_dots({"occ_out": msg3})
    assert len(d3) == 2 and d3[0]["cls"] == "edge" and d3[1]["observed"] is False, d3
    # a numpy bool must not take the bool("false") path
    assert cp._as_bool(np.bool_(False)) is False and cp._as_bool("false") is None
    print(f"occ_out adapter strict: dropped-invalid {len(dots)} / {len(d2)}, numpy {len(d3)} OK")


def check_prob_clamp_and_drop() -> None:
    # Out-of-range numeric probs are clamped (mutant: drop the clamp line).
    hi = cp.build_scene_message({"dots": [{"x": 0.0, "y": 1.0, "cls": "white", "prob": 1.5, "observed": True}]}, seq=1, t=0)["dots"]
    lo = cp.build_scene_message({"dots": [{"x": 0.0, "y": 1.0, "cls": "white", "prob": -0.2, "observed": True}]}, seq=1, t=0)["dots"]
    assert len(hi) == 1 and hi[0]["prob"] == 1.0, hi
    assert len(lo) == 1 and lo[0]["prob"] == 0.0, lo
    # Missing or invalid prob / observed is dropped, not drawn at 1.0 / True.
    missing_p = cp.build_scene_message({"dots": [{"x": 0.0, "y": 1.0, "cls": "white", "observed": True}]}, seq=1, t=0)["dots"]
    missing_o = cp.build_scene_message({"dots": [{"x": 0.0, "y": 1.0, "cls": "white", "prob": 0.5}]}, seq=1, t=0)["dots"]
    bad_p = cp.build_scene_message({"dots": [{"x": 0.0, "y": 1.0, "cls": "white", "prob": "high", "observed": True}]}, seq=1, t=0)["dots"]
    bad_o = cp.build_scene_message({"dots": [{"x": 0.0, "y": 1.0, "cls": "white", "prob": 0.5, "observed": "false"}]}, seq=1, t=0)["dots"]
    assert missing_p == [] and missing_o == [] and bad_p == [] and bad_o == []
    print("prob clamp + drop missing/invalid OK")


def check_pose_not_invented() -> None:
    # Partial pose must not invent the rest; dots stay in ego frame.
    msg = cp.build_scene_message({
        "ego": {"speed_mps": 5.0, "yaw": 0.3},
        "dots": [{"x": 1.0, "y": 2.0, "cls": "white", "prob": 1.0, "observed": True}],
    }, seq=1, t=0)
    assert "x" not in msg["ego"] and "y" not in msg["ego"] and "yaw" not in msg["ego"], msg["ego"]
    assert msg["dots"][0]["x"] == 1.0 and msg["dots"][0]["y"] == 2.0
    # Complete pose converts ego-frame dots onto the wire in world frame.
    # yaw=π/2 (facing +Y): ego (0, 28) + pose (0, 8) -> world (0, 36).
    posed = cp.build_scene_message({
        "ego": {"speed_mps": 5.0, "x": 0.0, "y": 8.0, "yaw": 1.5707963},
        "dots": [{"x": 0.0, "y": 28.0, "cls": "white", "prob": 1.0, "observed": True}],
        "tracks": [{"id": 1, "cls": "vehicle", "x": 0.0, "y": 14.0, "l": 4.5, "w": 1.85, "h": 1.5}],
    }, seq=1, t=0)
    assert posed["ego"]["x"] == 0.0 and posed["ego"]["y"] == 8.0
    assert abs(posed["dots"][0]["x"] - 0.0) < 1e-6 and abs(posed["dots"][0]["y"] - 36.0) < 1e-6, posed["dots"]
    # tracks stay in ego frame on the wire
    assert posed["tracks"][0]["x"] == 0.0 and posed["tracks"][0]["y"] == 14.0
    print("pose: no invention, dots world-converted, tracks ego OK")

def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    check_dots()
    check_occ_out_contract_freeze()
    check_occ_out_adapter_strict()
    check_prob_clamp_and_drop()
    check_pose_not_invented()
    check_schema()
    check_perception_only()
    check_paint()
    check_schema_strict()
    check_nonblocking_no_listener()
    check_receiver_never_reads()
    check_roundtrip_and_rate()
    check_oversize()
    check_fail_safe()
    check_sendto_oserror()
    check_seq_increments_on_attempt()
    check_close_terminates_and_reaps()
    check_close_kill_fallback()
    check_flag_default_off()
    print("test_viz_cpp_publisher: all OK")


if __name__ == "__main__":
    main()
