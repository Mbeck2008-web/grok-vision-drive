#!/usr/bin/env python3
"""Live-loop stress for the override echo path. Not part of the offline suite (no test_ prefix).

Two measurements, each against the checkout given by ``--root`` (default: this repo), so the
same script times a base commit and a fix commit:

* ``override_live``: runs ``test_m6_retail.check_ffb_override_live`` unchanged, ``--runs``
  times, each in its own process, while ``--hog`` busy-loop processes load every core.
* ``takeover``: the real supervisor (stub cameras) with a fake mod that echoes at 10 Hz like
  gvd_main (``EGO_POLL_S``), stamping ``echo_seq``. After engage, the wheel is held 0.25 off the
  command; prints the ms from the first echo carrying that pull to ``player_steer`` on
  gvd_engage.json. An older checkout ignores ``echo_seq``.

    xvfb-run -a python scripts/stress_ffb_echo_live.py --runs 30 --hog 8
    xvfb-run -a python scripts/stress_ffb_echo_live.py --root ../base --mode takeover

The CPU hog makes run-to-run timing noisy; compare medians across many runs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _hog() -> None:
    x = 0
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def _child_env(root: Path) -> dict:
    return dict(os.environ, PYTHONPATH=str(root))


def run_override_live(root: Path) -> tuple[bool, str]:
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import test_m6_retail as t; "
        "t.check_ffb_override_live(); print('PASS')"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", code], cwd=str(root), env=_child_env(root),
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = res.returncode == 0 and "PASS" in res.stdout
    tail = (res.stdout + res.stderr).strip().splitlines()
    return ok, "" if ok else (tail[-1] if tail else f"rc={res.returncode}")


def worker_takeover(root: Path, pull: float, seconds: float) -> None:
    """Runs with cwd=root and PYTHONPATH=root: these imports are that checkout's modules."""
    sys.path.insert(0, str(root))
    from python.control.actuate import cmd_path, ego_path, engage_path, write_engage_flag
    from python.runtime import paths as gvd_paths
    from python.runtime.state_io import read_state

    def touch_bus() -> Path:
        pin = gvd_paths.predicted_gvd_docs_path()
        pin.mkdir(parents=True, exist_ok=True)
        abs_pin = pin if pin.is_absolute() else pin.resolve()
        payload = json.dumps({"lua_bus": str(abs_pin)})
        (pin / "gvd_link.json").write_text(payload, encoding="utf-8")
        if abs_pin != pin:
            (abs_pin / "gvd_link.json").write_text(payload, encoding="utf-8")
        return abs_pin

    def read(p: Path) -> dict:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    touch_bus()
    if ego_path().exists():
        ego_path().unlink()
    write_engage_flag(True)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(root / "python" / "run_vision.py"), "--backend", "stub", "--hz", "20",
         "--encode", "cpu", "--allow-preview-drive"],
        cwd=str(root), env=_child_env(root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    out = {"ms": None, "reason": "timeout"}
    echo_k = 0
    next_echo = time.time()
    last_hb = 0.0
    pull_at = None       # wall time the wheel is pulled (after engage is live)
    first_pull = None    # wall time of the first echo carrying the pull
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                out["reason"] = f"exited {proc.returncode}"
                break
            now = time.time()
            root_bus = touch_bus()
            eng = read(engage_path())
            if eng.get("engaged") is True and now - last_hb >= 1.0:
                eng["mtime"] = now
                engage_path().write_text(json.dumps(eng), encoding="utf-8")
                last_hb = now
            if now >= next_echo:
                cmd = read(cmd_path())
                seq = int(cmd.get("seq", -1)) if cmd else -1
                if seq >= 0:
                    echo_k += 1
                    steer = float(cmd.get("steer", 0.0))
                    pulled = pull_at is not None and now >= pull_at
                    if pulled and first_pull is None:
                        first_pull = now
                    ego_path().write_text(json.dumps({
                        "speed_mps": 5.5,
                        "steering_input": steer + pull if pulled else steer,
                        "throttle_input": float(cmd.get("throttle", 0.0)),
                        "brake_input": float(cmd.get("brake", 0.0)),
                        "applied_seq": seq,
                        "echo_seq": echo_k,
                        "applying": True,
                        "mtime": int(now),
                        "lua_bus": str(root_bus),
                    }), encoding="utf-8")
                next_echo = max(next_echo + 0.1, now + 0.05)
            if pull_at is None:
                st = read_state() or {}
                if st.get("engaged") is True and st.get("cmd_applied") is True:
                    pull_at = now + 0.6
            elif first_pull is not None and eng.get("engaged") is False:
                out = {"ms": (now - first_pull) * 1000.0, "reason": str(eng.get("disengage_reason"))}
                break
            time.sleep(0.005)
    finally:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        write_engage_flag(False)
    print("RESULT " + json.dumps(out))


def run_takeover(root: Path, pull: float) -> dict:
    res = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--root", str(root), "--pull", str(pull)],
        cwd=str(root), env=_child_env(root), capture_output=True, text=True, timeout=120,
    )
    for line in res.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"ms": None, "reason": f"no result rc={res.returncode}: {res.stderr.strip()[-200:]}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--mode", choices=("override_live", "takeover", "both"), default="both")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--takeover-runs", type=int, default=15)
    ap.add_argument("--hog", type=int, default=os.cpu_count() or 2, help="busy-loop processes (0 = none)")
    ap.add_argument("--pull", type=float, default=0.25)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.worker:
        worker_takeover(root, args.pull, seconds=20.0)
        return

    hogs = [mp.Process(target=_hog, daemon=True) for _ in range(max(0, args.hog))]
    for h in hogs:
        h.start()
    print(f"root {root}  hog {len(hogs)} procs", flush=True)
    try:
        if args.mode in ("override_live", "both"):
            ok = 0
            fails: list[str] = []
            for i in range(args.runs):
                passed, why = run_override_live(root)
                ok += passed
                if not passed:
                    fails.append(why)
                print(f"  override_live {i + 1}/{args.runs}: {'PASS' if passed else 'FAIL ' + why}", flush=True)
            print(f"check_ffb_override_live: {ok}/{args.runs} passed, {args.runs - ok} failed", flush=True)
        if args.mode in ("takeover", "both"):
            ms: list[float] = []
            other: list[str] = []
            for i in range(args.takeover_runs):
                r = run_takeover(root, args.pull)
                if r.get("reason") == "player_steer" and r.get("ms") is not None:
                    ms.append(float(r["ms"]))
                else:
                    other.append(str(r.get("reason")))
                print(f"  takeover {i + 1}/{args.takeover_runs}: {r}", flush=True)
            if ms:
                print(
                    f"takeover ms (first pulled echo -> player_steer), n={len(ms)}: "
                    f"median {statistics.median(ms):.0f} mean {statistics.mean(ms):.0f} "
                    f"min {min(ms):.0f} max {max(ms):.0f}; other outcomes {other}",
                    flush=True,
                )
            else:
                print(f"takeover: no player_steer; outcomes {other}", flush=True)
    finally:
        for h in hogs:
            h.terminate()
        for h in hogs:
            h.join(timeout=2)


if __name__ == "__main__":
    main()
