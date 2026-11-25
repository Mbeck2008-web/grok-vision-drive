#!/usr/bin/env python3
"""Scripted mutants for python/train/ort_bench.py: each must make scripts/test_ort_bench.py fail.

Dev tool, not part of the offline suite (it runs the test once per mutant).
Each mutant edits a copy of python/ + scripts/ + config/ in a temp folder; the
checkout is never modified. No new dependency (mutmut not needed).

    python scripts/ort_bench_mutants.py [--jobs 3] [--out mutants.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "python/train/ort_bench.py"

# (id, what it breaks, old, new[, target]). ``old`` must occur exactly once; target defaults to TARGET.
MUTANTS = [
    ("tol_inverted", "tolerance comparison inverted (<= -> >)",
     "return bool(np.all(diff <= bound))", "return bool(np.all(diff > bound))"),
    ("tol_any", "all -> any (one good element passes)",
     "return bool(np.all(diff <= bound))", "return bool(np.any(diff <= bound))"),
    ("tol_strict_ge", "<= -> >=",
     "return bool(np.all(diff <= bound))", "return bool(np.all(diff >= bound))"),
    ("atol_widened", "ATOL widened 1e-4 -> 1e-1", "\nPARITY_ATOL = 1e-4\n", "\nPARITY_ATOL = 1e-1\n"),
    ("rtol_widened", "RTOL widened 1e-3 -> 1.0", "\nPARITY_RTOL = 1e-3\n", "\nPARITY_RTOL = 1.0\n"),
    ("bound_rtol_only", "atol dropped from the bound (narrowing still changes the gate)",
     "bound = atol + rtol * np.abs(ref[fin_ref])", "bound = rtol * np.abs(ref[fin_ref])"),
    ("shape_check_gone", "shape mismatch no longer fails",
     "    if ref.shape != got.shape:\n        return False\n    fin_ref",
     "    if ref.shape != got.shape:\n        return True\n    fin_ref"),
    ("nonfinite_ignored", "non-finite values no longer compared (inf vs -inf, NaN vs NaN pass)",
     "    if not np.array_equal(ref[~fin_ref], got[~fin_got]):\n        return False",
     "    if not np.array_equal(ref[~fin_ref], got[~fin_got]):\n        pass"),
    ("missing_output_ok", "a missing output does not fail parity",
     "                res.ok = False\n                res.failures.append(f\"tick {tick}: output {name!r} missing\")\n",
     ""),
    ("dropped_outputs", "compare skips the last 4 outputs (lane_gt + state)",
     "    for name in names:\n        worst_abs", "    for name in list(names)[:-4]:\n        worst_abs"),
    ("compare_first_tick_only", "only tick 0 compared",
     "for tick, (r, g) in enumerate(zip(ref, got)):", "for tick, (r, g) in enumerate(zip(ref[:1], got[:1])):"),
    ("ort_layout_bgr", "ORT fed channel-reversed strip",
     'full = {"strip": feed["strip"], "dt"', 'full = {"strip": feed["strip"][:, ::-1].copy(), "dt"'),
    ("ort_layout_nhwc", "ORT fed NHWC bytes as NCHW",
     'full = {"strip": feed["strip"], "dt"',
     'full = {"strip": np.ascontiguousarray(feed["strip"].transpose(0, 2, 3, 1)).reshape(feed["strip"].shape), "dt"'),
    ("ort_state_not_fed", "ORT state not fed back (cold every tick)",
     "        results.append(tick)\n        state = {k: tick[k] for k in (\"h\", \"tokens\", \"mask\")}\n    return results\n\n\n@dataclass",
     "        results.append(tick)\n    return results\n\n\n@dataclass"),
    ("export_other_weights", "export writes a differently seeded net (perturbed weights)",
     "    export_scene_onnx(path, net)\n", "    export_scene_onnx(path, build_net(999))\n"),
    ("opset_changed", "asserted opset changed", "SCENE_OPSET = 17", "SCENE_OPSET = 18"),
    ("fallback_inverted", "CPU-fallback detection inverted",
     "if p != want}", "if p == want}"),
    ("memcpy_wrong_op", "memcpy counter looks for the wrong op name",
     'MEMCPY_OPS = ("MemcpyToHost", "MemcpyFromHost")', 'MEMCPY_OPS = ("MemcpyToDevice", "MemcpyFromHost")'),
    ("bench_no_state", "bench tick does not feed state back",
     "        for k in (\"h\", \"tokens\", \"mask\"):\n            feed[k] = by[k]\n", "        pass\n"),
    # Skeptic PR #247 @ 483541b survivors (M1-M6), now guarded.
    ("bench_skip_tick", "timed loop does not run tick()",
     "        t0 = time.perf_counter()\n        tick()\n", "        t0 = time.perf_counter()\n"),
    ("bench_seconds", "latency reported in seconds, not ms",
     "ms.append((time.perf_counter() - t0) * 1000.0)", "ms.append(time.perf_counter() - t0)"),
    ("cuda_silent_fallback", "ep=cuda landing on CPU no longer raises",
     "    if want not in got:\n        raise RuntimeError", "    if False:\n        raise RuntimeError"),
    ("rss_gib", "Linux ru_maxrss KiB divided by 1024*1024 (GiB)",
     "else float(kb) / 1024.0", "else float(kb) / (1024.0 * 1024.0)"),
    ("vram_delta_inverted", "VRAM delta = baseline - peak",
     "delta = self.peak_device_mb - self.baseline_mb", "delta = self.baseline_mb - self.peak_device_mb"),
    ("cli_parity_exit_zero", "CLI returns 0 on parity failure",
     "file=sys.stderr)\n        return 1", "file=sys.stderr)\n        return 0", "scripts/ort_bench.py"),
]


def run_one(mutant: tuple, python: str) -> dict:
    mid, desc, old, new = mutant[:4]
    target_rel = mutant[4] if len(mutant) > 4 else TARGET
    with tempfile.TemporaryDirectory(prefix=f"gvd_mut_{mid}_") as name:
        root = Path(name)
        for sub in ("python", "scripts", "config"):
            shutil.copytree(ROOT / sub, root / sub, ignore=shutil.ignore_patterns("__pycache__"))
        target = root / target_rel
        src = target.read_text(encoding="utf-8")
        count = src.count(old)
        if count != 1:
            return {"id": mid, "desc": desc, "status": "invalid", "detail": f"pattern found {count}x"}
        target.write_text(src.replace(old, new), encoding="utf-8")
        t0 = time.time()
        try:
            res = subprocess.run([python, "-W", "ignore", str(root / "scripts" / "test_ort_bench.py")],
                                 capture_output=True, text=True, timeout=600, cwd=str(root))
            rc = res.returncode
            tail = [ln[:160] for ln in (res.stdout + res.stderr).splitlines() if "Error" in ln or "assert" in ln][-2:]
        except subprocess.TimeoutExpired:
            rc, tail = -9, ["timeout"]
        return {"id": mid, "desc": desc, "status": "killed" if rc != 0 else "SURVIVED", "rc": rc,
                "seconds": round(time.time() - t0, 1), "detail": tail}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ort_bench scripted mutants")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default="", help="comma-separated mutant ids")
    args = ap.parse_args(argv)
    chosen = [m for m in MUTANTS if not args.only or m[0] in args.only.split(",")]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda m: run_one(m, sys.executable), chosen))
    for r in results:
        print(f"{r['status']:9s} {r['id']:24s} {r['desc']}  {r.get('detail')}", flush=True)
    killed = sum(r["status"] == "killed" for r in results)
    survived = sum(r["status"] == "SURVIVED" for r in results)
    invalid = sum(r["status"] == "invalid" for r in results)
    print(f"mutants: {len(results)} total, {killed} killed, {survived} survived, {invalid} invalid", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if survived == 0 and invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
