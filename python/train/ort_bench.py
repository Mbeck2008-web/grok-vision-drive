"""PyTorch vs ONNX Runtime FP32: export, parity, op/EP report and latency bench for SceneNet.

Offline / bench tooling only. Nothing here is imported by the live supervisor,
the safety path (heartbeat / AEB / engage) or ``scene_lane_runtime``; it feeds
no live input (no odometry, no controls). Phase 0 decision D4: PyTorch
trains, ORT CUDA FP32 deploys. This module measures that path.

Export reuses ``scene_net.export_scene_onnx`` (opset ``SCENE_OPSET`` = 17, the
same graph the live runtime loads): inputs ``strip`` 1x3x192xW, ``dt`` 1,
``h`` 1x1x512, ``tokens`` 1xRxD, ``mask`` 1xR. The recurrent state is carried
across ticks exactly as ``SceneLaneRuntime.observe`` does: each tick's ``h`` /
``tokens`` / ``mask`` outputs are the next tick's inputs.

PARITY TOLERANCE (stated once, never widened; 2026-11-23):
    every element of every output on every tick must satisfy
        |ort - torch| <= PARITY_ATOL + PARITY_RTOL * |torch|
    with PARITY_ATOL = 1e-4 and PARITY_RTOL = 1e-3, for CPU EP and CUDA EP.
A NaN/Inf mismatch, a missing output or a shape mismatch fails parity outright.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-3
SCENE_OPSET = 17  # fixed by scene_net._trace_scene_onnx; asserted on every export
DEFAULT_TICKS = 6
DEFAULT_SEED = 1234
# Tick spacing of the rollout (s). First tick is 0.0, like a cold live runtime.
ROLLOUT_DT = (0.0, 0.1, 0.1, 0.05, 0.2, 0.1, 0.1, 0.1)
MEMCPY_OPS = ("MemcpyToHost", "MemcpyFromHost")
EP_NAMES = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
# Same cap the live YOLO session uses (perception/detect.py): 4 GB inference budget.
CUDA_MEM_LIMIT = 4 * 1024 ** 3


def base_name(name: str) -> str:
    """``h.1`` -> ``h``: the tracer renames pass-through state tensors (same rule as the live runtime)."""
    from python.train.scene_lane_runtime import _base_name

    return _base_name(name)


class ParityError(AssertionError):
    """ORT and PyTorch disagree beyond the stated tolerance."""


def within_tolerance(ref: np.ndarray, got: np.ndarray, atol: float = PARITY_ATOL, rtol: float = PARITY_RTOL) -> bool:
    """True when every element of ``got`` is within ``atol + rtol*|ref|`` of ``ref``.

    Shapes must match exactly; NaN/Inf anywhere is a failure unless both sides
    hold the identical non-finite value at that element.
    """
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape:
        return False
    fin_ref = np.isfinite(ref)
    fin_got = np.isfinite(got)
    # Non-finite values must match exactly (NaN never does). A non-finite value
    # at a position where ref is finite fails below (inf/NaN diff).
    if not np.array_equal(ref[~fin_ref], got[~fin_got]):
        return False
    diff = np.abs(got[fin_ref] - ref[fin_ref])
    bound = atol + rtol * np.abs(ref[fin_ref])
    return bool(np.all(diff <= bound))


def diff_stats(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape or ref.size == 0:
        return {"max_abs": float("inf"), "max_rel": float("inf")}
    diff = np.abs(got - ref)
    rel = diff / np.maximum(np.abs(ref), 1e-12)
    return {"max_abs": float(np.nanmax(diff)), "max_rel": float(np.nanmax(rel))}


def _need_torch() -> Any:
    import torch  # noqa: F401

    return torch


def build_net(seed: int = DEFAULT_SEED) -> Any:
    """The current default SceneNet (small preset, 8 cams), seeded, eval mode, CPU."""
    torch = _need_torch()
    from python.train.scene_net import SceneNet

    torch.manual_seed(int(seed))
    net = SceneNet()
    return net.to("cpu").eval()


def export(net: Any, path: Path | str) -> Path:
    """Export through the existing ``export_scene_onnx`` and assert the opset."""
    import onnx

    from python.train.scene_net import export_scene_onnx

    path = Path(path)
    export_scene_onnx(path, net)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opsets = {imp.domain or "ai.onnx": int(imp.version) for imp in model.opset_import}
    if opsets.get("ai.onnx") != SCENE_OPSET:
        raise RuntimeError(f"export opset {opsets} != {SCENE_OPSET}")
    return path


def make_inputs(strip_shape: Sequence[int], ticks: int = DEFAULT_TICKS, seed: int = DEFAULT_SEED) -> list[dict[str, np.ndarray]]:
    """Seeded per-tick strips in [0, 1] (the live NCHW float layout of ``_strip_nchw``) and dt values.

    State tensors are not here: each side carries its own state forward.
    """
    rng = np.random.default_rng(int(seed))
    shape = tuple(int(d) for d in strip_shape)
    out = []
    for tick in range(int(ticks)):
        strip = rng.random(shape, dtype=np.float32)
        dt = np.asarray([ROLLOUT_DT[tick % len(ROLLOUT_DT)]], dtype=np.float32)
        out.append({"strip": strip, "dt": dt})
    return out


def net_strip_shape(net: Any) -> tuple[int, int, int, int]:
    from python.train.scene_net import CANVAS_H, CANVAS_W

    return (1, 3, int(getattr(net, "canvas_h", CANVAS_H)), int(getattr(net, "canvas_w", CANVAS_W)))


def initial_state_np(net: Any) -> dict[str, np.ndarray]:
    state = net.initial_state(device="cpu")
    return {k: getattr(state, k).detach().cpu().numpy().astype(np.float32) for k in ("h", "tokens", "mask")}


def _fixed(shape: Any) -> tuple[int, ...]:
    dims = tuple(shape or ())
    if not dims or not all(isinstance(d, int) and d > 0 for d in dims):
        raise ValueError(f"graph input has non-fixed dims {list(dims)}")
    return dims


def session_strip_shape(sess: Any) -> tuple[int, ...]:
    for item in sess.get_inputs():
        if base_name(item.name) == "strip":
            return _fixed(item.shape)
    raise ValueError("graph has no strip input")


def session_initial_state(sess: Any) -> dict[str, np.ndarray]:
    """Cold state from the graph's own input shapes, filled like ``SceneLaneRuntime``.

    Torch-free, so an ORT-only bench never imports torch.
    """
    from python.train.scene_net import RESIDUAL_S

    out = {}
    for item in sess.get_inputs():
        key = base_name(item.name)
        if key in ("h", "tokens", "mask"):
            fill = float(RESIDUAL_S) + 1.0 if key == "mask" else 0.0
            out[key] = np.full(_fixed(item.shape), fill, dtype=np.float32)
    missing = {"h", "tokens", "mask"} - set(out)
    if missing:
        raise ValueError(f"graph lacks state inputs {sorted(missing)}")
    return out


def output_names(net: Any) -> list[str]:
    from python.train.scene_net import OUTPUT_FIELDS

    return list(OUTPUT_FIELDS) + ["h", "tokens", "mask"]


def torch_rollout(net: Any, inputs: Sequence[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """PyTorch reference through the same ``_OnnxScene`` wrapper the export traces."""
    torch = _need_torch()
    from python.train.scene_net import _OnnxScene

    wrapper = _OnnxScene(net).eval()
    names = output_names(net)
    state = initial_state_np(net)
    results = []
    with torch.no_grad():
        for feed in inputs:
            args = [torch.from_numpy(feed["strip"]), torch.from_numpy(feed["dt"])]
            args += [torch.from_numpy(state[k]) for k in ("h", "tokens", "mask")]
            outs = wrapper(*args)
            tick = {name: t.detach().cpu().numpy().astype(np.float32) for name, t in zip(names, outs)}
            results.append(tick)
            state = {k: tick[k] for k in ("h", "tokens", "mask")}
    return results


def session_options(profile_dir: Path | None = None, optimized_path: Path | None = None, threads: int = 0) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = int(threads)
    if profile_dir is not None:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(profile_dir) / "ort_profile")
    if optimized_path is not None:
        so.optimized_model_filepath = str(optimized_path)
    return so


def providers_for(ep: str) -> list[Any]:
    if ep == "cpu":
        return ["CPUExecutionProvider"]
    if ep == "cuda":
        return [
            ("CUDAExecutionProvider", {"device_id": 0, "gpu_mem_limit": CUDA_MEM_LIMIT,
                                       "arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
    raise ValueError(f"unknown ep {ep!r}")


def make_session(path: Path | str, ep: str = "cpu", **so_kwargs: Any) -> Any:
    """Session on exactly ``ep``. A CUDA request that silently lands on CPU raises."""
    import onnxruntime as ort

    if ep == "cuda" and hasattr(ort, "preload_dlls"):
        # pip nvidia-* wheels (requirements-ort.txt) put the CUDA/cuDNN DLLs in site-packages.
        try:
            ort.preload_dlls()
        except Exception:
            pass

    sess = ort.InferenceSession(str(path), sess_options=session_options(**so_kwargs), providers=providers_for(ep))
    want = EP_NAMES[ep]
    got = sess.get_providers()
    if want not in got:
        raise RuntimeError(f"requested {want}, session has {got} (EP failed to load)")
    return sess


def ort_rollout(sess: Any, inputs: Sequence[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """ORT rollout, state fed back by output name like ``SceneLaneRuntime.observe``."""
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    state = session_initial_state(sess)
    results = []
    for feed in inputs:
        full = {"strip": feed["strip"], "dt": feed["dt"], **state}
        outs = sess.run(None, {name: full[base_name(name)] for name in in_names})
        tick = {base_name(name): np.asarray(v, dtype=np.float32) for name, v in zip(out_names, outs)}
        results.append(tick)
        state = {k: tick[k] for k in ("h", "tokens", "mask")}
    return results


@dataclass
class ParityResult:
    ok: bool
    atol: float
    rtol: float
    ticks: int
    per_output: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "atol": self.atol, "rtol": self.rtol, "ticks": self.ticks,
                "per_output": self.per_output, "failures": self.failures}


def compare(ref: Sequence[dict[str, np.ndarray]], got: Sequence[dict[str, np.ndarray]],
            names: Sequence[str]) -> ParityResult:
    """Every reference output on every tick must exist in ``got`` and be within tolerance."""
    res = ParityResult(ok=True, atol=PARITY_ATOL, rtol=PARITY_RTOL, ticks=len(ref))
    if len(got) != len(ref):
        res.ok = False
        res.failures.append(f"tick count {len(got)} != {len(ref)}")
        return res
    for name in names:
        worst_abs = 0.0
        worst_rel = 0.0
        shape = None
        for tick, (r, g) in enumerate(zip(ref, got)):
            if name not in r or name not in g:
                res.ok = False
                res.failures.append(f"tick {tick}: output {name!r} missing")
                worst_abs = worst_rel = float("inf")
                continue
            shape = list(np.shape(r[name]))
            stats = diff_stats(r[name], g[name])
            worst_abs = max(worst_abs, stats["max_abs"])
            worst_rel = max(worst_rel, stats["max_rel"])
            if not within_tolerance(r[name], g[name]):
                res.ok = False
                res.failures.append(f"tick {tick}: {name} max_abs={stats['max_abs']:.3g} max_rel={stats['max_rel']:.3g}")
        res.per_output[name] = {"shape": shape, "max_abs": worst_abs, "max_rel": worst_rel}
    return res


def check_parity(path: Path | str, net: Any, ep: str = "cpu", ticks: int = DEFAULT_TICKS,
                 seed: int = DEFAULT_SEED) -> ParityResult:
    inputs = make_inputs(net_strip_shape(net), ticks=ticks, seed=seed)
    ref = torch_rollout(net, inputs)
    sess = make_session(path, ep)
    got = ort_rollout(sess, inputs)
    return compare(ref, got, output_names(net))


def _parse_profile(profile_path: Path) -> dict[str, Any]:
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    nodes: dict[str, dict[str, str]] = {}
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        args = ev.get("args") or {}
        provider = args.get("provider")
        op = args.get("op_name")
        if not provider or not op:
            continue
        name = str(ev.get("name", "")).rsplit("_kernel_time", 1)[0]
        nodes[name] = {"op": op, "provider": provider}
    return nodes


def memcpy_counts(model: Any) -> dict[str, int]:
    """``MemcpyToHost`` / ``MemcpyFromHost`` nodes the partitioner put in an optimized graph."""
    ops = [n.op_type for n in model.graph.node]
    return {op: ops.count(op) for op in MEMCPY_OPS}


def op_report(path: Path | str, ep: str = "cpu", workdir: Path | None = None) -> dict[str, Any]:
    """Which EP runs each node (ORT profiling), plus Memcpy nodes in the partitioned graph.

    The optimized model ORT writes after partitioning is where the partitioner's
    ``MemcpyToHost`` / ``MemcpyFromHost`` nodes appear; they are counted there.
    """
    import onnx

    own_tmp = None
    if workdir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="gvd_ort_ops_")
        workdir = Path(own_tmp.name)
    workdir = Path(workdir)
    try:
        optimized = workdir / f"optimized_{ep}.onnx"
        sess = make_session(path, ep, profile_dir=workdir, optimized_path=optimized)
        inputs = make_inputs(session_strip_shape(sess), ticks=2)
        ort_rollout(sess, inputs)
        profile = Path(sess.end_profiling())
        nodes = _parse_profile(profile)
        by_ep: dict[str, dict[str, int]] = {}
        for info in nodes.values():
            ops = by_ep.setdefault(info["provider"], {})
            ops[info["op"]] = ops.get(info["op"], 0) + 1
        opt_model = onnx.load(str(optimized))
        opt_ops = [n.op_type for n in opt_model.graph.node]
        memcpy = memcpy_counts(opt_model)
        src_model = onnx.load(str(path))
        want = EP_NAMES[ep]
        fallback = {p: ops for p, ops in by_ep.items() if p != want}
        report = {
            "ep": ep,
            "session_providers": sess.get_providers(),
            "source_nodes": len(src_model.graph.node),
            "source_op_types": sorted({n.op_type for n in src_model.graph.node}),
            "optimized_nodes": len(opt_ops),
            "profiled_nodes": len(nodes),
            "nodes_by_provider": {p: sum(ops.values()) for p, ops in by_ep.items()},
            "ops_by_provider": by_ep,
            "fallback_ops": fallback,
            "memcpy_nodes": memcpy,
            "memcpy_total": int(sum(memcpy.values())),
        }
        return report
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()


def peak_rss_mb() -> float | None:
    """Process peak RSS in MiB (POSIX ru_maxrss; psutil on Windows if present)."""
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(kb) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(kb) / 1024.0
    except ImportError:
        pass
    try:
        import psutil  # type: ignore

        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / (1024.0 ** 2)
    except Exception:
        return None


def _pct(samples: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def _lat_stats(ms: Sequence[float]) -> dict[str, float]:
    return {"n": len(ms), "mean_ms": float(np.mean(ms)), "p50_ms": _pct(ms, 50), "p90_ms": _pct(ms, 90),
            "p99_ms": _pct(ms, 99), "max_ms": float(np.max(ms))}


def io_bytes(sess: Any) -> dict[str, int]:
    """Bytes crossing host<->device per tick when feeds/fetches are numpy (no IO binding)."""
    feed = {"strip": make_inputs(session_strip_shape(sess), ticks=1)[0]["strip"], "dt": np.zeros(1, np.float32),
            **session_initial_state(sess)}
    h2d = int(sum(feed[base_name(i.name)].nbytes for i in sess.get_inputs()))
    outs = sess.run(None, {i.name: feed[base_name(i.name)] for i in sess.get_inputs()})
    d2h = int(sum(np.asarray(o).nbytes for o in outs))
    return {"h2d_bytes_per_tick": h2d, "d2h_bytes_per_tick": d2h}


def bench(path: Path | str, net: Any = None, ep: str = "cpu", warmup: int = 10, iters: int = 100,
          threads: int = 0) -> dict[str, Any]:
    """Latency of one live tick (session.run with fed-back state).

    ``net`` given: the same tick in PyTorch eager (CPU) is timed too. ``net``
    None: ORT only, torch is never imported (clean RSS for the deploy path).
    ``threads`` 0 = library default for both; set it on a loaded host.
    """
    rss0 = peak_rss_mb()
    sess = make_session(path, ep, threads=threads)
    rss_sess = peak_rss_mb()
    feed = {"strip": make_inputs(session_strip_shape(sess), ticks=1)[0]["strip"],
            "dt": np.asarray([0.1], np.float32), **session_initial_state(sess)}
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    def tick() -> None:
        outs = sess.run(None, {n: feed[base_name(n)] for n in in_names})
        by = {base_name(n): v for n, v in zip(out_names, outs)}
        for k in ("h", "tokens", "mask"):
            feed[k] = by[k]

    for _ in range(int(warmup)):
        tick()
    ms = []
    for _ in range(int(iters)):
        t0 = time.perf_counter()
        tick()
        ms.append((time.perf_counter() - t0) * 1000.0)
    out: dict[str, Any] = {
        "ep": ep,
        "session_providers": sess.get_providers(),
        "threads": int(threads),
        "warmup": int(warmup),
        "ort": _lat_stats(ms),
        "peak_rss_mb_before_session": rss0,
        "peak_rss_mb_after_session_create": rss_sess,
        "peak_rss_mb_after_ort_runs": peak_rss_mb(),
        "torch_imported": sys.modules.get("torch") is not None,
        # Non-zero proves the timed tick carried the GRU state like the live loop.
        "state_h_absmax_after": float(np.abs(feed["h"]).max()),
        **io_bytes(sess),
    }
    if net is not None:
        torch = _need_torch()
        from python.train.scene_net import _OnnxScene

        if threads:
            torch.set_num_threads(int(threads))
        wrapper = _OnnxScene(net).eval()
        st = initial_state_np(net)
        targs = [torch.from_numpy(feed["strip"]), torch.from_numpy(np.asarray([0.1], np.float32))]
        state = [torch.from_numpy(st[k]) for k in ("h", "tokens", "mask")]
        tms = []
        with torch.no_grad():
            for i in range(int(warmup) + int(iters)):
                t0 = time.perf_counter()
                outs = wrapper(*targs, *state)
                el = (time.perf_counter() - t0) * 1000.0
                state = list(outs[-3:])
                if i >= int(warmup):
                    tms.append(el)
        out["torch_cpu"] = _lat_stats(tms)
        out["torch_threads"] = int(torch.get_num_threads())
        out["peak_rss_mb_after_torch"] = peak_rss_mb()
    return out


class VramSampler:
    """Polls ``nvidia-smi`` in a thread: device memory.used and this pid's used_memory (MiB).

    On Windows WDDM the per-process column is often ``[N/A]``; then only the
    device delta (peak - baseline before the session) is reported, and it
    includes any other GPU user (BeamNG must be closed for a clean number).
    """

    def __init__(self, period_s: float = 0.05, gpu: int = 0) -> None:
        import shutil
        import threading

        self.exe = shutil.which("nvidia-smi")
        self.period_s = float(period_s)
        self.gpu = int(gpu)
        self.pid = os.getpid()
        self.baseline_mb: float | None = None
        self.peak_device_mb: float | None = None
        self.peak_pid_mb: float | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _query(self, args: list[str]) -> list[str]:
        import subprocess

        try:
            res = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=5)
        except Exception:
            return []
        return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]

    def device_used_mb(self) -> float | None:
        rows = self._query(["-i", str(self.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        try:
            return float(rows[0])
        except (IndexError, ValueError):
            return None

    def pid_used_mb(self) -> float | None:
        for row in self._query(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]):
            parts = [x.strip() for x in row.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid):
                try:
                    return float(parts[1])
                except ValueError:
                    return None
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            dev = self.device_used_mb()
            pid = self.pid_used_mb()
            self.samples += 1
            if dev is not None:
                self.peak_device_mb = dev if self.peak_device_mb is None else max(self.peak_device_mb, dev)
            if pid is not None:
                self.peak_pid_mb = pid if self.peak_pid_mb is None else max(self.peak_pid_mb, pid)
            self._stop.wait(self.period_s)

    def __enter__(self) -> "VramSampler":
        if self.exe:
            self.baseline_mb = self.device_used_mb()
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.exe:
            self._stop.set()
            self._thread.join(timeout=10)

    def report(self) -> dict[str, Any]:
        if not self.exe:
            return {"available": False, "reason": "nvidia-smi not on PATH"}
        delta = None
        if self.peak_device_mb is not None and self.baseline_mb is not None:
            delta = self.peak_device_mb - self.baseline_mb
        return {"available": True, "samples": self.samples, "device_baseline_mb": self.baseline_mb,
                "device_peak_mb": self.peak_device_mb, "device_delta_mb": delta, "pid_peak_mb": self.peak_pid_mb,
                "budget_mb": CUDA_MEM_LIMIT / 1024 ** 2}


def run_all(out_dir: Path | str, ep: str = "cpu", ticks: int = DEFAULT_TICKS, warmup: int = 10,
            iters: int = 100, seed: int = DEFAULT_SEED, threads: int = 0) -> dict[str, Any]:
    """Export, parity, op report and bench (with PyTorch CPU reference) in one process."""
    import onnx
    import onnxruntime as ort

    torch = _need_torch()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = build_net(seed)
    from python.train.scene_net import count_module_parameters

    onnx_path = export(net, out_dir / "scene_net_bench.onnx")
    parity = check_parity(onnx_path, net, ep=ep, ticks=ticks, seed=seed)
    ops = op_report(onnx_path, ep=ep, workdir=out_dir)
    if ep == "cuda":
        with VramSampler() as vram:
            perf = bench(onnx_path, net, ep=ep, warmup=warmup, iters=iters, threads=threads)
        perf["vram"] = vram.report()
    else:
        perf = bench(onnx_path, net, ep=ep, warmup=warmup, iters=iters, threads=threads)
    return {
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "ort_device": ort.get_device(),
                     "ort_available_providers": ort.get_available_providers()},
        "model": {"params": int(count_module_parameters(net)), "seed": int(seed), "opset": SCENE_OPSET,
                  "onnx_bytes": onnx_path.stat().st_size,
                  "inputs": {i.name: list(i.shape) for i in make_session(onnx_path, "cpu").get_inputs()}},
        "parity": parity.as_dict(),
        "ops": ops,
        "bench": perf,
    }


def run_ort_only(onnx_path: Path | str, ep: str = "cpu", warmup: int = 10, iters: int = 100,
                 threads: int = 0, workdir: Path | None = None) -> dict[str, Any]:
    """Op report and bench of an existing ONNX without PyTorch (deploy-path RSS/VRAM)."""
    import onnxruntime as ort

    ops = op_report(onnx_path, ep=ep, workdir=workdir)
    if ep == "cuda":
        with VramSampler() as vram:
            perf = bench(onnx_path, None, ep=ep, warmup=warmup, iters=iters, threads=threads)
        perf["vram"] = vram.report()
    else:
        perf = bench(onnx_path, None, ep=ep, warmup=warmup, iters=iters, threads=threads)
    return {
        "versions": {"python": sys.version.split()[0], "onnxruntime": ort.__version__, "ort_device": ort.get_device(),
                     "ort_available_providers": ort.get_available_providers(), "torch_imported": sys.modules.get("torch") is not None},
        "model": {"onnx": str(onnx_path), "onnx_bytes": Path(onnx_path).stat().st_size},
        "ops": ops,
        "bench": perf,
    }
