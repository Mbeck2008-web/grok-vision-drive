"""GPU-aware SceneNet sizes.

``small`` is today's net: the stem widths and the GRU width stay the module
constants, and ``scene_net_kwargs("small")`` is empty. ``auto`` picks the
largest preset whose frame time fits the latency budget. VRAM, system RAM,
and CPU threads are a sanity check after that, with BeamNG reserved first.

Per-frame times are estimates unless a startup micro-benchmark is cached for
this GPU and driver. A measured BeamNG VRAM reading wins over the per-tier
default when ``--game-vram-reserve-gb`` is not set. The flag wins over both.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PRESET_ORDER = ("tiny", "small", "medium", "large", "xl")

# Host RAM left for BeamNG, the OS, and the rest of the supervisor.
GAME_RAM_HEADROOM_GB = 8.0
# Below this many logical CPUs, auto will not pass ``small``.
CPU_THREADS_FOR_LARGER = 8
# Free VRAM kept back from the vision budget when a live reading exists.
FREE_VRAM_MARGIN_GB = 0.5
# Runtime step-down. Any one of these moves one size down.
FREE_VRAM_FLOOR_GB = 0.75
GAME_FPS_FLOOR = 30.0
# Unmeasured anchor: small, fp32, compute 6.1, long side 640, GPU to ourselves.
# Not a benchmark. 11.34 TFLOPS is the GTX 1080 Ti boost fp32 from the GeForce 10 table.
SMALL_FP32_MS = 25.0
ANCHOR_FP32_TFLOPS = 11.34
# Share of the GPU left with BeamNG while the net runs. Frame time is divided by the rest.
GAME_COMPUTE_SHARE = 0.35
LIVE_LONG_SIDE = 640


@dataclass(frozen=True)
class Preset:
    """One architecture. ``stem`` / ``gru`` of None means today's constants."""

    name: str
    stem: tuple[int, ...] | None
    gru: int | None
    long_side: int
    est_vram_gb: float
    est_host_gb: float


PRESETS: dict[str, Preset] = {
    "tiny": Preset("tiny", (16, 32, 64, 64), 256, 512, 1.0, 0.4),
    "small": Preset("small", None, None, 640, 2.2, 0.6),
    "medium": Preset("medium", (48, 96, 160, 160), 512, 640, 4.8, 1.2),
    "large": Preset("large", (64, 128, 192, 192), 768, 768, 7.5, 2.0),
    "xl": Preset("xl", (96, 160, 256, 256), 1024, 768, 12.0, 3.5),
}


@dataclass(frozen=True)
class GpuFacts:
    """What startup could see. Tests pass this in; they do not need a device."""

    name: str = "cpu"
    vram_gb: float = 0.0
    capability: tuple[int, int] | None = None
    free_vram_gb: float | None = None
    ram_gb: float | None = None
    cpu_count: int | None = None


@dataclass
class SizePlan:
    """The size a run will build, and the sentence that explains it."""

    preset: str
    auto_preset: str
    precision: str
    long_side: int
    game_reserve_gb: float
    reserve_source: str
    vision_budget_gb: float
    est_vram_gb: float
    est_ms: float
    requested: str
    warnings: list[str] = field(default_factory=list)
    reason: str = ""
    prefer_precision: str = "fp32"
    backend: str = "eager"
    limit_ms: float = 100.0
    bench_source: str = "estimate"
    tensor_cores: bool = False

    @property
    def net_kwargs(self) -> dict:
        """Keyword args for ``SceneNet``. Empty for ``small``."""
        return scene_net_kwargs(self.preset)

    @property
    def line(self) -> str:
        hz = 1000.0 / self.est_ms if self.est_ms > 0 else 0.0
        return (
            f"[GVD] model-size {self.preset} ({self.reason}). "
            f"auto would pick {self.auto_preset}; "
            f"precision {self.precision} (prefer {self.prefer_precision}), backend {self.backend}; "
            f"~{self.est_ms:.0f} ms vs {self.limit_ms:.0f} ms budget ({self.bench_source}); "
            f"BeamNG reserve {self.game_reserve_gb:.2f} GB ({self.reserve_source}); "
            f"vision budget {self.vision_budget_gb:.2f} GB; "
            f"estimate {self.est_vram_gb:.2f} GB, ~{hz:.0f} Hz"
        )


class PresetMismatch(ValueError):
    """The checkpoint was built for a different preset than this run asked for."""


def scene_net_kwargs(name: str) -> dict:
    """Constructor kwargs. ``small`` returns ``{}`` so the default call is unchanged."""
    preset = PRESETS.get(str(name or "small"))
    if preset is None:
        known = ", ".join(PRESET_ORDER)
        raise ValueError(f"unknown model size {name!r}; expected one of {known}")
    if preset.name == "small":
        return {}
    kwargs: dict = {"preset": preset.name}
    if preset.stem is not None:
        kwargs["stem_channels"] = preset.stem
    if preset.gru is not None:
        kwargs["gru_hidden"] = preset.gru
    return kwargs


def checkpoint_preset(meta: dict | None) -> str:
    """A checkpoint with no preset field is ``small``."""
    if not isinstance(meta, dict):
        return "small"
    raw = meta.get("preset")
    if raw in (None, ""):
        return "small"
    name = str(raw)
    if name not in PRESETS:
        raise PresetMismatch(
            f"checkpoint preset {name!r} is not a GVD size "
            f"({', '.join(PRESET_ORDER)}); a file with no preset field is small"
        )
    return name


def refuse_preset_mismatch(requested: str | None, meta: dict | None) -> None:
    """Raise when a pinned size does not match the file. ``auto`` is resolved by the caller."""
    if requested in (None, "", "auto"):
        return
    have = checkpoint_preset(meta)
    want = str(requested)
    if want not in PRESETS:
        raise PresetMismatch(f"requested model size {want!r} is not one of {', '.join(PRESET_ORDER)}")
    if want != have:
        raise PresetMismatch(
            f"checkpoint preset is {have!r} (a file with no preset field is small); "
            f"this run asked for {want!r}"
        )


def default_game_reserve_gb(vram_gb: float) -> float:
    """Conservative BeamNG VRAM reserve for a card of this size. Not a measurement."""
    vram = float(vram_gb or 0.0)
    if vram <= 4:
        return 2.5
    if vram <= 6:
        return 3.5
    if vram <= 8:
        return 4.5
    if vram <= 12:
        return 6.0
    if vram <= 16:
        return 7.5
    if vram <= 24:
        return 8.5
    return 10.0


def default_vision_cap_gb(vram_gb: float) -> float:
    """Vision-infer cap. An 11 GB card stays at 4 GB."""
    vram = float(vram_gb or 0.0)
    if vram <= 0:
        return 0.0
    if vram <= 6:
        return 1.6
    if vram <= 8:
        return 2.5
    if vram <= 12:
        return 4.0
    if vram <= 16:
        return 6.0
    if vram <= 24:
        return 8.0
    return 12.0


def _capability_factor(capability: tuple[int, int] | None) -> float:
    """Rough fp32 throughput vs a 6.1 card. Estimate, not a spec-sheet TFLOP."""
    if capability is None:
        return 1.0
    major, minor = int(capability[0]), int(capability[1])
    if major >= 12:
        return 16.0
    if major >= 10:
        return 12.0
    if major >= 9:
        return 8.0
    if major == 8 and minor >= 9:
        return 6.0
    if major == 8 and minor >= 6:
        return 4.0
    if major == 8:
        return 3.0
    if major == 7 and minor >= 5:
        return 1.8
    if major == 7:
        return 1.5
    return 1.0


def estimate_ms(
    preset: Preset,
    capability: tuple[int, int] | None,
    precision: str,
    *,
    long_side: int | None = None,
    tflops: float | None = None,
    name: str = "",
) -> float:
    """Milliseconds per frame at ``long_side`` while BeamNG keeps a share of the GPU.

    The 25 ms anchor is unmeasured. ``tflops`` replaces the capability factor when
    a startup micro-benchmark measured the card. Tensor-core precisions use a
    0.6 multiplier that is a guess, not a timed kernel.
    """
    if tflops is not None and float(tflops) > 0:
        factor = float(tflops) / ANCHOR_FP32_TFLOPS
    else:
        factor = _capability_factor(capability)
    relative = float(preset.est_vram_gb) / PRESETS["small"].est_vram_gb
    side = LIVE_LONG_SIDE if not long_side else int(long_side)
    spatial = (float(side) / float(preset.long_side)) ** 2
    ms = SMALL_FP32_MS * relative * spatial / max(factor, 1e-6)
    tensor = precision in ("fp16", "bf16", "tf32") and capability is not None and capability >= (7, 0)
    if tensor and not _gtx_16_series(name):
        ms *= 0.6
    share = min(0.9, max(0.0, GAME_COMPUTE_SHARE))
    return ms / (1.0 - share)


def vram_cost(preset: Preset, *, batch: int, long_side: int | None) -> float:
    side = int(long_side) if long_side else int(preset.long_side)
    scale = (float(side) / float(preset.long_side)) ** 2
    copies = 1 if batch is None or int(batch) < 1 else int(batch)
    return float(preset.est_vram_gb) * scale * copies


def _reserve(facts: GpuFacts, game_reserve_gb: float | None, measured: float | None) -> tuple[float, str]:
    if game_reserve_gb is not None:
        return max(0.0, float(game_reserve_gb)), "flag"
    if measured is not None:
        return max(0.0, float(measured)), "measured"
    return default_game_reserve_gb(facts.vram_gb), "tier default"


def _budget(
    facts: GpuFacts,
    reserve_gb: float,
    vram_budget_gb: float | None,
    free_vram_gb: float | None,
) -> float:
    if float(facts.vram_gb or 0.0) <= 0.0:
        room = 0.0
    else:
        room = max(0.0, float(facts.vram_gb) - float(reserve_gb))
    cap = float(vram_budget_gb) if vram_budget_gb is not None else default_vision_cap_gb(facts.vram_gb)
    budget = min(room, max(0.0, cap))
    if free_vram_gb is not None:
        budget = min(budget, max(0.0, float(free_vram_gb) - FREE_VRAM_MARGIN_GB))
    return budget


def _host_ok(preset: Preset, facts: GpuFacts) -> bool:
    if facts.ram_gb is None:
        return True
    return float(preset.est_host_gb) <= float(facts.ram_gb) - GAME_RAM_HEADROOM_GB


def _cpu_ok(preset: Preset, facts: GpuFacts) -> bool:
    if facts.cpu_count is None or int(facts.cpu_count) >= CPU_THREADS_FOR_LARGER:
        return True
    return PRESET_ORDER.index(preset.name) <= PRESET_ORDER.index("small")


def _fits(
    preset: Preset,
    facts: GpuFacts,
    budget: float,
    *,
    batch: int,
    long_side: int | None,
    frame_ms: float,
    limit_ms: float | None,
) -> bool:
    """Latency first. VRAM, RAM, and CPU are the sanity check."""
    if limit_ms is not None and limit_ms > 0 and frame_ms > float(limit_ms) + 1e-6:
        return False
    if vram_cost(preset, batch=batch, long_side=long_side) > budget + 1e-6:
        return False
    if not _host_ok(preset, facts):
        return False
    if not _cpu_ok(preset, facts):
        return False
    return True


def select_model(
    requested: str | None,
    facts: GpuFacts | None = None,
    *,
    vram_budget_gb: float | None = None,
    game_reserve_gb: float | None = None,
    measured_beamng_vram_gb: float | None = None,
    free_vram_gb: float | None = None,
    long_side: int | None = None,
    batch: int = 1,
    precision: str = "auto",
    target_hz: float = 10.0,
    max_infer_ms: float | None = None,
    bench_ms: dict[str, float] | None = None,
    tflops: float | None = None,
) -> SizePlan:
    """Pick a preset. ``small`` (the default) does not resize the net.

    ``auto`` takes the largest preset whose frame time is inside the latency
    budget. A measured ``bench_ms`` map wins over the static estimate. VRAM
    only drops a preset that would crowd BeamNG off the card.
    """
    facts = facts or GpuFacts()
    want = str(requested or "small")
    if want not in ("auto",) and want not in PRESETS:
        raise ValueError(f"unknown model size {want!r}")
    explicit = str(precision or "auto").lower()
    # The default small path must not import torch. run_vision's stub loop
    # checks that. Auto and an explicit precision are allowed to probe.
    if want == "small" and explicit in ("", "auto") and not bench_ms:
        accel = AccelChoice("fp32", "fp32", (), False, "eager", "eager", ())
        applied = "fp32"
        prec_warnings = []
    elif explicit in ("", "auto"):
        accel = pick_accel(facts.capability, facts.name)
        applied = accel.applied_precision
        prec_warnings = list(accel.notes) if want == "auto" else []
    else:
        accel = pick_accel(facts.capability, facts.name)
        applied, prec_warnings = resolve_precision(precision, facts.capability, facts.name)
    reserve, source = _reserve(facts, game_reserve_gb, measured_beamng_vram_gb)
    free = facts.free_vram_gb if free_vram_gb is None else free_vram_gb
    budget = _budget(facts, reserve, vram_budget_gb, free)
    warnings = list(prec_warnings)
    if long_side is not None and int(long_side) > 1920:
        warnings.append(f"live input long side {int(long_side)} is above 1920")
    if max_infer_ms is not None and float(max_infer_ms) > 0:
        limit_ms = float(max_infer_ms)
    elif target_hz and float(target_hz) > 0:
        limit_ms = 1000.0 / float(target_hz)
    else:
        limit_ms = None
    bench = {str(key): float(value) for key, value in (bench_ms or {}).items()}
    bench_source = "benchmark" if bench else "estimate"

    def frame_ms(name: str) -> float:
        if name in bench:
            return bench[name]
        return estimate_ms(
            PRESETS[name],
            facts.capability,
            applied,
            long_side=long_side,
            tflops=tflops,
            name=facts.name,
        )

    def cost_of(name: str) -> float:
        return vram_cost(PRESETS[name], batch=batch, long_side=long_side)

    fitting = [
        name
        for name in PRESET_ORDER
        if _fits(
            PRESETS[name],
            facts,
            budget,
            batch=batch,
            long_side=long_side,
            frame_ms=frame_ms(name),
            limit_ms=limit_ms,
        )
    ]
    auto_name = fitting[-1] if fitting else "tiny"
    if not fitting and want == "auto":
        warnings.append(
            f"no preset meets {limit_ms:.0f} ms and a {budget:.2f} GB vision budget "
            f"after reserving {reserve:.2f} GB for BeamNG; auto stays on tiny"
            if limit_ms
            else f"no preset fits a {budget:.2f} GB vision budget; auto stays on tiny"
        )
    if want == "auto":
        chosen = auto_name
        limit_txt = f"{limit_ms:.0f} ms" if limit_ms else "no latency cap"
        reason = (
            f"auto, largest preset at or under {limit_txt} "
            f"({bench_source}); VRAM sanity {budget:.2f} GB after {reserve:.2f} GB for BeamNG"
        )
    else:
        chosen = want
        reason = "default small; today's net" if want == "small" else "manual"
        if want != "small" and limit_ms is not None and frame_ms(want) > limit_ms + 1e-6:
            warnings.append(
                f"{want} estimates {frame_ms(want):.0f} ms and the latency budget is {limit_ms:.0f} ms"
            )
        if want != "small" and cost_of(want) > budget + 1e-6:
            warnings.append(
                f"{want} estimates {cost_of(want):.2f} GB and the vision budget is "
                f"{budget:.2f} GB after reserving {reserve:.2f} GB for BeamNG"
            )
        if want != "small" and not _host_ok(PRESETS[want], facts):
            warnings.append(
                f"{want} estimates {PRESETS[want].est_host_gb:.1f} GB of process RAM; "
                f"keeping {GAME_RAM_HEADROOM_GB:.0f} GB for BeamNG leaves less than that"
            )
        if want != "small" and not _cpu_ok(PRESETS[want], facts):
            warnings.append(
                f"{want} on {facts.cpu_count} logical CPUs leaves BeamNG short of threads; "
                f"auto would stop at small"
            )
    preset = PRESETS[chosen]
    side = int(long_side) if long_side else int(preset.long_side)
    return SizePlan(
        preset=chosen,
        auto_preset=auto_name,
        precision=applied,
        long_side=side,
        game_reserve_gb=reserve,
        reserve_source=source,
        vision_budget_gb=budget,
        est_vram_gb=cost_of(chosen),
        est_ms=frame_ms(chosen),
        requested=want,
        warnings=warnings,
        reason=reason,
        prefer_precision=accel.prefer_precision,
        backend=accel.backend_applied,
        limit_ms=float(limit_ms or 0.0),
        bench_source=bench_source,
        tensor_cores=accel.tensor_cores,
    )


def _gtx_16_series(name: str) -> bool:
    """TU116 / TU117. Compute 7.5, and no tensor cores."""
    text = str(name or "").lower().replace("-", " ")
    return any(token in text for token in ("gtx 16", "gtx16", "1630", "1650", "1660"))


@dataclass(frozen=True)
class AccelChoice:
    """Fastest precision the card can run, and the backend that will actually be used."""

    prefer_precision: str
    applied_precision: str
    tensor_precisions: tuple[str, ...]
    tensor_cores: bool
    backend_prefer: str
    backend_applied: str
    notes: tuple[str, ...]


def tensor_core_precisions(capability: tuple[int, int] | None, name: str = "") -> tuple[str, ...]:
    """Precisions the tensor cores on this chip advertise. Empty when there are none.

    Pascal has no tensor cores. GTX 16-series is compute 7.5 and also has none.
    FP8 is Ada (8.9) and newer. FP4 is Blackwell (10.0 / 12.0).
    """
    if capability is None or _gtx_16_series(name) or capability < (7, 0):
        return ()
    if capability >= (10, 0):
        return ("fp4", "fp8", "bf16", "tf32", "fp16", "int8")
    if capability >= (8, 9):
        return ("fp8", "bf16", "tf32", "fp16", "int8")
    if capability >= (8, 0):
        return ("bf16", "tf32", "fp16", "int8", "int4")
    if capability >= (7, 5):
        return ("fp16", "int8", "int4")
    return ("fp16",)


def _runnable_precision(capability: tuple[int, int] | None, name: str) -> str:
    """The fastest precision this tree will actually execute.

    INT8, FP8, and FP4 need a calibrated engine that is not built here.
    bf16 is Ampere and newer. fp16 tensor cores start at Volta, never on Pascal
    and never on a GTX 16-series name.
    """
    if capability is None or capability < (7, 0) or _gtx_16_series(name):
        return "fp32"
    if capability >= (8, 0):
        return "bf16"
    return "fp16"


def _prefer_precision(capability: tuple[int, int] | None, name: str) -> str:
    """Fastest advertised path, including ones this tree does not execute."""
    if capability is None:
        return "fp32"
    if capability < (7, 0) or _gtx_16_series(name):
        return "int8"
    precisions = tensor_core_precisions(capability, name)
    for cand in ("fp4", "fp8", "int8", "bf16", "fp16"):
        if cand in precisions:
            return cand
    return "fp32"


def _probe_backend() -> tuple[str, str, tuple[str, ...]]:
    """Prefer TensorRT, then ONNX Runtime, torch.compile, CUDA graphs, eager.

    A missing package or a missing engine is a note, not an exception.
    TensorRT and ONNX Runtime are not applied: this tree has no scene engine.
    """
    notes: list[str] = []
    try:
        import tensorrt  # noqa: F401

        notes.append("tensorrt imports, and no calibrated scene engine is built; not using it")
    except Exception:
        notes.append("tensorrt is not installed; skipping it")
    try:
        import onnxruntime  # noqa: F401

        notes.append("onnxruntime imports, and no scene-net session is built; not using it")
    except Exception:
        notes.append("onnxruntime is not installed; skipping it")
    compile_ok = False
    cuda_ok = False
    try:
        import torch

        compile_ok = hasattr(torch, "compile")
        cuda_ok = bool(torch.cuda.is_available())
    except Exception as exc:
        notes.append(f"torch backend probe failed ({exc})")
    if compile_ok:
        return "torch_compile", "torch_compile", tuple(notes)
    if cuda_ok:
        notes.append("CUDA graphs need --exp-cuda-graph to capture; this picker does not capture one")
        return "cuda_graph", "eager", tuple(notes)
    return "eager", "eager", tuple(notes)


def pick_accel(capability: tuple[int, int] | None, name: str = "") -> AccelChoice:
    """Auto precision and backend. Never raises."""
    try:
        prefer = _prefer_precision(capability, name)
        applied = _runnable_precision(capability, name)
        precisions = tensor_core_precisions(capability, name)
        prefer_backend, applied_backend, notes = _probe_backend()
        extra: list[str] = []
        if prefer != applied:
            if applied == "fp32" and (capability is None or capability < (7, 0)):
                extra.append(
                    "Pascal keeps fp32; INT8 via DP4A is the fast path and no calibrated engine is built"
                )
            elif _gtx_16_series(name):
                extra.append("GTX 16-series has no tensor cores; auto stays fp32")
            elif prefer in ("fp8", "fp4", "int8"):
                extra.append(
                    f"{prefer} is the fast path on this card and no calibrated engine is built; "
                    f"running {applied}"
                )
        return AccelChoice(
            prefer_precision=prefer,
            applied_precision=applied,
            tensor_precisions=precisions,
            tensor_cores=bool(precisions),
            backend_prefer=prefer_backend,
            backend_applied=applied_backend,
            notes=tuple(extra + list(notes)),
        )
    except Exception as exc:
        return AccelChoice("fp32", "fp32", (), False, "eager", "eager", (f"accel probe skipped ({exc})",))


def resolve_precision(
    choice: str | None,
    capability: tuple[int, int] | None,
    name: str = "",
) -> tuple[str, list[str]]:
    """Resolve a precision flag. ``auto`` uses :func:`pick_accel`.

    Pascal and GTX 16-series never apply fp16. INT8, FP8, and FP4 are accepted
    as requests and not executed.
    """
    picked = str(choice or "auto").lower()
    if picked in ("", "auto"):
        accel = pick_accel(capability, name)
        return accel.applied_precision, list(accel.notes)
    if picked == "fp32":
        return "fp32", []
    if picked == "fp16":
        if _gtx_16_series(name) or capability is None or capability < (7, 0):
            why = (
                "GTX 16-series has no tensor cores; fp16 stays fp32"
                if _gtx_16_series(name)
                else "fp16 is not applied below compute capability 7.0; Pascal stays fp32"
            )
            return "fp32", [why]
        return "fp16", []
    if picked == "bf16":
        if capability is None or capability < (8, 0) or _gtx_16_series(name):
            return "fp32", ["bf16 needs compute capability 8.0 or newer; staying on fp32"]
        return "bf16", []
    if picked == "int8":
        return "fp32", [
            "int8 was requested; GVD has no calibrated INT8 engine, so this run stays fp32"
        ]
    if picked == "fp8":
        applied = _runnable_precision(capability, name)
        if capability is None or capability < (8, 9):
            return applied if applied != "fp8" else "fp32", [
                "fp8 needs Ada (8.9) or newer; staying on the runnable precision"
            ]
        return applied, ["fp8 was requested and no FP8 kernel is built; staying on " + applied]
    raise ValueError(
        f"unknown precision {choice!r}; expected auto, fp32, fp16, bf16, int8, or fp8"
    )


def prepare_backend(net: object, backend: str) -> tuple[object, str, list[str]]:
    """Try the chosen backend. Any failure returns the original net and eager."""
    try:
        if backend == "torch_compile":
            import torch

            if not hasattr(torch, "compile"):
                return net, "eager", ["torch.compile is missing; staying on eager"]
            return torch.compile(net), "torch_compile", []
        if backend == "cuda_graph":
            import torch

            if not torch.cuda.is_available():
                return net, "eager", ["CUDA graphs need a CUDA device; staying on eager"]
            return net, "eager", [
                "CUDA graph capture stays on --exp-cuda-graph; this picker does not capture a graph"
            ]
        if backend == "tensorrt":
            return net, "eager", ["no TensorRT engine is built; staying on eager"]
        if backend == "onnxruntime":
            return net, "eager", ["no ONNX Runtime session is built for this preset; staying on eager"]
        return net, "eager", []
    except Exception as exc:
        return net, "eager", [f"{backend} failed ({exc}); staying on eager"]


def apply_runtime_knobs(
    capability: tuple[int, int] | None,
    name: str = "",
    *,
    enabled: bool,
) -> list[str]:
    """Turn on TF32 for Ampere and newer when auto-size asked for it. Never raises."""
    if not enabled or capability is None or capability < (8, 0) or _gtx_16_series(name):
        return []
    try:
        import torch

        if not torch.cuda.is_available():
            return ["TF32 not applied; no CUDA device"]
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return ["TF32 matmul on"]
    except Exception as exc:
        return [f"TF32 not applied ({exc})"]


def step_down_for_pressure(
    preset: str,
    *,
    free_vram_gb: float | None = None,
    game_fps: float | None = None,
    infer_ms: float | None = None,
    loop_hz: float | None = None,
    max_infer_ms: float | None = None,
    target_hz: float | None = None,
    free_floor_gb: float = FREE_VRAM_FLOOR_GB,
    fps_floor: float = GAME_FPS_FLOOR,
) -> tuple[str, bool, str]:
    """Move one size down when latency, loop rate, BeamNG fps, or free VRAM gives way.

    Returns ``(preset, stepped, reason)``. Never raises.
    """
    current = str(preset or "small")
    if current not in PRESETS:
        current = "small"
    try:
        reasons: list[str] = []
        if infer_ms is not None and max_infer_ms is not None and float(infer_ms) > float(max_infer_ms):
            reasons.append(
                f"infer latency {float(infer_ms):.1f} ms is over {float(max_infer_ms):.1f} ms"
            )
        if (
            loop_hz is not None
            and target_hz is not None
            and float(loop_hz) > 0
            and float(target_hz) > 0
            and float(loop_hz) < float(target_hz)
        ):
            reasons.append(f"loop {float(loop_hz):.1f} Hz is under target {float(target_hz):.1f} Hz")
        if free_vram_gb is not None and float(free_vram_gb) < float(free_floor_gb):
            reasons.append(
                f"free VRAM {float(free_vram_gb):.2f} GB is under {float(free_floor_gb):.2f} GB"
            )
        if game_fps is not None and float(game_fps) < float(fps_floor):
            reasons.append(f"BeamNG fps {float(game_fps):.1f} is under {float(fps_floor):.1f}")
        if not reasons:
            return current, False, "headroom ok"
        index = PRESET_ORDER.index(current)
        if index <= 0:
            return current, False, "; ".join(reasons) + "; already on tiny"
        nxt = PRESET_ORDER[index - 1]
        return nxt, True, "; ".join(reasons)
    except Exception as exc:
        return current, False, f"pressure check skipped ({exc})"


def read_free_vram_gb() -> float | None:
    """Free VRAM in GB from nvidia-smi, or None when the tool is missing."""
    try:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=0.4,
            stderr=subprocess.DEVNULL,
        )
        line = out.strip().splitlines()[0].strip()
        return float(line) / 1024.0
    except Exception:
        return None


def _system_ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return float(pages) * float(size) / (1024.0 ** 3)
    except (OSError, ValueError, AttributeError):
        return None


def _measured_from_env() -> float | None:
    raw = os.environ.get("GVD_BEAMNG_VRAM_GB", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def detect_gpu() -> GpuFacts:
    """torch.cuda, then nvidia-smi, then a CPU record. Never raises."""
    name = "cpu"
    vram = 0.0
    cap: tuple[int, int] | None = None
    free: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name = str(getattr(props, "name", "cuda"))
            total = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
            vram = float(total) / (1024.0 ** 3)
            raw_cap = torch.cuda.get_device_capability(0)
            cap = (int(raw_cap[0]), int(raw_cap[1]))
            try:
                free_b, _total_b = torch.cuda.mem_get_info()
                free = float(free_b) / (1024.0 ** 3)
            except Exception:
                free = None
    except Exception:
        name, vram, cap, free = "cpu", 0.0, None, None
    if name == "cpu":
        smi = _nvidia_smi_facts()
        if smi is not None:
            name, vram, free = smi
    return GpuFacts(
        name=name,
        vram_gb=vram,
        capability=cap,
        free_vram_gb=free,
        ram_gb=_system_ram_gb(),
        cpu_count=os.cpu_count(),
    )


def _nvidia_smi_facts() -> tuple[str, float, float | None] | None:
    try:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=0.4,
            stderr=subprocess.DEVNULL,
        )
        parts = [part.strip() for part in out.strip().splitlines()[0].split(",")]
        if len(parts) < 2:
            return None
        total = float(parts[1]) / 1024.0
        free = float(parts[2]) / 1024.0 if len(parts) > 2 else None
        return parts[0], total, free
    except Exception:
        return None


def add_model_size_args(parser: argparse.ArgumentParser) -> None:
    """Shared flags. The default size is ``small`` so today's net stays today's net."""
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["auto", *PRESET_ORDER],
        help="SceneNet preset. small is today's net. auto picks the largest preset inside the latency budget, then checks VRAM.",
    )
    parser.add_argument(
        "--vram-budget-gb",
        type=float,
        default=None,
        help="Vision VRAM cap in GB. Default is 4 on an 11 GB card, after the BeamNG reserve.",
    )
    parser.add_argument(
        "--game-vram-reserve-gb",
        type=float,
        default=None,
        help="VRAM held back for BeamNG. Default is a conservative per-tier estimate, or GVD_BEAMNG_VRAM_GB when that is set.",
    )
    parser.add_argument(
        "--live-input-long-side",
        type=int,
        default=0,
        help="Camera long side in pixels for the live resize. 0 keeps the preset or the hardware yaml.",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "int8", "fp8"],
        help="auto picks the fastest precision this tree can run. Pascal and GTX 16-series stay fp32. fp8 and int8 fall back when no engine is built.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=10.0,
        help="Auto keeps the largest preset whose estimate is at least this fast. --max-infer-ms wins when both are set.",
    )
    parser.add_argument(
        "--max-infer-ms",
        type=float,
        default=None,
        help="Latency budget in milliseconds. Overrides --target-hz for preset selection and for the runtime step-down.",
    )


def _bench_cache_path() -> Path:
    root = os.environ.get("GVD_DATA_DIR") or os.environ.get("XDG_CACHE_HOME")
    if not root:
        root = str(Path.home() / ".cache")
    return Path(root) / "gvd" / "model_size_bench.json"


def _driver_version() -> str:
    try:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return "unknown"
        out = subprocess.check_output(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=0.4,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0].strip() or "unknown"
    except Exception:
        return "unknown"


def _bench_identity(facts: GpuFacts) -> dict[str, str]:
    cap = "" if facts.capability is None else f"{facts.capability[0]}.{facts.capability[1]}"
    return {"gpu": facts.name, "capability": cap, "driver": _driver_version()}


def _read_bench_cache(facts: GpuFacts) -> dict[str, float] | None:
    path = _bench_cache_path()
    if not path.is_file():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        return None
    if blob.get("identity") != _bench_identity(facts):
        return None
    presets = blob.get("presets")
    if not isinstance(presets, dict):
        return None
    return {str(key): float(value) for key, value in presets.items() if key in PRESETS}


def _write_bench_cache(facts: GpuFacts, presets: dict[str, float]) -> None:
    path = _bench_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"identity": _bench_identity(facts), "presets": presets}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def measure_card_tflops() -> float | None:
    """TFLOP/s of one small CUDA conv. None when CUDA is missing. Never raises."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda")
        layer = torch.nn.Conv2d(3, 32, 3, padding=1).to(device)
        data = torch.zeros(1, 3, 32, 64, device=device)
        for _ in range(2):
            layer(data)
        torch.cuda.synchronize()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        iters = 8
        start.record()
        for _ in range(iters):
            layer(data)
        end.record()
        torch.cuda.synchronize()
        kernel_ms = float(start.elapsed_time(end)) / float(iters)
        if kernel_ms <= 0:
            return None
        # One output element is 3*3*3 MACs, and a MAC is two flops.
        flops = 32 * 64 * 32 * 3 * 3 * 3 * 2
        return (flops / (kernel_ms / 1e3)) / 1e12
    except Exception:
        return None


def lookup_benchmark(facts: GpuFacts) -> dict[str, float] | None:
    """Cached per GPU and driver. A miss times the card once, then stores the five presets.

    CPU and any probe error return None so the static estimate is used.
    """
    try:
        cached = _read_bench_cache(facts)
        if cached:
            return cached
        tflops = measure_card_tflops()
        if tflops is None:
            return None
        accel = pick_accel(facts.capability, facts.name)
        presets = {
            name: estimate_ms(
                PRESETS[name],
                facts.capability,
                accel.applied_precision,
                tflops=tflops,
                name=facts.name,
            )
            for name in PRESET_ORDER
        }
        try:
            _write_bench_cache(facts, presets)
        except Exception:
            pass
        return presets
    except Exception:
        return None


def _sizing_requested(ns: argparse.Namespace) -> bool:
    """True when this process asked for something other than today's net."""
    if str(getattr(ns, "model_size", "small") or "small") != "small":
        return True
    if str(getattr(ns, "precision", "auto") or "auto") != "auto":
        return True
    if getattr(ns, "max_infer_ms", None) not in (None, 0, 0.0):
        return True
    if getattr(ns, "vram_budget_gb", None) is not None:
        return True
    if getattr(ns, "game_vram_reserve_gb", None) is not None:
        return True
    return False


def plan_from_namespace(
    ns: argparse.Namespace,
    facts: GpuFacts | None = None,
    *,
    detect: bool | None = None,
) -> SizePlan:
    """Build a plan from CLI args. Detects the GPU when sizing was requested.

    ``detect=False`` skips torch and nvidia-smi. The live supervisor uses that
    for the default ``small`` path so a stub loop does not import torch.
    """
    side = int(getattr(ns, "live_input_long_side", 0) or 0)
    if side <= 0:
        side = int(getattr(ns, "live_long_side", 0) or 0)
    batch = getattr(ns, "batch", None)
    if batch is None:
        batch = getattr(ns, "infer_batch", 1)
    if detect is None:
        detect = _sizing_requested(ns)
    if facts is not None:
        seen = facts
    elif detect:
        seen = detect_gpu()
    else:
        seen = GpuFacts()
    bench = None
    if str(getattr(ns, "model_size", "small") or "small") == "auto":
        bench = lookup_benchmark(seen)
    return select_model(
        getattr(ns, "model_size", "small"),
        seen,
        vram_budget_gb=getattr(ns, "vram_budget_gb", None),
        game_reserve_gb=getattr(ns, "game_vram_reserve_gb", None),
        measured_beamng_vram_gb=_measured_from_env(),
        long_side=side or None,
        batch=int(batch or 1),
        precision=str(getattr(ns, "precision", "auto") or "auto"),
        target_hz=float(getattr(ns, "target_hz", 10.0) or 10.0),
        max_infer_ms=getattr(ns, "max_infer_ms", None),
        bench_ms=bench,
    )
