"""Preset selection, BeamNG reserve, and the runtime step-down. No GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.train.model_size import (  # noqa: E402
    PRESET_ORDER,
    GpuFacts,
    PresetMismatch,
    checkpoint_preset,
    pick_accel,
    prepare_backend,
    refuse_preset_mismatch,
    resolve_precision,
    scene_net_kwargs,
    select_model,
    step_down_for_pressure,
)


def _gpu(name: str, vram: float, major: int, minor: int, **extra) -> GpuFacts:
    return GpuFacts(name=name, vram_gb=vram, capability=(major, minor), **extra)


def check_mocked_gpus() -> None:
    ti = select_model("auto", _gpu("GeForce GTX 1080 Ti", 11, 6, 1))
    low = select_model("auto", _gpu("GeForce GTX 1060", 6, 6, 1))
    big = select_model("auto", _gpu("GeForce RTX 3090", 24, 8, 6))
    ada = select_model("auto", _gpu("GeForce RTX 4090", 24, 8, 9))
    cpu = select_model("auto", GpuFacts())
    assert ti.preset == "small", ti
    assert ti.net_kwargs == {}
    assert ti.reserve_source == "tier default"
    assert ti.game_reserve_gb == 6.0
    assert low.preset == "tiny", low
    assert big.preset == "large", big
    assert ada.preset == "large", ada
    assert cpu.preset == "tiny", cpu
    assert PRESET_ORDER.index(low.preset) < PRESET_ORDER.index(ti.preset) <= PRESET_ORDER.index(big.preset)


def check_default_stays_small() -> None:
    plan = select_model("small", _gpu("GeForce RTX 3090", 24, 8, 6))
    assert plan.preset == "small"
    assert plan.net_kwargs == {}
    assert scene_net_kwargs("small") == {}
    assert plan.auto_preset == "large"
    assert plan.warnings == []


def check_game_reserve_knob() -> None:
    card = _gpu("GeForce GTX 1080 Ti", 11, 6, 1)
    held = select_model("auto", card, game_reserve_gb=9.5)
    assert held.reserve_source == "flag"
    assert held.game_reserve_gb == 9.5
    assert held.preset == "tiny", held
    measured = select_model("auto", _gpu("GeForce RTX 3090", 24, 8, 6), measured_beamng_vram_gb=20)
    assert measured.reserve_source == "measured"
    assert measured.preset == "small", measured
    flag_wins = select_model(
        "auto",
        _gpu("GeForce GTX 1060", 6, 6, 1),
        game_reserve_gb=1.0,
        measured_beamng_vram_gb=5.0,
    )
    assert flag_wins.reserve_source == "flag"


def check_ram_and_cpu_headroom() -> None:
    ram = select_model("auto", _gpu("GeForce RTX 3090", 24, 8, 6, ram_gb=9.0))
    assert ram.preset == "small", ram
    cores = select_model("auto", _gpu("GeForce RTX 3090", 24, 8, 6, cpu_count=4))
    assert cores.preset == "small", cores


def check_low_free_vram_picks_smaller() -> None:
    card = _gpu("GeForce RTX 3090", 24, 8, 6)
    roomy = select_model("auto", card, free_vram_gb=20.0)
    tight = select_model("auto", card, free_vram_gb=2.0)
    assert roomy.preset == "large", roomy
    assert tight.preset == "tiny", tight
    assert PRESET_ORDER.index(tight.preset) < PRESET_ORDER.index(roomy.preset)


def check_manual_over_budget_warns() -> None:
    plan = select_model("xl", _gpu("GeForce GTX 1080 Ti", 11, 6, 1))
    assert plan.preset == "xl"
    assert plan.warnings
    assert any("xl" in note and "BeamNG" in note for note in plan.warnings)


def check_pascal_precision() -> None:
    applied, notes = resolve_precision("fp16", (6, 1))
    assert applied == "fp32"
    assert notes
    applied, notes = resolve_precision("auto", (6, 1), "GeForce GTX 1080 Ti")
    assert applied == "fp32"
    assert any("Pascal" in note for note in notes)
    applied, notes = resolve_precision("fp16", (8, 6))
    assert applied == "fp16" and notes == []
    applied, notes = resolve_precision("auto", (7, 5), "GeForce GTX 1660 Ti")
    assert applied == "fp32" and notes
    applied, notes = resolve_precision("int8", (8, 6))
    assert applied == "fp32"
    assert any("INT8" in note or "int8" in note for note in notes)
    applied, _notes = resolve_precision("auto", (8, 6), "GeForce RTX 3090")
    assert applied == "bf16"


def check_accel_paths() -> None:
    pascal = pick_accel((6, 1), "GeForce GTX 1080 Ti")
    assert pascal.applied_precision == "fp32"
    assert pascal.tensor_cores is False
    assert pascal.prefer_precision == "int8"
    assert "fp16" not in pascal.tensor_precisions
    assert any("Pascal" in note or "fp32" in note for note in pascal.notes)
    turing = pick_accel((7, 5), "GeForce RTX 2080 Ti")
    assert turing.tensor_cores is True
    assert turing.applied_precision == "fp16"
    assert "fp16" in turing.tensor_precisions and "int8" in turing.tensor_precisions
    assert turing.backend_applied in ("eager", "torch_compile")
    sixteen = pick_accel((7, 5), "GeForce GTX 1660")
    assert sixteen.tensor_cores is False
    assert sixteen.applied_precision == "fp32"
    ampere = pick_accel((8, 6), "GeForce RTX 3090")
    assert "bf16" in ampere.tensor_precisions and "tf32" in ampere.tensor_precisions
    assert ampere.applied_precision == "bf16"
    ada = pick_accel((8, 9), "GeForce RTX 4090")
    assert "fp8" in ada.tensor_precisions
    assert ada.applied_precision == "bf16"
    blackwell = pick_accel((12, 0), "GeForce RTX 5090")
    assert "fp4" in blackwell.tensor_precisions and "fp8" in blackwell.tensor_precisions
    net, used, notes = prepare_backend(object(), "tensorrt")
    assert used == "eager" and notes
    net, used, notes = prepare_backend(object(), "onnxruntime")
    assert used == "eager" and notes


def check_benchmark_latency_budget() -> None:
    card = GpuFacts(name="NVIDIA A100", vram_gb=80, capability=(8, 0))
    bench = {"tiny": 5.0, "small": 12.0, "medium": 28.0, "large": 55.0, "xl": 140.0}
    roomy = select_model("auto", card, bench_ms=bench, target_hz=10)
    assert roomy.preset == "large", roomy
    assert roomy.bench_source == "benchmark"
    tight = select_model("auto", card, bench_ms=bench, max_infer_ms=30)
    assert tight.preset == "medium", tight
    assert tight.limit_ms == 30


def check_latency_step_down() -> None:
    nxt, stepped, why = step_down_for_pressure("large", infer_ms=40, max_infer_ms=20)
    assert stepped and nxt == "medium" and "latency" in why
    nxt, stepped, why = step_down_for_pressure("medium", loop_hz=6, target_hz=10)
    assert stepped and nxt == "small" and "Hz" in why
    nxt, stepped, _why = step_down_for_pressure("small", infer_ms="slow", max_infer_ms=10)  # type: ignore[arg-type]
    assert nxt == "small" and not stepped
    nxt, stepped, why = step_down_for_pressure("large", loop_hz=0, target_hz=10, infer_ms=5, max_infer_ms=20)
    assert nxt == "large" and not stepped and why == "headroom ok"


def check_step_down_never_raises() -> None:
    nxt, stepped, why = step_down_for_pressure("large", free_vram_gb=0.2)
    assert stepped and nxt == "medium" and "free VRAM" in why
    nxt, stepped, why = step_down_for_pressure("small", game_fps=12)
    assert stepped and nxt == "tiny" and "fps" in why
    nxt, stepped, _why = step_down_for_pressure("tiny", free_vram_gb=0.1, game_fps=1)
    assert nxt == "tiny" and not stepped
    nxt, stepped, _why = step_down_for_pressure("large", free_vram_gb="not-a-number")  # type: ignore[arg-type]
    assert nxt == "large" and not stepped
    nxt, stepped, why = step_down_for_pressure("medium", free_vram_gb=8, game_fps=60)
    assert nxt == "medium" and not stepped and why == "headroom ok"


def check_old_checkpoint_is_small() -> None:
    assert checkpoint_preset({}) == "small"
    assert checkpoint_preset({"preset": None}) == "small"
    assert checkpoint_preset({"preset": "tiny"}) == "tiny"
    refuse_preset_mismatch("small", {})
    refuse_preset_mismatch(None, {"preset": "large"})
    try:
        refuse_preset_mismatch("medium", {})
    except PresetMismatch as exc:
        text = str(exc)
        assert "small" in text and "medium" in text
    else:
        raise AssertionError("a medium request against an old checkpoint must fail")


def main() -> None:
    check_mocked_gpus()
    check_default_stays_small()
    check_game_reserve_knob()
    check_ram_and_cpu_headroom()
    check_low_free_vram_picks_smaller()
    check_manual_over_budget_warns()
    check_pascal_precision()
    check_accel_paths()
    check_benchmark_latency_budget()
    check_latency_step_down()
    check_step_down_never_raises()
    check_old_checkpoint_is_small()
    print("test_gpu_model_size: ok")


if __name__ == "__main__":
    main()
