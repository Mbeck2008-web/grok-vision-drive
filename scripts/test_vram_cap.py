#!/usr/bin/env python3
"""hardware.yaml max_inference_vram_gb is applied to detector sessions.

The 4 GB cap was stored in config/hardware.yaml and never read. YOLO ONNX
sessions asked for CUDA with no arena limit.

    py -3 scripts/test_vram_cap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.perception.detect import (  # noqa: E402
    load_max_inference_vram_gb,
    onnx_providers,
    weight_file_exceeds_vram_cap,
)


def check_reads_hardware_yaml() -> None:
    cap = load_max_inference_vram_gb()
    assert cap == 4.0, cap
    print(f"max_inference_vram_gb={cap} OK")


def check_cuda_provider_carries_limit() -> None:
    providers = onnx_providers(available=["CUDAExecutionProvider", "CPUExecutionProvider"], cap_gb=4.0)
    assert providers[0][0] == "CUDAExecutionProvider", providers
    assert providers[0][1]["gpu_mem_limit"] == 4 * 1024 ** 3, providers
    print("CUDA provider gpu_mem_limit is 4 GiB OK")


def check_oversize_weight_is_refused() -> None:
    assert weight_file_exceeds_vram_cap(size_bytes=5 * 1024 ** 3, cap_gb=4.0)
    assert not weight_file_exceeds_vram_cap(size_bytes=100_000_000, cap_gb=4.0)
    print("weight file over the cap is refused OK")


def main() -> None:
    check_reads_hardware_yaml()
    check_cuda_provider_carries_limit()
    check_oversize_weight_is_refused()
    print("test_vram_cap: OK")


if __name__ == "__main__":
    main()
