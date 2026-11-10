# GPU tiers for GVD model size

GVD's live net shares the GPU with BeamNG. Auto-size picks the **largest preset whose frame time fits the latency budget** (`--target-hz`, default 10 Hz, or `--max-infer-ms` when that is set). VRAM is the sanity check after that: BeamNG is reserved first, and a preset that would crowd the game off the card is skipped even when it is fast enough.

The millisecond columns are estimates unless a startup micro-benchmark has been cached for this GPU and driver (`model_size_bench.json` under `GVD_DATA_DIR/gvd` or `~/.cache/gvd`). The static estimate starts from an unmeasured anchor: small, fp32, compute 6.1, long side 640, 25 ms with the GPU to ourselves. BeamNG is then given 35% of the GPU, so the published frame time is that anchor divided by 0.65. A cached benchmark replaces the capability factor with TFLOP/s measured from one small CUDA conv on the card. None of these numbers are a timed SceneNet, and none of them are BeamNG's frame rate.

## What is measured and what is reserved

Card memory, compute capability, bandwidth, and the fp32 rates in the tables are from the pages listed at the bottom. Those pages were opened while this note was written. INT8 peak TOPS are **not** on those pages for Pascal. The TensorRT note describes DP4A (a 4-element INT8 dot product) and a 50 ms / ~5× result for a different network on DRIVE PX. That 5× is not SceneNet and not a 1080 Ti.

BeamNG's VRAM column is **not** a measurement. `default_game_reserve_gb` in `python/train/model_size.py` is a conservative floor for that total-VRAM band. A real reading replaces it: set `GVD_BEAMNG_VRAM_GB`, or pass `--game-vram-reserve-gb`. The flag wins over the environment variable, and the environment variable wins over the floor.

The vision cap is the sanity check. On an 11 GB card it stays at 4 GB unless `--vram-budget-gb` says otherwise. The vision budget is the smaller of (card − BeamNG reserve) and that cap. A live free-VRAM reading, when we have one, shrinks the budget further by 0.5 GB.

System RAM keeps 8 GB for BeamNG and the OS before a larger preset is eligible. Fewer than 8 logical CPUs stops auto at `small`, so the game keeps cores. The supervisor process itself still has to stay under 12 GB RSS.

## Presets

`small` is today's SceneNet: stem `(32, 64, 128, 128)`, GRU 512, long side 640, about 2.2 million weights. `--model-size` defaults to `small`, and that path does not pass new constructor arguments. `auto` is the opt-in.

| Preset | Stem | GRU | Long side | Vision estimate | Process RAM estimate |
| --- | --- | --- | --- | --- | --- |
| tiny | 16, 32, 64, 64 | 256 | 512 | 1.0 GB | 0.4 GB |
| small | today's net | 512 | 640 | 2.2 GB | 0.6 GB |
| medium | 48, 96, 160, 160 | 512 | 640 | 4.8 GB | 1.2 GB |
| large | 64, 128, 192, 192 | 768 | 768 | 7.5 GB | 2.0 GB |
| xl | 96, 160, 256, 256 | 1024 | 768 | 12.0 GB | 3.5 GB |

The vision estimates are weights plus a rough activation bill at batch 1. They are not a profile.

`--model-size auto` (and a manual `medium` / `large` / `xl`) can exceed 4 GB for **training**. The paper vision estimates are medium 4.8 GB, large 7.5 GB, and xl 12.0 GB. An independent training-side estimate for xl at batch 1, eight cameras, T=16 is about 4.64 GB (weights plus activations; larger `--batch 0` auto-fit can fill the card, as the trainer already does). Live `SceneCkptRuntime` inference stays on CPU. The same independent estimate puts live inference at about 5.5–37 MB (main-only) and 34–170 MB (eight-camera canvas) from tiny through xl, so the 4 GB deploy gate still holds. Default `--model-size small` stays under that cap on paper and in that estimate.

Pascal (compute 6.1) never runs fp16. The fast path on that card is INT8 via DP4A, and this tree has no calibrated INT8 engine, so the run stays fp32. Volta and Turing tensor-core boards auto-apply fp16. Ampere and newer auto-apply bf16, and TF32 matmul is turned on when CUDA is present and `--model-size auto`. Ada (8.9) adds FP8. Blackwell (10.0 / 12.0) adds FP4. FP8 and FP4 are recorded and not executed. GTX 16-series boards are compute 7.5 and have no tensor cores, so a 16-series name stays fp32. A Quadro / NVIDIA T1000 (TU117, compute 7.5, no tensor cores) and a nameless compute 7.5 record are not name-matched, so they follow the generic 7.5 path rather than the 16-series fp32 rule.

Optional backends, in order, are TensorRT, ONNX Runtime, `torch.compile`, and CUDA graphs. A missing package or a missing scene engine falls back to eager PyTorch and the log says so. CUDA graph capture itself stays on `--exp-cuda-graph`. A manual size that misses the latency budget or the VRAM check prints a warning and is still built.

At run time, one size down is logged (and the loaded weights stay up) when infer latency is over the budget, the loop is under `--target-hz`, BeamNG fps is under 30, or free VRAM is under 0.75 GB. The step does not raise. The next launch is what builds the smaller net.

## BeamNG reserve by card size

| Card VRAM | BeamNG reserve (estimate) | Vision cap |
| --- | ---: | ---: |
| 4 GB | 2.5 GB | 0 (no fit; auto stays on tiny) |
| 6 GB | 3.5 GB | 1.6 GB |
| 8 GB | 4.5 GB | 2.5 GB |
| 11–12 GB | 6.0 GB | 4.0 GB |
| 16 GB | 7.5 GB | 6.0 GB |
| 24 GB | 8.5 GB | 8.0 GB |
| 32 GB and up | 10.0 GB | 12.0 GB |

## What auto picks

Same function as the tests, with no free-VRAM reading and no cached benchmark, so a card that is already full of BeamNG will pick smaller at startup than the row shows. Milliseconds are the shared-GPU estimate at long side 640 for the preset that was picked. Hz is `1000 / ms`. The capability factor (1.5 at 7.0, 1.8 at 7.5, 3 at 8.0, 4 at 8.6, 6 at 8.9, 16 at 12.0) is a stand-in, not the fp32 TFLOP column. fp16 and bf16 multiply by 0.6, which is a guess about tensor cores.

| Card | BeamNG reserve | Auto preset | Precision | Est. ms | Est. Hz |
| --- | ---: | --- | --- | ---: | ---: |
| GTX 1060 6 GB | 3.5 | tiny | fp32 | 27 | 37 |
| GTX 1080 Ti 11 GB | 6.0 | small | fp32 | 38 | 26 |
| GTX 1660 Ti 6 GB | 3.5 | tiny | fp32 | 15 | 67 |
| RTX 2060 6 GB | 3.5 | tiny | fp16 | 9 | 111 |
| RTX 2080 Ti 11 GB | 6.0 | small | fp16 | 13 | 77 |
| Tesla T4 16 GB | 7.5 | medium | fp16 | 28 | 36 |
| Tesla V100 16 GB | 7.5 | medium | fp16 | 34 | 29 |
| RTX 3060 12 GB | 6.0 | small | bf16 | 6 | 167 |
| RTX 3090 24 GB | 8.5 | large | bf16 | 14 | 71 |
| RTX A4000 16 GB | 7.5 | medium | bf16 | 13 | 77 |
| RTX A5000 24 GB | 8.5 | large | bf16 | 14 | 71 |
| A100 40 or 80 GB | 10.0 | xl | bf16 | 29 | 34 |
| RTX 4070 12 GB | 6.0 | small | bf16 | 4 | 250 |
| RTX 4090 24 GB | 8.5 | large | bf16 | 9 | 111 |
| L4 24 GB | 8.5 | large | bf16 | 9 | 111 |
| RTX 5070 12 GB | 6.0 | small | bf16 | 1 | 1000 |
| RTX 5090 32 GB | 10.0 | xl | bf16 | 3 | 333 |
| Tesla P40 24 GB | 8.5 | large | fp32 | 91 | 11 |
| CPU only | 2.5 floor, no VRAM | tiny | fp32 | 27 | — |

Hertz above the live camera rate (about 10–15) is the factor running ahead of the loop. BeamNG and the grab still set the frame rate. On a 3090, xl estimates 22 ms and still loses: its 12 GB vision estimate is over the 8 GB sanity cap. P40 large sits just under the 100 ms line. A mocked A100 whose benchmark says xl takes 140 ms picks `large` at a 10 Hz budget and `medium` at `--max-infer-ms 30`. That is `check_benchmark_latency_budget`.

## Compute

Architecture and tensor-core precisions are by compute capability, not by a made-up TOPS number.

| SM | Architecture | Tensor-core precisions | What auto runs |
| --- | --- | --- | --- |
| 6.1 | Pascal | none. INT8 is DP4A on the CUDA cores | fp32. INT8 is preferred and not built |
| 7.0 | Volta | FP16 | fp16 |
| 7.5 | Turing | FP16, INT8, INT4. GTX 16-series omitted the tensor cores | fp16 on RTX. fp32 on GTX 16 |
| 8.0 / 8.6 | Ampere | TF32, BF16, FP16, INT8, INT4 | bf16, and TF32 matmul when CUDA is up |
| 8.9 | Ada | the Ampere set, plus FP8 | bf16. FP8 is recorded and not built |
| 10.0 / 12.0 | Blackwell | the Ada set, plus FP4 (NVFP4 / MXFP4 on the data-center parts) | bf16. FP8 and FP4 are recorded and not built |

Estimated milliseconds at long side 640 with BeamNG holding 35% of the GPU. Same formula as `estimate_ms`. A blank means that preset misses a 100 ms budget.

| Card | tiny | small | medium | large | xl |
| --- | ---: | ---: | ---: | ---: | ---: |
| GTX 1060 / 1080 Ti / P40 (fp32, sm 6.1) | 27 | 38 | 84 | 91 | 146 |
| GTX 1660 Ti (fp32, sm 7.5, no tensor cores) | 15 | 21 | 47 | 51 | 81 |
| RTX 2060 / 2080 Ti / T4 (fp16, sm 7.5) | 9 | 13 | 28 | 30 | 49 |
| V100 (fp16, sm 7.0) | 11 | 15 | 34 | 36 | 58 |
| RTX 3090 / A5000 (bf16, sm 8.6) | 4 | 6 | 13 | 14 | 22 |
| A100 (bf16, sm 8.0) | 5 | 8 | 17 | 18 | 29 |
| RTX 4090 / L4 (bf16, sm 8.9) | 3 | 4 | 8 | 9 | 15 |
| RTX 5090 (bf16, sm 12.0) | 1 | 1 | 3 | 3 | 5 |

Per-SKU fp32 and bandwidth below use the desktop rows whose cells survived the page extract. A dash is an empty cell in that extract, not a guess. FP32 is the boost figure, in TFLOPS. INT8 peak TOPS are not filled in.

### GeForce 10, Pascal, sm 6.1

| Card | Memory | Bandwidth | FP32 |
| --- | --- | ---: | ---: |
| GTX 1050 2 GB | 2 GB | 112 GB/s | 1.86 |
| GTX 1050 3 GB | 3 GB | 84 GB/s | 2.33 |
| GTX 1050 Ti | 4 GB | 112 GB/s | 2.14 |
| GTX 1060 3 GB | 3 GB | 192 GB/s | 3.94 |
| GTX 1060 5 GB | 5 GB | 160 GB/s | 4.37 |
| GTX 1060 6 GB | 6 GB | 192 or 216 GB/s | — |
| GTX 1070 | 8 GB | 256 GB/s | 6.46 |
| GTX 1070 Ti | — | — | 8.19 |
| GTX 1080 | — | 320 or 352 GB/s | 8.87 |
| GTX 1080 Ti | 11 GB | 484 GB/s | 11.34 |

The 1080 Ti half-precision rate in that same table is 166 GFLOPS at base against 10609 GFLOPS fp32. That is why fp16 stays off.

### GeForce 16, Turing without tensor cores, sm 7.5

| Card | Memory | Bandwidth | FP32 |
| --- | --- | ---: | ---: |
| GTX 1630 | 4 GB | 96 GB/s | 1.83 |
| GTX 1650 GDDR5 | 4 GB | 128 GB/s | 2.98 |
| GTX 1650 GDDR6 | 4 GB | 192 GB/s | 2.85 |
| GTX 1650 Super | 4 GB | — | 4.42 |
| GTX 1660 | 6 GB | — | 5.03 |
| GTX 1660 Super | 6 GB | 336 GB/s | — |
| GTX 1660 Ti | 6 GB | 288 GB/s | 5.44 |

### GeForce 20, Turing tensor cores, sm 7.5

| Card | Memory | Bandwidth | FP32 |
| --- | --- | ---: | ---: |
| RTX 2060 | 6 GB | 336 GB/s | 6.45 |
| RTX 2060 12 GB | 12 GB | — | 7.18 |
| RTX 2060 Super | 8 GB | 448 GB/s | — |
| RTX 2080 Ti | 11 GB | 616 GB/s | 13.45 |
| Titan RTX | 24 GB | 672 GB/s | 16.31 |

### GeForce 30, Ampere, sm 8.6

| Card | Memory | Bandwidth | FP32 |
| --- | --- | ---: | ---: |
| RTX 3050 | 6 GB | 168 GB/s | 6.77 |
| RTX 3060 | 12 GB | 360 GB/s | 12.74 |
| RTX 3060 Ti | 8 GB | 448 GB/s | 16.2 |
| RTX 3080 | 10 GB | 760 GB/s | 29.77 |
| RTX 3080 12 GB | 12 GB | 912 GB/s | 30.64 |
| RTX 3090 | 24 GB | 936 GB/s | 35.58 |
| RTX 3090 Ti | — | 1008 GB/s | 39.99 |

### GeForce 40, Ada, sm 8.9

| Card | Memory | Bandwidth | FP32 |
| --- | --- | ---: | ---: |
| RTX 4060 | 8 GB | 272 GB/s | 15.11 |
| RTX 4060 Ti | 8 GB cell empty; 16 GB board | 288 GB/s on the 8 GB row | 22.06 |
| RTX 4070 | 12 GB | 504 GB/s | 29.15 |
| RTX 4070 Ti Super | 16 GB | 672 GB/s | 44.1 |
| RTX 4080 | 16 GB | 717 GB/s | 48.74 |
| RTX 4090 D | 24 GB | 1008 GB/s | 73.54 |
| RTX 4090 | size cell empty; the D row beside it is 24 GB | — | 82.58 |

### GeForce 50, Blackwell, sm 12.0

| Card | Memory | Bandwidth |
| --- | --- | ---: |
| RTX 5050 | 8 GB GDDR6 | 320 GB/s |
| RTX 5060 | size cell empty, GDDR7 | 448 GB/s |
| RTX 5060 Ti | 8 or 16 GB | — |
| RTX 5070 | 12 GB | 672 GB/s |
| RTX 5070 Ti | 16 GB | 896 GB/s |
| RTX 5080 | 16 GB in the IGN passage on the same page; size cell empty | 960 GB/s |
| RTX 5090 | 32 GB GDDR7 | 1792 GB/s |

FP32 TFLOPS for the 50-series were not in the extract used here.

### Data center and workstation

| Card | SM | Memory | Bandwidth | FP32 | Tensor |
| --- | --- | --- | ---: | ---: | --- |
| Tesla P40 | 6.1 | 24 GB GDDR5 | 346 GB/s | 10.0–11.8 | none, DP4A |
| Tesla V100 SXM | 7.0 | 16 or 32 GB HBM2 | 900 GB/s | 14.9 | FP16 |
| Tesla T4 | 7.5 | 16 GB GDDR6 | 320 GB/s | 8.1 | FP16, INT8 |
| A100 | 8.0 | 40 or 80 GB HBM2 | 1555 GB/s | 19.5 | TF32, BF16, FP16, INT8 |
| RTX A4000 | 8.6 | 16 GB GDDR6 | 448 GB/s | 19.2 | Ampere tensor cores, 192 of them |
| RTX A5000 | 8.6 | 24 GB GDDR6 | 768 GB/s | 27.8 | Ampere tensor cores, 256 of them |
| L4 | 8.9 | 24 GB GDDR6 | 300 GB/s | 30.3 | Ada, including FP8 |

A mocked 3090 with 2.0 GB free picks `tiny` instead of `large`. That is `check_low_free_vram_picks_smaller`.

## Cards

Compute capability is the NVIDIA developer table. Memory and fp32 rates are the desktop rows that survived the page extract. A blank memory cell means the extract did not include a number, not that the card has no memory. Laptop boards of the same chip are omitted.

fp16 / tensor / INT8, in short:

- Compute 6.1 (GTX 10-series, Titan Xp/X, Tesla P40): no tensor cores. The 1080 Ti row's half-precision rate is far below its fp32 rate (166 vs 10609 GFLOPS at base in the GeForce 10 table), which is why fp16 stays off. INT8 dot-product (DP4A) exists from sm_61. NVIDIA's "about 5×" figure is for a different network on DRIVE PX, not for SceneNet.
- GTX 16-series: Turing, compute 7.5, **no** tensor cores and no RT cores. The Wikipedia architecture section says the tensor and RT cores were omitted.
- Compute 7.0 (V100) and 7.5 (T4, RTX 20-series, Titan RTX): tensor cores, fp16.
- Compute 8.0 / 8.6 / 8.9 / 12.0: tensor cores, fp16, TF32 and bf16 from Ampere onward. INT8 on tensor cores from Turing onward.

### GeForce 10 (compute 6.1)

| Card | Memory | fp32 boost, GFLOPS |
| --- | --- | ---: |
| GTX 1050 2 GB | 2 GB | 1862 |
| GTX 1050 3 GB | 3 GB | 2332 |
| GTX 1050 Ti | 4 GB | 2138 |
| GTX 1060 3 GB | 3 GB | 3935 |
| GTX 1060 5 GB | 5 GB | 4372 |
| GTX 1060 | 6 GB | (cell paired with the 6 GB row) |
| GTX 1070 | 8 GB | 6463 |
| GTX 1080 Ti | 11 GB | 11340 |

The GTX 1080 and GTX 1070 Ti memory cells were empty in the extract. The 1080 row does list 320 / 352 GB/s. BeamNG reserve for the 11 GB 1080 Ti is 6 GB, and the vision cap is 4 GB, so auto stays on `small`.

### GeForce 16 (compute 7.5, no tensor cores)

| Card | Memory | fp32 boost, GFLOPS |
| --- | --- | ---: |
| GTX 1630 | 4 GB | 1828 |
| GTX 1650 | 4 GB | 2984 (GDDR5) |
| GTX 1650 Super | 4 GB (called out as 4 GB in the PC Gamer passage on the same page) | 4416 |
| GTX 1660 | 6 GB | 5027 |
| GTX 1660 Super | 6 GB | (same core config as the 1660) |
| GTX 1660 Ti | 6 GB | 5437 |

### GeForce 20 (compute 7.5)

| Card | Memory | fp32 boost, TFLOPS |
| --- | --- | ---: |
| RTX 2060 | 6 GB | 6.451 |
| RTX 2060 12 GB | 12 GB | 7.181 |
| RTX 2060 Super | 8 GB | (row continues the TU106 Super config) |
| RTX 2080 Ti | 11 GB | 13.448 |
| Titan RTX | 24 GB | 16.312 |

Several 2070 / 2080 memory cells were empty in the extract. Treat those as unknown here rather than filled in.

### GeForce 30 (compute 8.6)

| Card | Memory | fp32 boost, TFLOPS |
| --- | --- | ---: |
| RTX 3050 | 6 GB | 6.774 |
| RTX 3060 | 12 GB | 12.74 |
| RTX 3060 Ti | 8 GB | 16.2 |
| RTX 3080 | 10 GB | 29.77 |
| RTX 3080 12 GB | 12 GB | 30.64 |
| RTX 3090 | 24 GB | 35.58 |

### GeForce 40 (compute 8.9)

| Card | Memory | fp32 boost, TFLOPS |
| --- | --- | ---: |
| RTX 4060 | 8 GB | 15.11 |
| RTX 4060 Ti 16 GB | 16 GB | (same core config as the 8 GB board; that size cell was empty) |
| RTX 4070 | 12 GB | 29.15 |
| RTX 4070 Ti Super | 16 GB | 44.1 |
| RTX 4080 12 GB | 12 GB, unlaunched | 40.09 |
| RTX 4080 | 16 GB | 48.74 |
| RTX 4090 D | 24 GB | 73.54 |
| RTX 4090 | size cell empty in the extract; the D row beside it is 24 GB | 82.58 |

### GeForce 50 (compute 12.0)

Desktop memory from the GeForce 50 table. 5050 is GDDR6; the rest of that row is GDDR7. The 5060 and 5080 size cells were empty. The 5080's 16 GB figure is in the IGN passage on the same page ("the 16 GB of VRAM").

| Card | Memory | Bandwidth |
| --- | --- | ---: |
| RTX 5050 | 8 GB GDDR6 | 320 GB/s |
| RTX 5060 | size cell empty, GDDR7 | 448 GB/s |
| RTX 5060 Ti | 8 or 16 GB | — |
| RTX 5070 | 12 GB | 672 GB/s |
| RTX 5070 Ti | 16 GB | 896 GB/s |
| RTX 5080 | 16 GB in the IGN passage; size cell empty | 960 GB/s |
| RTX 5090 | 32 GB GDDR7 | 1792 GB/s |

### Data center and workstation

| Card | Memory | Compute | fp32 | Tensor |
| --- | --- | --- | --- | --- |
| Tesla P40 | 24 GB GDDR5 | 6.1 | 10.01–11.76 TFLOPS | no |
| Tesla V100 | 16 or 32 GB HBM2 | 7.0 | 14.90 TFLOPS (SXM) | yes, fp16 |
| Tesla T4 | 16 GB GDDR6 | 7.5 | 8.1 TFLOPS | yes |
| A100 | 40 or 80 GB HBM2 | 8.0 | 19.5 TFLOPS | yes |
| L4 | 24 GB GDDR6 | 8.9 | 30.3 TFLOPS | yes |
| RTX A4000 | 16 GB GDDR6 ECC | 8.6 | 19.2 TFLOPS | yes, 192 third-gen |
| RTX A5000 | 24 GB GDDR6 ECC | 8.6 | 27.8 TFLOPS | yes, 256 third-gen |

## Sources

Opened for this note:

- CUDA compute capability, including the 10 / 16 / 20 / 30 / 40 / 50 and data-center columns: <https://developer.nvidia.com/cuda/gpus>
- GeForce 10 memory and GFLOPS: <https://en.wikipedia.org/wiki/GeForce_10_series>
- GeForce 16, including the "no tensor cores" architecture paragraph: <https://en.wikipedia.org/wiki/GeForce_16_series>
- GeForce 20: <https://en.wikipedia.org/wiki/GeForce_20_series>
- GeForce 30: <https://en.wikipedia.org/wiki/GeForce_30_series>
- GeForce 40 and the 4090 table: <https://en.wikipedia.org/wiki/GeForce_40_series> and <https://en.wikipedia.org/wiki/GeForce_RTX_4090>
- GeForce 50: <https://en.wikipedia.org/wiki/GeForce_50_series>
- Tesla P40, V100, T4, A100, L4: <https://en.wikipedia.org/wiki/Nvidia_Tesla>
- RTX A4000 16 GB and RTX A5000 24 GB: <https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a4000-datasheet.pdf> and <https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf>
- INT8 on Pascal via DP4A, and the DRIVE PX speedup that is **not** a SceneNet number: <https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/>
- Tensor-core generations (Volta FP16, Turing INT8/INT4, Ampere TF32/BF16/INT8): <https://www.nvidia.com/en-eu/data-center/tensorcore/>
- Ada FP8 tensor cores: <https://www.nvidia.com/en-us/technologies/ada-architecture/>
- Blackwell FP8 and FP4, including NVFP4: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- Blackwell tensor-core precision list (NVFP4, FP8/FP6, INT8, BF16, FP16, TF32): <https://www.nvidia.com/en-us/data-center/tensor-cores/>
