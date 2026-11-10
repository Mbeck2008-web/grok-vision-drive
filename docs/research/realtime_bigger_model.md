# A bigger vision model on a shared GTX 1080 Ti

Research note. No code change is proposed as a patch in this file. Estimates are labeled. Citations are pages that were opened while writing this note.

## What is actually small

The live scene net on this tree is a GroupNorm stem (`32, 64, 128, 128`), eight horizontal bins, a 512-wide GRU, and linear heads. `architecture_parameter_count()` on this commit is **2,203,694** parameters. In fp32 that is about **8.4 MB of weights**. The 4 GB cap in `config/hardware.yaml` (`max_inference_vram_gb`) is not spent on those weights. It is spent on activations, the detector session, and whatever BeamNG already holds on the same 11 GB card. The supervisor is on the CPU and must stay under 12 GB RSS. The live picture is resized to a long side of about 640.

Pascal (sm_61) has no tensor cores. fp16 is a poor default. INT8 dot-products of four bytes into an int32 (DP4A) are the fast low-precision path, and they need compute capability 6.1 or newer. A GTX 1080 is in that set. NVIDIA's TensorRT write-up says the Pascal GPU in DRIVE PX runs DP4A, and that an INT8 TensorRT engine of one detection network was about 5× the throughput of that network's original Caffe fp32 run, at about 50 ms on that board. That 5× figure is **not** a measurement of SceneNet. It is a different model on a different Pascal board ([Fast INT8 Inference for Autonomous Vehicles with TensorRT 3](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)).

So the useful question is not "compress the 2.2M net." The useful question is "what extra capacity can we turn on at 10 Hz without crossing 4 GB, and how do we prove it on the holdout."

`--exp-distill-feats` (landed with #194, `python/train/exp_distill.py`) already caches a frozen teacher's tokens for **train rows only**. Holdout rows are not in that cache. The path is training-only and is not exported. A bigger teacher can stay offline.

## Ranked ideas

Rank is expected holdout gain per week of work, inside the 4 GB cap, not peak ImageNet accuracy.

### 1. Run the expensive stem on a clock, and the GRU every frame

**What.** Keep a feature token from the last keyframe. Step the GRU on every frame. Recompute the stem only when the picture moved, or every Nth frame.

**Why it fits.** The live loop is video, not a pile of stills. Shelhamer et al. schedule layers at different rates because deep features change more slowly than pixels ([Clockwork Convnets for Video Semantic Segmentation](https://arxiv.org/abs/1608.03609)). A fatter stem is then affordable: it runs a few times a second, not ten. Peak VRAM is the fat stem's one forward, which we can still cap at 4 GB. Average GPU time drops.

**Estimate.** If a future stem is 4× today's compute and it runs every 4th frame, average stem time is about today's, and the GRU (already small) still runs at 10 Hz. Peak VRAM is unchanged versus running that stem every frame. These are estimates, not a profile.

**Risk.** A curve, a cut-in, or a respawn between keyframes is exactly when the token is stale. The clock has to be data-dependent (their adaptive clock), not a fixed skip. Warping pixels with ego motion would invent geometry the model did not see. Do not do that. Reuse features, or rerun.

**Read.** [Clockwork Convnets](https://arxiv.org/abs/1608.03609). Reference code: [shelhamer/clockwork-fcn](https://github.com/shelhamer/clockwork-fcn).

### 2. Distill a fat offline teacher into a student that still fits

**What.** Train a large camera model offline, where the 4 GB cap does not apply and BeamNG is closed. Teach the live net with `--exp-distill-feats`: match the teacher's tokens on train rows, then score the student on the untouched holdout.

**Why it fits.** Hinton, Vinyals, and Dean separate a cumbersome training model from a small deployed model ([Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)). The teacher never loads in the supervisor. The student is the only thing that has to sit in 4 GB. The flag already refuses a missing teacher and does not write the holdout into the cache (`python/train/exp_distill.py`).

**Estimate.** Quality moves as far as the student can represent. A student that is only today's 2.2M net will not absorb a huge teacher. Plan on a student a few times wider in the stem, still a few tens of MB of weights. Latency of the student is whatever that stem costs at 640. VRAM of the teacher is an offline number and must not be added to the live budget. Estimates.

**Risk.** The current cache stores tokens, not a trained big backbone. Someone still has to train the teacher. If the teacher's tokens use map or depth features, the student learns a shortcut it cannot run live. Teacher inputs stay cameras.

**Read.** [Hinton, Vinyals, Dean](https://arxiv.org/abs/1503.02531). Repo hook: `python/train/exp_distill.py`.

### 3. Run the fat net only on the road, not on the whole 640 frame

**What.** A cheap full-frame pass marks the road region. The large stem runs as sparse blocks inside that region.

**Why it fits.** Ren et al. take a cheap mask and skip dense convolution outside it, with a tiled kernel that is fast on a real GPU, not just a FLOP count ([SBNet: Sparse Blocks Network for Fast Inference](https://arxiv.org/abs/1801.02108)). Most of a BeamNG frame is sky, hood, and shoulders. The lane head does not need those pixels at full cost. Activations shrink with the mask, which is the part of the 4 GB budget that matters.

**Estimate.** If the mask keeps about a third of the feature map, activation bytes drop by a similar factor, and a wider stem can replace today's. Wall-clock on Pascal is not the paper's number. Their speed-up was measured on a different task (LiDAR detection). Treat a 1.5–2× stem budget as a hope, and measure it. Estimate.

**Risk.** A bad mask drops the curve or the lead car. The mask has to come from the previous frame's own output or from a tiny net, not from the game road mesh. SBNet's published code path is not our GRU.

**Read.** [SBNet](https://arxiv.org/abs/1801.02108).

### 4. Cheap context at low resolution, a thin path at high resolution

**What.** Downscale the whole frame for a context backbone. Keep a narrow vertical crop, or the center of the 640 image, for a small high-resolution detail path. Fuse them into the GRU.

**Why it fits.** Yu et al. split spatial detail from a fast downsampling context path and fuse them ([BiSeNet](https://arxiv.org/abs/1808.00897)). They report 68.4% mean IoU on Cityscapes test at 2048×1024 and 105 FPS on a Titan XP. That FPS is not our number and not a 1080 Ti. The shape of the idea still matches us: context can be 160 px wide; the lane offset lives in a thin central strip. Two small activations fit in 4 GB more easily than one fat full-resolution map.

**Estimate.** A 4× linear downscale is about 16× fewer context pixels. The detail path is a crop, not a second full frame. Peak VRAM should fall versus a full-resolution wide net of the same depth, by a large fraction of the activation tensor. Latency should drop with the pixel count. Both are estimates until profiled. Their Cityscapes speed is a different card and a different head.

**Risk.** A crop that misses the lane on a sharp bend is a silent failure. The crop has to follow the previous center, and it has to widen when confidence falls. Fusion can wash out the metre-scale lane head if the two paths are not scaled.

**Read.** [BiSeNet](https://arxiv.org/abs/1808.00897).

### 5. INT8 the larger stem with TensorRT, after it exists

**What.** Post-training calibration, then an INT8 TensorRT engine for the conv stem. Keep the GRU in fp32 if its state quantizes badly. Optionally fine-tune with integer-only quantization-aware training.

**Why it fits.** DP4A is the Pascal INT8 instruction, documented for sm_61 including the GTX 1080 ([TensorRT INT8 calibration notes](https://kezunlin.me/blog/tensorrt-int8-inference/), and the NVIDIA post above). Jacob et al. train so the deployed net is integer arithmetic and show the accuracy/latency trade on CPUs even for MobileNets ([Integer-arithmetic-only inference](https://arxiv.org/abs/1712.05877)). Weights of a 20M-parameter stem are about 20 MB in int8 versus 80 MB in fp32. That is still nothing next to 4 GB. The real saving is bandwidth and the chance to run a wider conv before the 10 Hz deadline.

**Estimate.** Do not budget the blog's 5× onto SceneNet. For a conv-heavy stem, a 1.5–3× faster stem than the same stem in fp32 is a reasonable planning range on Pascal, and it can miss. Activation tensors stay fp32 at the GRU boundary unless we quantize those too. Peak VRAM drops only if the engine does not keep a parallel fp32 copy. Estimate.

**Risk.** Per-tensor scales on lane metres can bias the 8 m center even when classification-style metrics look fine. GRU and GroupNorm need a fallback path. TensorRT and the numpy checkpoint are two runtimes; the holdout has to be scored in the engine we ship. Calibration images must be our cameras, not a generic set.

**Read.** [NVIDIA TensorRT INT8 blog](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/). [Jacob et al.](https://arxiv.org/abs/1712.05877). Engine repo: [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT).

### 6. Grow the stem, then cut channels

**What.** Train a wider stem. During training, push channel scales toward zero. Delete the dead channels and fine-tune. Ship the thin net.

**Why it fits.** Liu et al. do this with batch-norm scales and report, on VGG, a multi-pass result of about 20× fewer parameters and 5× fewer FLOPs on classification sets ([Network Slimming](https://arxiv.org/abs/1708.06519)). Our stem is GroupNorm, not batch-norm, so the scale has to sit on the GroupNorm affine or on an extra multiplier. Fang et al. build a dependency graph so coupled layers are pruned together, including recurrent nets, and ship it as Torch-Pruning ([DepGraph](https://arxiv.org/abs/2301.12900), [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)). Structured pruning stays dense. Pascal likes dense conv. Unstructured sparsity would not.

**Estimate.** A 2× channel cut is about 4× fewer conv MACs in those layers and a similar cut in activation bytes. Whether holdout RMS survives is the whole experiment. The 20× / 5× figures are theirs, on other models. Estimate for us: try 30–50% of stem channels removed and expect a small RMS hit until fine-tune. VRAM of the shipped net drops with the activation width.

**Risk.** GroupNorm plus the GRU's input size are coupled. Pruning the stem without the graph cuts the wrong tensor and the checkpoint will not load. Pruning the GRU hidden size changes the pause format.

**Read.** [Network Slimming](https://arxiv.org/abs/1708.06519). [DepGraph](https://arxiv.org/abs/2301.12900).

### 7. Exit early when the confidence head is already sure

**What.** After a short prefix of the net, read the existing confidence. If it is high and the scene is not a fresh respawn, skip the rest of the stack. Hard frames run the full net.

**Why it fits.** Teerapittayanon et al. add side branches that leave early when the sample is easy ([BranchyNet](https://arxiv.org/abs/1709.01686)). Huang et al. stack early exits in one dense multi-scale net so compute can be spent on the hard inputs ([MSDNet](https://arxiv.org/abs/1703.09844)). We already have a confidence output. The 4 GB peak is the full net, which still runs on hard frames, so the cap does not go up. Average latency does.

**Estimate.** If half the frames exit after the stem, average frame time could fall by a few tens of percent. Peak VRAM does not fall. Both estimates. Their papers measure classification, not an 8 m lane RMS.

**Risk.** The frames we care about (curves, trucks, low sun) are the ones that must not exit. An exit rule trained on average confidence will skip them. The exit has to be forbidden for the first frames after a reset, and the holdout RMS has to be reported on the hard tail, not only the mean.

**Read.** [BranchyNet](https://arxiv.org/abs/1709.01686). [MSDNet](https://arxiv.org/abs/1703.09844).

### 8. Mix time by shifting channels, not by a bigger GRU

**What.** If the student grows a temporal stack, shift a fraction of channels across the last few frames instead of adding a 3D conv or a second GRU.

**Why it fits.** Lin, Gan, and Han insert a temporal shift into a 2D net and describe it as temporal modeling at zero extra parameters and zero extra multiply-adds, with an online variant for low latency ([TSM](https://arxiv.org/abs/1811.08383)). Our GRU is already the temporal model. TSM is a way to widen the camera tower without a second recurrent state. The state is a slice of activations we already paid for. It stays inside the same 4 GB envelope as that tower.

**Estimate.** No extra parameters for the shift itself. Latency is a memory permute plus the wider tower we chose to add. Their 13 ms and 35 ms figures are Jetson Nano and a phone, not this desktop. Do not copy them into a GVD budget.

**Risk.** The shift assumes a steady frame interval. Our loop is not a fixed training clip. A stalled frame or a respawn shifts stale channels into the current token. Online TSM, not the offline bi-directional one, is the only variant that is legal live.

**Read.** [TSM](https://arxiv.org/abs/1811.08383). Code they link: [mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module).

### 9. Put the non-lane heads on the Intel UHD

**What.** Keep the lane stem on the 1080 Ti. Compile the lamp, sign, and brake heads with OpenVINO and run them on the integrated GPU.

**Why it fits.** OpenVINO's GPU plugin is an OpenCL plugin for Intel integrated and discrete GPUs. On a machine with an iGPU, that device is `GPU.0` ([OpenVINO GPU device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)). The UHD does not share the 1080 Ti's 4 GB. The plugin's integer types are for quantized ops. The same page says to prefer fp16 on their GPU and to use static shapes. Our live shape is fixed (long side 640, then the canvas), so static shapes are natural.

**Estimate.** The heads are small linear maps off a 512-d vector. Moving them off the NVIDIA GPU saves a rounding error of VRAM and almost no NVIDIA time. The possible win is overlap: the UHD runs heads while the 1080 Ti is in BeamNG's frame. Sync can add 1–3 ms. Estimate. RSS of a second runtime has to stay inside the 12 GB supervisor cap.

**Risk.** Two loaders, two copies of the feature, and a CPU thread parked in the OpenCL driver. Their own notes say the driver can spin a core, and that some ops silently fall back to CPU. A UHD is not a second 1080 Ti. Do not put the stem there.

**Read.** [OpenVINO GPU device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html). Toolkit: [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino).

### 10. Buy the 4 GB back from BeamNG's frame, not from the net

**What.** Lower the game's render resolution, or the number of views, so the card has spare memory and spare time. Leave `live_input_long_side` at about 640.

**Why it fits.** The net never sees the game's full framebuffer if we resize to 640. A cheaper game frame does not add model parameters. It changes how much of the 11 GB and how many SMs BeamNG holds. That is the only way the 4 GB cap and a fatter net coexist when the game is up.

**Estimate.** No paper. The gain is "the inference process is not fighting a 1080p or 1440p scene for the same card." It does not by itself move holdout RMS. Measure BeamNG's VRAM with and without the cut before believing it.

**Risk.** A blurrier game image, after resize to 640, can still hurt the net if the resize was the only thing keeping detail. Cut game resolution in steps and rescore the holdout on frames grabbed at the new size.

**Read.** None. This is a machine setup, not a model result.

### Not now: stream weights from RAM

The live weights are about 8.4 MB. Streaming them from host memory does not move a 4 GB activation peak. Revisit only if a student crosses a few hundred MB and still misses the cap. No paper was opened that makes streaming the right next step at this size.

## One first experiment

Do this before any new backbone, any INT8 engine, or any live BeamNG session.

**Question.** How stale can the stem token be before the holdout 8 m center RMS moves?

**Setup.** One existing scene checkpoint. No extra training. On the holdout strips only:

- A. Current forward, stem and GRU every frame.
- B. Stem every 2nd frame. GRU every frame. Hold the token on the skipped frame.
- C. Stem every 4th frame. Same GRU rule.
- D. Same as C, but force a stem rerun when the previous confidence is under 0.40 or the frame index jumps (a respawn).

**Metrics, all three, on the same rows.**

- Holdout 8 m center RMS, the number training already uses to pick a checkpoint.
- Milliseconds per frame. On this machine a CPU timing is only a proxy. The number that matters is one process on the 1080 Ti with the 4 GB cap applied, BeamNG closed, so the measurement is the net and not the game.
- Peak VRAM of that process. For A–D on today's net the peak should barely move. Write that down. The experiment is allowed to show "no VRAM change." The decision it supports is whether a later, fatter stem may run at 2–4 Hz.

**Stop rule.** If C (or D, if C loses more than about 2 cm of RMS) is not clearly cheaper per frame than A, do not build a clockwork stem. Spend the next run on distillation (#194) into a wider student instead.

**What this does not prove.** It does not prove INT8, TensorRT, or a UHD offload. Those wait until a stem exists whose activations, not whose 8 MB of weights, are the bill.

## Sources

- [Fast INT8 Inference for Autonomous Vehicles with TensorRT 3](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)
- [TensorRT INT8 calibration notes, DP4A on sm_61](https://kezunlin.me/blog/tensorrt-int8-inference/)
- [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
- [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900)
- [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](https://arxiv.org/abs/1709.01686)
- [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/abs/1703.09844)
- [Clockwork Convnets for Video Semantic Segmentation](https://arxiv.org/abs/1608.03609)
- [shelhamer/clockwork-fcn](https://github.com/shelhamer/clockwork-fcn)
- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)
- [mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module)
- [SBNet: Sparse Blocks Network for Fast Inference](https://arxiv.org/abs/1801.02108)
- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)
- [OpenVINO GPU device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
- [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
