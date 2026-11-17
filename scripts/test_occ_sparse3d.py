#!/usr/bin/env python3
"""Fully sparse multi-camera occupancy core: lift, fuse, memory, read, profiles. No dense volume.

Spec: docs/specs/occ_sparse.md ("Architecture"). CPU only; memory numbers are estimates.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import skip_check  # noqa: E402


def _torch():
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(min(4, torch.get_num_threads()))  # shared CI/box CPUs: avoid oversubscription
    return torch



def _core(cams, rows=12, cols=16, ch=8, keep=4096):
    torch = _torch()
    from python.train.occ_sparse3d import SparseOcc3D

    torch.manual_seed(0)
    return SparseOcc3D(ch, cams, rows, cols, keep=keep).eval()


def check_fusion_geometry() -> None:
    """All 8 cams lift into ONE ego voxel set; front cams land ahead, rear lands behind."""
    from python.data.strip_writer import STRIP_ORDER
    from python.train.occ_sparse3d import FULL_GRID, lift_voxels
    from python.train.occ_sparse import GRID_MIN, GRID_SHAPE, VOXEL_M

    vox = lift_voxels(STRIP_ORDER, 12, 16)
    assert vox.shape[0] == 8 and (vox >= 0).any(axis=(1, 2, 3)).all(), "every cam must reach the grid"

    def ys(ids):
        return GRID_MIN[1] + ((ids // GRID_SHAPE[2]) % GRID_SHAPE[1] + 0.5) * VOXEL_M

    by = {c: vox[i][vox[i] >= 0] for i, c in enumerate(STRIP_ORDER)}
    for front in ("narrow", "main", "wide"):
        assert (ys(by[front]) > 0).all(), front
    assert (ys(by["rear"]) < 0).all(), "rear must look behind"
    union = np.unique(np.concatenate(list(by.values())))
    print(f"  fusion: 8 cams x 12x16 px x 12 depth bins reach {len(union)} frustum voxels of {FULL_GRID}; "
          "only top-2 bins per ray are written")


def check_never_dense() -> None:
    """No tensor in the core is ever grid-sized: the largest has fewer elements than the full 0.5 m grid x C."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_never_dense", "PyTorch is not installed")
        return
    from python.data.strip_writer import STRIP_ORDER
    from python.train.occ_sparse3d import FEAT_C, FULL_GRID

    net = _core(STRIP_ORDER)
    seen = []
    orig_zeros = torch.Tensor.new_zeros

    def spy(self, *size, **kw):
        out = orig_zeros(self, *size, **kw)
        seen.append(out.numel())
        return out

    torch.Tensor.new_zeros = spy
    try:
        s, st = net.step(torch.randn(1, 8, 8, 12, 16), 0.1)
        s, st = net.step(torch.randn(1, 8, 8, 12, 16), 0.1, st)
    finally:
        torch.Tensor.new_zeros = orig_zeros
    stored = int((s.ids >= 0).sum())
    with torch.no_grad():
        ids, _ = net.splat(torch.randn(1, 8, 8, 12, 16))
    active = int(torch.unique(ids[ids >= 0]).numel())
    frustum = int(net.frustum.numel())
    assert max(seen) <= (net.samples + net.keep) * FEAT_C < FULL_GRID * FEAT_C, (max(seen), stored)
    assert active <= net.samples and active < frustum // 2, (active, net.samples, frustum)
    assert stored <= net.keep and s.ids.shape[1] == net.keep and s.feat.shape == (1, net.keep, FEAT_C)
    print(f"  sparse: per tick {active} active (cap {net.samples}) vs {frustum} frustum vs {FULL_GRID} grid voxels; "
          f"stored {stored} (cap {net.keep}); largest buffer {max(seen)} vs full-grid x C {FULL_GRID * FEAT_C}")


def check_grid_growth_flat() -> None:
    """Grow range x2 and resolution x2 (full grid x32): stored activations stay flat, buffers never track the grid."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_grid_growth", "PyTorch is not installed")
        return
    import python.train.occ_sparse3d as S
    from python.data.strip_writer import STRIP_ORDER

    saved = (S.GRID_MIN, S.GRID_SHAPE, S.VOXEL_M, S.FULL_GRID)
    rows = []
    try:
        for name, lo, hi, vm in (("base", (-16, -16, -1), (16, 40, 3), 0.5), ("range x2", (-32, -32, -2), (32, 80, 6), 0.5),
                                 ("range x2 + res x2", (-32, -32, -2), (32, 80, 6), 0.25)):
            S.GRID_MIN, S.VOXEL_M = lo, vm
            S.GRID_SHAPE = tuple(int(round((h - l) / vm)) for l, h in zip(lo, hi))
            S.FULL_GRID = int(np.prod(S.GRID_SHAPE))
            net = _core(STRIP_ORDER, keep=2048)
            seen = []
            orig = torch.Tensor.new_zeros

            def spy(self, *size, **kw):
                out = orig(self, *size, **kw)
                seen.append(out.numel())
                return out

            torch.Tensor.new_zeros = spy
            try:
                with torch.no_grad():
                    s, st = net.step(torch.randn(1, 8, 8, 12, 16), 0.1)
                    s, st = net.step(torch.randn(1, 8, 8, 12, 16), 0.1, st)
            finally:
                torch.Tensor.new_zeros = orig
            rows.append((name, S.FULL_GRID, int(net.frustum.numel()), int(s.feat.numel()), max(seen)))
    finally:
        S.GRID_MIN, S.GRID_SHAPE, S.VOXEL_M, S.FULL_GRID = saved
    stored = {r[3] for r in rows}
    cap = (8 * 12 * 16 * S.TOPK_D + 2048) * S.FEAT_C  # candidates + carried memory, independent of the grid
    assert len(stored) == 1, rows
    assert rows[-1][1] >= 32 * rows[0][1] and all(r[4] <= cap for r in rows), (cap, rows)
    for r in rows:
        print(f"  grid {r[0]}: full {r[1]}, frustum {r[2]}, stored feat {r[3]} floats, largest buffer {r[4]} (cap {cap})")


def check_splat_touches_only_own_voxels() -> None:
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_splat", "PyTorch is not installed")
        return
    from python.data.strip_writer import STRIP_ORDER

    net = _core(STRIP_ORDER)
    maps = torch.randn(1, 8, 8, 12, 16)
    with torch.no_grad():
        ids0, f0 = net.splat(maps)
        k = STRIP_ORDER.index("pillarL")
        bump = maps.clone()
        bump[:, k] = torch.randn_like(bump[:, k])
        ids1, f1 = net.splat(bump)
    per = ids0.shape[1] // 8
    cam = torch.arange(ids0.shape[1]) // per
    other = cam != k
    assert torch.equal(ids0[0, other], ids1[0, other]) and torch.allclose(f0[0, other], f1[0, other]), "other cams moved"
    own = set(net.vox[k][net.vox[k] >= 0].tolist())
    mine = ids1[0, ~other]
    wrote = set(mine[mine >= 0].tolist())
    assert wrote and wrote <= own and per == 12 * 16 * 2, "a cam writes only its own ray voxels, 2 bins per ray"
    print(f"  splat: pillarL change wrote {len(wrote)} voxels (top-2 bins), all inside its {len(own)} frustum voxels")


def check_reduced_cams() -> None:
    """main+wide and main+wide+narrow build and run end to end through SceneNet; a wrong cam count is refused."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_reduced", "PyTorch is not installed")
        return
    from python.train.scene_net import SceneNet

    for cams in (("wide", "main"), ("wide", "main", "narrow")):
        net = SceneNet(cams=cams, occ_sparse=True).eval()
        strip = torch.zeros(1, 3, int(net.canvas_h), int(net.canvas_w))
        with torch.no_grad():
            pred, _ = net.forward(strip, 0.1, net.initial_state())
            maps = pred.occ_maps.unsqueeze(0)  # forward() returns the unbatched tick
            vox, _ = net.occ_core.step(maps, 0.1)
            out = net.occ_head(pred.occ_feat.reshape(1, -1), torch.arange(32).reshape(1, -1), vox)
        assert maps.shape[1] == len(cams) and tuple(out.shape) == (1, 32, 6), (maps.shape, out.shape)
    try:
        net.occ_core.splat(torch.zeros(1, 2, *maps.shape[2:]))
    except ValueError:
        pass
    else:
        raise AssertionError("3-cam core must refuse 2-cam maps")
    print("  reduced: main+wide and main+wide+narrow run; wrong cam count raises")


def check_stale_cam_masked() -> None:
    """A cam older than stale_s cannot write: its maps stop mattering, its private voxels are not stored."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_stale", "PyTorch is not installed")
        return
    from python.data.cam_hold import CAM_AGE_ORDER
    from python.data.strip_writer import STRIP_ORDER

    net = _core(STRIP_ORDER)
    maps = torch.randn(1, 8, 8, 12, 16)
    ages8 = torch.zeros(1, 8)
    ages8[0, list(CAM_AGE_ORDER).index("rear")] = 1.0  # 1 Hz cam, a second old
    ages = net.ages_for(ages8)
    k = STRIP_ORDER.index("rear")
    other = maps.clone()
    other[:, k] = torch.randn_like(other[:, k]) * 10
    with torch.no_grad():
        a, b = net.fuse(maps, ages), net.fuse(other, ages)
        assert torch.equal(a.ids, b.ids) and torch.allclose(a.feat, b.feat), "stale rear still writes"
        a, b = net.fuse(maps), net.fuse(other)
        assert not (torch.equal(a.ids, b.ids) and torch.allclose(a.feat, b.feat)), "fresh rear must write"
        rear_only = set(net.vox[k][net.vox[k] >= 0].tolist())
        for i in range(8):
            if i != k:
                rear_only -= set(net.vox[i][net.vox[i] >= 0].tolist())
        stored = set(net.fuse(maps, ages).ids[0].tolist())
        fresh_stored = set(net.fuse(maps).ids[0].tolist())
    assert rear_only and not (rear_only & stored), "voxels only a stale cam sees must not be stored"
    assert rear_only & fresh_stored, "fresh rear voxels are stored"
    print(f"  stale: rear age 1.0 s > {net.stale_s} s masked; {len(rear_only)} rear-only voxels left to memory")


def check_memory_ego_aligned() -> None:
    """Ego drives forward through a static scene: a carried voxel stays put in WORLD space (exact coords)."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_memory", "PyTorch is not installed")
        return
    from python.train.occ_sparse3d import _centers, voxel_id, warp_ids

    ids = torch.as_tensor(voxel_id(np.array([1.25]), np.array([21.25]), np.array([0.25]))).reshape(1, 1)
    ego_y = 0.0
    for _ in range(4):  # 10 m/s, dt 0.2 -> 2 m per tick
        ids = warp_ids(ids, torch.tensor([2.0]), torch.tensor([0.0]))
        ego_y += 2.0
        x, y, _z = _centers(ids)[0, 0].tolist()
        assert abs(x - 1.25) < 1e-6 and abs(y + ego_y - 21.25) < 1e-6, (x, y, ego_y)
    turned = warp_ids(torch.as_tensor(voxel_id(np.array([0.25]), np.array([9.25]), np.array([0.25]))).reshape(1, 1),
                      torch.tensor([0.0]), torch.tensor([math.pi / 2]))
    x, y, _z = _centers(turned)[0, 0].tolist()
    assert abs(x - 9.25) < 0.51 and abs(y + 0.25) < 0.51, f"left turn puts what was ahead on the right: {(x, y)}"
    side = warp_ids(ids, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]))
    assert abs(_centers(side)[0, 0, 0].item() - (1.25 - 1.0)) < 1e-6, "ego 1 m right -> static voxel 1 m further left"
    gone = warp_ids(ids, torch.tensor([30.0]), torch.tensor([0.0]))
    assert int(gone[0, 0]) == -1, "a voxel that leaves the grid is dropped"
    print("  memory: static voxel world y stays 21.25 m over 4 ticks at 2 m/tick (0.000 m err, exact transform); "
          "90 deg left -> right; out-of-grid dropped")


def check_step_ego_sources_and_grads() -> None:
    """Flag off: memory warps by the camera-only VO; flag on: by speed*dt. Loss at t reaches t-1's maps."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_step", "PyTorch is not installed")
        return
    from python.data.strip_writer import STRIP_ORDER
    from python.train.occ_sparse3d import SparseState, warp_ids

    net = _core(STRIP_ORDER)
    with torch.no_grad():
        net.vo[-1].weight.zero_()
        net.vo[-1].bias.copy_(torch.tensor([0.0, 3.0, 0.0]))  # camera-only VO (dx, dy fwd, dyaw): 3 m forward
    maps = torch.randn(1, 8, 8, 12, 16)
    with torch.no_grad():
        first, st = net.step(maps, 0.2)
        nxt, st2 = net.step(torch.zeros_like(maps), 0.2, st)
    assert torch.allclose(st2.vo, torch.tensor([[0.0, 3.0, 0.0]]))
    want = set(warp_ids(first.ids, torch.tensor([3.0]), torch.tensor([0.0]))[0].tolist()) - {-1}
    assert want and want <= set(nxt.ids[0].tolist()) | set(first.ids[0].tolist()), "VO-warped memory missing"
    carried = want & set(nxt.ids[0].tolist())
    assert len(carried) > 0.5 * len(want), (len(carried), len(want))
    try:
        net.step(maps, 0.2, st, torch.tensor([[10.0, 0.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("vehicle ego must be refused while ego_from_vehicle is off")
    net.ego_from_vehicle = True
    with torch.no_grad():
        veh, _ = net.step(torch.zeros_like(maps), 0.2, st, torch.tensor([[10.0, 0.0]]))
    want_v = set(warp_ids(first.ids, torch.tensor([2.0]), torch.tensor([0.0]))[0].tolist()) - {-1}
    assert len(want_v & set(veh.ids[0].tolist())) > 0.5 * len(want_v)
    net = _core(STRIP_ORDER).train()
    seq = torch.randn(1, 2, 8, 8, 12, 16, requires_grad=True)
    outs, _ = net.window(seq, 0.1)
    from python.train.occ_sparse3d import gather

    gather(outs[1], outs[0].ids.clamp(min=0)).pow(2).sum().backward()
    g = float(seq.grad[0, 0].abs().sum())
    assert g > 0, "temporal memory must backprop into the previous tick"
    print(f"  step: flag off -> VO warp (3 m), {len(carried)}/{len(want)} carried; flag on -> speed*dt; "
          f"grad to t-1 maps {g:.3g}")


def check_vo_drift_unproven() -> None:
    """Honest gate: VO drift in metres needs recorded clips + training; untrained VO must not be called accurate."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_vo", "PyTorch is not installed")
        return
    from python.data.strip_writer import STRIP_ORDER
    from python.train.occ_sparse3d import SparseState

    net = _core(STRIP_ORDER)
    maps = torch.randn(1, 8, 8, 12, 16)
    with torch.no_grad():
        _s, st = net.step(maps, 0.1)
        _s, st = net.step(maps, 0.1, st)
    assert st.vo.shape == (1, 3) and torch.isfinite(st.vo).all()
    print(f"  vo: camera-only estimate wired (dx, dy fwd, dyaw; untrained dy {float(st.vo[0, 1]):+.3f} m); clip drift UNPROVEN until "
          "trained on recorded pose deltas")


def check_overfit_with_flow() -> None:
    """Cams -> sparse set -> query head overfits occ, class AND per-cell flow on a fixed tick."""
    torch = _torch()
    if torch is None:
        skip_check("occ_sparse_overfit", "PyTorch is not installed")
        return
    from python.data.strip_writer import STRIP_ORDER
    from python.train.occ_sparse import SparseOccHead, build_queries, occ_loss
    from python.train.occ_sparse3d import FEAT_C

    torch.manual_seed(0)
    net = _core(STRIP_ORDER, keep=2048).train()
    rng = np.random.default_rng(3)
    active = net.frustum.numpy()
    labels = []
    for _ in range(2):
        occ = np.unique(rng.choice(active, 40, replace=False)).astype(np.int32)  # labels where cams see
        free = np.setdiff1d(rng.choice(active, 60, replace=False), occ).astype(np.int32)
        cls = rng.integers(0, 3, len(occ)).astype(np.uint8)
        flow = np.where(cls[:, None] == 2, rng.normal(0, 3, (len(occ), 2)), 0.0).astype(np.float16)
        labels.append({"occ_idx": occ, "occ_cls": cls, "free_idx": free, "occ_flow": flow,
                       "flow_valid": np.ones(len(occ), np.uint8)})
    q, t_occ, t_cls, m, t_flow, f_m = build_queries(labels, k=128, seed=0, with_flow=True)
    head = SparseOccHead(32, vox_ch=FEAT_C + 1)
    maps = torch.randn(2, 8, 8, 12, 16)
    feat = torch.zeros(2, 32)  # scene feature off: everything must come through the sparse set
    opt = torch.optim.Adam(list(head.parameters()) + list(net.parameters()), lr=5e-3)
    first = first_flow = None
    for _ in range(150):
        opt.zero_grad()
        vox, _ = net.step(maps, 0.1)
        out = head(feat, q, vox)
        loss = occ_loss(out, t_occ, t_cls, m, t_flow, f_m)
        ferr = float(((out[..., 4:6] - t_flow).abs().sum(-1) * f_m).sum() / f_m.sum())
        first = float(loss) if first is None else first
        first_flow = ferr if first_flow is None else first_flow
        loss.backward()
        opt.step()
    acc = float((((out[..., 0] > 0).float() == t_occ).float() * m).sum() / m.sum())
    assert float(loss) < 0.25 * first and acc > 0.9 and ferr < 0.5 * first_flow, (first, float(loss), acc, first_flow, ferr)
    print(f"  overfit(sparse+flow): loss {first:.3f} -> {float(loss):.4f}, occ acc {acc:.3f}, "
          f"flow L1 {first_flow:.2f} -> {ferr:.3f} m/s")


def check_profiles() -> None:
    """Each profile builds and runs; weights carry across res/frames/keep, refuse cleanly across width/cams."""
    torch = _torch()
    if torch is None:
        skip_check("occ_profiles", "PyTorch is not installed")
        return
    from dataclasses import replace

    from python.train import occ_profiles as P

    for name in P.PROFILES:
        e = P.estimate(name)
        assert e["logits"][-1] == 6 and e["stored"] <= P.get(name).keep < e["full_grid"], e
        print(f"  profile {name}: {e['params']} params ({e['occ_params']} occ), canvas {e['canvas']}, "
              f"active {e['active_voxels']} (cap {e['candidates_cap']}) / frustum {e['frustum_voxels']} / "
              f"stored {e['stored']} / full {e['full_grid']} voxels, memory state {e['temporal_state_kb']} KB, "
              f"~{e['est_infer_mb_cpu_estimate']} MB net-only (CPU estimate, not VRAM)")
    base = P.get("1080ti")
    state = P.build(base).state_dict()
    for knob in (replace(base, name="res", scale=0.75), replace(base, name="frames", frames=8, stale_s=1.1),
                 replace(base, name="keep", keep=1024), replace(base, name="fp16", precision="fp16")):
        P.load_into(state, base, knob)
    for bad, why in ((P.get("high"), "preset"), (replace(base, name="2cam", cams=("wide", "main")), "cams")):
        try:
            P.load_into(state, base, bad)
        except P.ProfileMismatch as err:
            assert why in str(err), err
        else:
            raise AssertionError(f"{bad.name} must refuse a 1080ti checkpoint")
        try:  # and the guard is not hiding a load that would have worked
            P.build(bad).load_state_dict(state, strict=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{bad.name} unexpectedly loads 1080ti weights; drop it from retrain_knobs")
    print("  profiles: res/frames/keep/precision load 1080ti weights; high (width) and 2-cam refuse with ProfileMismatch")


_POISON = r'''
import sys, importlib.abc
BLOCK = ("cv2", "PIL", "imageio", "python.data", "python.run_vision", "python.train.scene_net", "python.train.occ_sparse3d")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.startswith(BLOCK):
            raise ImportError("viz path touched image code: " + name)
for m in [m for m in sys.modules if m.startswith(BLOCK)]:
    del sys.modules[m]
sys.meta_path.insert(0, Block())
class Poison:
    def __getattr__(self, k): raise RuntimeError("viz read a camera array")
    def __array__(self, *a, **k): raise RuntimeError("viz read a camera array")
    def __getitem__(self, k): raise RuntimeError("viz read a camera array")
    def __len__(self): raise RuntimeError("viz read a camera array")
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/scripts"]
from test_occ_viz import _pred, _road_occ
from python.viz.occ_scene import MotionTracker, occ_to_viz, validate_message
tr = MotionTracker()
for f in range(3):
    out = dict(_pred(f), occ=_road_occ())
    out.update(strip=Poison(), frames=Poison(), image=Poison(), occ_maps=Poison())
    msg = occ_to_viz(out, tracker=tr, seq=f, t=0.1 * f, ego_speed_mps=10.0)
    assert not validate_message(msg), validate_message(msg)
print("ok", len(msg["occ"]), sorted(m for m in sys.modules if m.startswith(("cv2", "PIL", "python.data"))))
'''


def check_viz_never_touches_images() -> None:
    """Downstream of the heads: image imports blocked + camera arrays poisoned, occ_to_viz still runs."""
    out = subprocess.run(
        [sys.executable, "-c", _POISON, str(ROOT)], cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("ok") and last.endswith("[]"), last
    print(f"  image-free: occ_to_viz x3 with cv2/PIL/python.data blocked and strip/frames poisoned -> {last}")


def check_runtime_has_no_gt_pose() -> None:
    """Ego motion for the warp is a caller input (VO, or vehicle signals behind --ego-from-vehicle); never sim truth."""
    import ast

    banned = ("beamngpy", "scenetree", "vehicle_labels", "occ_labels", "python.data.raw_frames", "python.sim")
    for rel in ("python/train/occ_sparse3d.py", "python/train/occ_profiles.py", "python/viz/occ_scene.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        hits = [m for m in names if any(b in m for b in banned)]
        assert not hits, (rel, hits)
        src = (ROOT / rel).read_text(encoding="utf-8")
        extra = ("tech[", "speed_mps") if "occ_scene" not in rel else ()  # viz shows v1 ego speed by contract
        for key in extra + ("gt_pose", "scenetree", "get_vehicle_pose", "\"pos\"]"):
            assert key not in src, (rel, key)
    print("  odometry: occ_sparse3d / occ_profiles / occ_scene import no GT pose source; ego is an explicit input")


def _timed(fn) -> None:
    import time

    t0 = time.monotonic()
    fn()
    print(f"    ({fn.__name__} {time.monotonic() - t0:.1f}s)", flush=True)


def main() -> int:
    for fn in (check_runtime_has_no_gt_pose, check_fusion_geometry, check_viz_never_touches_images,
               check_never_dense, check_grid_growth_flat, check_splat_touches_only_own_voxels, check_stale_cam_masked,
               check_memory_ego_aligned, check_step_ego_sources_and_grads, check_vo_drift_unproven,
               check_reduced_cams, check_overfit_with_flow, check_profiles):
        _timed(fn)
    print("OK test_occ_sparse3d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
