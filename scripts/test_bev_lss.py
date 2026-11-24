#!/usr/bin/env python3
"""LSS BEV head (python/train/bev_lss.py): shape, geometry, visibility, gradient, size.

Each check names the guarantee (G1..G10) from the module docstring.
CPU only. The memory figure is a CPU-derived estimate, not a GPU measurement.
"""
# gvd-ci: requires torch
from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import yaml

from python.train import bev_lss as bl

torch.manual_seed(0)
N_CAMS = 8
GIB = 1024**3
# Stated before measuring; do not widen.
PARAMS_EXPECTED = 147_414
PARAMS_MAX = 200_000  # < 10% of SceneNet's 2,203,694
INFER_BUDGET_BYTES = 4 * GIB
ACT_FRACTION_MAX = 0.25  # BEV head activations must fit in a quarter of the 4 GB budget
ONNX_ATOL = 1e-3
ONNX_RTOL = 1e-4


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))


def _cam_ids() -> list[str]:
    from python.data.strip_writer import SECTORS

    return [s.cam_id for s in SECTORS]


def _rig(batch: int = 1) -> dict:
    return bl.rig_tensors(bl.rig_geometry(_cam_ids(), _cfg()), batch=batch)


def _taps(batch: int = 1, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = (
        torch.randn(batch, N_CAMS, 64, 48, 64),
        torch.randn(batch, N_CAMS, 128, 24, 32).relu(),
        torch.randn(batch, N_CAMS, 128, 12, 16).relu(),
    )
    for t in out:
        t.requires_grad_(requires_grad)
    return out


def _simple_cam(pos=(0.0, 0.0, 1.0), yaw=0.0, pitch=0.0, fx=100.0) -> tuple[torch.Tensor, torch.Tensor]:
    K = torch.tensor([[fx, 0.0, 128.0], [0.0, fx, 96.0], [0.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    T = torch.from_numpy(bl.cam_to_ego(pos, yaw, pitch)).view(1, 1, 4, 4)
    return K, T


# ---------------------------------------------------------------- G4 geometry inputs
def check_intrinsics_from_fov() -> None:
    K = bl.camera_intrinsics(34.0, 640, 480)
    assert abs(K[0, 0] - 96.0 / math.tan(math.radians(17.0))) < 1e-3, K
    assert K[0, 0] == K[1, 1] and K[0, 2] == 128.0 and K[1, 2] == 96.0
    wide = bl.camera_intrinsics(88.7, 640, 360)  # 16:9 letterboxed to 256x144 at row 24
    assert abs(wide[1, 1] - 72.0 / math.tan(math.radians(44.35))) < 1e-3, wide
    assert wide[1, 2] == 96.0 and bl.content_rect(640, 360) == (256, 144, 0, 24)
    geoms = {g.cam_id: g for g in bl.rig_geometry(_cam_ids(), _cfg())}
    fx = {k: round(float(g.K[0, 0]), 1) for k, g in geoms.items()}
    # audit table (redesigner.md item 9)
    assert fx == {"repeatL": 182.9, "pillarL": 149.8, "wide": 73.7, "main": 314.0, "narrow": 508.1,
                  "pillarR": 149.8, "repeatR": 182.9, "rear": 85.8}, fx
    mask = bl.pixel_valid_mask([geoms["wide"], geoms["main"]])
    assert mask.shape == (2, 24, 32)
    assert mask[1].all() and mask[0, 3:21].all() and not mask[0, :3].any() and not mask[0, 21:].any()
    for bad in (0.0, 180.0, -5.0):
        try:
            bl.camera_intrinsics(bad, 640, 480)
        except ValueError:
            continue
        raise AssertionError(f"fov {bad} accepted")
    try:
        bl.rig_geometry(["nope"], _cfg())
    except KeyError:
        pass
    else:
        raise AssertionError("unknown camera accepted")
    print("intrinsics_from_fov")


def check_extrinsics_axes() -> None:
    T = bl.cam_to_ego((1.0, 2.0, 3.0), 0.0, 0.0)
    assert np.allclose(T[:3, :3] @ [0, 0, 1], [0, 1, 0])  # optical axis -> +y forward
    assert np.allclose(T[:3, :3] @ [1, 0, 0], [1, 0, 0])  # image right -> +x right
    assert np.allclose(T[:3, :3] @ [0, 1, 0], [0, 0, -1])  # image down -> -z
    assert np.allclose(T[:3, 3], [1, 2, 3])
    right = bl.cam_to_ego((0, 0, 0), 90.0, 0.0)
    assert np.allclose(right[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)  # yaw +90 looks right
    up = bl.cam_to_ego((0, 0, 0), 0.0, 10.0)
    assert (up[:3, :3] @ [0, 0, 1])[2] > 0  # pitch + looks up
    for yaw, pitch in ((-78.0, -6.0), (150.0, -8.0)):
        R = bl.cam_to_ego((0, 0, 0), yaw, pitch)[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1.0) < 1e-6
    print("extrinsics_axes")


def check_depth_bins() -> None:
    d = bl.depth_bin_centers()
    assert d.shape == (40,) and np.all(np.diff(d) > 0) and 1.0 < d[0] and d[-1] < 48.0
    assert np.allclose(np.diff(np.log(d)), np.diff(np.log(d))[0], rtol=1e-4)  # log spaced
    for args in ((0, 1.0, 48.0), (40, 0.0, 48.0), (40, 5.0, 5.0)):
        try:
            bl.depth_bin_centers(*args)
        except ValueError:
            continue
        raise AssertionError(f"bad bins {args} accepted")
    print("depth_bins")


# ---------------------------------------------------------------- G2 / G3 grid
def check_bev_cell_convention() -> None:
    assert bl.bev_cell(0.0, 0.0) == (64, 64)
    assert bl.bev_cell(-0.01, 0.01) == (63, 63)
    assert bl.bev_cell(-32.0, 31.99) == (0, 0)  # far-left-forward corner
    assert bl.bev_cell(31.99, -31.99) == (127, 127)
    assert bl.bev_cell(0.0, 10.0) == (44, 64)  # forward is up
    assert bl.bev_cell(10.0, 0.0) == (64, 84)  # right is right
    for x, y in ((32.0, 0.0), (0.0, 32.01), (-32.01, 0.0), (0.0, -32.0)):
        assert bl.bev_cell(x, y) is None, (x, y)
    pts = torch.tensor([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, -2.01],
                        [0.0, 0.0, -2.0], [31.99, -31.99, 3.99], [-32.01, 0.0, 0.0]])
    idx, ok = bl.bev_flat_index(pts)
    assert ok.tolist() == [True, True, False, False, False, True, True, False], ok
    assert idx[0].item() == 44 * 128 + 64 and idx[1].item() == 64 * 128 + 84 and idx[6].item() == 127 * 128 + 127
    print("bev_cell_convention")


def check_splat_sum_and_drop() -> None:
    torch.manual_seed(1)
    b, p, c = 2, 500, 3
    pts = (torch.rand(b, p, 3) - 0.5) * torch.tensor([90.0, 90.0, 10.0])
    vals = torch.rand(b, p, c)
    idx, ok = bl.bev_flat_index(pts)
    out = bl.splat(vals, idx, ok)
    assert out.shape == (b, c, 128, 128)
    ref = torch.zeros(b, c, 128 * 128)
    for bi in range(b):
        for pi in range(p):
            if ok[bi, pi]:
                ref[bi, :, idx[bi, pi]] += vals[bi, pi]
    assert torch.allclose(out.reshape(b, c, -1), ref, atol=1e-5)
    # dropped points do not leak into any cell, including the edges and the discard slot
    kept = (vals * ok[..., None]).sum(dim=1)
    assert torch.allclose(out.sum(dim=(2, 3)), kept, atol=1e-4)
    assert 0 < int(ok.sum()) < b * p  # the sample has both kinds
    # batches do not bleed into each other
    solo = bl.splat(vals[1:], idx[1:], ok[1:])
    assert torch.allclose(solo[0], out[1], atol=1e-6)
    print("splat_sum_and_drop")


# ---------------------------------------------------------------- frustum geometry
def check_frustum_point_lands_in_expected_cell() -> None:
    K, T = _simple_cam(pos=(0.0, 0.0, 1.0), fx=100.0)
    depths = torch.tensor([5.0, 20.0])
    pts = bl.frustum_ego_points(K, T, depths, (24, 32))
    assert pts.shape == (1, 1, 2, 24, 32, 3)
    # pixel (row 11, col 20): u = 164, v = 92
    u, v = 164.0, 92.0
    for di, d in enumerate((5.0, 20.0)):
        want = torch.tensor([(u - 128.0) / 100.0 * d, d, 1.0 - (v - 96.0) / 100.0 * d])
        assert torch.allclose(pts[0, 0, di, 11, 20], want, atol=1e-5), (pts[0, 0, di, 11, 20], want)
    # main camera from cameras.yaml: centre-ish pixel at a known depth
    geoms = {g.cam_id: g for g in bl.rig_geometry(_cam_ids(), _cfg())}
    g = geoms["main"]
    Km = torch.from_numpy(g.K).view(1, 1, 3, 3)
    Tm = torch.from_numpy(g.T).view(1, 1, 4, 4)
    d = 10.0
    p = bl.frustum_ego_points(Km, Tm, torch.tensor([d]), (24, 32))[0, 0, 0, 12, 16]
    # pixel (row 12, col 16): u = 132, v = 100, small offsets from the axis; independent closed form
    xc, yc = (132.0 - 128.0) / g.K[0, 0] * d, (100.0 - 96.0) / g.K[1, 1] * d
    pitch = math.radians(-6.0)
    want_y = 1.20 + d * math.cos(pitch) + yc * math.sin(pitch)
    want_z = 1.26 + d * math.sin(pitch) - yc * math.cos(pitch)
    assert torch.allclose(p, torch.tensor([0.0 + xc, want_y, want_z]), atol=1e-4), (p, xc, want_y, want_z)
    assert bl.bev_cell(float(p[0]), float(p[1])) == (math.floor((32 - want_y) / 0.5), math.floor((xc + 32) / 0.5))
    # yawed cameras land on their side
    pl = bl.frustum_ego_points(torch.from_numpy(geoms["pillarL"].K).view(1, 1, 3, 3),
                               torch.from_numpy(geoms["pillarL"].T).view(1, 1, 4, 4), torch.tensor([10.0]), (24, 32))
    assert float(pl[0, 0, 0, 12, 16, 0]) < -8.0
    rr = bl.frustum_ego_points(torch.from_numpy(geoms["rear"].K).view(1, 1, 3, 3),
                               torch.from_numpy(geoms["rear"].T).view(1, 1, 4, 4), torch.tensor([10.0]), (24, 32))
    assert float(rr[0, 0, 0, 12, 16, 1]) < -10.0
    print("frustum_point_lands_in_expected_cell")


class _OneHotSpatial(torch.nn.Module):
    """Stub spatial path: all depth mass in one bin at every pixel, context = 1."""

    def __init__(self, bin_index: int, depth_bins: int = bl.DEPTH_BINS) -> None:
        super().__init__()
        self.bin_index, self.depth_bins = bin_index, depth_bins

    def forward(self, f4, f8, f16, cam_feat):
        n, _c, h, w = f8.shape
        prob = torch.zeros(n, self.depth_bins, h, w)
        prob[:, self.bin_index] = 1.0
        return prob, torch.ones(n, bl.CTX_CH, h, w)


def check_head_point_in_expected_cell_and_drop() -> None:
    K, T = _simple_cam(pos=(0.0, 0.0, 1.0), fx=100.0)
    head = bl.BevLssHead()
    depths = head.depths
    k = int(torch.argmin((depths - 10.0).abs()))
    head.spatial = _OneHotSpatial(k)
    only = torch.zeros(1, 24, 32)
    only[0, 12, 16] = 1.0
    f4, f8, f16 = (t[:, :1] for t in _taps())
    out = head(f4, f8, f16, K, T, pixel_valid=only)
    d = float(depths[k])
    x = (132.0 - 128.0) / 100.0 * d
    z = 1.0 - (100.0 - 96.0) / 100.0 * d
    assert bl.BEV_Z_MIN_M <= z < bl.BEV_Z_MAX_M
    row, col = bl.bev_cell(x, d)
    hit = out["hit_mass"][0, 0]
    assert abs(float(hit[row, col]) - 1.0) < 1e-6, float(hit[row, col])
    assert float(hit.sum()) == float(hit[row, col])  # nothing else
    # hit beyond the grid edge (last bin, ~46 m > 32 m): the hit is dropped, not clamped onto row 0
    head.spatial = _OneHotSpatial(bl.DEPTH_BINS - 1)
    out = head(f4, f8, f16, K, T, pixel_valid=only)
    assert float(out["hit_mass"].sum()) == 0.0
    assert float(out["hit_mass"][0, 0, 0].sum()) == 0.0 and float(out["pass_mass"].sum()) > 0.0
    # a camera wholly outside the grid (40 m ahead, looking forward): every point dropped, all unknown
    Kb, Tb = _simple_cam(pos=(0.0, 40.0, 1.0), fx=100.0)
    out = head(f4, f8, f16, Kb, Tb)
    assert float(out["hit_mass"].sum()) == 0.0 and float(out["pass_mass"].sum()) == 0.0
    assert torch.all(out["logits"][:, bl.CH_UNKNOWN] == bl.FORCE_LOGIT)
    assert torch.all(out["logits"][:, bl.CH_FREE] == -bl.FORCE_LOGIT)
    print("head_point_in_expected_cell_and_drop")


# ---------------------------------------------------------------- G9 ego compensation
def check_ego_compensation_optional() -> None:
    K, T = _simple_cam()
    d = torch.tensor([10.0])
    base = bl.frustum_ego_points(K, T, d, (24, 32))
    for dt, v in ((None, None), (torch.tensor([[0.05]]), None), (None, torch.tensor([20.0]))):
        same = bl.frustum_ego_points(K, T, d, (24, 32), time_offset_s=dt, ego_speed_mps=v)
        assert torch.equal(same, base)
    moved = bl.frustum_ego_points(K, T, d, (24, 32), time_offset_s=torch.tensor([[0.05]]),
                                  ego_speed_mps=torch.tensor([20.0]))
    delta = moved - base
    assert torch.allclose(delta[..., 1], torch.full_like(delta[..., 1], 1.0), atol=1e-5)  # 20 m/s * 50 ms
    assert torch.all(delta[..., 0] == 0) and torch.all(delta[..., 2] == 0)
    # two cameras, two offsets, two batch speeds
    pts = torch.zeros(2, 2, 3, 3)
    out = bl.ego_compensate(pts, torch.tensor([[0.1, -0.1], [0.2, 0.0]]), torch.tensor([10.0, 5.0]))
    assert torch.allclose(out[:, :, 0, 1], torch.tensor([[1.0, -1.0], [1.0, 0.0]]))
    # 2 cells at 0.5 m
    r0 = bl.bev_cell(float(base[0, 0, 0, 12, 16, 0]), float(base[0, 0, 0, 12, 16, 1]))
    r1 = bl.bev_cell(float(moved[0, 0, 0, 12, 16, 0]), float(moved[0, 0, 0, 12, 16, 1]))
    assert r1 == (r0[0] - 2, r0[1]), (r0, r1)
    # not wired to anything live: the head never reads a speed it was not given
    import inspect

    sig = inspect.signature(bl.BevLssHead.forward)
    assert sig.parameters["ego_speed_mps"].default is None and sig.parameters["time_offset_s"].default is None
    src = Path(bl.__file__).read_text(encoding="utf-8")
    for live in ("wheel", "supervisor", "gvd_state", "udp", "socket"):
        assert live not in src.lower().replace("wheel speed", ""), live
    print("ego_compensation_optional")


# ---------------------------------------------------------------- G1 shapes
def check_shapes_every_stage() -> None:
    from python.train.scene_net import SceneNet

    stem = SceneNet().stem
    x = torch.randn(N_CAMS, 3, 192, 256)
    f4, f8, f16 = bl.stem_taps(stem, x)
    assert tuple(f4.shape) == (8, 64, 48, 64) and tuple(f8.shape) == (8, 128, 24, 32) and tuple(f16.shape) == (8, 128, 12, 16)
    assert torch.equal(f16, stem.features(x))  # taps are read-only views of the same stem
    head = bl.BevLssHead()
    rig = _rig()
    cam_feat = head.camera_features(rig["K"], rig["T"])
    assert tuple(cam_feat.shape) == (1, 8, bl.CAM_FEAT)
    main = _cam_ids().index("main")
    assert abs(float(cam_feat[0, main, 0]) - 314.0 / 300.0) < 1e-3 and abs(float(cam_feat[0, main, 2]) - 1.0) < 1e-6
    prob, ctx = head.spatial(f4, f8, f16, cam_feat[0])
    assert tuple(prob.shape) == (8, 40, 24, 32) and tuple(ctx.shape) == (8, 32, 24, 32)
    assert torch.allclose(prob.sum(1), torch.ones(8, 24, 32), atol=1e-5)
    enc = head.encoder(torch.randn(1, bl.CTX_CH + 2, 128, 128))
    assert tuple(enc.shape) == (1, 14, 128, 128)
    with torch.no_grad():
        for batch in (1, 2):
            taps = _taps(batch)
            r = _rig(batch)
            out = head(*(t.detach() for t in taps), r["K"], r["T"], pixel_valid=r["pixel_valid"])
            assert tuple(out["logits"].shape) == (batch, len(bl.BEV_CHANNELS), 128, 128)
            assert tuple(out["hit_mass"].shape) == (batch, 1, 128, 128)
            assert tuple(out["pass_mass"].shape) == (batch, 1, 128, 128)
            assert tuple(out["depth_prob"].shape) == (batch, 8, 40, 24, 32)
    assert len(bl.BEV_CHANNELS) == 14 and "pedestrian" not in " ".join(bl.BEV_CHANNELS)
    print("shapes_every_stage")


# ---------------------------------------------------------------- G5 unknown vs free
def check_visibility_by_construction() -> None:
    head = bl.BevLssHead()
    rig = _rig()
    with torch.no_grad():
        out = head(*_taps(), rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    logits, hit, passed = out["logits"][0], out["hit_mass"][0, 0], out["pass_mass"][0, 0]
    seen = (hit + passed) > 0
    assert 0.2 < float(seen.float().mean()) < 1.0  # some cells seen, some not (near-ego blind ring, gaps)
    assert torch.all(logits[bl.CH_UNKNOWN][~seen] == bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_FREE][~seen] == -bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_FREE][passed <= 0] == -bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_UNKNOWN][seen].abs() < bl.FORCE_LOGIT)  # learned where seen
    assert bool(seen[64 - 30, 64])  # 15 m ahead is seen
    # apply_visibility in isolation: other channels untouched
    raw = torch.randn(1, 14, 4, 4)
    h = torch.zeros(1, 1, 4, 4)
    p = torch.zeros(1, 1, 4, 4)
    h[..., 0, 0] = 1.0
    p[..., 1, 1] = 1.0
    got = bl.apply_visibility(raw, h, p)
    keep = [i for i in range(14) if i not in (bl.CH_FREE, bl.CH_UNKNOWN)]
    assert torch.equal(got[:, keep], raw[:, keep])
    assert got[0, bl.CH_UNKNOWN, 0, 0] == raw[0, bl.CH_UNKNOWN, 0, 0] and got[0, bl.CH_UNKNOWN, 1, 1] == raw[0, bl.CH_UNKNOWN, 1, 1]
    assert got[0, bl.CH_UNKNOWN, 2, 2] == bl.FORCE_LOGIT
    assert got[0, bl.CH_FREE, 0, 0] == -bl.FORCE_LOGIT  # hit only, nothing passed: not free
    assert got[0, bl.CH_FREE, 1, 1] == raw[0, bl.CH_FREE, 1, 1]
    print("visibility_by_construction")


def check_pass_mass_is_survival() -> None:
    """pass_mass counts depth mass beyond the bin: a ray's own hit cell gets none of its own hit."""
    K, T = _simple_cam(pos=(0.0, 0.0, 1.0), fx=100.0)
    head = bl.BevLssHead()
    k = 20
    head.spatial = _OneHotSpatial(k)
    only = torch.zeros(1, 24, 32)
    only[0, 12, 16] = 1.0
    out = head(*(t[:, :1] for t in _taps()), K, T, pixel_valid=only)
    total_pass = float(out["pass_mass"].sum())
    # one-hot at bin k: bins 0..k-1 each pass mass 1 if inside the grid, bins > k pass 0
    pts = bl.frustum_ego_points(K, T, head.depths, (24, 32))[0, 0, :, 12, 16]
    _idx, ok = bl.bev_flat_index(pts)
    assert abs(total_pass - float(ok[:k].sum())) < 1e-5, (total_pass, ok[:k].sum())
    assert abs(float(out["hit_mass"].sum()) - 1.0) < 1e-6
    print("pass_mass_is_survival")


# ---------------------------------------------------------------- G7 no NaN
def check_zero_valid_cameras_no_nan() -> None:
    head = bl.BevLssHead()
    rig = _rig(2)
    taps = _taps(2, requires_grad=True)
    valid = torch.zeros(2, 8)
    valid[1, 3] = 1.0  # batch 1 has one camera, batch 0 has none
    out = head(*taps, rig["K"], rig["T"], cam_valid=valid, pixel_valid=rig["pixel_valid"])
    for name in ("logits", "hit_mass", "pass_mass", "depth_prob"):
        assert torch.isfinite(out[name]).all(), name
    assert float(out["hit_mass"][0].detach().sum()) == 0.0 and float(out["pass_mass"][0].detach().sum()) == 0.0
    assert torch.all(out["logits"][0, bl.CH_UNKNOWN] == bl.FORCE_LOGIT)
    assert torch.all(out["logits"][0, bl.CH_FREE] == -bl.FORCE_LOGIT)
    assert float(out["hit_mass"][1].detach().sum()) > 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], torch.zeros_like(out["logits"]))
    loss.backward()
    for t in taps:
        assert torch.isfinite(t.grad).all()
    for name, p in head.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name
    # NaN-free with extreme but finite taps
    with torch.no_grad():
        big = head(*(t.detach() * 1e4 for t in taps), rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    assert torch.isfinite(big["logits"]).all()
    print("zero_valid_cameras_no_nan")


# ---------------------------------------------------------------- G6 gradients
def check_gradients_reach_stem_and_depth() -> None:
    from python.train.scene_net import SceneNet

    stem = SceneNet().stem
    head = bl.BevLssHead()
    rig = _rig()
    x = torch.randn(N_CAMS, 3, 192, 256)
    f4, f8, f16 = bl.stem_taps(stem, x)
    for t in (f4, f8, f16):
        t.retain_grad()
    out = head(f4[None], f8[None], f16[None], rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    target = torch.rand_like(out["logits"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    for name, t in (("s4", f4), ("s8", f8), ("s16", f16)):
        assert t.grad is not None and float(t.grad.abs().sum()) > 0, name
    assert float(stem.conv[0].weight.grad.abs().sum()) > 0
    for name in ("spatial.depth.weight", "spatial.ctx.weight", "spatial.film.weight", "spatial.lat4.weight",
                 "spatial.lat16.weight", "encoder.head.weight"):
        g = dict(head.named_parameters())[name].grad
        assert g is not None and float(g.abs().sum()) > 0, name
    # focal conditioning matters: changing fx changes the depth distribution
    with torch.no_grad():
        taps = _taps()
        a = head(*taps, rig["K"], rig["T"])["depth_prob"]
        K2 = rig["K"].clone()
        K2[..., 0, 0] *= 2.0
        b = head(*taps, K2, rig["T"])["depth_prob"]
    assert not torch.allclose(a, b)
    print("gradients_reach_stem_and_depth")


# ---------------------------------------------------------------- G8 determinism
def check_deterministic() -> None:
    head = bl.BevLssHead().eval()
    rig = _rig()
    taps = _taps()
    with torch.no_grad():
        a = head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
        b = head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    for k in a:
        assert torch.equal(a[k], b[k]), k
    print("deterministic")


# ---------------------------------------------------------------- G10 size
def _param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def check_param_count() -> None:
    head = bl.BevLssHead()
    n = _param_count(head)
    print(f"  BevLssHead params {n} (spatial {_param_count(head.spatial)}, encoder {_param_count(head.encoder)})")
    assert n == PARAMS_EXPECTED, n
    assert n <= PARAMS_MAX
    print("param_count")


def check_activation_memory_estimate() -> None:
    """CPU-derived estimate, NOT a GPU measurement: profiler-counted CPU allocations of one
    8-camera batch-1 fp32 inference forward (sum of every allocation, an upper bound on peak)."""
    from torch.profiler import ProfilerActivity, profile

    head = bl.BevLssHead().eval()
    rig = _rig()
    taps = _taps()
    with torch.no_grad():
        head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    alloc = sum(e.cpu_memory_usage for e in prof.events() if e.cpu_memory_usage > 0)
    lifted = N_CAMS * bl.DEPTH_BINS * bl.CTX_CH * 24 * 32 * 4  # the largest single stage (analytic)
    params = _param_count(head) * 4
    total = alloc + params
    print(f"  CPU-derived estimate: allocations {alloc / 2**20:.1f} MiB + params {params / 2**20:.2f} MiB "
          f"= {total / 2**20:.1f} MiB = {100 * total / INFER_BUDGET_BYTES:.2f}% of 4 GiB "
          f"(lift tensor {lifted / 2**20:.1f} MiB). Not GPU-measured.")
    assert alloc > lifted  # the profiler did see the forward
    assert total <= ACT_FRACTION_MAX * INFER_BUDGET_BYTES, total
    print("activation_memory_estimate")


# ---------------------------------------------------------------- ONNX
def check_onnx_export_parity() -> None:
    from gvd_test_env import skip_check

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as exc:
        skip_check("bev_lss onnx_export_parity", f"onnx/onnxruntime missing: {exc}")
        return

    class _Wrap(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, f4, f8, f16, K, T, pv):
            out = self.head(f4, f8, f16, K, T, pixel_valid=pv)
            return out["logits"], out["hit_mass"], out["pass_mass"]

    wrap = _Wrap(bl.BevLssHead().eval())
    rig = _rig()
    args = (*_taps(), rig["K"], rig["T"], rig["pixel_valid"])
    buf = io.BytesIO()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(wrap, args, buf, dynamo=False, opset_version=17)
        except TypeError:
            torch.onnx.export(wrap, args, buf, opset_version=17)
    model = onnx.load_from_string(buf.getvalue())
    scatters = [n for n in model.graph.node if n.op_type.startswith("Scatter")]
    assert scatters, "splat must export as a scatter"
    for node in scatters:
        reduction = {a.name: a.s for a in node.attribute}.get("reduction", b"none")
        assert reduction == b"add", (node.op_type, reduction)  # a plain Scatter drops repeated indices
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    feeds = {i.name: a.numpy() for i, a in zip(sess.get_inputs(), args)}
    got = sess.run(None, feeds)
    with torch.no_grad():
        want = [t.numpy() for t in wrap(*args)]
    for name, g, w in zip(("logits", "hit_mass", "pass_mass"), got, want):
        np.testing.assert_allclose(g, w, atol=ONNX_ATOL, rtol=ONNX_RTOL, err_msg=name)
    print("onnx_export_parity")


# ---------------------------------------------------------------- default off, driving path untouched
def check_default_off() -> None:
    assert bl.DEFAULT_ENABLED is False
    assert bl.build_bev_head() is None
    assert bl.build_bev_head({}) is None
    assert bl.build_bev_head({"bev_lss": {"enabled": False}}) is None
    head = bl.build_bev_head({"bev_lss": {"enabled": True}})
    assert isinstance(head, bl.BevLssHead) and head.depths.numel() == bl.DEPTH_BINS
    import python.train.scene_net as sn

    assert "bev_lss" not in Path(sn.__file__).read_text(encoding="utf-8")
    assert not any(n.startswith("bev") for n, _ in sn.SceneNet().named_modules() if n)
    print("default_off")


def main() -> None:
    check_intrinsics_from_fov()
    check_extrinsics_axes()
    check_depth_bins()
    check_bev_cell_convention()
    check_splat_sum_and_drop()
    check_frustum_point_lands_in_expected_cell()
    check_head_point_in_expected_cell_and_drop()
    check_ego_compensation_optional()
    check_shapes_every_stage()
    check_visibility_by_construction()
    check_pass_mass_is_survival()
    check_zero_valid_cameras_no_nan()
    check_gradients_reach_stem_and_depth()
    check_deterministic()
    check_param_count()
    check_activation_memory_estimate()
    check_onnx_export_parity()
    check_default_off()
    print("test_bev_lss: OK")


if __name__ == "__main__":
    main()
