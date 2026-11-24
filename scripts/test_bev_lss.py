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
from python.sensors.rig_overrides import BASE_RIG_MODEL

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
    return bl.rig_tensors(bl.rig_geometry(_cam_ids(), _cfg(), model=BASE_RIG_MODEL), batch=batch)


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
    geoms = {g.cam_id: g for g in bl.rig_geometry(_cam_ids(), _cfg(), model=BASE_RIG_MODEL)}
    fx = {k: round(float(g.K[0, 0]), 1) for k, g in geoms.items()}
    # audit table (redesigner.md item 9), by hand: fy = (content_h / 2) / tan(fov_v / 2) with content_h = 192
    # for 640x480 and 144 for 640x360, e.g. main 96 / tan(17 deg) = 314.0, rear 72 / tan(40 deg) = 85.8
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
        bl.rig_geometry(["nope"], _cfg(), model=BASE_RIG_MODEL)
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
    # main camera from cameras.yaml: centre-ish pixel at a known depth (uses the module K; the fully
    # hand-derived version with literal numbers is check_geometry_hand_points)
    geoms = {g.cam_id: g for g in bl.rig_geometry(_cam_ids(), _cfg(), model=BASE_RIG_MODEL)}
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
    clear = out["clear_mass"][0, 0]
    seen = clear > 0  # unblocked observation; residual mass behind a hit is not "seen"
    assert 0.2 < float(seen.float().mean()) < 1.0  # some cells seen, some not (near-ego blind ring, gaps)
    assert torch.all(logits[bl.CH_UNKNOWN][~seen] == bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_FREE][~seen] == -bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_FREE][passed <= 0] == -bl.FORCE_LOGIT)
    frac = passed / (hit + passed + 1e-12)
    assert torch.all(logits[bl.CH_FREE][frac < bl.PASS_FRAC_TAU] == -bl.FORCE_LOGIT)
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



def check_pass_fraction_gate() -> None:
    """observed_free is allowed only when pass / (hit + pass) >= PASS_FRAC_TAU (default 0.5)."""
    raw = torch.randn(1, 14, 2, 2)
    h = torch.zeros(1, 1, 2, 2)
    p = torch.zeros(1, 1, 2, 2)
    # below 0.5: hit=0.9, pass=0.1 (frac=0.1), still > 0 so a `passed > 0` gate would allow free
    h[..., 0, 0] = 0.9
    p[..., 0, 0] = 0.1
    # above 0.5: hit=0.4, pass=0.6 (frac=0.6)
    h[..., 0, 1] = 0.4
    p[..., 0, 1] = 0.6
    got = bl.apply_visibility(raw, h, p)
    assert got[0, bl.CH_FREE, 0, 0] == -bl.FORCE_LOGIT, "sub-threshold pass share must force free low"
    assert got[0, bl.CH_FREE, 0, 1] == raw[0, bl.CH_FREE, 0, 1], "at-or-above-threshold pass share may be free"
    assert bl.PASS_FRAC_TAU == 0.5, bl.PASS_FRAC_TAU
    print("pass_fraction_gate")


def check_wall_behind_is_unknown_not_free() -> None:
    """G5: a confident wall makes every cell behind it unknown (and occluded), never free.

    Wall at bin 12 (3.4 m) of the main camera, softmax margins 8, 15 and 25
    (p_hit 0.987, ~1, 1). All 776 cells only reached beyond the wall must have
    free forced low and unknown (and occluded) forced high.
    """
    class Peaked(torch.nn.Module):
        def __init__(self, k, m):
            super().__init__()
            self.k, self.m = k, m

        def forward(self, f4, f8, f16, cam):
            n, _, h, w = f8.shape
            lg = torch.zeros(n, bl.DEPTH_BINS, h, w)
            lg[:, self.k] = self.m
            return lg.softmax(1), torch.ones(n, bl.CTX_CH, h, w)

    cfg = _cfg()
    geoms = bl.rig_geometry(["main"], cfg, model=BASE_RIG_MODEL)
    rig = bl.rig_tensors(geoms)
    head = bl.BevLssHead().eval()
    k = 12
    centre = float(head.depths[k])
    assert 3.3 < centre < 3.5, centre  # 3.4 m ahead
    ch_occ = bl.BEV_CHANNELS.index("occluded")
    f4 = torch.randn(1, 1, 64, 48, 64)
    f8 = torch.randn(1, 1, 128, 24, 32)
    f16 = torch.randn(1, 1, 128, 12, 16)
    behind_n = None
    for m in (8.0, 15.0, 25.0):
        head.spatial = Peaked(k, m)
        with torch.no_grad():
            out = head(f4, f8, f16, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
        lg = out["logits"][0]
        pts = bl.frustum_ego_points(rig["K"], rig["T"], head.depths, (24, 32))[0, 0]
        idx, ok = bl.bev_flat_index(pts)
        behind = torch.zeros(128 * 128, dtype=torch.bool)
        front = torch.zeros(128 * 128, dtype=torch.bool)
        in_front = torch.zeros(128 * 128, dtype=torch.bool)
        behind[idx[k + 1 :][ok[k + 1 :]]] = True
        front[idx[: k + 1][ok[: k + 1]]] = True
        in_front[idx[:k][ok[:k]]] = True
        behind = (behind & ~front).view(128, 128)
        in_front = (in_front & ~behind.view(-1)).view(128, 128)
        n_behind = int(behind.sum())
        if behind_n is None:
            behind_n = n_behind
        assert n_behind == 776, n_behind
        p_hit = float(torch.softmax(torch.tensor([m] + [0.0] * (bl.DEPTH_BINS - 1)), 0)[0])
        free_open = behind & (lg[bl.CH_FREE] != -bl.FORCE_LOGIT)
        unk_learned = behind & (lg[bl.CH_UNKNOWN] != bl.FORCE_LOGIT)
        occ_open = behind & (lg[ch_occ] != bl.FORCE_LOGIT)
        assert int(free_open.sum()) == 0, (
            f"margin {m}: p_hit={p_hit:.8f} free NOT forced low={int(free_open.sum())} of {n_behind}"
        )
        assert int(unk_learned.sum()) == 0, (
            f"margin {m}: p_hit={p_hit:.8f} unknown NOT forced high={int(unk_learned.sum())} of {n_behind}"
        )
        assert int(occ_open.sum()) == 0, (
            f"margin {m}: p_hit={p_hit:.8f} occluded NOT forced high={int(occ_open.sum())} of {n_behind}"
        )
        # rays that have not yet hit the wall stay learned, not forced unknown / not-free
        assert int(in_front.sum()) > 0
        assert torch.all(lg[bl.CH_UNKNOWN][in_front].abs() < bl.FORCE_LOGIT), m
        assert not torch.all(lg[bl.CH_FREE][in_front] == -bl.FORCE_LOGIT), m
    assert 0.0 < bl.OCCLUDE_HIT_TAU <= 0.987, bl.OCCLUDE_HIT_TAU
    src = Path(bl.__file__).read_text(encoding="utf-8")
    assert "PASS_FRAC_TAU" in src and "OCCLUDE_HIT_TAU" in src
    assert "pass_mass / (hit_mass + pass_mass)" in src or "pass_mass/(hit_mass+pass_mass)" in src
    print("wall_behind_is_unknown_not_free")


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
    small = bl.build_bev_head({"bev_lss": {"enabled": True, "depth_bins": 20}})
    assert small.depths.numel() == 20 and small.spatial.depth.out_channels == 20
    assert bl.rig_tensors(bl.rig_geometry(["main"], _cfg(), model=BASE_RIG_MODEL))["K"].shape == (1, 1, 3, 3)  # default batch 1
    import python.train.scene_net as sn

    assert "bev_lss" not in Path(sn.__file__).read_text(encoding="utf-8")
    assert not any(n.startswith("bev") for n, _ in sn.SceneNet().named_modules() if n)
    import subprocess

    probe = (
        "import sys\n"
        "import python.run_vision  # noqa: F401\n"
        "import python.perception.pipeline  # noqa: F401\n"
        "assert 'python.train.bev_lss' not in sys.modules, "
        "sorted(m for m in sys.modules if 'bev' in m)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
             "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    print("default_off")



def check_rig_geometry_applies_vehicle_overrides() -> None:
    """rig_geometry applies cameras.yaml vehicle_overrides via rig_for_model, or raises."""
    from python.collect.level_session import EGO_MODEL_DEFAULT
    from python.sensors.rig_overrides import BASE_RIG_MODEL

    cfg = _cfg()
    assert EGO_MODEL_DEFAULT == "sbr"
    assert "vehicle_overrides" in cfg and "sbr" in cfg["vehicle_overrides"]
    sbr_pos = {cid: list(map(float, p["pos_m"])) for cid, p in cfg["vehicle_overrides"]["sbr"].items()}
    assert sbr_pos["repeatL"] == [-1.10, 0.00, 0.90] and sbr_pos["repeatR"] == [1.10, 0.00, 0.90]
    try:
        sbr = {g.cam_id: g for g in bl.rig_geometry(["repeatL", "repeatR"], cfg, model="sbr")}
        base = {g.cam_id: g for g in bl.rig_geometry(["repeatL", "repeatR"], cfg, model=BASE_RIG_MODEL)}
    except TypeError as err:
        raise AssertionError("rig_geometry must take model and apply rig_for_model") from err
    assert np.allclose(sbr["repeatL"].T[:3, 3], [-1.10, 0.00, 0.90], atol=1e-5), sbr["repeatL"].T[:3, 3]
    assert np.allclose(sbr["repeatR"].T[:3, 3], [1.10, 0.00, 0.90], atol=1e-5), sbr["repeatR"].T[:3, 3]
    assert np.allclose(base["repeatL"].T[:3, 3], [-0.81, 1.25, 0.67], atol=1e-5), base["repeatL"].T[:3, 3]
    assert np.allclose(base["repeatR"].T[:3, 3], [0.81, 1.25, 0.67], atol=1e-5), base["repeatR"].T[:3, 3]
    # 1.25 m / 0.29 m mount error on the default eSBR is about 2 cells at 0.5 m
    for cid, depth, base_cell, sbr_cell in (
        ("repeatL", 7.5, (74, 54), (76, 54)),
        ("repeatL", 20.0, (95, 41), (97, 41)),
        ("repeatR", 7.5, (74, 72), (76, 73)),
        ("repeatR", 20.0, (96, 84), (98, 85)),
    ):
        def _cell(geom, d):
            K = torch.from_numpy(geom.K).view(1, 1, 3, 3)
            T = torch.from_numpy(geom.T).view(1, 1, 4, 4)
            pts = bl.frustum_ego_points(K, T, torch.tensor([d], dtype=torch.float32), (24, 32))
            mid = pts[0, 0, 0, 12, 16]
            return bl.bev_cell(float(mid[0]), float(mid[1]))

        assert _cell(base[cid], depth) == base_cell, (cid, depth, _cell(base[cid], depth))
        assert _cell(sbr[cid], depth) == sbr_cell, (cid, depth, _cell(sbr[cid], depth))
    try:
        bl.rig_geometry(["repeatL"], cfg)
    except ValueError as err:
        msg = str(err).lower()
        assert "model" in msg or "override" in msg, err
    else:
        raise AssertionError("rig_geometry must raise when vehicle_overrides is present and no model is given")
    print("rig_geometry_applies_vehicle_overrides")


# ---------------------------------------------------------------- mutation-guided edge checks
def _raises(exc, fn, *args, match: str = "") -> None:
    try:
        fn(*args)
    except exc as err:
        assert match in str(err), (match, str(err))
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise {exc.__name__}")


def check_geometry_helper_edges() -> None:
    assert bl.content_rect(640, 480) == (256, 192, 0, 0)
    assert bl.content_rect(1000, 100) == (256, 26, 0, 83)  # very wide: letterbox rows
    assert bl.content_rect(100, 1000) == (19, 192, 118, 0)  # very tall: pillarbox columns
    assert all(isinstance(v, int) for v in bl.content_rect(1000, 100) + bl.content_rect(100, 1000))
    assert bl.content_rect(640, 480, 128, 96) == (128, 96, 0, 0)
    assert bl.content_rect(100000, 1) == (256, 1, 0, 95)  # degenerate sizes clamp to one pixel, not two
    assert bl.content_rect(1, 100000) == (1, 192, 127, 0)
    K = bl.camera_intrinsics(90.0, 640, 360)  # 256x144 at row 24, tan(45) = 1
    assert np.allclose(K, [[72.0, 0.0, 128.0], [0.0, 72.0, 96.0], [0.0, 0.0, 1.0]], atol=1e-4), K
    Ks = bl.camera_intrinsics(90.0, 640, 480, 128, 96)  # crop size is honoured
    assert np.allclose(Ks, [[48.0, 0.0, 64.0], [0.0, 48.0, 48.0], [0.0, 0.0, 1.0]], atol=1e-4), Ks
    assert bl.camera_intrinsics(0.5, 640, 480)[0, 0] > 10000  # tiny but legal fov
    _raises(ValueError, bl.camera_intrinsics, 0.0, 640, 480, match="fov_v")
    assert bl.depth_bin_centers(1, 1.0, 4.0).tolist() == [2.5]  # one bin is legal
    _raises(ValueError, bl.depth_bin_centers, 0, 1.0, 4.0, match="bad depth bins")
    _raises(ValueError, bl.depth_bin_centers, 4, 0.0, 4.0, match="bad depth bins")
    cfg = _cfg()
    junk = {"cameras": ["not a camera", {"no_id": 1}] + list(cfg["cameras"])}
    geoms = {g.cam_id: g for g in bl.rig_geometry(["main", "pillarL"], junk)}
    assert geoms["pillarL"].yaw_deg == -78.0 and geoms["main"].yaw_deg == 0.0
    _raises(KeyError, bl.rig_geometry, ["nope"], cfg, match="nope")
    # letterbox mask boundaries: centres are 4 + 8k, so x0 = 60 and x0 + nw = 124 sit on a centre
    g = bl.CamGeom("t", np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), (64, 48, 60, 44), 0.0)
    m = bl.pixel_valid_mask([g])[0]
    assert m.dtype == np.float32
    cols = np.flatnonzero(m.any(axis=0)).tolist()
    rows = np.flatnonzero(m.any(axis=1)).tolist()
    assert cols == list(range(7, 15)), cols  # u = 60 .. 116 kept, 124 dropped
    assert rows == list(range(5, 11)), rows  # v = 44 .. 84 kept, 92 dropped
    assert m.sum() == len(cols) * len(rows)

    class _FakeStem(torch.nn.Module):
        conv = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, 2, 1), torch.nn.ReLU())

    _raises(RuntimeError, bl.stem_taps, _FakeStem(), torch.zeros(1, 3, 16, 16), match="need 4")
    print("geometry_helper_edges")


def check_flat_index_edges() -> None:
    half = bl.BEV_HALF_M
    pts = torch.tensor([
        [-half, half - 0.01, 0.0],  # row 0, col 0: kept
        [half - 0.01, -half + 0.01, 0.0],  # row 127, col 127: kept
        [half, 0.0, 0.0],  # col 128: dropped
        [-half - 0.01, 0.0, 0.0],  # col -1: dropped
        [0.0, half + 0.01, 0.0],  # row -1: dropped
        [0.0, -half, 0.0],  # row 128: dropped
    ])
    idx, ok = bl.bev_flat_index(pts)
    assert ok.tolist() == [True, True, False, False, False, False], ok
    assert idx[0].item() == 0 and idx[1].item() == 128 * 128 - 1
    print("flat_index_edges")



def check_depth_already_hit_boundary() -> None:
    """A bin whose prior cumulative mass is exactly OCCLUDE_HIT_TAU is already occluded."""
    tau = bl.OCCLUDE_HIT_TAU
    d = 4
    p = torch.zeros(1, 1, d, 1, 1)
    p[0, 0, 0] = tau
    p[0, 0, 1] = 1.0 - tau
    blocked = bl.depth_already_hit(p, tau, dim=2)
    assert not bool(blocked[0, 0, 0]), "hit bin itself is not yet behind"
    assert bool(blocked[0, 0, 1]), "prior == tau must count as a confident hit (>=, not >)"
    assert bool(blocked[0, 0, 2]) and bool(blocked[0, 0, 3])
    q = torch.zeros(1, 1, d, 1, 1)
    just = min(tau - 1e-6, tau * 0.999)
    q[0, 0, 0] = just
    q[0, 0, 1] = 1.0 - just
    below = bl.depth_already_hit(q, tau, dim=2)
    assert not bool(below[0, 0, 0]) and not bool(below[0, 0, 1]), (float(just), tau)
    print("depth_already_hit_boundary")


def check_depth_survival_reference() -> None:
    torch.manual_seed(3)
    prob = torch.rand(2, 3, 7, 4, 5).softmax(dim=2)
    got = bl.depth_survival(prob, dim=2)
    ref = torch.zeros_like(prob)
    for i in range(7):
        ref[:, :, i] = prob[:, :, i + 1 :].sum(dim=2)
    assert torch.allclose(got, ref, atol=1e-6)
    assert torch.all(got >= 0) and torch.all(got[:, :, -1] == 0)
    flat = torch.tensor([[0.5, 0.3, 0.2]])  # default dim=1 (N x D)
    assert torch.allclose(bl.depth_survival(flat), torch.tensor([[0.5, 0.2, 0.0]]))
    print("depth_survival_reference")


class _ConstSpatial(torch.nn.Module):
    """Stub spatial path: one-hot depth in ``bin_index``, context = ``value`` on every channel."""

    def __init__(self, bin_index: int, value: float) -> None:
        super().__init__()
        self.bin_index, self.value = bin_index, value

    def forward(self, f4, f8, f16, cam_feat):
        n, _c, h, w = f8.shape
        prob = torch.zeros(n, bl.DEPTH_BINS, h, w, dtype=f8.dtype)
        prob[:, self.bin_index] = 1.0
        ctx = torch.full((n, bl.CTX_CH, h, w), self.value, dtype=f8.dtype)
        ctx[:, 1] = -self.value
        return prob, ctx


def check_lift_and_head_compensation() -> None:
    K, T = _simple_cam(pos=(0.0, 0.0, 1.0), fx=100.0)
    head = bl.BevLssHead()
    k = int(torch.argmin((head.depths - 10.0).abs()))
    head.spatial = _ConstSpatial(k, 2.0)
    only = torch.zeros(1, 24, 32)
    only[0, 12, 16] = 1.0
    f4, f8, f16 = (t[:, :1] for t in _taps())
    out = head(f4, f8, f16, K, T, pixel_valid=only)
    hit = out["hit_mass"][0, 0]
    feat = out["bev_feat"][0]
    assert tuple(out["bev_feat"].shape) == (1, bl.CTX_CH, 128, 128)
    assert torch.allclose(feat[0], 2.0 * hit) and torch.allclose(feat[1], -2.0 * hit)  # lift = depth x context
    row, col = (int(v) for v in torch.nonzero(hit)[0])
    moved = head(f4, f8, f16, K, T, pixel_valid=only, time_offset_s=torch.tensor([[0.05]]),
                 ego_speed_mps=torch.tensor([20.0]))["hit_mass"][0, 0]
    assert torch.nonzero(moved).tolist() == [[row - 2, col]], (row, col, torch.nonzero(moved).tolist())
    # per-batch pixel mask (B x N x h x w): batch 1 masks the only pixel
    taps2 = tuple(torch.cat([t, t]) for t in (f4, f8, f16))
    pv = torch.stack([only, torch.zeros_like(only)])
    out2 = head(*taps2, K.expand(2, -1, -1, -1), T.expand(2, -1, -1, -1), pixel_valid=pv)
    assert float(out2["hit_mass"][0].sum()) == 1.0 and float(out2["hit_mass"][1].sum()) == 0.0
    # float64 end to end (K, T, taps and weights in double): no silent float32 casts
    headd = bl.BevLssHead().double()
    outd = headd(*(t.double() for t in (f4, f8, f16)), K.double(), T.double())
    assert outd["logits"].dtype == torch.float64 and outd["hit_mass"].dtype == torch.float64
    ptsd = bl.ego_compensate(torch.zeros(1, 1, 2, 3, dtype=torch.float64), torch.tensor([[0.1]]), torch.tensor([3.0]))
    assert ptsd.dtype == torch.float64
    print("lift_and_head_compensation")


def check_camera_features_values() -> None:
    rig = _rig()
    feat = bl.BevLssHead.camera_features(rig["K"], rig["T"])[0]
    cfg = {c["id"]: c for c in _cfg()["cameras"]}
    for i, cid in enumerate(_cam_ids()):
        yaw = math.radians(cfg[cid]["yaw_deg"])
        fx = bl.camera_intrinsics(cfg[cid]["fov_v"], *cfg[cid]["live_res"])[0, 0]
        want = torch.tensor([fx / bl.FX_REF_PX, math.sin(yaw), math.cos(yaw), cfg[cid]["pos_m"][2]])
        assert torch.allclose(feat[i], want, atol=1e-4), (cid, feat[i], want)
    print("camera_features_values")


def check_architecture_pinned() -> None:
    """The design claims in the docstrings: dilated s16 context (2, 4), a stride-2 BEV U-Net, DW-sep blocks
    with no conv bias before GroupNorm and a ReLU at the end, and a depth buffer kept out of checkpoints."""
    head = bl.BevLssHead()
    dil = [blk.block[0].dilation[0] for blk in head.spatial.ctx16]
    assert dil == [2, 4], dil
    down = head.encoder.down
    assert down[0][0].stride == (2, 2) and [blk.block[0].dilation[0] for blk in down[1:]] == [1, 2]
    blk = bl._dws(8, 16)
    assert blk[0].dilation == (1, 1) and blk[0].stride == (1, 1) and blk[0].groups == 8
    assert isinstance(blk[-1], torch.nn.ReLU) and len(blk) == 5
    assert bl._Res(8).block[0].dilation == (1, 1)
    for m in head.modules():
        if isinstance(m, torch.nn.Sequential):
            for a, b in zip(m, list(m)[1:]):
                if isinstance(a, torch.nn.Conv2d) and isinstance(b, torch.nn.GroupNorm):
                    assert a.bias is None
    assert isinstance(head.encoder.inp[-1], torch.nn.ReLU)
    assert "depths" not in head.state_dict() and "depths" in dict(head.named_buffers())
    print("architecture_pinned")


def check_forward_reference() -> None:
    """SpatialPath, BevEncoder and _Res against an explicit composition of their own submodules.

    REGRESSION-ONLY: the reference is rebuilt from the modules' own layers (no captured numbers), so it pins
    the wiring, not any guarantee. No G1..G10 claim rests on this check alone; geometry is pinned to
    hand-computed points in check_geometry_hand_points.
    """
    import torch.nn.functional as F

    torch.manual_seed(4)
    sp = bl.SpatialPath().eval()
    with torch.no_grad():
        sp.film.weight.normal_(0, 0.5)
        sp.film.bias.normal_(0, 0.5)
        f4, f8, f16 = torch.randn(2, 64, 48, 64), torch.randn(2, 128, 24, 32), torch.randn(2, 128, 12, 16)
        cam = torch.randn(2, bl.CAM_FEAT)
        prob, ctx = sp(f4, f8, f16, cam)
        p16 = sp.lat16(sp.ctx16[1](sp.ctx16[0](f16)))
        p8 = sp.lat8(f8) + p16.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pooled = sp.lat4(f4)
        pooled = 0.25 * (pooled[..., 0::2, 0::2] + pooled[..., 1::2, 0::2] + pooled[..., 0::2, 1::2] + pooled[..., 1::2, 1::2])
        p8 = p8 + pooled
        gb = sp.film(cam)
        gamma, beta = gb[:, : bl.FPN_CH, None, None], gb[:, bl.FPN_CH :, None, None]
        p8 = sp.fuse8(p8 * (1.0 + gamma) + beta)
        assert torch.allclose(prob, sp.depth(p8).softmax(dim=1), atol=1e-5)
        assert torch.allclose(ctx, sp.ctx(p8), atol=1e-5)
        enc = bl.BevEncoder(6).eval()
        x = torch.randn(1, 6, 32, 32)
        a = enc.inp(x)
        b = F.interpolate(enc.down(a), size=(32, 32), mode="bilinear", align_corners=False)
        assert torch.allclose(enc(x), enc.head(enc.up(torch.cat([a, b], dim=1))), atol=1e-5)
        res = bl._Res(8).eval()
        y = torch.randn(1, 8, 6, 6)
        assert torch.allclose(res(y), y + res.block(y), atol=1e-6)
        assert not torch.allclose(res(y), y)  # the block contributes
    print("forward_reference")


# ---------------------------------------------------------------- hand-computed geometry (independent of the module)
def check_geometry_hand_points() -> None:
    """G2/G3/G4/G9 pinned to numbers written out by hand from pinhole math and the cameras.yaml mounts.

    None of the expected numbers below is read back from bev_lss; each comes from the derivation beside it.
    """
    cfg = {c["id"]: c for c in _cfg()["cameras"]}
    # the mounts the literals were derived from; if cameras.yaml changes, re-derive by hand
    assert cfg["main"]["pos_m"] == [0.00, 1.20, 1.26] and cfg["main"]["pitch_deg"] == -6 and cfg["main"]["fov_v"] == 34.0
    assert cfg["pillarL"]["pos_m"] == [-0.88, 0.40, 1.18] and cfg["pillarL"]["yaw_deg"] == -78
    assert cfg["pillarL"]["pitch_deg"] == -6 and cfg["pillarL"]["fov_v"] == 65.3
    assert cfg["rear"]["pos_m"] == [0.00, -2.25, 0.78] and cfg["rear"]["yaw_deg"] == 180 and cfg["rear"]["pitch_deg"] == -10
    assert cfg["rear"]["fov_v"] == 80.0 and cfg["rear"]["live_res"] == [640, 360]

    # --- intrinsics. 640x480 -> scale min(256/640, 192/480) = 0.4 -> 256x192 at (0, 0): cx = 128, cy = 96.
    # 640x360 -> scale 0.4 -> 256x144, y0 = (192 - 144) / 2 = 24: cx = 128, cy = 24 + 72 = 96.
    # fy = (content_h / 2) / tan(fov_v / 2), fx = fy.
    hand_f = {
        ("main", 34.0, 640, 480): 314.00185,  # 96 / tan(17.00 deg) = 96 / 0.3057307
        ("narrow", 21.4, 640, 480): 508.06564,  # 96 / tan(10.70 deg) = 96 / 0.1889522
        ("pillarL", 65.3, 640, 480): 149.82280,  # 96 / tan(32.65 deg) = 96 / 0.6407568
        ("repeatL", 55.4, 640, 480): 182.85305,  # 96 / tan(27.70 deg) = 96 / 0.5250052
        ("wide", 88.7, 640, 360): 73.65245,  # 72 / tan(44.35 deg) = 72 / 0.9775548
        ("rear", 80.0, 640, 360): 85.80626,  # 72 / tan(40.00 deg) = 72 / 0.8390996
    }
    for (cid, fov, w, h), f in hand_f.items():
        K = bl.camera_intrinsics(fov, w, h)
        assert np.allclose(K, [[f, 0.0, 128.0], [0.0, f, 96.0], [0.0, 0.0, 1.0]], atol=1e-3), (cid, K, f)

    # --- extrinsics. Columns are the camera axes (right, down, forward) in ego coords:
    # fwd = (sin y cos p, cos y cos p, sin p), right = (cos y, -sin y, 0), down = fwd x right.
    # main, yaw 0, pitch -6: sin 6 = 0.1045285, cos 6 = 0.9945219
    #   right = (1, 0, 0); fwd = (0, 0.9945219, -0.1045285); down = (0, -0.1045285, -0.9945219)
    Tm = bl.cam_to_ego([0.00, 1.20, 1.26], 0.0, -6.0)
    assert np.allclose(Tm, [[1.0, 0.0, 0.0, 0.0],
                            [0.0, -0.1045285, 0.9945219, 1.20],
                            [0.0, -0.9945219, -0.1045285, 1.26],
                            [0.0, 0.0, 0.0, 1.0]], atol=1e-6), Tm
    # pillarL, yaw -78, pitch -6: sin 78 = 0.9781476, cos 78 = 0.2079117
    #   right = (0.2079117, 0.9781476, 0)
    #   fwd = (-0.9781476 * 0.9945219, 0.2079117 * 0.9945219, -0.1045285) = (-0.9727892, 0.2067727, -0.1045285)
    #   down = fwd x right = (0.1045285 * 0.9781476, -0.1045285 * 0.2079117, -0.9727892 * 0.9781476 - 0.2067727 * 0.2079117)
    #        = (0.1022443, -0.0217327, -0.9945219)
    Tp = bl.cam_to_ego([-0.88, 0.40, 1.18], -78.0, -6.0)
    assert Tp.dtype == np.float32 and Tp.shape == (4, 4)
    assert np.allclose(Tp, [[0.2079117, 0.1022443, -0.9727892, -0.88],
                            [0.9781476, -0.0217327, 0.2067727, 0.40],
                            [0.0, -0.9945219, -0.1045285, 1.18],
                            [0.0, 0.0, 0.0, 1.0]], atol=1e-6), Tp
    # rear, yaw 180, pitch -10: sin 10 = 0.1736482, cos 10 = 0.9848078
    #   right = (-1, 0, 0); fwd = (0, -0.9848078, -0.1736482); down = (0, 0.1736482, -0.9848078)
    Tr = bl.cam_to_ego([0.00, -2.25, 0.78], 180.0, -10.0)
    assert np.allclose(Tr, [[-1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.1736482, -0.9848078, -2.25],
                            [0.0, -0.9848078, -0.1736482, 0.78],
                            [0.0, 0.0, 0.0, 1.0]], atol=1e-6), Tr

    # --- frustum points. Feature pixel (row 12, col 16) at stride 8: u = 16*8 + 4 = 132, v = 12*8 + 4 = 100.
    # Camera point at depth 10: (xc, yc, zc) = ((132 - 128) / f * 10, (100 - 96) / f * 10, 10). Ego = R p + t.
    # main, f = 314.00185: xc = yc = 0.1273878
    #   x = 0.1273878; y = 1.20 - 0.1045285 * 0.1273878 + 0.9945219 * 10 = 11.1319033
    #   z = 1.26 - 0.9945219 * 0.1273878 - 0.1045285 * 10 = 0.0880254
    #   cell: row = floor((32 - 11.1319) / 0.5) = 41, col = floor((0.1274 + 32) / 0.5) = 64
    # pillarL, f = 149.82280: xc = yc = 0.2669821
    #   x = -0.88 + 0.2079117 * 0.2669821 + 0.1022443 * 0.2669821 - 0.9727892 * 10 = -10.5250860
    #   y = 0.40 + 0.9781476 * 0.2669821 - 0.0217327 * 0.2669821 + 0.2067727 * 10 = 2.7230729
    #   z = 1.18 - 0.9945219 * 0.2669821 - 0.1045285 * 10 = -0.1308041
    #   cell: row = floor((32 - 2.7231) / 0.5) = 58, col = floor((-10.5251 + 32) / 0.5) = 42
    # rear, f = 85.80626: xc = yc = 0.4661665
    #   x = -0.4661665; y = -2.25 + 0.1736482 * 0.4661665 - 0.9848078 * 10 = -12.0171286
    #   z = 0.78 - 0.9848078 * 0.4661665 - 0.1736482 * 10 = -1.4155661
    #   cell: row = floor((32 + 12.0171) / 0.5) = 88, col = floor((32 - 0.4662) / 0.5) = 63
    hand_pts = {
        "main": ((314.00185, 96.0), Tm, (0.1273878, 11.1319033, 0.0880254), (41, 64)),
        "pillarL": ((149.82280, 96.0), Tp, (-10.5250860, 2.7230729, -0.1308041), (58, 42)),
        "rear": ((85.80626, 96.0), Tr, (-0.4661665, -12.0171286, -1.4155661), (88, 63)),
    }
    for cid, ((f, cy), T_np, want, cell) in hand_pts.items():
        K = torch.tensor([[f, 0.0, 128.0], [0.0, f, cy], [0.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        T = torch.from_numpy(T_np).view(1, 1, 4, 4)
        p = bl.frustum_ego_points(K, T, torch.tensor([10.0]), (24, 32))
        assert p.shape == (1, 1, 1, 24, 32, 3)
        assert torch.allclose(p[0, 0, 0, 12, 16], torch.tensor(want), atol=1e-4), (cid, p[0, 0, 0, 12, 16], want)
        assert bl.bev_cell(want[0], want[1]) == cell, cid
        idx, ok = bl.bev_flat_index(p[0, 0, 0, 12, 16])
        assert bool(ok) and int(idx) == cell[0] * 128 + cell[1], (cid, int(idx))
    # the image corner pixel (row 0, col 0): u = v = 4, simple camera f = 100 at (0, 0, 1), yaw 0, pitch 0,
    # depth 2: cam = ((4 - 128) / 100 * 2, (4 - 96) / 100 * 2, 2) = (-2.48, -1.84, 2) -> ego (-2.48, 2, 1 + 1.84)
    # and the last pixel (row 23, col 31): u = 252, v = 188 -> cam (2.48, 1.84, 2) -> ego (2.48, 2, 1 - 1.84)
    K1 = torch.tensor([[100.0, 0.0, 128.0], [0.0, 100.0, 96.0], [0.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    T1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]).view(1, 1, 4, 4)
    p1 = bl.frustum_ego_points(K1, T1, torch.tensor([2.0]), (24, 32))[0, 0, 0]
    assert torch.allclose(p1[0, 0], torch.tensor([-2.48, 2.0, 2.84]), atol=1e-5), p1[0, 0]
    assert torch.allclose(p1[23, 31], torch.tensor([2.48, 2.0, -0.84]), atol=1e-5), p1[23, 31]
    assert torch.allclose(p1[0, 31], torch.tensor([2.48, 2.0, 2.84]), atol=1e-5), p1[0, 31]  # u varies along columns
    assert torch.allclose(p1[23, 0], torch.tensor([-2.48, 2.0, -0.84]), atol=1e-5), p1[23, 0]  # v varies along rows

    # --- splat into hand-picked cells. row = floor((32 - y) / 0.5), col = floor((x + 32) / 0.5).
    #   (0.25, 0.25, 0): row floor(63.5) = 63, col floor(64.5) = 64, twice (1 + 2 = 3)
    #   (-31.75, 31.75, 0): row 0, col 0 (5); (31.75, -31.75, 3.9): row 127, col 127 (7)
    #   (0, 0, 4.0): z = 4 is outside [-2, 4), dropped; (32.0, 0, 0): col 128, dropped
    pts = torch.tensor([[[0.25, 0.25, 0.0], [0.25, 0.25, 0.0], [-31.75, 31.75, 0.0], [31.75, -31.75, 3.9],
                         [0.0, 0.0, 4.0], [32.0, 0.0, 0.0]]])
    vals = torch.tensor([[[1.0], [2.0], [5.0], [7.0], [100.0], [1000.0]]])
    idx, ok = bl.bev_flat_index(pts)
    assert ok.tolist() == [[True, True, True, True, False, False]]
    assert idx[0, :4].tolist() == [63 * 128 + 64, 63 * 128 + 64, 0, 127 * 128 + 127]
    grid = bl.splat(vals, idx, ok)[0, 0]
    assert float(grid[63, 64]) == 3.0 and float(grid[0, 0]) == 5.0 and float(grid[127, 127]) == 7.0
    assert float(grid.sum()) == 15.0  # 100 and 1000 went nowhere

    # --- ego compensation (G9): v = 12.5 m/s, dt = +0.04 s -> +0.5 m along +y (one cell forward = row - 1)
    moved = bl.ego_compensate(torch.tensor([[[1.0, 2.0, 0.5]]]), torch.tensor([[0.04]]), torch.tensor([12.5]))
    assert torch.allclose(moved, torch.tensor([[[1.0, 2.5, 0.5]]]), atol=1e-6), moved
    assert bl.bev_cell(1.0, 2.5) == (59, 66) and bl.bev_cell(1.0, 2.0) == (60, 66)
    print("geometry_hand_points")


def check_device_follows_inputs() -> None:
    """Every tensor the head creates lives on the inputs' device (no hidden CPU tensors in the graph).

    Run on PyTorch's ``meta`` device (shapes only, no data, no GPU): a CPU tensor mixed into a meta
    computation raises, so this pins device propagation on a CPU-only host.
    """
    meta = torch.device("meta")
    K, T = _simple_cam()
    p = bl.frustum_ego_points(K.to(meta), T.to(meta), torch.tensor([5.0, 20.0], device=meta), (24, 32))
    assert p.device.type == "meta" and tuple(p.shape) == (1, 1, 2, 24, 32, 3)
    vals = torch.empty(2, 10, 3, device=meta)
    grid = bl.splat(vals, torch.zeros(2, 10, dtype=torch.long, device=meta), torch.ones(2, 10, dtype=torch.bool, device=meta))
    assert grid.device.type == "meta" and tuple(grid.shape) == (2, 3, 128, 128)
    head = bl.BevLssHead().to(meta)
    rig = _rig()
    taps = (torch.empty(1, N_CAMS, 64, 48, 64, device=meta), torch.empty(1, N_CAMS, 128, 24, 32, device=meta),
            torch.empty(1, N_CAMS, 128, 12, 16, device=meta))
    for kw in ({}, {"cam_valid": torch.ones(1, N_CAMS, device=meta), "pixel_valid": rig["pixel_valid"].to(meta)}):
        out = head(*taps, rig["K"].to(meta), rig["T"].to(meta), **kw)
        assert all(v.device.type == "meta" for v in out.values())
        assert tuple(out["logits"].shape) == (1, len(bl.BEV_CHANNELS), 128, 128)
    print("device_follows_inputs")


def check_dtype_follows_model() -> None:
    """Output dtype is the model's dtype, whatever dtype the masks, speeds or torch's global default have."""
    # ego_compensate: float32 points, float64 speed / offset -> float32 points, same numbers
    # (0, 10, 0) with v = 10 m/s, dt = 0.1 s -> (0, 11, 0)
    pts = torch.tensor([[[[0.0, 10.0, 0.0]]]])
    out = bl.ego_compensate(pts, torch.tensor([[0.1]], dtype=torch.float64), torch.tensor([10.0], dtype=torch.float64))
    assert out.dtype == torch.float32, out.dtype
    assert torch.allclose(out, torch.tensor([[[[0.0, 11.0, 0.0]]]]), atol=1e-6)
    out = bl.ego_compensate(pts, torch.tensor([[0.1]]), torch.tensor([10.0], dtype=torch.float64))
    assert out.dtype == torch.float32, out.dtype
    out = bl.ego_compensate(pts, torch.tensor([[0.1]], dtype=torch.float64), torch.tensor([10.0]))
    assert out.dtype == torch.float32, out.dtype
    # head: float32 model, masks in float64 / bool (numpy default and boolean masks are common)
    torch.manual_seed(7)
    head = bl.BevLssHead().eval()
    rig = _rig()
    f4, f8, f16 = _taps()
    with torch.no_grad():
        ref = head(f4, f8, f16, rig["K"], rig["T"], cam_valid=torch.ones(1, N_CAMS), pixel_valid=rig["pixel_valid"])
        for cv, pv in ((torch.ones(1, N_CAMS, dtype=torch.float64), rig["pixel_valid"]),
                       (torch.ones(1, N_CAMS), rig["pixel_valid"].double()),
                       (torch.ones(1, N_CAMS, dtype=torch.bool), rig["pixel_valid"].bool())):
            got = head(f4, f8, f16, rig["K"], rig["T"], cam_valid=cv, pixel_valid=pv)
            assert got["logits"].dtype == torch.float32 and got["hit_mass"].dtype == torch.float32
            assert torch.equal(got["logits"], ref["logits"]) and torch.equal(got["hit_mass"], ref["hit_mass"])
        # global default dtype float64 (some training scripts set it): the float32 head still runs in float32
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            got = head(f4, f8, f16, rig["K"], rig["T"])
        finally:
            torch.set_default_dtype(old)
        assert got["logits"].dtype == torch.float32
        plain = head(f4, f8, f16, rig["K"], rig["T"])
        assert torch.equal(got["logits"], plain["logits"])
    # the lift geometry runs in K's dtype: a float64 depth buffer must not upcast the float32 frustum
    K, T = _simple_cam(pos=(0.0, 0.0, 1.0), fx=100.0)
    headg = bl.BevLssHead()
    headg.spatial = _ConstSpatial(int(torch.argmin((headg.depths - 10.0).abs())), 1.0)
    headg.depths = headg.depths.double()
    seen = []
    orig = bl.bev_flat_index

    def _spy(points):
        seen.append(points.dtype)
        return orig(points)

    bl.bev_flat_index = _spy
    try:
        headg(*(t[:, :1] for t in _taps()), K, T)
    finally:
        bl.bev_flat_index = orig
    assert seen == [torch.float32], seen
    print("dtype_follows_model")


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
    check_pass_fraction_gate()
    check_wall_behind_is_unknown_not_free()
    check_pass_mass_is_survival()
    check_zero_valid_cameras_no_nan()
    check_gradients_reach_stem_and_depth()
    check_deterministic()
    check_param_count()
    check_activation_memory_estimate()
    check_onnx_export_parity()
    check_default_off()
    check_rig_geometry_applies_vehicle_overrides()
    check_geometry_helper_edges()
    check_flat_index_edges()
    check_depth_survival_reference()
    check_depth_already_hit_boundary()
    check_lift_and_head_compensation()
    check_camera_features_values()
    check_architecture_pinned()
    check_forward_reference()
    check_geometry_hand_points()
    check_device_follows_inputs()
    check_dtype_follows_model()
    print("test_bev_lss: OK")


if __name__ == "__main__":
    main()
