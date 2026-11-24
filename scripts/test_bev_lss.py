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
    logits = out["logits"][0]
    clear, through = out["clear_mass"][0, 0], out["through_mass"][0, 0]
    seen = clear > 0  # a valid ray reached the cell (transmittance >= REACH_TAU)
    # untrained head: depth near uniform over the log bins, so a ray more likely than not reaches only
    # about its median depth (~7 m). Near cells are seen, 15 m is not (G5: unknown, never free).
    assert 0.0 < float(seen.float().mean()) < 1.0, float(seen.float().mean())
    assert bool(seen[64 - 8, 64]) and not bool(seen[64 - 30, 64])
    assert torch.all(logits[bl.CH_UNKNOWN][~seen] == bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_FREE][~seen] == -bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_FREE][through <= 0] == -bl.FORCE_LOGIT)
    assert torch.all(logits[bl.CH_UNKNOWN][seen].abs() < bl.FORCE_LOGIT)  # learned where seen
    assert torch.all(logits[bl.CH_FREE][(through > 0) & seen].abs() < bl.FORCE_LOGIT)
    # a head 95% sure of the true surface of an empty flat road (ray-cast) on all 8 cameras sees a good part
    # of the grid, 15 m ahead included. (Was a head sure of bin 35 on every pixel; with the band measured
    # from each ray's surface that fixture puts the "ground" 32 m out along every ray, under the road.)
    far, _ = _scene_run(_cam_ids(), [])
    seen_far = far["clear_mass"][0, 0] > 0
    assert 0.2 < float(seen_far.float().mean()) < 1.0, float(seen_far.float().mean())
    assert bool(seen_far[64 - 30, 64])  # 15 m ahead is seen
    assert torch.all(far["logits"][0, bl.CH_UNKNOWN][~seen_far] == bl.FORCE_LOGIT)
    assert torch.all(far["logits"][0, bl.CH_UNKNOWN][seen_far].abs() < bl.FORCE_LOGIT)
    # apply_visibility in isolation: other channels untouched
    raw = torch.randn(1, 14, 4, 4)
    c = torch.zeros(1, 1, 4, 4)
    t = torch.zeros(1, 1, 4, 4)
    c[..., 0, 0] = 1.0  # reached, nothing passed through (the hit cell)
    c[..., 1, 1] = 1.0
    t[..., 1, 1] = 0.7  # reached and passed through
    got = bl.apply_visibility(raw, c, t)
    keep = [i for i in range(14) if i not in (bl.CH_FREE, bl.CH_UNKNOWN)]
    assert torch.equal(got[:, keep], raw[:, keep])
    assert got[0, bl.CH_UNKNOWN, 0, 0] == raw[0, bl.CH_UNKNOWN, 0, 0] and got[0, bl.CH_UNKNOWN, 1, 1] == raw[0, bl.CH_UNKNOWN, 1, 1]
    assert got[0, bl.CH_UNKNOWN, 2, 2] == bl.FORCE_LOGIT
    assert got[0, bl.CH_FREE, 0, 0] == -bl.FORCE_LOGIT  # reached, nothing passed: not free
    assert got[0, bl.CH_FREE, 1, 1] == raw[0, bl.CH_FREE, 1, 1]
    # through without reach (possible when FREE_PASS_TAU < REACH_TAU) is still not free
    t2 = torch.zeros(1, 1, 4, 4)
    t2[..., 2, 2] = 0.4
    got2 = bl.apply_visibility(raw, c, t2)
    assert got2[0, bl.CH_FREE, 2, 2] == -bl.FORCE_LOGIT and got2[0, bl.CH_UNKNOWN, 2, 2] == bl.FORCE_LOGIT
    print("visibility_by_construction")


def check_visibility_taus_pinned() -> None:
    """Both visibility cutoffs are the majority decision (0.5): no cutoff is tuned to a wall fixture."""
    assert bl.REACH_TAU == 0.5, bl.REACH_TAU
    assert bl.FREE_PASS_TAU == 0.5, bl.FREE_PASS_TAU
    head = bl.BevLssHead()
    assert head.reach_tau == 0.5 and head.free_pass_tau == 0.5
    for bad in (0.0, -0.1, 1.5, float("nan")):
        _raises(ValueError, lambda: bl.BevLssHead(reach_tau=bad), match="reach_tau")
        _raises(ValueError, lambda: bl.BevLssHead(free_pass_tau=bad), match="free_pass_tau")
    bl.BevLssHead(reach_tau=1.0, free_pass_tau=1.0)  # 1.0 is a legal (strictest) cutoff
    print("visibility_taus_pinned")


def check_transmittance_boundary() -> None:
    """Transmittance exactly equal to the cutoff counts (>=, not >), for reach and for pass-through."""
    tau_r, tau_f = bl.REACH_TAU, bl.FREE_PASS_TAU
    p = torch.tensor([0.5, 0.5, 0.0, 0.0]).view(1, 1, 4, 1, 1)  # exact in float32
    reach = bl.depth_transmittance(p, dim=2).view(-1)
    assert torch.equal(reach, torch.tensor([1.0, 0.5, 0.0, 0.0])), reach
    assert torch.equal(bl.depth_survival(p, dim=2).view(-1), torch.tensor([0.5, 0.0, 0.0, 0.0]))
    # cell 0: reached exactly at tau and passed exactly at tau; cell 1: just below both; cell 2: reached, pass just below
    reach_v = torch.tensor([tau_r, tau_r * (1 - 1e-6), tau_r]).view(1, 1, 3, 1, 1)
    pass_v = torch.tensor([tau_f, tau_f * (1 - 1e-6), tau_f * (1 - 1e-6)]).view(1, 1, 3, 1, 1)
    reached, through = bl.visibility_gates(reach_v, pass_v, tau_r, tau_f)
    assert reached.view(-1).tolist() == [1.0, 0.0, 1.0], reached.view(-1)
    assert through.view(-1).tolist() == [1.0, 0.0, 0.0], through.view(-1)
    print("transmittance_boundary")


class _FixedDepth(torch.nn.Module):
    """Spatial stand-in: depth distribution ``p`` (D, or D x h x w) at every pixel, or a list of them, one per
    camera; unit context."""

    def __init__(self, p) -> None:
        super().__init__()
        self.ps = list(p) if isinstance(p, (list, tuple)) else [p]

    def forward(self, f4, f8, f16, cam):
        n, _, h, w = f8.shape
        per = [p.view(-1, 1, 1).expand(-1, h, w) if p.dim() == 1 else p for p in self.ps]
        return torch.stack([per[i % len(per)] for i in range(n)]).contiguous(), torch.ones(n, bl.CTX_CH, h, w)


def _peaked(k: int, p_hit: float) -> torch.Tensor:
    """p_hit at bin k, the rest spread evenly over the other bins (softmax with one raised logit)."""
    m = math.log((bl.DEPTH_BINS - 1) * p_hit / (1.0 - p_hit))
    lg = torch.zeros(bl.DEPTH_BINS)
    lg[k] = m
    return lg.softmax(0)


def _bimodal(kw: int, p_wall: float, kf: int, eps: float = 1e-6) -> torch.Tensor:
    p = torch.full((bl.DEPTH_BINS,), eps)
    p[kw] = p_wall
    p[kf] = 1.0 - p_wall
    return p / p.sum()


def _cells(rig: dict, head, lo: int, hi: int, out: dict) -> torch.Tensor:
    """Bool 128x128: cells some valid in-band visibility sample governed by bins lo..hi-1 lands in.

    Samples are the bin centres (centre i: reach T_i) and the edges between bins (edge j: reach S_{j+1}
    = T_{j+2}, so edge j counts as bin j + 2). Only samples of rays with a surface, less than the band
    above the local ground height the head reported (``out``), in a cell that has one, count.
    """
    m = torch.zeros(128 * 128, dtype=torch.bool)
    pv = rig["pixel_valid"][0].bool()[None]
    g = out["ground_height"][0, 0].reshape(-1)
    has_g = out["has_ground"][0, 0].reshape(-1) > 0
    reach = bl.depth_transmittance(out["depth_prob"], dim=2)
    reached, _ = bl.visibility_gates(reach, reach, head.reach_tau, head.free_pass_tau)
    _, has_surf = bl.ray_surface(reached, dim=2)
    has_surf = has_surf[0, 0] > 0  # 1 x h x w
    for depths, shift in ((head.depths, 0), (head.edges, 2)):
        pts = bl.frustum_ego_points(rig["K"], rig["T"], depths, (24, 32))[0, 0]
        idx, ok = bl.bev_flat_index(pts)
        ok = ok & pv & has_surf & has_g[idx] & ((pts[..., 2] - g[idx]) < head.ground_band_m)
        a, b = max(lo - shift, 0), max(hi - shift, 0)
        m[idx[a:b][ok[a:b]]] = True
    return m.view(128, 128)


def _run_fixed(cam: str, p: torch.Tensor, head=None) -> tuple[dict, dict, torch.nn.Module]:
    rig = bl.rig_tensors(bl.rig_geometry([cam], _cfg(), model=BASE_RIG_MODEL))
    head = (head or bl.BevLssHead()).eval()
    head.spatial = _FixedDepth(p)
    taps = (torch.zeros(1, 1, 64, 48, 64), torch.zeros(1, 1, 128, 24, 32), torch.zeros(1, 1, 128, 12, 16))
    with torch.no_grad():
        out = head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    return out, rig, head


def _forced(lg: torch.Tensor, region: torch.Tensor) -> tuple[int, int, int]:
    """(free forced low, unknown forced high, occluded forced high) counts inside ``region``."""
    return (
        int((region & (lg[bl.CH_FREE] == -bl.FORCE_LOGIT)).sum()),
        int((region & (lg[bl.CH_UNKNOWN] == bl.FORCE_LOGIT)).sum()),
        int((region & (lg[bl.CH_OCCLUDED] == bl.FORCE_LOGIT)).sum()),
    )


def check_wall_behind_is_unknown_not_free() -> None:
    """G5: every cell only reached behind a wall the head is >= 50% sure of is unknown + occluded, never free.

    Walls on main (bins 12 and 24), pillarL (bin 16) and rear (bin 20), with the rest of the depth
    mass spread evenly (p_hit 0.6, 0.8, 0.94, 0.987, 0.999999). Cells in front of the wall stay learned.
    """
    # near-ground centre + edge samples: 54 with two-scale ground (the fixture's only surfaces
    # are the wall's, so cells more than GROUND_WINDOW_CELLS // 2 from it have no ground and stay plain
    # unknown; the wide-scale window trims six fringe cells versus the v5 60); 908 with the v4
    # ego-frame band, 776 with centres only over the whole z band
    expect_behind = {("main", 12): 54}
    for cam, k in (("main", 12), ("main", 24), ("pillarL", 16), ("rear", 20)):
        for p_hit in (0.6, 0.8, 0.94, 0.987, 0.999999):
            out, rig, head = _run_fixed(cam, _peaked(k, p_hit))
            lg = out["logits"][0]
            front = _cells(rig, head, 0, k + 1, out)
            behind = _cells(rig, head, k + 1, bl.DEPTH_BINS, out) & ~front
            in_front = _cells(rig, head, 0, k, out) & ~behind
            n = int(behind.sum())
            assert n > 0, (cam, k)
            if (cam, k) in expect_behind:
                assert n == expect_behind[(cam, k)], (cam, k, n)
            got = _forced(lg, behind)
            assert got == (n, n, n), f"{cam} bin {k} p_hit {p_hit}: forced (free, unknown, occluded) {got} of {n}"
            assert int(in_front.sum()) > 0
            assert torch.all(lg[bl.CH_UNKNOWN][in_front].abs() < bl.FORCE_LOGIT), (cam, k, p_hit)
            assert torch.all(lg[bl.CH_OCCLUDED][in_front].abs() < bl.FORCE_LOGIT), (cam, k, p_hit)
            assert not torch.all(lg[bl.CH_FREE][in_front] == -bl.FORCE_LOGIT), (cam, k, p_hit)
    print("wall_behind_is_unknown_not_free")


def check_bimodal_depth_occludes_between_modes() -> None:
    """Two-guess depth: wall p at 3.4 m (bin 12), 1 - p at bin 30. Between the modes the ray got through
    with probability 1 - p only, so for p > 0.5 those cells are unknown + occluded, never free; for
    p < 0.5 they are seen and may be free; p == 0.5 is seen and may be free (>=). main, pillarL, rear.
    """
    kw, kf = 12, 30
    for cam in ("main", "pillarL", "rear"):
        for p_wall in (0.6, 0.8, 0.94):
            out, rig, head = _run_fixed(cam, _bimodal(kw, p_wall, kf))
            lg = out["logits"][0]
            mid = _cells(rig, head, kw + 1, kf, out) & ~_cells(rig, head, 0, kw + 1, out)
            n = int(mid.sum())
            assert n > 0, cam
            got = _forced(lg, mid)
            assert got == (n, n, n), f"{cam} p_wall {p_wall}: forced (free, unknown, occluded) {got} of {n}"
        for p_wall, eps in ((0.4, 1e-6), (0.5, 0.0)):
            out, rig, head = _run_fixed(cam, _bimodal(kw, p_wall, kf, eps))
            lg = out["logits"][0]
            mid = _cells(rig, head, kw + 1, kf, out) & ~_cells(rig, head, 0, kw + 1, out)
            n = int(mid.sum())
            got = _forced(lg, mid)
            assert got == (0, 0, 0), f"{cam} p_wall {p_wall}: see-through cells forced {got} of {n}"
    print("bimodal_depth_occludes_between_modes")


def _hand_masses(head, band: float) -> None:
    """One pixel's clear / through / occluded masses equal a hand sum over its near-ground samples.

    Level camera at 3.0 m, pixel row 14 (v = 116, cy = 96): height z = 3.0 - 0.2 d. Depth mass 0.25 /
    0.375 / 0.375 at bins 24 / 26 / 34, so the ray's surface (last reached bin) is bin 26 and the only
    ground height is that point's z, over the 13 x 13 cells around its cell. Centre i counts reach T_i /
    pass S_i; the edge between bins j and j + 1 counts S_{j+1} for both. Samples outside that window or
    band (or more) above the surface skip.
    """
    K, T = _simple_cam(pos=(0.0, 0.0, 3.0), fx=100.0)
    p = torch.zeros(bl.DEPTH_BINS)
    p[24], p[26], p[34] = 0.25, 0.375, 0.375  # exact in float32; bin 24 (S = 0.75 < T = 1) in the window
    head.spatial = _FixedDepth(p)
    only = torch.zeros(1, 24, 32)
    only[0, 14, 16] = 1.0
    with torch.no_grad():
        out = head(*(t[:, :1] for t in _taps()), K, T, pixel_valid=only)

    def cell(d):
        pt = bl.frustum_ego_points(K, T, torch.tensor([float(d)], dtype=torch.float32), (24, 32))[0, 0, 0, 14, 16]
        idx, ok = bl.bev_flat_index(pt[None])
        return float(pt[2]), int(idx[0]) // 128, int(idx[0]) % 128, bool(ok[0])

    centres = [float(c) for c in bl.depth_bin_centers()]
    z_s, r_s, c_s, ok_s = cell(centres[26])
    assert ok_s and abs(z_s - (3.0 - 0.2 * centres[26])) < 1e-4
    half = bl.GROUND_WINDOW_CELLS // 2
    assert abs(float(out["ground_height"][0, 0, r_s, c_s]) - z_s) < 1e-4
    assert int(out["has_ground"].sum()) == bl.GROUND_WINDOW_CELLS ** 2
    edges = list(np.geomspace(bl.DEPTH_MIN_M, bl.DEPTH_MAX_M, bl.DEPTH_BINS + 1)[1:-1])
    samples = [(c, float(p[i:].sum()), float(p[i + 1 :].sum()), "c") for i, c in enumerate(centres)]
    samples += [(e, float(p[j + 2 :].sum()), float(p[j + 2 :].sum()), "e") for j, e in enumerate(edges)]
    clear = through = occ = 0.0
    above = set()
    for d, reach, passed, kind in samples:
        z, r, c, ok = cell(d)
        if not ok or abs(r - r_s) > half or abs(c - c_s) > half:
            continue
        if z - z_s >= band:
            above.add(kind)
            continue
        clear += reach if reach >= 0.5 else 0.0
        through += passed if passed >= 0.5 else 0.0
        occ += (1.0 - reach) if reach < 0.5 else 0.0
    assert clear > 0 and through > 0 and occ > 0
    assert above == ({"e"} if band <= 0.6 else set()), (band, above)  # the 0.5 m band drops an edge sample
    assert abs(float(out["clear_mass"].sum()) - clear) < 1e-5, (band, float(out["clear_mass"].sum()), clear)
    assert abs(float(out["through_mass"].sum()) - through) < 1e-5, (band, float(out["through_mass"].sum()), through)
    assert abs(float(out["occluded_mass"].sum()) - occ) < 1e-5, (band, float(out["occluded_mass"].sum()), occ)


def check_visibility_masses_values() -> None:
    """Hand-summed masses at the default 0.5 m band (near samples 0.5 m or more above the surface skip) and
    at a 1.0 m band passed through the config (every sample in the window counts, centres and edges alike)."""
    _hand_masses(bl.BevLssHead(), 0.5)
    _hand_masses(bl.build_bev_head({"bev_lss": {"enabled": True, "ground_band_m": 1.0}}), 1.0)
    print("visibility_masses_values")


def check_occlusion_branches() -> None:
    """Occluded is forced only where valid rays landed and none reached: not where another ray saw the
    cell, not from an invalid camera, not where no ray landed at all."""
    raw = torch.randn(1, 14, 1, 3)
    clear = torch.tensor([0.0, 0.7, 0.0]).view(1, 1, 1, 3)
    through = torch.zeros(1, 1, 1, 3)
    occ = torch.tensor([0.4, 0.4, 0.0]).view(1, 1, 1, 3)  # cell 0 behind, cell 1 behind for one ray but seen by another, cell 2 no ray
    got = bl.apply_visibility(raw, clear, through, occluded_mass=occ)
    assert got[0, bl.CH_OCCLUDED, 0, 0] == bl.FORCE_LOGIT
    assert got[0, bl.CH_OCCLUDED, 0, 1] == raw[0, bl.CH_OCCLUDED, 0, 1], "seen by another ray: occluded stays learned"
    assert got[0, bl.CH_UNKNOWN, 0, 1] == raw[0, bl.CH_UNKNOWN, 0, 1]
    assert got[0, bl.CH_OCCLUDED, 0, 2] == raw[0, bl.CH_OCCLUDED, 0, 2], "no ray landed: occluded stays learned"
    assert got[0, bl.CH_UNKNOWN, 0, 2] == bl.FORCE_LOGIT
    # head level: cells outside every frustum keep a learned occluded logit
    out, rig, head = _run_fixed("main", _peaked(12, 0.94))
    lg = out["logits"][0]
    landed = _cells(rig, head, 0, bl.DEPTH_BINS, out)
    assert int((~landed).sum()) > 0
    assert torch.all(lg[bl.CH_OCCLUDED][~landed].abs() < bl.FORCE_LOGIT)
    # head level: main has a 94% wall at 3.4 m; the forward "wide" camera is sure of a far surface (bin 35)
    # and reaches some of the cells behind main's wall. Those cells are seen, not occluded.
    rig2 = bl.rig_tensors(bl.rig_geometry(["main", "wide"], _cfg(), model=BASE_RIG_MODEL))
    head2 = bl.BevLssHead().eval()
    head2.spatial = _FixedDepth([_peaked(12, 0.94), _peaked(35, 0.987)])
    taps2 = (torch.zeros(1, 2, 64, 48, 64), torch.zeros(1, 2, 128, 24, 32), torch.zeros(1, 2, 128, 12, 16))
    with torch.no_grad():
        both = head2(*taps2, rig2["K"], rig2["T"], pixel_valid=rig2["pixel_valid"])
        # wide invalid: identical masses and forcing to main alone
        only_main = head2(*taps2, rig2["K"], rig2["T"], cam_valid=torch.tensor([[1.0, 0.0]]), pixel_valid=rig2["pixel_valid"])
    for name in ("clear_mass", "through_mass", "occluded_mass", "hit_mass", "pass_mass"):
        assert torch.equal(only_main[name], out[name]), name
    for ch in (bl.CH_FREE, bl.CH_UNKNOWN, bl.CH_OCCLUDED):
        assert torch.equal(only_main["logits"][0, ch] == bl.FORCE_LOGIT, lg[ch] == bl.FORCE_LOGIT), ch
        assert torch.equal(only_main["logits"][0, ch] == -bl.FORCE_LOGIT, lg[ch] == -bl.FORCE_LOGIT), ch
    lgb = both["logits"][0]
    shared = (out["occluded_mass"][0, 0] > 0) & (out["clear_mass"][0, 0] <= 0) & (both["clear_mass"][0, 0] > 0)
    assert int(shared.sum()) > 0, "fixture: some main-occluded cells must be reached by wide"
    assert torch.all(lgb[bl.CH_OCCLUDED][shared].abs() < bl.FORCE_LOGIT)
    assert torch.all(lgb[bl.CH_UNKNOWN][shared].abs() < bl.FORCE_LOGIT)
    print("occlusion_branches")


def check_visibility_taus_plumbed() -> None:
    """Non-default cutoffs reach the head through build_bev_head and change the forcing."""
    kw, kf = 12, 30
    cfg_head = bl.build_bev_head({"bev_lss": {"enabled": True, "reach_tau": 0.3, "free_pass_tau": 0.7}})
    assert cfg_head.reach_tau == 0.3 and cfg_head.free_pass_tau == 0.7
    # p_wall 0.6: between the modes reach = pass = 0.4. Default 0.5 forces; reach_tau 0.3 lets it be seen,
    # but free_pass_tau 0.7 still forbids free.
    # The near-ground cells between the modes depend on the run (the ray's surface is its last reached
    # bin, so a different reach_tau moves the local ground): each run's own between-mode cells, n > 0.
    def run(p_wall, head=None):
        out, rig, h = _run_fixed("main", _bimodal(kw, p_wall, kf), head=head)
        mid = _cells(rig, h, kw + 1, kf, out) & ~_cells(rig, h, 0, kw + 1, out)
        n = int(mid.sum())
        assert n > 0, (p_wall, h.reach_tau, h.free_pass_tau)
        return _forced(out["logits"][0], mid), n

    got, n = run(0.6)
    assert got == (n, n, n), got
    got, n = run(0.6, cfg_head)
    assert got == (n, 0, 0), got
    # p_wall 0.4: pass between modes 0.6. Default: free allowed; free_pass_tau 0.7: free forced low, still seen.
    got, n = run(0.4)
    assert got == (0, 0, 0), got
    strict = bl.build_bev_head({"bev_lss": {"enabled": True, "free_pass_tau": 0.7}})
    assert strict.reach_tau == bl.REACH_TAU and strict.free_pass_tau == 0.7
    got, n = run(0.4, strict)
    assert got == (n, 0, 0), got
    # reach_tau alone: 0.7 makes the 0.4-wall's between cells (reach 0.6) unseen and occluded
    tight = bl.build_bev_head({"bev_lss": {"enabled": True, "reach_tau": 0.7}})
    assert tight.reach_tau == 0.7 and tight.free_pass_tau == bl.FREE_PASS_TAU
    got, n = run(0.4, tight)
    assert got == (n, n, n), got
    print("visibility_taus_plumbed")


def check_depth_transmittance_reference() -> None:
    torch.manual_seed(3)
    prob = torch.rand(2, 3, 7, 4, 5).softmax(dim=2)
    got = bl.depth_transmittance(prob, dim=2)
    ref = torch.zeros_like(prob)
    for i in range(7):
        ref[:, :, i] = prob[:, :, i:].sum(dim=2)
    assert torch.allclose(got, ref, atol=1e-6)
    assert torch.allclose(got[:, :, 0], torch.ones_like(got[:, :, 0]), atol=1e-6)
    assert torch.allclose(got, bl.depth_survival(prob, dim=2) + prob, atol=1e-6)
    flat = torch.tensor([[0.5, 0.3, 0.2]])  # default dim=1 (N x D)
    assert torch.allclose(bl.depth_transmittance(flat), torch.tensor([[1.0, 0.5, 0.2]]))
    print("depth_transmittance_reference")


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
    # the lift geometry runs in K's dtype: float64 depth / edge buffers must not upcast the float32 frustum
    K, T = _simple_cam(pos=(0.0, 0.0, 1.0), fx=100.0)
    headg = bl.BevLssHead()
    headg.spatial = _ConstSpatial(int(torch.argmin((headg.depths - 10.0).abs())), 1.0)
    headg.depths = headg.depths.double()
    headg.edges = headg.edges.double()
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
    assert seen == [torch.float32, torch.float32], seen  # bin centres, then the visibility edges
    print("dtype_follows_model")


# ---------------------------------------------------------------- G5 ground band: rays over low occluders
def _ray_box(o, d, lo, hi):
    t0, t1 = -np.inf, np.inf
    for a in range(3):
        if abs(d[a]) < 1e-12:
            if o[a] < lo[a] or o[a] > hi[a]:
                return None
        else:
            ta, tb = (lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return t0 if (t1 >= t0 and t1 > 0) else None


def _scene_depth(geoms, boxes, p_true: float = 0.95, grade: float = 0.0, z0: float = 0.0) -> torch.Tensor:
    """Ray-cast ground plane z = grade * y + z0 (ego frame; a grade is also ego pitch on a flat road) plus
    boxes: p_true at the true z-depth bin, the rest uniform. Rays that hit nothing within 48 m put p_true
    in the last bin (sky)."""
    log_c = np.log(bl.depth_bin_centers())
    P = torch.zeros(len(geoms), bl.DEPTH_BINS, 24, 32)
    for n, g in enumerate(geoms):
        K, T = g.K.astype(np.float64), g.T.astype(np.float64)
        o = T[:3, 3]
        for r in range(24):
            for c in range(32):
                rc = np.array([((c + 0.5) * 8 - K[0, 2]) / K[0, 0], ((r + 0.5) * 8 - K[1, 2]) / K[1, 1], 1.0])
                d = T[:3, :3] @ rc  # ego displacement per metre of z-depth
                den = d[2] - grade * d[1]
                t_g = (grade * o[1] + z0 - o[2]) / den if abs(den) > 1e-12 else -1.0
                ts = [t_g] if t_g > 0 else []
                for lo, hi in boxes:
                    t = _ray_box(o, d, np.array(lo), np.array(hi))
                    if t is not None:
                        ts.append(t)
                z = min(ts) if ts else 60.0
                k = int(np.argmin(np.abs(log_c - np.log(max(z, 1.0))))) if z <= bl.DEPTH_MAX_M else bl.DEPTH_BINS - 1
                P[n, :, r, c] = (1.0 - p_true) / (bl.DEPTH_BINS - 1)
                P[n, k, r, c] = p_true
    return P


def _scene_run(cams, boxes, head=None, grade: float = 0.0, z0: float = 0.0):
    geoms = bl.rig_geometry(cams, _cfg(), model=BASE_RIG_MODEL)
    rig = bl.rig_tensors(geoms)
    head = (head or bl.BevLssHead()).eval()
    P = _scene_depth(geoms, boxes, grade=grade, z0=z0)
    head.spatial = _FixedDepth([P[i] for i in range(len(cams))])
    nc = len(cams)
    taps = (torch.zeros(1, nc, 64, 48, 64), torch.zeros(1, nc, 128, 24, 32), torch.zeros(1, nc, 128, 12, 16))
    with torch.no_grad():
        out = head(*taps, rig["K"], rig["T"], pixel_valid=rig["pixel_valid"])
    return out, geoms


def _shadow_core(geoms, lo, hi, margin: float = 0.3, grade: float = 0.0, z0: float = 0.0) -> list[tuple[int, int]]:
    """Cells beyond the box whose ground point's ray passes >= margin inside the box for every camera that
    has it in view (Skeptic's shadow-core definition, pr241v3_logs/scene_probe.py)."""
    lo2 = np.array(lo, dtype=np.float64) + margin
    hi2 = np.array(hi, dtype=np.float64) - margin
    lo2[2] = lo[2]
    cells = []
    for r in range(64):
        for c in range(128):
            y, x = 32 - 0.5 * (r + 0.5), -32 + 0.5 * (c + 0.5)
            if y <= hi[1] + 0.5:
                continue
            inview, hidden = 0, True
            for g in geoms:
                o = g.T[:3, 3].astype(np.float64)
                tgt = np.array([x, y, grade * y + z0])
                cam = np.linalg.inv(g.T.astype(np.float64)) @ np.append(tgt, 1.0)
                if cam[2] <= 0:
                    continue
                u = g.K[0, 0] * cam[0] / cam[2] + g.K[0, 2]
                v = g.K[1, 1] * cam[1] / cam[2] + g.K[1, 2]
                nw, nh, x0, y0 = g.content
                if not (x0 <= u < x0 + nw and y0 <= v < y0 + nh):
                    continue
                inview += 1
                t = _ray_box(o, tgt - o, lo2, hi2)
                if t is None or t >= 1.0:
                    hidden = False
                    break
            if hidden and inview > 0:
                cells.append((r, c))
    return cells


def _footprint(lo, hi, whole: bool = False) -> list[tuple[int, int]]:
    """Cells whose centre is inside the box's ground footprint, or (``whole``) whose whole 0.5 m square is."""
    m = 0.25 if whole else 0.0
    return [
        (r, c)
        for r in range(64)
        for c in range(128)
        if lo[0] + m <= -32 + 0.5 * (c + 0.5) <= hi[0] - m and lo[1] + m <= 32 - 0.5 * (r + 0.5) <= hi[1] - m
    ]


CAR_BOX = ((-0.9, 8.0, 0.0), (0.9, 12.5, 1.5))  # 1.8 x 4.5 x 1.5 m car, 8-12.5 m ahead
VAN_BOX = ((-1.0, 8.0, 0.0), (1.0, 13.0, 2.5))  # 2 x 5 x 2.5 m van


def _box_on(lo, hi, grade: float = 0.0, z0: float = 0.0, height: float | None = None):
    """The box moved onto the ground plane z = grade * y + z0 at its centre (Skeptic's slope_probe)."""
    zb = grade * 0.5 * (lo[1] + hi[1]) + z0
    h = hi[2] - lo[2] if height is None else height
    return (lo[0], lo[1], zb), (hi[0], hi[1], zb + h)


def _occluder_case(name, cams, lo, hi, grade=0.0, z0=0.0, min_core=1, check_front=False, front_rows=range(50, 58)) -> None:
    out, geoms = _scene_run(cams, [(lo, hi)], grade=grade, z0=z0)
    lg = out["logits"][0]
    free_ok = lg[bl.CH_FREE] != -bl.FORCE_LOGIT
    seen = lg[bl.CH_UNKNOWN] != bl.FORCE_LOGIT
    core = _shadow_core(geoms, lo, hi, grade=grade, z0=z0)
    assert len(core) >= min_core, (name, cams, grade, z0, len(core))
    nf = sum(bool(free_ok[r, c]) for r, c in core)
    ns = sum(bool(seen[r, c]) for r, c in core)
    assert nf == 0 and ns == 0, f"{name} {cams} grade {grade} z0 {z0}: shadow core {len(core)}: free-eligible {nf}, seen {ns}"
    # cells wholly under the vehicle: never free. (A centre-inside cell that sticks out past the side,
    # e.g. x -1.0..-0.5 for the car's x >= -0.9, may be opened by a ray passing that 0.1 m strip.)
    fp = _footprint(lo, hi, whole=True)
    assert len(fp) >= 18, (name, len(fp))
    nfp = sum(bool(free_ok[r, c]) for r, c in fp)
    assert nfp == 0, f"{name} {cams} grade {grade} z0 {z0}: footprint {len(fp)}: free-eligible {nfp}"
    if check_front:
        # ground in front of the car (3-7 m, centre columns) is seen and free-eligible
        front = [(r, c) for r in front_rows for c in (63, 64)]  # rows 50-57: y 3.25-6.75 m
        assert all(bool(free_ok[r, c]) and bool(seen[r, c]) for r, c in front), (name, cams, grade)


def check_low_occluder_hides_ground() -> None:
    """G5: rays that skim over a car roof do not open the hidden ground behind it or the car's own cells.

    Ray-cast scene (ground plane, a 1.5 m car or a 2.5 m van at 8-12.5 m, depth head 95% sure of the true
    surface) on main alone and on narrow + main + wide. Shadow-core cells and footprint cells: never free,
    never seen. Open ground in front of the car stays free-eligible. Also on a grade (= ego pitch): the car
    at -6 / +3 / +6 %, a 0.6 m obstacle at +3 / +6 % and on flat ground 0.23 m below the ego origin
    (Skeptic's slope_probe C-E: the ego-frame band leaked 12 of 16 hidden cells there).
    """
    for name, (lo, hi) in (("car", CAR_BOX), ("van", VAN_BOX)):
        for cams in (["main"], ["narrow", "main", "wide"]):
            _occluder_case(name, cams, lo, hi, min_core=201, check_front=True)
    for grade in (-0.06, 0.03, 0.06):
        for cams in (["main"], ["narrow", "main", "wide"]):
            # -6 % on main alone: main's lowest ground ray lands 4.07 m out (the road falls away), so the
            # 3.25-3.5 m row is out of its view; the front check starts at 3.5 m there
            rows = range(50, 57) if (grade < 0 and cams == ["main"]) else range(50, 58)
            box = _box_on(*CAR_BOX, grade=grade)
            _occluder_case("car", cams, *box, grade=grade, min_core=100, check_front=True, front_rows=rows)
    for grade, z0 in ((0.03, 0.0), (0.06, 0.0), (0.0, -0.23)):
        lo, hi = _box_on(*CAR_BOX, grade=grade, z0=z0, height=0.6)
        _occluder_case("0.6 m obstacle", ["main"], lo, hi, grade=grade, z0=z0)
    # no occluder: the centre column at 5, 10 and 15 m is free-eligible on main (realistic head, 15 m floor)
    out, _ = _scene_run(["main"], [])
    free_ok = out["logits"][0, bl.CH_FREE] != -bl.FORCE_LOGIT
    for ahead in (5.0, 10.0, 15.0):
        assert bool(free_ok[int((32 - ahead) / 0.5), 64]), ahead
    print("low_occluder_hides_ground")


def _barrier_hidden(geoms, lo, hi, below_top: float = 0.1) -> list[tuple[int, int]]:
    """Cells beyond the barrier whose sight line to the ground centre passes at least
    ``below_top`` under the barrier top for every in-view camera (Skeptic wall_leak)."""
    lo2 = (lo[0] + 0.3, lo[1], lo[2])
    hi2 = (hi[0] - 0.3, hi[1], hi[2] - below_top)
    cells: list[tuple[int, int]] = []
    for r in range(bl.BEV_N):
        for c in range(bl.BEV_N):
            y, x = 32 - 0.5 * (r + 0.5), -32 + 0.5 * (c + 0.5)
            if y <= hi[1] + 0.2:
                continue
            inview, hidden = 0, True
            for g in geoms:
                o = g.T[:3, 3].astype(np.float64)
                tgt = np.array([x, y, 0.0])
                cam = np.linalg.inv(g.T.astype(np.float64)) @ np.append(tgt, 1.0)
                if cam[2] <= 0:
                    continue
                u = g.K[0, 0] * cam[0] / cam[2] + g.K[0, 2]
                v = g.K[1, 1] * cam[1] / cam[2] + g.K[1, 2]
                nw, nh, x0, y0 = g.content
                if not (x0 <= u < x0 + nw and y0 <= v < y0 + nh):
                    continue
                inview += 1
                t = _ray_box(o, tgt - o, np.array(lo2), np.array(hi2))
                if t is None or t >= 1.0:
                    hidden = False
                    break
            if hidden and inview > 0:
                cells.append((r, c))
    return cells


def check_wide_low_barrier_hides_ground() -> None:
    """G5: a barrier wider than the ±3 m window must not reopen hidden ground as free.

    12 m-wide wall at y = 10 m, height 0.6 / 0.8 / 1.0 m, on main alone and on all 8 cameras.
    A cell is hidden when the sight line to its ground centre passes ≥ 0.1 m under the wall top
    (Skeptic wall_leak). Hidden cells are never free-eligible.
    """
    cams8 = [c["id"] for c in _cfg()["cameras"]]
    assert len(cams8) == 8, cams8
    for height in (0.6, 0.8, 1.0):
        lo, hi = (-6.0, 10.0, 0.0), (6.0, 10.3, height)
        for cams in (["main"], cams8):
            out, geoms = _scene_run(cams, [(lo, hi)])
            free_ok = out["logits"][0, bl.CH_FREE] != -bl.FORCE_LOGIT
            hid = _barrier_hidden(geoms, lo, hi)
            assert len(hid) >= 1, (height, cams, len(hid))
            nf = sum(bool(free_ok[r, c]) for r, c in hid)
            assert nf == 0, (
                f"barrier {height} m cams={len(cams)}: hidden {len(hid)} free-eligible {nf}"
            )


def check_empty_road_on_grade_and_pitch() -> None:
    """G5: the band follows the road, not the ego frame. Empty road on main and on narrow + main + wide
    at +-1 deg ego pitch (ground z = tan(+-1 deg) y), a +3 % grade and the ground 0.23 m below the ego
    origin: the centre column is seen and free-eligible at 5, 10 and 15 m (Skeptic's slope_probe B: the
    ego-frame band kept 128 free cells at +3 % and none at 15 m)."""
    for grade, z0 in ((math.tan(math.radians(1.0)), 0.0), (math.tan(math.radians(-1.0)), 0.0), (0.03, 0.0), (0.0, -0.23)):
        for cams in (["main"], ["narrow", "main", "wide"]):
            out, _ = _scene_run(cams, [], grade=grade, z0=z0)
            lg = out["logits"][0]
            free_ok = lg[bl.CH_FREE] != -bl.FORCE_LOGIT
            seen = lg[bl.CH_UNKNOWN] != bl.FORCE_LOGIT
            for ahead in (5.0, 10.0, 15.0):
                r = int((32 - ahead) / 0.5)
                assert bool(free_ok[r, 64]) and bool(seen[r, 64]), (grade, z0, cams, ahead)
            assert int(free_ok.sum()) >= 300, (grade, z0, cams, int(free_ok.sum()))
    print("empty_road_on_grade_and_pitch")


def check_sky_rays_never_free() -> None:
    """A ray above the horizon with its depth mass in the last bin (sky) has no surface and counts nowhere,
    even where its low points pass through road another ray saw."""
    K, T = _simple_cam(pos=(0.0, 0.0, 0.3), fx=100.0)
    p = torch.full((bl.DEPTH_BINS,), 0.05 / (bl.DEPTH_BINS - 1))
    p[-1] = 0.95
    taps = (torch.zeros(1, 1, 64, 48, 64), torch.zeros(1, 1, 128, 24, 32), torch.zeros(1, 1, 128, 12, 16))
    head = bl.BevLssHead().eval()
    head.spatial = _FixedDepth(p)
    sky = torch.zeros(1, 24, 32)
    sky[0, 11, 16] = 1.0  # v = 92 < cy = 96: the ray rises 0.04 m per m, 0.3 -> 0.5 m by 5 m
    with torch.no_grad():
        out = head(*taps, K, T, pixel_valid=sky)
    assert float(out["pass_mass"].sum()) > 0  # the ray does land in the grid (inside the z band)
    assert float(out["through_mass"].sum()) == 0.0 and float(out["clear_mass"].sum()) == 0.0
    assert float(out["occluded_mass"].sum()) == 0.0
    assert torch.all(out["logits"][0, bl.CH_FREE] == -bl.FORCE_LOGIT)
    assert torch.all(out["logits"][0, bl.CH_UNKNOWN] == bl.FORCE_LOGIT)
    # a ground ray (row 14 falls 0.2 m per m) puts a surface 1.5 m ahead; the sky ray's points 1-4.5 m
    # ahead are under 0.5 m above that ground, yet the sky ray still adds nothing (the check bites: they
    # would count if its surface flag were ignored)
    p2 = torch.full((bl.DEPTH_BINS,), 0.05 / (bl.DEPTH_BINS - 1))
    k = int(torch.argmin((torch.log(torch.from_numpy(bl.depth_bin_centers())) - math.log(1.5)).abs()))
    p2[k] = 0.95 + p2[k]
    p2 = p2 / p2.sum()
    pm = torch.stack([p] * 24)[:, :, None].expand(24, bl.DEPTH_BINS, 32).permute(1, 0, 2).clone()
    pm[:, 14, :] = p2[:, None]
    head.spatial = _FixedDepth(pm)
    ground_only = torch.zeros(1, 24, 32)
    ground_only[0, 14, 16] = 1.0
    with torch.no_grad():
        out_g = head(*taps, K, T, pixel_valid=ground_only)
        out_b = head(*taps, K, T, pixel_valid=ground_only + sky)
    assert float(out_g["has_ground"].sum()) > 0
    for name in ("clear_mass", "through_mass", "occluded_mass"):
        assert torch.equal(out_b[name], out_g[name]), name
    # the sky ray's low points do sit inside the band over that ground (so dropping them is the surface rule)
    g = out_g["ground_height"][0, 0].reshape(-1)
    has_g = out_g["has_ground"][0, 0].reshape(-1) > 0
    pts = bl.frustum_ego_points(K, T, head.depths, (24, 32))[0, 0, :, 11, 16]
    idx, ok = bl.bev_flat_index(pts)
    assert bool((ok & has_g[idx] & ((pts[:, 2] - g[idx]) < head.ground_band_m)).any())
    print("sky_rays_never_free")


def check_ray_surface_and_ground_height() -> None:
    """ray_surface: one-hot of the last reached bin, none if the ray is still reached at the last bin.
    ground_height: windowed soft minimum of the surface heights, no ground where the window has none."""
    reached = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    surf, has = bl.ray_surface(reached, dim=1)
    assert surf.tolist() == [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], surf
    assert has.view(-1).tolist() == [1, 0, 1]
    k = bl.GROUND_SOFTMIN_K
    z_sum = torch.zeros(1, 1, 128, 128)
    w_sum = torch.zeros(1, 1, 128, 128)

    def put(r, c, z, w=1.0):
        z_sum[0, 0, r, c] += w * math.exp(-k * (z - bl.BEV_Z_MIN_M))
        w_sum[0, 0, r, c] += w

    put(40, 64, 0.3)
    g, hg = bl.ground_height(z_sum, w_sum)
    half = bl.GROUND_WINDOW_CELLS // 2
    assert int(hg.sum()) == bl.GROUND_WINDOW_CELLS ** 2, int(hg.sum())
    assert bool(hg[0, 0, 40 - half, 64 + half]) and not bool(hg[0, 0, 40 - half - 1, 64])
    assert abs(float(g[0, 0, 40 + half, 64 - half]) - 0.3) < 1e-4
    assert float(g[0, 0, 0, 0]) == 0.0
    # a roof point 1.5 m above the road in the same window barely moves the soft minimum
    put(42, 64, 1.8, w=4.0)
    g2, _ = bl.ground_height(z_sum, w_sum)
    assert 0.3 <= float(g2[0, 0, 41, 64]) < 0.3 + math.log(5.0) / k + 1e-4, float(g2[0, 0, 41, 64])
    print("ray_surface_and_ground_height")


def check_ground_band_pinned_and_plumbed() -> None:
    """GROUND_BAND_M is 0.5 m, the window 13 cells, k 10 / m; the band is validated and a non-default value
    reaches the head through the config."""
    assert bl.GROUND_BAND_M == 0.5, bl.GROUND_BAND_M
    assert bl.GROUND_WINDOW_CELLS == 13 and bl.GROUND_SOFTMIN_K == 10.0
    assert bl.GROUND_WIDE_WINDOW_CELLS == 21 and bl.GROUND_SLOPE_CAP == 0.15
    assert bl.RAISED_OCC_M == 0.3 and bl.OCC_WINDOW_CELLS == 3 and bl.SIGHTLINE_STEPS == 12 and bl.SIGHTLINE_CLEARANCE_M == 0.05 and bl.SIGHTLINE_MARGIN_M == 1.5
    assert bl.BevLssHead().ground_band_m == 0.5
    span = bl.BEV_Z_MAX_M - bl.BEV_Z_MIN_M
    for bad in (0.0, -0.5, span + 0.1, float("nan")):
        _raises(ValueError, lambda: bl.BevLssHead(ground_band_m=bad), match="ground_band_m")
    bl.BevLssHead(ground_band_m=span)  # the whole z band is legal
    head = bl.build_bev_head({"bev_lss": {"enabled": True, "ground_band_m": 2.0}})
    assert head.ground_band_m == 2.0 and head.reach_tau == bl.REACH_TAU
    # 2.0 m band: rays over the 1.5 m car roof count again, so the car's footprint opens (default stays shut)
    out, _ = _scene_run(["main"], [CAR_BOX], head=head)
    free_ok = out["logits"][0, bl.CH_FREE] != -bl.FORCE_LOGIT
    assert sum(bool(free_ok[r, c]) for r, c in _footprint(*CAR_BOX)) > 0
    # boundary: a point exactly band above the ground is outside (z - ground < band), like the BEV z band
    z = torch.tensor([0.25, 0.5 - 1e-6, 0.5, 0.5 + 1e-6, 3.0]) + 1.0
    g = torch.full_like(z, 1.0)
    assert bl.near_ground(z, g, 0.5).tolist() == [True, True, False, False, False]
    assert bl.near_ground(z, g, 2.0).tolist() == [True, True, True, True, False]
    print("ground_band_pinned_and_plumbed")

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
    check_visibility_taus_pinned()
    check_transmittance_boundary()
    check_wall_behind_is_unknown_not_free()
    check_bimodal_depth_occludes_between_modes()
    check_visibility_masses_values()
    check_occlusion_branches()
    check_visibility_taus_plumbed()
    check_empty_road_on_grade_and_pitch()
    check_ray_surface_and_ground_height()
    check_ground_band_pinned_and_plumbed()
    check_sky_rays_never_free()
    check_low_occluder_hides_ground()
    check_wide_low_barrier_hides_ground()
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
    check_depth_transmittance_reference()
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
