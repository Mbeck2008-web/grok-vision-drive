#!/usr/bin/env python3
"""Property + fuzz checks for Reality View stable fill near range: road_snap / ground_clamp /
edge cut / far cap / depth filter / camera ownership (yield_mask), through the REAL per-camera
render path.

The per-camera path lives in tools/rv_stable_preview.py (``cam_frame``: road_snap -> ground_clamp
-> depth-edge cut -> far_to_cap -> valid -> ownership), followed in its main loop by
CamDepthFilter.update and unproject. This file does not copy ``cam_frame``: it parses the preview
source and executes the real ``cam_frame`` / ``proc_res`` / ``fy_full`` defs and the real module
constants (CAMS, SKY, EGO_MASK, FY_REF, CAM_POSE loop, YIELD_TO, YIELD) against a fake preds
npz and fake RGB files. Only the two-line filter call from the main loop is mirrored
(``_filter_step``). Camera rig = config/cameras.yaml (checked).

Invariants (bounds and where they come from):
A  no rendered surface below the road plane: every valid pixel's unprojected z >= -TOL_Z.
   TOL_Z = 1 mm (judgment: >> the ~1e-7 m float32 inverse-depth round trip in cam_frame,
   << 1 chase pixel). Ground plane z = 0 is the plane road_snap / ground_clamp themselves use
   (python/viz/rv_stable_fill.py, ground_z=0). Near rows = pixels whose ray meets the ground
   within NEAR_M = 15 m horizontal range (Michael's 0-15 m), not where the point ended up.
B  seam: (1) exact flat ground fed to two forward cams leaves no seam > SEAM_PX = 1 chase pixel in
   0-15 m (chase = preview pose A, default --width/--height 480x270, parsed from the preview);
   per 0.25 m cell (GRID_CELL_M) the two cams' median z are projected into the chase and compared.
   (2) open-road main/wide seam median <= SEAM_MEDIAN_M = 0.15 m (GVD's gate, 2026-11-26).
   Tunnel wide-pillar seams are scoped out by Michael: reported only, never gated.
C  sky stays sky: a sky-labelled camera pixel never becomes a valid point (any depth: NaN, inf,
   0, negative, huge; any road label over it); in compose, an up-ray chase pixel farther than the
   fill reach (iters * (win // 2) = 4 px, fill_gaps defaults) from every observed pixel is sky.
D  far stays at cap: in cam_frame a non-sky up-ray pixel with input depth >= cap (incl. +inf and
   exactly cap) comes out exactly at cap and valid, and input below cap is never pulled closer
   (relative 1e-6 for the float32 inverse round trip) and never past cap. After the depth filter
   (the next runtime step), a static at-cap scene stays at cap within CAP_TOL_M = 1 mm.
Y  ownership (yield_mask, when present): wide never keeps a pixel whose ray (200 m probe, as the
   code defines it) lands >= 4 px inside main; pillars are unaffected on the config rig; the mask
   is depth-independent (NaN/inf depth); empty / full overlap, 1x1 cams, no better cam behave.
O  objects keep their height: a pole, car face or curb on the road within the snap range (25 m)
   keeps z_render >= z_gt - OBJ_TOL_M (0.05 m, judgment) on every rendered object pixel more than
   2 px (the 5x5 edge-cut window) inside its silhouette, with the net's road label at the class
   grid resolution (12 x 16 cells, majority).
H  height bias by GT range bin (clip015 f40/f160, needs the clip: GVD_RV_CLIP015 / _PREDS / _GT,
   skipped visibly otherwise): rendered road z - GT road z per bin 0-5/5-10/10-15/15-25/25-60 m.
   Near road 0-15 m: median |dz| <= 0.08 m and p90 |dz| <= 0.20 m (Redesigner's pre-set gate;
   KNOWN_FAILING, belongs to gvd/rv-ground-plane). 25-60 m: no worse than the 4df1176 snap
   range (25 m) on the same input. Synthetic graded road: same 25-60 m gate, always runs.
   Snap on vs off (clip015 f40/f160): per cam, road median |dz| and p90 |dz| with snap on are no
   worse (+1 mm) than with --road-snap 0 in 0-5, 5-10, 10-15 m.
   Holdout (Skeptic): every clip gate is re-run on >= 2 held-out clip015 frames >= 40 frames from
   f40/f160 (GVD_RV_HOLDOUT_FRAMES to pin them) with the same thresholds; pass on f40/f160 + fail on
   holdout = overfit counterexample.
G  GT is measurement only: cam_frame in pred mode never reads GT depth (no gt root at all), and
   python/viz/rv_stable_fill.py never reads GT (no state/ego/gt names).

KNOWN_FAILING (real bugs on this head; they report XFAIL and the run fails with XPASS when one
starts passing, so each turns into a fail->pass test for the fix). See KNOWN_FAILING below.

RV_PROPS_REPORT=1 prints PASS/FAIL per property without XFAIL/XPASS gating (before/after tables).
"""
from __future__ import annotations

import ast
import atexit
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gvd_test_env import skip_check  # noqa: E402

try:
    from hypothesis import HealthCheck, example, given, settings
    from hypothesis import strategies as st
except ImportError:  # dev-only dep (requirements-dev.txt); CI fails instead of skipping
    given = None

from python.viz import rv_stable_fill as sf  # noqa: E402
from python.viz.reality_view_c import camera_from_angles  # noqa: E402

PREVIEW = ROOT / "tools" / "rv_stable_preview.py"

# ------------------------------------------------------------------ thresholds (committed first)
TOL_Z = 1e-3            # m, below-road tolerance (judgment, see A)
NEAR_M = 15.0           # m, Michael's near range
SEAM_PX = 1.0           # chase px at the preview default size (see B)
SEAM_MEDIAN_M = 0.15    # m, open-road main/wide seam median (GVD gate)
NEAR_MEDIAN_M = 0.08    # m, clip015 near road 0-15 m median |dz| (Redesigner's pre-set gate)
NEAR_P90_M = 0.20       # m, clip015 near road 0-15 m p90 |dz|
BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 25.0), (25.0, 60.0))
BASELINE_SNAP_MAX_M = 25.0   # 4df1176 python/viz/rv_stable_fill.py ROAD_SNAP_MAX_M
NO_WORSE_TOL_M = 1e-3   # m, "no worse" slack for the 25-60 m bin
CAP_TOL_M = 1e-3        # m
CAP_REL = 1e-6          # relative, the float32 inverse-depth round trip in cam_frame (1/d -> float32 -> 1/x)
OBJ_TOL_M = 0.05        # m, object height kept (judgment)
YIELD_MARGIN_PX = 4.0   # yield_mask default margin_px
OBJ_SNAP_RANGE_M = 25.0  # Michael: poles/cars/curbs keep their height at 25 m

CAM_SPECS = {  # config/cameras.yaml (check_cam_specs_match_config)
    "narrow": {"id": "narrow", "pos_m": [0.03, 1.20, 1.28], "yaw_deg": 0.0, "pitch_deg": -4.0, "fov_v": 21.4, "resolution": [640, 480], "far_m": 800.0},
    "main": {"id": "main", "pos_m": [0.00, 1.20, 1.26], "yaw_deg": 0.0, "pitch_deg": -6.0, "fov_v": 34.0, "resolution": [640, 480], "far_m": 300.0},
    "wide": {"id": "wide", "pos_m": [-0.03, 1.20, 1.24], "yaw_deg": 0.0, "pitch_deg": -8.0, "fov_v": 88.7, "resolution": [640, 360], "far_m": 300.0},
    "pillarL": {"id": "pillarL", "pos_m": [-0.88, 0.40, 1.18], "yaw_deg": -78.0, "pitch_deg": -6.0, "fov_v": 65.3, "resolution": [640, 480], "far_m": 100.0},
    "pillarR": {"id": "pillarR", "pos_m": [0.88, 0.40, 1.18], "yaw_deg": 78.0, "pitch_deg": -6.0, "fov_v": 65.3, "resolution": [640, 480], "far_m": 100.0},
}
FWD = ("narrow", "main", "wide")
ROAD_ID = 2
OBJ_ID = 13
NONROAD_ID = 0


SKIPPED = "skipped"


# ------------------------------------------------------------------ real preview code, loaded
class _FakeNpz(dict):
    @property
    def files(self):
        return list(self.keys())


def _argparse_defaults(tree: ast.Module) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            dest = str(node.args[0].value).lstrip("-").replace("-", "_")
            val = None
            for kw in node.keywords:
                if kw.arg == "default":
                    val = ast.literal_eval(kw.value)
            out[dest] = val
    return out


_WANT_FN = {"fy_full", "proc_res", "cam_frame", "footprint"}
_WANT_AS = {"CAMS", "SKY", "CHASE_POS", "CHASE_PITCH", "CHASE_FOV", "EGO_BOX", "EGO_MASK", "FY_REF", "CAM_POSE",
            "CLAMP_STATS", "FAR_STATS", "SNAP_STATS", "YIELD_TO", "YIELD"}


def _assign_names(node) -> set:
    tg = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = set()
    for t in tg:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                names.add(n.id)
    return names


class Preview:
    """The preview's own per-camera code, executed against fake inputs (or a real clip)."""

    def __init__(self, overrides: dict | None = None, *, root: Path | None = None, P=None, cams: dict | None = None):
        tree = ast.parse(PREVIEW.read_text(encoding="utf-8"))
        self.defaults = _argparse_defaults(tree)
        a = dict(self.defaults)
        a.update(depth_src="pred", pose_src="vo")
        a.update(overrides or {})
        self.a = types.SimpleNamespace(**a)
        self.tmp = Path(tempfile.mkdtemp(prefix="rvprops-"))
        atexit.register(shutil.rmtree, self.tmp, True)
        self.fake = root is None
        self.root = self.tmp if root is None else Path(root)
        self.cams_json = dict(cams if cams is not None else CAM_SPECS)
        self.P = _FakeNpz() if P is None else P
        body = []
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name in _WANT_FN:
                body.append(n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)) and _assign_names(n) and _assign_names(n) <= _WANT_AS:
                body.append(n)
            elif isinstance(n, ast.For) and "cams_json.items()" in ast.unparse(n.iter):
                body.append(n)
        self.ns = {"math": math, "json": json, "np": np, "cv2": cv2, "Path": Path, "S": sf,
                   "camera_from_angles": camera_from_angles, "a": self.a, "root": self.root,
                   "cams_json": self.cams_json, "P": self.P}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(PREVIEW), "exec"), self.ns)
        self.cam = self.ns["CAM_POSE"]
        self.has_yield = "YIELD" in self.ns

    def run(self, cid: str, depth: np.ndarray, sem: np.ndarray, idx: int = 0):
        """cam_frame(cid, idx) with the net's depth = ``depth`` (metres at proc res) and labels ``sem``."""
        c = self.cams_json[cid]
        W0, H0 = c["resolution"]
        if self.fake:
            p = self.root / "raw" / cid / "000"
            p.mkdir(parents=True, exist_ok=True)
            f = p / f"{idx:06d}.png"
            if not f.exists():
                cv2.imwrite(str(f), np.full((H0, W0, 3), 128, np.uint8))
            fs = (self.ns["fy_full"](c) * 192.0 / H0) / self.ns["FY_REF"]
            for k in [k for k in self.P if k.startswith(f"{cid}/{idx}/")]:
                del self.P[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                self.P[f"{cid}/{idx}/0/d"] = np.asarray(depth, np.float64) / fs
            self.P[f"{cid}/{idx}/0/x0"] = np.array([0, W0])
            self.P[f"{cid}/{idx}/0/s"] = np.asarray(sem)
        with np.errstate(all="ignore"):
            return self.ns["cam_frame"](cid, idx)

    def chase(self):
        n = self.ns
        return sf.chase_camera(n["CHASE_POS"], n["CHASE_PITCH"], n["CHASE_FOV"], int(self.defaults["width"]),
                               int(self.defaults["height"]), 0.0)


def _filter_step(F: sf.CamDepthFilter, cam, depth, valid, pose, t):
    """Mirror of tools/rv_stable_preview.py main loop (the --depth-filter 1 branch)."""
    conf = np.clip(1.0 / (1.0 + depth / 20.0), 0.2, 1.0)
    d, _ = F.update(cam, depth, valid, pose, t, conf=np.where(valid, conf, 0.0))
    return d


_PV: Preview | None = None


def pv() -> Preview:
    global _PV
    if _PV is None:
        _PV = Preview()
    return _PV


# ------------------------------------------------------------------ geometry helpers
def geom(cam):
    rays = sf._pixel_rays(cam)
    dz = rays[..., 2]
    down = dz < -1e-6
    hit = np.where(down, cam.pos[2] / np.where(down, -dz, 1.0), np.inf)
    t = np.where(down, hit, np.nan)
    gx = cam.pos[0] + t * rays[..., 0]
    gy = cam.pos[1] + t * rays[..., 1]
    R = np.where(down, np.hypot(gx, gy), np.inf)
    return {"rays": rays, "dz": dz, "down": down, "hit": hit, "R": R}


def heights(cam, d):
    return cam.pos[2] + np.asarray(d, np.float64) * sf._pixel_rays(cam)[..., 2]


def exact_ground(cam, road: bool):
    g = geom(cam)
    depth = np.where(g["down"], g["hit"], 1e4)
    sem = np.where(g["down"], ROAD_ID if road else NONROAD_ID, sf_sky())
    return depth, sem, g


def sf_sky() -> int:
    return int(pv().ns["SKY"])


def tunnel_fixture(cam):
    """scripts/test_rv_stable_fill.py S9 shape: 1.0x at the top, 0.5x mid, 2.5x at the bottom; road rect."""
    g = geom(cam)
    H, W = g["hit"].shape
    ys, xs = np.mgrid[0:H, 0:W]
    bend = np.interp(ys, [0, H // 2, H - 1], [1.0, 0.5, 2.5])
    road = (ys >= H * 0.45) & (xs >= W * 0.2) & (xs < W * 0.8)
    depth = np.where(g["down"], g["hit"] * bend, 1e4)
    sem = np.where(~g["down"], sf_sky(), np.where(road, ROAD_ID, NONROAD_ID))
    return depth, sem, g


def near_min_z(cid, d, v, g):
    cam = pv().cam[cid]
    z = heights(cam, d)
    m = v & (g["R"] <= NEAR_M)
    if not m.any():
        return 0.0, None
    i = np.unravel_index(int(np.argmin(np.where(m, z, np.inf))), z.shape)
    return float(z[i]), (int(i[0]), int(i[1]), round(float(g["R"][i]), 2))


def coarse(fine: np.ndarray, rows: int = 12, cols: int = 16) -> np.ndarray:
    """Majority label per class-grid cell (the net's 12x16 class grid)."""
    H, W = fine.shape
    out = np.zeros((rows, cols), np.int64)
    for r in range(rows):
        for c in range(cols):
            blk = fine[r * H // rows:(r + 1) * H // rows, c * W // cols:(c + 1) * W // cols].ravel()
            vals, cnt = np.unique(blk, return_counts=True)
            out[r, c] = vals[int(np.argmax(cnt))]
    return out


# ------------------------------------------------------------------ seam
def _cells(p, lo, hi):
    r = np.hypot(p[:, 0], p[:, 1])
    k = (r >= lo) & (r <= hi) & (p[:, 1] > 0)
    ix = np.floor(p[k, 0] / sf.GRID_CELL_M).astype(np.int64)
    iy = np.floor(p[k, 1] / sf.GRID_CELL_M).astype(np.int64)
    zz = p[k, 2]
    out = {}
    order = np.lexsort((iy, ix))
    ix, iy, zz = ix[order], iy[order], zz[order]
    if ix.size == 0:
        return out
    brk = np.nonzero((np.diff(ix) != 0) | (np.diff(iy) != 0))[0] + 1
    for s, e in zip(np.r_[0, brk], np.r_[brk, ix.size]):
        out[(int(ix[s]), int(iy[s]))] = float(np.median(zz[s:e]))
    return out


def seam(pa, pb, lo, hi, chase=None):
    """(max chase-px gap, median |dz| m, n common cells, worst cell) between two cams' points."""
    A, B = _cells(pa, lo, hi), _cells(pb, lo, hi)
    common = sorted(set(A) & set(B))
    if not common:
        return 0.0, 0.0, 0, None
    c = np.array([[(a + 0.5) * sf.GRID_CELL_M, (b + 0.5) * sf.GRID_CELL_M] for a, b in common])
    za = np.array([A[k] for k in common])
    zb = np.array([B[k] for k in common])
    ch = chase or pv().chase()
    _, va, _ = sf.project(ch, np.c_[c, za])
    _, vb, _ = sf.project(ch, np.c_[c, zb])
    dv = np.abs(va - vb)
    i = int(np.argmax(dv))
    return float(dv[i]), float(np.median(np.abs(za - zb))), len(common), (c[i].round(2).tolist(), round(float(za[i]), 4), round(float(zb[i]), 4))


def points(cid, d, v):
    cam = pv().cam[cid]
    ys, xs = np.nonzero(v)
    return sf.unproject(cam, d, ys, xs)


# ------------------------------------------------------------------ graded road (GT for measurement)
def graded_gt(cam, grade: float, y0: float):
    """Planar depth of the road z = grade * max(0, y - y0) (flat, then a steady climb)."""
    rays = sf._pixel_rays(cam)
    dy, dz = rays[..., 1], rays[..., 2]
    h, cy = cam.pos[2], cam.pos[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_flat = np.where(dz < -1e-9, h / -dz, np.inf)
        flat_ok = cy + t_flat * dy <= y0
        den = grade * dy - dz
        t_sl = np.where(den > 1e-9, (h - grade * (cy - y0)) / den, np.inf)
        sl_ok = (t_sl > 0) & (cy + t_sl * dy > y0)
    t = np.where(flat_ok, t_flat, np.where(sl_ok, t_sl, np.inf))
    return t


def bias_bins(cam, d_render, d_gt, mask):
    """{bin: (median dz, p90 |dz|, n)} of rendered z - GT z on ``mask``, binned by GT ground range."""
    ys, xs = np.nonzero(mask & np.isfinite(d_gt) & np.isfinite(d_render))
    if ys.size == 0:
        return {b: (float("nan"), float("nan"), 0) for b in BINS}
    pr = sf.unproject(cam, d_render, ys, xs)
    pg = sf.unproject(cam, d_gt, ys, xs)
    dzv = pr[:, 2] - pg[:, 2]
    rng = np.hypot(pg[:, 0], pg[:, 1])
    out = {}
    for lo, hi in BINS:
        k = (rng >= lo) & (rng < hi)
        out[(lo, hi)] = (float(np.median(dzv[k])) if k.any() else float("nan"),
                         float(np.percentile(np.abs(dzv[k]), 90)) if k.any() else float("nan"), int(k.sum()))
    return out


# ------------------------------------------------------------------ objects on the road
def object_scene(cam, kind: str, rng_m: float, lat_m: float):
    """GT depth of flat ground + one upright face, the net's coarse labels and the object mask."""
    g = geom(cam)
    rays = g["rays"]
    if kind == "pole":
        w, h_ = 0.15, 2.5
    elif kind == "car":
        w, h_ = 1.8, 1.5
    else:  # curb: a 0.15 m step along the road edge, facing the car
        w, h_ = 3.0, 0.15
    x0, x1 = lat_m - w / 2, lat_m + w / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(rays[..., 1] > 1e-9, (rng_m - cam.pos[1]) / rays[..., 1], np.inf)
        X = cam.pos[0] + t * rays[..., 0]
        Z = cam.pos[2] + t * rays[..., 2]
    face = np.isfinite(t) & (X >= x0) & (X <= x1) & (Z >= 0) & (Z <= h_) & (t < g["hit"])
    gt = np.where(face, t, np.where(g["down"], g["hit"], np.inf))
    fine = np.where(face, OBJ_ID, np.where(~g["down"], sf_sky(), ROAD_ID))
    return gt, fine, face, g


# ------------------------------------------------------------------ deterministic checks (pass)
def check_cam_specs_match_config() -> None:
    try:
        import yaml
    except ImportError:
        skip_check("cam_specs_match_config", "pyyaml not installed")
        return SKIPPED
    cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))
    by = {c["id"]: c for c in cfg["cameras"]}
    for cid, s in CAM_SPECS.items():
        c = by[cid]
        assert [float(v) for v in c["pos_m"]] == s["pos_m"], cid
        assert float(c.get("yaw_deg") or 0.0) == s["yaw_deg"] and float(c["pitch_deg"]) == s["pitch_deg"], cid
        assert float(c["fov_v"]) == s["fov_v"] and list(c["live_res"]) == s["resolution"] and float(c["far_m"]) == s["far_m"], cid


def check_harness_runs_real_preview_code() -> None:
    p = pv()
    assert "cam_frame" in p.ns and callable(p.ns["cam_frame"])
    assert p.a.clamp_margin == 1.0 and p.a.road_snap == 1 and p.a.ground_clamp == 1 and p.a.max_depth == "cap"
    for cid in FWD:
        cam = p.cam[cid]
        assert (cam.width, cam.height) == p.ns["proc_res"](CAM_SPECS[cid])
        depth, sem, g = exact_ground(cam, road=False)
        _, d, v, sky = p.run(cid, depth, sem)
        assert d.shape == (cam.height, cam.width) and v.any() and sky.any()


def check_no_gt_in_runtime() -> None:
    """G: pred mode reads no GT (no gt root exists at all) and the runtime module names no GT/state."""
    p = Preview({"gt_depth_root": "/nonexistent-gt-root"})
    depth, sem, _ = exact_ground(p.cam["main"], road=True)
    p.run("main", depth, sem)  # would raise on the GT branch (cv2.imread None .astype)
    src = (ROOT / "python" / "viz" / "rv_stable_fill.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    leak = {x for x in names if x.lower() in {"gt", "gt_depth", "state", "ego_pos", "depth_src", "pose_src"}}
    assert not leak, f"GT/state names in runtime module: {leak}"
    cf = next(n for n in ast.parse(PREVIEW.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "cam_frame")
    for node in ast.walk(cf):
        if isinstance(node, ast.Attribute) and node.attr == "gt_depth_root":
            pass
    gt_uses = [n for n in ast.walk(cf) if isinstance(n, ast.If) and "gt_depth_root" in ast.unparse(n.body[0]) if n.body]
    for n in gt_uses:
        assert "depth_src" in ast.unparse(n.test) and "gt" in ast.unparse(n.test), "GT read must be gated by --depth-src gt"


def check_exact_ground_near_rows_single_frame() -> None:
    """A on exact flat ground, one frame, with and without a road label: holds on 4df1176."""
    p = pv()
    for cid in FWD:
        for road in (False, True):
            depth, sem, g = exact_ground(p.cam[cid], road)
            _, d, v, _ = p.run(cid, depth, sem)
            z, where = near_min_z(cid, d, v, g)
            assert z >= -TOL_Z, (cid, road, z, where)


def check_seam_exact_ground_near() -> None:
    """B(1): exact ground into narrow/main/wide leaves no seam > 1 chase px in 0-15 m."""
    p = pv()
    pts = {}
    for cid in FWD:
        depth, sem, _ = exact_ground(p.cam[cid], road=False)
        _, d, v, _ = p.run(cid, depth, sem)
        pts[cid] = points(cid, d, v)
    for a_, b_ in (("wide", "main"), ("wide", "narrow"), ("main", "narrow")):
        px, med, n, worst = seam(pts[a_], pts[b_], 0.0, NEAR_M)
        assert px <= SEAM_PX, (a_, b_, px, worst)


def open_road_seam():
    """Main/wide seam on an open road (road label on every ground pixel, the S9 net-like bend)."""
    p = pv()
    pts = {}
    for cid in ("main", "wide"):
        depth, _, g = tunnel_fixture(p.cam[cid])
        sem = np.where(g["down"], ROAD_ID, sf_sky())
        _, d, v, _ = p.run(cid, depth, sem)
        pts[cid] = points(cid, d, v)
    return seam(pts["main"], pts["wide"], 0.0, 60.0)


def check_seam_open_road_median() -> None:
    px, med, n, worst = open_road_seam()
    assert n == 0 or med <= SEAM_MEDIAN_M, (med, n, px, worst)


def check_sky_all_sky_frame() -> None:
    p = pv()
    for cid in FWD:
        cam = p.cam[cid]
        for fill in (1e4, np.nan, np.inf, 0.0, -5.0):
            depth = np.full((cam.height, cam.width), fill)
            sem = np.full(depth.shape, sf_sky())
            _, d, v, sky = p.run(cid, depth, sem)
            assert sky.all() and not v.any(), (cid, fill)


def check_far_exactly_at_cap() -> None:
    p = pv()
    for cid in FWD:
        cam = p.cam[cid]
        cap = sf.depth_cap_m(CAM_SPECS[cid]["far_m"])
        g = geom(cam)
        for val in (cap, cap * (1 + 1e-9), 1e6, np.inf):
            depth = np.full(g["hit"].shape, val)
            sem = np.full(depth.shape, NONROAD_ID)
            _, d, v, _ = p.run(cid, depth, sem)
            up = ~g["down"] & v
            assert up.any() and np.all(d[up] <= cap) and np.all(d[up] >= cap * (1 - CAP_REL)), (cid, val, float(d[up].min()), float(d[up].max()))


def check_yield_edges() -> None:
    """Y fuzz edges: no better cam, empty overlap, full overlap, 1x1, zero-size refused, depth-free."""
    if not hasattr(sf, "yield_mask"):
        skip_check("yield_edges", "no yield_mask on this revision")
        return SKIPPED
    p = pv()
    wide, main = p.cam["wide"], p.cam["main"]
    assert not sf.yield_mask(wide, []).any()
    back = camera_from_angles("rear", (0.0, -2.0, 1.0), 180.0, 0.0, 60.0, 64, 48, near_m=0.05, far_m=100.0)
    assert not sf.yield_mask(wide, [back]).any(), "empty overlap must not yield"
    full = sf.yield_mask(main, [main])
    u = np.arange(main.width) + 0.5
    v = np.arange(main.height) + 0.5
    inner = (v[:, None] >= YIELD_MARGIN_PX) & (v[:, None] <= main.height - YIELD_MARGIN_PX) & (u[None, :] >= YIELD_MARGIN_PX) & (u[None, :] <= main.width - YIELD_MARGIN_PX)
    assert np.array_equal(full, inner), "full overlap: exactly the >= margin interior yields"
    one = camera_from_angles("wide", (0.0, 1.2, 1.24), 0.0, -8.0, 88.7, 1, 1, near_m=0.05, far_m=300.0)
    assert sf.yield_mask(one, [main]).shape == (1, 1) and bool(sf.yield_mask(one, [main])[0, 0])
    try:
        camera_from_angles("wide", (0.0, 1.2, 1.24), 0.0, -8.0, 88.7, 0, 10)
        raise AssertionError("zero-size camera must be refused")
    except ValueError:
        pass
    if p.has_yield:
        Y = p.ns["YIELD"]["wide"]
        for val in (np.nan, np.inf, 5.0):
            depth = np.full((wide.height, wide.width), val)
            _, _, v, _ = p.run("wide", depth, np.full(depth.shape, NONROAD_ID))
            assert not (v & Y).any(), ("wide kept a yielded pixel", val)


def check_yield_pillars_unaffected() -> None:
    p = pv()
    if not p.has_yield:
        skip_check("yield_pillars_unaffected", "no ownership on this revision")
        return SKIPPED
    for cid in ("pillarL", "pillarR"):
        if cid in p.ns["YIELD"]:
            assert not p.ns["YIELD"][cid].any(), (cid, float(p.ns["YIELD"][cid].mean()))
        cam = p.cam[cid]
        depth, sem, _ = exact_ground(cam, road=True)
        _, d_on, v_on, _ = p.run(cid, depth, sem)
        off = Preview({"ownership": 0})
        _, d_off, v_off, _ = off.run(cid, depth, sem)
        assert np.array_equal(v_on, v_off) and np.array_equal(d_on, d_off), cid


def check_yield_wide_never_wins_inside_main() -> None:
    p = pv()
    if not p.has_yield:
        skip_check("yield_wide_never_wins_inside_main", "no ownership on this revision")
        return SKIPPED
    wide, main = p.cam["wide"], p.cam["main"]
    rays = sf._pixel_rays(wide).reshape(-1, 3)
    probe = np.asarray(wide.pos) + rays * (200.0 / np.linalg.norm(rays, axis=1, keepdims=True))
    u, v, z = sf.project(main, probe)
    m = YIELD_MARGIN_PX
    inside = (z > 0) & (u >= m) & (u <= main.width - m) & (v >= m) & (v <= main.height - m)
    depth, sem, _ = exact_ground(wide, road=True)
    _, _, val, _ = p.run("wide", depth, sem)
    assert not (val.ravel() & inside).any(), int((val.ravel() & inside).sum())


# ------------------------------------------------------------------ pinned counterexamples
def check_pin_label_edge_dip() -> None:
    """A: the S9 tunnel fixture on the wide cam (road rect 45%..100% rows, 20%..80% cols).

    Edge band of the road label (not snapped) sits next to snapped road; the depth-edge cut
    (tools/rv_stable_preview.py ``far_side`` / ``np.where(cut & ~sky & ~snapped, far_side, depth)``)
    runs AFTER ground_clamp and moves the band pixel to the farther ground 2 rows up: below the road.
    """
    p = pv()
    depth, sem, g = tunnel_fixture(p.cam["wide"])
    _, d, v, _ = p.run("wide", depth, sem)
    z, where = near_min_z("wide", d, v, g)
    assert z >= -TOL_Z, f"wide label-edge pixel below road at near range: z={z:.4f} m (row, col, ground range m)={where}"


def check_pin_object_base_dip() -> None:
    """A: exact flat ground + a 3x12 px upright face 2% nearer than the ground at ~8 m, wide cam, col 40..52.

    The ground pixels under the face see a big inverse-depth step, get cut, and are re-placed at the
    5x5 max depth (ground 2 rows farther): the road under any near object dips.
    """
    p = pv()
    cam = p.cam["wide"]
    depth, sem, g = exact_ground(cam, road=False)
    col = 46
    r0 = int(np.argmin(np.abs(np.where(g["down"], g["R"], 1e9)[:, col] - 8.0)))
    depth = depth.copy()
    depth[r0 - 3:r0, col - 6:col + 6] = g["hit"][r0, col] * 0.98
    _, d, v, _ = p.run("wide", depth, sem)
    z, where = near_min_z("wide", d, v, g)
    assert z >= -TOL_Z, f"road under a near object below road: z={z:.4f} m at {where}"


def check_pin_horizon_band_dip() -> None:
    """A (whole frame): exact flat ground, no road label, wide cam. Rows just past 15 m are cut and
    pushed to the next-farther rows' depth (up to the sky's 1e4 -> cap): road from ~15.5 m bends down."""
    p = pv()
    depth, sem, g = exact_ground(p.cam["wide"], road=False)
    _, d, v, _ = p.run("wide", depth, sem)
    z = heights(p.cam["wide"], d)
    m = v & g["down"] & (g["R"] <= 30.0)
    i = np.unravel_index(int(np.argmin(np.where(m, z, np.inf))), z.shape)
    bad = m & (z < -TOL_Z)
    first = float(g["R"][bad].min()) if bad.any() else float("nan")
    assert z[i] >= -TOL_Z, (f"exact ground rendered below road: z={z[i]:.4f} m at row {i[0]} col {i[1]} ground range "
                            f"{g['R'][i]:.2f} m; dips start at ground range {first:.2f} m")


def check_pin_filter_unsnaps_road() -> None:
    """A: exact ground, road everywhere, perfect pose, 3 frames at 1 m/frame through CamDepthFilter.

    The filter blends the previous surface point's planar depth into a pixel whose ray is not that
    point's ray (floor lookup, python/viz/rv_stable_fill.py CamDepthFilter.update ``out_l[cy, cx] = ...``),
    after snap/clamp, and nothing re-clamps: snapped road ripples below the road."""
    p = pv()
    cam = p.cam["wide"]
    depth, sem, g = exact_ground(cam, road=True)
    F = sf.CamDepthFilter()
    for k in range(3):
        _, d, v, _ = p.run("wide", depth, sem)
        d = _filter_step(F, cam, d, v, (0.0, 1.0 * k, 0.0), 0.1 * k)
    z, where = near_min_z("wide", d, v, g)
    assert z >= -TOL_Z, f"filtered snapped road below road: z={z:.4f} m at {where}"


def check_pin_filter_far_past_cap() -> None:
    """D: a static non-sky wall beyond the cap (net 1e5 m), wide cam, yaw 3 deg/frame, perfect pose:
    after the filter, at-cap pixels leave the cap both ways (planar depth of the carried point grows
    when it rotates toward the axis)."""
    p = pv()
    cam = p.cam["wide"]
    cap = sf.depth_cap_m(CAM_SPECS["wide"]["far_m"])
    g = geom(cam)
    depth = np.full(g["hit"].shape, 1e5)
    sem = np.full(depth.shape, NONROAD_ID)
    F = sf.CamDepthFilter()
    for k in range(2):
        _, d, v, _ = p.run("wide", depth, sem)
        d = _filter_step(F, cam, d, v, (0.0, 0.0, math.radians(3.0) * k), 0.1 * k)
    m = v & ~g["down"]
    lo, hi = float(d[m].min()), float(d[m].max())
    assert cap - CAP_TOL_M <= lo and hi <= cap + CAP_TOL_M, f"cap {cap} m, filtered at-cap range [{lo:.2f}, {hi:.2f}] m"


def check_pin_clamp_inf_then_cap() -> None:
    """A (unit): ground_clamp leaves +inf on a downward ray (``np.isfinite(d) & (d > zmax)``), then
    far_to_cap draws it AT the cap along that ray: deep under the road. Not reachable from the pred
    path at this head (cam_frame's depth is <= 1e6 by construction), reachable by any +inf caller."""
    cam = pv().cam["wide"]
    g = geom(cam)
    d = np.where(g["down"], np.inf, 50.0)
    out, _ = sf.ground_clamp(cam, d, margin=1.0)
    out, _ = sf.far_to_cap(out, np.zeros(d.shape, bool), 250.0)
    z = heights(cam, out)
    assert z[g["down"]].min() >= -TOL_Z, f"+inf depth on a downward ray -> z={z[g['down']].min():.2f} m"


def check_pin_objects_snapped_flat() -> None:
    """O: with the net's 12x16 class grid, thin objects on the road share a road-majority cell, sit
    >= 4 px inside the road label and are snapped onto z = 0 (python/viz/rv_stable_fill.py road_snap).
    Main cam, lat 0.5 m: a curb at 20 m and a pole at 24 m (inside the 25 m snap range)."""
    p = pv()
    bad = []
    for cid, kind, r in (("main", "curb", 20.0), ("main", "pole", 24.0)):
        cam = p.cam[cid]
        gt, fine, face, _ = object_scene(cam, kind, r, 0.5)
        _, d, v, _ = p.run(cid, np.where(np.isfinite(gt), gt, 1e4), coarse(fine))
        zg, zr = heights(cam, gt), heights(cam, d)
        flat = face & v & (zg > OBJ_TOL_M) & (np.abs(zr) <= TOL_Z)
        if flat.any():
            bad.append(f"{kind}@{r:g}m: {int(flat.sum())}/{int((face & v).sum())} px flattened, GT z up to {zg[flat].max():.2f} m -> 0.00")
    assert not bad, "; ".join(bad)


def graded_road_bins(snap_max_m=None):
    """Bias bins (main + wide road pixels) for a road that climbs 3% from 20 m; net depth = GT."""
    over = {} if snap_max_m is None else {"road_snap_max_m": snap_max_m}
    p = Preview(over) if over else pv()
    out = {}
    for cid in ("main", "wide"):
        cam = p.cam[cid]
        gt = graded_gt(cam, 0.03, 20.0)
        sem = np.where(np.isfinite(gt), ROAD_ID, sf_sky())
        depth = np.where(np.isfinite(gt), gt, 1e4)
        _, d, v, _ = p.run(cid, depth, sem)
        out[cid] = bias_bins(cam, d, gt, v & (sem == ROAD_ID))
    return out


def check_graded_road_far_bin_no_worse() -> None:
    """H synthetic: 25-60 m road bias on a 3% climb is no worse than with the 4df1176 snap range (25 m)."""
    p = pv()
    if "road_snap_max_m" not in p.defaults and sf.ROAD_SNAP_MAX_M == BASELINE_SNAP_MAX_M:
        return  # this IS the baseline (4df1176): nothing to compare
    cur = graded_road_bins()
    base = graded_road_bins(BASELINE_SNAP_MAX_M)
    for cid in cur:
        c, b = cur[cid][(25.0, 60.0)], base[cid][(25.0, 60.0)]
        if c[2] == 0:
            continue
        assert abs(c[0]) <= abs(b[0]) + NO_WORSE_TOL_M and c[1] <= b[1] + NO_WORSE_TOL_M, (cid, "now", c, "4df1176 range", b)


# ------------------------------------------------------------------ clip015 (real data, optional)
def gt_at_proc(g_full: np.ndarray, pw: int, ph: int) -> np.ndarray:
    """GT (full res) sampled at each processing pixel's centre (no nearest-resize half-pixel shift)."""
    H0, W0 = g_full.shape[:2]
    yi = np.minimum(((np.arange(ph) + 0.5) * H0 / ph).astype(np.int64), H0 - 1)
    xi = np.minimum(((np.arange(pw) + 0.5) * W0 / pw).astype(np.int64), W0 - 1)
    return g_full[yi[:, None], xi[None, :]]


DEFAULT_CLIP = Path("/workspace/rv-preview/v2src/clip015")
DEFAULT_PREDS = Path("/workspace/rv-preview/v2/preds_clip015.npz")
DEFAULT_GT = Path("/workspace/rv-preview/v2src/gt_eval_only")
TRACER_HOLDOUT = (100, 200)
CLEAN_HOLDOUT = (70,)


def clip015_paths():
    clip = Path(os.environ["GVD_RV_CLIP015"]) if os.environ.get("GVD_RV_CLIP015") else DEFAULT_CLIP
    preds = Path(os.environ["GVD_RV_CLIP015_PREDS"]) if os.environ.get("GVD_RV_CLIP015_PREDS") else DEFAULT_PREDS
    gt = Path(os.environ["GVD_RV_CLIP015_GT"]) if os.environ.get("GVD_RV_CLIP015_GT") else DEFAULT_GT
    if clip.joinpath("cameras.json").is_file() and preds.is_file() and gt.is_dir():
        return clip, preds, gt
    return None


def clip_metrics(clip: Path, preds: Path, gt_root: Path, frames=(40, 160), overrides: dict | None = None):
    """Bias bins (rendered road z - GT road z, main+wide, net road label) and seams on real frames."""
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(overrides, root=clip, P=P, cams=cams)
    def _frames(cid):
        return sorted({int(k.split("/")[1]) for k in P.files if k.startswith(cid + "/") and k.endswith("/d")})
    def _nearest(cid, target):
        frs = _frames(cid)
        return min(frs, key=lambda i: (abs(i - target), i)) if frs else None
    acc = {b: [] for b in BINS}
    acc_cam: dict = {}
    near = []
    seams_open, seams_wp = [], []
    used = {}
    for f in frames:
        pts = {}
        for cid in [c for c in ("main", "wide", "pillarL", "pillarR") if c in p.cam]:
            idx = _nearest(cid, f)
            if idx is None:
                continue
            used.setdefault(cid, {})[f] = idx
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            _, d, v, sky = fr
            cam = p.cam[cid]
            c = cams[cid]
            W0, _ = c["resolution"]
            pw, ph = p.ns["proc_res"](c)
            ys, xs = np.nonzero(v)
            pts[cid] = sf.unproject(cam, d, ys, xs)
            if cid not in ("main", "wide"):
                continue
            g = cv2.imread(str(gt_root / f"{cid}_depth" / "000" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
            if g is None:
                continue
            g = gt_at_proc(g.astype(np.float32) / 100.0, pw, ph)
            gtd = np.where((g > 0) & (g < min(c["far_m"] * sf.GT_NO_HIT_FRAC, 655.0)), g, np.nan)
            road = np.zeros((ph, pw), bool)
            sx = pw / W0
            k = 0
            while f"{cid}/{idx}/{k}/s" in P.files:
                x0, cw = P[f"{cid}/{idx}/{k}/x0"]
                a0, a1 = int(round(x0 * sx)), int(round((x0 + cw) * sx))
                s = P[f"{cid}/{idx}/{k}/s"]
                road[:, a0:a1] |= cv2.resize(np.isin(s, sf.ROAD_CLASS_IDS).astype(np.uint8), (a1 - a0, ph),
                                             interpolation=cv2.INTER_NEAREST) > 0
                k += 1
            ys, xs = np.nonzero(v & road & ~sky & np.isfinite(gtd))
            pr = sf.unproject(cam, d, ys, xs)
            pg = sf.unproject(cam, gtd, ys, xs)
            rr = np.hypot(pg[:, 0], pg[:, 1])
            dzv = pr[:, 2] - pg[:, 2]
            for lo, hi in BINS:
                acc[(lo, hi)].append(dzv[(rr >= lo) & (rr < hi)])
                acc_cam.setdefault(cid, {b: [] for b in BINS})[(lo, hi)].append(dzv[(rr >= lo) & (rr < hi)])
            near.append(dzv[rr < NEAR_M])
        if "main" in pts and "wide" in pts:
            seams_open.append((f, seam(pts["main"], pts["wide"], 0.0, 60.0)))
        for pl in ("pillarL", "pillarR"):
            if pl in pts and "wide" in pts:
                seams_wp.append((f, pl, seam(pts["wide"], pts[pl], 0.0, 60.0)))
    def stats(x):
        if x.size == 0:
            return (float("nan"), float("nan"), float("nan"), 0)
        return (float(np.median(x)), float(np.median(np.abs(x))), float(np.percentile(np.abs(x), 90)), int(x.size))
    bins = {b: stats(np.concatenate(parts) if parts else np.zeros(0)) for b, parts in acc.items()}
    by_cam = {c: {b: stats(np.concatenate(v) if v else np.zeros(0)) for b, v in d_.items()} for c, d_ in acc_cam.items()}
    return {"bins": bins, "bins_by_cam": by_cam, "near": stats(np.concatenate(near) if near else np.zeros(0)),
            "seam_open": seams_open, "seam_wide_pillar": seams_wp, "used": used}


_CLIP: dict = {}


GATE_FRAMES = (40, 160)
HOLDOUT_MIN_GAP = 40     # Skeptic: held-out frames >= 40 frames from f40/f160 (no fitting on the test set)
HOLDOUT_N = 2            # at least 2 held-out frames
SNAP_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0))


def holdout_frames(P_files, n: int = 4) -> tuple:
    """Frames in the preds with every cam key, >= HOLDOUT_MIN_GAP from each gate frame (env override
    GVD_RV_HOLDOUT_FRAMES=a,b,..), up to ``n`` spread evenly."""
    env = os.environ.get("GVD_RV_HOLDOUT_FRAMES", "").strip()
    if env:
        fr = [int(x) for x in env.split(",") if x.strip()]
    else:
        fr = sorted({int(k.split("/")[1]) for k in P_files if k.startswith("main/") and k.endswith("/0/d")})
    fr = [f for f in fr if all(abs(f - g) >= HOLDOUT_MIN_GAP for g in GATE_FRAMES)]
    if len(fr) > n:
        fr = [fr[int(round(i * (len(fr) - 1) / (n - 1)))] for i in range(n)]
    return tuple(fr)


def _clip(overrides=None, frames=GATE_FRAMES):
    key = json.dumps([overrides or {}, list(frames)], sort_keys=True)
    if key not in _CLIP:
        _CLIP[key] = clip_metrics(*clip015_paths(), frames=frames, overrides=overrides)
    return _CLIP[key]


def _holdout() -> tuple:
    return TRACER_HOLDOUT


def _near_gate(m, label):
    _, med, p90, n = m["near"]
    assert n > 0 and med <= NEAR_MEDIAN_M and p90 <= NEAR_P90_M, f"{label} near road 0-15 m: median |dz| {med:.3f} m, p90 {p90:.3f} m, n {n}"


def _snap_gate(frames, label):
    on = _clip(None, frames)["bins_by_cam"]
    off = _clip({"road_snap": 0}, frames)["bins_by_cam"]
    bad = []
    for cid in on:
        for b in SNAP_BINS:
            c, o = on[cid][b], off.get(cid, {}).get(b, (np.nan,) * 4)
            if c[3] == 0 or o[3] == 0:
                continue
            if c[1] > o[1] + NO_WORSE_TOL_M or c[2] > o[2] + NO_WORSE_TOL_M:
                bad.append(f"{cid} {b[0]:g}-{b[1]:g} m: snap on median|dz| {c[1]:.3f}/p90 {c[2]:.3f} vs off {o[1]:.3f}/{o[2]:.3f}")
    assert not bad, f"{label}: " + "; ".join(bad)


def check_clip015_snap_no_worse_than_off() -> None:
    """Per cam, road |dz| with snap on is no worse than snap off in 0-5/5-10/10-15 m (f40, f160)."""
    if clip015_paths() is None:
        skip_check("clip015_snap_no_worse_than_off", "clip015 not on this host")
        return SKIPPED
    _snap_gate(GATE_FRAMES, "f40/f160")


def check_holdout_snap_no_worse_than_off() -> None:
    if clip015_paths() is None:
        skip_check("holdout_snap_no_worse_than_off", "clip015 not on this host")
        return SKIPPED
    _snap_gate(_holdout(), "holdout")


def check_holdout_near_road_bias() -> None:
    """Same near-road gate on held-out frames (a pass on f40/f160 + fail here = overfit counterexample)."""
    if clip015_paths() is None:
        skip_check("holdout_near_road_bias", "clip015 not on this host")
        return SKIPPED
    fr = _holdout()
    m = _clip(None, fr)
    print(f"holdout {fr} bins:", _fmt_bins(m["bins"]), flush=True)
    _near_gate(m, f"holdout {fr}")


def check_holdout_open_road_seam() -> None:
    if clip015_paths() is None:
        skip_check("holdout_open_road_seam", "clip015 not on this host")
        return SKIPPED
    for f, (px, med, n, worst) in _clip(None, _holdout())["seam_open"]:
        assert n == 0 or med <= SEAM_MEDIAN_M, ("holdout", f, med, n, worst)


def check_clean_holdout_near_road_bias() -> None:
    if clip015_paths() is None:
        skip_check("clean_holdout_near_road_bias", "clip015 not on this host")
        return SKIPPED
    m = _clip(None, CLEAN_HOLDOUT)
    print(f"clean holdout {CLEAN_HOLDOUT} bins:", _fmt_bins(m["bins"]), flush=True)
    _near_gate(m, f"clean holdout {CLEAN_HOLDOUT}")


def check_clean_holdout_snap_no_worse_than_off() -> None:
    if clip015_paths() is None:
        skip_check("clean_holdout_snap_no_worse_than_off", "clip015 not on this host")
        return SKIPPED
    _snap_gate(CLEAN_HOLDOUT, "clean holdout f70")


def check_clean_holdout_open_road_seam() -> None:
    if clip015_paths() is None:
        skip_check("clean_holdout_open_road_seam", "clip015 not on this host")
        return SKIPPED
    for f, (px, med, n, worst) in _clip(None, CLEAN_HOLDOUT)["seam_open"]:
        assert n == 0 or med <= SEAM_MEDIAN_M, ("clean holdout", f, med, n, worst)


def check_no_fitted_z0() -> None:
    """Flag a z0 or lateral offset in runtime code that looks fitted on clip015 (e.g. +0.09)."""
    bad = []
    for rel in ("python/viz/rv_stable_fill.py", "tools/rv_stable_preview.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        if "0.09" in text or "+0.09" in text:
            bad.append(rel)
    assert not bad, f"possible fitted z0=+0.09 (or 0.09) in {bad}"


def _fmt_bins(bins):
    return {f"{lo:g}-{hi:g}": tuple(round(v, 4) if isinstance(v, float) else v for v in t) for (lo, hi), t in bins.items()}


def check_clip015_near_road_bias() -> None:
    """H on real clip015 f40/f160: near road 0-15 m median |dz| <= 0.08 m and p90 |dz| <= 0.20 m."""
    if clip015_paths() is None:
        skip_check("clip015_near_road_bias", "clip015 not on this host (set GVD_RV_CLIP015, GVD_RV_CLIP015_PREDS, GVD_RV_CLIP015_GT)")
        return SKIPPED
    m = _clip()
    print("clip015 bins (median dz, median |dz|, p90 |dz|, n):", _fmt_bins(m["bins"]), flush=True)
    print("clip015 wide-pillar seams (report only, scoped out):", m["seam_wide_pillar"], flush=True)
    _near_gate(m, "f40/f160")


def check_clip015_far_bin_no_worse() -> None:
    """H on real clip015: 25-60 m bin no worse than with the 4df1176 snap range (25 m)."""
    if clip015_paths() is None:
        skip_check("clip015_far_bin_no_worse", "clip015 not on this host")
        return SKIPPED
    if "road_snap_max_m" not in pv().defaults:
        return  # 4df1176 itself
    cur = _clip()["bins"][(25.0, 60.0)]
    base = _clip({"road_snap_max_m": BASELINE_SNAP_MAX_M})["bins"][(25.0, 60.0)]
    assert cur[3] == 0 or (cur[1] <= base[1] + NO_WORSE_TOL_M and cur[2] <= base[2] + NO_WORSE_TOL_M), (cur, base)


def check_clip015_open_road_seam() -> None:
    if clip015_paths() is None:
        skip_check("clip015_open_road_seam", "clip015 not on this host")
        return SKIPPED
    for f, (px, med, n, worst) in _clip()["seam_open"]:
        assert n == 0 or med <= SEAM_MEDIAN_M, (f, med, n, worst)


def check_clip_harness_selftest() -> None:
    """The clip015 measurement path on a synthetic clip (flat ground, preds = GT): every bin ~0."""
    tmp = Path(tempfile.mkdtemp(prefix="rvclip-"))
    atexit.register(shutil.rmtree, tmp, True)
    cams = [dict(CAM_SPECS[c]) for c in ("main", "wide")]
    (tmp / "cameras.json").write_text(json.dumps({"cameras": cams}))
    gt_root = tmp / "gt"
    store = {}
    for c in cams:
        cid = c["id"]
        W0, H0 = c["resolution"]
        pw, ph = pv().ns["proc_res"](c)
        cam = pv().cam[cid]
        g = geom(cam)
        fs = (pv().ns["fy_full"](c) * 192.0 / H0) / pv().ns["FY_REF"]
        for f in GATE_FRAMES:
            (tmp / "raw" / cid / "000").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp / "raw" / cid / "000" / f"{f:06d}.png"), np.full((H0, W0, 3), 128, np.uint8))
            store[f"{cid}/{f}/0/d"] = np.where(g["down"], g["hit"], 1e4) / fs
            store[f"{cid}/{f}/0/x0"] = np.array([0, W0])
            store[f"{cid}/{f}/0/s"] = np.where(g["down"], ROAD_ID, sf_sky())
            camf = camera_from_angles(cid, c["pos_m"], c["yaw_deg"], c["pitch_deg"], c["fov_v"], W0, H0, near_m=0.05, far_m=400.0)
            gf = geom(camf)
            gfull = np.where(gf["down"], np.minimum(gf["hit"], 600.0), 0.0)
            (gt_root / f"{cid}_depth" / "000").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(gt_root / f"{cid}_depth" / "000" / f"{f:06d}.png"), np.round(gfull * 100).astype(np.uint16))
    np.savez(tmp / "preds.npz", **store)
    m = clip_metrics(tmp, tmp / "preds.npz", gt_root)
    for (lo, hi), (md, mad, p90, n) in m["bins"].items():
        if hi <= 25.0:
            assert n > 0 and abs(md) < 0.02 and p90 < 0.05, ((lo, hi), md, p90, n)
    off = clip_metrics(tmp, tmp / "preds.npz", gt_root, overrides={"road_snap": 0})
    assert set(off["bins_by_cam"]) == {"main", "wide"}
    assert holdout_frames(["main/0/0/d", "main/40/0/d", "main/80/0/d", "main/100/0/d", "main/200/0/d", "main/201/0/d"]) == (0, 80, 200, 201)


# ------------------------------------------------------------------ hypothesis props
def _settings(n):
    return settings(max_examples=n, deadline=None, derandomize=True, database=None,
                    suppress_health_check=list(HealthCheck), print_blob=False)


def _build_props():
    if given is None:
        return []
    knot = st.floats(0.3, 3.0, allow_nan=False)

    @_settings(40)
    @given(cid=st.sampled_from(FWD), k0=knot, k1=knot, k2=knot,
           r0=st.floats(0.3, 0.95), c0=st.floats(0.0, 0.5), c1=st.floats(0.5, 1.0))
    def prop_near_rows_no_dip(cid, k0, k1, k2, r0, c0, c1):
        """A: any net-like bend + any road rect -> no valid near-row pixel below the road."""
        p = pv()
        g = geom(p.cam[cid])
        H, W = g["hit"].shape
        ys, xs = np.mgrid[0:H, 0:W]
        bend = np.interp(ys, [0, H // 2, H - 1], [k0, k1, k2])
        road = (ys >= H * r0) & (xs >= W * c0) & (xs < W * c1)
        depth = np.where(g["down"], g["hit"] * bend, 1e4)
        sem = np.where(~g["down"], sf_sky(), np.where(road, ROAD_ID, NONROAD_ID))
        _, d, v, _ = p.run(cid, depth, sem)
        z, where = near_min_z(cid, d, v, g)
        assert z >= -TOL_Z, f"{cid}: near-row pixel below road z={z:.4f} m at {where}"

    @_settings(30)
    @given(cid=st.sampled_from(FWD), k0=knot, k1=knot, k2=knot, r0=st.floats(0.3, 0.95))
    def prop_road_core_on_plane(cid, k0, k1, k2, r0):
        """A/S9: eroded road within the snap range lands on the plane through the whole cam_frame."""
        p = pv()
        g = geom(p.cam[cid])
        H, W = g["hit"].shape
        ys, xs = np.mgrid[0:H, 0:W]
        bend = np.interp(ys, [0, H // 2, H - 1], [k0, k1, k2])
        road = (ys >= H * r0) & g["down"]
        depth = np.where(g["down"], g["hit"] * bend, 1e4)
        sem = np.where(~g["down"], sf_sky(), np.where(road, ROAD_ID, NONROAD_ID))
        _, d, v, _ = p.run(cid, depth, sem)
        core = sf._erode(road, sf.ROAD_SNAP_ERODE_PX) & (g["hit"] <= sf.ROAD_SNAP_MAX_M) & v
        z = heights(p.cam[cid], d)
        assert not core.any() or np.max(np.abs(z[core])) <= TOL_Z, float(np.max(np.abs(z[core])))

    @_settings(30)
    @given(cid=st.sampled_from(FWD), step=st.floats(0.0, 2.0), n=st.integers(2, 4))
    def prop_filter_keeps_ground(cid, step, n):
        """A after the depth filter: exact ground, road everywhere, perfect pose, straight drive."""
        p = pv()
        cam = p.cam[cid]
        depth, sem, g = exact_ground(cam, road=True)
        F = sf.CamDepthFilter()
        for k in range(n):
            _, d, v, _ = p.run(cid, depth, sem)
            d = _filter_step(F, cam, d, v, (0.0, step * k, 0.0), 0.1 * k)
        z, where = near_min_z(cid, d, v, g)
        assert z >= -TOL_Z, f"{cid} step {step} m: z={z:.4f} m at {where}"

    @_settings(30)
    @given(cid=st.sampled_from(FWD), yaw_deg=st.floats(-5.0, 5.0))
    def prop_filter_keeps_cap(cid, yaw_deg):
        """D after the depth filter: a static at-cap scene stays at cap under a pure yaw."""
        p = pv()
        cam = p.cam[cid]
        cap = sf.depth_cap_m(CAM_SPECS[cid]["far_m"])
        g = geom(cam)
        depth = np.full(g["hit"].shape, 1e5)
        sem = np.full(depth.shape, NONROAD_ID)
        F = sf.CamDepthFilter()
        for k in range(2):
            _, d, v, _ = p.run(cid, depth, sem)
            d = _filter_step(F, cam, d, v, (0.0, 0.0, math.radians(yaw_deg) * k), 0.1 * k)
        m = v & ~g["down"]
        if m.any():
            assert cap - CAP_TOL_M <= d[m].min() and d[m].max() <= cap + CAP_TOL_M, (cid, yaw_deg, float(d[m].min()), float(d[m].max()))

    special = st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -1.0, 1e-9, 0.3, 1e6, 1e300])
    @_settings(40)
    @given(cid=st.sampled_from(FWD), sky_r=st.floats(0.0, 1.0), road_r=st.floats(0.0, 1.0),
           fill=st.one_of(special, st.floats(0.01, 1e4)), seed=st.integers(0, 2**16))
    def prop_sky_never_valid(cid, sky_r, road_r, fill, seed):
        """C: sky-labelled pixels never become points, whatever the depth or road label over them."""
        p = pv()
        cam = p.cam[cid]
        H, W = cam.height, cam.width
        rng = np.random.default_rng(seed)
        depth = np.where(rng.random((H, W)) < 0.5, fill, rng.uniform(0.1, 400.0, (H, W)))
        ys = np.mgrid[0:H, 0:W][0]
        sky = (ys < H * sky_r) | (rng.random((H, W)) < 0.05)
        sem = np.where(sky, sf_sky(), np.where(ys >= H * road_r, ROAD_ID, NONROAD_ID))
        _, d, v, skym = p.run(cid, depth, sem)
        assert np.array_equal(skym, sky)
        assert not (v & sky).any(), int((v & sky).sum())
        assert np.all(np.isfinite(d[v])) and np.all(d[v] > 0)

    @_settings(30)
    @given(cid=st.sampled_from(FWD), val=st.one_of(st.floats(0.31, 1e7), st.just(np.inf)), seed=st.integers(0, 999))
    def prop_far_at_cap_cam_frame(cid, val, seed):
        """D in cam_frame: non-sky up-ray pixels: >= cap -> exactly cap; below cap -> never closer, never past cap."""
        p = pv()
        cam = p.cam[cid]
        cap = sf.depth_cap_m(CAM_SPECS[cid]["far_m"])
        g = geom(cam)
        rng = np.random.default_rng(seed)
        depth = np.where(rng.random(g["hit"].shape) < 0.5, val, cap)
        _, d, v, _ = p.run(cid, depth, np.full(depth.shape, NONROAD_ID))
        up = ~g["down"]
        if p.has_yield and cid in p.ns["YIELD"]:
            up = up & ~p.ns["YIELD"][cid]   # owned by main (ownership), not a far pixel dropped
        assert np.all(d[v] <= cap)
        big = up & (depth >= cap)
        assert np.all(v[big]) and np.all(d[big] >= cap * (1 - CAP_REL)), float(d[big].min())
        small = up & (depth < cap)
        if small.any():
            assert np.all(d[small] >= depth[small] * (1 - CAP_REL)), float(np.min(d[small] / depth[small]))

    @_settings(40)
    @given(H=st.integers(1, 64), W=st.integers(1, 64), pitch=st.floats(-60.0, 30.0), fov=st.floats(5.0, 170.0),
           h=st.floats(-1.0, 5.0), fill=special, seed=st.integers(0, 2**16), erode=st.integers(0, 6), maxm=st.floats(0.0, 100.0))
    def prop_unit_clamp_snap(H, W, pitch, fov, h, fill, seed, erode, maxm):
        """A unit: ground_clamp (margin 1.0) never leaves a finite downward depth below the road and
        never touches NaN; road_snap puts snapped pixels on the plane and leaves the rest bit-identical."""
        cam = camera_from_angles("wide", (0.0, 1.2, h), 0.0, pitch, fov, W, H, near_m=0.05, far_m=400.0)
        rng = np.random.default_rng(seed)
        d = np.where(rng.random((H, W)) < 0.3, fill, rng.uniform(0.01, 500.0, (H, W)))
        road = rng.random((H, W)) < 0.7
        out, m = sf.ground_clamp(cam, d, margin=1.0)
        z = heights(cam, out)
        dz = sf._pixel_rays(cam)[..., 2]
        fin = np.isfinite(out) & (dz < -1e-6) & (h > 0)
        assert not fin.any() or z[fin].min() >= -TOL_Z, float(z[fin].min())
        assert np.array_equal(np.isnan(out), np.isnan(d))
        out2, m2 = sf.road_snap(cam, d, road, erode_px=erode, max_hit_m=maxm)
        if m2.any():
            assert np.max(np.abs(heights(cam, out2)[m2])) <= TOL_Z
        assert np.array_equal(out2[~m2], d[~m2], equal_nan=True)

    @_settings(25)
    @given(seed=st.integers(0, 2**16), n=st.integers(0, 3000), fill=st.booleans())
    def prop_compose_sky_stays_sky(seed, n, fill):
        """C in compose: sky never overlaps painted pixels; up-ray pixels beyond the fill reach are sky."""
        rng = np.random.default_rng(seed)
        chase = sf.chase_camera((0.0, -4.0, 2.6), -12.0, 60.0, 96, 54, 0.0)
        pts = np.c_[rng.uniform(-20, 20, n), rng.uniform(0, 60, n), rng.uniform(-0.5, 3.0, n)]
        r = sf.compose(chase, pts, np.full((n, 3), 200, np.uint8), np.full(n, 0.3), sky_rgb=(120, 160, 220),
                       ground_rgb=(90, 90, 90), fill=fill)
        mk = r.masks
        assert not (mk["sky"] & (mk["observed"] | mk["filled"])).any()
        up = sf._pixel_rays(chase)[..., 2] >= -1e-9
        near_obs = cv2.dilate(mk["observed"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        assert np.all(mk["sky"][up & ~near_obs])

    @_settings(25)
    @given(yaw=st.floats(-180, 180), pitch=st.floats(-30, 30), fov=st.floats(10, 150), W=st.integers(1, 80), H=st.integers(1, 60),
           margin=st.floats(0.0, 10.0))
    def prop_yield_matches_definition(yaw, pitch, fov, W, H, margin):
        """Y: for any overlap geometry (none .. full), yield == 'ray lands >= margin inside the better cam'."""
        if not hasattr(sf, "yield_mask"):
            return
        main = pv().cam["main"]
        cam = camera_from_angles("wide", (-0.03, 1.2, 1.24), yaw, pitch, fov, W, H, near_m=0.05, far_m=300.0)
        y = sf.yield_mask(cam, [main], margin_px=margin)
        rays = sf._pixel_rays(cam).reshape(-1, 3)
        u, v, z = sf.project(main, np.asarray(cam.pos) + rays * (200.0 / np.linalg.norm(rays, axis=1, keepdims=True)))
        want = (z > 0) & (u >= margin) & (u <= main.width - margin) & (v >= margin) & (v <= main.height - margin)
        assert y.shape == (H, W) and np.array_equal(y.ravel(), want)

    @_settings(30)
    @given(cid=st.sampled_from(("main", "wide")), kind=st.sampled_from(("pole", "car", "curb")),
           rng_m=st.floats(4.0, OBJ_SNAP_RANGE_M), lat=st.floats(-3.0, 3.0))
    @example(cid="main", kind="car", rng_m=20.0, lat=0.0)
    def prop_objects_keep_height(cid, kind, rng_m, lat):
        """O: a pole / car face / curb on the road within 25 m keeps its height (coarse net road label)."""
        p = pv()
        cam = p.cam[cid]
        gt, fine, face, g = object_scene(cam, kind, rng_m, lat)
        sem = coarse(fine)
        depth = np.where(np.isfinite(gt), gt, 1e4)
        _, d, v, _ = p.run(cid, depth, sem)
        inner = sf._erode(face, 2) & v
        if not inner.any():
            return
        zr = heights(cam, d)[inner]
        zg = heights(cam, gt)[inner]
        lost = zg - zr
        i = int(np.argmax(lost))
        assert lost[i] <= OBJ_TOL_M, f"{kind} at {rng_m:.2f} m lat {lat:.2f} ({cid}): pixel at GT z {zg[i]:.3f} m rendered at {zr[i]:.3f} m"

    return [prop_near_rows_no_dip, prop_road_core_on_plane, prop_filter_keeps_ground, prop_filter_keeps_cap,
            prop_sky_never_valid, prop_far_at_cap_cam_frame, prop_unit_clamp_snap, prop_compose_sky_stays_sky,
            prop_yield_matches_definition, prop_objects_keep_height]


PROPS = _build_props()

CHECKS = [
    check_cam_specs_match_config,
    check_harness_runs_real_preview_code,
    check_no_gt_in_runtime,
    check_exact_ground_near_rows_single_frame,
    check_seam_exact_ground_near,
    check_seam_open_road_median,
    check_sky_all_sky_frame,
    check_far_exactly_at_cap,
    check_yield_edges,
    check_yield_pillars_unaffected,
    check_yield_wide_never_wins_inside_main,
    check_graded_road_far_bin_no_worse,
    check_clip_harness_selftest,
    check_clip015_near_road_bias,
    check_clip015_far_bin_no_worse,
    check_clip015_open_road_seam,
    check_clip015_snap_no_worse_than_off,
    check_holdout_snap_no_worse_than_off,
    check_holdout_near_road_bias,
    check_holdout_open_road_seam,
    check_clean_holdout_snap_no_worse_than_off,
    check_clean_holdout_near_road_bias,
    check_clean_holdout_open_road_seam,
    check_no_fitted_z0,
    check_pin_label_edge_dip,
    check_pin_object_base_dip,
    check_pin_horizon_band_dip,
    check_pin_filter_unsnaps_road,
    check_pin_filter_far_past_cap,
    check_pin_clamp_inf_then_cap,
    check_pin_objects_snapped_flat,
]

# name -> the counterexample it fails on at 783fedd (real runs, rv-evidence/props_report_783fedd.log).
# Responsible code: E = tools/rv_stable_preview.py cam_frame depth-edge cut
#   ``depth = np.where(cut & ~sky & ~snapped, far_side, depth)`` (runs AFTER ground_clamp, nothing re-clamps);
# F = python/viz/rv_stable_fill.py CamDepthFilter.update ``out_l[cy, cx] = ...`` (runs after cam_frame,
#   blends a neighbouring ray's carried point; nothing re-clamps or re-caps);
# N = road_snap on the net's coarse 12x16 class grid (python/viz/rv_stable_fill.py road_snap).
KNOWN_FAILING: dict[str, str] = {
    "check_pin_label_edge_dip": "E: S9 tunnel fixture, wide: label-edge px z=-0.2376 m at row 89 col 67, ground range 13.97 m",
    "check_pin_object_base_dip": "E: exact ground + 3x12 px face 2% nearer at ~8 m, wide col 46: road under it z=-0.1276 m at 9.53 m",
    "check_pin_horizon_band_dip": "E: exact flat ground, no road label, wide: dips from 17.76 m, z=-1.0168 m at 29.90 m",
    "check_pin_filter_unsnaps_road": "F: exact ground, road everywhere, perfect pose, 3 frames x 1 m, wide: z=-0.0533 m at 14.96 m",
    "check_pin_filter_far_past_cap": "F: non-sky wall at 1e5 m, wide, yaw 3 deg/frame: at-cap px span [239.98, 260.43] m, cap 250",
    "check_pin_clamp_inf_then_cap": "ground_clamp keeps +inf on a downward ray, far_to_cap draws it at cap: z=-274.22 m (unit; not reachable from preds)",
    "check_pin_objects_snapped_flat": "N: main curb@20 m 50/100 px flattened (GT 0.10 m -> 0.00); pole@24 m 2/32 px flattened",
    "prop_near_rows_no_dip": "E: narrow, bend knots (1.0, 1.0, 0.5), road rows >= 50%, full width: z=-0.0320 m at row 142 col 3, 9.28 m",
    "prop_filter_keeps_ground": "F: narrow, 1.0 m/frame, 2 frames, perfect pose, exact ground: z=-0.0063 m at 14.52 m",
    "prop_filter_keeps_cap": "F: narrow, yaw 1.0 deg/frame: at-cap px span [249.49, 250.51] m",
    "prop_objects_keep_height": "N: main pole at 4.0 m, lat 0.0: px at GT z 1.114 m rendered at 0.000 m",
    "check_clip015_near_road_bias": "f40/f160 near road 0-15 m: median |dz| 0.114 m, p90 0.489 m, n 96786 (gate 0.08/0.20; gvd/rv-ground-plane)",
    "check_clean_holdout_near_road_bias": "f70: median |dz| 0.074 m passes, p90 0.713 m fails, n 56069",
    "check_clip015_open_road_seam": "f40 main/wide seam median |dz| 5.24 m at [-0.62, 9.88] (0.0 vs 7.50 m)",
    "check_holdout_open_road_seam": "f200 main/wide seam median |dz| 0.328 m at [-1.38, 4.88]",
    "check_clean_holdout_open_road_seam": "f70 main/wide seam median |dz| 5.65 m at [-1.38, 6.62] (0.0 vs 5.11 m)",
    "check_clip015_snap_no_worse_than_off": "f40/f160: main 10-15 m median 0.215 vs 0.142 off; wide 10-15 m 0.542 vs 0.495",
    "check_holdout_snap_no_worse_than_off": "f100/f200: main 10-15 m median 0.173 vs 0.103 off; wide 10-15 m p90 0.696 vs 0.626",
    "check_clean_holdout_snap_no_worse_than_off": "f70: main 5-10 m median 0.073 vs 0.065; wide 0-5 m 0.050 vs 0.048",
}

_REPORT = os.environ.get("RV_PROPS_REPORT", "").strip() in {"1", "true", "yes"}


def _first_line(e: BaseException) -> str:
    msg = str(e).strip().splitlines()
    out = (msg[0] if msg else type(e).__name__)[:400]
    notes = [n.replace("\n", " ") for n in getattr(e, "__notes__", []) if "Falsifying" in n or "prop_" in n]
    return out + (" | " + " ".join(notes)[:600] if notes else "")


def _run_one(fn) -> bool:
    name = fn.__name__
    try:
        if fn() == SKIPPED:
            print(f"SKIP {name}" + (" (known failing; not evaluated here)" if name in KNOWN_FAILING else ""), flush=True)
            return True
    except AssertionError as e:
        if _REPORT:
            print(f"FAIL {name}: {_first_line(e)}", flush=True)
            return True
        if name in KNOWN_FAILING:
            print(f"XFAIL {name}: {_first_line(e)}", flush=True)
            return True
        print(f"FAIL {name}: {_first_line(e)}", flush=True)
        traceback.print_exc()
        return False
    if name in KNOWN_FAILING and not _REPORT:
        print(f"XPASS {name}: passes now; remove it from KNOWN_FAILING ({KNOWN_FAILING[name]})", flush=True)
        return False
    print(f"{'PASS' if _REPORT else 'ok'} {name}", flush=True)
    return True


def _ci_requires_hypothesis() -> bool:
    def _on(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
    return _on("CI") or _on("GITHUB_ACTIONS")


def _run_hypothesis_props_or_skip() -> bool:
    if given is not None:
        return all([_run_one(p) for p in PROPS])
    if _ci_requires_hypothesis():
        raise SystemExit(
            "FAIL hypothesis_properties: hypothesis is not installed under CI "
            f"(CI={os.environ.get('CI')!r}, GITHUB_ACTIONS={os.environ.get('GITHUB_ACTIONS')!r}). "
            "CI must install requirements-dev.txt (hypothesis==6.170.17)."
        )
    skip_check("hypothesis_properties", "hypothesis not installed (dev-only dep; CI installs it)")
    return True


def main() -> None:
    ok = all([_run_one(fn) for fn in CHECKS])
    ok = _run_hypothesis_props_or_skip() and ok
    n_x = len(KNOWN_FAILING)
    if not ok:
        raise SystemExit("test_rv_stable_props: FAILED")
    print(f"test_rv_stable_props: OK ({n_x} known failing)")


if __name__ == "__main__":
    main()


# Pytest entrypoints: a KNOWN_FAILING check passes here while it still fails; XPASS fails.
def _assert_ok(fn) -> None:
    assert _run_one(fn), fn.__name__


def test_checks() -> None:
    for fn in CHECKS:
        _assert_ok(fn)


def test_hypothesis_props() -> None:
    assert _run_hypothesis_props_or_skip()
