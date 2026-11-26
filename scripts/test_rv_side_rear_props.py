#!/usr/bin/env python3
"""Property + fuzz checks for Reality View side / rear cameras (pillarL/R, repeatL/R, rear, + main for
the near front) through the REAL per-camera render path (tools/rv_stable_preview.py ``cam_frame``,
parsed and executed against fake preds, as in scripts/test_rv_stable_props.py on gvd/rv-stable-props).

Thresholds (committed before any gvd/rv-side-rear fix commit exists):
W  road lateral width, per cam in {pillarL, pillarR, repeatL, repeatR, rear, main} and per range bin
   0-5 / 5-15 / 15-30 m: rendered road width minus GT road width, per 1 m longitudinal slice of the
   cam's visible road; median |err| <= 10% of GT width and p90 |err| <= 20% (GVD gate, 2026-11-26).
   Rendered road = valid points from pixels the NET labels road (runtime notion; it is what
   road_snap uses); GT road = GT depth + GT road label on the same cam pixels (eval only).
E  left / right road edge lateral offset vs GT, per cam and bin: REPORT ONLY.
R  rear hole fraction: share of GT road pixels in the rear and rear-quarter chase views with no
   rendered (observed or gap-filled) surface; the ground prior is a guess and counts as a hole.
   <= 2% at 0-15 m, <= 5% at 15-30 m. Rear / rear-quarter view (judgment, the preview has none):
   the preview chase (pose A, 480x270) swung about the ego origin by 180 / +-135 deg.
I  invariants: rendered road never extends past the GT road edge by > 0.3 m at 0-5 m; objects
   above the road keep their height (>= GT z - 0.05 m, > 2 px inside the silhouette); sky stays sky;
   no NaN in any valid depth or rendered point.
F  fuzz: side-cam depth NaN / inf / 0 / negative, empty masks (all sky, no road), extreme extrinsic
   yaw (any yaw incl. +-180 and wrapped values).
H  holdout (Skeptic): every clip gate also runs on >= 2 clip015 frames >= 40 frames from f40/f160
   with the same thresholds; f40/f160 pass + holdout fail = overfit counterexample.
Clip gates need GVD_RV_CLIP015 (clip dir), GVD_RV_CLIP015_PREDS, GVD_RV_CLIP015_GT (GT depth root)
and GVD_RV_CLIP015_GTSEG (GT label root, <cam>_seg/000/%06d.png; road ids GVD_RV_GTSEG_ROAD_IDS,
default 2); they skip visibly without them. Synthetic versions (flat ground, 7 m road, net depth =
GT, net label = the 12x16 class grid majority of the GT label) always run.
GT is eval-only: it never enters cam_frame (pred mode).
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
WIDTH_BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))
WIDTH_MED_FRAC = 0.10
WIDTH_P90_FRAC = 0.20
HOLE_BINS = {(0.0, 15.0): 0.02, (15.0, 30.0): 0.05}
EDGE_OVERSHOOT_M = 0.3   # at 0-5 m
EDGE_NEAR_M = 5.0
OBJ_TOL_M = 0.05
TOL_Z = 1e-3
SLICE_M = 1.0
MIN_SLICE_PTS = 3
ROAD_HALF_M = 3.5        # synthetic road: x in [-3.5, 3.5] (one 7 m carriageway)
GATE_FRAMES = (40, 160)
HOLDOUT_MIN_GAP = 40
HOLDOUT_N = 2
REAR_VIEWS_DEG = (180.0, 135.0, -135.0)

CAM_SPECS = {  # config/cameras.yaml (check_cam_specs_match_config)
    "narrow": {"id": "narrow", "pos_m": [0.03, 1.20, 1.28], "yaw_deg": 0.0, "pitch_deg": -4.0, "fov_v": 21.4, "resolution": [640, 480], "far_m": 800.0},
    "main": {"id": "main", "pos_m": [0.00, 1.20, 1.26], "yaw_deg": 0.0, "pitch_deg": -6.0, "fov_v": 34.0, "resolution": [640, 480], "far_m": 300.0},
    "wide": {"id": "wide", "pos_m": [-0.03, 1.20, 1.24], "yaw_deg": 0.0, "pitch_deg": -8.0, "fov_v": 88.7, "resolution": [640, 360], "far_m": 300.0},
    "pillarL": {"id": "pillarL", "pos_m": [-0.88, 0.40, 1.18], "yaw_deg": -78.0, "pitch_deg": -6.0, "fov_v": 65.3, "resolution": [640, 480], "far_m": 100.0},
    "pillarR": {"id": "pillarR", "pos_m": [0.88, 0.40, 1.18], "yaw_deg": 78.0, "pitch_deg": -6.0, "fov_v": 65.3, "resolution": [640, 480], "far_m": 100.0},
    "repeatL": {"id": "repeatL", "pos_m": [-0.81, 1.25, 0.67], "yaw_deg": -150.0, "pitch_deg": -8.0, "fov_v": 55.4, "resolution": [640, 480], "far_m": 100.0},
    "repeatR": {"id": "repeatR", "pos_m": [0.81, 1.25, 0.67], "yaw_deg": 150.0, "pitch_deg": -8.0, "fov_v": 55.4, "resolution": [640, 480], "far_m": 100.0},
    "rear": {"id": "rear", "pos_m": [0.00, -2.25, 0.78], "yaw_deg": 180.0, "pitch_deg": -10.0, "fov_v": 80.0, "resolution": [640, 360], "far_m": 100.0},
}
SIDE = ("pillarL", "pillarR", "repeatL", "repeatR", "rear")
WIDTH_CAMS = SIDE + ("main",)
ROAD_ID = 2
OBJ_ID = 13
NONROAD_ID = 0


SKIPPED = "skipped"


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


def gt_at_proc(g_full: np.ndarray, pw: int, ph: int) -> np.ndarray:
    """GT (full res) sampled at each processing pixel's centre (no nearest-resize half-pixel shift)."""
    H0, W0 = g_full.shape[:2]
    yi = np.minimum(((np.arange(ph) + 0.5) * H0 / ph).astype(np.int64), H0 - 1)
    xi = np.minimum(((np.arange(pw) + 0.5) * W0 / pw).astype(np.int64), W0 - 1)
    return g_full[yi[:, None], xi[None, :]]


def _settings(n):
    return settings(max_examples=n, deadline=None, derandomize=True, database=None,
                    suppress_health_check=list(HealthCheck), print_blob=False)


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


def sf_sky() -> int:
    return int(pv().ns["SKY"])


def ego_box_filter(pts: np.ndarray) -> np.ndarray:
    """Mirror of the preview main loop's EGO_BOX cut (points inside the ego car are dropped)."""
    hw, y0, y1, z0, z1 = pv().ns["EGO_BOX"]
    inside = (np.abs(pts[:, 0]) <= hw) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
    return ~inside


# ------------------------------------------------------------------ synthetic scene (GT, eval only)
def ground_scene(cam, half=ROAD_HALF_M, centre=0.0):
    """Exact flat-ground depth, GT road label (|x - centre| <= half) and the net's coarse label."""
    g = geom(cam)
    rays = g["rays"]
    t = np.where(g["down"], g["hit"], np.nan)
    X = cam.pos[0] + t * rays[..., 0]
    gt_road = g["down"] & (np.abs(X - centre) <= half)
    fine = np.where(~g["down"], sf_sky(), np.where(gt_road, ROAD_ID, NONROAD_ID))
    depth = np.where(g["down"], g["hit"], 1e4)
    return depth, fine, coarse(fine), gt_road, g


def side_object_scene(cam, kind: str, x0: float, y0: float):
    """Flat ground + one upright face in the plane x = x0 (parallel to the car), y from y0."""
    g = geom(cam)
    rays = g["rays"]
    if kind == "pole":
        L, h_ = 0.15, 2.5
    elif kind == "car":
        L, h_ = 4.2, 1.5
    else:  # curb
        L, h_ = 6.0, 0.15
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(rays[..., 0]) > 1e-9, (x0 - cam.pos[0]) / rays[..., 0], np.inf)
        t = np.where(t > 0, t, np.inf)
        Y = cam.pos[1] + t * rays[..., 1]
        Z = cam.pos[2] + t * rays[..., 2]
    face = np.isfinite(t) & (Y >= y0) & (Y <= y0 + L) & (Z >= 0) & (Z <= h_) & (t < g["hit"])
    gt = np.where(face, t, np.where(g["down"], g["hit"], np.inf))
    fine = np.where(face, OBJ_ID, np.where(~g["down"], sf_sky(), ROAD_ID))
    return gt, fine, face, g


# ------------------------------------------------------------------ measurements
def width_errors(cam, d_r, road_r, d_g, road_g):
    """Per 1 m longitudinal slice (in the cam's own forward axis): (range m, GT width, err, left err, right err).

    Rendered road = points of ``road_r`` pixels at depth ``d_r``; GT road = ``road_g`` at ``d_g``.
    Only points on the ground (|z| <= STATIC_MAX_HEIGHT_M) count. Slices are cut along the ego
    y axis for the side/rear cams' lateral road and widths are measured along x."""
    out = []
    ys, xs = np.nonzero(road_r & np.isfinite(d_r))
    pr = sf.unproject(cam, d_r, ys, xs) if ys.size else np.zeros((0, 3))
    ys, xs = np.nonzero(road_g & np.isfinite(d_g))
    pg = sf.unproject(cam, d_g, ys, xs) if ys.size else np.zeros((0, 3))
    pr = pr[np.abs(pr[:, 2]) <= sf.STATIC_MAX_HEIGHT_M]
    pg = pg[np.abs(pg[:, 2]) <= sf.STATIC_MAX_HEIGHT_M]
    if pg.shape[0] == 0:
        return out
    k_g = np.floor(pg[:, 1] / SLICE_M).astype(np.int64)
    k_r = np.floor(pr[:, 1] / SLICE_M).astype(np.int64)
    for k in np.unique(k_g):
        a = pg[k_g == k]
        b = pr[k_r == k]
        if a.shape[0] < MIN_SLICE_PTS:
            continue
        wg = float(a[:, 0].max() - a[:, 0].min())
        if wg < 0.5:
            continue
        rng = float(np.median(np.hypot(a[:, 0], a[:, 1])))
        if b.shape[0] < MIN_SLICE_PTS:
            out.append((rng, wg, -wg, float("nan"), float("nan")))
            continue
        wr = float(b[:, 0].max() - b[:, 0].min())
        out.append((rng, wg, wr - wg, float(b[:, 0].min() - a[:, 0].min()), float(b[:, 0].max() - a[:, 0].max())))
    return out


def width_stats(rows):
    """{bin: (median |err|/GT, p90 |err|/GT, median left offset, median right offset, n slices)}."""
    out = {}
    for lo, hi in WIDTH_BINS:
        r = [x for x in rows if lo <= x[0] < hi]
        if not r:
            out[(lo, hi)] = (float("nan"), float("nan"), float("nan"), float("nan"), 0)
            continue
        rel = np.array([abs(x[2]) / x[1] for x in r])
        le = np.array([x[3] for x in r])
        ri = np.array([x[4] for x in r])
        out[(lo, hi)] = (float(np.median(rel)), float(np.percentile(rel, 90)),
                         float(np.nanmedian(le)) if np.isfinite(le).any() else float("nan"),
                         float(np.nanmedian(ri)) if np.isfinite(ri).any() else float("nan"), len(r))
    return out


def width_gate(stats_by_cam, label):
    bad = []
    for cid, st_ in stats_by_cam.items():
        for b, (med, p90, _, _, n) in st_.items():
            if n and not (med <= WIDTH_MED_FRAC and p90 <= WIDTH_P90_FRAC):
                bad.append(f"{cid} {b[0]:g}-{b[1]:g} m: median |err| {100 * med:.1f}% p90 {100 * p90:.1f}% (n {n})")
    assert not bad, f"{label} road width: " + "; ".join(bad)


def edge_overshoot(cam, d_r, road_r, gt_half=ROAD_HALF_M, centre=0.0):
    """Max lateral overshoot (m) of rendered road ground points past the GT edge within 0-5 m."""
    ys, xs = np.nonzero(road_r & np.isfinite(d_r))
    if ys.size == 0:
        return 0.0, None
    p = sf.unproject(cam, d_r, ys, xs)
    p = p[(np.abs(p[:, 2]) <= sf.STATIC_MAX_HEIGHT_M) & (np.hypot(p[:, 0], p[:, 1]) <= EDGE_NEAR_M)]
    if p.shape[0] == 0:
        return 0.0, None
    over = np.abs(p[:, 0] - centre) - gt_half
    i = int(np.argmax(over))
    return float(over[i]), p[i].round(3).tolist()


def rear_views():
    n = pv().ns
    d = pv().defaults
    return {deg: sf.chase_camera(n["CHASE_POS"], n["CHASE_PITCH"], n["CHASE_FOV"], int(d["width"]), int(d["height"]),
                                 math.radians(deg)) for deg in REAR_VIEWS_DEG}


def hole_fraction(chase, rend_pts, gt_road_pts):
    """{bin: (hole frac, n GT road px)}: GT road px in ``chase`` (GT road points splatted + gap fill)
    without an observed or filled rendered surface."""
    def paint(pts):
        if pts.shape[0] == 0:
            z = np.zeros((chase.height, chase.width), bool)
            return z, np.zeros(z.shape)
        r = sf.compose(chase, pts, np.full((pts.shape[0], 3), 128, np.uint8), np.full(pts.shape[0], 0.3),
                       sky_rgb=None, ground_rgb=None, fill=True)
        return r.masks["observed"] | r.masks["filled"], r.depth
    gt_m, _ = paint(gt_road_pts)
    rd_m, _ = paint(rend_pts)
    rays = sf._pixel_rays(chase)
    dz = rays[..., 2]
    down = dz < -1e-9
    t = np.where(down, -chase.pos[2] / np.where(down, dz, -1.0), np.nan)
    rng = np.hypot(chase.pos[0] + t * rays[..., 0], chase.pos[1] + t * rays[..., 1])
    out = {}
    for (lo, hi) in HOLE_BINS:
        m = gt_m & down & (rng >= lo) & (rng < hi)
        n = int(m.sum())
        out[(lo, hi)] = (float((m & ~rd_m).sum()) / n if n else 0.0, n)
    return out


def hole_gate(fr_by_view, label):
    bad = []
    for deg, bins in fr_by_view.items():
        for b, (f, n) in bins.items():
            if n and f > HOLE_BINS[b]:
                bad.append(f"view {deg:g} deg {b[0]:g}-{b[1]:g} m: holes {100 * f:.2f}% of {n} GT road px (max {100 * HOLE_BINS[b]:g}%)")
    assert not bad, f"{label} rear holes: " + "; ".join(bad)


def synthetic_frame(cams=None, label_fn=None):
    """Every cam through cam_frame on the synthetic road; returns per-cam dicts."""
    p = pv()
    out = {}
    for cid in cams or p.cam:
        cam = p.cam[cid]
        depth, fine, net, gt_road, g = ground_scene(cam)
        lab = net if label_fn is None else label_fn(fine)
        _, d, v, sky = p.run(cid, depth, lab)
        if cid in p.ns["EGO_MASK"]:
            r0, c0, c1 = p.ns["EGO_MASK"][cid]
            gt_road = gt_road.copy()
            gt_road[int(r0 * cam.height):, int(c0 * cam.width):int(c1 * cam.width)] = False  # hood, not road
        out[cid] = {"cam": cam, "d": d, "v": v, "sky": sky, "net_road": np.asarray(cv2.resize(
            (lab == ROAD_ID).astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0),
            "gt_d": np.where(g["down"], g["hit"], np.nan), "gt_road": gt_road}
    return out


def synthetic_rear_holes():
    fr = synthetic_frame()
    rend, gtp = [], []
    for cid, x in fr.items():
        cam = x["cam"]
        ys, xs = np.nonzero(x["v"])
        p = sf.unproject(cam, x["d"], ys, xs)
        rend.append(p[ego_box_filter(p)])
        ys, xs = np.nonzero(x["gt_road"])
        q = sf.unproject(cam, x["gt_d"], ys, xs)
        gtp.append(q[ego_box_filter(q)])
    rend = np.concatenate(rend)
    gtp = np.concatenate(gtp)
    return {deg: hole_fraction(ch, rend, gtp) for deg, ch in rear_views().items()}


# ------------------------------------------------------------------ deterministic checks
def check_cam_specs_match_config() -> None:
    try:
        import yaml
    except ImportError:
        skip_check("cam_specs_match_config", "pyyaml not installed")
        return SKIPPED
    by = {c["id"]: c for c in yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text(encoding="utf-8"))["cameras"]}
    for cid, s in CAM_SPECS.items():
        c = by[cid]
        assert [float(v) for v in c["pos_m"]] == s["pos_m"] and float(c.get("yaw_deg") or 0.0) == s["yaw_deg"], cid
        assert float(c["pitch_deg"]) == s["pitch_deg"] and float(c["fov_v"]) == s["fov_v"], cid
        assert list(c["live_res"]) == s["resolution"] and float(c["far_m"]) == s["far_m"], cid


def check_harness_side_cams() -> None:
    p = pv()
    for cid in SIDE:
        assert cid in p.cam, cid
        depth, fine, net, gt_road, g = ground_scene(p.cam[cid])
        _, d, v, _ = p.run(cid, depth, net)
        assert v.any() and gt_road.any(), cid


def check_synthetic_width_exact_labels() -> None:
    """W with net depth = GT AND net label = GT label: whatever fails here is the render path itself."""
    _synthetic_width(lambda fine: fine, "synthetic exact labels")


def check_synthetic_width() -> None:
    """W on the synthetic road (exact depth, net label = 12x16 majority of GT)."""
    _synthetic_width(None, "synthetic 12x16 labels")


def _synthetic_width(label_fn, label) -> None:
    fr = synthetic_frame(WIDTH_CAMS, label_fn)
    stats = {cid: width_stats(width_errors(x["cam"], x["d"], x["v"] & x["net_road"], x["gt_d"], x["gt_road"]))
             for cid, x in fr.items()}
    print(f"{label} width (median|err|/GT, p90, left off m, right off m, n):",
          {c: {f"{lo:g}-{hi:g}": tuple(round(v, 3) if isinstance(v, float) else v for v in t) for (lo, hi), t in s.items()}
           for c, s in stats.items()}, flush=True)
    width_gate(stats, label)


def check_synthetic_edge_overshoot_exact_labels() -> None:
    """I with net label = GT label: the render path alone."""
    _edge(lambda fine: fine)


def check_synthetic_edge_overshoot() -> None:
    """I: rendered road <= 0.3 m past the GT edge at 0-5 m (synthetic, net label = class grid majority)."""
    _edge(None)


def _edge(label_fn) -> None:
    fr = synthetic_frame(WIDTH_CAMS, label_fn)
    bad = []
    for cid, x in fr.items():
        o, where = edge_overshoot(x["cam"], x["d"], x["v"] & x["net_road"])
        if o > EDGE_OVERSHOOT_M:
            bad.append(f"{cid}: {o:.2f} m past the GT edge at {where}")
    assert not bad, "; ".join(bad)


def check_synthetic_rear_holes() -> None:
    h = synthetic_rear_holes()
    print("synthetic rear holes:", {deg: {f"{lo:g}-{hi:g}": (round(100 * f, 2), n) for (lo, hi), (f, n) in b.items()} for deg, b in h.items()}, flush=True)
    hole_gate(h, "synthetic")


def check_no_nan_and_sky() -> None:
    p = pv()
    for cid in SIDE:
        cam = p.cam[cid]
        H, W = cam.height, cam.width
        for fill in (np.nan, np.inf, -np.inf, 0.0, -3.0, 1e300):
            depth = np.full((H, W), fill)
            for sem in (np.full((H, W), sf_sky()), np.full((H, W), NONROAD_ID), np.full((H, W), ROAD_ID)):
                _, d, v, sky = p.run(cid, depth, sem)
                assert not (v & sky).any() and np.all(np.isfinite(d[v])), (cid, fill, int(sem[0, 0]))


def check_side_objects_keep_height_pins() -> None:
    """I: a car parked 2.5 m to the right at y 0..4.2 m, a pole and a curb, seen by pillarR / repeatR."""
    p = pv()
    bad = []
    for cid in ("pillarR", "repeatR", "pillarL"):
        for kind, x0, y0 in (("car", 2.5, 0.0), ("pole", 3.0, 1.0), ("curb", 3.5, -3.0), ("car", -2.5, -1.0)):
            cam = p.cam[cid]
            gt, fine, face, _ = side_object_scene(cam, kind, x0, y0)
            _, d, v, _ = p.run(cid, np.where(np.isfinite(gt), gt, 1e4), coarse(fine))
            inner = sf._erode(face, 2) & v
            if not inner.any():
                continue
            lost = heights(cam, gt)[inner] - heights(cam, d)[inner]
            if lost.max() > OBJ_TOL_M:
                bad.append(f"{cid} {kind}@x{x0:g},y{y0:g}: lost {lost.max():.3f} m on {int((lost > OBJ_TOL_M).sum())}/{int(inner.sum())} px")
    assert not bad, "; ".join(bad)


# ------------------------------------------------------------------ clip015 (real data, optional)
def clip_paths():
    keys = ("GVD_RV_CLIP015", "GVD_RV_CLIP015_PREDS", "GVD_RV_CLIP015_GT", "GVD_RV_CLIP015_GTSEG")
    v = [os.environ.get(k, "") for k in keys]
    if all(v) and Path(v[0], "cameras.json").is_file() and Path(v[1]).is_file() and Path(v[2]).is_dir() and Path(v[3]).is_dir():
        return tuple(Path(x) for x in v)
    return None


def _road_ids():
    return tuple(int(x) for x in os.environ.get("GVD_RV_GTSEG_ROAD_IDS", "2").split(",") if x.strip())


def holdout_frames(P_files, n: int = 4) -> tuple:
    env = os.environ.get("GVD_RV_HOLDOUT_FRAMES", "").strip()
    fr = [int(x) for x in env.split(",") if x.strip()] if env else sorted(
        {int(k.split("/")[1]) for k in P_files if k.startswith("main/") and k.endswith("/0/d")})
    fr = [f for f in fr if all(abs(f - g) >= HOLDOUT_MIN_GAP for g in GATE_FRAMES)]
    if len(fr) > n:
        fr = [fr[int(round(i * (len(fr) - 1) / (n - 1)))] for i in range(n)]
    return tuple(fr)


def clip_measure(clip, preds, gt_root, seg_root, frames):
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    rows = {}
    over = {}
    hole_acc = {deg: {b: [0, 0] for b in HOLE_BINS} for deg in REAR_VIEWS_DEG}
    for f in frames:
        rend, gtp = [], []
        for cid in [c for c in p.cam if c in CAM_SPECS]:
            fr = p.run(cid, None, None, idx=f)
            if fr is None:
                continue
            _, d, v, sky = fr
            cam = p.cam[cid]
            c = cams[cid]
            W0, _ = c["resolution"]
            pw, ph = p.ns["proc_res"](c)
            g = cv2.imread(str(gt_root / f"{cid}_depth" / "000" / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
            s = cv2.imread(str(seg_root / f"{cid}_seg" / "000" / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
            if g is None or s is None:
                continue
            g = gt_at_proc(g.astype(np.float32) / 100.0, pw, ph)
            gtd = np.where((g > 0) & (g < min(c["far_m"] * sf.GT_NO_HIT_FRAC, 655.0)), g, np.nan)
            s = gt_at_proc(s if s.ndim == 2 else s[..., 0], pw, ph)
            gt_road = np.isin(s, _road_ids()) & np.isfinite(gtd)
            net_road = np.zeros((ph, pw), bool)
            sx = pw / W0
            k = 0
            while f"{cid}/{f}/{k}/s" in P.files:
                x0, cw = P[f"{cid}/{f}/{k}/x0"]
                a0, a1 = int(round(x0 * sx)), int(round((x0 + cw) * sx))
                net_road[:, a0:a1] |= cv2.resize(np.isin(P[f"{cid}/{f}/{k}/s"], sf.ROAD_CLASS_IDS).astype(np.uint8),
                                                 (a1 - a0, ph), interpolation=cv2.INTER_NEAREST) > 0
                k += 1
            ys, xs = np.nonzero(v)
            q = sf.unproject(cam, d, ys, xs)
            assert np.all(np.isfinite(q)), (cid, f, "NaN in rendered points")
            rend.append(q[ego_box_filter(q)])
            ys, xs = np.nonzero(gt_road)
            q = sf.unproject(cam, gtd, ys, xs)
            gtp.append(q[ego_box_filter(q)])
            if cid in WIDTH_CAMS:
                rows.setdefault(cid, []).extend(width_errors(cam, d, v & net_road, gtd, gt_road))
                # edge overshoot vs the GT road edge in each 1 m slice (GT width from the GT points)
                ys, xs = np.nonzero(v & net_road)
                pr = sf.unproject(cam, d, ys, xs)
                pr = pr[(np.abs(pr[:, 2]) <= sf.STATIC_MAX_HEIGHT_M) & (np.hypot(pr[:, 0], pr[:, 1]) <= EDGE_NEAR_M)]
                pg = gtp[-1]
                pg = pg[np.hypot(pg[:, 0], pg[:, 1]) <= EDGE_NEAR_M + 1.0]
                if pr.shape[0] and pg.shape[0]:
                    kg = np.floor(pg[:, 1] / SLICE_M)
                    for kk in np.unique(np.floor(pr[:, 1] / SLICE_M)):
                        a = pg[kg == kk]
                        b = pr[np.floor(pr[:, 1] / SLICE_M) == kk]
                        if a.shape[0] >= MIN_SLICE_PTS:
                            o = max(a[:, 0].min() - b[:, 0].min(), b[:, 0].max() - a[:, 0].max())
                            over[cid] = max(over.get(cid, -1e9), float(o))
        if rend and gtp:
            R, G = np.concatenate(rend), np.concatenate(gtp)
            for deg, ch in rear_views().items():
                for b, (fh, n) in hole_fraction(ch, R, G).items():
                    hole_acc[deg][b][0] += fh * n
                    hole_acc[deg][b][1] += n
    width = {cid: width_stats(r) for cid, r in rows.items()}
    holes = {deg: {b: ((v[0] / v[1]) if v[1] else 0.0, v[1]) for b, v in bins.items()} for deg, bins in hole_acc.items()}
    return {"width": width, "overshoot": over, "holes": holes}


_CLIP: dict = {}


def _clip(frames):
    if frames not in _CLIP:
        _CLIP[frames] = clip_measure(*clip_paths(), frames)
    return _CLIP[frames]


def _clip_holdout():
    fr = holdout_frames(np.load(clip_paths()[1]).files)
    assert len(fr) >= HOLDOUT_N, f"holdout: need >= {HOLDOUT_N} frames >= {HOLDOUT_MIN_GAP} from {GATE_FRAMES}, have {fr}"
    return fr


def _clip_check(name, frames_fn, gate):
    if clip_paths() is None:
        skip_check(name, "clip015 + GT depth + GT labels not on this host (GVD_RV_CLIP015, _PREDS, _GT, _GTSEG)")
        return SKIPPED
    frames = frames_fn()
    m = _clip(frames)
    print(f"{name} frames {frames}: width {m['width']} overshoot {m['overshoot']} holes {m['holes']}", flush=True)
    gate(m, f"clip015 {frames}")


def check_clip015_width():
    return _clip_check("clip015_width", lambda: GATE_FRAMES, lambda m, l: width_gate(m["width"], l))


def check_holdout_width():
    return _clip_check("holdout_width", _clip_holdout, lambda m, l: width_gate(m["width"], l))


def check_clip015_rear_holes():
    return _clip_check("clip015_rear_holes", lambda: GATE_FRAMES, lambda m, l: hole_gate(m["holes"], l))


def check_holdout_rear_holes():
    return _clip_check("holdout_rear_holes", _clip_holdout, lambda m, l: hole_gate(m["holes"], l))


def _over_gate(m, l):
    bad = {c: round(o, 3) for c, o in m["overshoot"].items() if o > EDGE_OVERSHOOT_M}
    assert not bad, f"{l} road past GT edge at 0-5 m (m): {bad}"


def check_clip015_edge_overshoot():
    return _clip_check("clip015_edge_overshoot", lambda: GATE_FRAMES, _over_gate)


def check_holdout_edge_overshoot():
    return _clip_check("holdout_edge_overshoot", _clip_holdout, _over_gate)


def check_clip_harness_selftest() -> None:
    """The clip path on a synthetic clip (flat ground, 7 m road, preds = GT, labels = GT at full res)."""
    tmp = Path(tempfile.mkdtemp(prefix="rvsideclip-"))
    atexit.register(shutil.rmtree, tmp, True)
    cams = [dict(CAM_SPECS[c]) for c in CAM_SPECS]
    (tmp / "cameras.json").write_text(json.dumps({"cameras": cams}))
    store = {}
    for c in cams:
        cid = c["id"]
        W0, H0 = c["resolution"]
        cam = pv().cam[cid]
        g = geom(cam)
        fs = (pv().ns["fy_full"](c) * 192.0 / H0) / pv().ns["FY_REF"]
        camf = camera_from_angles(cid, c["pos_m"], c["yaw_deg"], c["pitch_deg"], c["fov_v"], W0, H0, near_m=0.05, far_m=400.0)
        gf = geom(camf)
        Xf = camf.pos[0] + np.where(gf["down"], gf["hit"], np.nan) * gf["rays"][..., 0]
        segf = np.where(gf["down"] & (np.abs(Xf) <= ROAD_HALF_M), ROAD_ID, NONROAD_ID).astype(np.uint8)
        depth, fine, net, gt_road, _ = ground_scene(cam)
        for f in GATE_FRAMES:
            for sub in (f"raw/{cid}/000", f"gt/{cid}_depth/000", f"seg/{cid}_seg/000"):
                (tmp / sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp / f"raw/{cid}/000/{f:06d}.png"), np.full((H0, W0, 3), 128, np.uint8))
            store[f"{cid}/{f}/0/d"] = depth / fs
            store[f"{cid}/{f}/0/x0"] = np.array([0, W0])
            store[f"{cid}/{f}/0/s"] = fine
            cv2.imwrite(str(tmp / f"gt/{cid}_depth/000/{f:06d}.png"), np.round(np.where(gf["down"], np.minimum(gf["hit"], 600.0), 0.0) * 100).astype(np.uint16))
            cv2.imwrite(str(tmp / f"seg/{cid}_seg/000/{f:06d}.png"), segf)
    np.savez(tmp / "preds.npz", **store)
    m = clip_measure(tmp, tmp / "preds.npz", tmp / "gt", tmp / "seg", GATE_FRAMES)
    for cid in ("pillarL", "pillarR", "rear"):
        t = m["width"][cid][(0.0, 5.0)]   # 0-5 m only: past ~10 m the rear's own edge-cut dip shows up
        assert t[4] > 0 and t[0] <= 0.05, (cid, t)
    assert holdout_frames(["main/0/0/d", "main/40/0/d", "main/100/0/d", "main/250/0/d"]) == (0, 100, 250)


# ------------------------------------------------------------------ hypothesis props
def _build_props():
    if given is None:
        return []
    special = st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -1.0, 1e-9, 0.3, 1e6, 1e300])

    @_settings(40)
    @given(cid=st.sampled_from(SIDE), fill=st.one_of(special, st.floats(0.01, 1e4)), frac=st.floats(0.0, 1.0),
           sky_r=st.floats(0.0, 1.0), road=st.booleans(), seed=st.integers(0, 2**16))
    def prop_side_depth_fuzz(cid, fill, frac, sky_r, road, seed):
        """F: side-cam depth with NaN/inf/0/negative, any sky / road masks (incl. empty): no NaN out,
        sky never valid, valid depth > 0 and <= the cam's cap."""
        p = pv()
        cam = p.cam[cid]
        H, W = cam.height, cam.width
        rng = np.random.default_rng(seed)
        g = geom(cam)
        depth = np.where(rng.random((H, W)) < frac, fill, np.where(g["down"], g["hit"], 1e4))
        ys = np.mgrid[0:H, 0:W][0]
        sky = ys < H * sky_r
        sem = np.where(sky, sf_sky(), ROAD_ID if road else NONROAD_ID)
        _, d, v, skym = p.run(cid, depth, sem)
        cap = sf.depth_cap_m(CAM_SPECS[cid]["far_m"])
        assert not (v & skym).any()
        assert np.all(np.isfinite(d[v])) and np.all(d[v] > 0) and np.all(d[v] <= cap)

    @_settings(30)
    @given(cid=st.sampled_from(SIDE), yaw=st.floats(-720.0, 720.0), pitch=st.floats(-30.0, 10.0))
    def prop_extreme_yaw(cid, yaw, pitch):
        """F: any extrinsic yaw (wrapped, +-180, multiple turns): exact flat ground gives no NaN and no
        point below the road within 0-5 m; road-label points stay inside the GT road + 0.3 m at 0-5 m
        when the net label equals the GT label."""
        cams = {k: dict(v) for k, v in CAM_SPECS.items()}
        cams[cid]["yaw_deg"] = yaw
        cams[cid]["pitch_deg"] = pitch
        p = Preview(cams=cams)
        cam = p.cam[cid]
        g = geom(cam)
        if not g["down"].any():
            return
        rays = g["rays"]
        X = cam.pos[0] + np.where(g["down"], g["hit"], np.nan) * rays[..., 0]
        gt_road = g["down"] & (np.abs(X) <= ROAD_HALF_M)
        fine = np.where(~g["down"], int(p.ns["SKY"]), np.where(gt_road, ROAD_ID, NONROAD_ID))
        _, d, v, _ = p.run(cid, np.where(g["down"], g["hit"], 1e4), fine)
        ys, xs = np.nonzero(v)
        q = sf.unproject(cam, d, ys, xs)
        assert np.all(np.isfinite(q))
        near = g["R"][ys, xs] <= EDGE_NEAR_M
        assert not near.any() or q[near, 2].min() >= -TOL_Z, float(q[near, 2].min())
        o, where = edge_overshoot(cam, d, v & gt_road)
        assert o <= EDGE_OVERSHOOT_M, (o, where)

    @_settings(30)
    @given(cid=st.sampled_from(("pillarL", "pillarR", "repeatL", "repeatR")), kind=st.sampled_from(("pole", "car", "curb")),
           x0=st.floats(1.5, 6.0), y0=st.floats(-8.0, 4.0))
    def prop_side_objects_keep_height(cid, kind, x0, y0):
        """I: a pole / parked car / curb beside the car keeps its height (net label = 12x16 majority)."""
        p = pv()
        cam = p.cam[cid]
        sgn = -1.0 if cid.endswith("L") else 1.0
        gt, fine, face, _ = side_object_scene(cam, kind, sgn * x0, y0)
        _, d, v, _ = p.run(cid, np.where(np.isfinite(gt), gt, 1e4), coarse(fine))
        inner = sf._erode(face, 2) & v
        if not inner.any():
            return
        lost = heights(cam, gt)[inner] - heights(cam, d)[inner]
        assert lost.max() <= OBJ_TOL_M, f"{cid} {kind} x0 {sgn * x0:.2f} y0 {y0:.2f}: lost {lost.max():.3f} m"

    return [prop_side_depth_fuzz, prop_extreme_yaw, prop_side_objects_keep_height]


PROPS = _build_props()

CHECKS = [
    check_cam_specs_match_config,
    check_harness_side_cams,
    check_clip_harness_selftest,
    check_no_nan_and_sky,
    check_synthetic_width_exact_labels,
    check_synthetic_width,
    check_synthetic_edge_overshoot_exact_labels,
    check_synthetic_edge_overshoot,
    check_synthetic_rear_holes,
    check_side_objects_keep_height_pins,
    check_clip015_width,
    check_holdout_width,
    check_clip015_edge_overshoot,
    check_holdout_edge_overshoot,
    check_clip015_rear_holes,
    check_holdout_rear_holes,
]

# name -> counterexample on 783fedd (filled from real runs after the thresholds commit).
KNOWN_FAILING: dict[str, str] = {}

_REPORT = os.environ.get("RV_PROPS_REPORT", "").strip() in {"1", "true", "yes"}


def main() -> None:
    ok = all([_run_one(fn) for fn in CHECKS])
    ok = _run_hypothesis_props_or_skip() and ok
    if not ok:
        raise SystemExit("test_rv_side_rear_props: FAILED")
    print(f"test_rv_side_rear_props: OK ({len(KNOWN_FAILING)} known failing)")


if __name__ == "__main__":
    main()


def test_checks() -> None:
    for fn in CHECKS:
        assert _run_one(fn), fn.__name__


def test_hypothesis_props() -> None:
    assert _run_hypothesis_props_or_skip()
