#!/usr/bin/env python3
"""Property + fuzz checks for Reality View side / rear cameras (pillarL/R, repeatL/R, rear, + main for
the near front) through the REAL per-camera render path (tools/rv_stable_preview.py ``cam_frame``,
parsed and executed against fake preds, as in scripts/test_rv_stable_props.py on gvd/rv-stable-props).

Thresholds (committed before any gvd/rv-side-rear fix commit exists; clip015 has GT depth only,
no GT seg — GVD 2026-11-26, superseding the seg-based width gate):
W  (a) edge-position error of pred-road pixels vs GT depth at the same pixel:
   |x_rendered - x_GTdepth|, gated vs GT lateral distance |x_GTdepth|: median |err| <= 10% and
   p90 |err| <= 20% of median |x_GTdepth| in bins 0-5 and 5-15 m (15-30 m is report-only);
   plus edge overshoot (|x_r| - |x_gt|) <= 0.3 m at 0-5 m.
   (b) snap misplacement, all of {wide, pillarL, pillarR, repeatL, repeatR, rear}, GATED:
   <= 1% of snapped pixels with GT z > 0.30 m at 0-15 m, and <= 3% at 15-30 m.
   Reason (Tracer / Skeptic): hard 0 unreachable: one-sided z_pred>+0.30 gate catches only
   46-100% of off-ground px (logs/rv_side4.txt); GT cm quantization + pose jitter push true
   road just over 0.30 m past 15 m; hence <=1% 0-15 m, <=3% 15-30 m (Skeptic).
   Denominator (GATED, Skeptic ruling 2026-11-26): snapped px that are still DRAWN, i.e. valid in
   cam_frame after the runtime yield (rv_stable_preview L115) and ego-mask (L118) drops, with
   finite GT depth, counted per cam and range bin (range = hypot(x,y) of that pixel's GT
   unprojection). Numerator: those px with GT z > 0.30 m at that px's GT depth (unchanged).
   Limits unchanged. Reason: hood/yield px are dropped before render with snap on or off, so
   the earlier gate counted px Michael never sees. All-snapped px (incl. yield/hood-dropped)
   is printed every run as INFO, both as k/n and %, n shown at both ranges. The mask comes
   from the runtime path itself (cam_frame's own `valid`); no mask array is copied into
   this test. check_runtime_mask_growth fails if the yield / ego masked px count on dev
   frames f40/f110/f160 grows beyond what 783fedd masks (recorded below).
   p99 |GT z| of drawn snapped px is printed as a road-noise number, info only (no noise margin
   in the gate).
   Note: net mask too wide is not measured: GT road = net px within 1 m of plane.
   The old seg-based width gate is kept but skips unless GVD_RV_CLIP015_GTSEG is set
   (clip015 has no GT seg).
E  left / right edge offset vs GT depth, per cam and bin: REPORT ONLY.
R  rear hole fraction on Michael's still, pose B, frame 110: share of GT-ground pixels
   (|z| <= STATIC_MAX_HEIGHT_M; no GT seg) in the pose-B chase with no rendered (observed or
   filled) surface; ground prior counts as a hole. <= 2% at 0-15 m, <= 5% at 15-30 m.
   Pose B is not in tools/rv_stable_preview.py on this head (only pose A). Source:
   /workspace/rv-preview/v2/abl/poseB/render_stats_B_full.json chase_pose
   pos (0, -9, 4.5) m, pitch -16 deg, fov_v 70; still rv_v3_still_poseB_frame110.png.
S  snap on is no worse than snap off for 0-15 m |z_render - z_GT| (median and p90), per
   side cam and rear (and wide). Catches a rig-wide z0 that pushes side edges in.
P  footprint coverage: on snapped ground px at 0-15 m, row-to-row 3D spacing <= the
   preview footprint() clipped size (6*d/fy) for >= 95% of those px, per cam.
B  behind-car strip holes, GATED (not report-only): GT-road cells 0.25 m, |x|<=12, y<0, r<=30,
   |z|<1, net road at GT depth, outside the ego box; cells built from rows f-10..f+10 moved
   into the current frame by VO; hole = the cell centre projects onto a chase pixel with no
   splat (3x3 dilate, no gap fill; 960x540 rigid chase; memory on: v2 accumulate / stable
   ground grid). Pose A strip hole frac <= v2 measured on 10893ee (this run): f40 0.1456
   (n=158), f160 0.0000 (n=159). Historian published 0.146/0.000 (same measurement, rounded).
   Pose B f110 strip <= v2 measured 0.1058 (n=1503); Historian did not publish f110.
   Dev frames only (f40/f160/f110); f60/f130 untouched. Far-hole gate (pose B, <=2% at 0-15 m
   and <=5% at 15-30 m, fill on, |z|<=STATIC_MAX_HEIGHT_M) is kept.
A  width (gate a: edge-position median <=10% / p90 <=20% of median |x_gt| in 0-5 and 5-15 m,
   plus overshoot <=0.3 m at 0-5 m) and holes (<=2% / <=5%) are also gated in the RENDER
   (chase) frame, identically to the capture-frame gates. Pose A and pose B, dev frames.
N  net mask too wide is not measured: GT road = net px within 1 m of plane.
I  snapped pixels are not exempt from the depth-edge cut (tools/rv_stable_preview.py
   ``cut & ~sky & ~snapped``): an off-plane road-labelled pixel at a depth edge must still
   be cut (moved to the 5x5 far-side). NEW vs a56f514 (no preview/snap).
I  objects above the road keep their height; a wall/car/curb top labelled road stays put
   (snap must not move it outward; it keeps height and depth); sky stays sky; no NaN.
F  fuzz: side-cam depth NaN/inf/0/negative, empty masks, extreme extrinsic yaw.
H  holdout: the same gates on clip015 f100 and f200 (nearest sparse side/rear index;
   Tracer printed numbers on these while tuning) AND on clean holdout f70 (nobody has
   printed this frame). A pass on f40/f160 + fail on holdout = overfit counterexample.
Default clip paths (override with GVD_RV_CLIP015 / _PREDS / _GT; read-only):
  /workspace/rv-preview/v2src/clip015, /workspace/rv-preview/v2/preds_clip015.npz,
  /workspace/rv-preview/v2src/gt_eval_only. Never write there. Never touch
  /workspace/rv-preview/drift_eval/clip_005 (locked holdout).
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
TRACER_HOLDOUT = (100, 200)   # Tracer already printed numbers on these while tuning the gate
CLEAN_HOLDOUT = (70,)         # nobody has printed this frame (clean holdout)
HOLDOUT_MIN_GAP = 40
HOLDOUT_N = 2
REAR_VIEWS_DEG = (180.0, 135.0, -135.0)
# (b) snap misplacement (GVD final gate, 2026-11-26). No noise-margin constant.
SNAP_BAD_Z = 0.30
SNAP_BAD_NEAR = 0.01          # <= 1% of snapped px with GT z > 0.30 m, at 0-15 m
SNAP_BAD_FAR = 0.03           # <= 3% at 15-30 m
SNAP_PLACE_CAMS = ("wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
EDGE_MED_FRAC = 0.10
EDGE_P90_FRAC = 0.20
FOOTPRINT_MIN_FRAC = 0.95
# Pose B: not in tools/rv_stable_preview.py (only pose A = CHASE_POS). From
# /workspace/rv-preview/v2/abl/poseB/render_stats_B_full.json "chase_pose".
POSE_B_POS = (0.0, -9.0, 4.5)
POSE_B_PITCH = -16.0
POSE_B_FOV = 70.0
POSE_B_FRAME = 110
# Behind-car strip gates = v2 10893ee measured here (rv-side-evidence/v2_verify_10893ee.json).
# Historian published pose A 0.146/0.000; we measured 0.1456 (n=158) / 0.0000 (n=159) — same
# method, their rounding. Pose B f110 was not published by Historian; we measured 0.1058 (n=1503).
V2_POSE_A_STRIP = {40: 0.1456, 160: 0.0000}
V2_POSE_B_STRIP = {110: 0.1058}
V2_POSE_A_REAR_HOLES = {40: 0.146, 160: 0.000}  # Historian rounded, kept as a published-ref note
B017D12_POSE_A_REAR_HOLES = {40: 0.279, 160: 0.371}
HIST_783FEDD_POSE_A_REAR_HOLES = {40: 0.222, 160: 0.277}
BEHIND_X_M = 12.0
BEHIND_R_M = 30.0
BEHIND_Z_M = 1.0
STRIP_W, STRIP_H = 960, 540
STRIP_WIN = 10
STRIP_CAMS = ("pillarL", "pillarR", "repeatL", "repeatR", "rear", "main", "wide")  # not narrow
DEFAULT_CLIP = Path("/workspace/rv-preview/v2src/clip015")
DEFAULT_PREDS = Path("/workspace/rv-preview/v2/preds_clip015.npz")
DEFAULT_GT = Path("/workspace/rv-preview/v2src/gt_eval_only")

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


_WANT_FN = {"fy_full", "proc_res", "cam_frame", "footprint", "phase_dx"}


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
        self.snap_hook = {"mask": None}

        class _SHook:
            def __init__(self, real, hook):
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_hook", hook)

            def __getattr__(self, name):
                return getattr(object.__getattribute__(self, "_real"), name)

            def road_snap(self, *a, **k):
                out, m = object.__getattribute__(self, "_real").road_snap(*a, **k)
                object.__getattribute__(self, "_hook")["mask"] = m
                return out, m

        self.ns["S"] = _SHook(sf, self.snap_hook)
        self.last_snap = None

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
        self.snap_hook["mask"] = None
        with np.errstate(all="ignore"):
            out = self.ns["cam_frame"](cid, idx)
        self.last_snap = self.snap_hook.get("mask")
        return out

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
    """(clip, preds, gt) if present. GTSEG is optional and is not required for the default gates."""
    clip = Path(os.environ["GVD_RV_CLIP015"]) if os.environ.get("GVD_RV_CLIP015") else DEFAULT_CLIP
    preds = Path(os.environ["GVD_RV_CLIP015_PREDS"]) if os.environ.get("GVD_RV_CLIP015_PREDS") else DEFAULT_PREDS
    gt = Path(os.environ["GVD_RV_CLIP015_GT"]) if os.environ.get("GVD_RV_CLIP015_GT") else DEFAULT_GT
    if clip.joinpath("cameras.json").is_file() and preds.is_file() and gt.is_dir():
        return clip, preds, gt
    return None


def clip_seg_path():
    p = os.environ.get("GVD_RV_CLIP015_GTSEG", "")
    return Path(p) if p and Path(p).is_dir() else None


def pred_frames(P, cid):
    return sorted({int(k.split("/")[1]) for k in P.files if k.startswith(cid + "/") and k.endswith("/d")})


def nearest_idx(frames, target):
    return min(frames, key=lambda i: (abs(i - target), i)) if frames else None


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
        clip, preds, gt = clip_paths()
        _CLIP[frames] = clip_measure(clip, preds, gt, clip_seg_path(), frames)
    return _CLIP[frames]


def _clip_holdout():
    return TRACER_HOLDOUT


def _clip_clean_holdout():
    return CLEAN_HOLDOUT


def _clip_check(name, frames_fn, gate):
    if clip_paths() is None:
        skip_check(name, "clip015 + GT depth not on this host")
        return SKIPPED
    if clip_seg_path() is None:
        skip_check(name, "no GT seg on clip015 (GVD_RV_CLIP015_GTSEG not set); width is gated by edge-position vs GT depth instead")
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



# ------------------------------------------------------------------ clip015 GT-depth gates (no GT seg)
_DCLIP: dict = {}


def _load_gt_depth(gt_root: Path, cid: str, idx: int, c: dict, pw: int, ph: int):
    g = cv2.imread(str(gt_root / f"{cid}_depth" / "000" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    if g is None:
        return None
    g = gt_at_proc(g.astype(np.float32) / 100.0, pw, ph)
    return np.where((g > 0) & (g < min(c["far_m"] * sf.GT_NO_HIT_FRAC, 655.0)), g, np.nan)


def _net_road(P, cid, idx, W0, pw, ph):
    road = np.zeros((ph, pw), bool)
    sx = pw / W0
    k = 0
    while f"{cid}/{idx}/{k}/s" in P.files:
        x0, cw = P[f"{cid}/{idx}/{k}/x0"]
        a0, a1 = int(round(x0 * sx)), int(round((x0 + cw) * sx))
        road[:, a0:a1] |= cv2.resize(np.isin(P[f"{cid}/{idx}/{k}/s"], sf.ROAD_CLASS_IDS).astype(np.uint8),
                                     (a1 - a0, ph), interpolation=cv2.INTER_NEAREST) > 0
        k += 1
    return road


def clip_depth_metrics(frames, overrides=None):
    """Per-cam measurements from pred-road / snap vs GT depth (no GT seg). Sparse cams use nearest i."""
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(overrides, root=clip, P=P, cams=cams)
    avail = {cid: pred_frames(P, cid) for cid in p.cam}
    snap_rows = []          # (cid, target, used, lo, hi, n_drawn, n_bad_drawn, p99_abs_z_drawn, n_all_snapped, n_bad_all)
    edge_rows = []          # (cid, target, lo, hi, med_err, p90_err, med_lat, n, max_overshoot)
    dz_rows = []            # (cid, target, med_|dz|, p90_|dz|, n) in 0-15 m
    foot_rows = []          # (cid, target, frac_ok, n)
    used = {}
    for target in frames:
        for cid in list(p.cam):
            idx = nearest_idx(avail.get(cid) or [], target)
            if idx is None:
                continue
            used.setdefault(cid, {})[target] = idx
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            _, d, v, sky = fr
            snap = p.last_snap
            if snap is None:
                snap = np.zeros(d.shape, bool)
            cam = p.cam[cid]
            c = cams[cid]
            W0, _ = c["resolution"]
            pw, ph = p.ns["proc_res"](c)
            gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
            if gtd is None:
                continue
            net = _net_road(P, cid, idx, W0, pw, ph)
            # (b) snap misplacement. GATED denominator = snapped px still DRAWN: valid in cam_frame, i.e.
            # after the runtime yield (L115) and ego-mask (L118) drops (Skeptic ruling 2026-11-26).
            # `v` comes from the same cam_frame call; no mask is copied into the test.
            # INFO denominator = all snapped px with finite GT depth.
            ys, xs = np.nonzero(snap & np.isfinite(gtd))
            if ys.size:
                pg = sf.unproject(cam, gtd, ys, xs)
                rng = np.hypot(pg[:, 0], pg[:, 1])
                drawn_px = v[ys, xs]
            else:
                pg = np.zeros((0, 3))
                rng = np.zeros(0)
                drawn_px = np.zeros(0, bool)
            for lo, hi in ((0.0, 15.0), (15.0, 30.0)):
                ka = (rng >= lo) & (rng < hi)
                kd = ka & drawn_px
                n_all = int(ka.sum())
                n_bad_all = int((pg[ka, 2] > SNAP_BAD_Z).sum()) if n_all else 0
                n = int(kd.sum())
                n_bad = int((pg[kd, 2] > SNAP_BAD_Z).sum()) if n else 0
                p99 = float(np.percentile(np.abs(pg[kd, 2]), 99)) if n else float("nan")
                snap_rows.append((cid, target, idx, lo, hi, n, n_bad, p99, n_all, n_bad_all))
            # (a) edge position of pred-road pixels vs GT depth
            ys, xs = np.nonzero(v & net & np.isfinite(gtd) & np.isfinite(d))
            if ys.size:
                pr = sf.unproject(cam, d, ys, xs)
                pg = sf.unproject(cam, gtd, ys, xs)
                rng = np.hypot(pg[:, 0], pg[:, 1])
                err = np.abs(pr[:, 0] - pg[:, 0])
                over = np.abs(pr[:, 0]) - np.abs(pg[:, 0])
                for lo, hi in ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0)):
                    k = (rng >= lo) & (rng < hi)
                    if not k.any():
                        continue
                    edge_rows.append((cid, target, lo, hi,
                                      float(np.median(err[k])), float(np.percentile(err[k], 90)),
                                      float(np.median(np.abs(pg[k, 0]))), int(k.sum()),
                                      float(over[k].max()) if k.any() else 0.0))
            # snap-on vs off uses |dz| at 0-15 m (pred-road or any valid with GT)
            ys, xs = np.nonzero(v & np.isfinite(gtd) & np.isfinite(d))
            if ys.size:
                pr = sf.unproject(cam, d, ys, xs)
                pg = sf.unproject(cam, gtd, ys, xs)
                k = np.hypot(pg[:, 0], pg[:, 1]) < 15.0
                if k.any():
                    adz = np.abs(pr[k, 2] - pg[k, 2])
                    dz_rows.append((cid, target, float(np.median(adz)), float(np.percentile(adz, 90)), int(k.sum())))
            # footprint coverage on snapped ground px at 0-15 m
            if snap.any() and "footprint" in p.ns:
                fp = p.ns["footprint"](cam, d)
                ys, xs = np.nonzero(snap & np.isfinite(d) & np.isfinite(gtd))
                if ys.size:
                    pr = sf.unproject(cam, d, ys, xs)
                    pg = sf.unproject(cam, gtd, ys, xs)
                    near = np.hypot(pg[:, 0], pg[:, 1]) < 15.0
                    # row-to-row: same column, next row also snapped with finite depth
                    ok = 0
                    n = 0
                    H, W = d.shape
                    for y, x, keep in zip(ys, xs, near):
                        if not keep or y + 1 >= H or not snap[y + 1, x] or not np.isfinite(d[y + 1, x]):
                            continue
                        p0 = sf.unproject(cam, d, np.array([y]), np.array([x]))[0]
                        p1 = sf.unproject(cam, d, np.array([y + 1]), np.array([x]))[0]
                        n += 1
                        if np.linalg.norm(p1 - p0) <= fp[y, x] + 1e-9:
                            ok += 1
                    if n:
                        foot_rows.append((cid, target, ok / n, n))
    return {"snap": snap_rows, "edge": edge_rows, "dz": dz_rows, "foot": foot_rows, "used": used}


def _dclip(frames, overrides=None):
    key = (tuple(frames), json.dumps(overrides or {}, sort_keys=True))
    if key not in _DCLIP:
        _DCLIP[key] = clip_depth_metrics(frames, overrides)
    return _DCLIP[key]


def _need_clip(name):
    if clip_paths() is None:
        skip_check(name, "clip015 + GT depth not on this host")
        return False
    return True


def _pct(k, n):
    return "n/a (n=0)" if n == 0 else f"{100 * k / n:.2f}%"


def print_snap_both(rows, label):
    """Both numbers every run, per cam / range bin / frame, k/n and %, n always shown:
    DRAWN-only (GATED) and all-snapped (INFO)."""
    print(f"{label} gate (b) per cam/bin/frame: DRAWN-only (GATED) | all-snapped (INFO); k/n = snapped px with "
          f"GT z>{SNAP_BAD_Z} m at that px's GT depth / snapped px with finite GT depth "
          f"(drawn = valid in cam_frame after runtime yield L115 + ego mask L118)", flush=True)
    pooled = {}
    for cid, tgt, idx, lo, hi, n, n_bad, p99, n_all, n_bad_all in rows:
        print(f"  {cid:8s} f{tgt}->i{idx} {lo:g}-{hi:g} m: DRAWN(gated) {n_bad}/{n} {_pct(n_bad, n)} | "
              f"all-snapped(info) {n_bad_all}/{n_all} {_pct(n_bad_all, n_all)} | p99|z| drawn {p99:.3f} m", flush=True)
        a = pooled.setdefault((cid, lo, hi), [0, 0, 0, 0])
        a[0] += n_bad; a[1] += n; a[2] += n_bad_all; a[3] += n_all
    for (cid, lo, hi), (k, n, ka, na) in sorted(pooled.items()):
        print(f"  {cid:8s} POOLED {lo:g}-{hi:g} m: DRAWN(gated) {k}/{n} {_pct(k, n)} | "
              f"all-snapped(info) {ka}/{na} {_pct(ka, na)}", flush=True)


def snap_misplace_gate(m, label):
    """<= 1% DRAWN snapped px with GT z > 0.30 m at 0-15 m; <= 3% at 15-30 m. Per cam, side/rear/wide.

    Denominator (GATED): snapped px that are still drawn after the runtime yield/ego masks (valid in
    cam_frame, rv_stable_preview L115/L118), with finite GT depth, per cam and range bin
    (range = hypot(x, y) of that pixel's GT unprojection).
    Numerator: those px whose GT z > 0.30 m at that same pixel's GT depth.
    All-snapped (incl. yield/hood-dropped px) is printed as INFO only.
    Reason: hood/yield px are dropped before render with snap on or off, so the earlier gate counted
    px Michael never sees (Skeptic ruling).
    """
    print(f"{label} snap misplace gate (b): denom = snapped px still DRAWN (valid in cam_frame after runtime "
          f"yield/ego masks) with finite GT depth, per cam/bin (range = hypot(x,y) of that px's GT unprojection); "
          f"numer = those px with GT z > {SNAP_BAD_Z} m at that px's GT depth. Not all road px, not z_pred. "
          f"All-snapped is INFO only. p99 |GT z| is road noise, not a margin. "
          f"Note: net mask too wide is not measured: GT road = net px within 1 m of plane.",
          flush=True)
    print_snap_both(m["snap"], label)
    print(f"{label} nearest frames:", m["used"], flush=True)
    acc = {}  # (cid, lo, hi) -> [n_bad, n]
    for cid, tgt, idx, lo, hi, n, n_bad, p99, n_all, n_bad_all in m["snap"]:
        if cid not in SNAP_PLACE_CAMS:
            continue
        a = acc.setdefault((cid, lo, hi), [0, 0])
        a[0] += n_bad
        a[1] += n
    bad = []
    for (cid, lo, hi), (n_bad, n) in sorted(acc.items()):
        if n == 0:
            continue
        frac = n_bad / n
        lim = SNAP_BAD_NEAR if hi <= 15.0 else SNAP_BAD_FAR
        if frac > lim:
            bad.append(f"{cid} {lo:g}-{hi:g} m: {100 * frac:.2f}% ({n_bad}/{n}) drawn snapped px with GT z > {SNAP_BAD_Z} m (max {100 * lim:g}%)")
    assert not bad, f"{label} snap misplace: " + "; ".join(bad)


def edge_position_gate(m, label, report_far=True):
    print(f"{label} edge position (med err, p90, med |x_gt|, n, max overshoot):",
          [(r[0], r[1], f"{r[3]:g}-{r[4]:g}", tuple(round(x, 3) if isinstance(x, float) else x for x in r[5:]))
           for r in m["edge"]], flush=True)
    acc = {}
    for cid, tgt, lo, hi, med, p90, lat, n, over in m["edge"]:
        acc.setdefault((cid, lo, hi), []).append((med, p90, lat, n, over))
    bad, over_bad = [], []
    for (cid, lo, hi), rows in acc.items():
        n = sum(r[3] for r in rows)
        # re-weight is approximate; use worst frame in the set (each row is one frame)
        for med, p90, lat, n_i, over in rows:
            if n_i == 0 or lat < 0.25:
                continue
            if hi <= 15.0 and (med > EDGE_MED_FRAC * lat + 1e-9 or p90 > EDGE_P90_FRAC * lat + 1e-9):
                bad.append(f"{cid} {lo:g}-{hi:g} m: median {med:.3f} m p90 {p90:.3f} m vs 10%/20% of |x_gt| {lat:.3f} m")
            if lo == 0.0 and hi == 5.0 and over > EDGE_OVERSHOOT_M:
                over_bad.append(f"{cid} 0-5 m overshoot {over:.3f} m")
    assert not bad, f"{label} edge position: " + "; ".join(bad)
    assert not over_bad, f"{label} edge overshoot: " + "; ".join(over_bad)


def snap_no_worse_gate(on, off, label):
    print(f"{label} |dz| 0-15 m snap ON:", on["dz"], "OFF:", off["dz"], flush=True)
    def by_cam(rows):
        d = {}
        for cid, tgt, med, p90, n in rows:
            d.setdefault(cid, []).append((med, p90, n))
        return d
    A, B = by_cam(on["dz"]), by_cam(off["dz"])
    bad = []
    for cid in SNAP_PLACE_CAMS:
        if cid not in A or cid not in B:
            continue
        # compare per-frame paired by order (same targets)
        for (m1, p1, n1), (m0, p0, n0) in zip(A[cid], B[cid]):
            if n1 == 0 or n0 == 0:
                continue
            if m1 > m0 + 1e-3 or p1 > p0 + 1e-3:
                bad.append(f"{cid}: snap on median {m1:.3f}/p90 {p1:.3f} vs off {m0:.3f}/{p0:.3f}")
    assert not bad, f"{label} snap on worse than off (0-15 m |dz|): " + "; ".join(bad)


def footprint_gate(m, label):
    print(f"{label} footprint coverage:", m["foot"], flush=True)
    acc = {}
    for cid, tgt, frac, n in m["foot"]:
        a = acc.setdefault(cid, [0, 0])
        a[0] += frac * n
        a[1] += n
    bad = []
    for cid, (ok, n) in acc.items():
        if n and ok / n < FOOTPRINT_MIN_FRAC:
            bad.append(f"{cid}: {100 * ok / n:.1f}% of {n} snapped 0-15 m px have row spacing <= footprint (min {100 * FOOTPRINT_MIN_FRAC:g}%)")
    assert not bad, f"{label} footprint: " + "; ".join(bad)


def pose_b_chase():
    return sf.chase_camera(POSE_B_POS, POSE_B_PITCH, POSE_B_FOV,
                           int(pv().defaults["width"]), int(pv().defaults["height"]), 0.0)


def clip_poseB_holes(target=POSE_B_FRAME):
    """Rear-hole fraction in pose B at frame 110 (nearest sparse index). GT-ground = |z| <= STATIC_MAX_HEIGHT_M."""
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    rend, gtp = [], []
    used = {}
    for cid in p.cam:
        idx = nearest_idx(pred_frames(P, cid), target)
        if idx is None:
            continue
        used[cid] = idx
        fr = p.run(cid, None, None, idx=idx)
        if fr is None:
            continue
        _, d, v, _ = fr
        cam = p.cam[cid]
        c = cams[cid]
        pw, ph = p.ns["proc_res"](c)
        gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
        ys, xs = np.nonzero(v & np.isfinite(d))
        if ys.size:
            q = sf.unproject(cam, d, ys, xs)
            rend.append(q[ego_box_filter(q)])
        if gtd is not None:
            ys, xs = np.nonzero(np.isfinite(gtd))
            q = sf.unproject(cam, gtd, ys, xs)
            q = q[np.abs(q[:, 2]) <= sf.STATIC_MAX_HEIGHT_M]
            if q.size:
                gtp.append(q[ego_box_filter(q)])
    chase = pose_b_chase()
    holes = hole_fraction(chase, np.concatenate(rend) if rend else np.zeros((0, 3)),
                          np.concatenate(gtp) if gtp else np.zeros((0, 3)))
    return {"holes": {0.0: holes}, "used": used, "chase": "poseB"}  # one view, keyed like hole_gate expects? 


# Runtime yield / ego mask counts on 783fedd (dev frames), per frame and cam, measured by running the
# runtime cam_frame three ways on the same frame/pred index: normal, ownership off (a.ownership=0),
# EGO_MASK emptied. yield px = valid-with-ownership-off & ~valid; ego px = valid-with-EGO_MASK-off & ~valid.
# Recorded from this test on 783fedd (rv-side-evidence/mask_counts_783fedd.log). Growth fails.
MASK_FRAMES = (40, 110, 160)
MASK_LIMITS: dict = {   # {(frame, cam): (yield_px, ego_px)}, 783fedd, dev frames (cams not listed drop 0)
    (40, "pillarL"): (0, 0), (40, "pillarR"): (0, 0), (40, "wide"): (3998, 9216),
    (110, "pillarL"): (0, 0), (110, "pillarR"): (0, 0), (110, "wide"): (2814, 9216),
    (160, "pillarL"): (0, 0), (160, "pillarR"): (0, 0), (160, "wide"): (2814, 9216),
}


def runtime_mask_counts(frames=MASK_FRAMES):
    """{(frame, cid): (yield_px, ego_px, idx)} from the runtime cam_frame; also checks the dropped px lie
    only in the runtime's own YIELD / EGO_MASK regions (subset)."""
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    out, stray = {}, []
    for target in frames:
        for cid in p.cam:
            if cid not in p.ns.get("YIELD_TO", {}) and cid not in p.ns.get("EGO_MASK", {}):
                continue
            idx = nearest_idx(pred_frames(P, cid), target)
            if idx is None:
                continue
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            v = fr[2]
            own, ego_mask = p.a.ownership, p.ns["EGO_MASK"]
            try:
                p.a.ownership = 0
                v_no_yield = p.run(cid, None, None, idx=idx)[2]
            finally:
                p.a.ownership = own
            try:
                p.ns["EGO_MASK"] = {}
                v_no_ego = p.run(cid, None, None, idx=idx)[2]
            finally:
                p.ns["EGO_MASK"] = ego_mask
            y_px = v_no_yield & ~v
            e_px = v_no_ego & ~v
            ph, pw = v.shape
            region_y = np.asarray(p.ns["YIELD"][cid], bool) if cid in p.ns.get("YIELD", {}) else np.zeros(v.shape, bool)
            region_e = np.zeros(v.shape, bool)
            if cid in ego_mask:
                r0, c0, c1 = ego_mask[cid]
                region_e[int(r0 * ph):, int(c0 * pw):int(c1 * pw)] = True
            if (y_px & ~region_y).any() or (e_px & ~region_e).any():
                stray.append(f"f{target} {cid}: {(y_px & ~region_y).sum()} yield-dropped px outside YIELD, "
                             f"{(e_px & ~region_e).sum()} ego-dropped px outside EGO_MASK")
            out[(target, cid)] = (int(y_px.sum()), int(e_px.sum()), idx)
    return out, stray


def check_runtime_mask_growth() -> None:
    """The runtime yield / ego masks must not drop more px than 783fedd on dev frames f40/f110/f160.

    A mask that grows hides more snapped px from the drawn-only denominator of gate (b) (it shrinks the
    denominator), so a fix could pass the gate by masking more. Counts come from the runtime cam_frame
    (L115 / L118) on the same frame; limits are the 783fedd counts. Subset check: dropped px lie inside
    the runtime's own YIELD / EGO_MASK regions.
    """
    if not _need_clip("runtime_mask_growth"):
        return SKIPPED
    if not MASK_LIMITS:
        raise AssertionError("MASK_LIMITS not recorded")
    cur, stray = runtime_mask_counts()
    print("runtime mask px (yield, ego) per frame/cam [limit]:",
          {f"f{f}/{c}": (v[0], v[1], "limit", MASK_LIMITS.get((f, c))) for (f, c), v in sorted(cur.items())}, flush=True)
    bad = []
    for (f, c), (y, e, idx) in sorted(cur.items()):
        ly, le = MASK_LIMITS.get((f, c), (0, 0))
        if y > ly or e > le:
            bad.append(f"f{f} {c}: yield {y} (max {ly}), ego {e} (max {le})")
    assert not bad, "runtime masks grew beyond 783fedd: " + "; ".join(bad)
    assert not stray, "masked px outside the runtime's own YIELD/EGO_MASK region: " + "; ".join(stray)


def check_clip015_snap_misplace() -> None:
    if not _need_clip("clip015_snap_misplace"):
        return SKIPPED
    snap_misplace_gate(_dclip(GATE_FRAMES), "f40/f160")


def check_holdout_snap_misplace() -> None:
    if not _need_clip("holdout_snap_misplace"):
        return SKIPPED
    snap_misplace_gate(_dclip(TRACER_HOLDOUT), "holdout f100/f200 (Tracer printed these while tuning)")


def check_clean_holdout_snap_misplace() -> None:
    if not _need_clip("clean_holdout_snap_misplace"):
        return SKIPPED
    snap_misplace_gate(_dclip(CLEAN_HOLDOUT), "clean holdout f70")


def check_clip015_edge_position() -> None:
    if not _need_clip("clip015_edge_position"):
        return SKIPPED
    edge_position_gate(_dclip(GATE_FRAMES), "f40/f160")


def check_holdout_edge_position() -> None:
    if not _need_clip("holdout_edge_position"):
        return SKIPPED
    edge_position_gate(_dclip(TRACER_HOLDOUT), "holdout f100/f200")


def check_clean_holdout_edge_position() -> None:
    if not _need_clip("clean_holdout_edge_position"):
        return SKIPPED
    edge_position_gate(_dclip(CLEAN_HOLDOUT), "clean holdout f70")


def check_clip015_snap_no_worse_than_off() -> None:
    if not _need_clip("clip015_snap_no_worse_than_off"):
        return SKIPPED
    snap_no_worse_gate(_dclip(GATE_FRAMES), _dclip(GATE_FRAMES, {"road_snap": 0}), "f40/f160")


def check_holdout_snap_no_worse_than_off() -> None:
    if not _need_clip("holdout_snap_no_worse_than_off"):
        return SKIPPED
    snap_no_worse_gate(_dclip(TRACER_HOLDOUT), _dclip(TRACER_HOLDOUT, {"road_snap": 0}), "holdout f100/f200")


def check_clean_holdout_snap_no_worse_than_off() -> None:
    if not _need_clip("clean_holdout_snap_no_worse_than_off"):
        return SKIPPED
    snap_no_worse_gate(_dclip(CLEAN_HOLDOUT), _dclip(CLEAN_HOLDOUT, {"road_snap": 0}), "clean holdout f70")


def check_clip015_footprint() -> None:
    if not _need_clip("clip015_footprint"):
        return SKIPPED
    footprint_gate(_dclip(GATE_FRAMES), "f40/f160")


def check_holdout_footprint() -> None:
    if not _need_clip("holdout_footprint"):
        return SKIPPED
    footprint_gate(_dclip(TRACER_HOLDOUT), "holdout f100/f200")


def check_clean_holdout_footprint() -> None:
    if not _need_clip("clean_holdout_footprint"):
        return SKIPPED
    footprint_gate(_dclip(CLEAN_HOLDOUT), "clean holdout f70")


def check_clip015_rear_holes_poseB() -> None:
    """Rear holes on pose B at frame 110 (Michael's still). Same 2% / 5% gates."""
    if not _need_clip("clip015_rear_holes_poseB"):
        return SKIPPED
    m = clip_poseB_holes()
    print(f"pose B frame {POSE_B_FRAME} nearest:", m["used"], "holes:", m["holes"], flush=True)
    hole_gate(m["holes"], f"pose B f{POSE_B_FRAME}")



def check_snapped_pixels_not_exempt_from_edge_cut() -> None:
    """Snapped pixels must still take the depth-edge cut.

    tools/rv_stable_preview.py does ``depth = np.where(cut & ~sky & ~snapped, far_side, depth)``,
    so a snapped pixel at a cut is left at the ground-plane hit. An off-plane road-labelled
    pixel at a depth edge must still be cut. NEW vs a56f514 (no rv_stable_preview / road_snap).
    """
    p = pv()
    cam = p.cam["wide"]
    g = geom(cam)
    H, W = g["hit"].shape
    ys, xs = np.mgrid[0:H, 0:W]
    # Left: road-labelled, 2.5x too far (snaps onto the plane). Right: not road, also 2.5x
    # (clamp leaves a depth step vs the snapped hit). The 5x5 far-side of snapped edge
    # pixels is the unsnapped far depth; they must still be cut, not left at the plane hit.
    raw = np.where(g["down"], g["hit"] * 2.5, 1e4)
    road = g["down"] & (xs < W // 2)
    sem = np.where(~g["down"], sf_sky(), np.where(road, ROAD_ID, NONROAD_ID))
    _, d, v, sky = p.run("wide", raw, sem)
    snap = p.last_snap
    assert snap is not None and snap.any() and snap[:, : W // 2].any()
    # Replay snap+clamp, then the cut, to identify snapped pixels that sit on a depth edge.
    pre, snapped = sf.road_snap(cam, raw, (sem == ROAD_ID) & (sem != sf_sky()), max_hit_m=sf.ROAD_SNAP_MAX_M)
    pre, _ = sf.ground_clamp(cam, pre, margin=float(p.a.clamp_margin or 1.0))
    inv = 1.0 / np.maximum(pre, 1e-3)
    gy, gx = np.gradient(inv)
    cut = np.hypot(gx, gy) > 0.12 * inv
    far_side = cv2.dilate(pre.astype(np.float32), np.ones((5, 5), np.uint8))
    need = cut & ~sky & snapped & (np.abs(far_side - pre) > 1e-3)
    assert need.any(), "fixture must place a snapped pixel on a depth edge"
    kept = need & (np.abs(d - pre) <= 1e-3)
    n_need, n_kept = int(need.sum()), int(kept.sum())
    assert not kept.any(), (
        f"{n_kept}/{n_need} snapped px at a depth edge were exempt from the cut "
        f"(tools/rv_stable_preview.py: `cut & ~sky & ~snapped`); NEW vs a56f514"
    )


def pose_a_chase():
    n = pv().ns
    d = pv().defaults
    return sf.chase_camera(n["CHASE_POS"], n["CHASE_PITCH"], n["CHASE_FOV"],
                           int(d["width"]), int(d["height"]), 0.0)


def behind_car_strip_frac(chase, rend_pts, gt_pts):
    """Hole fraction of behind-car GT-road cells (0.25 m, |x|<=12, y<0, r<=30, |z|<1).

    Hole = the cell centre projects onto a chase pixel with no splat (3x3 dilate, no gap fill).
    Current-frame points only (no ground-grid memory, no f-10..f+10 window).
    Returns (hole_frac, n_visible_cells).
    """
    g = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    if g.shape[0] == 0:
        return float("nan"), 0
    sel = (np.abs(g[:, 0]) <= BEHIND_X_M) & (g[:, 1] < 0.0) & (np.hypot(g[:, 0], g[:, 1]) <= BEHIND_R_M) & (np.abs(g[:, 2]) < BEHIND_Z_M)
    g = g[sel]
    if g.shape[0] == 0:
        return float("nan"), 0
    ix = np.floor(g[:, 0] / sf.GRID_CELL_M).astype(np.int64)
    iy = np.floor(g[:, 1] / sf.GRID_CELL_M).astype(np.int64)
    keys = np.unique(np.stack([ix, iy], 1), axis=0)
    centres = (keys + 0.5) * sf.GRID_CELL_M
    zc = np.zeros(len(centres))
    r = sf.compose(chase, np.asarray(rend_pts).reshape(-1, 3) if len(rend_pts) else np.zeros((0, 3)),
                   np.full((max(len(rend_pts), 0), 3), 128, np.uint8) if len(rend_pts) else np.zeros((0, 3), np.uint8),
                   np.full(len(rend_pts), 0.3) if len(rend_pts) else np.zeros(0),
                   sky_rgb=None, ground_rgb=None, fill=False)
    obs = cv2.dilate(r.masks["observed"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    u, v, z = sf.project(chase, np.c_[centres, zc])
    vis = np.isfinite(z) & (z > chase.near_m) & (u >= 0) & (u < chase.width) & (v >= 0) & (v < chase.height)
    if not vis.any():
        return float("nan"), 0
    ui, vi = np.floor(u[vis]).astype(np.int64), np.floor(v[vis]).astype(np.int64)
    ui = np.clip(ui, 0, chase.width - 1)
    vi = np.clip(vi, 0, chase.height - 1)
    n = int(vis.sum())
    return float((~obs[vi, ui]).mean()), n


def clip_behind_car_strip(targets, poses):
    """Load clip frames (nearest sparse index) and measure the behind-car strip on each pose."""
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    out = {}
    for target in targets:
        rend, gtp = [], []
        used = {}
        for cid in p.cam:
            idx = nearest_idx(pred_frames(P, cid), target)
            if idx is None:
                continue
            used[cid] = idx
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            _, d, v, sky = fr
            cam = p.cam[cid]
            c = cams[cid]
            pw, ph = p.ns["proc_res"](c)
            W0, _ = c["resolution"]
            gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
            net = _net_road(P, cid, idx, W0, pw, ph)
            ys, xs = np.nonzero(v & np.isfinite(d))
            if ys.size:
                q = sf.unproject(cam, d, ys, xs)
                rend.append(q[ego_box_filter(q)])
            if gtd is not None:
                ys, xs = np.nonzero(net & np.isfinite(gtd))
                if ys.size:
                    q = sf.unproject(cam, gtd, ys, xs)
                    q = q[np.abs(q[:, 2]) < BEHIND_Z_M]
                    if q.size:
                        gtp.append(q[ego_box_filter(q)])
        R = np.concatenate(rend) if rend else np.zeros((0, 3))
        G = np.concatenate(gtp) if gtp else np.zeros((0, 3))
        out[target] = {"used": used}
        for name, ch in poses:
            frac, n = behind_car_strip_frac(ch, R, G)
            out[target][name] = (frac, n)
    return out


def strip_chase(name: str):
    """960x540 rigid chase used for the v2-comparable behind-car strip."""
    if name == "poseA":
        pos, pitch, fov = (0.0, -4.0, 2.6), -12.0, 60.0
    elif name == "poseB":
        pos, pitch, fov = POSE_B_POS, POSE_B_PITCH, POSE_B_FOV
    else:
        raise ValueError(name)
    return sf.chase_camera(pos, pitch, fov, STRIP_W, STRIP_H, 0.0)


def behind_car_strip_frac_hires(chase, rend_pts, gt_pts):
    """Same as behind_car_strip_frac but splat radius matches the 960-wide preview (max(2, W//120)=8)."""
    g = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    if g.shape[0] == 0:
        return float("nan"), 0
    sel = ((np.abs(g[:, 0]) <= BEHIND_X_M) & (g[:, 1] < 0.0) & (np.hypot(g[:, 0], g[:, 1]) <= BEHIND_R_M)
           & (np.abs(g[:, 2]) < BEHIND_Z_M))
    g = g[sel]
    if g.shape[0] == 0:
        return float("nan"), 0
    ix = np.floor(g[:, 0] / sf.GRID_CELL_M).astype(np.int64)
    iy = np.floor(g[:, 1] / sf.GRID_CELL_M).astype(np.int64)
    keys, inv = np.unique(np.stack([ix, iy], 1), axis=0, return_inverse=True)
    zc = np.zeros(len(keys))
    np.add.at(zc, inv, g[:, 2])
    zc /= np.bincount(inv, minlength=len(keys))
    centres = np.c_[(keys + 0.5) * sf.GRID_CELL_M, zc]
    rend = np.asarray(rend_pts, np.float64).reshape(-1, 3)
    n_r = len(rend)
    if n_r == 0:
        obs = np.zeros((chase.height, chase.width), bool)
    else:
        r = sf.splat(chase, rend, np.full((n_r, 3), 128, np.uint8), np.full(n_r, 0.3),
                     splat_scale=1.0, max_radius_px=max(2, chase.width // 120))
        obs = r[2] if isinstance(r, tuple) else r.masks["observed"]
    obs = cv2.dilate(np.asarray(obs).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    u, v, z = sf.project(chase, centres)
    vis = np.isfinite(z) & (z > chase.near_m) & (u >= 0) & (u < chase.width) & (v >= 0) & (v < chase.height)
    if not vis.any():
        return float("nan"), 0
    ui = np.clip(np.floor(u[vis]).astype(np.int64), 0, chase.width - 1)
    vi = np.clip(np.floor(v[vis]).astype(np.int64), 0, chase.height - 1)
    n = int(vis.sum())
    return float((~obs[vi, ui]).mean()), n


def _gt_road_row(p, P, gt_root, cams, idx, present):
    """GT road points (net class, |z|<1, outside ego) in that row's camera/ego frame. Eval only."""
    out = []
    for cid in STRIP_CAMS:
        if cid not in p.cam or cid not in present:
            continue
        c = cams[cid]
        pw, ph = p.ns["proc_res"](c)
        W0, _ = c["resolution"]
        gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
        if gtd is None:
            continue
        if not any(f"{cid}/{idx}/{k}/s" in P.files for k in range(8)):
            continue
        net = _net_road(P, cid, idx, W0, pw, ph)
        ys, xs = np.nonzero(net & np.isfinite(gtd) & (gtd > 0.5) & (gtd < 200))
        if not ys.size:
            continue
        q = sf.unproject(p.cam[cid], gtd, ys, xs)
        q = q[np.abs(q[:, 2]) < BEHIND_Z_M]
        if q.size:
            out.append(q[ego_box_filter(q)])
    return np.concatenate(out) if out else np.zeros((0, 3))


def clip_stable_timeline(targets, win=STRIP_WIN):
    """Walk clip015 with preview VO + CamDepthFilter + GroundGrid (memory).

    Returns per target: cur/mem point clouds in the current ego frame, GT-window
    points (f+/-win, transformed in), and current-only GT (held captures, for
    render-frame width/holes). Dev frames only; does not touch f60/f130 unless
    they fall in a requested window (callers must not request them).
    """
    if any(t in (60, 130) for t in targets):
        raise ValueError("f60/f130 are reserved holdouts and must not be requested")
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    rows = [json.loads(l) for l in (clip / "state.jsonl").read_text().splitlines() if l.strip()]
    P = np.load(preds)
    prev = Preview(root=clip, P=P, cams=cams)
    phase_dx = prev.ns["phase_dx"]
    footprint = prev.ns["footprint"]
    filters = {cid: sf.CamDepthFilter() for cid in prev.cam}
    grid = sf.GroundGrid()
    latest = {}  # cid -> (pts, cols, foot, pose)
    odo = [0.0, 0.0, 0.0]
    prev_band = None
    prev_t = None
    FXM = prev.ns["fy_full"](cams["main"]) * 256 / 640
    want = set(int(t) for t in targets)
    max_i = max(want) + int(win)
    gtwin = {}
    out = {}
    for row in rows:
        i = int(row["i"])
        t = float(row["t"])
        if i > max_i:
            break
        main_png = clip / "raw" / "main" / "000" / f"{i:06d}.png"
        band = cv2.resize(cv2.imread(str(main_png), cv2.IMREAD_GRAYSCALE), (256, 192),
                          interpolation=cv2.INTER_AREA)[:80].astype(np.float64)
        if prev_band is not None:
            dt = max(0.0, t - prev_t)
            dyaw = math.atan2(phase_dx(prev_band, band), FXM)
            mid = odo[2] + 0.5 * dyaw
            dist = float(row["veh"]["wheelspeed"]) * dt
            odo = [odo[0] - math.sin(mid) * dist, odo[1] + math.cos(mid) * dist, odo[2] + dyaw]
        prev_band, prev_t = band, t
        pose = tuple(odo)
        present = set((row.get("cams") or {}).keys())
        if any(abs(i - w) <= win for w in want):
            gtwin[i] = (_gt_road_row(prev, P, gt_root, cams, i, present), pose)
        if i > max(want):
            continue
        for cid in prev.cam:
            if cid not in present:
                continue
            if not (clip / "raw" / cid / "000" / f"{i:06d}.png").is_file():
                continue
            fr = prev.run(cid, None, None, idx=i)
            if fr is None:
                continue
            rgb, depth, valid, sky = fr
            cam = prev.cam[cid]
            if bool(prev.a.depth_filter):
                conf = np.clip(1.0 / (1.0 + depth / 20.0), 0.2, 1.0)
                depth, _ = filters[cid].update(cam, depth, valid, pose, t, conf=np.where(valid, conf, 0.0))
            ys, xs = np.nonzero(valid)
            if not ys.size:
                continue
            pts = sf.unproject(cam, depth, ys, xs)
            cols = rgb[ys, xs] if rgb.ndim == 3 else np.full((ys.size, 3), 128, np.uint8)
            foot = footprint(cam, depth)[ys, xs]
            keep = ego_box_filter(pts)
            pts, cols, foot = pts[keep], cols[keep], foot[keep]
            latest[cid] = (pts, cols, foot, pose)
            st = pts[:, 2] <= sf.STATIC_MAX_HEIGHT_M
            if st.any():
                wpts = sf.transform_points(pts[st], pose, (0.0, 0.0, 0.0))
                dist = np.hypot(pts[st, 0], pts[st, 1])
                grid.add(wpts, cols[st].astype(np.float64), 1.0 / (1.0 + dist / 15.0), t)
        if i not in want:
            continue
        parts_p, parts_c, parts_f, gt_now = [], [], [], []
        used = {}
        for cid, (pts, cols, foot, ps) in latest.items():
            q = sf.transform_points(pts, ps, pose)
            parts_p.append(q)
            parts_c.append(cols)
            parts_f.append(foot)
            used[cid] = "held"
        cur = np.concatenate(parts_p) if parts_p else np.zeros((0, 3))
        gp, gc = grid.live(t, pose[:2])
        if gp.shape[0]:
            gl = sf.transform_points(gp, (0.0, 0.0, 0.0), pose)
            mem = np.concatenate([cur, gl]) if cur.size else gl
            mem_c = np.concatenate(parts_c + [gc.astype(np.uint8)]) if parts_c else gc.astype(np.uint8)
            mem_f = np.concatenate(parts_f + [np.full(gl.shape[0], sf.GRID_CELL_M * 1.3)]) if parts_f else np.full(gl.shape[0], sf.GRID_CELL_M * 1.3)
        else:
            mem, mem_c, mem_f = cur, (np.concatenate(parts_c) if parts_c else np.zeros((0, 3), np.uint8)), (np.concatenate(parts_f) if parts_f else np.zeros(0))
        Gs = [sf.transform_points(pts, pj, pose) for j, (pts, pj) in gtwin.items()
              if abs(j - i) <= win and pts.shape[0]]
        G = np.concatenate(Gs) if Gs else np.zeros((0, 3))
        # current-held GT (no window) for render-frame width/holes
        Gcur = []
        for cid, (pts, cols, foot, ps) in latest.items():
            # GT at the last capture we have for this cam is not stored; use this row if present else skip
            pass
        out[i] = {"pose": pose, "cur": cur, "mem": mem, "mem_cols": mem_c, "mem_foot": mem_f,
                  "gt_window": G, "used": used, "cols": (np.concatenate(parts_c) if parts_c else np.zeros((0, 3), np.uint8)),
                  "foot": (np.concatenate(parts_f) if parts_f else np.zeros(0))}
    # second pass: current-held GT using each cam's last raw/pred that we can load at the target
    # (re-derive from gtwin at the exact target row only, plus any held rows — approximate:
    #  use all gtwin rows? No: capture-identical render uses held current points vs GT at those
    #  captures. We stored only window points. For render-frame we gather GT at the target row
    #  plus whatever is in latest by re-reading at target from _gt_road_row if present, else empty
    #  from other cams. Good enough: use gt_window for strip; for render-frame use G at exact
    #  target plus unprojected current cam_frame? The render-frame helpers below take cur + gt.
    return out


def measure_behind_car_strip(tl):
    """tl from clip_stable_timeline. Returns {frame: {poseA/poseB: (frac, n)}} using MEMORY points."""
    out = {}
    for f, rec in tl.items():
        out[f] = {}
        for name in ("poseA", "poseB"):
            out[f][name] = behind_car_strip_frac_hires(strip_chase(name), rec["mem"], rec["gt_window"])
    return out


def check_behind_car_strip_holes() -> None:
    """Gated behind-car strip: pose A f40/f160 and pose B f110 <= v2 measured (10893ee, this run)."""
    if not _need_clip("behind_car_strip_holes"):
        return SKIPPED
    targets = (40, 160, POSE_B_FRAME)
    tl = clip_stable_timeline(targets)
    m = measure_behind_car_strip(tl)
    print("behind-car strip (mem, 960x540, f+/-10 window, 3x3 dilate, no fill). "
          "v2 10893ee measured here: pose A f40 0.1456 (n=158), f160 0.0000 (n=159) "
          "(Historian published 0.146/0.000, same method, rounded); "
          "pose B f110 0.1058 (n=1503), not published by Historian. "
          "Historian 783fedd pose A (their run) 0.222/0.277.", flush=True)
    bad = []
    for f, rec in m.items():
        bits = [f"{k} {rec[k][0]:.4f} (n={rec[k][1]})" for k in ("poseA", "poseB") if k in rec]
        print(f"  f{f}: " + ", ".join(bits), flush=True)
        if f in V2_POSE_A_STRIP:
            frac, n = rec["poseA"]
            lim = V2_POSE_A_STRIP[f]
            if n and frac > lim + 1e-12:
                bad.append(f"pose A f{f} strip holes {frac:.4f} of {n} cells (max v2 {lim:.4f})")
        if f in V2_POSE_B_STRIP:
            frac, n = rec["poseB"]
            lim = V2_POSE_B_STRIP[f]
            if n and frac > lim + 1e-12:
                bad.append(f"pose B f{f} strip holes {frac:.4f} of {n} cells (max v2 {lim:.4f})")
    assert not bad, "behind-car strip vs v2: " + "; ".join(bad)


def render_frame_edge_rows(chase, rend_pts, gt_pts):
    """Same-pixel |x_r - x_gt| in a chase render, binned by GT range. fill off."""
    def paint(pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            z = np.full((chase.height, chase.width), np.nan)
            return z, np.zeros(z.shape, bool)
        r = sf.compose(chase, pts, np.full((pts.shape[0], 3), 128, np.uint8), np.full(pts.shape[0], 0.3),
                       sky_rgb=None, ground_rgb=None, fill=False)
        return r.depth, r.masks["observed"]
    rd, ro = paint(rend_pts)
    gd, go = paint(gt_pts)
    both = ro & go & np.isfinite(rd) & np.isfinite(gd)
    ys, xs = np.nonzero(both)
    rows = []
    if not ys.size:
        return rows
    pr = sf.unproject(chase, rd, ys, xs)
    pg = sf.unproject(chase, gd, ys, xs)
    rng = np.hypot(pg[:, 0], pg[:, 1])
    err = np.abs(pr[:, 0] - pg[:, 0])
    over = np.abs(pr[:, 0]) - np.abs(pg[:, 0])
    for lo, hi in ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0)):
        k = (rng >= lo) & (rng < hi)
        if not k.any():
            continue
        rows.append((lo, hi, float(np.median(err[k])), float(np.percentile(err[k], 90)),
                     float(np.median(np.abs(pg[k, 0]))), int(k.sum()),
                     float(over[k].max())))
    return rows


def render_edge_gate(rows_by_view, label):
    """Identical thresholds to capture-frame gate (a): med <=10%, p90 <=20% of |x_gt|, overshoot <=0.3 m at 0-5 m."""
    print(f"{label} render-frame edge (med, p90, med |x_gt|, n, max overshoot):", rows_by_view, flush=True)
    bad, over_bad = [], []
    for view, rows in rows_by_view.items():
        for lo, hi, med, p90, lat, n, over in rows:
            if n == 0 or lat < 0.25:
                continue
            if hi <= 15.0 and (med > EDGE_MED_FRAC * lat + 1e-9 or p90 > EDGE_P90_FRAC * lat + 1e-9):
                bad.append(f"{view} {lo:g}-{hi:g} m: median {med:.3f} m p90 {p90:.3f} m vs 10%/20% of |x_gt| {lat:.3f} m")
            if lo == 0.0 and hi == 5.0 and over > EDGE_OVERSHOOT_M:
                over_bad.append(f"{view} 0-5 m overshoot {over:.3f} m")
    msg = []
    if bad:
        msg.append("edge position: " + "; ".join(bad))
    if over_bad:
        msg.append("edge overshoot: " + "; ".join(over_bad))
    assert not msg, f"{label} render-frame " + " | ".join(msg)


def check_clip015_render_edge_and_holes() -> None:
    """Gate (a) and far-hole gates in the RENDER (chase) frame, identical thresholds, dev frames only."""
    if not _need_clip("clip015_render_edge_and_holes"):
        return SKIPPED
    # current-frame points (no grid) — same ingredients as capture-frame cam_frame, viewed in chase
    poses = (("poseA", pose_a_chase()), ("poseB", pose_b_chase()))
    gathered = clip_behind_car_strip((40, 160, POSE_B_FRAME), poses)
    # clip_behind_car_strip stores used + pose hole fracs of the *strip*, not general holes.
    # Re-collect rend/gt via the same helper path: rebuild from a light loop.
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    edge = {}
    holes = {}
    for target in (40, 160, POSE_B_FRAME):
        rend, gtp = [], []
        for cid in p.cam:
            idx = nearest_idx(pred_frames(P, cid), target)
            if idx is None:
                continue
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            _, d, v, _ = fr
            cam = p.cam[cid]
            c = cams[cid]
            pw, ph = p.ns["proc_res"](c)
            W0, _ = c["resolution"]
            gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
            net = _net_road(P, cid, idx, W0, pw, ph)
            ys, xs = np.nonzero(v & np.isfinite(d))
            if ys.size:
                q = sf.unproject(cam, d, ys, xs)
                rend.append(q[ego_box_filter(q)])
            if gtd is not None:
                # GT road = net px within 1 m of plane (note: net mask too wide is not measured)
                ys, xs = np.nonzero(net & np.isfinite(gtd))
                if ys.size:
                    q = sf.unproject(cam, gtd, ys, xs)
                    q = q[np.abs(q[:, 2]) < BEHIND_Z_M]
                    if q.size:
                        gtp.append(q[ego_box_filter(q)])
        R = np.concatenate(rend) if rend else np.zeros((0, 3))
        G = np.concatenate(gtp) if gtp else np.zeros((0, 3))
        for name, ch in poses:
            key = f"{name}-f{target}"
            edge[key] = render_frame_edge_rows(ch, R, G)
            holes[key] = hole_fraction(ch, R, G)
    print("render-frame holes (fill on, GT-ground |z|<=STATIC_MAX_HEIGHT_M for the kept far-hole "
          "definition on pose B; render edge uses GT road = net px within 1 m of plane):",
          {k: v for k, v in holes.items()}, flush=True)
    render_edge_gate(edge, "f40/f110/f160")
    # holes gated identically: <=2% at 0-15 m, <=5% at 15-30 m, each pose/frame
    fr_by_view = {}
    for key, bins in holes.items():
        fr_by_view[key] = bins
    # hole_gate expects numeric view keys; reuse its logic
    bad = []
    for view, bins in fr_by_view.items():
        for b, (f, n) in bins.items():
            if n and f > HOLE_BINS[b]:
                bad.append(f"{view} {b[0]:g}-{b[1]:g} m: holes {100 * f:.2f}% of {n} GT road px (max {100 * HOLE_BINS[b]:g}%)")
    assert not bad, "render-frame holes: " + "; ".join(bad)


def clip_capture_holes(frames):
    """Capture-frame holes: net px within 1 m of plane that have no valid rendered depth, by range."""
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    rows = []  # (cid, target, lo, hi, n_hole, n)
    used = {}
    for target in frames:
        for cid in p.cam:
            idx = nearest_idx(pred_frames(P, cid), target)
            if idx is None:
                continue
            used.setdefault(cid, {})[target] = idx
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            _, d, v, _ = fr
            cam = p.cam[cid]
            c = cams[cid]
            pw, ph = p.ns["proc_res"](c)
            W0, _ = c["resolution"]
            gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
            if gtd is None:
                continue
            net = _net_road(P, cid, idx, W0, pw, ph)
            ys, xs = np.nonzero(net & np.isfinite(gtd))
            if not ys.size:
                continue
            pg = sf.unproject(cam, gtd, ys, xs)
            on = np.abs(pg[:, 2]) < BEHIND_Z_M  # within 1 m of plane
            if not on.any():
                continue
            ys, xs, pg = ys[on], xs[on], pg[on]
            rng = np.hypot(pg[:, 0], pg[:, 1])
            owned = np.ones(d.shape, bool)
            if bool(p.a.ownership) and cid in p.ns.get("YIELD", {}):
                owned &= ~np.asarray(p.ns["YIELD"][cid], bool)
            if cid in p.ns.get("EGO_MASK", {}):
                r0, c0, c1 = p.ns["EGO_MASK"][cid]
                ph_, pw_ = d.shape
                owned[int(r0 * ph_):, int(c0 * pw_):int(c1 * pw_)] = False
            hole = (~v[ys, xs]) & owned[ys, xs]
            owned_px = owned[ys, xs]
            for lo, hi in HOLE_BINS:
                k = (rng >= lo) & (rng < hi) & owned_px
                n = int(k.sum())
                n_h = int(hole[k].sum()) if n else 0
                rows.append((cid, target, lo, hi, n_h, n))
    return {"rows": rows, "used": used}


def capture_hole_gate(m, label):
    """Identical to the far-hole gate: <=2% at 0-15 m, <=5% at 15-30 m, per cam (pooled over frames)."""
    print(f"{label} capture-frame holes (GT road = net px within 1 m of plane; hole = no valid render depth):",
          m["rows"], "used", m["used"], flush=True)
    acc = {}
    for cid, tgt, lo, hi, n_h, n in m["rows"]:
        a = acc.setdefault((cid, lo, hi), [0, 0])
        a[0] += n_h
        a[1] += n
    bad = []
    for (cid, lo, hi), (n_h, n) in sorted(acc.items()):
        if n == 0:
            continue
        frac = n_h / n
        lim = HOLE_BINS[(lo, hi)]
        if frac > lim:
            bad.append(f"{cid} {lo:g}-{hi:g} m: holes {100 * frac:.2f}% ({n_h}/{n}) (max {100 * lim:g}%)")
    assert not bad, f"{label} capture-frame holes: " + "; ".join(bad)


def check_clip015_capture_holes() -> None:
    if not _need_clip("clip015_capture_holes"):
        return SKIPPED
    capture_hole_gate(clip_capture_holes((40, 160, POSE_B_FRAME)), "f40/f110/f160")




def check_road_labelled_object_top_stays_put() -> None:
    """A wall / car / curb top labelled road stays put: snap must not move it outward, and it keeps height and depth."""
    p = pv()
    bad = []
    for cid, kind, x0, y0 in (("pillarR", "car", 2.5, 0.0), ("pillarR", "curb", 3.5, -3.0),
                              ("wide", "car", 0.0, 8.0), ("main", "car", 0.0, 10.0), ("pillarL", "pole", -3.0, 1.0)):
        cam = p.cam.get(cid)
        if cam is None:
            continue
        if cid in ("wide", "main"):
            # a face ahead, x = lateral, y = range: reuse object_scene-like geometry along y
            g = geom(cam)
            rays = g["rays"]
            w, h_ = (1.8, 1.5) if kind == "car" else (0.15, 2.5)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(rays[..., 1] > 1e-9, (y0 - cam.pos[1]) / rays[..., 1], np.inf)
                X = cam.pos[0] + t * rays[..., 0]
                Z = cam.pos[2] + t * rays[..., 2]
            face = np.isfinite(t) & (np.abs(X) <= w / 2) & (Z >= 0) & (Z <= h_) & (t < g["hit"])
            gt = np.where(face, t, np.where(g["down"], g["hit"], np.inf))
        else:
            gt, fine0, face, g = side_object_scene(cam, kind, x0, y0)
        if not face.any():
            continue
        zf = heights(cam, np.where(np.isfinite(gt), gt, 0.0))
        zmax = float(zf[face].max())
        top = face & (zf >= zmax - 0.15)   # the top 0.15 m of the object, labelled road
        if not top.any():
            continue
        fine = np.where(top, ROAD_ID, np.where(face, OBJ_ID, np.where(~g["down"], sf_sky(), ROAD_ID)))
        raw = np.where(np.isfinite(gt), gt, 1e4)
        _, d, v, _ = p.run(cid, raw, fine)
        ys, xs = np.nonzero(top)
        if ys.size == 0:
            continue
        # snap must not pull these pixels onto the ground or farther along the ray
        moved = (np.abs(d[top] - raw[top]) > 0.05) & v[top]
        z_r = heights(cam, d)
        lost = zf[top] - z_r[top]
        outward = (d[top] > raw[top] + 0.05) & np.isfinite(d[top])
        if outward.any() or (lost[v[top]] > OBJ_TOL_M).any() if v[top].any() else False:
            bad.append(f"{cid} {kind} top: {int(outward.sum())} px moved outward, "
                       f"max height lost {float(lost[v[top]].max()) if v[top].any() else 0:.3f} m, "
                       f"GT z up to {zmax:.2f} m")
    assert not bad, "; ".join(bad)


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
    check_snapped_pixels_not_exempt_from_edge_cut,
    check_road_labelled_object_top_stays_put,
    check_behind_car_strip_holes,
    check_clip015_render_edge_and_holes,
    check_clip015_capture_holes,
    check_runtime_mask_growth,
    check_clip015_snap_misplace,
    check_holdout_snap_misplace,
    check_clean_holdout_snap_misplace,
    check_clip015_edge_position,
    check_holdout_edge_position,
    check_clean_holdout_edge_position,
    check_clip015_snap_no_worse_than_off,
    check_holdout_snap_no_worse_than_off,
    check_clean_holdout_snap_no_worse_than_off,
    check_clip015_footprint,
    check_holdout_footprint,
    check_clean_holdout_footprint,
    check_clip015_rear_holes_poseB,
    check_clip015_width,
    check_holdout_width,
    check_clip015_edge_overshoot,
    check_holdout_edge_overshoot,
    check_clip015_rear_holes,
    check_holdout_rear_holes,
]

# name -> counterexample on 783fedd (real runs after thresholds commit 8b53e6b;
# rv-side-evidence/props_report_783fedd.log). E = depth-edge cut in tools/rv_stable_preview.py cam_frame
# (after ground_clamp: far rows pushed below the road, out of the ground band); L = the net's 12x16 class
# grid (road label bleeds past the edge); N = road_snap flattens objects inside a road-majority cell.
KNOWN_FAILING: dict[str, str] = {
    "check_synthetic_width_exact_labels": "E: exact depth + exact labels: repeatL/R 15-30 m median |err| 30.8% p90 100%; rear 15-30 m 96.8%/100%",
    "check_synthetic_width": "L+E: pillarL/R 0-5 m median 14.8% p90 47.3%; repeatL/R 0-5 m 16.6%/23.0%, 5-15 m 28.1%/48.2%; rear 15-30 m",
    "check_synthetic_edge_overshoot": "L: road 0.35 m (pillars) / 0.71 m (repeats) past the GT edge at 0-5 m (12x16 label bleed)",
    "check_side_objects_keep_height_pins": "N: pillarR parked car x2.5 lost 1.07 m on 88 px; pole x3 lost 1.07 m on 351/391 px; curbs lost 0.11 m",
    "check_snapped_pixels_not_exempt_from_edge_cut": "NEW vs a56f514: 456/456 snapped px at a depth edge were exempt from the cut (tools/rv_stable_preview.py: `cut & ~sky & ~snapped`, added at 4df1176). wide, left-half road 2.5x / right unsnapped 2.5x, clamp on.",
    "prop_side_objects_keep_height": "N: pillarL pole at x=-2.0, y0=0.0: lost 0.592 m",
    "check_road_labelled_object_top_stays_put": "N: pillarR car top 218 px moved outward (GT z 1.50 m); curb top 1487 px, height lost 0.15 m; main car top also snapped outward",
    "check_clip015_snap_misplace": "f40/f160 DRAWN-only: pillarR 18.10% (7317/40432; f40 36.90%/f160 3.50%), repeatR 4.65% (1835/39434), wide 10.12% (3555/35133; was 15.00% all-snapped 7461/49726) at 0-15 m; wide 61.64% (188/305; all-snapped 59.35% 219/369) at 15-30 m (GT z>0.30)",
    "check_holdout_snap_misplace": "f100/f200 DRAWN-only: wide 4.46% (1562/34986; was 15.00% all-snapped) and pillarR 4.44% at 0-15 m; wide 30.80% (182/591; was 37.67%), rear 6.25%, repeatL 5.29% at 15-30 m",
    "check_clean_holdout_snap_misplace": "f70 DRAWN-only: pillarR 55.49%, pillarL 41.55%, repeatR 24.60%, wide 11.39% (2067/18147; was 14.34% all-snapped), rear 5.16% at 0-15 m; wide 79.38% (77/97), rear 9.44% at 15-30 m",
    "check_clip015_edge_position": "f40/f160: wide 5-15 m median 1.334/p90 5.119 vs 10%/20% of |x_gt| 4.36 m; also narrow/main 5-15 p90 over the 20% gate",
    "check_holdout_edge_position": "f100/f200: wide 5-15 m median 0.551/p90 5.453 vs |x_gt| 5.19 m; narrow/main p90 over at 5-15 m",
    "check_clean_holdout_edge_position": "f70: wide 5-15 m median 2.183/p90 4.595; pillarL 5-15 m 2.453/5.904; pillarR 0-5 m 0.568/6.308",
    "check_clip015_snap_no_worse_than_off": "f40/f160: pillarR |dz| median 0.246 vs 0.184 off; pillarL 0.049/0.603 vs 0.048/0.593",
    "check_holdout_snap_no_worse_than_off": "f100/f200: pillarR p90 0.436 vs 0.285 off",
    "check_clean_holdout_snap_no_worse_than_off": "f70: pillarR median 0.048 vs 0.035; repeatR p90 0.555 vs 0.415; pillarL 0.098 vs 0.094",
    "check_clip015_footprint": "f40/f160: narrow 45.3%, main 70.4%, wide 83.8%, pillarR 81.9%, repeatL 93.0% (min 95%)",
    "check_holdout_footprint": "f100/f200: narrow 46.9%, main 72.6%, wide 88.1%, pillarR 89.2%, repeatR 93.4%",
    "check_clean_holdout_footprint": "f70: pillarL 18.3%, narrow 49.5%, pillarR 58.8%, main 72.2%, wide 86.0%",
    "check_behind_car_strip_holes": "pose A f40 0.2000 of 160 cells (max v2 0.1456); pose A f160 0.2250 of 160 (max v2 0.0000). Pose B f110 0.0054 of 1491 PASSES (max v2 0.1058). This test: n=160/160; Historian 783fedd was 0.2215/0.2767 (n=158/159) — same definition, small n/frac difference.",
    "check_clip015_render_edge_and_holes": "render-frame edge: poseA f40 5-15 m median 0.303/p90 2.651 vs 10%/20% of |x_gt| 4.245 m; poseB f40 5-15 p90 1.381 vs 3.981 m; poseA f160 5-15 p90 2.044 vs 5.429 m; poseA f110 5-15 p90 2.394 vs 5.540 m; poseB f110 5-15 p90 1.282 vs 5.223 m; overshoot poseA-f40 0-5 m 2.792 m, poseB-f40 0-5 m 1.385 m. Render-frame holes PASS (all poses/frames <=2%/<=5%).",
}

_REPORT = os.environ.get("RV_PROPS_REPORT", "").strip() in {"1", "true", "yes"}


def main() -> None:
    print("Note: net mask too wide is not measured: GT road = net px within 1 m of plane.", flush=True)
    print("gate (b): denom = snapped px still DRAWN (valid in cam_frame after runtime yield L115 / ego mask L118) "
          "with finite GT depth, per cam/bin (range = hypot(x,y) of that px's GT unprojection); "
          "numer = those px with GT z > 0.30 m at that px's GT depth. All-snapped is INFO.", flush=True)
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
