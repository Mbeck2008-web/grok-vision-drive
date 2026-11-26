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
import contextlib
import inspect
import atexit
import json
import math
import os
import shutil
import subprocess
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


def snap_hook_bound_args(orig, a, k) -> tuple:
    """Bind a snap function's (*a, **k) by parameter name (inspect.signature).

    road_snap is ``(cam, depth, road, ...)``; snap_road_for_cam is ``(cam_id, cam, depth, road, ...)``.
    Positional a[1] is depth only for road_snap — using it for snap_road_for_cam treats the CameraPose as
    depth and crashes (``float() argument ... not 'CameraPose'``). Always pull ``depth`` and ``road`` by
    name; raise if either parameter is missing.
    """
    try:
        ba = inspect.signature(orig).bind(*a, **k)
    except TypeError as e:
        raise TypeError(f"snap hook could not bind {getattr(orig, '__name__', orig)}: {e}") from e
    ba.apply_defaults()
    missing = [n for n in ("depth", "road") if n not in ba.arguments]
    if missing:
        raise TypeError(f"{getattr(orig, '__name__', orig)} has no parameter {missing} "
                        f"(got {list(ba.arguments)}); cannot hook snap")
    return ba.arguments["depth"], ba.arguments["road"], ba


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
        self.snap_hook = {"mask": None, "pre": None, "calls": []}

        class _SHook:
            def __init__(self, real, hook):
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_hook", hook)

            def __getattr__(self, name):
                return getattr(object.__getattribute__(self, "_real"), name)

            def _wrap(self, name, *a, **k):
                hk = object.__getattribute__(self, "_hook")
                orig = getattr(object.__getattribute__(self, "_real"), name)
                dep, road, ba = snap_hook_bound_args(orig, a, k)
                cam = ba.arguments.get("cam", ba.arguments.get("cam_id"))
                if not hk.get("_in"):
                    hk["pre"] = None if dep is None else np.array(dep, np.float64, copy=True)
                hk["calls"].append((name, getattr(cam, "name", None) or getattr(cam, "cam_id", None) or cam))
                hk["_in"] = hk.get("_in", 0) + 1
                try:
                    out, m = orig(*a, **k)
                finally:
                    hk["_in"] -= 1
                hk["mask"] = m
                return out, m

            def road_snap(self, *a, **k):
                return self._wrap("road_snap", *a, **k)

            def snap_road_for_cam(self, *a, **k):   # per-cam snap entry point, if the runtime has one
                return self._wrap("snap_road_for_cam", *a, **k)

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
        self.snap_hook["pre"] = None
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
        if name in SEAL_GUARD_REPORT_ONLY:
            print(f"INFO {name}: {_first_line(e)}", flush=True)
            return True
        if _REPORT:
            print(f"FAIL {name}: {_first_line(e)}", flush=True)
            return True
        if name in KNOWN_FAILING:
            print(f"XFAIL {name}: {_first_line(e)}", flush=True)
            return True
        print(f"FAIL {name}: {_first_line(e)}", flush=True)
        traceback.print_exc()
        return False
    if name in SEAL_GUARD_REPORT_ONLY:
        print(f"ok {name} (report-only; seal guard band, not a gate)", flush=True)
        return True
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
        if in_seal_guard(idx):
            print(f"  seal guard: f{target} {cid} pred idx {idx} in {SEAL_GUARD_RANGES} -> not measured (gating)", flush=True)
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
            if in_seal_guard(idx):
                print(f"  seal guard: f{target} {cid} pred idx {idx} in {SEAL_GUARD_RANGES} -> not measured (gating)", flush=True)
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



@contextlib.contextmanager
def cut_clamp_tap():
    """Record the runtime cam_frame's calls to the two geometry stages it shares with the renderer:
    ("cut", depth_in, sky_arg, ground_hit, depth_out) for sf.edge_cut_far_side and ("clamp", depth_in, depth_out) for
    sf.ground_clamp, in call order. The preview reaches them through ``S.<name>`` at call time, so patching the
    module attributes is seen by cam_frame; both are restored on exit."""
    log: list = []
    oc, og = sf.edge_cut_far_side, sf.ground_clamp

    def cut(depth, sky, ground_hit, *a, **k):
        out = oc(depth, sky, ground_hit, *a, **k)
        log.append(("cut", np.array(depth, np.float64), np.array(sky, bool), np.array(ground_hit, np.float64), np.array(out)))
        return out

    def clamp(cam, depth, *a, **k):
        out = og(cam, depth, *a, **k)
        log.append(("clamp", np.array(depth, np.float64), np.array(out[0])))
        return out

    sf.edge_cut_far_side, sf.ground_clamp = cut, clamp
    try:
        yield log
    finally:
        sf.edge_cut_far_side, sf.ground_clamp = oc, og


def run_tapped(cid, raw, lab):
    """(cam_frame outputs, tap log) for one synthetic input."""
    with cut_clamp_tap() as log:
        out = pv().run(cid, raw, lab)
    return out, log


def check_snapped_pixels_not_exempt_from_edge_cut() -> None:
    """Snapped pixels must still take the depth-edge cut (tightening: retargeted from wide, which no longer snaps).

    The runtime must hand ``edge_cut_far_side`` exactly the sky mask: adding the snapped px to it (the old
    ``cut & ~sky & ~snapped``, as a ``sky | snapped`` argument) exempts them from the cut. With clamp margin 1.0 a
    snapped px sits at its own ground hit, so the cut can only move it where the far side is equal; the exemption is
    therefore not visible in the output on a flat scene, and the argument is what is pinned. Cams: main and pillarR
    (both in ROAD_SNAP_CAMS); left half road at 2.5x depth (snaps onto the plane), right half non-road at 2.5x."""
    p = pv()
    snap_cams = resolve_road_snap_cams()
    bad = []
    for cid in ("main", "pillarR"):
        assert cid in snap_cams, f"{cid} must be in ROAD_SNAP_CAMS for this check (got {sorted(snap_cams)})"
        cam = p.cam[cid]
        g = geom(cam)
        H, W = g["hit"].shape
        ys, xs = np.mgrid[0:H, 0:W]
        raw = np.where(g["down"], g["hit"] * 2.5, 1e4)
        road = g["down"] & (xs < W // 2)
        sem = np.where(~g["down"], sf_sky(), np.where(road, ROAD_ID, NONROAD_ID))
        fr, log = run_tapped(cid, raw, sem)
        sky, snap = fr[3], p.last_snap
        assert snap is not None and snap.any() and snap[:, : W // 2].any(), f"{cid}: fixture must snap road px"
        cuts = [e for e in log if e[0] == "cut"]
        assert len(cuts) == 1, f"{cid}: cam_frame called the edge cut {len(cuts)} times"
        sky_arg = cuts[0][2]
        extra = int((sky_arg & ~sky).sum())
        if extra or not np.array_equal(sky_arg, sky):
            bad.append(f"{cid}: edge cut got {extra} non-sky px in its sky mask ({int(snap.sum())} snapped px are exempt from the cut)")
    assert not bad, "; ".join(bad)


UNDER_ROAD_TOL_M = 0.05   # no rendered point below the road plane by more than this (z < -0.05 m)


def cut_clamp_order_scenes(cid):
    """(name, raw net depth, coarse labels) fed to cam_frame for the order / under-road check: the flat synthetic
    ground with the net's 12x16 labels, and a left-road / right-non-road scene at 2.5x depth (net depth under the road)."""
    cam = pv().cam[cid]
    depth, fine, net, gt_road, g = ground_scene(cam)
    H, W = g["hit"].shape
    xs = np.mgrid[0:H, 0:W][1]
    half = np.where(g["down"], g["hit"] * np.where(xs < W // 2, 1.0, 2.5), 1e4)
    return [("ground", depth, net), ("left road / right 2.5x deep", half, net)]


def check_cut_then_clamp_order_and_nothing_under_road() -> None:
    """38c9827: the depth-edge cut runs first and the ground clamp after it, on every snapping cam, and nothing is
    rendered under the road. (1) call order cam_frame -> edge_cut_far_side then ground_clamp, with the clamp's input
    exactly the cut's output (a clamp-then-cut order fails here); (2) valid rendered points with z < -0.05 m: 0 per cam
    on the synthetic scenes (the count is printed). Swapping the order alone leaves the guarded cut under-road-free on
    these scenes, so (1) is what pins the order."""
    bad = []
    counts = {}
    for cid in sorted(resolve_road_snap_cams()):
        cam = pv().cam[cid]
        n_under = 0
        for name, raw, lab in cut_clamp_order_scenes(cid):
            fr, log = run_tapped(cid, raw, lab)
            _, d, v, _ = fr
            kinds = [e[0] for e in log]
            if kinds != ["cut", "clamp"]:
                bad.append(f"{cid} {name}: cam_frame stage order {kinds} (want ['cut', 'clamp'])")
                continue
            if not np.array_equal(log[1][1], log[0][4], equal_nan=True):
                bad.append(f"{cid} {name}: ground_clamp input is not the edge cut's output")
            under = v & np.isfinite(d) & (heights(cam, d) < -UNDER_ROAD_TOL_M)
            n_under += int(under.sum())
            if under.any():
                bad.append(f"{cid} {name}: {int(under.sum())} rendered px below the road plane by > {UNDER_ROAD_TOL_M} m")
        counts[cid] = n_under
    print(f"  rendered px below the road plane (z < -{UNDER_ROAD_TOL_M} m), synthetic scenes, per snap cam: {counts}", flush=True)
    assert not bad, "; ".join(bad)


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
            if in_seal_guard(idx):
                print(f"  seal guard: f{target} {cid} pred idx {idx} in {SEAL_GUARD_RANGES} -> not measured (gating)", flush=True)
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
            if in_seal_guard(idx):
                print(f"  seal guard: f{target} {cid} pred idx {idx} in {SEAL_GUARD_RANGES} -> not measured (gating)", flush=True)
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
            if in_seal_guard(idx):
                print(f"  seal guard: f{target} {cid} pred idx {idx} in {SEAL_GUARD_RANGES} -> not measured (gating)", flush=True)
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


# ------------------------------------------------------------------ ripple (rings) + isolated streak (GVD 2026-11-26)
# Michael picked WIDE SNAP OFF (1:08 PM). With wide snap off he sees (1) concentric rings on the road in the
# pose-B chase and (2) a single line of 3-D points shooting off into the distance. Thresholds below were
# set by rule on dev threshold frames RS_THRESH_FRAMES (never f40/f110/f160, never burned f60/f130, never
# the sealed held-out pair #2), then verified on f40/f110/f160. See the commit message for the table.
RS_THRESH_FRAMES = (10, 25, 50, 120, 140, 155)  # rule: clip015 multiples of 5 outside the seal guard band (frame AND every cam's pred idx), not 40/60/110/130/160, >=10 from them, no shared sparse pred idx; greedy >=15 apart from f10 (f170 dropped 13:51: inside the guard band)
RS_VERIFY_FRAMES = (40, 110, 160)               # Michael's before/after still frames
RS_BURNED = (60, 130)                           # one-time unseal on 2026-11-26: burned
HELDOUT2_SEALED = (85, 185)                     # sealed held-out pair #2 (rv-side-evidence/heldout2_SEALED.md); never measured here
HELDOUT2_SEALED_IDX = {"main": (85, 185), "narrow": (85, 185), "wide": (85, 185), "pillarL": (86, 184), "pillarR": (85, 184),
                       "rear": (86, 183), "repeatL": (83, 184), "repeatR": (85, 187)}
# SEAL GUARD BAND (Skeptic, binding 13:51): no dev measurement / threshold / fixture / tuning may use a frame or a
# pred idx within +-20 of a sealed pred idx (83-87, 183-187) -> excluded 63-107 and 163-207 (kills f70/f80/f100/f200).
# Verify frame f110: rear's nearest pred idx is 107 (in band) -> rear is not measured at f110.
SEAL_GUARD_M = 20
SEAL_GUARD_RANGES = ((83 - SEAL_GUARD_M, 87 + SEAL_GUARD_M), (183 - SEAL_GUARD_M, 187 + SEAL_GUARD_M))  # sealed pred idxs 83-87, 183-187 -> 63-107, 163-207


def in_seal_guard(i) -> bool:
    return i is not None and any(lo <= int(i) <= hi for lo, hi in SEAL_GUARD_RANGES)


RS_CAMS = ("wide", "main", "narrow", "pillarL", "pillarR", "repeatL", "repeatR", "rear")
RIP_BIN_M = 0.5            # range bin for the residual profile
RIP_R_MAX = 30.0
RIP_GT_Z = 1.0             # GT road = net road px whose GT point is within 1 m of z=0 (as the strip gate)
RIP_MIN_PX = 10            # px per range bin for the bin to count
RIP_MIN_BINS = 10          # bins per cam for the profile metrics to count (else n/a, printed)
RIP_ZC_EPS = 0.01          # zero-crossing dead band (m): sign changes of the detrended residual beyond +-1 cm
RIP_DETREND_BINS = 11      # local linear ground fit over 11 bins = 5.5 m centred on each bin
RIP_DZ_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 30.0))
RIP_PTP_BIN = (5.0, 15.0)
RIP_BAND = 0.25            # synthetic banding: every other 12x16 row 25% short (raised by 0.25 x cam height; far-side errors are clamped to the road)
RIP_METRICS = ("std", "ptp_5_15", "pfrac", "zc") + tuple(f"dz_{lo:g}_{hi:g}" for lo, hi in RIP_DZ_BINS)
STREAK_Y = (15.0, 20.0)
STREAK_X = (0.5, 1.0)
STREAK_K = 3
STREAK_Z_M = 5.0           # ray run: > 5 m of range along one pixel column
STREAK_RAY_X = 1.0         # ray-run points must be isolated (< STREAK_K neighbours within 1.0 m)
FOOT_GAP_K = 1.05           # Tracer: row spacing > 1.05 x footprint = a visible gap (stripe)
FOOT_GAP_MAX = 1.0 - FOOTPRINT_MIN_FRAC   # = 5%: the committed P gate (>= 95% of rows covered), not tuned here
FOOT_GAP_BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))
COLMONO_RISE_M = 100.0     # INFO only (Skeptic: circular vs the fix): drawn depth rising > 100 m up one column; Tracer's number, not derived
GC_RATIO, GC_ABS = 1.5, 20.0   # GT-consistency: drawn depth >= 1.5 x GT AND > GT + 20 m where GT has the nearer surface (Skeptic's numbers, stated not derived)
FR_MIN_M = 30.0            # false-removal: GT surface past 30 m
FR_MARGIN = 0.02           # kept fraction >= head (worst threshold frame) - 2 pp
HOLE_BANDS = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))
GC_LIMITS: dict = {"wide": 531, "main": 1584, "narrow": 3637, "pillarL": 701, "pillarR": 978, "repeatL": 732, "repeatR": 834, "rear": 716}  # min(ON, v2) + GT-path, worst threshold frame
FR_LIMITS: dict = {"wide": 0.0397, "main": 0.5822, "narrow": 0.9175, "pillarL": 0.2314, "pillarR": None, "repeatL": 0.3118, "repeatR": 0.2822, "rear": 0.0287}  # head (snap OFF) pooled threshold frames - 2 pp; None: n < 200
RINGA_LIMITS: dict = {"0-5": 0.0136, "5-15": 0.0443, "15-30": 0.0}  # min(ON, v2) + GT, worst threshold frame
STREAK_START_MAX_M = 30.0  # a ray run must start inside the 0-30 m field (the line starts near the car and shoots off)
SURF_REL, SURF_ABS = 0.10, 0.5   # on the GT surface: within 10% / 0.5 m of GT depth anywhere in the px's 3x3 GT neighbourhood
# Limits (rule: no worse than the better of 783fedd snap ON and v2 10893ee, worst threshold frame of
# each, plus the GT-depth floor of the same metric as margin; see RS_LIMIT_RULE). None = n/a for that cam.
RIPPLE_LIMITS: dict = {   # {cam: {metric: limit}}; rule on RS_THRESH_FRAMES (rv-side-evidence/ripple/rule.log)
    'wide': {'std': 0.164, 'ptp_5_15': 0.436, 'pfrac': 0.561, 'zc': 15, 'dz_0_5': 0.08, 'dz_5_10': 0.163, 'dz_10_15': 0.431, 'dz_15_30': 0.417},
    'main': {'std': 0.107, 'ptp_5_15': 0.051, 'pfrac': 0.486, 'zc': 13, 'dz_0_5': 0.09, 'dz_5_10': 0.125, 'dz_10_15': 0.22, 'dz_15_30': 0.266},
    'narrow': {'std': 0.076, 'ptp_5_15': 0.011, 'pfrac': 0.409, 'zc': 11, 'dz_0_5': None, 'dz_5_10': 0.143, 'dz_10_15': 0.265, 'dz_15_30': 0.299},
    'pillarL': {'std': 0.194, 'ptp_5_15': 0.258, 'pfrac': 0.673, 'zc': 8, 'dz_0_5': 0.033, 'dz_5_10': 0.093, 'dz_10_15': 0.347, 'dz_15_30': 0.258},
    'pillarR': {'std': 0.185, 'ptp_5_15': 0.722, 'pfrac': 1.1, 'zc': 4, 'dz_0_5': 0.046, 'dz_5_10': 0.524, 'dz_10_15': 0.2, 'dz_15_30': 1.717},
    'repeatL': {'std': 0.118, 'ptp_5_15': 0.229, 'pfrac': 0.801, 'zc': 17, 'dz_0_5': 0.041, 'dz_5_10': 0.106, 'dz_10_15': 0.212, 'dz_15_30': 0.36},
    'repeatR': {'std': 0.148, 'ptp_5_15': 0.26, 'pfrac': 0.531, 'zc': 21, 'dz_0_5': 0.037, 'dz_5_10': 0.118, 'dz_10_15': 0.212, 'dz_15_30': 0.424},
    'rear': {'std': 0.222, 'ptp_5_15': 0.386, 'pfrac': 0.465, 'zc': 27, 'dz_0_5': 0.062, 'dz_5_10': 0.109, 'dz_10_15': 0.196, 'dz_15_30': 0.503},
}
WIDE_BOWL_LIMIT = 0.543    # rule: min(ON 0.523, v2 2.607) + GT 0.020 (worst threshold frame)
PLANE_PULL_TOL: dict = {"0-5": 0.122, "5-10": 0.048, "10-15": 0.056, "15-30": 0.059}  # upsample tolerance, worst threshold frame, wide
ISO_LIMITS: dict = {(15.0, 0.5): 14987, (15.0, 1.0): 5427, (20.0, 0.5): 14876, (20.0, 1.0): 5426}  # pooled all cams; min(ON, v2) worst threshold frame, no margin (GT path has MORE: 20672/8829/20631/8823)
STREAK_RAY_LIMIT = 0       # rays above STREAK_Z_M per cam (GT floor stated in the commit)
RS_LIMIT_RULE = ("limit = min(max_f ON, max_f v2) + max_f GT-path, per cam and metric, over RS_THRESH_FRAMES; "
                 "ISO pooled: min(max_f ON, max_f v2), no margin; false-removal: head pooled - 2 pp; rays/colmono/footgap: INFO")


def rs_frames_ok(frames, used: dict | None = None) -> None:
    """Never measure burned or sealed frames (nor the sealed pair's sparse-cam pred indices)."""
    bad = [f for f in frames if f in RS_BURNED or f in HELDOUT2_SEALED or in_seal_guard(f)]
    for cid, idxs in (used or {}).items():
        bad += [f"{cid}:{i}" for i in idxs if i in HELDOUT2_SEALED_IDX.get(cid, ()) or in_seal_guard(i)]
    assert not bad, f"burned/sealed frames requested: {bad}"


def rs_record(cam, d, v, gtd, road, cap: float = math.inf) -> dict:
    """Per-cam record for the ripple/streak metrics; works for any pipeline (head, v2, GT path). cap = the
    pipeline's own far cap (far px are drawn at the cap by design; the GT surface is capped the same way)."""
    H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W]
    ys, xs = ys.ravel(), xs.ravel()
    with np.errstate(all="ignore"):
        q = sf.unproject(cam, np.where(np.isfinite(d), d, 0.0), ys, xs).reshape(H, W, 3)
        qg = sf.unproject(cam, np.where(np.isfinite(gtd), gtd, 0.0), ys, xs).reshape(H, W, 3)
    return {"d": np.asarray(d, np.float64), "v": np.asarray(v, bool) & np.isfinite(d) & (np.asarray(d) > 0),
            "gtd": np.asarray(gtd, np.float64), "road": np.asarray(road, bool), "q": q, "qg": qg, "cap": float(cap),
            "cam_z": float(np.asarray(cam.pos)[2])}


def ripple_metrics(rec) -> dict:
    """GT-road px drawn after cam_frame: dz = z_render - z_GT at the same px, range = hypot(x, y) of the GT point.
    Profile = per 0.5 m range bin median dz over 0-30 m (bins with >= RIP_MIN_PX px). Detrend = local ground fit:
    per bin, a least-squares line in range over the valid bins within +-2.5 m (RIP_DETREND_BINS), subtracted. std = std of the detrended profile; ptp_5_15 = its max-min in 5-15 m;
    pfrac = share of non-DC power in the dominant FFT bin of the detrended profile (gaps linearly interpolated);
    zc = sign changes of the detrended profile outside +-RIP_ZC_EPS; dz_lo_hi = median |dz| over px in that range."""
    gtd, qg, q = rec["gtd"], rec["qg"], rec["q"]
    m = rec["v"] & rec["road"] & np.isfinite(gtd) & (np.abs(qg[..., 2]) <= RIP_GT_Z)
    R = np.hypot(qg[..., 0], qg[..., 1])[m]
    dz = (q[..., 2] - qg[..., 2])[m]
    out = {"n_px": int(m.sum())}
    for lo, hi in RIP_DZ_BINS:
        k = (R >= lo) & (R < hi)
        out[f"dz_{lo:g}_{hi:g}"] = float(np.median(np.abs(dz[k]))) if k.sum() >= RIP_MIN_PX else None
        out[f"n_{lo:g}_{hi:g}"] = int(k.sum())
    nb = int(round(RIP_R_MAX / RIP_BIN_M))
    b = np.floor(R / RIP_BIN_M).astype(np.int64)
    ok = (b >= 0) & (b < nb)
    cnt = np.bincount(b[ok], minlength=nb)
    med = np.full(nb, np.nan)
    order = np.argsort(b[ok], kind="stable")
    bs, ds = b[ok][order], dz[ok][order]
    starts = np.searchsorted(bs, np.arange(nb))
    ends = np.searchsorted(bs, np.arange(nb), side="right")
    for i in range(nb):
        if cnt[i] >= RIP_MIN_PX:
            med[i] = float(np.median(ds[starts[i]:ends[i]]))
    good = np.isfinite(med)
    out["n_bins"] = int(good.sum())
    centres = (np.arange(nb) + 0.5) * RIP_BIN_M
    out["profile"] = [None if not np.isfinite(x) else round(float(x), 4) for x in med]
    if good.sum() < RIP_MIN_BINS:
        out.update(std=None, ptp_5_15=None, pfrac=None, zc=None, resid=None)
        return out
    e = np.full(nb, np.nan)
    h = RIP_DETREND_BINS // 2
    for i in np.nonzero(good)[0]:     # local ground fit: linear in range over a 5 m window around the bin
        lo, hi = max(0, i - h), min(nb, i + h + 1)
        k = good[lo:hi]
        if k.sum() >= 3:
            cf = np.polyfit(centres[lo:hi][k], med[lo:hi][k], 1)
            e[i] = med[i] - np.polyval(cf, centres[i])
    good = np.isfinite(e)
    if good.sum() < RIP_MIN_BINS:
        out.update(std=None, ptp_5_15=None, pfrac=None, zc=None, resid=None)
        return out
    out["resid"] = [None if not np.isfinite(x) else round(float(x), 4) for x in e]
    out["std"] = float(np.std(e[good]))
    k = good & (centres >= RIP_PTP_BIN[0]) & (centres < RIP_PTP_BIN[1])
    out["ptp_5_15"] = float(np.ptp(e[k])) if k.sum() >= 2 else None
    i0, i1 = int(np.nonzero(good)[0][0]), int(np.nonzero(good)[0][-1])
    xs_ = np.arange(i0, i1 + 1)
    x = np.interp(xs_, np.nonzero(good)[0], e[good])
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    out["pfrac"] = float(spec[1:].max() / spec[1:].sum()) if spec[1:].sum() > 0 else 0.0
    sgn = np.sign(np.where(np.abs(e[good]) > RIP_ZC_EPS, e[good], 0.0))
    sgn = sgn[sgn != 0]
    out["zc"] = int(np.count_nonzero(np.diff(sgn) != 0))
    return out


def _surface_ok(d, gtd, cap=math.inf):
    """Rendered depth within SURF_REL/SURF_ABS of the GT depth anywhere in the px's 3x3 GT neighbourhood
    (so the GT path's own nearest-resize half-pixel shift is not counted as off-surface); GT capped at the
    pipeline's far cap, since far px are drawn at the cap by design."""
    g = np.where(np.isfinite(gtd), np.minimum(gtd, cap), np.nan)
    gmin = -cv2.dilate(np.where(np.isfinite(g), -g, -np.inf).astype(np.float32), np.ones((3, 3), np.uint8))
    gmax = cv2.dilate(np.where(np.isfinite(g), g, -np.inf).astype(np.float32), np.ones((3, 3), np.uint8))
    with np.errstate(invalid="ignore"):
        return np.isfinite(gmax) & (d >= gmin * (1 - SURF_REL) - SURF_ABS) & (d <= gmax * (1 + SURF_REL) + SURF_ABS)


def streak_metrics(recs: dict) -> dict:
    """Rendered 3-D point set of one frame (every cam's drawn px after cam_frame, ego box cut; NO sky exemption):
    iso[(Y, X)] = points past Y m (hypot x,y) with < STREAK_K neighbours within X m (per cam and total).
    Ray runs, per cam: consecutive rows of one pixel column whose points are off the GT surface (_surface_ok
    false) and isolated (< STREAK_K neighbours within STREAK_RAY_X), starting within STREAK_START_MAX_M and
    spanning > STREAK_Z_M of range."""
    from scipy.spatial import cKDTree
    pts, own, pix = [], [], []
    for cid, r in recs.items():
        ys, xs = np.nonzero(r["v"])
        q = r["q"][ys, xs]
        k = ego_box_filter(q)
        pts.append(q[k]); own += [cid] * int(k.sum()); pix.append(np.stack([ys[k], xs[k]], 1))
    if not pts:
        return {"n_pts": 0, "iso": {}, "rays": {}}
    P = np.concatenate(pts); own = np.array(own); pix = np.concatenate(pix)
    tree = cKDTree(P)
    R = np.hypot(P[:, 0], P[:, 1])
    out = {"n_pts": int(len(P)), "iso": {}, "rays": {}}
    for Y in STREAK_Y:
        far = R > Y
        for X in STREAK_X:
            c = tree.query_ball_point(P[far], X, return_length=True) - 1 if far.any() else np.zeros(0, int)
            iso = c < STREAK_K
            out["iso"][(Y, X)] = {"total": int(iso.sum()), "n_far": int(far.sum()),
                                  "by_cam": {cid: int((own[far][iso] == cid).sum()) for cid in recs}}
    nbr = tree.query_ball_point(P, STREAK_RAY_X, return_length=True) - 1
    isolated = nbr < STREAK_K
    for cid, r in recs.items():
        sel = own == cid
        H, W = r["d"].shape
        bad = np.zeros((H, W), bool)
        rng = np.full((H, W), np.nan)
        yx = pix[sel]
        rng[yx[:, 0], yx[:, 1]] = R[sel]
        bad[yx[:, 0], yx[:, 1]] = isolated[sel]
        bad &= ~_surface_ok(r["d"], r["gtd"], r.get("cap", math.inf))
        runs = []
        for c in np.nonzero(bad.any(0))[0]:
            col = np.concatenate([[False], bad[:, c], [False]])
            dif = np.diff(col.astype(np.int8))
            for a0, a1 in zip(np.nonzero(dif == 1)[0], np.nonzero(dif == -1)[0]):
                seg = rng[a0:a1, c]
                span = float(np.nanmax(seg) - np.nanmin(seg))
                if span > STREAK_Z_M and float(np.nanmin(seg)) <= STREAK_START_MAX_M:
                    j = a0 + int(np.nanargmin(seg))
                    runs.append({"span_m": round(span, 2), "col": int(c), "rows": [int(a0), int(a1 - 1)],
                                 "start_row": int(j), "r_start_m": round(float(np.nanmin(seg)), 2),
                                 "r_end_m": round(float(np.nanmax(seg)), 2)})
        runs.sort(key=lambda z: -z["span_m"])
        out["rays"][cid] = {"count": len(runs), "max_span_m": runs[0]["span_m"] if runs else 0.0, "top": runs[:5],
                            "n_px": int(sel.sum())}
    out["colmono"] = {}
    for cid, r in recs.items():
        cm = col_monotonic(r)
        out["colmono"][cid] = {"count": len(cm), "top": cm[:5]}
    return out


def _raw_nn(p: Preview, P, cid, idx, c, pw, ph):
    """The net's raw 12x16 depth (x fs), nearest-upsampled to processing res (no inverse-depth interpolation)."""
    W0, H0 = c["resolution"]
    fs = (p.ns["fy_full"](c) * 192.0 / H0) / p.ns["FY_REF"]
    out = np.full((ph, pw), np.nan)
    sx = pw / W0
    k = 0
    while f"{cid}/{idx}/{k}/d" in P.files:
        x0, cw = P[f"{cid}/{idx}/{k}/x0"]
        a0, a1 = int(round(x0 * sx)), int(round((x0 + cw) * sx))
        out[:, a0:a1] = cv2.resize((P[f"{cid}/{idx}/{k}/d"] * fs).astype(np.float32), (a1 - a0, ph), interpolation=cv2.INTER_NEAREST)
        k += 1
    return out


def footprint_gap(rec) -> dict:
    """Source 1 (Tracer): drawn GT-ground px (|GT z| <= STATIC_MAX_HEIGHT_M) whose next row down is also drawn
    GT ground; gap = 3-D spacing of the two rendered points > FOOT_GAP_K x footprint (preview footprint(),
    capped at 6 d / fy). Per range bin and total."""
    v, qg, q, f = rec["v"], rec["qg"], rec["q"], rec["foot"]
    gnd = v & np.isfinite(rec["gtd"]) & (np.abs(qg[..., 2]) <= sf.STATIC_MAX_HEIGHT_M)
    pair = gnd[:-1] & gnd[1:]
    sp = np.linalg.norm(q[:-1] - q[1:], axis=-1)
    gap = pair & (sp > FOOT_GAP_K * f[:-1])
    Rg = np.hypot(qg[:-1, :, 0], qg[:-1, :, 1])
    out = {"n": int(pair.sum()), "k": int(gap.sum()), "frac": float(gap.sum() / pair.sum()) if pair.sum() else None}
    for lo, hi in FOOT_GAP_BINS:
        m = pair & (Rg >= lo) & (Rg < hi)
        out[f"{lo:g}-{hi:g}"] = (int((gap & m).sum()), int(m.sum()))
    return out


def coarse_row_bowl(rec) -> dict:
    """Source 2 (Tracer): wide height bowl. Median dz (z_render - z_GT) of drawn GT-road px per 12x16 coarse
    row; bowl = max - min over rows with >= RIP_MIN_PX px."""
    m = rec["v"] & rec["road"] & np.isfinite(rec["gtd"]) & (np.abs(rec["qg"][..., 2]) <= RIP_GT_Z)
    H = m.shape[0]
    dz = rec["q"][..., 2] - rec["qg"][..., 2]
    rows = {}
    for r in range(12):
        mm = m[r * H // 12:(r + 1) * H // 12]
        if mm.sum() >= RIP_MIN_PX:
            rows[r] = round(float(np.median(dz[r * H // 12:(r + 1) * H // 12][mm])), 3)
    return {"rows": rows, "bowl": (max(rows.values()) - min(rows.values())) if len(rows) >= 2 else None,
            "n": int(m.sum())}


def plane_pull(rec) -> dict:
    """Anti-plane-pull (wide, snap off): per range bin, median of z(final drawn depth) - z(raw upsampled net
    depth, the cam_frame depth before any snap/clamp/cut) on drawn GT-road px, and the upsample tolerance
    = median |z(raw upsampled) - z(raw 12x16 nearest)| on the same px."""
    raw, nn = rec.get("raw"), rec.get("raw_nn")
    if raw is None or nn is None:
        return {}
    m = rec["v"] & rec["road"] & np.isfinite(rec["gtd"]) & (np.abs(rec["qg"][..., 2]) <= RIP_GT_Z) & np.isfinite(raw) & np.isfinite(nn)
    Rg = np.hypot(rec["qg"][..., 0], rec["qg"][..., 1])
    rays_z = rec["q"][..., 2] - rec["cam_z"]
    with np.errstate(all="ignore"):
        per = rays_z / np.where(rec["d"] > 0, rec["d"], np.nan)      # z per unit planar depth along each px ray
    out = {}
    for lo, hi in RIP_DZ_BINS:
        k = m & (Rg >= lo) & (Rg < hi)
        if k.sum() < RIP_MIN_PX:
            out[f"{lo:g}-{hi:g}"] = None
            continue
        z_f = rec["q"][..., 2][k]
        z_r = rec["cam_z"] + per[k] * raw[k]
        z_n = rec["cam_z"] + per[k] * nn[k]
        out[f"{lo:g}-{hi:g}"] = {"pull": float(np.median(z_f - z_r)), "tol": float(np.median(np.abs(z_r - z_n))), "n": int(k.sum())}
    return out


def col_monotonic(rec, road_ok=False) -> list:
    """Streak along one column (Tracer f160 narrow / f200 car smear): consecutive drawn, non-road (net), off-GT-
    surface px in one column whose depth keeps rising going up the column (>= 80% of steps non-decreasing)
    by more than COLMONO_RISE_M. Returns runs (col, rows, depth range, GT median)."""
    d, v, g = rec["d"], rec["v"], rec["gtd"]
    bad = v & ~_surface_ok(d, g, rec.get("cap", math.inf))
    if not road_ok:
        bad &= ~rec["road"]
    runs = []
    for c in np.nonzero(bad.any(0))[0]:
        col = np.concatenate([[False], bad[:, c], [False]])
        dif = np.diff(col.astype(np.int8))
        for a0, a1 in zip(np.nonzero(dif == 1)[0], np.nonzero(dif == -1)[0]):
            seg = d[a0:a1, c][::-1]           # bottom -> top
            if seg.size < 3:
                continue
            rise = float(seg.max() - seg.min())
            up = np.diff(seg)
            if rise > COLMONO_RISE_M and np.mean(up >= 0) >= 0.8 and seg[-1] > seg[0]:
                gs = g[a0:a1, c]
                runs.append({"col": int(c), "rows": [int(a0), int(a1 - 1)], "d_min": round(float(seg.min()), 1),
                             "d_max": round(float(seg.max()), 1), "gt_med": round(float(np.nanmedian(gs)), 1) if np.isfinite(gs).any() else None})
    runs.sort(key=lambda z: -(z["d_max"] - z["d_min"]))
    return runs


def gt_consistency(rec) -> dict:
    """Eval-only: drawn px whose depth exceeds GT (capped at the pipeline's far cap; the FARTHEST GT in the px's
    3x3 neighbourhood, as _surface_ok, so the GT path's own half-pixel resample at silhouettes is not counted) by
    >= GC_RATIO x and > GC_ABS m, i.e. GT has a nearer surface on that ray. No sky exemption (sky is never drawn on head)."""
    g = np.where(np.isfinite(rec["gtd"]), np.minimum(rec["gtd"], rec.get("cap", math.inf)), -np.inf)
    gmax = cv2.dilate(g.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.float64)   # farthest GT in the 3x3 (as _surface_ok)
    with np.errstate(invalid="ignore"):
        bad = rec["v"] & np.isfinite(rec["gtd"]) & (rec["d"] >= GC_RATIO * gmax) & (rec["d"] > gmax + GC_ABS)
    ys, xs = np.nonzero(bad)
    return {"count": int(bad.sum()), "n": int(rec["v"].sum()),
            "cols": [int(xs.min()), int(xs.max())] if xs.size else None, "rows": [int(ys.min()), int(ys.max())] if ys.size else None}


def false_removal(rec) -> dict:
    """Px with a GT surface past FR_MIN_M (GT range) where the raw net depth (cam_frame's depth before
    snap/clamp/cut) is within SURF_REL / SURF_ABS of GT: the share still drawn (valid)."""
    raw = rec.get("raw")
    if raw is None:
        return {}
    g = rec["gtd"]
    Rg = np.hypot(rec["qg"][..., 0], rec["qg"][..., 1])
    with np.errstate(invalid="ignore"):
        good = np.isfinite(g) & (Rg > FR_MIN_M) & np.isfinite(raw) & (np.abs(raw - g) <= np.maximum(SURF_REL * g, SURF_ABS))
    return {"n": int(good.sum()), "kept": int((good & rec["v"]).sum()),
            "frac": float((good & rec["v"]).sum() / good.sum()) if good.sum() else None}


def ring_a_holes(recs: dict, W: int = 960, H: int = 540) -> dict:
    """Ring-A gate: pose-B render (current-frame drawn points, preview footprint, splat radius W//120, gap fill on)
    vs GT ground (|GT z| <= STATIC_MAX_HEIGHT_M, GT range <= 30 m, splatted the same way): share of GT-ground
    chase px left unpainted, per chase-ground range band; attributed per cam = among the GT-ground px of that cam."""
    chase = sf.chase_camera(POSE_B_POS, POSE_B_PITCH, POSE_B_FOV, W, H, 0.0)
    rad = max(2, W // 120)
    def paint(pts, foot):
        if not len(pts):
            return np.zeros((H, W), bool)
        r = sf.compose(chase, pts, np.full((len(pts), 3), 128, np.uint8), foot, sky_rgb=None, ground_rgb=None,
                       max_radius_px=rad, fill=True)
        return r.masks["observed"] | r.masks["filled"]
    rp, rf, gm = [], [], {}
    for cid, r in recs.items():
        ys, xs = np.nonzero(r["v"])
        q = r["q"][ys, xs]
        k = ego_box_filter(q)
        rp.append(q[k]); rf.append(r["foot"][ys, xs][k])
        gq = r["qg"]
        m = np.isfinite(r["gtd"]) & (np.abs(gq[..., 2]) <= sf.STATIC_MAX_HEIGHT_M) & (np.hypot(gq[..., 0], gq[..., 1]) <= 30.0)
        ys, xs = np.nonzero(m)
        qq = gq[ys, xs]
        kk = ego_box_filter(qq)
        fg = r.get("foot_gt")
        gf = fg[ys, xs][kk] if fg is not None else np.full(int(kk.sum()), 0.3)
        gm[cid] = paint(qq[kk], gf)
    drawn = paint(np.concatenate(rp) if rp else np.zeros((0, 3)), np.concatenate(rf) if rf else np.zeros(0))
    rays = sf._pixel_rays(chase)
    dz = rays[..., 2]
    down = dz < -1e-9
    t = np.where(down, -chase.pos[2] / np.where(down, dz, -1.0), np.nan)
    rng = np.hypot(chase.pos[0] + t * rays[..., 0], chase.pos[1] + t * rays[..., 1])
    gall = np.zeros((H, W), bool)
    for m in gm.values():
        gall |= m
    out = {}
    for lo, hi in HOLE_BANDS:
        b = gall & (rng >= lo) & (rng < hi)
        hole = b & ~drawn
        out[f"{lo:g}-{hi:g}"] = {"frac": float(hole.sum() / b.sum()) if b.sum() else None, "n": int(b.sum()), "k": int(hole.sum()),
                                 "by_cam": {cid: [int((m & hole & (rng >= lo) & (rng < hi)).sum()), int((m & (rng >= lo) & (rng < hi)).sum())] for cid, m in gm.items()}}
    return out


def _wide_snap_off(p: Preview) -> None:
    """Michael's pick: road snap on every cam except wide. Uses ROAD_SNAP_CAMS when the code has it
    (module or preview), else wraps road_snap for the wide CameraPose only (other cams unchanged)."""
    if "ROAD_SNAP_CAMS" in p.ns:
        p.ns["ROAD_SNAP_CAMS"] = tuple(c for c in p.ns["ROAD_SNAP_CAMS"] if c != "wide")
        return
    hook = p.ns["S"]
    real = object.__getattribute__(hook, "_real")
    h = object.__getattribute__(hook, "_hook")
    if hasattr(real, "ROAD_SNAP_CAMS"):
        object.__setattr__(hook, "ROAD_SNAP_CAMS", tuple(c for c in real.ROAD_SNAP_CAMS if c != "wide"))
        return
    wide = p.cam["wide"]

    def road_snap(*a, **k):
        cam = a[0] if a else k["cam"]
        depth = np.asarray(a[1] if len(a) > 1 else k["depth"], np.float64)
        h["pre"] = depth.copy()
        h["calls"].append(("road_snap", getattr(cam, "cam_id", id(cam))))
        if cam is wide:
            m = np.zeros(depth.shape, bool)
            h["mask"] = m
            return depth.copy(), m
        out, m = real.road_snap(*a, **k)
        h["mask"] = m
        return out, m

    object.__setattr__(hook, "road_snap", road_snap)


def rs_collect(mode: str, frames, info_in_guard: bool = False) -> dict:
    """{frame: {cid: rs_record}} through the runtime cam_frame. mode: 'on' (head as is), 'off' (wide snap off),
    'gt' (GT depth through the same cam_frame path, eval only: the floor)."""
    clip, preds, gt_root = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    ov = {"depth_src": "gt", "gt_depth_root": str(gt_root)} if mode == "gt" else None
    p = Preview(ov, root=clip, P=P, cams=cams)
    if mode == "off":
        _wide_snap_off(p)
    avail = {cid: pred_frames(P, cid) for cid in p.cam}
    used = {cid: [i for i in (nearest_idx(avail.get(cid) or [], f) for f in frames) if not in_seal_guard(i)] for cid in p.cam}
    if not info_in_guard:   # only the opt-in INFO f200 fixture may bypass the guard band (never sealed frames)
        rs_frames_ok(frames, used)
    else:
        assert not set(frames) & set(HELDOUT2_SEALED) and not set(frames) & set(RS_BURNED), frames
    out = {}
    for f in frames:
        out[f] = {}
        for cid in RS_CAMS:
            idx = nearest_idx(avail.get(cid) or [], f)
            if idx is None:
                continue
            if in_seal_guard(idx) and not info_in_guard:   # seal guard band: e.g. rear idx 107 at f110 is never measured
                print(f"  seal guard: f{f} {cid} pred idx {idx} in {SEAL_GUARD_RANGES} -> not measured", flush=True)
                continue
            fr = p.run(cid, None, None, idx=idx)
            if fr is None:
                continue
            _, d, v, _ = fr
            c = cams[cid]
            pw, ph = p.ns["proc_res"](c)
            gtd = _load_gt_depth(gt_root, cid, idx, c, pw, ph)
            if gtd is None:
                continue
            pre = p.snap_hook.get("pre")
            out[f][cid] = rs_record(p.cam[cid], d, v, gtd, _net_road(P, cid, idx, c["resolution"][0], pw, ph),
                                    cap=sf.depth_cap_m(c["far_m"]))
            out[f][cid].update(idx=idx, foot=p.ns["footprint"](p.cam[cid], np.where(np.isfinite(d), d, 0.0)),
                               foot_gt=p.ns["footprint"](p.cam[cid], np.where(np.isfinite(gtd), gtd, 0.0)),
                               raw=pre, raw_nn=_raw_nn(p, P, cid, idx, c, pw, ph) if mode != "gt" else None,
                               snap=None if p.last_snap is None else np.array(p.last_snap, bool, copy=True), cam=p.cam[cid])
    return out


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_ripple(rows: dict, label: str) -> None:
    for f, per in rows.items():
        for cid, m in per.items():
            print(f"  ripple {label} f{f} {cid}: n_px={m['n_px']} n_bins={m['n_bins']} std={_fmt(m['std'])} "
                  f"ptp_5_15={_fmt(m['ptp_5_15'])} pfrac={_fmt(m['pfrac'])} zc={_fmt(m['zc'])} "
                  + " ".join(f"|dz|{lo:g}-{hi:g}={_fmt(m[f'dz_{lo:g}_{hi:g}'])}(n={m[f'n_{lo:g}_{hi:g}']})"
                             for lo, hi in RIP_DZ_BINS), flush=True)


def print_streak(rows: dict, label: str) -> None:
    for f, s in rows.items():
        iso = " ".join(f"Y{Y:g}/X{X:g}={v['total']}/{v['n_far']}" for (Y, X), v in s["iso"].items())
        print(f"  streak {label} f{f}: n_pts={s['n_pts']} isolated(k<{STREAK_K}) {iso}", flush=True)
        for cid, r in s["rays"].items():
            if r["count"]:
                t = r["top"][0]
                print(f"    rays {label} f{f} {cid}: {r['count']} runs > {STREAK_Z_M:g} m (n_px={r['n_px']}), max {r['max_span_m']} m: "
                      f"col {t['col']} rows {t['rows'][0]}-{t['rows'][1]}, starts row {t['start_row']} at {t['r_start_m']} m -> {t['r_end_m']} m",
                      flush=True)
        for cid, r in s.get("colmono", {}).items():
            if r["count"]:
                t = r["top"][0]
                print(f"    colmono {label} f{f} {cid}: {r['count']} runs; top col {t['col']} rows {t['rows'][0]}-{t['rows'][1]} "
                      f"depth {t['d_min']}->{t['d_max']} m (GT median {t['gt_med']} m)", flush=True)


def ripple_gate(rows: dict, label: str) -> list:
    bad = []
    for f, per in rows.items():
        for cid, m in per.items():
            for k, lim in (RIPPLE_LIMITS.get(cid) or {}).items():
                if lim is None or m.get(k) is None:
                    continue
                if m[k] > lim:
                    bad.append(f"{label} f{f} {cid} {k} {_fmt(m[k])} > {_fmt(lim)}"
                               + (f" (n={m['n_' + k[3:]]})" if k.startswith("dz_") else f" (n_bins={m['n_bins']}, n_px={m['n_px']})"))
    return bad


def streak_gate(rows: dict, label: str, recs: dict | None = None) -> list:
    """GATES (Skeptic): (a) isolated far points pooled over cams <= ISO_LIMITS (no sky exemption); (b) eval-only
    GT-consistency count per cam <= GC_LIMITS. Ray runs and column-monotonic runs are printed as INFO."""
    bad = []
    for f, s_ in rows.items():
        for (Y, X), v in s_["iso"].items():
            lim = ISO_LIMITS.get((Y, X))
            if lim is not None and v["total"] > lim:
                bad.append(f"{label} f{f} isolated Y{Y:g}/X{X:g} {v['total']} > {lim} (n_far={v['n_far']}, by cam {v['by_cam']})")
        for cid, r in ((recs or {}).get(f) or {}).items():
            gc = gt_consistency(r)
            print(f"    GT-consistency {label} f{f} {cid}: {gc['count']}/{gc['n']} drawn px >= {GC_RATIO}x GT and > GT+{GC_ABS:g} m "
                  f"(cols {gc['cols']}, rows {gc['rows']}; limit {GC_LIMITS.get(cid)})", flush=True)
            if GC_LIMITS.get(cid) is not None and gc["count"] > GC_LIMITS[cid]:
                bad.append(f"{label} f{f} {cid} GT-consistency {gc['count']} > {GC_LIMITS[cid]} (cols {gc['cols']}, rows {gc['rows']})")
    return bad


_RS_CACHE: dict = {}


def _rs(mode, frames):
    key = (mode, tuple(frames))
    if key not in _RS_CACHE:
        _RS_CACHE[key] = rs_collect(mode, frames)
    return _RS_CACHE[key]


RIP_DZ_KEYS = ("dz_0_5", "dz_5_10", "dz_10_15", "dz_15_30")
RIP_PERIOD_KEYS = ("std", "ptp_5_15", "pfrac", "zc")


def _ripple_check(cams, keys, label) -> None:
    data = _rs("off", RS_VERIFY_FRAMES)
    rows = {f: {cid: ripple_metrics(r) for cid, r in per.items() if cid in cams} for f, per in data.items()}
    print_ripple(rows, label)
    bad = [b for b in ripple_gate(rows, label) if any(f" {k} " in b for k in keys)]
    assert not bad, "; ".join(bad)


def check_clip015_ripple_wide_dz() -> None:
    """Ripple (rings) gate, wide snap OFF, verify frames: per-range median |dz| on GT-road drawn px (0-5/5-10/10-15/
    15-30 m) <= RIPPLE_LIMITS['wide'] (the wide coarse-row ripple dz)."""
    if not _need_clip("clip015_ripple_wide_dz"):
        return SKIPPED
    _ripple_check(("wide",), RIP_DZ_KEYS, "snapOFF-wide-dz")


def check_clip015_ripple_wide_periodicity() -> None:
    """Ripple (rings) gate, wide snap OFF, verify frames: detrended-profile std / ptp_5_15 / pfrac / zero crossings."""
    if not _need_clip("clip015_ripple_wide_periodicity"):
        return SKIPPED
    _ripple_check(("wide",), RIP_PERIOD_KEYS, "snapOFF-wide-period")


def check_clip015_ripple_other_cams() -> None:
    """Ripple gate on the non-wide cams (unchanged by the wide snap toggle), same metrics and rule."""
    if not _need_clip("clip015_ripple_other_cams"):
        return SKIPPED
    _ripple_check(tuple(c for c in RS_CAMS if c != "wide"), RIP_DZ_KEYS + RIP_PERIOD_KEYS, "snapOFF-others")


def check_clip015_streak_wide_snap_off() -> None:
    """Isolated-point / ray-streak gate on the verify frames with wide snap OFF (no sky exemption)."""
    if not _need_clip("clip015_streak_wide_snap_off"):
        return SKIPPED
    data = _rs("off", RS_VERIFY_FRAMES)
    rows = {f: streak_metrics(per) for f, per in data.items()}
    print_streak(rows, "snapOFF")
    bad = streak_gate(rows, "snapOFF", data)
    assert not bad, "; ".join(bad[:12]) + (f" (+{len(bad) - 12} more)" if len(bad) > 12 else "")


def check_snap_hook_binds_depth_by_name() -> None:
    """The snap hook must bind ``depth`` / ``road`` by parameter name, not by positional a[1].

    snap_road_for_cam's signature is (cam_id, cam, depth, road): a[1] is a CameraPose, so the old hook
    crashed with TypeError: float() argument ... not 'CameraPose'. This check builds a function with that
    exact signature, passes a non-float cam object, and asserts the hook reads the depth array. A mutant
    that still uses positional a[1] is caught (it would treat cam as depth). Missing depth/road params fail loudly.
    """
    class Cam:
        name = "narrow"

    def snap_road_for_cam(cam_id, cam, depth, road):  # exact runtime signature
        return np.array(depth, np.float64, copy=True), np.asarray(road, bool)

    cam = Cam()
    depth = np.arange(12, dtype=np.float64).reshape(3, 4)
    road = np.ones((3, 4), bool)
    args = ("narrow", cam, depth, road)
    dep, rd, ba = snap_hook_bound_args(snap_road_for_cam, args, {})
    assert dep is depth, f"depth bound to {type(dep).__name__}, not the array (cam={type(cam).__name__})"
    assert rd is road
    assert ba.arguments["cam"] is cam and ba.arguments["cam_id"] == "narrow"
    pre = np.array(dep, np.float64, copy=True)
    assert np.array_equal(pre, depth)

    def mutant_positional_a1(orig, a, k):  # the old hook
        return a[1] if len(a) > 1 else k.get("depth")

    mut = mutant_positional_a1(snap_road_for_cam, args, {})
    assert mut is cam, "mutant fixture no longer takes a[1] as depth (test bug)"
    try:
        np.array(mut, np.float64, copy=True)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("mutant positional a[1] was not caught (cam coerced to an array)")
    # also: a function with no depth/road param must fail loudly
    def no_depth(cam, mask):
        return cam, mask
    try:
        snap_hook_bound_args(no_depth, (cam, road), {})
    except TypeError as e:
        assert "depth" in str(e) and "road" in str(e), e
    else:
        raise AssertionError("missing depth/road parameters were not rejected")
    # and the Preview hook uses the binder: wrap a fake snap_road_for_cam through _SHook
    p = Preview()
    hook = p.ns["S"]
    object.__setattr__(hook, "_real", types.SimpleNamespace(snap_road_for_cam=snap_road_for_cam,
                                                            road_snap=sf.road_snap))
    p.snap_hook.update(mask=None, pre=None, calls=[], _in=0)
    out, m = hook.snap_road_for_cam("narrow", cam, depth, road)
    assert np.array_equal(out, depth) and np.array_equal(p.snap_hook["pre"], depth), p.snap_hook
    assert p.snap_hook["calls"] and p.snap_hook["calls"][0][0] == "snap_road_for_cam"
    print(f"  hook bound depth by name for snap_road_for_cam(cam_id, cam={type(cam).__name__}, depth, road); "
          f"mutant a[1] caught (got {type(mut).__name__}); missing-param TypeError ok", flush=True)


def check_snap_hook_hit() -> None:
    """Harness self-test (Redesigner 2026-11-26): the snap hook must be hit by the runtime's real per-cam snap
    path, or gate (b) passes vacuously. Dev f40, every cam: >= 1 hooked snap call per cam whose frame has net
    road px; if the runtime has snap_road_for_cam it must be among the hooked calls."""
    if not _need_clip("snap_hook_hit"):
        return SKIPPED
    clip, preds, _ = clip_paths()
    cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
    P = np.load(preds)
    p = Preview(root=clip, P=P, cams=cams)
    bad, seen = [], {}
    for cid in p.cam:
        idx = nearest_idx(pred_frames(P, cid), 40)
        if idx is None:
            continue
        p.snap_hook["calls"] = []
        fr = p.run(cid, None, None, idx=idx)
        if fr is None:
            continue
        c = cams[cid]
        pw, ph = p.ns["proc_res"](c)
        has_road = bool(_net_road(P, cid, idx, c["resolution"][0], pw, ph).any())
        calls = list(p.snap_hook["calls"])
        seen[cid] = (len(calls), has_road, int(p.last_snap.sum()) if p.last_snap is not None else None)
        if p.a.road_snap and has_road and not calls:
            bad.append(f"{cid}: road px but 0 hooked snap calls")
        if hasattr(sf, "snap_road_for_cam") and not any(n == "snap_road_for_cam" for n, _ in calls):
            bad.append(f"{cid}: runtime has snap_road_for_cam but the hook never saw it")
    print(f"  snap hook calls per cam f40 (calls, net road present, snapped px): {seen}", flush=True)
    assert not bad, "; ".join(bad)


def info_clip015_footprint_row_gaps() -> None:
    """INFO (Skeptic: circular vs the fix). Ripple source 1 (Tracer): footprint striping. Fraction of drawn
    GT-ground adjacent row pairs with 3-D spacing > 1.05 x footprint, per cam and frame (reference 5%, the
    committed P gate). Snap-independent. The gate for ring A is check_clip015_ringA_poseB_holes."""
    if not _need_clip("clip015_footprint_row_gaps"):
        return SKIPPED
    bad = []
    for f, per in _rs("off", RS_VERIFY_FRAMES).items():
        for cid, r in per.items():
            fg = footprint_gap(r)
            print(f"  footgap f{f} {cid}: {fg['k']}/{fg['n']} = {_fmt(fg['frac'], 4)} "
                  + " ".join(f"{lo:g}-{hi:g}:{fg[f'{lo:g}-{hi:g}'][0]}/{fg[f'{lo:g}-{hi:g}'][1]}" for lo, hi in FOOT_GAP_BINS), flush=True)
            if fg["frac"] is not None and fg["frac"] > FOOT_GAP_MAX:
                bad.append(f"f{f} {cid} {fg['frac']:.2%} ({fg['k']}/{fg['n']})")
    if bad:   # INFO (Skeptic: circular vs the fix that grows the splat to the next point); the ring-A gate is the drawn hole fraction
        print(f"  INFO footgap over {FOOT_GAP_MAX:.0%}: " + "; ".join(bad), flush=True)


def check_clip015_wide_coarse_row_bowl() -> None:
    """Ripple source 2 (Tracer): wide height bowl, wide snap OFF. Per 12x16 coarse row median dz of drawn GT-road
    px; bowl (max - min over rows) <= WIDE_BOWL_LIMIT (rule: no worse than the better of 783fedd snap ON and v2
    on the threshold frames, + GT-path floor)."""
    if not _need_clip("clip015_wide_coarse_row_bowl"):
        return SKIPPED
    bad = []
    for f, per in _rs("off", RS_VERIFY_FRAMES).items():
        if "wide" not in per:
            continue
        b = coarse_row_bowl(per["wide"])
        print(f"  bowl f{f} wide: n={b['n']} rows(median dz)={b['rows']} bowl={_fmt(b['bowl'])} (limit {WIDE_BOWL_LIMIT})", flush=True)
        if b["bowl"] is not None and WIDE_BOWL_LIMIT is not None and b["bowl"] > WIDE_BOWL_LIMIT:
            bad.append(f"f{f} wide bowl {b['bowl']:.3f} > {WIDE_BOWL_LIMIT} rows {b['rows']}")
    assert not bad, "; ".join(bad)


def _plane_pull_check(bins) -> None:
    bad = []
    for f, per in _rs("off", RS_VERIFY_FRAMES).items():
        if "wide" not in per:
            continue
        pp = plane_pull(per["wide"])
        print(f"  plane-pull f{f} wide: " + " ".join(
            f"{k}: pull {_fmt(v['pull'])} tol(this frame) {_fmt(v['tol'])} n={v['n']} limit {PLANE_PULL_TOL.get(k)}" if v else f"{k}: n/a"
            for k, v in pp.items() if k in bins), flush=True)
        for k, v in pp.items():
            if k in bins and v and PLANE_PULL_TOL.get(k) is not None and abs(v["pull"]) > PLANE_PULL_TOL[k]:
                bad.append(f"f{f} wide {k} |pull| {abs(v['pull']):.3f} > {PLANE_PULL_TOL[k]} (n={v['n']})")
    assert not bad, "; ".join(bad)


def check_wide_snap_off_no_plane_pull_0_5m() -> None:
    """The 0-5 m bin of the anti-plane-pull guard (same definition and tolerance). Fails on head: ground_clamp
    (clamp margin 1.0) lifts below-road net depth onto the road plane at 0-5 m."""
    if not _need_clip("wide_snap_off_no_plane_pull_0_5m"):
        return SKIPPED
    _plane_pull_check(("0-5",))


def check_wide_snap_off_no_plane_pull() -> None:
    """GUARD (GVD ruling, must pass): with wide snap OFF, the final drawn wide depth on GT-road px matches the raw
    net depth (cam_frame's upsampled depth before snap/clamp/cut): per range bin |median dz(final - raw)| <=
    PLANE_PULL_TOL[bin] (upsample tolerance, measured on head: median |z(raw upsampled) - z(raw 12x16 nearest)|,
    worst threshold frame). Catches any plane flatten / ground-grid pull on wide. Verify frames f40/f110/f160.
    Bins 5-10/10-15/15-30 m here (pass on head); 0-5 m is check_wide_snap_off_no_plane_pull_0_5m (fails on head:
    ground_clamp lifts below-road net depth to the plane)."""
    if not _need_clip("wide_snap_off_no_plane_pull"):
        return SKIPPED
    _plane_pull_check(("5-10", "10-15", "15-30"))


def check_clip015_ringA_poseB_holes() -> None:
    """Ring-A GATE (Skeptic): pose-B drawn-image hole fraction of GT ground (0-30 m) per range band, wide snap OFF,
    verify frames; <= RINGA_LIMITS (rule on threshold frames). Per-cam attribution printed."""
    if not _need_clip("clip015_ringA_poseB_holes"):
        return SKIPPED
    bad = []
    for f, per in _rs("off", RS_VERIFY_FRAMES).items():
        h = ring_a_holes(per)
        for band, v in h.items():
            att = " ".join(f"{c}:{k}/{n}" for c, (k, n) in v["by_cam"].items() if n)
            print(f"  ringA f{f} {band} m: {v['k']}/{v['n']} = {_fmt(v['frac'], 4)} (limit {RINGA_LIMITS.get(band)}) by cam {att}", flush=True)
            if v["frac"] is not None and RINGA_LIMITS.get(band) is not None and v["frac"] > RINGA_LIMITS[band]:
                bad.append(f"f{f} {band} m holes {v['frac']:.4f} > {RINGA_LIMITS[band]} ({v['k']}/{v['n']})")
    assert not bad, "; ".join(bad)


def check_clip015_false_removal() -> None:
    """FALSE-REMOVAL guard (must pass on head): px with a GT surface past 30 m whose raw net depth is within
    10% / 0.5 m of GT; the drawn share, pooled over the threshold frames, wide snap OFF, per cam >= FR_LIMITS
    (head - 2 pp). Catches a fix that deletes real far hills / buildings / terrain."""
    if not _need_clip("clip015_false_removal"):
        return SKIPPED
    bad = []
    pooled = {}
    for f, per in _rs("off", RS_THRESH_FRAMES).items():
        for cid, r in per.items():
            fr = false_removal(r)
            k, n = pooled.get(cid, (0, 0))
            pooled[cid] = (k + fr.get("kept", 0), n + fr.get("n", 0))
    for cid, (k, n) in pooled.items():
        lim = FR_LIMITS.get(cid)
        print(f"  false-removal {cid}: kept {k}/{n} = {_fmt(k / n if n else None, 4)} (limit {lim})", flush=True)
        if lim is not None and n and k / n < lim:
            bad.append(f"{cid} kept {k / n:.4f} < {lim} ({k}/{n})")
    assert not bad, "; ".join(bad)


def check_runtime_never_reads_gt() -> None:
    """GT is eval-only (Skeptic): no runtime module (python/**) mentions gt_eval_only / GT depth roots; in the
    preview, the GT depth root is read only inside `if a.depth_src == "gt"` (eval mode) and the road / sky class
    used by snap comes from the net's semantic preds (P[.../s]); the test's own GT masks never enter cam_frame."""
    bad = []
    for f in sorted((ROOT / "python").rglob("*.py")):
        t = f.read_text(encoding="utf-8", errors="ignore")
        for k in ("gt_eval_only", "gt_depth_root", "GVD_RV_CLIP015_GT"):
            if k in t:
                bad.append(f"{f.relative_to(ROOT)} mentions {k}")
    tree = ast.parse(PREVIEW.read_text(encoding="utf-8"))
    cf = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cam_frame")
    guarded = set()
    for n in ast.walk(cf):
        if isinstance(n, ast.If) and "depth_src" in ast.unparse(n.test) and '"gt"' in ast.unparse(n.test).replace("'", '"') and "==" in ast.unparse(n.test):
            guarded |= {id(x) for x in ast.walk(n)}
    for n in ast.walk(cf):
        if isinstance(n, ast.Attribute) and n.attr == "gt_depth_root" and id(n) not in guarded:
            bad.append(f"cam_frame reads a.gt_depth_root outside the depth_src == 'gt' branch (line {n.lineno})")
    src = ast.unparse(cf)
    if "road[:, a0:a1] |=" not in src or "isin(s, S.ROAD_CLASS_IDS)" not in src:
        bad.append("cam_frame road class no longer visibly from the net's semantic preds (P[.../s])")
    assert not bad, "; ".join(bad)


SNAP_REJECT_Z = 0.30
# Literal contract (Skeptic 15:11): ROAD_SNAP_CAMS must equal this set. Imported from the runtime when
# present; never used as the snap-cam list by itself (a dropped cam must not silently shrink the check).
EXPECTED_ROAD_SNAP_CAMS = frozenset({"main", "pillarL", "pillarR", "repeatL", "repeatR", "rear"})


def runtime_road_snap_cams(src=None):
    """The snap-cam set from the runtime (python.viz.rv_stable_fill / preview ns). None if the name is absent (this head)."""
    if src is not None:
        return frozenset(src.ROAD_SNAP_CAMS) if hasattr(src, "ROAD_SNAP_CAMS") else None
    if hasattr(sf, "ROAD_SNAP_CAMS"):
        return frozenset(sf.ROAD_SNAP_CAMS)
    return None


def assert_road_snap_cams(cams) -> None:
    """ROAD_SNAP_CAMS must equal the literal set; dropping a cam (e.g. pillarR) must FAIL, not shrink the check."""
    got = frozenset(cams)
    assert got == EXPECTED_ROAD_SNAP_CAMS, (
        f"ROAD_SNAP_CAMS {sorted(got)} != literal {sorted(EXPECTED_ROAD_SNAP_CAMS)} "
        f"(dropped {sorted(EXPECTED_ROAD_SNAP_CAMS - got)}; extra {sorted(got - EXPECTED_ROAD_SNAP_CAMS)})"
    )


def resolve_road_snap_cams(src=None):
    """The snap-cam set used by the reject check.

    The literal fallback is allowed only on heads *without* snap_road_for_cam. If
    ``hasattr(src, 'snap_road_for_cam')`` is true, a missing ROAD_SNAP_CAMS is a FAIL
    (clear message), not a fallback. When the set is present it must equal the literal.
    """
    src = sf if src is None else src
    cams = runtime_road_snap_cams(src)
    if hasattr(src, "snap_road_for_cam") and cams is None:
        raise AssertionError(
            "snap_road_for_cam is present but ROAD_SNAP_CAMS is missing; "
            "the literal fallback is only allowed on heads without snap_road_for_cam"
        )
    if cams is None:
        return EXPECTED_ROAD_SNAP_CAMS
    assert_road_snap_cams(cams)
    return cams


def _snap_scene(cam):
    rz = sf._pixel_rays(cam)[..., 2]
    h = float(np.asarray(cam.pos)[2])
    down = rz < -1e-6
    hit = np.where(down, h / np.where(down, -rz, 1.0), np.inf)
    cand = down & (hit < 12.0) & (hit > 2.0)
    if not cand.any():
        return None
    H, W = cand.shape
    zpred = np.where(np.arange(W)[None, :] < W // 2, 0.5, 0.05) * np.ones((H, 1))
    d = np.where(cand, (h - zpred) / np.where(down, -rz, 1.0), 50.0)
    return {"d": d, "road": np.ones((H, W), bool), "cand": cand, "zpred": zpred, "raised": cand & (zpred > SNAP_REJECT_Z),
            "low": cand & (zpred < SNAP_REJECT_Z)}


def _snap_fns():
    """The runtime snap entry points present on this head (road_snap always; snap_road_for_cam if it exists)."""
    fns = [("road_snap", lambda cam, cid, d, road: sf.road_snap(cam, d, road), False)]
    if hasattr(sf, "snap_road_for_cam"):
        fns.append(("snap_road_for_cam", lambda cam, cid, d, road: sf.snap_road_for_cam(cid, cam, d, road), True))
    return fns


def snap_rejection_violations(snap_fn, snap_cams) -> list:
    """Reject+snap assertions only for cams in snap_cams (ROAD_SNAP_CAMS). Behavioural: a road-class px whose
    pred point sits 0.5 m above the ground must NOT be snapped; a road px at 0.05 m must still be snapped."""
    p = Preview()
    bad = []
    for cid in snap_cams:
        if cid not in p.cam:
            continue
        sc = _snap_scene(p.cam[cid])
        if sc is None:
            continue
        out, mask = snap_fn(p.cam[cid], cid, sc["d"].copy(), sc["road"])
        moved_raised = int((np.asarray(mask, bool) & sc["raised"]).sum())
        kept_low = int((np.asarray(mask, bool) & sc["low"]).sum())
        if moved_raised:
            bad.append(f"{cid}: {moved_raised}/{int(sc['raised'].sum())} road px with pred z 0.5 m snapped to the plane")
        if sc["low"].sum() > 50 and kept_low < 0.5 * int(sc["low"].sum()):
            bad.append(f"{cid}: only {kept_low}/{int(sc['low'].sum())} road px with pred z 0.05 m snapped (rejection not one-sided)")
    return bad


def snap_unlisted_cam_violations(snap_fn, snap_cams) -> list:
    """For every cam NOT in ROAD_SNAP_CAMS (narrow, and any other), snap_road_for_cam must change zero road px."""
    p = Preview()
    bad = []
    for cid, cam in p.cam.items():
        if cid in snap_cams:
            continue
        sc = _snap_scene(cam)
        if sc is None:
            continue
        src = sc["d"].copy()
        out, mask = snap_fn(cam, cid, src, sc["road"])
        n = int(np.asarray(mask, bool).sum())
        changed = int((~np.isclose(np.asarray(out, np.float64), src, equal_nan=True)).sum())
        if n or changed:
            bad.append(f"{cid}: snap_road_for_cam changed {changed} px / snapped {n} (must be 0; cam is not in ROAD_SNAP_CAMS)")
    return bad


def check_snap_rejects_raised_pred_z() -> None:
    """Skeptic (13:59 / 15:11): the one-sided pred-z > 0.30 m rejection must be in the snap path for cams in
    ROAD_SNAP_CAMS (imported from the runtime; must equal the literal
    {main, pillarL, pillarR, repeatL, repeatR, rear} so dropping a cam cannot silently shrink the check).
    The literal fallback is allowed only when snap_road_for_cam is absent; if that entry point exists,
    a missing ROAD_SNAP_CAMS is a FAIL, not a fallback. Behavioural (synthetic pred z 0.5 vs 0.05 m) +
    mutants: no rejection; a snap that acts on narrow; a ROAD_SNAP_CAMS that drops pillarR; snap_road_for_cam
    present with ROAD_SNAP_CAMS deleted. Positive control: stub snap_road_for_cam + literal set, set-equality
    passes (rejection aside). For every cam not in the set, snap_road_for_cam must change zero road px.
    Scoping correction, not a loosening. Expected FAIL on this head (road_snap has no rejection; normal FAIL,
    Fix item for Redesigner) with a raised-pred-z message on a snap cam, not narrow or a missing set."""
    # mutant: dropping pillarR from ROAD_SNAP_CAMS must FAIL the set-equality assertion
    dropped = EXPECTED_ROAD_SNAP_CAMS - {"pillarR"}
    try:
        assert_road_snap_cams(dropped)
    except AssertionError as e:
        dropped_caught = True
        print(f"  mutant drop pillarR from ROAD_SNAP_CAMS: caught ({e})", flush=True)
    else:
        dropped_caught = False
    assert dropped_caught, "mutant that drops pillarR from ROAD_SNAP_CAMS was not caught"

    def _stub_snap_road_for_cam(cam_id, cam, depth, road):  # exact runtime signature
        return depth, np.zeros(np.shape(depth), bool)

    def _snap_module_snapshot():
        return (hasattr(sf, "snap_road_for_cam"), getattr(sf, "snap_road_for_cam", None),
                hasattr(sf, "ROAD_SNAP_CAMS"), getattr(sf, "ROAD_SNAP_CAMS", None))

    def _restore_snap_module(had_fn, fn, had_set, sset):
        if had_fn:
            sf.snap_road_for_cam = fn
        elif hasattr(sf, "snap_road_for_cam"):
            delattr(sf, "snap_road_for_cam")
        if had_set:
            sf.ROAD_SNAP_CAMS = sset
        elif hasattr(sf, "ROAD_SNAP_CAMS"):
            delattr(sf, "ROAD_SNAP_CAMS")

    snap_state = _snap_module_snapshot()
    try:
        # mutant: snap_road_for_cam present, ROAD_SNAP_CAMS absent -> must FAIL, not fall back
        sf.snap_road_for_cam = _stub_snap_road_for_cam
        if hasattr(sf, "ROAD_SNAP_CAMS"):
            delattr(sf, "ROAD_SNAP_CAMS")
        try:
            resolve_road_snap_cams(sf)
            missing_caught, missing_msg = False, None
        except AssertionError as e:
            missing_caught, missing_msg = True, str(e)
        assert missing_caught, (
            "mutant: snap_road_for_cam present but ROAD_SNAP_CAMS absent was not caught (literal fallback used)"
        )
        print(f"  mutant missing ROAD_SNAP_CAMS with snap_road_for_cam: caught ({missing_msg})", flush=True)

        # positive control: stub snap_road_for_cam + ROAD_SNAP_CAMS equal to the literal -> set-equality passes
        sf.snap_road_for_cam = _stub_snap_road_for_cam
        sf.ROAD_SNAP_CAMS = tuple(sorted(EXPECTED_ROAD_SNAP_CAMS))
        ctrl = resolve_road_snap_cams(sf)
        assert_road_snap_cams(ctrl)
        print(f"  positive control stub snap_road_for_cam + ROAD_SNAP_CAMS == literal: set-equality passed ({sorted(ctrl)})",
              flush=True)
    finally:
        _restore_snap_module(*snap_state)
        # the head's real attributes must be back before any later snap call
        assert _snap_module_snapshot() == snap_state, "failed to restore sf snap attributes after mutants"

    snap_cams = resolve_road_snap_cams()
    if runtime_road_snap_cams() is not None:
        print(f"  ROAD_SNAP_CAMS (from runtime) {sorted(snap_cams)} == literal {sorted(EXPECTED_ROAD_SNAP_CAMS)}", flush=True)
    else:
        print(f"  ROAD_SNAP_CAMS absent on this head and snap_road_for_cam is absent; "
              f"reject+snap scoped to the literal fallback {sorted(snap_cams)}", flush=True)

    def mutant(cam, cid, d, road):   # rejection removed: snap decided on a ground-level depth, then applied to d
        rz = sf._pixel_rays(cam)[..., 2]
        h = float(np.asarray(cam.pos)[2])
        flat = np.where(rz < -1e-6, h / np.where(rz < -1e-6, -rz, 1.0), 1e4)
        _, m = sf.road_snap(cam, flat, road)
        return np.where(m, flat, d), m
    def reference(cam, cid, d, road):   # test-only positive control: road_snap + one-sided pred-z rejection
        rz = sf._pixel_rays(cam)[..., 2]
        z = float(np.asarray(cam.pos)[2]) + np.asarray(d, np.float64) * rz
        out, m = sf.road_snap(cam, d, road)
        m = m & ~(z > SNAP_REJECT_Z)
        return np.where(m, out, d), m
    def mutant_snaps_narrow(cam, cid, d, road):  # scoping mutant: snaps a cam that is not in ROAD_SNAP_CAMS
        return sf.road_snap(cam, d, road)

    mut = snap_rejection_violations(mutant, snap_cams)
    assert mut, "mutant without the pred-z rejection was not caught"
    ref = snap_rejection_violations(reference, snap_cams)
    assert not ref, f"positive control (road_snap + rejection) flagged: {ref}"
    narrow_mut = snap_unlisted_cam_violations(mutant_snaps_narrow, snap_cams)
    assert narrow_mut, "mutant that snaps a cam outside ROAD_SNAP_CAMS (narrow) was not caught"
    print(f"  mutant (snaps unlisted cam) caught: {narrow_mut}", flush=True)

    bad = []
    for name, fn, is_per_cam in _snap_fns():
        v = snap_rejection_violations(fn, snap_cams)
        print(f"  snap rejection {name} (cams {sorted(snap_cams)}): " +
              ("; ".join(v) if v else "raised road px kept, ground road px snapped"), flush=True)
        bad += [f"{name} {x}" for x in v]
        if is_per_cam:
            u = snap_unlisted_cam_violations(fn, snap_cams)
            print(f"  unlisted cams under {name}: " + ("; ".join(u) if u else "zero road px changed"), flush=True)
            bad += [f"{name} {x}" for x in u]
    print(f"  mutant (no rejection) caught: {len(mut)} cams", flush=True)
    assert not bad, "; ".join(bad)


SNAPB_ATTR_CAMS = ("pillarR", "repeatR", "pillarL")


def snap_b_attribution(r) -> dict:
    """Gate (b) attribution for one cam/frame on the head render (snap ON for this cam), 0-15 m:
    bad = snapped & drawn & GT z > SNAP_BAD_Z (gate (b) numerator); good = snapped & drawn & GT z <= SNAP_BAD_Z.
    (i) bad/n; (ii) the same bad px with snap disabled = the raw (pre-snap) net depth: share whose raw pred z is
    < SNAP_BAD_Z (the raised px is already drawn at road level without snap) and < 0.15; raw z p50; snap shift p50;
    (iii) share of bad px whose pred px class is road (net seg says road where GT is raised).
    INFO: good px that a raw-pred-z > 0.30 rejection would wrongly drop."""
    snap, pre = r.get("snap"), r.get("raw")
    if snap is None or pre is None:
        return {"n": 0}
    cam = r["cam"]
    m = snap & r["v"] & np.isfinite(r["gtd"]) & np.isfinite(pre) & (pre > 0)
    ys, xs = np.nonzero(m)
    if not ys.size:
        return {"n": 0}
    qg = r["qg"][ys, xs]
    keep = np.hypot(qg[:, 0], qg[:, 1]) < 15.0
    ys, xs, qg = ys[keep], xs[keep], qg[keep]
    zr = sf.unproject(cam, pre, ys, xs)[:, 2]
    zf = r["q"][ys, xs, 2]
    bad = qg[:, 2] > SNAP_BAD_Z
    nb, n = int(bad.sum()), int(ys.size)
    out = {"n": n, "bad": nb, "frac": nb / n if n else None}
    if nb:
        out.update(raw_lt030=float((zr[bad] < SNAP_BAD_Z).mean()), raw_lt015=float((zr[bad] < 0.15).mean()),
                   raw_z_p50=float(np.median(zr[bad])), gt_z_p50=float(np.median(qg[bad, 2])),
                   shift_p50=float(np.median(np.abs(zf[bad] - zr[bad]))),
                   pred_road=float(r["road"][ys[bad], xs[bad]].mean()))
    good = ~bad
    out["good"] = int(good.sum())
    out["good_rej030"] = float((zr[good] > SNAP_BAD_Z).mean()) if good.any() else None
    return out


def info_snap_b_attribution() -> None:
    """INFO (Skeptic 13:52 / Tracer rv_snapb.txt): pillarR/repeatR gate (b) attribution on dev threshold frames
    (guard band respected; not f40). Verdict printed: net seg+depth if most bad px are pred-road AND already at
    road level (raw z < 0.30) before snap; snap if they are not."""
    if not _need_clip("snap_b_attribution"):
        return SKIPPED
    tot = {}
    for f, per in _rs("on", RS_THRESH_FRAMES).items():
        for cid in SNAPB_ATTR_CAMS:
            if cid not in per:
                continue
            a = snap_b_attribution(per[cid])
            t = tot.setdefault(cid, {"n": 0, "bad": 0, "lt030": 0.0, "lt015": 0.0, "road": 0.0, "good": 0, "rej": 0.0})
            t["n"] += a["n"]; t["bad"] += a.get("bad", 0); t["good"] += a.get("good", 0)
            if a.get("bad"):
                t["lt030"] += a["raw_lt030"] * a["bad"]; t["lt015"] += a["raw_lt015"] * a["bad"]; t["road"] += a["pred_road"] * a["bad"]
            if a.get("good_rej030") is not None:
                t["rej"] += a["good_rej030"] * a["good"]
            print(f"  snap(b) attr f{f} {cid} idx {per[cid]['idx']}: (i) bad {a.get('bad', 0)}/{a['n']} = {_fmt(a.get('frac'), 4)}"
                  + (f" | (ii) raw z<0.30 {a['raw_lt030']:.0%} raw z<0.15 {a['raw_lt015']:.0%} raw z p50 {a['raw_z_p50']:.2f} "
                     f"GT z p50 {a['gt_z_p50']:.2f} snap shift p50 {a['shift_p50']:.3f} m | (iii) pred road {a['pred_road']:.0%}"
                     if a.get("bad") else "")
                  + (f" | residual after a pred-z>0.30 rejection (bad & raw z<=0.30): {int(round(a['raw_lt030'] * a['bad']))}" if a.get("bad") else "")
                  + f" | INFO good px raw z>0.30 (wrongly rejected by a z>0.30 rule) {_fmt(a.get('good_rej030'), 3)} of {a.get('good', 0)}", flush=True)
    for cid, t in tot.items():
        if not t["bad"]:
            print(f"  snap(b) attr pooled {cid}: 0 bad px of {t['n']}", flush=True)
            continue
        lt030, lt015, road = t["lt030"] / t["bad"], t["lt015"] / t["bad"], t["road"] / t["bad"]
        verdict = ("net seg+depth (raised px already at road level before snap)" if lt030 >= 0.5 else
                   f"snap moves them: pred class is road on {road:.0%} (snap only acts on pred road) but raw net depth keeps "
                   f"{1 - lt030:.0%} of them >= {SNAP_BAD_Z} m before snap")
        print(f"  snap(b) attr pooled {cid}: bad {t['bad']}/{t['n']} = {t['bad'] / t['n']:.2%}; raw z<0.30 {lt030:.0%}, "
              f"raw z<0.15 {lt015:.0%}, pred road {road:.0%} -> {verdict}; good-px wrongly rejected by z>0.30 "
              f"{(t['rej'] / t['good']) if t['good'] else float('nan'):.1%} of {t['good']}", flush=True)


F200_FIXTURE = 200   # BURNED for thresholds (Tracer holdout); regression fixture only: narrow car smear rows 42-78, cols 61-213


def info_streak_fixture_f200_narrow_car_smear() -> None:
    """Regression fixture (fail-first only; f200 is burned and never sets a threshold): clip015 f200 narrow car
    ~20 m away (GT) smeared 32-254 m along its rays, rows 42-78 / cols 61-213 (Tracer). The committed streak
    gates (isolated pooled + GT-consistency), unchanged, are applied to f200; column-monotonic runs are INFO."""
    if os.environ.get("RV_PROPS_F200_INFO", "") != "1":   # f200 is inside the seal guard band (163-207): INFO, opt-in only
        skip_check("streak_fixture_f200_narrow_car_smear", "f200 in the seal guard band; INFO only, set RV_PROPS_F200_INFO=1")
        return SKIPPED
    if not _need_clip("streak_fixture_f200_narrow_car_smear"):
        return SKIPPED
    data = rs_collect("off", (F200_FIXTURE,), info_in_guard=True)
    rows = {F200_FIXTURE: streak_metrics(data[F200_FIXTURE])}
    print_streak(rows, "f200fixture")
    r = data[F200_FIXTURE].get("narrow")
    if r is not None:
        g = np.where(np.isfinite(r["gtd"]), np.minimum(r["gtd"], r["cap"]), -np.inf)
        gmax = cv2.dilate(g.astype(np.float32), np.ones((3, 3), np.uint8))
        with np.errstate(invalid="ignore"):
            bad_px = r["v"] & np.isfinite(r["gtd"]) & (r["d"] >= GC_RATIO * gmax) & (r["d"] > gmax + GC_ABS)
        box = bad_px[42:79, 61:214]
        print(f"  f200 narrow (idx {r['idx']}): GT-consistency px in the fixture box rows 42-78/cols 61-213: {int(box.sum())} "
              f"(whole frame {int(bad_px.sum())})", flush=True)
    bad = streak_gate(rows, "f200fixture", data)
    print("  INFO f200 (not a gate, not out-of-sample evidence): " + ("; ".join(bad) if bad else "under the committed streak limits"), flush=True)


VIS_BAND_FRAC = 0.10   # band = the 10% of image rows just below the horizon line


def check_clip015_poseB_unknown_report() -> None:
    """Visibility (Tracer): sky is never 3-D points (valid = ~sky); the backdrop is a 2-D gradient in compose.
    REPORT ONLY: pose-B (960x540, fill/sky/prior on, current-frame points, wide snap OFF) share of 'unknown'
    px in the top half and in the band just below the horizon, per verify frame."""
    if not _need_clip("clip015_poseB_unknown_report"):
        return SKIPPED
    for f, per in _rs("off", RS_VERIFY_FRAMES).items():
        pts, cols, foot = [], [], []
        for cid, r in per.items():
            ys, xs = np.nonzero(r["v"])
            q = r["q"][ys, xs]
            k = ego_box_filter(q)
            pts.append(q[k]); cols.append(np.full((int(k.sum()), 3), 128, np.uint8)); foot.append(r["foot"][ys, xs][k])
        chase = sf.chase_camera(POSE_B_POS, POSE_B_PITCH, POSE_B_FOV, 960, 540, 0.0)
        out = sf.compose(chase, np.concatenate(pts), np.concatenate(cols), np.concatenate(foot), sky_rgb=(200, 220, 240),
                         ground_rgb=(128, 128, 128), max_radius_px=8, fill=True)
        unk = out.masks["unknown"]
        up = sf._pixel_rays(chase)[..., 2] >= -1e-9
        hz = np.argmax(~up, axis=0)
        rows = np.arange(540)[:, None]
        band = (rows >= hz[None, :]) & (rows < hz[None, :] + int(540 * VIS_BAND_FRAC))
        print(f"  poseB unknown f{f}: top half {unk[:270].mean():.4f} (n={unk[:270].size}), below-horizon band "
              f"{unk[band].mean():.4f} (n={int(band.sum())}), sky backdrop px {out.masks['sky'].mean():.4f}", flush=True)


# ---- synthetic: the metrics detect what they claim (fail-first on synthetic)
def _synthetic_wide_plane(banded: bool):
    """Flat road plane on the wide cam, every ground px labelled road, wide snap OFF. Depth is fed as
    the net outputs it: the plane sampled on the 12x16 grid (cell centres); cam_frame upsamples 1/d linearly,
    which is exact for a plane. banded=True scales every other grid row by 1 - RIP_BAND (banded net
    depth); banded=False feeds the exact plane samples."""
    p = Preview()
    _wide_snap_off(p)
    cam = p.cam["wide"]
    g = geom(cam)
    exact = np.where(g["down"], g["hit"], 1e4)
    fine = np.where(~g["down"], sf_sky(), ROAD_ID)
    H, W = exact.shape
    yi = ((np.arange(12) + 0.5) * H / 12).astype(int)
    xi = ((np.arange(16) + 0.5) * W / 16).astype(int)
    depth_in = exact[yi[:, None], xi[None, :]]
    if banded:
        depth_in = depth_in * np.where(np.arange(12) % 2 == 0, 1.0, 1.0 - RIP_BAND)[:, None]
    _, d, v, _ = p.run("wide", depth_in, coarse(fine))
    gtd = np.where(g["down"], g["hit"], np.nan)
    return rs_record(cam, d, v, gtd, fine == ROAD_ID, cap=sf.depth_cap_m(CAM_SPECS["wide"]["far_m"]))


def check_synthetic_ripple_signature() -> None:
    """A: banded (12x16) depth of a perfectly flat plane through cam_frame must fail at least one wide ripple
    limit; the exact plane must pass every wide ripple limit. (At the committed limits the banded plane is caught
    by the per-range |dz| limits; std / ptp / zc / pfrac of the banded plane stay under the rule limits - printed.)"""
    lim = RIPPLE_LIMITS.get("wide") or {}
    clean = ripple_metrics(_synthetic_wide_plane(False))
    band = ripple_metrics(_synthetic_wide_plane(True))
    for tag, m in (("exact", clean), ("banded", band)):
        print(f"  synthetic ripple {tag}: n_px={m['n_px']} n_bins={m['n_bins']} std={_fmt(m['std'])} ptp_5_15={_fmt(m['ptp_5_15'])} "
              f"pfrac={_fmt(m['pfrac'])} zc={_fmt(m['zc'])} " + " ".join(f"|dz|{lo:g}-{hi:g}={_fmt(m[f'dz_{lo:g}_{hi:g}'])}" for lo, hi in RIP_DZ_BINS), flush=True)
    over_clean = [k for k, l in lim.items() if l is not None and clean.get(k) is not None and clean[k] > l]
    over_band = [(k, round(band[k], 3), l) for k, l in lim.items() if l is not None and band.get(k) is not None and band[k] > l]
    print(f"  synthetic ripple: banded flagged by {over_band}; exact flagged by {over_clean}", flush=True)
    assert not over_clean, f"exact flat plane flagged: {[(k, clean[k], lim[k]) for k in over_clean]}"
    assert over_band, "banded plane not flagged by any wide ripple limit"


def _synthetic_scene_recs(spike_col: int | None = None):
    """Every cam sees exact flat ground through cam_frame (GT = the plane, labels exact). spike_col: one wide
    pixel column gets bad depth (3x the plane) on the rows whose ground range is 8-40 m, injected into the
    DRAWN depth after cam_frame: a 1-px column spike fed into cam_frame is clamped to the road (ground_clamp)
    or removed by the depth-edge cut before it is drawn, so it never reaches the point set."""
    p = Preview()
    recs = {}
    for cid in RS_CAMS:
        cam = p.cam[cid]
        g = geom(cam)
        exact = np.where(g["down"], g["hit"], 1e4)
        fine = np.where(~g["down"], sf_sky(), ROAD_ID)
        gtd = np.where(g["down"], g["hit"], np.nan)
        _, d, v, _ = p.run(cid, exact, fine)
        if cid == "wide" and spike_col is not None:
            rows = np.nonzero(v[:, spike_col] & g["down"][:, spike_col] & (g["R"][:, spike_col] > 8.0) & (g["R"][:, spike_col] < 40.0))[0]
            d = d.copy()
            d[rows, spike_col] = 3.0 * g["hit"][rows, spike_col]
        recs[cid] = rs_record(cam, d, v, gtd, fine == ROAD_ID, cap=sf.depth_cap_m(CAM_SPECS[cid]["far_m"]))
    return recs


def check_synthetic_streak() -> None:
    """B: a clean flat scene gives 0 ray runs; one bad wide column (drawn depth 3x the plane on the rows at
    8-40 m ground range) is caught as a ray run in exactly that column, and nothing else is flagged."""
    cr = _synthetic_scene_recs(None)
    clean = streak_metrics(cr)
    gcc = {c: gt_consistency(r) for c, r in cr.items()}
    # INFO: on an exact plane the horizon row is cut to its far side and sent to the cap (far_to_cap), so 1-2
    # full rows per cam read as cap vs GT ~35-115 m. Pre-existing pipeline behaviour, under GC_LIMITS.
    print("  synthetic clean GT-consistency (horizon row -> cap): " + " ".join(
        f"{c}:{g['count']} rows {g['rows']}" for c, g in gcc.items() if g["count"]), flush=True)
    over = {c: g["count"] for c, g in gcc.items() if GC_LIMITS.get(c) is not None and g["count"] > GC_LIMITS[c]}
    assert not over, f"clean flat scene over GC_LIMITS: {over}"
    n_clean = {c: r["count"] for c, r in clean["rays"].items() if r["count"]}
    iso = " ".join(f"Y{Y:g}/X{X:g}={v['total']}/{v['n_far']}" for (Y, X), v in clean["iso"].items())
    print(f"  synthetic streak clean: n_pts={clean['n_pts']} rays={n_clean or 0}; isolated far pts (flat ground, report only) {iso}", flush=True)
    assert not n_clean, f"clean flat scene has ray runs: {n_clean}"
    W = pv().cam["wide"].width
    col = W // 2 + W // 5
    sr = _synthetic_scene_recs(col)
    spk = streak_metrics(sr)
    def _gc_px(r):
        g = np.where(np.isfinite(r["gtd"]), np.minimum(r["gtd"], r["cap"]), -np.inf)
        gm = cv2.dilate(g.astype(np.float32), np.ones((3, 3), np.uint8))
        with np.errstate(invalid="ignore"):
            return r["v"] & np.isfinite(r["gtd"]) & (r["d"] >= GC_RATIO * gm) & (r["d"] > gm + GC_ABS)
    new = {c: _gc_px(sr[c]) & ~_gc_px(cr[c]) for c in sr}
    ys, xs = np.nonzero(new["wide"])
    print(f"  synthetic spike: GT-consistency px added vs clean: wide {xs.size} (cols {sorted(set(xs.tolist()))}, rows "
          f"{[int(ys.min()), int(ys.max())] if xs.size else None}); others {sum(int(m.sum()) for c, m in new.items() if c != 'wide')}", flush=True)
    assert xs.size >= 1 and set(xs.tolist()) == {col} and not any(m.any() for c, m in new.items() if c != "wide"), "spike not isolated by GT-consistency"
    r = spk["rays"]["wide"]
    others = {c: x["count"] for c, x in spk["rays"].items() if c != "wide" and x["count"]}
    print(f"  synthetic streak spike col {col}: wide rays={r['count']} top={r['top'][:1]} others={others or 0}", flush=True)
    assert r["count"] >= 1 and all(abs(t["col"] - col) <= 1 for t in r["top"]) and not others, \
        f"bad column {col} not caught (or others flagged): {r} {others}"


# ---- safety: the sky / any RV render option is render-only (Skeptic 2026-11-26)
RV_SKY_FLAG = "RV_SKY_BACKDROP"   # expected name of a future sky-backdrop flag; parametrized here once it lands
SAFETY_ROWS = 8                   # first clip015 rows (dev; far from burned/sealed frames)
RENDER_VARIANTS = (
    {}, {"sky": 0}, {"prior": 0}, {"fill": 0}, {"width": 960, "height": 540}, {"debug_color": "cam"},
    {"debug_color": "height"}, {"chase": "rigid"}, {"omega": 3.0},
)


def _h(*arrs) -> str:
    import hashlib
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode() + str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()


def _perception_trace(variant: dict, root=None, P=None, cams=None, rows=None) -> tuple[list, list]:
    """Run the preview's per-row pipeline (cam_frame -> CamDepthFilter -> GroundGrid -> compose) with one
    render variant; return the sha256 trace of everything that is not the picture (cam_frame outputs,
    filter in/out, grid adds, grid.live, CamEvidence inputs) and any in-place mutation found by compose."""
    over = {k: v for k, v in variant.items() if k != "__env"}
    p = Preview(over, root=root, P=P, cams=cams)
    a = p.a
    ns = p.ns
    filters = {c: sf.CamDepthFilter() for c in p.cam}
    grid = sf.GroundGrid()
    damp = sf.DampedHeading(omega=a.omega)
    trace, mut = [], []
    latest = {}
    sky_rgb = ground_rgb = None
    for n, row in enumerate(rows):
        i, t, pose = row["i"], row["t"], row["pose"]
        for cid in p.cam:
            if root is None:
                fr = p.run(cid, row["depth"][cid], row["sem"][cid], idx=i)
            else:
                if cid not in (row.get("cams") or {}) or not (root / "raw" / cid / "000" / f"{i:06d}.png").is_file():
                    continue
                fr = p.run(cid, None, None, idx=i)
            if fr is None:
                continue
            rgb, depth, valid, sky = fr
            trace.append(("cam_frame", i, cid, _h(rgb, depth, valid, sky)))
            cp = p.cam[cid]
            conf = np.clip(1.0 / (1.0 + depth / 20.0), 0.2, 1.0)
            cin = np.where(valid, conf, 0.0)
            trace.append(("filter_in", i, cid, _h(depth, valid, cin, np.array(pose), np.array([t]))))
            if a.depth_filter:
                depth, wgt = filters[cid].update(cp, depth, valid, pose, t, conf=cin)
                trace.append(("filter_out", i, cid, _h(depth, wgt)))
            trace.append(("evidence", i, cid, _h(depth, valid, sky)))
            ys, xs = np.nonzero(valid)
            pts = sf.unproject(cp, depth, ys, xs)
            cols = rgb[ys, xs]
            foot = ns["footprint"](cp, depth)[ys, xs]
            keep = ego_box_filter(pts)
            pts, cols, foot = pts[keep], cols[keep], foot[keep]
            latest[cid] = (pts, cols, foot, pose)
            if a.grid:
                st_ = pts[:, 2] <= sf.STATIC_MAX_HEIGHT_M
                w_in = sf.transform_points(pts[st_], pose, (0.0, 0.0, 0.0))
                c_in = cols[st_].astype(np.float64)
                k_in = 1.0 / (1.0 + np.hypot(pts[st_, 0], pts[st_, 1]) / 15.0)
                trace.append(("grid_add", i, cid, _h(w_in, c_in, k_in)))
                grid.add(w_in, c_in, k_in, t)
            if sky.any():
                sc = np.median(rgb[sky], axis=0)
                sky_rgb = sc if sky_rgb is None else 0.8 * sky_rgb + 0.2 * sc
        pp, pc, pf = [], [], []
        for cid, (pts, cols, foot, ps) in latest.items():
            pp.append(sf.transform_points(pts, ps, pose)); pc.append(cols); pf.append(foot)
        if a.grid:
            gp, gc = grid.live(t, pose[:2])
            trace.append(("grid_live", i, "-", _h(gp, gc)))
            if gp.shape[0]:
                gl = sf.transform_points(gp, (0.0, 0.0, 0.0), pose)
                pp.append(gl); pc.append(gc); pf.append(np.full(gl.shape[0], sf.GRID_CELL_M * 1.3))
                near = np.hypot(gl[:, 0], gl[:, 1]) < 12.0
                if near.sum() > 20:
                    g_ = np.median(gc[near], axis=0)
                    ground_rgb = g_ if ground_rgb is None else 0.8 * ground_rgb + 0.2 * g_
        if not pp:
            continue
        pts = np.concatenate(pp); cols = np.concatenate(pc); foot = np.concatenate(pf)
        yaw_off = (damp.step(pose[2], 0.05 if n else 0.0) - pose[2]) if a.chase == "damped" else 0.0
        chase = sf.chase_camera(POSE_B_POS, POSE_B_PITCH, POSE_B_FOV, int(a.width), int(a.height), yaw_off)
        held = [(c, _h(*v[:3])) for c, v in latest.items()]
        before = _h(pts, cols, foot)
        gstate = _h(*[np.array(v) for v in grid.cells.values()]) if grid.cells else ""
        fstate = [_h(f.depth, f.w, f.valid) for f in filters.values() if f.depth is not None]
        sf.compose(chase, pts, cols, foot, sky_rgb=(sky_rgb if a.sky else None),
                   ground_rgb=(ground_rgb if a.prior else None), max_radius_px=max(2, int(a.width) // 120), fill=bool(a.fill))
        if _h(pts, cols, foot) != before:
            mut.append(f"row {i}: compose mutated pts/cols/foot")
        if [(c, _h(*v[:3])) for c, v in latest.items()] != held:
            mut.append(f"row {i}: compose mutated held per-cam points")
        if (_h(*[np.array(v) for v in grid.cells.values()]) if grid.cells else "") != gstate:
            mut.append(f"row {i}: compose mutated the ground grid")
        if [_h(f.depth, f.w, f.valid) for f in filters.values() if f.depth is not None] != fstate:
            mut.append(f"row {i}: compose mutated a CamDepthFilter state")
    return trace, mut


def _rv_consumers() -> list:
    """Modules outside the RV render (python/viz, tools, scripts) that import an RV render module."""
    bad = []
    for sub in ("planning", "control", "perception", "runtime", "sensors"):
        for f in sorted((ROOT / "python" / sub).rglob("*.py")) if (ROOT / "python" / sub).is_dir() else []:
            s = f.read_text(encoding="utf-8", errors="ignore")
            if any(k in s for k in ("rv_stable_fill", "rv_multicam", "reality_view_c", "rv_contract")):
                bad.append(str(f.relative_to(ROOT)))
    return bad


RV_FORBIDDEN_DOC = "tools.rv_* or rv_stable_fill (any package path)"
AEB_PLANNER_MODULES = ("python.control.actuate", "python.control.override", "python.planning.speed",
                       "python.perception.cipv", "python.runtime.frame_gate")


def _rv_forbidden(mod: str) -> bool:
    parts = mod.split(".")
    return (parts[0] == "tools" and len(parts) > 1 and parts[1].startswith("rv_")) or "rv_stable_fill" in parts


def _module_imports(path: Path, modname: str) -> set:
    """Absolute module names imported anywhere in the file (incl. function-level / try imports; relative resolved)."""
    out = set()
    pkg = modname.split(".")[:-1] if path.name != "__init__.py" else modname.split(".")
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="ignore"))):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                base = ".".join(pkg[: len(pkg) - (n.level - 1)] + ([base] if base else []))
            out.add(base)
            out |= {f"{base}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("import_module", "__import__") \
                and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            out.add(n.args[0].value)
    return out


def rv_import_violations(root: Path, pkg: str = "python") -> list:
    """Static AST scan of root/pkg/**: modules that import a forbidden RV module directly or transitively (through
    other modules under the same tree)."""
    files = {}
    for f in sorted((root / pkg).rglob("*.py")):
        rel = f.relative_to(root).with_suffix("")
        mod = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
        files[mod] = f
    graph = {m: _module_imports(f, m) for m, f in files.items()}
    bad = []
    for start in sorted(files):
        if _rv_forbidden(start):
            continue
        seen, stack, path_of = {start}, [start], {start: [start]}
        hit = None
        while stack and hit is None:
            m = stack.pop()
            for t in sorted(graph.get(m, ())):
                if _rv_forbidden(t):
                    hit = path_of[m] + [t]
                    break
                for cand in (t, t.rsplit(".", 1)[0]):
                    if cand in files and cand not in seen:
                        seen.add(cand); path_of[cand] = path_of[m] + [cand]; stack.append(cand)
        if hit:
            bad.append(" -> ".join(hit))
    return bad


def rv_runtime_modules(mods, extra_path: Path | None = None) -> list:
    """Fresh interpreter: import the AEB / planner modules; return any rv_* module that ended up in sys.modules."""
    code = ("import sys, importlib\n"
            + (f"sys.path.insert(0, {str(extra_path)!r})\n" if extra_path else "")
            + f"for m in {list(mods)!r}: importlib.import_module(m)\n"
            + "print('\\n'.join(sorted(k for k in sys.modules if any(p.startswith('rv_') for p in k.split('.')) or 'rv_stable_fill' in k)))")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT)] + ([str(extra_path)] if extra_path else []))})
    assert r.returncode == 0, f"import of {mods} failed: {r.stderr[-800:]}"
    return [l for l in r.stdout.splitlines() if l.strip()]


def check_import_isolation_rv() -> None:
    """RENDER-ONLY SAFETY (sky-gateable; guard, must pass): nothing under python/ imports """ + "tools.rv_* or rv_stable_fill" + """,
    directly or transitively (AST scan of python/**); importing the AEB / planner modules (actuate, override,
    planning.speed, perception.cipv, runtime.frame_gate) in a fresh interpreter loads no rv_* module. Mutant self-test:
    a temp tree whose module imports tools.rv_stable_preview (via a helper) must be flagged by the static scan, and a
    temp module importing a stub rv_stable_fill must be flagged by the runtime check."""
    bad = rv_import_violations(ROOT)
    loaded = rv_runtime_modules(AEB_PLANNER_MODULES)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "python" / "control").mkdir(parents=True)
        (t / "python" / "__init__.py").write_text("")
        (t / "python" / "control" / "__init__.py").write_text("")
        (t / "python" / "control" / "helper.py").write_text("def f():\n    from tools import rv_stable_preview\n    return rv_stable_preview\n")
        (t / "python" / "control" / "mut_aeb.py").write_text("from . import helper\n")
        mut_static = rv_import_violations(t)
        (t / "rv_stable_fill.py").write_text("X = 1\n")
        (t / "mut_aeb_rt.py").write_text("import rv_stable_fill\n")
        mut_rt = rv_runtime_modules(("mut_aeb_rt",), extra_path=t)
    print(f"  import isolation: static violations {bad or 0}; rv_* loaded by {len(AEB_PLANNER_MODULES)} AEB/planner modules: "
          f"{loaded or 0}; mutant static {mut_static}; mutant runtime {mut_rt}", flush=True)
    assert any("python.control.mut_aeb -> python.control.helper -> tools" in m for m in mut_static), f"mutant (transitive) not caught: {mut_static}"
    assert "rv_stable_fill" in mut_rt, f"runtime mutant not caught: {mut_rt}"
    assert not bad, f"python/ imports {RV_FORBIDDEN_DOC}: {bad}"
    assert not loaded, f"AEB/planner imports load RV modules: {loaded}"


def check_threshold_frames_outside_seal_guard() -> None:
    """Seal guard band (Skeptic 13:51): threshold frames (and every cam's nearest pred idx at them) never fall in
    63-107 / 163-207; nor are they verify, burned or sealed frames. The f200 fixture is INFO and opt-in only."""
    clip = clip_paths()
    bad = [f for f in RS_THRESH_FRAMES if in_seal_guard(f) or f in RS_VERIFY_FRAMES or f in RS_BURNED or f in HELDOUT2_SEALED]
    if clip is not None:
        P = np.load(clip[1])
        for cid in RS_CAMS:
            av = pred_frames(P, cid)
            bad += [f"f{f}:{cid}:{nearest_idx(av, f)}" for f in RS_THRESH_FRAMES if in_seal_guard(nearest_idx(av, f))]
    assert SEAL_GUARD_RANGES == ((63, 107), (163, 207)), SEAL_GUARD_RANGES
    assert not bad, f"threshold frames in the seal guard band / reserved: {bad}"


SEAL_GUARD_BAND_CONSTANTS = frozenset({"TRACER_HOLDOUT", "CLEAN_HOLDOUT", "F200_FIXTURE"})
# Helpers a gating check may call; they must skip guard-band pred idxs.
_SEAL_GATING_HELPERS = frozenset({
    "clip_poseB_holes", "runtime_mask_counts", "clip_behind_car_strip", "clip_capture_holes",
    "clip_depth_metrics", "rs_collect", "_dclip", "_rs", "clip_measure",
})


def _seal_guard_refs(fn: ast.AST) -> list:
    """Names of TRACER_HOLDOUT / CLEAN_HOLDOUT / F200_FIXTURE, or a call to clip_stable_timeline,
    referenced in this function (those paths read guard-band frames)."""
    hits = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in SEAL_GUARD_BAND_CONSTANTS:
            hits.append(n.id)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "clip_stable_timeline":
            hits.append("clip_stable_timeline")
    return hits


def gating_check_guard_band_hits(src: str, report_only: set, skip: set) -> list:
    """AST scan: gating check_* functions must not reference holdout/f200 constants or the timeline walker."""
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    bad = []
    for name, node in funcs.items():
        if not name.startswith("check_") or name in report_only or name in skip:
            continue
        refs = _seal_guard_refs(node)
        if refs:
            bad.append(f"{name} references {refs}")
    return bad


def check_gating_checks_outside_seal_guard() -> None:
    """Seal guard band (Skeptic 14:09): no *gating* check or threshold derivation may read a frame or pred idx
    in 63-107 / 163-207. Report-only checks in SEAL_GUARD_REPORT_ONLY may still print those frames as INFO.
    Sibling of check_threshold_frames_outside_seal_guard."""
    src = Path(__file__).read_text(encoding="utf-8")
    skip = {"check_threshold_frames_outside_seal_guard", "check_gating_checks_outside_seal_guard"}
    hits = gating_check_guard_band_hits(src, set(SEAL_GUARD_REPORT_ONLY), skip)
    # mutant: a gating check that calls _dclip(TRACER_HOLDOUT) must be flagged
    mutant = "def check_mutant_holdout():\n    snap_misplace_gate(_dclip(TRACER_HOLDOUT), 'x')\n"
    mut_hits = gating_check_guard_band_hits(src + "\n" + mutant, set(SEAL_GUARD_REPORT_ONLY), skip)
    assert any("check_mutant_holdout" in h and "TRACER_HOLDOUT" in h for h in mut_hits), mut_hits
    assert not hits, "gating checks still reference seal-guard-band frames: " + "; ".join(hits)
    # no demoted name remains a KNOWN_FAILING gate
    still = sorted(SEAL_GUARD_REPORT_ONLY & set(KNOWN_FAILING))
    assert not still, f"report-only guard-band checks still in KNOWN_FAILING: {still}"
    # gating frame constants themselves, and every cam's pred idx at those frames after the skip, stay outside the band
    gating_frames = tuple(dict.fromkeys(list(GATE_FRAMES) + list(RS_THRESH_FRAMES) + list(RS_VERIFY_FRAMES)
                                        + list(MASK_FRAMES) + [POSE_B_FRAME]))
    bad = [f for f in gating_frames if f in RS_BURNED or f in HELDOUT2_SEALED]
    # a *frame number* may sit next to the band (f110=110 is allowed) so long as any cam whose nearest
    # pred idx falls in the band is skipped by the gating helpers (rear @ f110 idx 107).
    clip = clip_paths()
    used_in_band = []
    if clip is not None:
        P = np.load(clip[1])
        avail = {cid: pred_frames(P, cid) for cid in RS_CAMS}
        # helpers that a gating check still calls must not return a guard-band idx
        pairs = []
        for cid, idx in clip_poseB_holes().get("used", {}).items():
            pairs.append(("clip_poseB_holes", cid, idx))
        cap_used = clip_capture_holes((40, 160, POSE_B_FRAME)).get("used", {})
        for cid, by in cap_used.items():
            for idx in (by.values() if isinstance(by, dict) else [by]):
                pairs.append(("clip_capture_holes", cid, idx))
        for rec in clip_behind_car_strip((40, 160, POSE_B_FRAME), (("poseA", pose_a_chase()),)).values():
            for cid, idx in rec.get("used", {}).items():
                pairs.append(("clip_behind_car_strip", cid, idx))
        for (f, cid), (_, _, idx) in runtime_mask_counts()[0].items():
            pairs.append((f"runtime_mask_counts f{f}", cid, idx))
        for helper, cid, idx in pairs:
            if in_seal_guard(idx):
                used_in_band.append(f"{helper} {cid}:{idx}")
        # rs_collect / _rs on verify frames (ripple/streak/ringA/false-removal path)
        off = _rs("off", RS_VERIFY_FRAMES)
        for f, per in off.items():
            for cid, r in per.items():
                if in_seal_guard(r.get("idx")):
                    used_in_band.append(f"rs_collect f{f} {cid}:{r['idx']}")
        # GATE_FRAMES (f40/f160) idxs are all outside the band (no skip needed)
        for f in GATE_FRAMES + RS_THRESH_FRAMES:
            for cid, av in avail.items():
                idx = nearest_idx(av, f)
                if in_seal_guard(f) or in_seal_guard(idx):
                    bad.append(f"gating frame f{f} {cid}:{idx}")
    print(f"  gating frames {gating_frames}; AST hits {hits or 0}; helper idxs in band {used_in_band or 0}", flush=True)
    assert not bad, f"gating frame set in the seal guard band: {bad}"
    assert not used_in_band, f"gating helpers still measured a guard-band pred idx: {used_in_band}"


def check_render_options_do_not_touch_perception() -> None:
    """SAFETY (guard, must pass): sky backdrop on/off and every RV render option leave cam_frame outputs,
    CamDepthFilter inputs/outputs, GroundGrid inputs/outputs and CamEvidence byte-identical (sha256), and
    compose never mutates its inputs or the held per-cam points / grid / filter state in place. No planner /
    control / perception / runtime module imports an RV render module (AEB/planner read nothing from RV).
    On this head the sky toggle is --sky (2-D gradient backdrop in compose; sky px are never 3-D points).
    Expected future flag: RV_SKY_BACKDROP (env or module attr); toggled here too once it exists."""
    cons = _rv_consumers()
    assert not cons, f"planner/AEB-side modules import RV render code: {cons}"
    # planner / AEB inputs: plan_speed(ego_speed_mps, curvature, ttc_lead, aeb) (python/planning/speed.py) and the
    # actuator's AEB brake (aeb flag / ttc, python/control/actuate.py) come from perception state in run_vision.
    # RV is a separate child process (python/viz/rv_contract.RvClient): run_vision only launches and closes it.
    # select_cipv consumes tracker tracks (detector output), which are not in the clip015 preds, so the AEB inputs
    # cannot be driven from the clip offline: a fixed synthetic track / speed grid through select_cipv + plan_speed
    # (+ AEB_BRAKE_TTC) is hashed before and after every RV variant; check_import_isolation_rv covers the imports.
    rv_src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    rv_uses = sorted({n.attr for n in ast.walk(ast.parse(rv_src)) if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == "rv_client"})
    assert set(rv_uses) <= {"close"}, f"run_vision reads from the RV client: {rv_uses}"
    from python.planning.speed import plan_speed
    def _aeb_inputs_hash():
        outs = [plan_speed(ego_speed_mps=v, curvature=k, ttc_lead=t, aeb=a_) for v in (None, 0.0, 8.0, 25.0)
                for k in (0.0, 0.02) for t in (None, 0.8, 3.0) for a_ in ("none", "warn", "brake")]
        from python.perception.cipv import select_cipv
        from python.control.actuate import AEB_BRAKE_TTC
        tracks = [{"class": "vehicle", "x": x, "y": y, "speed_mps": sp} for x in (-1.5, 0.0, 0.9) for y in (3.0, 6.0, 25.0)
                  for sp in (-8.0, -0.01, 2.0)] + [{"class": "pedestrian", "x": 0.0, "y": 5.0, "speed_mps": -3.0}]
        cv = [select_cipv([t], ego_speed_mps=v) for t in tracks for v in (None, 0.0, 10.0)] + [select_cipv(tracks, ego_speed_mps=10.0)]
        enc = lambda x: {None: -1.0, "off": 0.0, "warn": 1.0, "brake": 2.0}.get(x, -2.0)
        return _h(np.array([[np.nan if o.target_v is None else o.target_v, np.nan if o.ttc_lead is None else o.ttc_lead,
                             enc(o.aeb)] for o in outs]
                           + [[np.nan if c.ttc_lead is None else c.ttc_lead, enc(c.aeb), -1.0 if c.track is None else c.track["y"]] for c in cv]
                           + [[AEB_BRAKE_TTC, 0.0, 0.0]], np.float64))
    aeb0 = _aeb_inputs_hash()
    variants = list(RENDER_VARIANTS)
    if hasattr(sf, RV_SKY_FLAG):
        variants.append({"__module_flag": 0})
    if clip_paths() is not None:
        clip, preds, _ = clip_paths()
        cams = {c["id"]: c for c in json.loads((clip / "cameras.json").read_text())["cameras"]}
        P = np.load(preds)
        rows, odo = [], [0.0, 0.0, 0.0]
        for n, l in enumerate((clip / "state.jsonl").read_text().splitlines()[:SAFETY_ROWS]):
            r = json.loads(l)
            odo = [odo[0], odo[1] + float(r["veh"]["wheelspeed"]) * 0.05 * (n > 0), odo[2] + 0.002 * n]
            rows.append({"i": int(r["i"]), "t": float(r["t"]), "pose": tuple(odo), "cams": r.get("cams")})
        kw = {"root": clip, "P": P, "cams": cams}
        rs_frames_ok([r["i"] for r in rows])
    else:
        rows = []
        for n in range(4):
            dep, sem = {}, {}
            for cid in CAM_SPECS:
                g = geom(pv().cam[cid])
                dep[cid] = np.where(g["down"], g["hit"] * (1.0 + 0.01 * n), 1e4)
                sem[cid] = coarse(np.where(~g["down"], sf_sky(), ROAD_ID))
            rows.append({"i": n, "t": 0.05 * n, "pose": (0.0, 0.5 * n, 0.01 * n), "depth": dep, "sem": sem})
        kw = {}
    base = None
    bad = []
    for var in variants:
        saved = None
        if "__module_flag" in var:
            saved = getattr(sf, RV_SKY_FLAG)
            setattr(sf, RV_SKY_FLAG, not saved)
        try:
            tr, mut = _perception_trace({k: v for k, v in var.items() if not k.startswith("__")}, rows=rows, **kw)
        finally:
            if saved is not None:
                setattr(sf, RV_SKY_FLAG, saved)
        bad += [f"{var}: {m}" for m in mut]
        if base is None:
            base = tr
            continue
        if tr != base:
            diff = next((x for x, y in zip(tr, base) if x != y), None) if len(tr) == len(base) else f"len {len(tr)} vs {len(base)}"
            bad.append(f"{var}: perception trace differs from base at {diff}")
    assert _aeb_inputs_hash() == aeb0, "planner/AEB outputs changed after running RV render variants (global state leak)"
    print(f"  safety: {len(variants)} render variants x {len(rows)} rows, {len(base or [])} hashed perception arrays/row-steps; "
          f"sky flag on this head: --sky (module {RV_SKY_FLAG}: {'present' if hasattr(sf, RV_SKY_FLAG) else 'absent'})", flush=True)
    assert not bad, "; ".join(bad[:6])


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
    check_cut_then_clamp_order_and_nothing_under_road,
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
    check_render_options_do_not_touch_perception,
    check_import_isolation_rv,
    check_threshold_frames_outside_seal_guard,
    check_gating_checks_outside_seal_guard,
    check_synthetic_ripple_signature,
    check_synthetic_streak,
    check_clip015_ripple_wide_dz,
    check_clip015_ripple_wide_periodicity,
    check_clip015_ripple_other_cams,
    check_clip015_streak_wide_snap_off,
    check_snap_hook_binds_depth_by_name,
    check_snap_hook_hit,
    info_clip015_footprint_row_gaps,
    check_clip015_ringA_poseB_holes,
    info_snap_b_attribution,
    check_snap_rejects_raised_pred_z,
    check_clip015_false_removal,
    check_wide_snap_off_no_plane_pull_0_5m,
    check_runtime_never_reads_gt,
    check_clip015_wide_coarse_row_bowl,
    check_wide_snap_off_no_plane_pull,
    info_streak_fixture_f200_narrow_car_smear,
    check_clip015_poseB_unknown_report,
]

# Checks that still *run* and *print* on frames / pred idxs inside the seal guard band (63-107, 163-207)
# but are REPORT-ONLY (Skeptic 14:09): they must not gate, must not set or feed a limit, and are not in
# KNOWN_FAILING. Keep the measurement code. info_* names were already report-only.
SEAL_GUARD_REPORT_ONLY = frozenset({
    "check_holdout_snap_misplace",            # f100/f200
    "check_clean_holdout_snap_misplace",      # f70
    "check_holdout_edge_position",            # f100/f200
    "check_clean_holdout_edge_position",      # f70
    "check_holdout_snap_no_worse_than_off",   # f100/f200
    "check_clean_holdout_snap_no_worse_than_off",  # f70
    "check_holdout_footprint",                # f100/f200
    "check_clean_holdout_footprint",          # f70
    "check_holdout_width",                    # f100/f200
    "check_holdout_edge_overshoot",           # f100/f200
    "check_holdout_rear_holes",               # f100/f200
    "check_behind_car_strip_holes",           # clip_stable_timeline walks 0..f160 (and GT window f+/-10), so it reads 63-107 and 163-170
    "info_streak_fixture_f200_narrow_car_smear",  # already INFO, f200
})

# name -> counterexample on 783fedd (real runs after thresholds commit 8b53e6b;
# rv-side-evidence/props_report_783fedd.log). E = depth-edge cut in tools/rv_stable_preview.py cam_frame
# (after ground_clamp: far rows pushed below the road, out of the ground band); L = the net's 12x16 class
# grid (road label bleeds past the edge); N = road_snap flattens objects inside a road-majority cell.
KNOWN_FAILING: dict[str, str] = {
    "check_synthetic_width": "L+E: pillarL/R 0-5 m median 14.8% p90 47.3%; repeatL/R 0-5 m 16.6%/23.0%, 5-15 m 28.1%/48.2%; rear 15-30 m",
    "check_synthetic_edge_overshoot": "L: road 0.35 m (pillars) / 0.71 m (repeats) past the GT edge at 0-5 m (12x16 label bleed)",
    "check_side_objects_keep_height_pins": "N: pillarR parked car x2.5 lost 1.07 m on 88 px; pole x3 lost 1.07 m on 351/391 px; curbs lost 0.11 m",
    "prop_side_objects_keep_height": "N: pillarL pole at x=-2.0, y0=0.0: lost 0.592 m",
    "check_road_labelled_object_top_stays_put": "N: pillarR car top 218 px moved outward (GT z 1.50 m); curb top 1487 px, height lost 0.15 m; main car top also snapped outward",
    # pillarR/repeatR (b): michael 2026-11-26 13:50 CT, net seg+depth, Trainer 2.0 -- residual after a pred-z>0.30 rejection only
    "check_clip015_snap_misplace": "f40/f160 DRAWN-only: pillarR 18.10% (7317/40432; f40 36.90%/f160 3.50%), repeatR 4.65% (1835/39434), wide 10.12% (3555/35133; was 15.00% all-snapped 7461/49726) at 0-15 m; wide 61.64% (188/305; all-snapped 59.35% 219/369) at 15-30 m (GT z>0.30) || pillarR/repeatR (b) attribution (Breaker, dev f50/f120 + Tracer swarm-2/logs/rv_snapb.txt f40, rv_snapb_more.txt f20/f50/f120/f150): residual after a one-sided pred-z>0.30 rejection = pred-road px with raw net z<=0.30 (net seg+depth): pillarR f50 2635/6274 (Tracer 2635), repeatR f50 779/1947 (Tracer 779), f120 0/3672 and 0/957 (Tracer 0) -> root cause of the residual 'net seg+depth (pred road on curb, raw z<=0.30), Trainer 2.0'; approval michael 2026-11-26 13:50 CT: \"Yeah, if this issue can be fixed with training on a large data set like the one that we're creating that's perfectly fine\". The rest of the (b) px (raw z>0.30) are snap-created: the rejection is ABSENT on dee9827 (check_snap_rejects_raised_pred_z FAILs, normal FAIL, Fix item Redesigner). Approval does not cover wide or other cams.",
    "check_clip015_edge_position": "f40/f160: wide 5-15 m median 1.334/p90 5.119 vs 10%/20% of |x_gt| 4.36 m; also narrow/main 5-15 p90 over the 20% gate",
    "check_clip015_snap_no_worse_than_off": "f40/f160: pillarR |dz| median 0.246 vs 0.184 off; pillarL 0.049/0.603 vs 0.048/0.593",
    # ---- ripple / streak / sky (swarm-4 Breaker, wide snap OFF, thresholds by rule on f10/25/50/120/140/155) ----
    # wide coarse-row dz: michael 2026-11-26 13:50 CT, net depth, Trainer 2.0
    "check_clip015_ripple_wide_dz": "NEW (wide snap OFF; snap ON passes these bins at f40/f110/f160): wide |dz| f110 0-5 m 0.130 > 0.080, 5-10 m 0.327 > 0.163; f160 0-5 0.103, 5-10 0.277; f40 15-30 0.435 > 0.417. Root cause: net depth (12x16 coarse-row bands, wide f110 R6..R9 +0.43/+0.19/+0.22/+0.07 m; Tracer swarm-2/logs/rv_ring2.txt), fix in training (Trainer 2.0), not runtime. Approval michael 2026-11-26 13:50 CT: \"Yeah, if this issue can be fixed with training on a large data set like the one that we're creating that's perfectly fine\". Limits unchanged.",
    "check_clip015_ripple_other_cams": "PRE-EXISTING (non-wide cams are not touched by the wide toggle; same with snap ON): f40 main std 0.110 > 0.107, ptp_5_15 0.150 > 0.051, |dz| 5-10 0.150 > 0.125, 10-15 0.345 > 0.220; f40 narrow std 0.100 > 0.076, ptp 0.191 > 0.011; plus pillar/repeat/rear bins (see props log). Not covered by the approval: Fix item.",
    "check_clip015_streak_wide_snap_off": "PRE-EXISTING (snap ON f110 15778 / f160 16007, same): isolated far Y15/X0.5 f110 15762 > 14987, f160 16016 > 14987; Y15/X1 f160 6115 > 5427. GT path itself is 16003/22963 at f110/f160 (sparse far geometry). GT-consistency passes. Not covered by the approval: Fix item.",
    "check_clip015_ringA_poseB_holes": "PRE-EXISTING, worse with snap OFF: pose-B f110 GT-ground 0-5 m hole frac 0.0290 (1738/59903) > 0.0136 (snap ON 0.0163, also over). Not covered by the approval: Fix item.",
    "check_wide_snap_off_no_plane_pull_0_5m": "PRE-EXISTING (ground_clamp margin 1.0 lifts below-road net depth to the plane): f40 wide 0-5 m |pull| 0.240 > 0.122 (n=11279). 5-30 m guard passes.",
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
