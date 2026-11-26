"""Offline Reality View preview with the stable-fill pipeline (python/viz/rv_stable_fill.py).

Runtime inputs: recorded RGB cameras, the net's predicted depth + sky (preds .npz),
wheel speed, camera-only yaw (phase correlation on main). CPU only.

Eval-only switches, labelled on every frame they touch and never part of the runtime:
  --pose-src gt   sim ego pose instead of camera-only odometry (jitter decomposition)
  --depth-src gt  sim depth instead of the net (the "GT-depth" ceiling comparison)
  --classify-dir  black-cause split of another render's frames (e.g. the v2 preview)
"""
import argparse, json, math, sys, time
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--preds", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--gt-depth-root", default="")
ap.add_argument("--width", type=int, default=480); ap.add_argument("--height", type=int, default=270)
ap.add_argument("--chase", default="damped", choices=["rigid", "damped"]); ap.add_argument("--omega", type=float, default=1.5)
ap.add_argument("--depth-filter", type=int, default=1); ap.add_argument("--grid", type=int, default=1)
ap.add_argument("--fill", type=int, default=1); ap.add_argument("--sky", type=int, default=1); ap.add_argument("--prior", type=int, default=1)
ap.add_argument("--ground-clamp", type=int, default=1); ap.add_argument("--clamp-margin", type=float, default=1.0)
ap.add_argument("--road-snap", type=int, default=1)
ap.add_argument("--road-snap-max-m", type=float, default=None, help="road snap range gate (default: module ROAD_SNAP_MAX_M)")
ap.add_argument("--ownership", type=int, default=1, help="each direction drawn by one camera: wide and pillars yield to main")
ap.add_argument("--debug-color", default="none", choices=["none", "cam", "height"], help="eval only: colour points by source camera or by height")
ap.add_argument("--max-depth", choices=["v2", "cap"], default="cap", help="v2: drop depth >= 120 m (old). cap: per-cam cap, far drawn at cap")
ap.add_argument("--pose-src", default="vo", choices=["vo", "gt"]); ap.add_argument("--depth-src", default="pred", choices=["pred", "gt"])
ap.add_argument("--classify-dir", default=""); ap.add_argument("--classify-crop", default="0,0,960,540")
ap.add_argument("--frames", default=""); ap.add_argument("--save-frames", type=int, default=1); ap.add_argument("--label", default="")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from python.viz import rv_stable_fill as S
from python.viz.reality_view_c import camera_from_angles

root = Path(a.clip); P = np.load(a.preds)
cams_json = {c["id"]: c for c in json.loads((root / "cameras.json").read_text())["cameras"]}
rows = [json.loads(l) for l in (root / "state.jsonl").read_text().splitlines() if l.strip()]
CAMS = ["narrow", "main", "wide", "pillarL", "pillarR", "repeatL", "repeatR", "rear"]
SKY = 21
CHASE_POS, CHASE_PITCH, CHASE_FOV = (0.0, -4.0, 2.6), -12.0, 60.0   # pose A, as in v2
EGO_BOX = (1.0, -2.45, 2.4, -0.5, 1.6)
EGO_MASK = {"wide": (0.80, 0.10, 0.90)}
def fy_full(c): return (c["resolution"][1] / 2) / math.tan(math.radians(c["fov_v"]) / 2)
FY_REF = fy_full(cams_json["main"]) * 192.0 / cams_json["main"]["resolution"][1]
def proc_res(c):
    w, h = c["resolution"]; return (256, 192) if abs(w / h - 4 / 3) < 0.01 else (320, 180)
CAM_POSE = {}
CLAMP_STATS: dict = {}
YIELD_TO = {"wide": ["main"], "pillarL": ["main"], "pillarR": ["main"]}
FAR_STATS: dict = {}
SNAP_STATS: dict = {}
CAM_COLOR = {"narrow": (255, 255, 0), "main": (0, 200, 0), "wide": (255, 0, 0), "pillarL": (0, 120, 255), "pillarR": (255, 0, 255), "repeatL": (0, 255, 255), "repeatR": (255, 140, 0), "rear": (128, 128, 128)}
for cid, c in cams_json.items():
    if cid in CAMS:
        pw, ph = proc_res(c)
        CAM_POSE[cid] = camera_from_angles(cid, c["pos_m"], c.get("yaw_deg") or 0.0, c["pitch_deg"], c["fov_v"], pw, ph, near_m=0.05, far_m=400.0)

YIELD = {c: S.yield_mask(CAM_POSE[c], [CAM_POSE[b] for b in bs]) for c, bs in YIELD_TO.items() if c in CAM_POSE}

def phase_dx(prev, cur):
    win = np.hanning(prev.shape[0])[:, None] * np.hanning(prev.shape[1])[None, :]
    fa = np.fft.fft2((prev - prev.mean()) * win); fb = np.fft.fft2((cur - cur.mean()) * win)
    r = fb * np.conj(fa); r /= np.maximum(np.abs(r), 1e-12); cc = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape); w = cc.shape[1]
    l, m, rr = cc[iy, (ix - 1) % w], cc[iy, ix], cc[iy, (ix + 1) % w]; den = l - 2 * m + rr
    dx = ix + (0.5 * (l - rr) / den if abs(den) > 1e-12 else 0.0)
    return dx - w if dx > w / 2 else dx

def cam_frame(cam, idx):
    c = cams_json[cam]; W0, H0 = c["resolution"]; pw, ph = proc_res(c); sx = pw / W0
    bgr = cv2.imread(str(root / "raw" / cam / "000" / f"{idx:06d}.png"), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    fs = (fy_full(c) * 192.0 / H0) / FY_REF
    inv_acc = np.zeros((ph, pw)); wacc = np.zeros((ph, pw)); sky = np.zeros((ph, pw), bool); road = np.zeros((ph, pw), bool)
    k = 0
    while f"{cam}/{idx}/{k}/d" in P.files:
        x0, cw = P[f"{cam}/{idx}/{k}/x0"]; d = P[f"{cam}/{idx}/{k}/d"] * fs; s = P[f"{cam}/{idx}/{k}/s"]
        a0, a1 = int(round(x0 * sx)), int(round((x0 + cw) * sx))
        inv_acc[:, a0:a1] += cv2.resize((1.0 / np.maximum(d, 0.1)).astype(np.float32), (a1 - a0, ph), interpolation=cv2.INTER_LINEAR)
        wacc[:, a0:a1] += 1
        sky[:, a0:a1] |= cv2.resize((s == SKY).astype(np.uint8), (a1 - a0, ph), interpolation=cv2.INTER_NEAREST) > 0
        road[:, a0:a1] |= cv2.resize(np.isin(s, S.ROAD_CLASS_IDS).astype(np.uint8), (a1 - a0, ph), interpolation=cv2.INTER_NEAREST) > 0
        k += 1
    if k == 0:
        return None
    depth = 1.0 / np.maximum(inv_acc / np.maximum(wacc, 1), 1e-6)
    if a.depth_src == "gt":
        g = cv2.imread(str(Path(a.gt_depth_root) / f"{cam}_depth" / "000" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
        g = cv2.resize(g, (pw, ph), interpolation=cv2.INTER_NEAREST)
        depth = np.where(g > 0, g, np.inf)
        sky = ~(g > 0) | (g >= min(c["far_m"] * S.GT_NO_HIT_FRAC, 655.0))  # far plane / uint16 cm saturation = no hit
    clamp_frac = 0.0
    snapped = np.zeros((ph, pw), bool)
    if a.road_snap and a.depth_src != "gt":
        depth, snapped = S.snap_road_for_cam(cam, CAM_POSE[cam], depth, road & ~sky, max_hit_m=a.road_snap_max_m or S.ROAD_SNAP_MAX_M)
    SNAP_STATS.setdefault(cam, []).append(float(np.mean(snapped)))
    # depth edges: re-place edge pixels on the far side (local max), so no sheet joins fg and bg
    inv = 1.0 / np.maximum(depth, 1e-3)
    gy, gx = np.gradient(inv)
    cut = np.hypot(gx, gy) > 0.12 * inv
    far_side = cv2.dilate(depth.astype(np.float32), np.ones((5, 5), np.uint8))
    # ... but never under the road: where the far side is past the pixel's own ground hit (a thin pole or the road
    # in front of a face would be pushed under the road, Breaker: -1.02 m at 29.9 m), keep the pixel's own depth
    cp_ = CAM_POSE[cam]; rz = S._pixel_rays(cp_)[..., 2]; dn = rz < -1e-6
    ghit = np.where(dn, float(cp_.pos[2]) / np.where(dn, -rz, 1.0), np.inf) * (a.clamp_margin if a.ground_clamp else 1.0)
    depth = np.where(cut & ~sky & ~snapped & (far_side <= ghit), far_side, depth)
    # and clamp after the cut, so nothing the cut touched can end up below the road
    if a.ground_clamp and a.depth_src != "gt":
        depth, cl = S.ground_clamp(CAM_POSE[cam], depth, margin=a.clamp_margin)
        clamp_frac = float(np.mean(cl & ~sky))
    CLAMP_STATS.setdefault(cam, []).append(clamp_frac)
    if a.max_depth == "cap":
        cap = S.depth_cap_m(c["far_m"])
        depth, farm = S.far_to_cap(depth, sky, cap)
        FAR_STATS.setdefault(cam, []).append(float(np.mean(farm)))
        valid = ~sky & np.isfinite(depth) & (depth > 0.3) & (depth <= cap)
    else:
        valid = ~sky & np.isfinite(depth) & (depth > 0.3) & (depth < S.FAR_M)
    if a.ownership and cam in YIELD_TO:
        valid &= ~YIELD[cam]
    if cam in EGO_MASK:
        r0, c0, c1 = EGO_MASK[cam]; valid[int(r0 * ph):, int(c0 * pw):int(c1 * pw)] = False
    return rgb, depth, valid, sky

def footprint(cam, depth):
    d = np.where(np.isfinite(depth), depth, 0.0); h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w]
    g = np.stack([((xs + 0.5 - cam.cx) / cam.fx) * d, -((ys + 0.5 - cam.cy) / cam.fy) * d, d], -1)
    sp = np.zeros((h, w))
    sp[:, :-1] = np.maximum(sp[:, :-1], np.linalg.norm(np.diff(g, axis=1), axis=-1))
    sp[:-1] = np.maximum(sp[:-1], np.linalg.norm(np.diff(g, axis=0), axis=-1))
    base = d / cam.fy
    return np.clip(sp, base, 6.0 * base)

def gt_pose(row, ref):
    d = row["ego"]["dir"]; hw = math.atan2(d[1], d[0]); p = np.array(row["ego"]["pos"][:2])
    hw0, p0 = ref
    dp = p - p0; f0 = np.array([math.cos(hw0), math.sin(hw0)]); l0 = np.array([-math.sin(hw0), math.cos(hw0)])
    return (float(-dp @ l0), float(dp @ f0), float(math.atan2(math.sin(hw - hw0), math.cos(hw - hw0))))

out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
want = {int(x) for x in a.frames.split(",") if x} if a.frames else None
filters = {c: S.CamDepthFilter() for c in CAMS}
grid = S.GroundGrid(); damp = S.DampedHeading(omega=a.omega)
latest = {}; odo = [0.0, 0.0, 0.0]; prev_band = None; prev_t = None; gt_ref = None
FXM = fy_full(cams_json["main"]) * 256 / 640
sky_rgb = None; ground_rgb = None
stats = {k: [] for k in ["fuse_cpu_ms", "render_cpu_ms", "black", "unknown", "observed", "filled", "ground_prior", "sky", "far", "chase_yaw_off_deg"]}
causes_acc = []; v2_causes = []; v2_black = []
cx0, cy0, cw0, ch0 = (int(v) for v in a.classify_crop.split(","))
chase_rigid_full = camera_from_angles("chase", CHASE_POS, 0.0, CHASE_PITCH, CHASE_FOV, cw0, ch0, near_m=0.05, far_m=400.0)
for n_row, row in enumerate(rows):
    i = int(row["i"]); t = float(row["t"])
    c0 = time.process_time()
    band = cv2.resize(cv2.imread(str(root / "raw/main/000" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE), (256, 192), interpolation=cv2.INTER_AREA)[:80].astype(np.float64)
    if gt_ref is None:
        d0 = row["ego"]["dir"]; gt_ref = (math.atan2(d0[1], d0[0]), np.array(row["ego"]["pos"][:2]))
    if prev_band is not None:
        dt = max(0.0, t - prev_t); dyaw = math.atan2(phase_dx(prev_band, band), FXM)
        mid = odo[2] + 0.5 * dyaw; dist = float(row["veh"]["wheelspeed"]) * dt
        odo = [odo[0] - math.sin(mid) * dist, odo[1] + math.cos(mid) * dist, odo[2] + dyaw]
    else:
        dt = 0.0
    prev_band, prev_t = band, t
    pose = tuple(odo) if a.pose_src == "vo" else gt_pose(row, gt_ref)
    evidence = []; sky_cols = []
    for cam in CAMS:
        if cam not in (row.get("cams") or {}) or not (root / "raw" / cam / "000" / f"{i:06d}.png").is_file():
            continue
        fr = cam_frame(cam, i)
        if fr is None:
            continue
        rgb, depth, valid, sky = fr; cp = CAM_POSE[cam]
        if a.depth_filter:
            conf = np.clip(1.0 / (1.0 + depth / 20.0), 0.2, 1.0)
            depth, _ = filters[cam].update(cp, depth, valid, pose, t, conf=np.where(valid, conf, 0.0))
        evidence.append(S.CamEvidence(cp, depth, valid, sky))
        if sky.any() and cam in ("main", "wide", "narrow"):
            sky_cols.append(rgb[sky])
        ys, xs = np.nonzero(valid)
        pts = S.unproject(cp, depth, ys, xs); cols = rgb[ys, xs]; foot = footprint(cp, depth)[ys, xs]
        hw, y0, y1, z0, z1 = EGO_BOX
        inside = (np.abs(pts[:, 0]) <= hw) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
        pts, cols, foot = pts[~inside], cols[~inside], foot[~inside]
        if a.debug_color == "cam":
            cols = np.tile(np.array(CAM_COLOR[cam], np.uint8), (len(pts), 1))
        elif a.debug_color == "height":
            hz = np.clip((pts[:, 2] + 0.5) / 1.5, 0, 1)   # -0.5 m blue .. 0 green .. +1 m red
            cols = (cv2.applyColorMap((hz * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)[:, ::-1]).copy()
        latest[cam] = (pts, cols, foot, pose)
        if a.grid:
            st = pts[:, 2] <= S.STATIC_MAX_HEIGHT_M
            dist = np.hypot(pts[st, 0], pts[st, 1])
            wpts = S.transform_points(pts[st], pose, (0.0, 0.0, 0.0))
            grid.add(wpts, cols[st].astype(np.float64), 1.0 / (1.0 + dist / 15.0), t)
    if sky_cols:
        sc = np.median(np.concatenate(sky_cols), axis=0); sky_rgb = sc if sky_rgb is None else 0.8 * sky_rgb + 0.2 * sc
    parts_p, parts_c, parts_f = [], [], []
    for cam, (pts, cols, foot, ps) in latest.items():
        parts_p.append(S.transform_points(pts, ps, pose)); parts_c.append(cols); parts_f.append(foot)
    if a.grid:
        gp, gc = grid.live(t, pose[:2])
        if gp.shape[0]:
            gl = S.transform_points(gp, (0.0, 0.0, 0.0), pose)
            parts_p.append(gl); parts_c.append(gc); parts_f.append(np.full(gl.shape[0], S.GRID_CELL_M * 1.3))
            near = np.hypot(gl[:, 0], gl[:, 1]) < 12.0
            if near.sum() > 20:
                gcol = np.median(gc[near], axis=0); ground_rgb = gcol if ground_rgb is None else 0.8 * ground_rgb + 0.2 * gcol
    if ground_rgb is None and latest:
        allp = np.concatenate(parts_p); allc = np.concatenate(parts_c); m = (allp[:, 2] <= 0.35) & (np.hypot(allp[:, 0], allp[:, 1]) < 12)
        if m.sum() > 20:
            ground_rgb = np.median(allc[m], axis=0)
    c1 = time.process_time()
    yaw_off = (damp.step(pose[2], dt) - pose[2]) if a.chase == "damped" else 0.0
    if want is not None and i not in want:
        continue
    chase = S.chase_camera(CHASE_POS, CHASE_PITCH, CHASE_FOV, a.width, a.height, yaw_off)
    pts = np.concatenate(parts_p) if parts_p else np.zeros((0, 3)); cols = np.concatenate(parts_c) if parts_c else np.zeros((0, 3), np.uint8); foot = np.concatenate(parts_f) if parts_f else np.zeros(0)
    r = S.compose(chase, pts, cols, foot, sky_rgb=(sky_rgb if a.sky else None), ground_rgb=(ground_rgb if a.prior else None),
                  splat_scale=1.0, max_radius_px=max(2, a.width // 120), fill=bool(a.fill))
    c2 = time.process_time()
    stats["fuse_cpu_ms"].append((c1 - c0) * 1e3); stats["render_cpu_ms"].append((c2 - c1) * 1e3)
    stats["black"].append(S.black_fraction(r.rgb))
    for k in ("unknown", "observed", "filled", "ground_prior", "sky", "far"):
        stats[k].append(r.fraction(k))
    stats["chase_yaw_off_deg"].append(math.degrees(yaw_off))
    causes_acc.append(S.classify_unknown(chase, r.masks["unknown"] | (S.black_mask(r.rgb) & ~r.masks["observed"]), evidence))
    if a.classify_dir:
        f = Path(a.classify_dir) / f"f{n_row:04d}.png"
        if f.is_file():
            im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)[cy0:cy0 + ch0, cx0:cx0 + cw0]
            bm = S.black_mask(im); v2_black.append(float(bm.mean()))
            v2_causes.append(S.classify_unknown(chase_rigid_full, bm, evidence))
    if a.save_frames:
        img = cv2.cvtColor(r.rgb, cv2.COLOR_RGB2BGR)
        if a.label:
            cv2.putText(img, a.label, (6, 16 if a.height < 400 else 28), 0, 0.45 if a.height < 400 else 0.8, (255, 255, 255), 1 if a.height < 400 else 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "frames" / f"f{n_row:04d}.png"), img)
        np.savez_compressed(out / "frames" / f"m{n_row:04d}.npz", **{k: np.packbits(v) for k, v in r.masks.items()})
    if len(stats["black"]) % 25 == 1:
        print(f"frame {i}: black {100*stats['black'][-1]:.1f}% unknown {100*stats['unknown'][-1]:.1f}% fuse {stats['fuse_cpu_ms'][-1]:.0f} ms render {stats['render_cpu_ms'][-1]:.0f} ms", flush=True)

def summ(v): return {"mean": round(float(np.mean(v)), 5), "p95": round(float(np.percentile(v, 95)), 5)} if v else None
res = {k: summ(v) for k, v in stats.items()}
res["causes_mean_frac_of_all_px"] = {c: round(float(np.mean([d[c] for d in causes_acc])), 5) for c in S.CAUSES} if causes_acc else None
if v2_causes:
    res["v2_black_mean"] = round(float(np.mean(v2_black)), 5)
    res["v2_causes_mean_frac_of_all_px"] = {c: round(float(np.mean([d[c] for d in v2_causes])), 5) for c in S.CAUSES}
fr = sorted((out / "frames").glob("f*.png"))
if len(fr) >= 3:
    res["temporal_jitter"] = round(S.temporal_jitter([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in fr]), 4)
res["ground_clamp_frac_by_cam"] = {c: round(float(np.mean(v)), 5) for c, v in CLAMP_STATS.items()}
res["yield_frac_by_cam"] = {c: round(float(m.mean()), 4) for c, m in YIELD.items()}
res["road_snap_frac_by_cam"] = {c: round(float(np.mean(v)), 5) for c, v in SNAP_STATS.items()}
res["far_at_cap_frac_by_cam"] = {c: round(float(np.mean(v)), 5) for c, v in FAR_STATS.items()}
res["depth_cap_m_by_cam"] = {c: round(S.depth_cap_m(cams_json[c]["far_m"]), 2) for c in CAMS}
res["config"] = vars(a)
(out / "stats.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
