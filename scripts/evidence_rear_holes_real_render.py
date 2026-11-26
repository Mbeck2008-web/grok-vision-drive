"""Evidence (not a test; not collected by run_offline_suite, which only runs scripts/test_*.py): the rear-holes synthetic,
props 0.3 m fixed splat vs the REAL preview render path (point_footprint + GroundGrid + real splat cap).

Read-only measurement of the checked-out tree; no production change. Run from the repo root:
    python scripts/evidence_rear_holes_real_render.py            (optional arg: repo root, default = this file's repo)
"""
import sys, os, importlib.util, math
from pathlib import Path
import numpy as np
wt = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
os.chdir(wt)
spec = importlib.util.spec_from_file_location("props", wt / "scripts" / "test_rv_side_rear_props.py")
m = importlib.util.module_from_spec(spec); sys.modules["props"] = m; spec.loader.exec_module(m)
sf = m.sf
fr = m.synthetic_frame()                      # every cam through cam_frame on the synthetic road (same as check_synthetic_rear_holes)
rend, gtp, rend_foot = [], [], []
cam_pts = []
for cid, x in fr.items():
    cam = x["cam"]
    ys, xs = np.nonzero(x["v"])
    p = sf.unproject(cam, x["d"], ys, xs)
    foot = sf.point_footprint(cam, x["d"])[ys, xs]          # exactly what the live preview does (footprint(cp, depth)[ys, xs])
    k = m.ego_box_filter(p)
    rend.append(p[k]); rend_foot.append(foot[k])
    ys, xs = np.nonzero(x["gt_road"])
    q = sf.unproject(cam, x["gt_d"], ys, xs)
    gtp.append(q[m.ego_box_filter(q)])
rend = np.concatenate(rend); rend_foot = np.concatenate(rend_foot); gtp = np.concatenate(gtp)
W = int(m.pv().defaults["width"]); H = int(m.pv().defaults["height"])
cap_real = sf.splat_max_radius_px(W)

def painted(ch, pts, foot, max_r, fill=True):
    r = sf.compose(ch, pts, np.full((pts.shape[0], 3), 128, np.uint8), foot, sky_rgb=None, ground_rgb=None,
                   splat_scale=1.0, max_radius_px=max_r, fill=fill)
    return r.masks["observed"] | r.masks["filled"], r.masks["observed"]

# real path ground grid, one frame, pose (0,0,0), same weights/cell/foot as tools/rv_stable_preview.py
grid = sf.GroundGrid()
st = rend[:, 2] <= sf.STATIC_MAX_HEIGHT_M
dist = np.hypot(rend[st, 0], rend[st, 1])
grid.add(rend[st], np.full((int(st.sum()), 3), 128.0), 1.0 / (1.0 + dist / 15.0), 0.0)
gp, gc = grid.live(0.0, (0.0, 0.0))
gfoot = np.full(gp.shape[0], sf.GRID_CELL_M * 1.3)
print(f"chase {W}x{H}; real splat cap {cap_real}px (props default cap 4px); rendered pts {rend.shape[0]}; grid cells {gp.shape[0]}; GT road pts {gtp.shape[0]}")
def band_rng(ch):
    rays = sf._pixel_rays(ch); dz = rays[..., 2]; down = dz < -1e-9
    t = np.where(down, -ch.pos[2] / np.where(down, dz, -1.0), np.nan)
    return down, np.hypot(ch.pos[0] + t * rays[..., 0], ch.pos[1] + t * rays[..., 1])
rows = []
for deg, ch in m.rear_views().items():
    down, rng = band_rng(ch)
    gt_m, _ = painted(ch, gtp, np.full(gtp.shape[0], 0.3), 4)            # props' GT road mask (fixed 0.3 m, cap 4)
    sel = gt_m & down & (rng >= 15) & (rng < 30)
    n = int(sel.sum())
    variants = {
      "A props splat (0.3 m fixed, cap 4, fill)":                          (rend, np.full(rend.shape[0], 0.3), 4, True),
      "B real footprint, props cap 4, no grid, fill":                      (rend, rend_foot, 4, True),
      "C real footprint, real cap, no grid, fill":                         (rend, rend_foot, cap_real, True),
      "D real footprint + grid, real cap, fill (LIVE PATH)":               (np.concatenate([rend, gp]), np.concatenate([rend_foot, gfoot]), cap_real, True),
      "E same as D, observed only (no gap-fill)":                          (np.concatenate([rend, gp]), np.concatenate([rend_foot, gfoot]), cap_real, False),
    }
    for name, (pp, ff, mr, fl) in variants.items():
        pm, obs = painted(ch, pp, ff, mr, fl)
        mm = obs if name.startswith("E") else pm
        holes = int((sel & ~mm).sum())
        rows.append((deg, name, holes, n, 100.0 * holes / max(n, 1)))
for r in rows:
    print(f"{r[0]:>7g} deg  15-30 m  holes {r[2]:>4d} / {r[3]:>5d} = {r[4]:6.2f}%   {r[1]}")
# what the grid contributes at 15-30 m
d = np.hypot(gp[:, 0], gp[:, 1])
print(f"grid cells by range: <15 m {int((d<15).sum())}, 15-30 m {int(((d>=15)&(d<30)).sum())}, >=30 m {int((d>=30).sum())}; rendered pts 15-30 m (hypot) {int(((np.hypot(rend[:,0],rend[:,1])>=15)&(np.hypot(rend[:,0],rend[:,1])<30)).sum())}")
