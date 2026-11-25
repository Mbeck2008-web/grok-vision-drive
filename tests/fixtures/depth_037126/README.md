# depth_037126

Training-only depth and semantic planes of one authentic main-camera frame
(`sgt_2h_20261105`, strip `ctl_20261105_p01`, frame `037126`, west_coast_usa,
Hirochi eSBR, collector `train_gt` on main). The source tree was deleted from
E: on 2026-11-11 (approved); these two PNGs and `sem_classes.json` are the box
copy used by the Phase 0 audit (tracer section 4.2).

- `raw/main_depth/037126.png`: uint16 cm, 480x640, written as `depth01 * far_m * 100`.
- `raw/main_sem/037126.png`: uint8 class id (`sem_classes.json`).
- Camera: main, fov_v 34.0, 640x480, near 0.05 m, far 300 m, yaml pitch -6 deg,
  yaml `pos_m.z` 1.26 m (etk800 draft mount).
- Ego state at that tick: speed 22.8 m/s, `dir.z` -0.0018 (car pitch about 0.1 deg).

`scripts/test_recorder_v2.py` fits the road plane to this frame to decide
whether the stored value is a planar optical-axis depth or a ray length.
