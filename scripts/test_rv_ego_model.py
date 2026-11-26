#!/usr/bin/env python3
"""Offline CPU checks for the Reality View ego car (python/viz/rv_ego_model.py).

E1  The shipped GLB (cpp/viz/assets/tesla_model3_ego.glb) loads: <= 50k triangles, finite, indices in
    range, file < 5 MB, no Draco; ATTRIBUTION.txt credits David_Holiday under CC BY 4.0.
E2  Scale: the fitted bounding box is 1.85 m wide, 4.72 m long, 1.44 m tall.
E3  Placement: x-centred, wheels on z = 0, +Y forward (tail lamps behind the lengthwise centre, head
    lamps ahead), lengthwise centre on the old wire box centre, inside the old box width and +-0.1 m
    of its ends.
E4  Render: deterministic, alpha in 0..1, painted only inside the projected bounding box, back faces
    culled, the near triangle wins the z-test, geometry at or behind the near plane is dropped.
E5  Fallback: a missing or unreadable mesh file gives None, and draw_ego then draws exactly the old
    12-edge wire box; a Draco / non-GLB file is refused by load_glb.
E6  No new dependency: the module imports only numpy, the stdlib and reality_view_c.
"""
from __future__ import annotations

import ast
import json
import struct
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from python.viz import rv_ego_model as E  # noqa: E402
from python.viz.reality_view_c import camera_from_angles  # noqa: E402

POSE_A = dict(cam_id="chase", pos_m=(0.0, -4.0, 2.6), yaw_deg=0.0, pitch_deg=-12.0, fov_v_deg=60.0, width=320, height=180, near_m=0.05, far_m=100.0)


def _pose(**kw):
    return camera_from_angles(**{**POSE_A, **kw})


def _glb_bytes(gltf: dict, blob: bytes) -> bytes:
    j = json.dumps(gltf).encode()
    j += b" " * (-len(j) % 4)
    blob += b"\x00" * (-len(blob) % 4)
    return struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(j) + 8 + len(blob)) + struct.pack("<I4s", len(j), b"JSON") + j + struct.pack("<I4s", len(blob), b"BIN\x00") + blob


def _tri_glb(path: Path, tris: list[tuple[list[tuple[float, float, float]], tuple[float, float, float]]], draco: bool = False) -> None:
    """GLB of flat-coloured triangles: [(3 verts, rgb), ...]."""
    verts = np.array([v for t, _ in tris for v in t], np.float32)
    cols = np.array([[*c, 0.0] for t, c in tris for _ in t], np.float32)
    idx = np.arange(len(verts), dtype=np.uint32)
    parts = [idx.tobytes(), verts.tobytes(), cols.tobytes()]
    views, blob = [], bytearray()
    for p in parts:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(p)})
        blob += p
    prim = {"attributes": {"POSITION": 1, "COLOR_0": 2}, "indices": 0, "mode": 4}
    if draco:
        prim["extensions"] = {"KHR_draco_mesh_compression": {}}
    gltf = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [prim]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(verts), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(verts), "type": "VEC4"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    path.write_bytes(_glb_bytes(gltf, bytes(blob)))


def _raw_mesh() -> "E.EgoMesh":
    return E.load_glb(E.DEFAULT_GLB)


def check_load() -> None:
    assert E.DEFAULT_GLB.is_file(), E.DEFAULT_GLB
    assert E.DEFAULT_GLB.stat().st_size < 5 * 1024 * 1024
    m = _raw_mesh()
    assert 0 < m.n_tris <= 50_000, m.n_tris
    assert m.verts.dtype == np.float32 and m.verts.shape[1] == 3 and np.isfinite(m.verts).all()
    assert m.faces.min() >= 0 and m.faces.max() < len(m.verts)
    assert m.colors.shape == (len(m.verts), 4)
    txt = (E.DEFAULT_GLB.parent / "ATTRIBUTION.txt").read_text()
    assert "David_Holiday" in txt and "CC BY 4.0" in txt and "sketchfab.com/3d-models/tesla-model-3-123c10f376ec4f18b93c73afc382808b" in txt


def check_scale_and_placement() -> None:
    m = E.load_ego_mesh()
    assert m is not None
    lo, hi = m.bounds()
    ext = hi - lo
    assert np.allclose(ext, [E.MODEL3_WIDTH_M, E.MODEL3_LENGTH_M, E.MODEL3_HEIGHT_M], atol=1e-4), ext
    assert (E.MODEL3_WIDTH_M, E.MODEL3_LENGTH_M, E.MODEL3_HEIGHT_M) == (1.85, 4.72, 1.44)
    assert abs(lo[2]) < 1e-5, "wheels must sit on z=0"
    assert abs(0.5 * (lo[0] + hi[0])) < 1e-4, "x-centred"
    assert abs(0.5 * (lo[1] + hi[1]) - E.BOX_Y_CENTER) < 1e-4, "lengthwise centre = old box centre"
    # still inside the old box width, and within 0.1 m of its ends and height
    assert hi[0] <= E.BOX_HALF_W and lo[0] >= -E.BOX_HALF_W
    assert abs(lo[1] - E.BOX_Y0) < 0.1 and abs(hi[1] - E.BOX_Y1) < 0.1 and abs(hi[2] - E.BOX_H) < 0.1
    # forward axis: red lamps (alpha>0.5, r >> g) are at the tail, the others at the nose
    lamp = m.colors[:, 3] > 0.5
    red = lamp & (m.colors[:, 0] > 2.0 * m.colors[:, 1])
    head = lamp & ~red
    assert red.any() and head.any()
    assert m.verts[red, 1].mean() < E.BOX_Y_CENTER - 1.5 and m.verts[head, 1].mean() > E.BOX_Y_CENTER + 1.5
    # rear axle position follows the length scale: wheel arches sit ahead of / behind centre, not at the ends
    assert E.BOX_Y0 < m.verts[red, 1].min() < E.BOX_Y0 + 0.2
    # the ground rule holds for any source mesh, not only this one (source floating 0.3 m up, off-centre)
    raw = _raw_mesh()
    shifted = E.EgoMesh(raw.verts + np.array([0.7, 3.0, 0.3], np.float32), raw.colors, raw.faces)
    ms = E.fit_to_ego(shifted)
    assert abs(ms.bounds()[0][2]) < 1e-4 and abs(0.5 * (ms.bounds()[0][0] + ms.bounds()[1][0])) < 1e-4
    assert abs(0.5 * (ms.bounds()[0][1] + ms.bounds()[1][1]) - E.BOX_Y_CENTER) < 1e-4
    # bad targets are refused
    for kw in ({"length_m": 0.0}, {"width_m": -1.0}, {"height_m": float("nan")}):
        try:
            E.fit_to_ego(_raw_mesh(), **kw)
        except ValueError:
            continue
        raise AssertionError(f"fit_to_ego accepted {kw}")


def _proj_bbox(m: "E.EgoMesh", chase) -> tuple[float, float, float, float]:
    lo, hi = m.bounds()
    c = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float64)
    u, v, z = E._project(c, chase)
    assert (z > chase.near_m).all()
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def check_render() -> None:
    m = E.load_ego_mesh()
    chase = _pose()
    rgb, alpha = E.render_ego_layer(m, chase)
    assert rgb.shape == (180, 320, 3) and rgb.dtype == np.uint8 and alpha.shape == (180, 320) and alpha.dtype == np.float32
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0 and np.isfinite(alpha).all()
    assert (alpha > 0.99).mean() > 0.04, "car should cover a visible part of the chase view"
    rgb2, alpha2 = E.render_ego_layer(m, chase)
    assert rgb.tobytes() == rgb2.tobytes() and alpha.tobytes() == alpha2.tobytes(), "deterministic"
    u0, u1, v0, v1 = _proj_bbox(m, chase)
    ys, xs = np.nonzero(alpha > 0)
    assert xs.min() >= np.floor(u0) - 1 and xs.max() <= np.ceil(u1) + 1 and ys.min() >= np.floor(v0) - 1 and ys.max() <= np.ceil(v1) + 1
    assert ((alpha > 0.0) & (alpha < 1.0)).any(), "edges are anti-aliased"
    # shaded, not a flat silhouette: many distinct grey levels on the car
    assert len(np.unique(rgb[alpha > 0.99].reshape(-1, 3), axis=0)) > 30
    # composite touches only covered pixels and is bounded
    bg = np.full((180, 320, 3), (10, 200, 10), np.uint8)
    out = E.composite(bg.copy(), (rgb, alpha))
    assert (out[alpha == 0] == (10, 200, 10)).all() and (out[alpha > 0.99] == rgb[alpha > 0.99]).all()
    # camera inside the car: everything is behind or at the near plane or culled, nothing crashes
    inside = _pose(pos_m=(0.0, -0.025, 0.7), pitch_deg=0.0)
    _, a_in = E.render_ego_layer(m, inside)
    assert np.isfinite(a_in).all()


def check_zbuffer_and_culling() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.glb"
        # camera looks along +Y from y=-4 at z=0 (pitch 0). Near red triangle at y=0, far blue one at y=2, both facing -Y.
        near = ([(-1.0, 0.0, -1.0), (1.0, 0.0, -1.0), (0.0, 0.0, 1.0)], (1.0, 0.0, 0.0))
        far = ([(-3.0, 2.0, -3.0), (3.0, 2.0, -3.0), (0.0, 2.0, 3.0)], (0.0, 0.0, 1.0))
        _tri_glb(p, [far, near])
        m = E.load_glb(p)
        chase = _pose(pos_m=(0.0, -4.0, 0.0), pitch_deg=0.0, width=64, height=64)
        rgb, alpha = E.render_ego_layer(m, chase, supersample=1)
        # which way do these triangles face? flip the winding if they are culled
        if alpha.max() == 0:
            m = E.EgoMesh(m.verts, m.colors, m.faces[:, ::-1].copy())
            rgb, alpha = E.render_ego_layer(m, chase, supersample=1)
        assert alpha.max() == 1.0
        c = rgb[32, 32]
        assert c[0] > c[2] * 3, f"near (red) triangle must win the z-test, got {c}"
        edge = rgb[56, 32]  # only the big far triangle reaches here
        assert edge[2] > edge[0] * 3, f"far (blue) triangle visible outside the near one, got {edge}"
        # back faces are culled: reversing the winding of both gives an empty layer
        flipped = E.EgoMesh(m.verts, m.colors, m.faces[:, ::-1].copy())
        assert E.render_ego_layer(flipped, chase, supersample=1)[1].max() == 0.0
        # also from close up (a loose cull threshold would keep these)
        close = _pose(pos_m=(0.0, -1.0, 0.0), pitch_deg=0.0, width=64, height=64)
        assert E.render_ego_layer(m, close, supersample=1)[1].max() == 1.0
        assert E.render_ego_layer(flipped, close, supersample=1)[1].max() == 0.0
        # geometry behind the camera is dropped
        behind = _pose(pos_m=(0.0, 5.0, 0.0), pitch_deg=0.0, width=64, height=64)
        assert E.render_ego_layer(m, behind, supersample=1)[1].max() == 0.0


def check_fallback() -> None:
    chase = _pose()
    missing = ROOT / "does" / "not" / "exist.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a missing file is the normal fallback, not a warning
        assert E.load_ego_mesh(missing) is None
    base = np.zeros((180, 320, 3), np.uint8)
    img = base.copy()
    assert E.draw_ego(img, chase, None) == "box"
    segs = E.wire_box_segments(chase)
    assert len(segs) == 12
    # old geometry preserved: same 8 corners as rv_multicam.project_wire_box defaults
    assert (E.BOX_HALF_W, E.BOX_Y0, E.BOX_Y1, E.BOX_H) == (0.95, -2.35, 2.3, 1.45)
    exp = base.copy()
    for p0, p1 in segs:
        E._draw_segment(exp, p0, p1, E.WIRE_RGB)
    assert (img == exp).all() and img.any()
    assert tuple(img[img.any(2)][0]) == E.WIRE_RGB
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.glb"
        bad.write_bytes(b"not a glb at all, just text")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert E.load_ego_mesh(bad) is None
        assert any(issubclass(x.category, RuntimeWarning) for x in w), "unreadable mesh must warn"
        trunc = Path(d) / "trunc.glb"
        trunc.write_bytes(E.DEFAULT_GLB.read_bytes()[:3000])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert E.load_ego_mesh(trunc) is None
        dr = Path(d) / "draco.glb"
        _tri_glb(dr, [([(0, 0, 0), (1, 0, 0), (0, 0, 1)], (1, 1, 1))], draco=True)
        try:
            E.load_glb(dr)
        except ValueError:
            pass
        else:
            raise AssertionError("Draco GLB accepted")
    # model path: draw_ego with a real mesh paints the car, not the box colour
    m = E.load_ego_mesh()
    img2 = base.copy()
    assert E.draw_ego(img2, chase, m) == "model" and img2.any()
    assert not (img2 == np.array(E.WIRE_RGB, np.uint8)).all(2).any()


def check_no_new_deps() -> None:
    tree = ast.parse((ROOT / "python" / "viz" / "rv_ego_model.py").read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    allowed = {"__future__", "json", "struct", "warnings", "dataclasses", "pathlib", "numpy", "python"}
    assert mods <= allowed, mods - allowed


def main() -> None:
    check_load()
    check_scale_and_placement()
    check_render()
    check_zbuffer_and_culling()
    check_fallback()
    check_no_new_deps()
    print("test_rv_ego_model: OK")


if __name__ == "__main__":
    main()


def test_load() -> None:
    check_load()


def test_scale_and_placement() -> None:
    check_scale_and_placement()


def test_render() -> None:
    check_render()


def test_zbuffer_and_culling() -> None:
    check_zbuffer_and_culling()


def test_fallback() -> None:
    check_fallback()


def test_no_new_deps() -> None:
    check_no_new_deps()
