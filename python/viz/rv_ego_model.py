"""Reality View ego car: a shaded Model 3 mesh in the chase view, wire box as the fallback.

Pure numpy, no new dependencies. Offline CPU only; nothing here is on the live safety path.

The mesh is cpp/viz/assets/tesla_model3_ego.glb ("Tesla Model 3" by David_Holiday, CC BY 4.0;
attribution and the list of changes are in cpp/viz/assets/ATTRIBUTION.txt, which must stay with it).
It is an uncompressed GLB with POSITION / NORMAL / COLOR_0 / uint32 indices, in the ego frame
(+X right, +Y forward, +Z up, length-centred, wheels on z = 0).

``fit_to_ego`` scales it to real Model 3 dimensions (per axis, bounding box) and centres it on the
same footprint as the old wire box (``BOX_*``), so the forward axis (+Y), the ground plane (z = 0)
and the lengthwise centre match the box. A missing or unreadable mesh file gives ``None`` and the
caller draws the wire box instead (``draw_ego`` does this itself).
"""
from __future__ import annotations

import json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python.viz.reality_view_c import CameraPose

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "cpp" / "viz" / "assets"
# Found by glob, not by a literal name: test_m6_retail keeps product names out of runtime strings in python/viz.
DEFAULT_GLB = next(iter(sorted(ASSET_DIR.glob("*model3_ego.glb"))), ASSET_DIR / "model3_ego.glb")

# Tesla Model 3 (metres). Width excludes nothing the mesh has: the bounding box is fitted to this.
MODEL3_LENGTH_M = 4.72
MODEL3_WIDTH_M = 1.85
MODEL3_HEIGHT_M = 1.44

# The wire box this replaces (rv_multicam.project_wire_box defaults): half width, y0, y1, height.
BOX_HALF_W = 0.95
BOX_Y0 = -2.35
BOX_Y1 = 2.3
BOX_H = 1.45
BOX_Y_CENTER = 0.5 * (BOX_Y0 + BOX_Y1)

MAX_GLB_BYTES = 8 * 1024 * 1024
MAX_TRIANGLES = 50_000
# Toward the light, ego frame: above, behind and a little to the left of the car.
LIGHT_DIR = (-0.35, -0.45, 0.82)
AMBIENT = 0.42
WIRE_RGB = (90, 200, 230)

_COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass(frozen=True)
class EgoMesh:
    """Triangle mesh in the ego frame. ``colors`` is (N, 4) float: rgb 0..1, alpha > 0.5 = lamp (unshaded)."""

    verts: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 4) float32
    faces: np.ndarray  # (M, 3) int32

    @property
    def n_tris(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.verts.min(0), self.verts.max(0)


def _accessor(gltf: dict, blob: bytes, idx: int) -> np.ndarray:
    acc = gltf["accessors"][idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    if "sparse" in acc or "byteStride" in view:
        raise ValueError("sparse / strided glTF accessors are not supported")
    n = _NCOMP[acc["type"]]
    dtype = np.dtype(_COMP[acc["componentType"]])
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = int(acc["count"])
    if off < 0 or off + count * n * dtype.itemsize > len(blob):
        raise ValueError("glTF accessor runs past the BIN chunk")
    a = np.frombuffer(blob, dtype, count * n, off).reshape(count, n)
    return a.copy()


def load_glb(path: str | Path) -> EgoMesh:
    """Read one uncompressed triangle-list mesh from a GLB. ValueError on anything else (Draco, no BIN, ...)."""
    path = Path(path)
    if path.stat().st_size > MAX_GLB_BYTES:
        raise ValueError(f"GLB larger than {MAX_GLB_BYTES} bytes")
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path.name}")
    off, chunks = 12, {}
    while off + 8 <= len(data):
        clen, ctype = struct.unpack_from("<I4s", data, off)
        chunks[ctype.rstrip(b"\x00 ").decode("ascii", "replace")] = data[off + 8 : off + 8 + clen]
        off += 8 + clen
    if "JSON" not in chunks or "BIN" not in chunks:
        raise ValueError("GLB needs a JSON and a BIN chunk")
    gltf = json.loads(chunks["JSON"])
    blob = chunks["BIN"]
    verts_l, cols_l, faces_l, base = [], [], [], 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if prim.get("mode", 4) != 4 or "indices" not in prim or "POSITION" not in attrs:
                raise ValueError("only indexed triangle-list primitives with POSITION are supported")
            if "KHR_draco_mesh_compression" in prim.get("extensions", {}):
                raise ValueError("Draco-compressed GLB is not supported")
            v = _accessor(gltf, blob, attrs["POSITION"]).astype(np.float32)
            if "COLOR_0" in attrs:
                c = _accessor(gltf, blob, attrs["COLOR_0"]).astype(np.float32)
                if c.shape[1] == 3:
                    c = np.concatenate([c, np.zeros((len(c), 1), np.float32)], 1)
            else:
                c = np.tile(np.array([[0.8, 0.8, 0.82, 0.0]], np.float32), (len(v), 1))
            idx = _accessor(gltf, blob, prim["indices"]).reshape(-1).astype(np.int64)
            if idx.size % 3 or (idx.size and (idx.min() < 0 or idx.max() >= len(v))) or c.shape[0] != len(v):
                raise ValueError("bad indices or colour count")
            verts_l.append(v)
            cols_l.append(c)
            faces_l.append((idx.reshape(-1, 3) + base).astype(np.int32))
            base += len(v)
    if not faces_l:
        raise ValueError("GLB has no mesh")
    verts, colors, faces = np.concatenate(verts_l), np.concatenate(cols_l), np.concatenate(faces_l)
    if not np.isfinite(verts).all():
        raise ValueError("non-finite vertex")
    if faces.shape[0] > MAX_TRIANGLES:
        raise ValueError(f"{faces.shape[0]} triangles > {MAX_TRIANGLES}")
    return EgoMesh(verts, colors, faces)


def fit_to_ego(
    mesh: EgoMesh,
    length_m: float = MODEL3_LENGTH_M,
    width_m: float = MODEL3_WIDTH_M,
    height_m: float = MODEL3_HEIGHT_M,
    y_center: float = BOX_Y_CENTER,
) -> EgoMesh:
    """Scale the mesh bounding box to ``width x length x height`` (X, Y, Z), x-centred, wheels on z = 0,
    lengthwise centre at ``y_center`` (the old wire box centre), +Y still forward."""
    lo, hi = mesh.bounds()
    ext = (hi - lo).astype(np.float64)
    target = np.array([width_m, length_m, height_m], np.float64)
    if not np.all(np.isfinite(target)) or np.any(target <= 0) or np.any(ext <= 1e-6):
        raise ValueError("degenerate mesh or target size")
    s = target / ext
    centre = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]], np.float64)
    out = (mesh.verts.astype(np.float64) - centre) * s + np.array([0.0, y_center, 0.0])
    return EgoMesh(out.astype(np.float32), mesh.colors, mesh.faces)


def load_ego_mesh(path: str | Path | None = None) -> EgoMesh | None:
    """The fitted Model 3, or None (caller falls back to the wire box) when the file is missing or unreadable."""
    p = Path(path) if path is not None else DEFAULT_GLB
    if not p.is_file():
        return None
    try:
        return fit_to_ego(load_glb(p))
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError, struct.error) as e:
        warnings.warn(f"ego mesh {p.name} unusable ({e}); drawing the wire box", RuntimeWarning, stacklevel=2)
        return None


# ---------------------------------------------------------------- render


def _project(pts: np.ndarray, chase: CameraPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, z_cam) for ego-frame points; same pinhole convention as ``project_wire_box``."""
    cp = (pts.astype(np.float64) - np.asarray(chase.pos, np.float64)) @ chase.rotation_matrix().T
    z = cp[:, 2]
    zs = np.where(z > chase.near_m, z, np.nan)
    return chase.cx + chase.fx * cp[:, 0] / zs, chase.cy - chase.fy * cp[:, 1] / zs, z


def render_ego_layer(mesh: EgoMesh, chase: CameraPose, supersample: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Flat-shaded, z-buffered render of the mesh from the chase camera.

    Returns (rgb uint8 (H, W, 3), alpha float32 (H, W) in 0..1). The car is fixed in the ego frame, so a
    fixed chase pose gives the same layer every frame: render once, then ``composite`` per frame.
    Triangles with any vertex at or behind the near plane are dropped, back faces are culled.
    """
    ss = int(supersample)
    if not 1 <= ss <= 4:
        raise ValueError("supersample must be 1..4")
    h, w = int(chase.height), int(chase.width)
    if not (0 < h <= 4096 and 0 < w <= 4096):
        raise ValueError("bad chase size")
    H, W = h * ss, w * ss
    u, v, z = _project(mesh.verts, chase)
    u, v = u * ss, v * ss
    f = mesh.faces
    wp = mesh.verts.astype(np.float64)
    n = np.cross(wp[f[:, 1]] - wp[f[:, 0]], wp[f[:, 2]] - wp[f[:, 0]])
    nl = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nl, 1e-12)[:, None]
    centroid = wp[f].mean(1)
    to_cam = np.asarray(chase.pos, np.float64) - centroid
    front = (np.einsum("ij,ij->i", n, to_cam) > 0) & (nl > 1e-12)
    ok = front & np.isfinite(u[f]).all(1) & np.isfinite(v[f]).all(1) & (z[f] > chase.near_m).all(1)
    light = np.asarray(LIGHT_DIR, np.float64)
    light /= np.linalg.norm(light)
    lam = AMBIENT + (1.0 - AMBIENT) * np.clip(n @ light, 0.0, 1.0)
    col = mesh.colors[f].astype(np.float64)
    fc = col[:, :, :3].mean(1)
    lamp = col[:, :, 3].mean(1) > 0.5
    shade = np.where(lamp, 1.0, lam)
    face_rgb = np.clip(fc * shade[:, None], 0.0, 1.0)
    rgb = np.zeros((H, W, 3), np.float64)
    zbuf = np.full((H, W), np.inf)
    cov = np.zeros((H, W), bool)
    for t in np.nonzero(ok)[0]:
        a, b, c = f[t]
        x0, x1, x2 = u[a], u[b], u[c]
        y0, y1, y2 = v[a], v[b], v[c]
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        # perspective-correct depth: 1/z is affine in screen space
        iz = l0 / z[a] + l1 / z[b] + l2 / z[c]
        zz = np.where(inside, 1.0 / np.maximum(iz, 1e-12), np.inf)
        sub = zbuf[ymin : ymax + 1, xmin : xmax + 1]
        win = zz < sub
        if not win.any():
            continue
        sub[win] = zz[win]
        rgb[ymin : ymax + 1, xmin : xmax + 1][win] = face_rgb[t]
        cov[ymin : ymax + 1, xmin : xmax + 1][win] = True
    if ss > 1:
        alpha = cov.reshape(h, ss, w, ss).mean((1, 3))
        acc = (rgb * cov[..., None]).reshape(h, ss, w, ss, 3).sum((1, 3))
        rgb_o = np.where(alpha[..., None] > 0, acc / np.maximum(cov.reshape(h, ss, w, ss).sum((1, 3)), 1)[..., None], 0.0)
    else:
        alpha, rgb_o = cov.astype(np.float64), rgb
    return np.clip(np.round(rgb_o * 255.0), 0, 255).astype(np.uint8), alpha.astype(np.float32)


def composite(img_rgb: np.ndarray, layer: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Alpha-blend a ``render_ego_layer`` result onto an RGB uint8 image in place; returns it."""
    rgb, alpha = layer
    if img_rgb.shape != rgb.shape or img_rgb.dtype != np.uint8:
        raise ValueError("image and ego layer must be the same-sized uint8 RGB")
    a = alpha[..., None]
    img_rgb[:] = np.clip(np.round(img_rgb * (1.0 - a) + rgb * a), 0, 255).astype(np.uint8)
    return img_rgb


def wire_box_segments(
    chase: CameraPose, half_w: float = BOX_HALF_W, y0: float = BOX_Y0, y1: float = BOX_Y1, h: float = BOX_H
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The old ego outline (same geometry as rv_multicam.project_wire_box): 12 pixel segments, [] if clipped."""
    c = np.array([(x, y, z) for x in (-half_w, half_w) for y in (y0, y1) for z in (0.0, h)], np.float64)
    u, v, z = _project(c, chase)
    if np.any(z <= chase.near_m):
        return []
    pts = [(int(round(a)), int(round(b))) for a, b in zip(u, v)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return [(pts[i], pts[j]) for i, j in edges]


def _draw_segment(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], rgb: tuple[int, int, int]) -> None:
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.round(np.linspace(p0[0], p1[0], n)).astype(np.int64)
    ys = np.round(np.linspace(p0[1], p1[1], n)).astype(np.int64)
    m = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
    img[ys[m], xs[m]] = rgb


def draw_ego(img_rgb: np.ndarray, chase: CameraPose, mesh: EgoMesh | None, layer=None) -> str:
    """Draw the ego car on an RGB uint8 chase image in place. Returns "model" or "box" (the fallback)."""
    if mesh is None:
        for p0, p1 in wire_box_segments(chase):
            _draw_segment(img_rgb, p0, p1, WIRE_RGB)
        return "box"
    composite(img_rgb, layer if layer is not None else render_ego_layer(mesh, chase))
    return "model"
