#!/usr/bin/env python3
"""Build cpp/viz/assets/tesla_model3_ego.glb from David_Holiday's car-only Model 3.

Source (do not ship model3_scene.glb): the car-only extract of
"Tesla Model 3" by David_Holiday, CC BY 4.0.
Removes the embossed hood T badge, converts to the visualizer ego frame
(X right, Y forward, Z up, length-centered, wheels on the ground), and
writes vertex colours (body / glass / wheels / lamps) for the flat mesh shader.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
# Default source lives next to the UI refs, not in the repo.
DEFAULT_SRC = Path("/workspace/gvd-ui/tesla-models/tesla-model-3-david-holiday/model3_car_only.glb")
OUT = Path(__file__).resolve().parents[1] / "assets" / "tesla_model3_ego.glb"


def parse_glb(path: Path):
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise SystemExit(f"not a GLB: {path}")
    off = 12
    chunks = {}
    while off < len(data):
        clen, ctype = struct.unpack_from("<I4s", data, off)
        chunks[ctype.decode().strip("\x00")] = data[off + 8 : off + 8 + clen]
        off += 8 + clen
    return json.loads(chunks["JSON"]), chunks["BIN"]


def accessor_np(gltf, blob, acc_idx):
    acc = gltf["accessors"][acc_idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    typ, ctype = acc["type"], acc["componentType"]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[typ]
    dtype = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[ctype]
    count = acc["count"]
    if ncomp > 1:
        return np.frombuffer(blob, dtype=dtype, count=count * ncomp, offset=off).reshape(count, ncomp).copy()
    return np.frombuffer(blob, dtype=dtype, count=count, offset=off).copy()


def write_glb(path: Path, verts: np.ndarray, normals: np.ndarray, colors: np.ndarray, indices: np.ndarray, name: str):
    """Uncompressed GLB: one mesh, POSITION/NORMAL/COLOR_0 (vec4), uint32 indices."""
    assert verts.dtype == np.float32 and verts.ndim == 2 and verts.shape[1] == 3
    assert normals.shape == verts.shape and colors.shape == (len(verts), 4)
    indices = indices.astype(np.uint32, copy=False).reshape(-1)
    bin_parts = [indices.tobytes(), verts.tobytes(), normals.tobytes(), colors.astype(np.float32).tobytes()]
    # 4-byte align each
    blob = bytearray()
    views = []
    for part in bin_parts:
        if len(blob) % 4:
            blob.extend(b"\x00" * (4 - len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(part)})
        blob.extend(part)
    if len(blob) % 4:
        blob.extend(b"\x00" * (4 - len(blob) % 4))
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5125,
            "count": int(indices.size),
            "type": "SCALAR",
            "max": [int(indices.max())],
            "min": [int(indices.min())],
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": int(len(verts)),
            "type": "VEC3",
            "max": verts.max(0).tolist(),
            "min": verts.min(0).tolist(),
        },
        {"bufferView": 2, "componentType": 5126, "count": int(len(normals)), "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": int(len(colors)), "type": "VEC4"},
    ]
    gltf = {
        "asset": {"version": "2.0", "generator": "gvd_viz prepare_ego_glb"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 1, "NORMAL": 2, "COLOR_0": 3},
                        "indices": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    if len(json_bytes) % 4:
        json_bytes += b" " * (4 - len(json_bytes) % 4)
    length = 12 + 8 + len(json_bytes) + 8 + len(blob)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, length)  # glTF, v2, total
    out += struct.pack("<I4s", len(json_bytes), b"JSON")
    out += json_bytes
    out += struct.pack("<I4s", len(blob), b"BIN\x00")
    out += blob
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path, len(indices) // 3, len(verts)


def classify(cent: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Face labels in ego frame (X right, Y forward, Z up). 0 body, 1 glass, 2 wheel, 3 tail, 4 head."""
    x, y, z = cent[:, 0], cent[:, 1], cent[:, 2]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    lab = np.zeros(len(cent), dtype=np.int32)
    # wheels: only the axle barrels, not the rocker or bumper
    rear_axle = np.abs(y + 1.35) < 0.50
    front_axle = np.abs(y - 1.40) < 0.50
    lab[(z < 0.42) & (np.abs(x) > 0.58) & (rear_axle | front_axle)] = 2
    # windshield
    lab[(z > 0.78) & (y > 0.15) & (y < 1.05) & (nz > -0.15) & (ny < 0.92) & (np.abs(x) < 0.85)] = 1
    # rear window
    lab[(z > 0.85) & (y < -1.35) & (nz < 0.15) & (ny < 0.95)] = 1
    # side glass
    lab[(z > 0.80) & (np.abs(nx) > 0.55) & (y > -1.55) & (y < 0.85)] = 1
    # tail lamps: facing aft, outer, mid height
    lab[(y < -2.02) & (z > 0.52) & (z < 0.90) & (np.abs(x) > 0.48) & (ny < -0.15)] = 3
    # headlights: facing forward, outer
    lab[(y > 1.95) & (z > 0.48) & (z < 0.78) & (np.abs(x) > 0.40) & (ny > 0.15)] = 4
    return lab


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    gltf, blob = parse_glb(args.src)
    prim = gltf["meshes"][0]["primitives"][0]
    pos = accessor_np(gltf, blob, prim["attributes"]["POSITION"]).astype(np.float32)
    faces = accessor_np(gltf, blob, prim["indices"]).astype(np.int64).reshape(-1, 3)
    p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    cent = (p0 + p1 + p2) / 3.0
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    # Hood T badge: a tight cluster of small faces on the front of the hood
    # (source frame: X right, Y up, Z forward). Large hood panels that only
    # touch the region are kept (area threshold).
    badge = (
        (np.abs(cent[:, 0]) < 0.045)
        & (cent[:, 1] > 0.640)
        & (cent[:, 1] < 0.705)
        & (cent[:, 2] > 1.470)
        & (cent[:, 2] < 1.560)
        & (area < 5.0e-4)
    )
    print(f"source tris {len(faces)} verts {len(pos)}; badge tris {int(badge.sum())}")
    keep = ~badge
    faces = faces[keep]
    # drop unused verts
    used = np.unique(faces.ravel())
    remap = np.full(len(pos), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    pos = pos[used]
    faces = remap[faces]

    # Source (X right, Y up, Z forward) -> ego (X right, Y forward, Z up).
    # The permutation (x, z, y) would flip handedness, so X is mirrored.
    zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
    ymin = float(pos[:, 1].min())
    zmid = 0.5 * (zmin + zmax)
    ego = np.empty_like(pos)
    ego[:, 0] = -pos[:, 0]
    ego[:, 1] = pos[:, 2] - zmid
    ego[:, 2] = pos[:, 1] - ymin
    print(
        f"ego bbox min={ego.min(0)} max={ego.max(0)} size={ego.max(0) - ego.min(0)} "
        f"length={ego[:, 1].max() - ego[:, 1].min():.3f} m"
    )

    p0, p1, p2 = ego[faces[:, 0]], ego[faces[:, 1]], ego[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(ln, 1e-12)
    cent = (p0 + p1 + p2) / 3.0
    # outward-facing
    centre = np.array([0.0, 0.0, 0.7], dtype=np.float32)
    flip = np.sum(fn * (cent - centre), axis=1) < 0
    fn[flip] *= -1
    faces[flip] = faces[flip][:, ::-1]

    lab = classify(cent, fn)
    pal = np.array(
        [
            [0.92, 0.93, 0.95, 0.0],  # body
            [0.10, 0.11, 0.13, 0.0],  # glass
            [0.07, 0.07, 0.08, 0.0],  # wheel
            [1.00, 0.12, 0.10, 1.0],  # tail
            [0.95, 0.95, 0.90, 1.0],  # head
        ],
        dtype=np.float32,
    )
    # vertex smooth normals + majority colour
    vnor = np.zeros((len(ego), 3), dtype=np.float64)
    vacc = np.zeros((len(ego), 5), dtype=np.float64)
    for i in range(3):
        np.add.at(vnor, faces[:, i], fn)
        np.add.at(vacc, faces[:, i], np.eye(5)[lab])
    vnor = (vnor / np.maximum(np.linalg.norm(vnor, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    vlab = np.argmax(vacc, axis=1)
    vcol = pal[vlab]
    counts = {k: int((lab == i).sum()) for i, k in enumerate(["body", "glass", "wheel", "tail", "head"])}
    print("face labels", counts)

    path, tris, verts = write_glb(args.out, ego.astype(np.float32), vnor, vcol, faces.astype(np.uint32), "Tesla_Model_3")
    print(f"wrote {path}  tris={tris} verts={verts} bytes={path.stat().st_size}")
    if tris >= 30000:
        raise SystemExit(f"triangle count {tris} is not under 30k")


if __name__ == "__main__":
    main()
