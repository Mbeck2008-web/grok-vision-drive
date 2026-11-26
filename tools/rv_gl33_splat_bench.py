"""Step-4 proxy prove: GL 3.3 core, CPU-sorted instanced-quad Gaussian splat pass.

Not gvd_viz (no MSVC on IQU45, so gvd_viz is not built); same technique the spec
names for the live rasterizer (antimatter15-style, CPU sort, instanced quads).
Offscreen FBO: RGBA8 colour + RGBA32F alpha-weighted depth (the spec's depth buffer).
Prints exact numbers as JSON. Deterministic synthetic splats unless --npz is given.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from python.viz.rv_splat_sort import sort_splats  # noqa: E402

VS = """
#version 330 core
layout(location=0) in vec2 corner;
layout(location=1) in vec3 center;
layout(location=2) in vec4 color;
layout(location=3) in float sigma_m;
uniform mat4 view;
uniform mat4 proj;
uniform float fy_px;
uniform vec2 viewport;
out vec2 v_uv;
out vec4 v_col;
out float v_z;
void main() {
    vec4 vc = view * vec4(center, 1.0);
    float z = -vc.z;
    vec4 clip = proj * vc;
    float r_px = clamp(3.0 * sigma_m * fy_px / max(z, 0.05), 0.75, 64.0);
    vec2 off = corner * r_px * 2.0 / viewport;
    gl_Position = clip + vec4(off * clip.w, 0.0, 0.0);
    v_uv = corner;
    v_col = color;
    v_z = z;
}
"""
FS = """
#version 330 core
in vec2 v_uv;
in vec4 v_col;
in float v_z;
layout(location=0) out vec4 out_rgb;
layout(location=1) out vec4 out_depth;
void main() {
    float r2 = dot(v_uv, v_uv) * 9.0;
    if (r2 > 9.0) discard;
    float a = v_col.a * exp(-0.5 * r2);
    if (a < 1.0/255.0) discard;
    out_rgb = vec4(v_col.rgb * a, a);
    out_depth = vec4(v_z * a, 0.0, 0.0, a);
}
"""


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64); up = np.asarray(up, np.float64)
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m, f


def perspective(fov_v_deg, aspect, near, far):
    t = 1.0 / np.tan(np.radians(fov_v_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = t / aspect; m[1, 1] = t
    m[2, 2] = (far + near) / (near - far); m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    n_ground = int(n * 0.7)
    g = np.stack([rng.uniform(-20, 20, n_ground), rng.uniform(0, 64, n_ground), rng.normal(0, 0.02, n_ground)], 1)
    nw = n - n_ground
    side = np.where(rng.random(nw) < 0.5, -12.0, 12.0)
    w = np.stack([side + rng.normal(0, 0.3, nw), rng.uniform(0, 64, nw), rng.uniform(0, 6, nw)], 1)
    pos = np.concatenate([g, w]).astype(np.float32)
    col = np.concatenate([rng.uniform(0.2, 0.9, (n, 3)), np.full((n, 1), 0.9)], 1).astype(np.float32)
    sig = rng.uniform(0.05, 0.3, n).astype(np.float32)
    return pos, col, sig


def gpu_dedicated_mb(pid):
    ps = ("$s=(Get-Counter -Counter '\\GPU Process Memory(pid_%d_*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples;"
          "if($s){[int64](($s | Measure-Object CookedValue -Sum).Sum)}else{-1}") % pid
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.strip()
        v = int(out.splitlines()[-1])
        return round(v / 2**20, 1) if v >= 0 else None
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def nvsmi_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--npz", default="")
    ap.add_argument("--png", default="")
    a = ap.parse_args()
    proc = psutil.Process(os.getpid())
    nv_before = nvsmi_used_mb()
    rss_before = proc.memory_info().rss / 2**20

    if not glfw.init():
        raise SystemExit("glfw.init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    win = glfw.create_window(64, 64, "rv_gl33_splat_bench", None, None)
    if not win:
        raise SystemExit("glfw.create_window (GL 3.3 core) failed")
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context(require=330)
    info = {k: ctx.info[k] for k in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}

    if a.npz:
        z = np.load(a.npz)
        pos = z["pos"].astype(np.float32); col = z["col"].astype(np.float32); sig = z["sigma"].astype(np.float32)
        src = a.npz
    else:
        pos, col, sig = synthetic(a.n)
        src = "synthetic seed 0"
    n = int(pos.shape[0])

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    corners = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32)
    vbo_c = ctx.buffer(corners.tobytes())
    inst_dtype = np.dtype([("p", np.float32, 3), ("c", np.float32, 4), ("s", np.float32)])
    inst = np.empty(n, inst_dtype)
    vbo_i = ctx.buffer(reserve=inst.nbytes, dynamic=True)
    vao = ctx.vertex_array(prog, [(vbo_c, "2f", "corner"), (vbo_i, "3f 4f 1f/i", "center", "color", "sigma_m")])
    color_tex = ctx.texture((a.w, a.h), 4, dtype="f1")
    depth_tex = ctx.texture((a.w, a.h), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[color_tex, depth_tex])
    fov = 60.0
    proj = perspective(fov, a.w / a.h, 0.05, 200.0)
    fy = (a.h / 2.0) / np.tan(np.radians(fov) / 2.0)
    prog["proj"].write(proj.T.astype("f4").tobytes())
    prog["fy_px"].value = float(fy)
    prog["viewport"].value = (float(a.w), float(a.h))
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.disable(moderngl.DEPTH_TEST)

    sort_ms, upload_ms, draw_ms, total_ms = [], [], [], []
    total_frames = a.warmup + a.frames
    for i in range(total_frames):
        ang = 2 * np.pi * i / total_frames
        eye = (8 * np.sin(ang), -8 + 4 * np.cos(ang), 4.0)
        view, fwd = look_at(eye, (0.0, 12.0, 0.0))
        t0 = time.perf_counter()
        d = (pos - np.asarray(eye, np.float32)) @ fwd.astype(np.float32)
        order = sort_splats(d, 0.05, 200.0)  # same near/far as proj above
        inst["p"] = pos[order]; inst["c"] = col[order]; inst["s"] = sig[order]
        t1 = time.perf_counter()
        vbo_i.write(inst.tobytes())
        prog["view"].write(view.T.astype("f4").tobytes())
        t2 = time.perf_counter()
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        ctx.finish()
        t3 = time.perf_counter()
        if i >= a.warmup:
            sort_ms.append((t1 - t0) * 1e3); upload_ms.append((t2 - t1) * 1e3)
            draw_ms.append((t3 - t2) * 1e3); total_ms.append((t3 - t0) * 1e3)

    rgba = np.frombuffer(color_tex.read(), np.uint8).reshape(a.h, a.w, 4)
    dep = np.frombuffer(depth_tex.read(), np.float32).reshape(a.h, a.w, 4)
    acc = dep[..., 3]
    zbuf = np.where(acc > 1e-3, dep[..., 0] / np.maximum(acc, 1e-6), 0.0)
    gpu_mb = gpu_dedicated_mb(os.getpid())
    nv_during = nvsmi_used_mb()
    rss_mb = proc.memory_info().rss / 2**20
    if a.png:
        import zlib, struct
        img = rgba[::-1, :, :3].copy()
        raw = b"".join(b"\x00" + img[y].tobytes() for y in range(a.h))
        def chunk(t, dta):
            return struct.pack(">I", len(dta)) + t + dta + struct.pack(">I", zlib.crc32(t + dta) & 0xFFFFFFFF)
        open(a.png, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", a.w, a.h, 8, 2, 0, 0, 0))
                                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))

    def pct(v, p):
        return round(float(np.percentile(v, p)), 3)
    out = {
        "gl": info, "splats": n, "source": src, "res": [a.w, a.h], "frames": a.frames,
        "sort_ms_p50": pct(sort_ms, 50), "sort_ms_p95": pct(sort_ms, 95),
        "upload_ms_p95": pct(upload_ms, 95),
        "draw_finish_ms_p50": pct(draw_ms, 50), "draw_finish_ms_p95": pct(draw_ms, 95),
        "sort_plus_render_ms_p50": pct(total_ms, 50), "sort_plus_render_ms_p95": pct(total_ms, 95),
        "sort_plus_render_ms_max": round(max(total_ms), 3),
        "fps_at_p50": round(1000.0 / pct(total_ms, 50), 1),
        "process_gpu_dedicated_mb": gpu_mb,
        "nvidia_smi_used_mb_before": nv_before, "nvidia_smi_used_mb_during": nv_during,
        "rss_mb_before_gl": round(rss_before, 1), "rss_mb": round(rss_mb, 1),
        "depth_buffer_covered_px": int((acc > 1e-3).sum()),
        "depth_buffer_finite": bool(np.isfinite(zbuf).all()),
        "depth_buffer_median_m": round(float(np.median(zbuf[acc > 1e-3])), 3) if (acc > 1e-3).any() else None,
        "gate_p95_lt_33ms": pct(total_ms, 95) < 33.0,
        "gate_gpu_le_300mb": (gpu_mb <= 300) if isinstance(gpu_mb, float) else "unmeasured",
    }
    print(json.dumps(out, indent=1))
    vao.release(); vbo_i.release(); vbo_c.release(); fbo.release(); color_tex.release(); depth_tex.release(); prog.release()
    ctx.release()
    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
