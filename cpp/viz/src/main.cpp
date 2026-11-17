// gvd_viz: GVD early-FSD-beta visualizer (docs/specs/viz_cpp.md).
//   offline:  gvd_viz --offline scenes.jsonl --png out.png [--frame N] [--lens chase] [--size 1280x720]
//   bench:    gvd_viz --offline scenes.jsonl --bench 300
//   live:     gvd_viz --listen 47800        (window; needs a GLFW build)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <glad/gl.h>
#if GVD_VIZ_HAVE_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

#include "context.hpp"
#include "image_io.hpp"
#include "renderer.hpp"
#include "scene.hpp"
#include "udp.hpp"

using namespace gvd;

static bool parse3(const char* s, glm::vec3& v) {
  return std::sscanf(s, "%f,%f,%f", &v.x, &v.y, &v.z) == 3;
}

static int usage() {
  std::fprintf(stderr,
               "usage: gvd_viz (--offline FILE [--png OUT] [--png-seq PREFIX] [--frame N] [--bench N] | --listen PORT)\n"
               "  [--size WxH] [--lens chase|low|high|top|side|dash] [--eye x,y,z --look x,y,z --fov DEG]\n"
               "  [--palette neon|parking|dusk] [--background IMG] [--no-ego] [--no-hud] [--msaa N] [--info]\n");
  return 2;
}

static double median(std::vector<double> v) {
  if (v.empty()) return -1;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  std::string offline, png, png_seq, listen_port, bg_path;
  int frame = 0, bench = 0;
  bool info = false;
  RenderOptions o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    const char* v = nullptr;
    if (a == "--offline" && (v = next())) offline = v;
    else if (a == "--png" && (v = next())) png = v;
    else if (a == "--png-seq" && (v = next())) png_seq = v;
    else if (a == "--frame" && (v = next())) frame = std::atoi(v);
    else if (a == "--bench" && (v = next())) bench = std::atoi(v);
    else if (a == "--listen" && (v = next())) listen_port = v;
    else if (a == "--size" && (v = next())) {
      if (std::sscanf(v, "%dx%d", &o.w, &o.h) != 2 || o.w < 16 || o.h < 16 || o.w > 8192 || o.h > 8192) return usage();
    } else if (a == "--lens" && (v = next())) {
      if (!lens_named(v, o.lens)) return usage();
    } else if (a == "--eye" && (v = next())) {
      if (!parse3(v, o.lens.eye)) return usage();
    } else if (a == "--look" && (v = next())) {
      if (!parse3(v, o.lens.look)) return usage();
    } else if (a == "--fov" && (v = next())) o.lens.vfov_deg = static_cast<float>(std::atof(v));
    else if (a == "--palette" && (v = next())) {
      if (!palette_named(v, o.palette)) return usage();
    } else if (a == "--background" && (v = next())) bg_path = v;
    else if (a == "--msaa" && (v = next())) o.msaa = std::atoi(v);
    else if (a == "--no-ego") o.draw_ego = false;
    else if (a == "--no-hud") o.draw_hud = false;
    else if (a == "--info") info = true;
    else return usage();
  }
  if (offline.empty() == listen_port.empty() && !info) return usage();

  GlContext ctx;
  std::string err;
  bool live = !listen_port.empty();
  bool ok = live ? ctx.init_window(o.w, o.h, "GVD VIZ", true, &err) : ctx.init_headless(&err);
  if (!ok) {
    std::fprintf(stderr, "gvd_viz: no GL context: %s\n", err.c_str());
    return 3;
  }
  if (info) std::printf("renderer: %s\n", ctx.renderer_string().c_str());
  Renderer r;
  if (!r.init(&err)) {
    std::fprintf(stderr, "gvd_viz: renderer init failed: %s\n", err.c_str());
    return 4;
  }
  Image bg;
  if (!bg_path.empty()) {
    if (!read_image(bg_path, bg)) {
      std::fprintf(stderr, "gvd_viz: cannot read background %s\n", bg_path.c_str());
      return 5;
    }
    o.background = &bg;
  }
  if (info && offline.empty() && listen_port.empty()) return 0;

  if (!offline.empty()) {
    int bad = 0;
    std::vector<Scene> scenes = load_scene_file(offline, &bad);
    if (scenes.empty()) {
      std::fprintf(stderr, "gvd_viz: no valid scene in %s (%d bad lines)\n", offline.c_str(), bad);
      return 6;
    }
    if (bad) std::fprintf(stderr, "gvd_viz: skipped %d bad lines\n", bad);
    if (bench > 0) {
      std::vector<double> cpu, gpu, wall;
      for (int k = 0; k < bench + 2; ++k) {
        const Scene& s = scenes[static_cast<size_t>(k) % scenes.size()];
        auto t0 = std::chrono::steady_clock::now();
        r.render(s, o, nullptr);
        glFinish();
        auto t1 = std::chrono::steady_clock::now();
        if (k >= 2) {
          wall.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
          cpu.push_back(r.stats().cpu_build_ms);
          if (r.stats().gpu_ms >= 0) gpu.push_back(r.stats().gpu_ms);
        }
      }
      std::printf("{\"frames\": %d, \"size\": \"%dx%d\", \"msaa\": %d, \"wall_ms_median\": %.3f, "
                  "\"cpu_build_ms_median\": %.3f, \"gpu_ms_median\": %.3f, \"gl_bytes\": %zu, "
                  "\"capsules\": %d, \"renderer\": \"%s\"}\n",
                  bench, o.w, o.h, o.msaa, median(wall), median(cpu), median(gpu), r.gl_bytes(), r.stats().capsules,
                  ctx.renderer_string().c_str());
    }
    if (!png.empty()) {
      int idx = std::clamp(frame, 0, static_cast<int>(scenes.size()) - 1);
      Image img;
      if (!r.render(scenes[static_cast<size_t>(idx)], o, &img)) {
        std::fprintf(stderr, "gvd_viz: render failed\n");
        return 7;
      }
      if (!write_png(png, img)) {
        std::fprintf(stderr, "gvd_viz: cannot write %s\n", png.c_str());
        return 8;
      }
    }
    if (!png_seq.empty()) {
      for (size_t i = 0; i < scenes.size(); ++i) {
        Image img;
        if (!r.render(scenes[i], o, &img)) {
          std::fprintf(stderr, "gvd_viz: render failed at frame %zu\n", i);
          return 7;
        }
        char path[1024];
        std::snprintf(path, sizeof(path), "%s_%04zu.png", png_seq.c_str(), i);
        if (!write_png(path, img)) {
          std::fprintf(stderr, "gvd_viz: cannot write %s\n", path);
          return 8;
        }
      }
    }
    return 0;
  }

  // live: newest UDP scene each frame; the supervisor never waits on us.
  UdpReceiver rx;
  if (!rx.open(std::atoi(listen_port.c_str()), &err)) {
    std::fprintf(stderr, "gvd_viz: %s\n", err.c_str());
    return 9;
  }
  Scene cur;
  auto last_rx = std::chrono::steady_clock::now() - std::chrono::seconds(10);
  std::string msg;
  while (!ctx.should_close()) {
#if GVD_VIZ_HAVE_GLFW
    if (ctx.key_down(GLFW_KEY_ESCAPE)) break;
#endif
    if (rx.latest(msg)) {
      Scene s;
      if (parse_scene(msg, s)) {
        cur = std::move(s);
        last_rx = std::chrono::steady_clock::now();
      }
    }
    bool stale = scene_is_stale(last_rx);  // no signal for 1 s: empty road + ego
    Scene show = stale ? Scene{} : cur;
    int fw = 0, fh = 0;
    ctx.framebuffer_size(&fw, &fh);
    if (fw > 0 && fh > 0) {
      // The window may be any size; Renderer caps internal targets so
      // gl_bytes stays under 200 MB, then blits to the framebuffer.
      o.w = fw;
      o.h = fh;
      r.render(show, o, nullptr, true, fw, fh);
    }
    ctx.swap_and_poll();
  }
  return 0;
}
