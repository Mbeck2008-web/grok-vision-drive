// GL 3.3 core renderer for the early-FSD-beta look (docs/specs/viz_cpp.md).
#pragma once
#include <cstddef>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "scene.hpp"

namespace gvd {

// Default = named "chase": pulled back, 40 deg vfov, ego in the lower third,
// ~3 lanes and a long look-ahead visible at 16:9 (early FSD beta framing).
struct Lens {
  glm::vec3 eye{0.f, -32.f, 10.f};
  glm::vec3 look{0.f, 40.f, 0.f};
  float vfov_deg = 40.f;
};
// chase (live / offline default), low, high, top, side, dash (camera-ish, for overlay).
bool lens_named(const std::string& name, Lens& out);

enum class Palette { Neon, Parking, Dusk };
bool palette_named(const std::string& name, Palette& out);

struct RenderOptions {
  int w = 1280, h = 720;
  int msaa = 4;
  Lens lens;
  Palette palette = Palette::Neon;
  bool draw_ego = true;
  bool draw_hud = true;
  const Image* background = nullptr;  // offline overlay: drawn under the scene
};

// Hard cap on GL render-target + buffer memory (MSAA colour + depth + resolve).
constexpr size_t kMaxGlBytes = 200ull * 1024ull * 1024ull;

// Estimated bytes for the MSAA colour RB, depth RB, and resolve texture.
size_t estimate_target_bytes(int w, int h, int msaa);
// Shrink MSAA first, then resolution, so estimate + extra fits in budget.
void cap_render_targets(int& w, int& h, int& msaa, size_t extra = 0, size_t budget = kMaxGlBytes);

// Colours the renderer uses (linear 0..1). Exposed for tests.
glm::vec3 lane_rgb(Paint p);
glm::vec3 edge_rgb(Palette pal);
glm::vec3 track_rgb(const Track& t, int lead_id, Palette pal);

struct FrameStats {
  double cpu_build_ms = 0;  // CPU: build instance buffers
  double gpu_ms = -1;       // GL_TIME_ELAPSED of the previous frame (-1 = not ready)
  int capsules = 0, sprites = 0, ground_verts = 0;
};

class Renderer {
 public:
  ~Renderer();
  bool init(std::string* err);
  // Render into the internal MSAA target. If out != nullptr, read back RGBA8.
  // If blit_to_default is true, also blit to framebuffer 0 at (fb_w, fb_h).
  bool render(const Scene& s, const RenderOptions& o, Image* out, bool blit_to_default = false,
              int fb_w = 0, int fb_h = 0);
  const FrameStats& stats() const { return stats_; }
  // Bytes of GL memory this renderer allocated (targets + buffers + textures).
  size_t gl_bytes() const;
  // Internal target size actually allocated (after the 200 MB cap).
  void target_size(int* w, int* h, int* msaa) const;

 private:
  bool ensure_targets(int w, int h, int msaa);
  void upload_background(const Image* bg);
  struct Impl;
  Impl* p_ = nullptr;
  FrameStats stats_;
};

}  // namespace gvd
