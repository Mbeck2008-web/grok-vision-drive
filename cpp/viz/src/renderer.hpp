// GL 3.3 core renderer for the early-FSD-beta look (docs/specs/viz_cpp.md).
#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "scene.hpp"

namespace gvd {

struct Lens {
  glm::vec3 eye{0.f, -21.f, 6.6f};
  glm::vec3 look{0.f, 14.3f, 0.f};
  float vfov_deg = 30.f;
};
// chase (live default), low, high, top, side, dash (camera-ish, for overlay).
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

 private:
  bool ensure_targets(int w, int h, int msaa);
  void upload_background(const Image* bg);
  struct Impl;
  Impl* p_ = nullptr;
  FrameStats stats_;
};

}  // namespace gvd
