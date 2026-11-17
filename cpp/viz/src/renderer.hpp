// GL 3.3 core renderer for the early-FSD-beta look (docs/specs/viz_cpp.md).
#pragma once
#include <algorithm>
#include <cmath>
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

// CPU-side formula for MSAA colour RB + depth RB + resolve texture bytes.
// This is an estimate, not a measured GPU VRAM figure. `msaa` should be the
// actual sample count (GL_RENDERBUFFER_SAMPLES) when it is known.
size_t estimate_target_bytes(int w, int h, int msaa);
// Shrink MSAA first (powers of two only: 8→4→2→1, because drivers round
// 3→4 and 5–7→8), then resolution, so estimate + extra fits in budget.
void cap_render_targets(int& w, int& h, int& msaa, size_t extra = 0, size_t budget = kMaxGlBytes);
// Next lower power-of-two MSAA count (8→4→2→1; 1 and 0 stay as-is).
int msaa_step_down(int msaa);
// Power of two the driver would use for this request (3→4, 5–7→8; 0/1 stay).
int msaa_request_pow2(int msaa);

// Box wireframe thickness: 0.055 m in world space, floored at 1.9 px.
// Pixel half-width is max(world_m * focal_px / depth, floor_px). N2 (floor
// only, no distance scaling) is killed by the unit test of this helper and
// by a close-up render that expects a stroke thicker than the floor.
constexpr float kBoxEdgeWorldM = 0.055f;
constexpr float kBoxEdgeFloorPx = 1.9f;
inline float box_edge_pixel_radius(float world_m, float floor_px, float focal_px, float view_z) {
  float depth = std::max(1e-4f, -view_z);
  return std::max(floor_px, world_m * focal_px / depth);
}

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
  // CPU-side estimate of GL memory (targets + buffers + textures), using the
  // real GL_RENDERBUFFER_SAMPLES count. Not a measured GPU VRAM figure.
  size_t gl_bytes() const;
  // Internal target size actually allocated (after the 200 MB cap).
  void target_size(int* w, int* h, int* msaa) const;
  // Ego mesh: the David_Holiday Tesla Model 3 GLB when it loaded, else the
  // procedural fallback. triangle count is the number of GL triangles.
  bool ego_from_glb() const;
  int ego_triangle_count() const;
  const std::string& ego_glb_path() const;

 private:
  bool ensure_targets(int w, int h, int msaa);
  void upload_background(const Image* bg);
  struct Impl;
  Impl* p_ = nullptr;
  FrameStats stats_;
};

}  // namespace gvd
