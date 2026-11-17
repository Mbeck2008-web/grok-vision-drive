// ctest: render fixed scenes headless and check pixels / schema handling.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "context.hpp"
#include "image_io.hpp"
#include "renderer.hpp"
#include "scene.hpp"

using namespace gvd;

static int g_fail = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    if (!(cond)) {                                        \
      ++g_fail;                                           \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);    \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

struct Counts { long white = 0, yellow = 0, purple = 0, red = 0, green = 0, blue = 0, lit = 0; };

// Hue buckets on saturated / bright pixels. Glow halos are dim and fall in
// none of them unless they are bright enough to read as that colour.
static Counts classify(const Image& im, int y0 = 0, int y1 = -1) {
  Counts c;
  if (y1 < 0) y1 = im.h;
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < im.w; ++x) {
      const uint8_t* p = &im.rgba[(static_cast<size_t>(y) * im.w + x) * 4];
      int r = p[0], g = p[1], b = p[2];
      int mx = std::max(r, std::max(g, b)), mn = std::min(r, std::min(g, b));
      if (mx > 40) ++c.lit;
      if (mn > 185 && mx - mn < 40) { ++c.white; continue; }
      if (mx < 110) continue;
      if (r > 170 && g > 120 && b < 0.55 * g && r - g < 110) ++c.yellow;
      else if (r > 120 && b > 120 && g < 0.6 * std::min(r, b)) ++c.purple;
      else if (r > 150 && g < 0.45 * r && b < 0.45 * r) ++c.red;
      else if (g > 150 && r < 0.6 * g && b < 0.6 * g) ++c.green;
      else if (b > 150 && r < 0.6 * b && g < 0.75 * b) ++c.blue;
    }
  }
  return c;
}

static std::string head(const std::string& body) {
  return std::string("{\"v\":1,\"type\":\"gvd_scene\",\"seq\":1,\"t\":0") + (body.empty() ? "" : ",") + body + "}";
}

static Scene must(const std::string& js) {
  Scene s;
  std::string err;
  bool ok = parse_scene(js, s, &err);
  CHECK(ok, "parse failed: %s", err.c_str());
  return s;
}

static const char* kStraight = "\"lanes\":[{\"pts\":[[-1.8,-12],[-1.8,90]],\"paint\":\"%s\"},{\"pts\":[[1.8,-12],[1.8,90]],\"paint\":\"%s\"}]";

static std::string lanes(const char* a, const char* b) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), kStraight, a, b);
  return buf;
}

int main() {
  GlContext ctx;
  std::string err;
  if (!ctx.init_headless(&err)) {
    std::printf("FAIL no headless GL context: %s\n", err.c_str());
    return 1;
  }
  std::printf("GL: %s\n", ctx.renderer_string().c_str());
  Renderer r;
  if (!r.init(&err)) {
    std::printf("FAIL renderer init: %s\n", err.c_str());
    return 1;
  }
  RenderOptions o;
  o.w = 640;
  o.h = 360;
  o.draw_ego = false;
  o.draw_hud = false;
  auto shot = [&](const Scene& s, const RenderOptions& opt) {
    Image im;
    bool ok = r.render(s, opt, &im);
    CHECK(ok, "render returned false");
    return im;
  };

  // 1. schema: wrong version / type / junk are rejected, not drawn.
  {
    Scene s;
    CHECK(!parse_scene("{\"v\":2,\"type\":\"gvd_scene\"}", s), "v=2 accepted");
    CHECK(!parse_scene("{\"v\":1,\"type\":\"other\"}", s), "wrong type accepted");
    CHECK(!parse_scene("{\"type\":\"gvd_scene\"}", s), "missing v accepted");
    CHECK(!parse_scene("not json", s), "junk accepted");
    CHECK(parse_scene(head(""), s), "minimal v1 rejected");
  }

  // 2. empty scene, no ego/HUD: pure black.
  {
    Image im = shot(must(head("")), o);
    Counts c = classify(im);
    CHECK(c.lit == 0, "empty scene has %ld lit pixels", c.lit);
  }

  // 3. white paint: white lanes, no yellow, no purple, no red.
  {
    Image im = shot(must(head(lanes("white", "white"))), o);
    Counts c = classify(im);
    CHECK(c.white > 200, "white lanes: only %ld white px", c.white);
    CHECK(c.yellow == 0, "white lanes drew %ld yellow px", c.yellow);
    CHECK(c.purple == 0, "white lanes drew %ld purple px", c.purple);
    CHECK(c.red == 0, "white lanes drew %ld red px", c.red);
  }

  // 4. yellow paint is yellow; still nothing purple. Missing paint = white.
  {
    Image im = shot(must(head(lanes("yellow", "white"))), o);
    Counts c = classify(im);
    CHECK(c.yellow > 150, "yellow paint: only %ld yellow px", c.yellow);
    CHECK(c.white > 100, "white line beside yellow: only %ld white px", c.white);
    CHECK(c.purple == 0, "purple px %ld", c.purple);
    Image im2 = shot(must(head("\"lanes\":[{\"pts\":[[0,-12],[0,90]]}]")), o);
    Counts c2 = classify(im2);
    CHECK(c2.yellow == 0 && c2.white > 80, "no paint field: yellow=%ld white=%ld", c2.yellow, c2.white);
  }

  // 5. road edges are red, and only edges are red.
  {
    Image im = shot(must(head("\"edges\":[{\"pts\":[[-6,-12],[-6,90]]},{\"pts\":[[6,-12],[6,90]]}]")), o);
    Counts c = classify(im);
    CHECK(c.red > 150, "edges: only %ld red px", c.red);
    CHECK(c.yellow == 0 && c.purple == 0 && c.green == 0, "edges drew y=%ld p=%ld g=%ld", c.yellow, c.purple,
          c.green);
  }

  // 6. lead box green; another car blue and not green.
  {
    Image im = shot(must(head("\"lead_id\":1,\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708}]")), o);
    Counts c = classify(im);
    CHECK(c.green > 80, "lead box: only %ld green px", c.green);
    Image im2 = shot(must(head("\"lead_id\":7,\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708}]")), o);
    Counts c2 = classify(im2);
    CHECK(c2.blue > 80, "other car: only %ld blue px", c2.blue);
    CHECK(c2.green == 0, "non-lead car drew %ld green px", c2.green);
    Image im3 = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"lead\":true}]")), o);
    CHECK(classify(im3).green > 80, "track lead flag ignored");
  }

  // 7. GT / unknown keys never change a pixel.
  {
    std::string base = lanes("white", "white") + ",\"tracks\":[{\"id\":3,\"cls\":\"truck\",\"x\":3.6,\"y\":20,\"yaw\":1.57}]";
    std::string gt = base +
                     ",\"path_world\":{\"pts\":[[1,0],[1,40]]},\"debug_gt\":true,\"gt_lanes\":[{\"pts\":[[0,0],[0,40]],\"paint\":\"yellow\"}],"
                     "\"lanes_gt\":[{\"pts\":[[3,0],[3,40]]}],\"vehicles_gt\":[{\"x\":0,\"y\":10}]";
    Image a = shot(must(head(base)), o);
    Image b = shot(must(head(gt)), o);
    CHECK(a.rgba == b.rgba, "GT keys changed the picture");
  }

  // 8. traffic light: red lamp lights red in the upper half; stop sign adds
  //    a stop bar + zebra (white/grey ground) in the lower part.
  {
    Image im = shot(must(head("\"signs\":[{\"cls\":\"traffic_light\",\"x\":0,\"y\":30,\"state\":\"red\"}]")), o);
    Counts up = classify(im, 0, o.h / 2);
    CHECK(up.red > 20, "red light: only %ld red px up top", up.red);
    Image im2 = shot(must(head(lanes("white", "white") + ",\"signs\":[{\"cls\":\"stop_sign\",\"x\":-6,\"y\":18}]")), o);
    Image im3 = shot(must(head(lanes("white", "white"))), o);
    Counts c2 = classify(im2), c3 = classify(im3);
    CHECK(c2.lit > c3.lit + 400, "stop sign: no crosswalk (%ld vs %ld lit)", c2.lit, c3.lit);
    CHECK(c2.red > 10, "stop sign octagon not red");
  }

  // 9. path is blue beads.
  {
    Image im = shot(must(head("\"path\":{\"pts\":[[0,2],[0,40]]}")), o);
    Counts c = classify(im);
    CHECK(c.blue > 60, "path: only %ld blue px", c.blue);
    CHECK(c.green == 0 && c.red == 0, "path drew g=%ld r=%ld", c.green, c.red);
  }

  // 10. ego + HUD on: something lit at the bottom centre, still no purple.
  {
    RenderOptions e = o;
    e.draw_ego = true;
    e.draw_hud = true;
    Image im = shot(must(head("\"ego\":{\"speed_mps\":13.4,\"set_mps\":15,\"engaged\":true}")), e);
    Counts c = classify(im, o.h / 2, o.h);
    CHECK(c.white > 150, "ego/HUD: only %ld white px in the lower half", c.white);
    CHECK(c.purple == 0, "ego/HUD drew purple");
  }

  // 11. a full scene renders at two sizes back to back with no GL error.
  {
    std::string full = lanes("yellow", "white") +
                       ",\"edges\":[{\"pts\":[[-6,-12],[-6,90]]}],\"tracks\":[{\"id\":1,\"x\":0,\"y\":12}],"
                       "\"signs\":[{\"cls\":\"traffic_light\",\"x\":2,\"y\":40,\"state\":\"green\"}],\"path\":{\"pts\":[[0,2],[0,30]]}";
    Scene s = must(head(full));
    RenderOptions big = o;
    big.w = 1280;
    big.h = 720;
    big.draw_ego = big.draw_hud = true;
    Image a = shot(s, big);
    Image b = shot(s, o);
    CHECK(a.w == 1280 && b.w == 640, "size switch failed");
    Image c = shot(s, big);
    CHECK(a.rgba == c.rgba, "same scene, same size, different pixels after a resize");
  }

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
  return g_fail ? 1 : 0;
}
