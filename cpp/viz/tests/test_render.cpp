// ctest: render fixed scenes headless and check pixels / schema handling.
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "context.hpp"
#include "image_io.hpp"
#include "renderer.hpp"
#include "scene.hpp"
#include "udp.hpp"
#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>
#if GVD_VIZ_HAVE_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

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

struct Counts { long white = 0, yellow = 0, purple = 0, red = 0, green = 0, blue = 0, lit = 0,
                orange = 0, pink = 0, magenta = 0; };

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
      else if (r > 150 && b > 150 && g > 100) ++c.pink;
      else if (r > 120 && b > 120 && g < 0.6 * std::min(r, b)) {
        ++c.purple;  // also catches the purple-lane mutant and magenta trucks
        if (g <= 100) ++c.magenta;
      } else if (r > 170 && g >= static_cast<int>(0.45 * r) && g <= 170 && b < 0.55 * g && r - g >= 110) {
        ++c.orange;
      } else if (r > 150 && g < 0.45 * r && b < 0.45 * r) ++c.red;
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

// Independent dots along a segment. The viz must not invent extra samples.
static std::string dots_seg(float x0, float y0, float x1, float y1, const char* cls, float step,
                            int id0, float prob = 1.f, bool observed = true) {
  std::ostringstream o;
  const float dx = x1 - x0, dy = y1 - y0;
  const float len = std::sqrt(dx * dx + dy * dy);
  const int n = std::max(1, static_cast<int>(std::lround(len / step)));
  for (int i = 0; i <= n; ++i) {
    const float t = (n == 0) ? 0.f : static_cast<float>(i) / static_cast<float>(n);
    if (i) o << ',';
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "{\"id\":%d,\"x\":%.3f,\"y\":%.3f,\"cls\":\"%s\",\"prob\":%.3f,\"observed\":%s}",
                  id0 + i, x0 + t * dx, y0 + t * dy, cls, prob, observed ? "true" : "false");
    o << buf;
  }
  return o.str();
}

static std::string lanes(const char* a, const char* b) {
  return std::string("\"dots\":[") + dots_seg(-1.8f, -12.f, -1.8f, 90.f, a, 1.2f, 1) + "," +
         dots_seg(1.8f, -12.f, 1.8f, 90.f, b, 1.2f, 200) + "]";
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
  // Colour / schema pixel tests use a close framing so bead counts stay stable.
  // The live default chase is pinned separately in test 18.
  o.lens = Lens{{0.f, -21.f, 6.6f}, {0.f, 14.3f, 0.f}, 30.f};
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
    Image im2 = shot(must(head(std::string("\"dots\":[") + dots_seg(0.f, -12.f, 0.f, 90.f, "white", 1.2f, 1) + "]")), o);
    Counts c2 = classify(im2);
    CHECK(c2.yellow == 0 && c2.white > 80, "no paint field: yellow=%ld white=%ld", c2.yellow, c2.white);
  }

  // 5. road edges are red, and only edges are red.
  {
    Image im = shot(must(head(std::string("\"dots\":[") + dots_seg(-6.f, -12.f, -6.f, 90.f, "curb", 0.8f, 1) + "," +
                                  dots_seg(6.f, -12.f, 6.f, 90.f, "curb", 0.8f, 300) + "]")), o);
    Counts c = classify(im);
    CHECK(c.red > 150, "edges: only %ld red px", c.red);
    CHECK(c.yellow == 0 && c.purple == 0 && c.green == 0, "edges drew y=%ld p=%ld g=%ld", c.yellow, c.purple,
          c.green);
  }

  // 6. lead box green; another car blue and not green.
  {
    Image im = shot(must(head("\"lead_id\":1,\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    Counts c = classify(im);
    CHECK(c.green > 80, "lead box: only %ld green px", c.green);
    Image im2 = shot(must(head("\"lead_id\":7,\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    Counts c2 = classify(im2);
    CHECK(c2.blue > 80, "other car: only %ld blue px", c2.blue);
    CHECK(c2.green == 0, "non-lead car drew %ld green px", c2.green);
    Image im3 = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5,\"lead\":true}]")), o);
    CHECK(classify(im3).green > 80, "track lead flag ignored");
  }

  // 7. GT / unknown keys never change a pixel.
  {
    std::string base = lanes("white", "white") + ",\"tracks\":[{\"id\":3,\"cls\":\"truck\",\"x\":3.6,\"y\":20,\"yaw\":1.57,\"l\":7.6,\"w\":2.5,\"h\":3.0}]";
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
    Image im = shot(must(head(std::string("\"dots\":[") + dots_seg(0.f, 2.f, 0.f, 40.f, "path", 2.0f, 1) + "]")), o);
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
                       ",\"edges\":[{\"pts\":[[-6,-12],[-6,90]]}],\"tracks\":[{\"id\":1,\"x\":0,\"y\":12,\"l\":4.5,\"w\":1.85,\"h\":1.5}],"
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


  // 12. per-class box colours (red is edges/curbs only).
  {
    Image parked = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"parked\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":4.5,\"w\":1.9,\"h\":1.5}]")), o);
    Counts cp = classify(parked);
    // Uniform box strokes no longer add at vertices, so the orange bucket is
    // smaller than the old 12-capsule additive draw; colour identity is enough.
    CHECK(cp.orange > 40, "parked: only %ld orange px", cp.orange);
    CHECK(cp.red == 0, "parked drew %ld red px (red is edges/curbs only)", cp.red);
    Image obst = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"obstacle\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":4.4,\"w\":1.9,\"h\":1.6}]")), o);
    Counts co = classify(obst);
    CHECK(co.orange > 40, "obstacle: only %ld orange px", co.orange);
    CHECK(co.red == 0, "obstacle drew %ld red px (red is edges/curbs only)", co.red);
    // Cone's class prior is 0.4x0.4 m, too small for the class-colour bucket;
    // give it the same box size as a parked car so this asserts colour, not size.
    Image cone = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"cone\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":4.5,\"w\":1.9,\"h\":1.5}]")), o);
    Counts cn = classify(cone);
    CHECK(cn.red == 0, "cone drew %ld red px", cn.red);
    CHECK(cn.orange > 40, "cone: only %ld orange px", cn.orange);
    Image car = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    Counts cv = classify(car);
    CHECK(cv.blue > 80, "car: only %ld blue px", cv.blue);
    CHECK(cv.red == 0, "car drew %ld red px", cv.red);
    Image ped = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"pedestrian\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":0.7,\"w\":0.7,\"h\":1.7}]")), o);
    CHECK(classify(ped).pink > 80, "ped: only %ld pink px", classify(ped).pink);
    Image bike = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"bike\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":1.8,\"w\":0.6,\"h\":1.5}]")), o);
    Counts cb = classify(bike);
    CHECK(cb.yellow > 80, "bike: only %ld yellow px", cb.yellow);
    CHECK(cb.green == 0, "bike drew %ld green px", cb.green);
    Image cross = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":12,\"yaw\":0,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    Counts cc = classify(cross);
    CHECK(cc.yellow > 80, "cross traffic: only %ld yellow px", cc.yellow);
    CHECK(cc.green == 0, "cross traffic drew %ld green (lead) px", cc.green);
    Image truck = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"truck\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":7.6,\"w\":2.5,\"h\":3.0}]")), o);
    Counts ct = classify(truck);
    CHECK(ct.magenta > 80, "truck: only %ld magenta px", ct.magenta);
    // Linear RGB of parked/obstacle must not fall in the red bucket (g >= 0.45 r).
    for (const char* cls : {"parked", "obstacle", "debris", "cone"}) {
      Track tp;
      tp.cls = cls;
      glm::vec3 oc = track_rgb(tp, -1, Palette::Neon);
      CHECK(oc.g >= 0.45f * oc.r && oc.b < 0.4f * oc.g,
            "%s rgb (%.2f,%.2f,%.2f) is not orange (g < 0.45 r would be red)", cls, oc.r, oc.g, oc.b);
    }
    // white / yellow paint constants stay themselves; red is reserved for edges.
    glm::vec3 white = lane_rgb(Paint::White), yellow = lane_rgb(Paint::Yellow), edge = edge_rgb(Palette::Neon);
    CHECK(white.r > 0.9f && white.g > 0.9f && white.b > 0.9f, "white paint rgb (%.2f,%.2f,%.2f)", white.r, white.g, white.b);
    CHECK(yellow.r > 0.9f && yellow.g > 0.7f && yellow.b < 0.3f, "yellow paint rgb (%.2f,%.2f,%.2f)", yellow.r, yellow.g, yellow.b);
    CHECK(edge.r > 0.9f && edge.g < 0.45f * edge.r && edge.b < 0.45f * edge.r, "edge rgb (%.2f,%.2f,%.2f) is not red", edge.r, edge.g, edge.b);
  }

  // 13. UDP receiver drains the queue and keeps the newest datagram.
  {
    UdpReceiver rx;
    std::string e;
    CHECK(rx.open(0, &e), "udp open failed: %s", e.c_str());
    CHECK(rx.port() > 0, "ephemeral port is 0");
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
#else
    int s = socket(AF_INET, SOCK_DGRAM, 0);
#endif
    CHECK(s >= 0, "udp sender socket failed");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<unsigned short>(rx.port()));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const char* msgs[] = {"one", "two", "three-last"};
    for (const char* m : msgs) {
#ifdef _WIN32
      int n = sendto(s, m, static_cast<int>(std::strlen(m)), 0, reinterpret_cast<sockaddr*>(&a), sizeof(a));
#else
      long n = sendto(s, m, std::strlen(m), 0, reinterpret_cast<sockaddr*>(&a), sizeof(a));
#endif
      CHECK(n > 0, "sendto failed for %s", m);
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    std::string got;
    int drained = 0;
    CHECK(rx.latest(got, &drained), "latest() got nothing");
    CHECK(got == "three-last", "latest() kept \"%s\", want last datagram", got.c_str());
    CHECK(drained == 3, "drained %d, want 3 (a 1-datagram drain would report 1)", drained);
  }

  // 14. stale logic: > 1 s since last_rx blanks the scene; 1 s or less is fresh.
  {
    auto t0 = std::chrono::steady_clock::now();
    CHECK(!scene_is_stale(t0, t0), "t0 vs t0 is stale");
    CHECK(!scene_is_stale(t0, t0 + std::chrono::milliseconds(1000)), "exactly 1 s is stale (timeout is > 1 s)");
    CHECK(scene_is_stale(t0, t0 + std::chrono::milliseconds(1001)), "1001 ms is still fresh");
    CHECK(scene_is_stale(t0, t0 + std::chrono::seconds(2)), "2 s is still fresh");
    CHECK(!scene_is_stale(t0, t0 + std::chrono::milliseconds(999)), "999 ms is stale");
  }

  // 15. internal targets stay under the 200 MB VRAM cap, including a 4K 4x request.
  {
    int w = 3840, h = 2160, msaa = 4;
    size_t uncapped = estimate_target_bytes(w, h, msaa);
    CHECK(uncapped > kMaxGlBytes, "4K 4x estimate %zu already under cap (test is vacuous)", uncapped);
    cap_render_targets(w, h, msaa);
    size_t capped = estimate_target_bytes(w, h, msaa);
    CHECK(capped <= kMaxGlBytes, "capped 4K estimate %zu over budget", capped);
    CHECK(w == 3840 && h == 2160, "4K resolution was reduced before MSAA (%dx%d msaa=%d)", w, h, msaa);
    CHECK(msaa >= 1 && msaa < 4, "4K 4x MSAA was not reduced (msaa=%d)", msaa);
    // smaller sizes at 4x already fit
    for (auto wh : {std::pair<int,int>{1280,720}, {1920,1080}, {2560,1440}}) {
      int ww = wh.first, hh = wh.second, s = 4;
      cap_render_targets(ww, hh, s);
      CHECK(ww == wh.first && hh == wh.second && s == 4, "unexpected cap at %dx%d msaa=%d", ww, hh, s);
      CHECK(estimate_target_bytes(ww, hh, s) <= kMaxGlBytes, "estimate over cap at %dx%d", ww, hh);
    }
    RenderOptions big = o;
    big.w = 3840;
    big.h = 2160;
    big.msaa = 4;
    CHECK(r.render(must(head("")), big, nullptr), "capped 4K render failed");
    CHECK(r.gl_bytes() < kMaxGlBytes, "gl_bytes %zu over 200 MB after 4K request", r.gl_bytes());
    int tw = 0, th = 0, ts = 0;
    r.target_size(&tw, &th, &ts);
    CHECK(estimate_target_bytes(tw, th, ts) <= kMaxGlBytes, "allocated %dx%d msaa=%d over cap", tw, th, ts);
  }

  // 16. strict v1 enums, integer ids, caps, path_conf clamp / default.
  {
    Scene s;
    CHECK(!parse_scene(head("\"aeb\":\"maybe\""), s), "invalid aeb accepted");
    CHECK(!parse_scene(head("\"aeb\":1"), s), "numeric aeb accepted");
    CHECK(parse_scene(head("\"aeb\":\"warn\""), s) && s.aeb == "warn", "aeb warn rejected");
    CHECK(parse_scene(head("\"aeb\":\"brake\""), s) && s.aeb == "brake", "aeb brake rejected");
    CHECK(parse_scene(head(""), s) && s.aeb == "off" && s.path_conf == 1.f, "defaults: aeb=%s path_conf=%f", s.aeb.c_str(), s.path_conf);
    CHECK(parse_scene(head("\"path_conf\":1.8"), s) && s.path_conf == 1.f, "path_conf 1.8 not clamped, got %f", s.path_conf);
    CHECK(parse_scene(head("\"path_conf\":-0.2"), s) && s.path_conf == 0.f, "path_conf -0.2 not clamped, got %f", s.path_conf);
    CHECK(parse_scene(head("\"path_conf\":0.9"), s) && std::fabs(s.path_conf - 0.9f) < 1e-5, "path_conf 0.9 got %f", s.path_conf);
    CHECK(parse_scene(head("\"lanes\":[{\"pts\":[[0,-12],[0,90]],\"paint\":\"not_yellow\"}]"), s) && s.lanes.empty(),
          "substring paint not_yellow was accepted");
    CHECK(parse_scene(head("\"lanes\":[{\"pts\":[[0,-12],[0,90]],\"kind\":\"guessed\"}]"), s) && s.lanes.empty(),
          "invalid kind accepted as detected");
    CHECK(parse_scene(head("\"tracks\":[{\"id\":1.9,\"cls\":\"vehicle\",\"x\":0,\"y\":12}]"), s) && s.tracks.empty(),
          "float id accepted (truncated)");
    CHECK(parse_scene(head("\"lead_id\":4.7,\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":12}]"), s) && s.lead_id == -1,
          "float lead_id accepted, got %d", s.lead_id);
    CHECK(parse_scene(head("\"tracks\":[{\"id\":1,\"cls\":\"spaceship\",\"x\":0,\"y\":12}]"), s) && s.tracks.empty(),
          "unknown cls accepted");
    CHECK(parse_scene(head("\"signs\":[{\"cls\":\"traffic_light\",\"x\":0,\"y\":30,\"state\":\"maybe\"}]"), s) && s.signs.empty(),
          "invalid sign state accepted");
    CHECK(parse_scene(head("\"signs\":[{\"cls\":\"billboard\",\"x\":0,\"y\":30}]"), s) && s.signs.empty(),
          "unknown sign cls accepted");
    std::string many = "\"tracks\":[";
    for (int i = 0; i < 65; ++i) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%s{\"id\":%d,\"cls\":\"vehicle\",\"x\":0,\"y\":%d}", i ? "," : "", i + 1, 8 + i);
      many += buf;
    }
    many += "]";
    CHECK(parse_scene(head(many), s) && s.tracks.size() == 64, "track cap: got %zu", s.tracks.size());
    CHECK(s.tracks.front().id == 1 && s.tracks.back().id == 64, "track cap dropped the wrong end");
    std::string lanes = "\"lanes\":[";
    for (int i = 0; i < 20; ++i) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%s{\"pts\":[[%d,-12],[%d,90]],\"paint\":\"white\"}", i ? "," : "", i, i);
      lanes += buf;
    }
    lanes += "]";
    CHECK(parse_scene(head(lanes), s) && s.lanes.size() == 16, "lane cap: got %zu", s.lanes.size());
    // aliases are ignored, not accepted as the documented keys
    CHECK(parse_scene(head("\"lanes\":[{\"pts\":[[0,-12],[0,90]],\"paint_color\":\"yellow\"}]"), s)
              && s.lanes.size() == 1 && s.lanes[0].paint == Paint::White,
          "paint_color alias was treated as yellow");
    CHECK(parse_scene(head("\"lanes\":[{\"points\":[[0,-12],[0,90]],\"paint\":\"yellow\"}]"), s) && s.lanes.empty(),
          "points alias accepted as pts");
    CHECK(parse_scene(head("\"tracks\":[{\"id\":1,\"class\":\"parked\",\"x\":0,\"y\":12}]"), s)
              && s.tracks.size() == 1 && s.tracks[0].cls == "vehicle",
          "class alias accepted as cls, got %s", s.tracks.empty() ? "(none)" : s.tracks[0].cls.c_str());
    std::string long_path = "\"path\":{\"pts\":[";
    for (int i = 0; i < 200; ++i) {
      char buf[48];
      std::snprintf(buf, sizeof(buf), "%s[0,%d]", i ? "," : "", i);
      long_path += buf;
    }
    long_path += "]}";
    CHECK(parse_scene(head(long_path), s) && s.path.size() == 120 && s.path.front().y == 0.f && s.path.back().y == 199.f,
          "point cap: size=%zu first=%f last=%f", s.path.size(), s.path.empty() ? -1.f : s.path.front().y,
          s.path.empty() ? -1.f : s.path.back().y);
  }

  // 17. hidden GLFW window: GLFW_VISIBLE is false when requested (hint set after init).
  {
#if GVD_VIZ_HAVE_GLFW
    GlContext hidden;
    std::string e;
    if (hidden.init_window(64, 64, "gvd_viz hint test", false, &e, /*visible=*/false)) {
      CHECK(hidden.window() != nullptr, "hidden window is null");
      CHECK(glfwGetWindowAttrib(hidden.window(), GLFW_VISIBLE) == GLFW_FALSE,
            "hidden window is visible (GLFW_VISIBLE hint lost)");
    } else {
      std::printf("SKIP glfw hidden-window attrib (no display?): %s\n", e.c_str());
    }
#else
    std::printf("SKIP glfw hidden-window attrib (built without GLFW)\n");
#endif
  }

  // 18. default Lens is the named chase: pulled back, 40 deg vfov.
  {
    Lens d;
    CHECK(d.eye.x == 0.f && d.eye.y == -32.f && d.eye.z == 10.f,
          "default eye (%.1f,%.1f,%.1f)", d.eye.x, d.eye.y, d.eye.z);
    CHECK(d.look.x == 0.f && d.look.y == 40.f && d.look.z == 0.f,
          "default look (%.1f,%.1f,%.1f)", d.look.x, d.look.y, d.look.z);
    CHECK(d.vfov_deg == 40.f, "default vfov %.1f", d.vfov_deg);
    Lens c;
    CHECK(lens_named("chase", c), "named chase missing");
    CHECK(c.eye == d.eye && c.look == d.look && c.vfov_deg == d.vfov_deg,
          "named chase differs from the default Lens");
    RenderOptions ro;
    CHECK(ro.lens.eye == d.eye && ro.lens.look == d.look && ro.lens.vfov_deg == 40.f,
          "RenderOptions default lens is not chase");
  }


  // 19. box wireframe: corner brightness is within ~5% of mid-edge (no additive
  // glow pile-up where 12 capsules would overlap at a vertex).
  {
    Scene s = must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]"));
    Image im = shot(s, o);
    const Track& t = s.tracks[0];
    glm::vec3 d = t.l > 0.f ? glm::vec3(t.l, t.w, t.h) : glm::vec3(4.5f, 1.85f, 1.5f);
    // same vertex construction as add_box
    float cy = std::cos(t.yaw), sy = std::sin(t.yaw);
    glm::vec2 fwd(cy, sy), side(-sy, cy), ctr(t.x, t.y);
    glm::vec3 v[8];
    for (int i = 0; i < 8; ++i) {
      float fl = (i & 1) ? 0.5f : -0.5f, sl = (i & 2) ? 0.5f : -0.5f;
      glm::vec2 p = ctr + fwd * (fl * d.x) + side * (sl * d.y);
      v[i] = glm::vec3(p, (i & 4) ? d.z : 0.03f);
    }
    const float aspect = float(o.w) / float(o.h);
    glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
    glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, 0.3f, 600.f);
    auto to_img = [&](glm::vec3 w) -> glm::vec2 {
      glm::vec4 clip = proj * view * glm::vec4(w, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      float sx = (ndc.x * 0.5f + 0.5f) * float(o.w);
      float sy = (1.f - (ndc.y * 0.5f + 0.5f)) * float(o.h);
      return {sx, sy};
    };
    auto luma_at = [&](int x, int y) -> float {
      if (x < 0 || y < 0 || x >= im.w || y >= im.h) return 0.f;
      const uint8_t* p = &im.rgba[(static_cast<size_t>(y) * im.w + x) * 4];
      return (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f;
    };
    auto max_win = [&](glm::vec2 px, int rad) -> float {
      float m = 0.f;
      int cx = int(std::lround(px.x)), cy = int(std::lround(px.y));
      for (int dy = -rad; dy <= rad; ++dy)
        for (int dx = -rad; dx <= rad; ++dx) m = std::max(m, luma_at(cx + dx, cy + dy));
      return m;
    };
    // closer face (smaller y, toward the camera): verts 0,2,4,6
    const int face[4] = {0, 2, 4, 6};
    const int edges[4][2] = {{0, 2}, {4, 6}, {0, 4}, {2, 6}};
    float corner = 0.f, edge = 0.f;
    for (int i : face) corner = std::max(corner, max_win(to_img(v[i]), 4));
    for (auto& e : edges) edge = std::max(edge, max_win(to_img(0.5f * (v[e[0]] + v[e[1]])), 4));
    // also average mid-edges so a single bright sample cannot hide a dark line
    float edge_sum = 0.f;
    for (auto& e : edges) edge_sum += max_win(to_img(0.5f * (v[e[0]] + v[e[1]])), 4);
    float edge_mean = edge_sum / 4.f;
    float ref = edge_mean > 1e-4f ? edge_mean : edge;
    float rel = ref > 1e-4f ? std::fabs(corner - ref) / ref : 1.f;
    std::printf("box stroke: corner luma %.4f mid-edge max %.4f mean %.4f rel %.3f\n",
                corner, edge, edge_mean, rel);
    CHECK(ref > 0.05f, "box wireframe too dim to measure (edge luma %.4f)", ref);
    CHECK(rel <= 0.05f, "corner luma %.4f vs mid-edge %.4f (rel %.3f, want <= 0.05)", corner, ref, rel);
  }


  // 20. ego GLB (David_Holiday Tesla Model 3) loads, under 30k tris, no badge node.
  {
    CHECK(r.ego_from_glb(), "ego mesh did not load from tesla_model3_ego.glb (path '%s')",
          r.ego_glb_path().c_str());
    CHECK(r.ego_triangle_count() > 10000, "ego GLB too small (%d tris)", r.ego_triangle_count());
    CHECK(r.ego_triangle_count() < 30000, "ego GLB too heavy (%d tris, want < 30k)", r.ego_triangle_count());
    std::printf("ego GLB: %s  tris=%d\n", r.ego_glb_path().c_str(), r.ego_triangle_count());
    const std::string& path = r.ego_glb_path();
    std::ifstream f(path, std::ios::binary);
    CHECK(f.good(), "cannot open ego GLB %s", path.c_str());
    if (f) {
      auto u32 = [](const std::vector<char>& b, size_t o) {
        uint32_t v; std::memcpy(&v, b.data() + o, 4); return v;
      };
      f.seekg(0, std::ios::end);
      std::vector<char> buf(static_cast<size_t>(f.tellg()));
      f.seekg(0);
      f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      CHECK(buf.size() > 20 && u32(buf, 0) == 0x46546C67u, "ego asset is not a GLB");
      size_t off = 12, json_off = 0, json_n = 0;
      while (off + 8 <= buf.size()) {
        uint32_t clen = u32(buf, off), typ = u32(buf, off + 4);
        if (off + 8 + clen > buf.size()) break;
        if (typ == 0x4E4F534Au) { json_off = off + 8; json_n = clen; }
        off += 8 + clen;
      }
      CHECK(json_n > 0, "ego GLB has no JSON chunk");
      nlohmann::json gltf = nlohmann::json::parse(buf.data() + json_off, buf.data() + json_off + json_n, nullptr, false);
      CHECK(!gltf.is_discarded(), "ego GLB JSON parse failed");
      auto lower = [](std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
      };
      const char* kBadge[] = {"badge", "logo", "emblem", "wordmark", "hood_t", "tesla_t", "t_badge"};
      int nodes = 0;
      bool named = false;
      if (gltf.contains("nodes")) {
        for (const auto& n : gltf["nodes"]) {
          ++nodes;
          std::string name = n.value("name", "");
          if (name == "Tesla_Model_3") named = true;
          std::string ln = lower(name);
          for (const char* k : kBadge) {
            CHECK(ln.find(k) == std::string::npos, "badge node present: %s", name.c_str());
          }
        }
      }
      CHECK(nodes >= 1, "ego GLB has no nodes");
      CHECK(named, "ego GLB missing Tesla_Model_3 node");
      std::printf("ego GLB nodes=%d (no badge)\n", nodes);
    }
  }


  // 21. dropping one independent dot from the input removes only that dot.
  {
    auto to_img = [&](float x, float y) -> glm::vec2 {
      const float aspect = float(o.w) / float(o.h);
      glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
      glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, 0.3f, 600.f);
      glm::vec4 clip = proj * view * glm::vec4(x, y, 0.04f, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      return {(ndc.x * 0.5f + 0.5f) * float(o.w), (1.f - (ndc.y * 0.5f + 0.5f)) * float(o.h)};
    };
    auto win_luma = [&](const Image& im, glm::vec2 px, int rad) -> float {
      float m = 0.f;
      int cx = int(std::lround(px.x)), cy = int(std::lround(px.y));
      for (int dy = -rad; dy <= rad; ++dy)
        for (int dx = -rad; dx <= rad; ++dx) {
          int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= im.w || y >= im.h) continue;
          const uint8_t* p = &im.rgba[(static_cast<size_t>(y) * im.w + x) * 4];
          m = std::max(m, (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f);
        }
      return m;
    };
    const char* both =
        "\"dots\":[{\"id\":1,\"x\":-3.5,\"y\":22,\"cls\":\"white\",\"prob\":1,\"observed\":true},"
        "{\"id\":2,\"x\":3.5,\"y\":22,\"cls\":\"white\",\"prob\":1,\"observed\":true}]";
    const char* one =
        "\"dots\":[{\"id\":1,\"x\":-3.5,\"y\":22,\"cls\":\"white\",\"prob\":1,\"observed\":true}]";
    Image a = shot(must(head(both)), o);
    Image b = shot(must(head(one)), o);
    glm::vec2 p1 = to_img(-3.5f, 22.f), p2 = to_img(3.5f, 22.f);
    CHECK(win_luma(a, p1, 6) > 0.15f && win_luma(a, p2, 6) > 0.15f, "both dots should be lit");
    CHECK(win_luma(b, p1, 6) > 0.15f, "kept dot disappeared");
    CHECK(win_luma(b, p2, 6) < 0.05f, "dropped dot is still lit (luma %.3f)", win_luma(b, p2, 6));
    std::printf("drop one dot: kept %.3f dropped %.3f (was %.3f)\n", win_luma(b, p1, 6), win_luma(b, p2, 6),
                win_luma(a, p2, 6));
  }

  // 22. per-dot alpha follows that dot's prob.
  {
    auto to_img = [&](float x, float y) -> glm::vec2 {
      const float aspect = float(o.w) / float(o.h);
      glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
      glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, 0.3f, 600.f);
      glm::vec4 clip = proj * view * glm::vec4(x, y, 0.04f, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      return {(ndc.x * 0.5f + 0.5f) * float(o.w), (1.f - (ndc.y * 0.5f + 0.5f)) * float(o.h)};
    };
    auto win_luma = [&](const Image& im, glm::vec2 px, int rad) -> float {
      float m = 0.f;
      int cx = int(std::lround(px.x)), cy = int(std::lround(px.y));
      for (int dy = -rad; dy <= rad; ++dy)
        for (int dx = -rad; dx <= rad; ++dx) {
          int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= im.w || y >= im.h) continue;
          const uint8_t* p = &im.rgba[(static_cast<size_t>(y) * im.w + x) * 4];
          m = std::max(m, (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f);
        }
      return m;
    };
    Image im = shot(must(head("\"dots\":[{\"id\":1,\"x\":-3.5,\"y\":22,\"cls\":\"white\",\"prob\":1.0,\"observed\":true},"
                              "{\"id\":2,\"x\":3.5,\"y\":22,\"cls\":\"white\",\"prob\":0.25,\"observed\":true}]")),
                    o);
    float hi = win_luma(im, to_img(-3.5f, 22.f), 6);
    float lo = win_luma(im, to_img(3.5f, 22.f), 6);
    std::printf("alpha follows prob: p=1 luma %.3f  p=0.25 luma %.3f\n", hi, lo);
    CHECK(hi > 0.20f, "prob=1 dot too dim (%.3f)", hi);
    CHECK(lo > 0.02f, "prob=0.25 dot invisible (%.3f)", lo);
    CHECK(hi > lo * 1.6f, "prob=1 (%.3f) should be clearly brighter than prob=0.25 (%.3f)", hi, lo);
  }

  // 23. a new dot fades in over N frames rather than popping on.
  {
    auto win_luma = [&](const Image& im, int rad) -> float {
      const float aspect = float(o.w) / float(o.h);
      glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
      glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, 0.3f, 600.f);
      glm::vec4 clip = proj * view * glm::vec4(0.f, 22.f, 0.04f, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      int cx = int(std::lround((ndc.x * 0.5f + 0.5f) * float(o.w)));
      int cy = int(std::lround((1.f - (ndc.y * 0.5f + 0.5f)) * float(o.h)));
      float m = 0.f;
      for (int dy = -rad; dy <= rad; ++dy)
        for (int dx = -rad; dx <= rad; ++dx) {
          int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= im.w || y >= im.h) continue;
          const uint8_t* p = &im.rgba[(static_cast<size_t>(y) * im.w + x) * 4];
          m = std::max(m, (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f);
        }
      return m;
    };
    // Seed history at t=0 with an empty scene, then introduce the dot.
    shot(must("{\"v\":1,\"type\":\"gvd_scene\",\"seq\":1,\"t\":0}"), o);
    const char* dot = "{\"v\":1,\"type\":\"gvd_scene\",\"seq\":2,\"t\":%.3f,"
                      "\"dots\":[{\"id\":7,\"x\":0,\"y\":22,\"cls\":\"white\",\"prob\":1,\"observed\":true}]}";
    char buf[256];
    std::snprintf(buf, sizeof(buf), dot, 0.04);
    float l1 = win_luma(shot(must(buf), o), 6);
    std::snprintf(buf, sizeof(buf), dot, 0.08);
    float l2 = win_luma(shot(must(buf), o), 6);
    std::snprintf(buf, sizeof(buf), dot, 0.12);
    float l3 = win_luma(shot(must(buf), o), 6);
    std::snprintf(buf, sizeof(buf), dot, 0.28);
    float lN = win_luma(shot(must(buf), o), 6);
    std::printf("fade in: t=0.04 %.3f  0.08 %.3f  0.12 %.3f  0.28 %.3f\n", l1, l2, l3, lN);
    CHECK(l1 > 0.01f, "first fade frame is black (%.3f)", l1);
    CHECK(lN > l1 * 1.4f, "dot popped instead of fading (%.3f then %.3f)", l1, lN);
    CHECK(l2 >= l1 * 0.9f && l3 >= l2 * 0.9f, "fade is not increasing (%.3f %.3f %.3f)", l1, l2, l3);
  }

  // 24. a world-fixed dot shifts on screen by the right amount when ego moves.
  {
    auto to_img = [&](float x, float y) -> glm::vec2 {
      const float aspect = float(o.w) / float(o.h);
      glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
      glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, 0.3f, 600.f);
      glm::vec4 clip = proj * view * glm::vec4(x, y, 0.04f, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      return {(ndc.x * 0.5f + 0.5f) * float(o.w), (1.f - (ndc.y * 0.5f + 0.5f)) * float(o.h)};
    };
    auto peak = [&](const Image& im, glm::vec2 guess, int rad, int* ox, int* oy) -> float {
      int cx = int(std::lround(guess.x)), cy = int(std::lround(guess.y));
      float m = 0.f;
      int bx = cx, by = cy;
      for (int dy = -rad; dy <= rad; ++dy)
        for (int dx = -rad; dx <= rad; ++dx) {
          int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= im.w || y >= im.h) continue;
          const uint8_t* p = &im.rgba[(static_cast<size_t>(y) * im.w + x) * 4];
          float L = (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f;
          if (L > m) { m = L; bx = x; by = y; }
        }
      if (ox) *ox = bx;
      if (oy) *oy = by;
      return m;
    };
    // World yaw π/2 = facing +Y. Dot at world (0, 28). Ego at y=0 then y=8.
    Image a = shot(must(head("\"ego\":{\"x\":0,\"y\":0,\"yaw\":1.5708},"
                             "\"dots\":[{\"id\":1,\"x\":0,\"y\":28,\"cls\":\"white\",\"prob\":1,\"observed\":true}]")),
                   o);
    Image b = shot(must(head("\"ego\":{\"x\":0,\"y\":8,\"yaw\":1.5708},"
                             "\"dots\":[{\"id\":1,\"x\":0,\"y\":28,\"cls\":\"white\",\"prob\":1,\"observed\":true}]")),
                   o);
    glm::vec2 exp_a = to_img(0.f, 28.f), exp_b = to_img(0.f, 20.f);
    int ax, ay, bx, by;
    float la = peak(a, exp_a, 8, &ax, &ay);
    float lb = peak(b, exp_b, 8, &bx, &by);
    std::printf("world lock: ego0 peak (%d,%d) luma %.3f  ego+8 (%d,%d) luma %.3f  expected (%.1f,%.1f)->(%.1f,%.1f)\n",
                ax, ay, la, bx, by, lb, exp_a.x, exp_a.y, exp_b.x, exp_b.y);
    CHECK(la > 0.15f && lb > 0.15f, "world-fixed dot not visible (%.3f / %.3f)", la, lb);
    CHECK(std::fabs(float(ax) - exp_a.x) <= 3.f && std::fabs(float(ay) - exp_a.y) <= 3.f,
          "dot at ego 0 is at (%d,%d) want (%.1f,%.1f)", ax, ay, exp_a.x, exp_a.y);
    CHECK(std::fabs(float(bx) - exp_b.x) <= 3.f && std::fabs(float(by) - exp_b.y) <= 3.f,
          "dot after ego move is at (%d,%d) want (%.1f,%.1f)", bx, by, exp_b.x, exp_b.y);
  }

  // 25. different object size and yaw from the model output render differently.
  {
    Image mid = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    Image big = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":9.0,\"w\":2.5,\"h\":3.0}]")), o);
    Image yaw = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":0.0,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    CHECK(mid.rgba != big.rgba, "same pixels for 4.5 m and 9 m boxes");
    CHECK(mid.rgba != yaw.rgba, "same pixels for yaw=pi/2 and yaw=0");
    Counts cm = classify(mid), cb = classify(big);
    CHECK(cb.lit > cm.lit + 40, "larger box is not bigger on screen (%ld vs %ld lit)", cb.lit, cm.lit);
    std::printf("size/yaw: mid lit %ld  big lit %ld  yaw0 lit %ld\n", cm.lit, cb.lit, classify(yaw).lit);
  }

  // 26. a 2-vertex lane is two dots, not a resampled bead chain.
  {
    Image im = shot(must(head("\"lanes\":[{\"pts\":[[-1.8,-12],[-1.8,90]],\"paint\":\"white\"}]")), o);
    Counts c = classify(im);
    std::printf("no resample: 2-vertex lane white=%ld lit=%ld\n", c.white, c.lit);
    CHECK(c.lit > 5 && c.lit < 80, "2-vertex lane should be two beads, got %ld white / %ld lit px", c.white, c.lit);
    CHECK(c.yellow == 0 && c.purple == 0, "2-vertex lane drew y=%ld p=%ld", c.yellow, c.purple);
  }

  // 27. inferred (observed=false) dots are dimmer than observed ones.
  {
    auto win_luma = [&](const Image& im, float x, float y) -> float {
      const float aspect = float(o.w) / float(o.h);
      glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
      glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, 0.3f, 600.f);
      glm::vec4 clip = proj * view * glm::vec4(x, y, 0.04f, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      int cx = int(std::lround((ndc.x * 0.5f + 0.5f) * float(o.w)));
      int cy = int(std::lround((1.f - (ndc.y * 0.5f + 0.5f)) * float(o.h)));
      float m = 0.f;
      for (int dy = -6; dy <= 6; ++dy)
        for (int dx = -6; dx <= 6; ++dx) {
          int ix = cx + dx, iy = cy + dy;
          if (ix < 0 || iy < 0 || ix >= im.w || iy >= im.h) continue;
          const uint8_t* p = &im.rgba[(static_cast<size_t>(iy) * im.w + ix) * 4];
          m = std::max(m, (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f);
        }
      return m;
    };
    Image im = shot(must(head("\"dots\":[{\"id\":1,\"x\":-3.5,\"y\":22,\"cls\":\"white\",\"prob\":1,\"observed\":true},"
                              "{\"id\":2,\"x\":3.5,\"y\":22,\"cls\":\"white\",\"prob\":1,\"observed\":false}]")),
                    o);
    float ho = win_luma(im, -3.5f, 22.f), hi = win_luma(im, 3.5f, 22.f);
    std::printf("inferred dimmer: observed %.3f  inferred %.3f\n", ho, hi);
    CHECK(ho > hi * 1.4f, "inferred (%.3f) is not dimmer than observed (%.3f)", hi, ho);
  }

  // 28. a track with no model size is not drawn (no invented class prior).
  {
    Image none = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708}]")), o);
    Image sized = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]")), o);
    CHECK(classify(none).lit == 0, "sizeless track was drawn (%ld lit) using a fixed prior", classify(none).lit);
    CHECK(classify(sized).lit > 40, "sized track is invisible");
  }

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
  return g_fail ? 1 : 0;
}
