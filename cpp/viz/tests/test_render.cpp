// ctest: render fixed scenes headless and check pixels / schema handling.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
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


  // 12. per-class box colours (red is edges/curbs only).
  {
    Image parked = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"parked\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
    Counts cp = classify(parked);
    CHECK(cp.orange > 80, "parked: only %ld orange px", cp.orange);
    CHECK(cp.red == 0, "parked drew %ld red px (red is edges/curbs only)", cp.red);
    Image obst = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"obstacle\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
    Counts co = classify(obst);
    CHECK(co.orange > 80, "obstacle: only %ld orange px", co.orange);
    CHECK(co.red == 0, "obstacle drew %ld red px (red is edges/curbs only)", co.red);
    // Cone's class prior is 0.4x0.4 m, too small for an 80-px colour bucket;
    // give it the same box size as a parked car so this asserts colour, not size.
    Image cone = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"cone\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":4.5,\"w\":1.9,\"h\":1.5}]")), o);
    Counts cn = classify(cone);
    CHECK(cn.red == 0, "cone drew %ld red px", cn.red);
    CHECK(cn.orange > 80, "cone: only %ld orange px", cn.orange);
    Image car = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
    Counts cv = classify(car);
    CHECK(cv.blue > 80, "car: only %ld blue px", cv.blue);
    CHECK(cv.red == 0, "car drew %ld red px", cv.red);
    Image ped = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"pedestrian\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
    CHECK(classify(ped).pink > 80, "ped: only %ld pink px", classify(ped).pink);
    Image bike = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"bike\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
    Counts cb = classify(bike);
    CHECK(cb.yellow > 80, "bike: only %ld yellow px", cb.yellow);
    CHECK(cb.green == 0, "bike drew %ld green px", cb.green);
    Image cross = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":12,\"yaw\":0}]")), o);
    Counts cc = classify(cross);
    CHECK(cc.yellow > 80, "cross traffic: only %ld yellow px", cc.yellow);
    CHECK(cc.green == 0, "cross traffic drew %ld green (lead) px", cc.green);
    Image truck = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"truck\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
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

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
  return g_fail ? 1 : 0;
}
