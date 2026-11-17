// ctest: render fixed scenes headless and check pixels / schema handling.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
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
#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>
#if GVD_VIZ_HAVE_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif
#include <glad/gl.h>

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


// Compact SHA-256 (public-domain style) for pinning the ego GLB bytes.
static uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void sha256(const unsigned char* data, size_t len, unsigned char out[32]) {
  static const uint32_t K[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  auto process = [&](const unsigned char* blk) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)blk[i*4]<<24 | (uint32_t)blk[i*4+1]<<16 | (uint32_t)blk[i*4+2]<<8 | blk[i*4+3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
      uint32_t s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2]>>10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
      uint32_t ch = (e & f) ^ ((~e) & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  };
  unsigned char blk[64];
  size_t n = len, off = 0;
  while (n >= 64) { process(data + off); off += 64; n -= 64; }
  memcpy(blk, data + off, n);
  blk[n] = 0x80;
  if (n + 1 > 56) { memset(blk + n + 1, 0, 63 - n); process(blk); memset(blk, 0, 56); }
  else memset(blk + n + 1, 0, 55 - n);
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 0; i < 8; ++i) blk[63 - i] = (unsigned char)(bits >> (8 * i));
  process(blk);
  for (int i = 0; i < 8; ++i) {
    out[i*4] = (unsigned char)(h[i] >> 24);
    out[i*4+1] = (unsigned char)(h[i] >> 16);
    out[i*4+2] = (unsigned char)(h[i] >> 8);
    out[i*4+3] = (unsigned char)h[i];
  }
}
static std::string sha256_hex(const unsigned char* data, size_t len) {
  unsigned char d[32];
  sha256(data, len, d);
  static const char* kHex = "0123456789abcdef";
  std::string s(64, '0');
  for (int i = 0; i < 32; ++i) { s[i*2] = kHex[d[i] >> 4]; s[i*2+1] = kHex[d[i] & 0xf]; }
  return s;
}

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
  // 17. hidden GLFW window: must run *before* the EGL surfaceless context is
  // current. Creating a GLX context while EGL owns the thread fails on xvfb
  // with X_GLXMakeCurrent BadAccess and aborts the process.
  {
    CHECK(headless_glfw_visible() == false, "init_headless must pass visible=false");
#if GVD_VIZ_HAVE_GLFW
    GlContext hidden;
    std::string e;
    if (hidden.init_window(64, 64, "gvd_viz hint test", false, &e, /*visible=*/false)) {
      CHECK(hidden.window() != nullptr, "hidden window is null");
      CHECK(glfwGetWindowAttrib(hidden.window(), GLFW_VISIBLE) == GLFW_FALSE,
            "hidden window is visible (GLFW_VISIBLE hint lost)");
      if (hidden.window() && glfwGetWindowAttrib(hidden.window(), GLFW_VISIBLE) == GLFW_FALSE)
        std::printf("PASS glfw hidden-window attrib (init_window visible=false)\n");
    } else {
      std::printf("SKIP glfw hidden-window attrib (no display?): %s\n", e.c_str());
    }
#else
    std::printf("SKIP glfw hidden-window attrib (built without GLFW)\n");
#endif
  }

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
    // Uniform box strokes no longer add at vertices, so the orange bucket is
    // smaller than the old 12-capsule additive draw; colour identity is enough.
    CHECK(cp.orange > 40, "parked: only %ld orange px", cp.orange);
    CHECK(cp.red == 0, "parked drew %ld red px (red is edges/curbs only)", cp.red);
    Image obst = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"obstacle\",\"x\":0,\"y\":12,\"yaw\":1.5708}]")), o);
    Counts co = classify(obst);
    CHECK(co.orange > 40, "obstacle: only %ld orange px", co.orange);
    CHECK(co.red == 0, "obstacle drew %ld red px (red is edges/curbs only)", co.red);
    // Cone's class prior is 0.4x0.4 m, too small for the class-colour bucket;
    // give it the same box size as a parked car so this asserts colour, not size.
    Image cone = shot(must(head("\"tracks\":[{\"id\":1,\"cls\":\"cone\",\"x\":0,\"y\":12,\"yaw\":1.5708,\"l\":4.5,\"w\":1.9,\"h\":1.5}]")), o);
    Counts cn = classify(cone);
    CHECK(cn.red == 0, "cone drew %ld red px", cn.red);
    CHECK(cn.orange > 40, "cone: only %ld orange px", cn.orange);
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
    Scene cur;
    cur.speed_mps = 13.4f;
    cur.seq = 7;
    cur.engaged = true;
    Scene fresh = scene_to_show(cur, t0, t0 + std::chrono::milliseconds(999));
    CHECK(fresh.seq == 7 && std::fabs(fresh.speed_mps - 13.4f) < 1e-5f && fresh.engaged,
          "fresh scene_to_show dropped the live scene (seq=%lld speed=%f)", fresh.seq, fresh.speed_mps);
    Scene blank = scene_to_show(cur, t0, t0 + std::chrono::milliseconds(1001));
    CHECK(blank.seq == 0 && blank.speed_mps == 0.f && !blank.engaged && blank.tracks.empty(),
          "stale scene_to_show did not blank (seq=%lld speed=%f)", blank.seq, blank.speed_mps);
  }

  // 15. internal targets stay under the 200 MB VRAM cap, including a 4K 4x request.
  // Figures from estimate_target_bytes / gl_bytes are a CPU-side formula, not a
  // measured GPU VRAM figure. MSAA steps through powers of two only (8→4→2→1).
  {
    int w = 3840, h = 2160, msaa = 4;
    size_t uncapped = estimate_target_bytes(w, h, msaa);
    CHECK(uncapped > kMaxGlBytes, "4K 4x estimate %zu already under cap (test is vacuous)", uncapped);
    cap_render_targets(w, h, msaa);
    size_t capped = estimate_target_bytes(w, h, msaa);
    CHECK(capped <= kMaxGlBytes, "capped 4K estimate %zu over budget", capped);
    CHECK(w == 3840 && h == 2160, "4K resolution was reduced before MSAA (%dx%d msaa=%d)", w, h, msaa);
    CHECK(msaa == 2, "4K 4x must land exactly on MSAA 2, got msaa=%d", msaa);
    // smaller sizes at 4x already fit
    for (auto wh : {std::pair<int,int>{1280,720}, {1920,1080}, {2560,1440}}) {
      int ww = wh.first, hh = wh.second, s = 4;
      cap_render_targets(ww, hh, s);
      CHECK(ww == wh.first && hh == wh.second && s == 4, "unexpected cap at %dx%d msaa=%d", ww, hh, s);
      CHECK(estimate_target_bytes(ww, hh, s) <= kMaxGlBytes, "estimate over cap at %dx%d", ww, hh);
    }
    // background / extra term must tighten the cap (K9). 2560x1440 4x fits
    // alone (~132.7 MB) but not with an 80 MB extra; it must step to 2.
    {
      int ww = 2560, hh = 1440, s = 4;
      cap_render_targets(ww, hh, s, /*extra=*/0);
      CHECK(ww == 2560 && hh == 1440 && s == 4, "2560x1440 4x should fit with no extra");
      ww = 2560; hh = 1440; s = 4;
      const size_t extra = 80ull * 1024ull * 1024ull;
      cap_render_targets(ww, hh, s, extra);
      CHECK(estimate_target_bytes(ww, hh, s) + extra <= kMaxGlBytes,
            "extra term left estimate over budget (%dx%d msaa=%d)", ww, hh, s);
      CHECK(s == 2 && ww == 2560 && hh == 1440,
            "80 MB extra should step 2560x1440 4x to msaa 2, got %dx%d msaa=%d", ww, hh, s);
    }
    // Non-power-of-two requests must not stick at 3/5/6/7 (driver would
    // round those up). 5120x1440 at 3 would appear to fit, but 4 would not.
    {
      int ww = 5120, hh = 1440, s = 3;
      cap_render_targets(ww, hh, s);
      CHECK(s == 2 || s == 1 || s == 4 || s == 8, "non-pow2 sample count survived, msaa=%d", s);
      CHECK(s != 3, "msaa 3 request landed on 3 (driver would round up to 4, over budget)");
      CHECK(estimate_target_bytes(ww, hh, s) <= kMaxGlBytes, "5120x1440 msaa=%d over budget", s);
      ww = 5120; hh = 1440; s = 4;
      cap_render_targets(ww, hh, s);
      CHECK(ww == 5120 && hh == 1440 && s == 2, "5120x1440 4x should land on 2, got %dx%d msaa=%d", ww, hh, s);
    }
    // F4: 8→4 step must not skip to 2. 2560x1440 8x is ~250.7 MB (over
    // the cap) and 4x is ~132.7 MB (fits), so the first step must land on 4.
    {
      int ww = 2560, hh = 1440, s = 8;
      cap_render_targets(ww, hh, s);
      CHECK(ww == 2560 && hh == 1440 && s == 4,
            "2560x1440 8x must land on msaa 4, got %dx%d msaa=%d", ww, hh, s);
    }
    // K9: render() must pass the background's bytes into the cap. A 80 MB
    // background makes 2560x1440 4x miss the budget, so MSAA or size must drop.
    {
      Image bg;
      bg.w = 4096;
      bg.h = 5120;  // 4096*5120*4 = 83,886,080 bytes
      bg.rgba.assign(static_cast<size_t>(bg.w) * bg.h * 4, static_cast<uint8_t>(0));
      RenderOptions with_bg;
      with_bg.w = 2560;
      with_bg.h = 1440;
      with_bg.msaa = 4;
      with_bg.background = &bg;
      CHECK(r.render(must(head("")), with_bg, nullptr), "render with large background failed");
      int bw = 0, bh = 0, bs = 0;
      r.target_size(&bw, &bh, &bs);
      const size_t bg_bytes = static_cast<size_t>(bg.w) * bg.h * 4;
      CHECK(estimate_target_bytes(bw, bh, bs) + bg_bytes <= kMaxGlBytes,
            "background extra ignored: allocated %dx%d msaa=%d", bw, bh, bs);
      CHECK(bs < 4 || bw < 2560 || bh < 1440,
            "80 MB background did not tighten 2560x1440 4x (got %dx%d msaa=%d)", bw, bh, bs);
    }
    RenderOptions big = o;
    big.w = 3840;
    big.h = 2160;
    big.msaa = 4;
    CHECK(r.render(must(head("")), big, nullptr), "capped 4K render failed");
    CHECK(r.gl_bytes() <= kMaxGlBytes, "gl_bytes %zu over 200 MB after 4K request", r.gl_bytes());
    int tw = 0, th = 0, ts = 0;
    r.target_size(&tw, &th, &ts);
    CHECK(estimate_target_bytes(tw, th, ts) <= kMaxGlBytes, "allocated %dx%d msaa=%d over cap", tw, th, ts);
    // cap_render_targets itself lands on 4K MSAA 2 (tested above). The
    // allocated target uses the real GL_RENDERBUFFER_SAMPLES count; if the
    // driver rounded up past 2, resolution must shrink so the estimate fits.
    if (ts <= 2) {
      CHECK(tw == 3840 && th == 2160, "driver honored msaa %d but 4K was reduced to %dx%d", ts, tw, th);
    } else {
      CHECK(static_cast<long long>(tw) * th < 3840ll * 2160ll,
            "driver rounded MSAA to %d but 4K resolution was not reduced (%dx%d)", ts, tw, th);
    }
    // F2: target_size()'s ts comes from the same GL_RENDERBUFFER_SAMPLES
    // readback it is meant to check. Query the still-bound renderbuffer
    // (ensure_targets last binds the depth RB) so a "tmsaa = requested"
    // mutant cannot hide a driver round-up.
    {
      GLint rb = 0, gl_s = -1;
      glGetIntegerv(GL_RENDERBUFFER_BINDING, &rb);
      if (rb != 0) {
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &gl_s);
        if (gl_s < 1) gl_s = 1;
        CHECK(ts == gl_s, "target_size msaa=%d but GL_RENDERBUFFER_SAMPLES=%d (F2?)", ts, gl_s);
        std::printf("4K readback: target_size msaa=%d GL_RENDERBUFFER_SAMPLES=%d %dx%d gl_bytes=%zu\n",
                    ts, gl_s, tw, th, r.gl_bytes());
      } else {
        // Binding was cleared: on llvmpipe a 4K request still allocates 4
        // samples (~265 MB at full 4K) even when 2 was asked. gl_bytes must
        // use that real count, and the resolution must shrink to fit.
        const std::string rname = ctx.renderer_string();
        std::printf("4K readback: no RB bound; renderer=%s ts=%d %dx%d gl_bytes=%zu\n",
                    rname.c_str(), ts, tw, th, r.gl_bytes());
        if (rname.find("llvmpipe") != std::string::npos) {
          CHECK(ts == 4, "llvmpipe 4K tmsaa should be the real 4, got %d (F2 uses requested?)", ts);
          CHECK(static_cast<long long>(tw) * th < 3840ll * 2160ll,
                "llvmpipe 4K was not shrunk (%dx%d msaa=%d)", tw, th, ts);
          const size_t est4 = estimate_target_bytes(tw, th, 4);
          CHECK(r.gl_bytes() >= est4, "llvmpipe gl_bytes %zu < 4-sample estimate %zu", r.gl_bytes(), est4);
        }
      }
    }
    // F5: reuse guard must count the background extra on the same
    // renderer. 2560x1440 4x with no bg fits and stays 4x (~132.7 MB);
    // the same request plus an 84 MB background would be 216.6 MB if
    // the targets were reused, which is over the 200 MiB cap.
    {
      Renderer r2;
      std::string e2;
      CHECK(r2.init(&e2), "F5 renderer init failed: %s", e2.c_str());
      RenderOptions no_bg;
      no_bg.w = 2560;
      no_bg.h = 1440;
      no_bg.msaa = 4;
      no_bg.draw_ego = false;
      no_bg.draw_hud = false;
      CHECK(r2.render(must(head("")), no_bg, nullptr), "F5 first render (no bg) failed");
      int aw = 0, ah = 0, as_ = 0;
      r2.target_size(&aw, &ah, &as_);
      CHECK(aw == 2560 && ah == 1440 && as_ == 4,
            "F5 setup: 2560x1440 4x no-bg should keep 4x, got %dx%d msaa=%d", aw, ah, as_);
      CHECK(r2.gl_bytes() <= kMaxGlBytes, "F5 no-bg gl_bytes %zu over cap", r2.gl_bytes());
      Image bg;
      bg.w = 4096;
      bg.h = 5120;  // 4096*5120*4 = 83,886,080 bytes
      bg.rgba.assign(static_cast<size_t>(bg.w) * bg.h * 4, static_cast<uint8_t>(0));
      RenderOptions with_bg = no_bg;
      with_bg.background = &bg;
      CHECK(r2.render(must(head("")), with_bg, nullptr), "F5 second render (with bg) failed");
      CHECK(r2.gl_bytes() <= kMaxGlBytes, "F5 reuse ignored bg extra: gl_bytes %zu > 200 MiB", r2.gl_bytes());
      int bw = 0, bh = 0, bs = 0;
      r2.target_size(&bw, &bh, &bs);
      const size_t bg_bytes = static_cast<size_t>(bg.w) * bg.h * 4;
      CHECK(estimate_target_bytes(bw, bh, bs) + bg_bytes <= kMaxGlBytes,
            "F5 allocated %dx%d msaa=%d plus bg still over cap", bw, bh, bs);
      CHECK(bs < 4 || bw < 2560 || bh < 1440,
            "F5 84 MB bg did not tighten reused 2560x1440 4x (got %dx%d msaa=%d)", bw, bh, bs);
      std::printf("F5 reuse+bg: %dx%d msaa=%d gl_bytes=%zu\n", bw, bh, bs, r2.gl_bytes());
    }
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
    // K10: C++ edge cap
    std::string edges = "\"edges\":[";
    for (int i = 0; i < 12; ++i) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%s{\"pts\":[[%d,-12],[%d,90]]}", i ? "," : "", i, i);
      edges += buf;
    }
    edges += "]";
    CHECK(parse_scene(head(edges), s) && s.edges.size() == 8, "edge cap: got %zu", s.edges.size());
    CHECK(!s.edges.empty() && s.edges.front().pts[0].x == 0.f && s.edges.back().pts[0].x == 7.f,
          "edge cap dropped the wrong end");
    // K11: C++ sign cap
    std::string signs = "\"signs\":[";
    for (int i = 0; i < 20; ++i) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%s{\"cls\":\"traffic_light\",\"x\":%d,\"y\":30}", i ? "," : "", i);
      signs += buf;
    }
    signs += "]";
    CHECK(parse_scene(head(signs), s) && s.signs.size() == 16, "sign cap: got %zu", s.signs.size());
    CHECK(!s.signs.empty() && s.signs.front().x == 0.f && s.signs.back().x == 15.f, "sign cap dropped the wrong end");
    // K14: invalid edge kind is dropped, not accepted as detected
    CHECK(parse_scene(head("\"edges\":[{\"pts\":[[5,-12],[5,90]],\"kind\":\"guessed\"}]"), s) && s.edges.empty(),
          "invalid edge kind accepted");
    CHECK(parse_scene(head("\"edges\":[{\"pts\":[[5,-12],[5,90]],\"kind\":1}]"), s) && s.edges.empty(),
          "non-string edge kind accepted");
    // K15: integer ids that do not fit in int are rejected
    CHECK(parse_scene(head("\"tracks\":[{\"id\":2147483648,\"cls\":\"vehicle\",\"x\":0,\"y\":12}]"), s) && s.tracks.empty(),
          "out-of-range track id accepted");
    CHECK(parse_scene(head("\"lead_id\":2147483648"), s) && s.lead_id == -1,
          "out-of-range lead_id accepted, got %d", s.lead_id);
    CHECK(parse_scene(head("\"tracks\":[{\"id\":-2147483649,\"cls\":\"vehicle\",\"x\":0,\"y\":12}]"), s) && s.tracks.empty(),
          "under-range track id accepted");
    // K17: missing sign state is unknown, not red
    CHECK(parse_scene(head("\"signs\":[{\"cls\":\"traffic_light\",\"x\":0,\"y\":30}]"), s)
              && s.signs.size() == 1 && s.signs[0].state == "unknown",
          "missing sign state: got %s", s.signs.empty() ? "(none)" : s.signs[0].state.c_str());
    // K18: a non-string paint value drops the lane (it is not kept as white)
    CHECK(parse_scene(head("\"lanes\":[{\"pts\":[[0,-12],[0,90]],\"paint\":1}]"), s) && s.lanes.empty(),
          "non-string paint was kept as white");
    CHECK(parse_scene(head("\"lanes\":[{\"pts\":[[0,-12],[0,90]],\"paint\":true}]"), s) && s.lanes.empty(),
          "boolean paint was kept as white");
  }

  // 17. hidden GLFW window test runs before init_headless (see main).

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

  // 18b. box edge thickness: max(0.055 m * focal / depth, 1.9 px). Kills N2
  // (floor only, no distance scaling).
  {
    CHECK(kBoxEdgeWorldM == 0.055f && kBoxEdgeFloorPx == 1.9f,
          "thickness constants changed (world=%f floor=%f)", kBoxEdgeWorldM, kBoxEdgeFloorPx);
    const float f = 500.f;
    // Close: world term wins. Floor-only would stay at 1.9.
    float close = box_edge_pixel_radius(kBoxEdgeWorldM, kBoxEdgeFloorPx, f, -4.f);
    float world_close = kBoxEdgeWorldM * f / 4.f;  // 6.875 px
    CHECK(std::fabs(close - world_close) < 1e-4f, "close radius %f, want %f (world term)", close, world_close);
    CHECK(close > kBoxEdgeFloorPx + 2.f, "close radius %f did not exceed the 1.9 px floor (N2?)", close);
    // Far: the floor wins.
    float far = box_edge_pixel_radius(kBoxEdgeWorldM, kBoxEdgeFloorPx, f, -200.f);
    CHECK(std::fabs(far - kBoxEdgeFloorPx) < 1e-5f, "far radius %f, want the 1.9 px floor", far);
    // Render a close box and measure mid-edge half-max width; it must be
    // clearly thicker than a floor-only 1.9 px stroke.
    RenderOptions close_o = o;
    close_o.lens.eye = glm::vec3(0.f, -1.5f, 1.2f);
    close_o.lens.look = glm::vec3(0.f, 6.f, 0.6f);
    close_o.lens.vfov_deg = 40.f;
    close_o.draw_ego = false;
    close_o.draw_hud = false;
    // Closer than y=3 so the mid-edge world term is well above the 1.9 px
    // floor (at y=3, expect was ~4.06 px — zero margin vs half >= 4).
    Scene cs = must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":0,\"yaw\":1.5708,\"l\":4.5,\"w\":1.85,\"h\":1.5}]"));
    Image cim = shot(cs, close_o);
    const Track& ct = cs.tracks[0];
    glm::vec3 cd = ct.l > 0.f ? glm::vec3(ct.l, ct.w, ct.h) : glm::vec3(4.5f, 1.85f, 1.5f);
    float cy = std::cos(ct.yaw), sy = std::sin(ct.yaw);
    glm::vec2 fwd(cy, sy), side(-sy, cy), ctr(ct.x, ct.y);
    glm::vec3 v0 = glm::vec3(ctr + fwd * (0.5f * cd.x) + side * (-0.5f * cd.y), cd.z);
    glm::vec3 v1 = glm::vec3(ctr + fwd * (0.5f * cd.x) + side * (0.5f * cd.y), cd.z);
    glm::vec3 mid = 0.5f * (v0 + v1);
    const float aspect = float(close_o.w) / float(close_o.h);
    glm::mat4 view = glm::lookAt(close_o.lens.eye, close_o.lens.look, glm::vec3(0, 0, 1));
    glm::mat4 proj = glm::perspective(glm::radians(close_o.lens.vfov_deg), aspect, 0.3f, 600.f);
    auto to_img = [&](glm::vec3 w) -> glm::vec2 {
      glm::vec4 clip = proj * view * glm::vec4(w, 1.f);
      glm::vec3 ndc = glm::vec3(clip) / clip.w;
      return {(ndc.x * 0.5f + 0.5f) * float(close_o.w), (1.f - (ndc.y * 0.5f + 0.5f)) * float(close_o.h)};
    };
    glm::vec2 a = to_img(v0), b = to_img(v1), m = to_img(mid);
    glm::vec2 dir = glm::normalize(b - a);
    glm::vec2 nrm(-dir.y, dir.x);
    auto luma = [&](float x, float y) -> float {
      int ix = (int)std::round(x), iy = (int)std::round(y);
      if (ix < 0 || iy < 0 || ix >= cim.w || iy >= cim.h) return 0.f;
      const uint8_t* p = &cim.rgba[(static_cast<size_t>(iy) * cim.w + ix) * 4];
      return (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f;
    };
    float peak = 0.f;
    for (int s = -20; s <= 20; ++s) peak = std::max(peak, luma(m.x + nrm.x * s, m.y + nrm.y * s));
    CHECK(peak > 0.15f, "close-up box edge not found (peak luma %f)", peak);
    int half = 0;
    for (int s = 0; s <= 20; ++s) {
      if (luma(m.x + nrm.x * s, m.y + nrm.y * s) >= 0.5f * peak) half = s;
    }
    glm::vec4 emid = view * glm::vec4(mid, 1.f);
    float focal = proj[1][1] * 0.5f * float(close_o.h);
    float expect = box_edge_pixel_radius(kBoxEdgeWorldM, kBoxEdgeFloorPx, focal, emid.z);
    // Width is ~2*half. Floor-only is ~1.9 px plus a little AA (~2-3 px
    // half-width). The close-up mid-edge is sized so the world term sits
    // several pixels above that floor; require half to track expect with
    // margin, not a tight half>=4 against a 4 px expect.
    CHECK(expect > kBoxEdgeFloorPx + 3.f, "test setup is floor-limited (expect %f px)", expect);
    CHECK(half >= 5, "close-up half-width %d px looks like the 1.9 px floor only (N2); expect r=%.2f", half, expect);
    CHECK(static_cast<float>(half) + 2.f >= expect,
          "close-up half-width %d px is well below expect r=%.2f", half, expect);
    std::printf("box thickness: close half-width %d px (expect r=%.2f, floor=%.1f)\n", half, expect, kBoxEdgeFloorPx);
  }


  // 19. box wireframe: corner brightness is within ~5% of mid-edge (no additive
  // glow pile-up where 12 capsules would overlap at a vertex).
  {
    Scene s = must(head("\"tracks\":[{\"id\":1,\"cls\":\"vehicle\",\"x\":0,\"y\":14,\"yaw\":1.5708}]"));
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


  // 20. ego GLB (David_Holiday Tesla Model 3) loads, under 30k tris, SHA pinned,
  // and no hood-T badge triangles (node names alone are not enough: the
  // shipped file is one merged node).
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
      // Pin the exact bytes the prepare script reproduces (fa1ac245…).
      const char* kEgoSha =
          "fa1ac2450f36a48356abdf2390346e8c1190ffb986e031dc430124a59e10dae8";
      std::string got_sha = sha256_hex(reinterpret_cast<const unsigned char*>(buf.data()), buf.size());
      CHECK(got_sha == kEgoSha, "ego GLB sha256 %s, want %s", got_sha.c_str(), kEgoSha);
      size_t off = 12, json_off = 0, json_n = 0, bin_off = 0, bin_n = 0;
      while (off + 8 <= buf.size()) {
        uint32_t clen = u32(buf, off), typ = u32(buf, off + 4);
        if (off + 8 + clen > buf.size()) break;
        if (typ == 0x4E4F534Au) { json_off = off + 8; json_n = clen; }
        if (typ == 0x004E4942u) { bin_off = off + 8; bin_n = clen; }
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
      // Hood-T box from prepare_ego_glb.py, converted to the ego frame
      // with the same ymin/zmid that script uses on the kept verts:
      //   |x| < 0.045, y in (2.014792, 2.104792), z in (0.664687, 0.729687),
      //   area < 5e-4. A badge-restored mesh (26,942 tris) has 307 hits here.
      const float kBx = 0.045f, kBy0 = 2.014792f, kBy1 = 2.104792f, kBz0 = 0.664687f, kBz1 = 0.729687f, kArea = 5.0e-4f;
      int badge_tris = 0, ntris = 0;
      if (bin_n > 0 && gltf.contains("meshes") && !gltf["meshes"].empty() &&
          gltf["meshes"][0].contains("primitives") && !gltf["meshes"][0]["primitives"].empty()) {
        const auto& prim = gltf["meshes"][0]["primitives"][0];
        auto acc_of = [&](int ai) -> const nlohmann::json& { return gltf["accessors"].at(ai); };
        auto read_f32_vec3 = [&](int ai, std::vector<glm::vec3>& out) {
          const auto& acc = acc_of(ai);
          const auto& view = gltf["bufferViews"].at(acc.at("bufferView").get<int>());
          size_t o = bin_off + view.value("byteOffset", 0) + acc.value("byteOffset", 0);
          int count = acc.at("count").get<int>();
          out.resize(static_cast<size_t>(count));
          if (o + static_cast<size_t>(count) * 12 > buf.size()) { out.clear(); return; }
          std::memcpy(out.data(), buf.data() + o, static_cast<size_t>(count) * 12);
        };
        auto read_idx = [&](int ai, std::vector<uint32_t>& out) {
          const auto& acc = acc_of(ai);
          const auto& view = gltf["bufferViews"].at(acc.at("bufferView").get<int>());
          size_t o = bin_off + view.value("byteOffset", 0) + acc.value("byteOffset", 0);
          int count = acc.at("count").get<int>();
          int ctype = acc.at("componentType").get<int>();
          out.resize(static_cast<size_t>(count));
          if (ctype == 5125) {  // uint32
            if (o + static_cast<size_t>(count) * 4 > buf.size()) { out.clear(); return; }
            std::memcpy(out.data(), buf.data() + o, static_cast<size_t>(count) * 4);
          } else if (ctype == 5123) {  // uint16
            if (o + static_cast<size_t>(count) * 2 > buf.size()) { out.clear(); return; }
            for (int i = 0; i < count; ++i) {
              uint16_t v; std::memcpy(&v, buf.data() + o + static_cast<size_t>(i) * 2, 2); out[i] = v;
            }
          } else {
            out.clear();
          }
        };
        std::vector<glm::vec3> pos;
        std::vector<uint32_t> idx;
        read_f32_vec3(prim["attributes"].at("POSITION").get<int>(), pos);
        if (prim.contains("indices")) read_idx(prim.at("indices").get<int>(), idx);
        CHECK(!pos.empty() && idx.size() >= 3 && idx.size() % 3 == 0,
              "ego GLB mesh unreadable (verts=%zu idx=%zu)", pos.size(), idx.size());
        ntris = static_cast<int>(idx.size() / 3);
        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
          if (idx[i] >= pos.size() || idx[i+1] >= pos.size() || idx[i+2] >= pos.size()) continue;
          glm::vec3 p0 = pos[idx[i]], p1 = pos[idx[i+1]], p2 = pos[idx[i+2]];
          glm::vec3 c = (p0 + p1 + p2) / 3.f;
          glm::vec3 cr = glm::cross(p1 - p0, p2 - p0);
          float area = 0.5f * glm::length(cr);
          if (std::fabs(c.x) < kBx && c.y > kBy0 && c.y < kBy1 && c.z > kBz0 && c.z < kBz1 && area < kArea)
            ++badge_tris;
        }
      }
      CHECK(badge_tris == 0, "hood-T badge still present: %d tris in the badge box (%d total)", badge_tris, ntris);
      std::printf("ego GLB nodes=%d (no badge) sha256=%s badge_tris=%d\n", nodes, got_sha.c_str(), badge_tris);
    }
  }

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
  return g_fail ? 1 : 0;
}
