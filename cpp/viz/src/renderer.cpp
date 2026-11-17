#include "renderer.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace gvd {
namespace {


// ---- look constants (world metres unless noted) -------------------------
constexpr float kLaneStep = 1.2f;     // bead pitch along a lane line
constexpr float kLaneHalf = 0.10f;    // half-length of one dash
constexpr float kLaneRad = 0.10f;
constexpr float kEdgeStep = 0.8f;     // red curb beads
constexpr float kEdgeRad = 0.14f;
constexpr float kPathStep = 2.0f;
constexpr float kPathRad = 0.20f;
constexpr float kFarFadeFrom = 90.f;
constexpr float kFarCut = 160.f;
constexpr float kLightZ = 4.8f;
constexpr float kSignZ = 2.6f;

struct Prior { const char* cls; float l, w, h; };
const Prior kPriors[] = {
    {"pedestrian", 0.6f, 0.6f, 1.75f}, {"ped", 0.6f, 0.6f, 1.75f}, {"person", 0.6f, 0.6f, 1.75f},
    {"bicycle", 1.8f, 0.6f, 1.5f},     {"bike", 1.8f, 0.6f, 1.5f},  {"cyclist", 1.8f, 0.6f, 1.6f},
    {"motorcycle", 2.1f, 0.7f, 1.4f},  {"truck", 7.6f, 2.5f, 3.0f}, {"bus", 10.5f, 2.6f, 3.2f},
    {"trailer", 12.f, 2.5f, 3.4f},     {"obstacle", 4.4f, 1.9f, 1.6f}, {"parked", 4.5f, 1.9f, 1.5f},
    {"debris", 1.2f, 0.8f, 0.6f},      {"cone", 0.4f, 0.4f, 0.75f}, {"vehicle", 4.5f, 1.85f, 1.5f},
    {"car", 4.5f, 1.85f, 1.5f},
};

glm::vec3 dims_of(const Track& t) {
  glm::vec3 d(4.5f, 1.85f, 1.5f);
  for (const auto& p : kPriors) {
    if (t.cls == p.cls) { d = {p.l, p.w, p.h}; break; }
  }
  if (t.l > 0.2f) d.x = t.l;
  if (t.w > 0.2f) d.y = t.w;
  if (t.h > 0.2f) d.z = t.h;
  return d;
}

bool is_cls(const Track& t, std::initializer_list<const char*> names) {
  for (const char* n : names) if (t.cls == n) return true;
  return false;
}

// ---- GPU instance layouts -----------------------------------------------
struct Capsule {
  glm::vec3 p0, p1;
  glm::vec2 rad;    // radius_m, min_px
  glm::vec4 color;  // rgb, core intensity
  glm::vec2 glow;   // halo px, halo gain
};
struct BoxStroke {
  glm::vec3 v[8];
  glm::vec4 color;
  glm::vec2 rad;
  glm::vec2 glow;
};
// One stroke per box (min distance to all 12 edges). Set false to draw 12
// additive capsules instead — corners then glow brighter than mid-edges.
constexpr bool kBoxUniformStroke = true;
struct GVert { glm::vec3 pos; glm::vec4 color; glm::vec2 uv; float mode; };
struct Sprite { glm::vec3 pos; glm::vec2 size_m; float min_px; float type; glm::vec4 color; };
struct Hud { glm::vec4 rect; glm::vec4 color; glm::vec2 kind; };
struct MVert { glm::vec3 pos, nrm, col; float emissive; };

// ---- shaders ------------------------------------------------------------
const char* kCapsuleVS = R"(#version 330 core
layout(location=0) in vec2 a_corner;
layout(location=1) in vec3 a_p0;
layout(location=2) in vec3 a_p1;
layout(location=3) in vec2 a_rad;
layout(location=4) in vec4 a_color;
layout(location=5) in vec2 a_glow;
uniform mat4 u_view; uniform mat4 u_proj; uniform vec2 u_vp; uniform float u_near;
out vec2 v_px; flat out vec2 v_s0; flat out vec2 v_s1; flat out float v_r;
flat out vec4 v_color; flat out vec2 v_glow;
void main(){
  vec4 e0 = u_view*vec4(a_p0,1.0), e1 = u_view*vec4(a_p1,1.0);
  if (e0.z > -u_near && e1.z > -u_near) { gl_Position = vec4(2.0,2.0,2.0,1.0); return; }
  if (e0.z > -u_near) e0 = mix(e0, e1, (-u_near - e0.z)/(e1.z - e0.z));
  if (e1.z > -u_near) e1 = mix(e1, e0, (-u_near - e1.z)/(e0.z - e1.z));
  vec4 c0 = u_proj*e0, c1 = u_proj*e1;
  vec2 s0 = (c0.xy/c0.w*0.5+0.5)*u_vp, s1 = (c1.xy/c1.w*0.5+0.5)*u_vp;
  float f = u_proj[1][1]*0.5*u_vp.y;
  float r0 = a_rad.x*f/(-e0.z), r1 = a_rad.x*f/(-e1.z);
  float r = max(a_rad.y, 0.5*(r0+r1));
  float g = a_glow.x*clamp(r/2.5, 0.4, 1.0);
  float ext = r + g + 1.0;
  vec2 d = s1 - s0; float len = length(d);
  vec2 dir = len > 1e-3 ? d/len : vec2(1.0,0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  bool far_end = a_corner.x > 0.0;
  vec2 base = far_end ? s1 + dir*ext : s0 - dir*ext;
  vec2 px = base + nrm*a_corner.y*ext;
  float z = far_end ? c1.z/c1.w : c0.z/c0.w;
  gl_Position = vec4(px/u_vp*2.0-1.0, clamp(z,-1.0,1.0), 1.0);
  v_px = px; v_s0 = s0; v_s1 = s1; v_r = r; v_color = a_color; v_glow = vec2(g, a_glow.y);
})";
const char* kCapsuleFS = R"(#version 330 core
in vec2 v_px; flat in vec2 v_s0; flat in vec2 v_s1; flat in float v_r;
flat in vec4 v_color; flat in vec2 v_glow;
out vec4 o;
void main(){
  vec2 pa = v_px - v_s0, ba = v_s1 - v_s0;
  float h = dot(ba,ba) > 1e-6 ? clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0) : 0.0;
  float d = length(pa - ba*h);
  float core = clamp(v_r + 0.5 - d, 0.0, 1.0);
  float out_d = max(d - v_r, 0.0);
  float halo = v_glow.x > 0.0 ? v_glow.y*exp(-(out_d*out_d)/(0.35*v_glow.x*v_glow.x+1e-3)) : 0.0;
  // hot centre: the inner half of the bead goes a little toward white
  float hot = core*smoothstep(v_r, 0.0, d)*0.22;
  vec3 c = mix(v_color.rgb, vec3(1.0), hot);
  vec3 rgb = c*core*v_color.a + v_color.rgb*halo*(1.0-core);
  if (max(rgb.r, max(rgb.g, rgb.b)) < 0.004) discard;
  o = vec4(rgb, 0.0);  // additive (premultiplied, alpha 0)
})";

// One billboard per box; the fragment takes the min distance to all 12 edges
// so a corner is no brighter than a mid-edge (additive 12-capsule draws stack).
const char* kBoxVS = R"(#version 330 core
layout(location=0) in vec2 a_corner;
layout(location=1) in vec3 a_v0; layout(location=2) in vec3 a_v1;
layout(location=3) in vec3 a_v2; layout(location=4) in vec3 a_v3;
layout(location=5) in vec3 a_v4; layout(location=6) in vec3 a_v5;
layout(location=7) in vec3 a_v6; layout(location=8) in vec3 a_v7;
layout(location=9) in vec4 a_color; layout(location=10) in vec2 a_rad;
layout(location=11) in vec2 a_glow;
uniform mat4 u_view; uniform mat4 u_proj; uniform vec2 u_vp; uniform float u_near;
out vec2 v_px;
flat out vec3 v_w0, v_w1, v_w2, v_w3, v_w4, v_w5, v_w6, v_w7;
flat out vec4 v_color; flat out vec2 v_rad; flat out vec2 v_glow;
void main(){
  vec3 w[8] = vec3[](a_v0,a_v1,a_v2,a_v3,a_v4,a_v5,a_v6,a_v7);
  vec2 mn = vec2(1e9), mx = vec2(-1e9);
  int vis = 0;
  float f = u_proj[1][1]*0.5*u_vp.y;
  float ext = 2.0;
  for (int i = 0; i < 8; ++i) {
    vec4 e = u_view*vec4(w[i],1.0);
    if (e.z > -u_near) continue;
    vec4 c = u_proj*e;
    vec2 s = (c.xy/c.w*0.5+0.5)*u_vp;
    float r = max(a_rad.y, a_rad.x*f/(-e.z));
    float g = a_glow.x*clamp(r/2.5, 0.4, 1.0);
    ext = max(ext, r + g + 1.0);
    mn = min(mn, s); mx = max(mx, s);
    vis++;
  }
  if (vis == 0) { gl_Position = vec4(2.0,2.0,2.0,1.0); return; }
  mn -= ext; mx += ext;
  vec2 px = mix(mn, mx, a_corner*0.5+0.5);
  gl_Position = vec4(px/u_vp*2.0-1.0, 0.0, 1.0);
  v_px = px;
  v_w0=a_v0; v_w1=a_v1; v_w2=a_v2; v_w3=a_v3;
  v_w4=a_v4; v_w5=a_v5; v_w6=a_v6; v_w7=a_v7;
  v_color = a_color; v_rad = a_rad; v_glow = a_glow;
})";
const char* kBoxFS = R"(#version 330 core
in vec2 v_px;
flat in vec3 v_w0, v_w1, v_w2, v_w3, v_w4, v_w5, v_w6, v_w7;
flat in vec4 v_color; flat in vec2 v_rad; flat in vec2 v_glow;
uniform mat4 u_view; uniform mat4 u_proj; uniform vec2 u_vp; uniform float u_near;
out vec4 o;
bool project_edge(vec3 p0, vec3 p1, out vec2 s0, out vec2 s1, out float z0, out float z1, out float r, out float g){
  vec4 e0 = u_view*vec4(p0,1.0), e1 = u_view*vec4(p1,1.0);
  if (e0.z > -u_near && e1.z > -u_near) return false;
  if (e0.z > -u_near) e0 = mix(e0, e1, (-u_near - e0.z)/(e1.z - e0.z));
  if (e1.z > -u_near) e1 = mix(e1, e0, (-u_near - e1.z)/(e0.z - e1.z));
  vec4 c0 = u_proj*e0, c1 = u_proj*e1;
  s0 = (c0.xy/c0.w*0.5+0.5)*u_vp;
  s1 = (c1.xy/c1.w*0.5+0.5)*u_vp;
  z0 = c0.z/c0.w; z1 = c1.z/c1.w;
  float f = u_proj[1][1]*0.5*u_vp.y;
  float r0 = v_rad.x*f/(-e0.z), r1 = v_rad.x*f/(-e1.z);
  r = max(v_rad.y, 0.5*(r0+r1));
  g = v_glow.x*clamp(r/2.5, 0.4, 1.0);
  return true;
}
void main(){
  vec3 w[8] = vec3[](v_w0,v_w1,v_w2,v_w3,v_w4,v_w5,v_w6,v_w7);
  const int E[24] = int[](0,1, 1,3, 3,2, 2,0, 4,5, 5,7, 7,6, 6,4, 0,4, 1,5, 2,6, 3,7);
  float best_d = 1e9, best_r = 1.0, best_gpx = 0.0, best_z = 0.0;
  bool any = false;
  for (int i = 0; i < 12; ++i) {
    vec2 s0, s1; float z0, z1, r, gpx;
    if (!project_edge(w[E[i*2]], w[E[i*2+1]], s0, s1, z0, z1, r, gpx)) continue;
    vec2 pa = v_px - s0, ba = s1 - s0;
    float h = dot(ba,ba) > 1e-6 ? clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0) : 0.0;
    float d = length(pa - ba*h);
    if (!any || d < best_d) {
      best_d = d; best_r = r; best_gpx = gpx; best_z = mix(z0, z1, h);
      any = true;
    }
  }
  if (!any) discard;
  gl_FragDepth = clamp(best_z*0.5+0.5, 0.0, 1.0);
  float core = clamp(best_r + 0.5 - best_d, 0.0, 1.0);
  float out_d = max(best_d - best_r, 0.0);
  float halo = best_gpx > 0.0 ? v_glow.y*exp(-(out_d*out_d)/(0.35*best_gpx*best_gpx+1e-3)) : 0.0;
  float hot = core*smoothstep(best_r, 0.0, best_d)*0.22;
  vec3 c = mix(v_color.rgb, vec3(1.0), hot);
  vec3 rgb = c*core*v_color.a + v_color.rgb*halo*(1.0-core);
  if (max(rgb.r, max(rgb.g, rgb.b)) < 0.004) discard;
  o = vec4(rgb, 0.0);
})";


const char* kGroundVS = R"(#version 330 core
layout(location=0) in vec3 a_pos; layout(location=1) in vec4 a_color;
layout(location=2) in vec2 a_uv; layout(location=3) in float a_mode;
uniform mat4 u_vp_m;
out vec4 v_color; out vec2 v_uv; flat out float v_mode;
void main(){ gl_Position = u_vp_m*vec4(a_pos,1.0); v_color=a_color; v_uv=a_uv; v_mode=a_mode; })";
const char* kGroundFS = R"(#version 330 core
in vec4 v_color; in vec2 v_uv; flat in float v_mode; out vec4 o;
void main(){
  if (v_mode < 0.5) { o = vec4(v_color.rgb*v_color.a, v_color.a); return; }
  float r2 = dot(v_uv, v_uv);
  float k = v_color.a*exp(-3.2*r2)*(1.0 - smoothstep(0.85,1.0,sqrt(r2)));
  o = vec4(v_color.rgb*k, 0.0);
})";

const char* kSpriteVS = R"(#version 330 core
layout(location=0) in vec2 a_corner;
layout(location=1) in vec3 a_pos; layout(location=2) in vec2 a_size;
layout(location=3) in float a_minpx; layout(location=4) in float a_type; layout(location=5) in vec4 a_color;
uniform mat4 u_view; uniform mat4 u_proj; uniform vec2 u_vp;
out vec2 v_uv; flat out float v_type; flat out vec4 v_color;
void main(){
  vec4 e = u_view*vec4(a_pos,1.0);
  if (e.z > -0.5) { gl_Position = vec4(2.0,2.0,2.0,1.0); return; }
  vec4 c = u_proj*e;
  float f = u_proj[1][1]*0.5*u_vp.y;
  float hpx = max(a_minpx, a_size.y*f/(-e.z));
  vec2 half_px = vec2(hpx*a_size.x/a_size.y, hpx);
  vec2 ndc = c.xy/c.w + a_corner*half_px/u_vp*2.0;
  gl_Position = vec4(ndc, c.z/c.w, 1.0);
  v_uv = a_corner*0.5+0.5; v_type = a_type; v_color = a_color;
})";
const char* kSpriteFS = R"(#version 330 core
in vec2 v_uv; flat in float v_type; flat in vec4 v_color; out vec4 o;
float box(vec2 p, vec2 b, float r){ vec2 q = abs(p)-b+r; return length(max(q,0.0))+min(max(q.x,q.y),0.0)-r; }
void main(){
  vec2 p = v_uv*2.0-1.0;   // sprite space, x in [-1,1], y in [-1,1] (up)
  vec3 rgb = vec3(0.0); float a = 0.0;
  if (v_type < 0.5) {
    // traffic light: housing occupies the middle 46% x 70% of the quad; the
    // rest is room for the lamp glow.
    vec2 q = vec2(p.x/0.46, p.y/0.70);
    float dh = box(q, vec2(0.62, 0.95), 0.25);
    float fill = 1.0 - smoothstep(-0.02, 0.02, dh);
    float rim = 1.0 - smoothstep(0.0, 0.10, abs(dh+0.05));
    rgb = vec3(0.10)*fill + vec3(0.62)*rim*0.85; a = max(fill, rim);
    float ys[3] = float[3](0.58, 0.0, -0.58);
    for (int i=0;i<3;++i){
      float d = length(vec2(q.x, q.y-ys[i]));
      bool lit = (v_color.a > 0.5) && (abs(float(i) - v_color.a + 1.0) < 0.1);
      float lamp = 1.0 - smoothstep(0.30, 0.36, d);
      if (lit) {
        float glow = exp(-d*d*2.2)*0.9;
        rgb = mix(rgb, v_color.rgb, lamp) + v_color.rgb*glow*(1.0-lamp);
        a = max(a, max(lamp, glow));
      } else {
        rgb = mix(rgb, vec3(0.22), lamp*fill);
      }
    }
  } else if (v_type < 1.5) {
    // stop sign: red octagon, white rim
    vec2 q = abs(p);
    float d = max(max(q.x, q.y), (q.x+q.y)*0.7071) - 0.78;
    float fill = 1.0 - smoothstep(-0.02, 0.02, d);
    float rim = 1.0 - smoothstep(0.0, 0.06, abs(d+0.07));
    rgb = mix(vec3(0.85,0.08,0.08)*fill, vec3(0.95), rim*0.9); a = max(fill, rim);
    float glow = (1.0-fill)*exp(-max(d,0.0)*9.0)*0.35;
    rgb += vec3(0.9,0.1,0.1)*glow; a = max(a, glow);
  }
  if (a < 0.003) discard;
  o = vec4(rgb, a);   // premultiplied-ish over
})";

const char* kHudVS = R"(#version 330 core
layout(location=0) in vec2 a_corner;
layout(location=1) in vec4 a_rect; layout(location=2) in vec4 a_color; layout(location=3) in vec2 a_kind;
uniform vec2 u_vp;
out vec2 v_local; flat out vec4 v_rect; flat out vec4 v_color; flat out vec2 v_kind;
void main(){
  vec2 uv = a_corner*0.5+0.5;
  vec2 px = a_rect.xy + uv*a_rect.zw;
  gl_Position = vec4(px.x/u_vp.x*2.0-1.0, 1.0-px.y/u_vp.y*2.0, 0.0, 1.0);
  v_local = (uv-0.5)*a_rect.zw; v_rect=a_rect; v_color=a_color; v_kind=a_kind;
})";
const char* kHudFS = R"(#version 330 core
in vec2 v_local; flat in vec4 v_rect; flat in vec4 v_color; flat in vec2 v_kind; out vec4 o;
void main(){
  vec2 hb = v_rect.zw*0.5; float d;
  if (v_kind.x < 0.5) { float r=min(v_kind.y, min(hb.x,hb.y)); vec2 q=abs(v_local)-hb+r; d=length(max(q,0.0))+min(max(q.x,q.y),0.0)-r; }
  else if (v_kind.x < 1.5) { d = abs(length(v_local) - (min(hb.x,hb.y)-v_kind.y)) - v_kind.y*0.5; }
  else { d = length(v_local) - min(hb.x,hb.y); }
  float a = v_color.a*(1.0 - smoothstep(-0.5, 0.5, d));
  if (a < 0.003) discard;
  o = vec4(v_color.rgb*a, a);
})";

const char* kMeshVS = R"(#version 330 core
layout(location=0) in vec3 a_pos; layout(location=1) in vec3 a_nrm;
layout(location=2) in vec3 a_col; layout(location=3) in float a_emi;
uniform mat4 u_vp_m; out vec3 v_n; out vec3 v_col; out float v_emi; out vec3 v_w;
void main(){ gl_Position = u_vp_m*vec4(a_pos,1.0); v_n=a_nrm; v_col=a_col; v_emi=a_emi; v_w=a_pos; })";
const char* kMeshFS = R"(#version 330 core
in vec3 v_n; in vec3 v_col; in float v_emi; in vec3 v_w; uniform vec3 u_eye; uniform vec3 u_tint; out vec4 o;
void main(){
  vec3 n = normalize(v_n); vec3 l = normalize(vec3(-0.35,-0.45,0.82));
  vec3 vdir = normalize(u_eye - v_w);
  float lam = max(dot(n,l),0.0);
  float spec = pow(max(dot(n, normalize(l+vdir)),0.0), 24.0);
  float rim = pow(1.0 - max(dot(n, vdir), 0.0), 3.0);
  vec3 base = v_col*u_tint;
  vec3 c = base*(0.30 + 0.75*lam) + vec3(0.35)*spec + vec3(0.18)*rim;
  c = mix(c, v_col, v_emi);
  o = vec4(c, 1.0);
})";

const char* kBgVS = R"(#version 330 core
layout(location=0) in vec2 a_corner; out vec2 v_uv;
void main(){ v_uv = vec2(a_corner.x*0.5+0.5, 0.5-a_corner.y*0.5); gl_Position = vec4(a_corner,0.0,1.0); })";
const char* kBgFS = R"(#version 330 core
in vec2 v_uv; uniform sampler2D u_tex; uniform int u_mode; uniform vec3 u_top; uniform vec3 u_bot; out vec4 o;
void main(){
  if (u_mode == 1) { o = vec4(texture(u_tex, v_uv).rgb, 1.0); return; }
  o = vec4(mix(u_top, u_bot, v_uv.y), 1.0);
})";

GLuint compile(GLenum type, const char* src, std::string* err) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    if (err) *err = std::string("shader compile: ") + log;
    glDeleteShader(s);
    return 0;
  }
  return s;
}

GLuint program(const char* vs, const char* fs, std::string* err) {
  GLuint v = compile(GL_VERTEX_SHADER, vs, err);
  if (!v) return 0;
  GLuint f = compile(GL_FRAGMENT_SHADER, fs, err);
  if (!f) return 0;
  GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  glDeleteShader(v);
  glDeleteShader(f);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(p, sizeof(log), nullptr, log);
    if (err) *err = std::string("program link: ") + log;
    return 0;
  }
  return p;
}

// ---- ego mesh -------------------------------------------------------------
struct Sec { float y, hw, sill, belt, rhw, roof; };
// Low-poly hatch/sedan, ~4.7 m x 1.92 m, rear at y=-2.35.
const Sec kEgo[] = {
    {-2.36f, 0.86f, 0.36f, 0.84f, 0.60f, 0.92f}, {-2.10f, 0.94f, 0.30f, 0.90f, 0.62f, 1.00f},
    {-1.55f, 0.97f, 0.27f, 0.94f, 0.60f, 1.30f}, {-0.70f, 0.97f, 0.27f, 0.95f, 0.62f, 1.44f},
    {0.35f, 0.97f, 0.27f, 0.95f, 0.64f, 1.44f},  {1.00f, 0.96f, 0.27f, 0.93f, 0.72f, 1.10f},
    {1.60f, 0.94f, 0.29f, 0.82f, 0.80f, 0.86f},  {2.36f, 0.78f, 0.36f, 0.66f, 0.66f, 0.70f},
};

void tri(std::vector<MVert>& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 col, float emi = 0.f) {
  glm::vec3 n = glm::cross(b - a, c - a);
  float L = glm::length(n);
  n = L > 1e-6f ? n / L : glm::vec3(0, 0, 1);
  m.push_back({a, n, col, emi});
  m.push_back({b, n, col, emi});
  m.push_back({c, n, col, emi});
}
void quad(std::vector<MVert>& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 col, float emi = 0.f) {
  tri(m, a, b, c, col, emi);
  tri(m, a, c, d, col, emi);
}

std::vector<MVert> build_ego() {
  std::vector<MVert> m;
  const glm::vec3 body(0.92f, 0.93f, 0.95f), glass(0.10f, 0.11f, 0.13f), lamp(1.0f, 0.12f, 0.10f),
      dark(0.16f, 0.16f, 0.17f), head(0.95f, 0.95f, 0.9f);
  const int n = sizeof(kEgo) / sizeof(kEgo[0]);
  for (int i = 0; i + 1 < n; ++i) {
    const Sec& a = kEgo[i];
    const Sec& b = kEgo[i + 1];
    for (int s = -1; s <= 1; s += 2) {
      float sa = static_cast<float>(s);
      glm::vec3 a0(sa * a.hw, a.y, a.sill), a1(sa * a.hw, a.y, a.belt), a2(sa * a.rhw, a.y, a.roof);
      glm::vec3 b0(sa * b.hw, b.y, b.sill), b1(sa * b.hw, b.y, b.belt), b2(sa * b.rhw, b.y, b.roof);
      bool glassy = (a.roof - a.belt > 0.3f) && (b.roof - b.belt > 0.3f);
      if (s < 0) {
        quad(m, a0, a1, b1, b0, body);
        quad(m, a1, a2, b2, b1, glassy ? glass : body);
      } else {
        quad(m, a0, b0, b1, a1, body);
        quad(m, a1, b1, b2, a2, glassy ? glass : body);
      }
    }
    // top: roof panel flat, slopes (windscreen / rear glass) glassy
    glm::vec3 al(-a.rhw, a.y, a.roof), ar(a.rhw, a.y, a.roof), bl(-b.rhw, b.y, b.roof), br(b.rhw, b.y, b.roof);
    bool slope = std::fabs(a.roof - b.roof) > 0.25f;
    bool cabin = (a.roof - a.belt > 0.3f) && (b.roof - b.belt > 0.3f);
    quad(m, al, ar, br, bl, (slope || cabin) ? glass : body);  // glass roof
    // floor
    quad(m, glm::vec3(-a.hw, a.y, a.sill), glm::vec3(-b.hw, b.y, b.sill), glm::vec3(b.hw, b.y, b.sill),
         glm::vec3(a.hw, a.y, a.sill), dark);
  }
  // caps
  for (int end = 0; end < 2; ++end) {
    const Sec& c = end == 0 ? kEgo[0] : kEgo[n - 1];
    float sgn = end == 0 ? -1.f : 1.f;
    glm::vec3 l0(-c.hw, c.y, c.sill), r0(c.hw, c.y, c.sill), l1(-c.hw, c.y, c.belt), r1(c.hw, c.y, c.belt),
        l2(-c.rhw, c.y, c.roof), r2(c.rhw, c.y, c.roof);
    if (end == 0) {
      quad(m, l0, r0, r1, l1, body);
      quad(m, l1, r1, r2, l2, body);
      // tail lamps: emissive strips just behind the cap
      float y = c.y - 0.01f, z0 = c.belt - 0.16f, z1 = c.belt - 0.04f;
      quad(m, {-c.hw + 0.02f, y, z0}, {-c.hw + 0.42f, y, z0}, {-c.hw + 0.42f, y, z1}, {-c.hw + 0.02f, y, z1}, lamp, 1.f);
      quad(m, {c.hw - 0.42f, y, z0}, {c.hw - 0.02f, y, z0}, {c.hw - 0.02f, y, z1}, {c.hw - 0.42f, y, z1}, lamp, 1.f);
    } else {
      quad(m, r0, l0, l1, r1, body);
      quad(m, r1, l1, l2, r2, body);
      float y = c.y + 0.01f, z0 = c.belt - 0.14f, z1 = c.belt - 0.04f;
      quad(m, {c.hw - 0.02f, y, z0}, {c.hw - 0.36f, y, z0}, {c.hw - 0.36f, y, z1}, {c.hw - 0.02f, y, z1}, head, 1.f);
      quad(m, {-c.hw + 0.36f, y, z0}, {-c.hw + 0.02f, y, z0}, {-c.hw + 0.02f, y, z1}, {-c.hw + 0.36f, y, z1}, head, 1.f);
    }
    (void)sgn;
  }
  // wheels: dark boxes poking below the sill
  for (float wy : {-1.45f, 1.40f}) {
    for (float sx : {-1.f, 1.f}) {
      float x0 = sx * 0.97f, x1 = sx * 0.80f, y0 = wy - 0.34f, y1 = wy + 0.34f, z0 = 0.0f, z1 = 0.62f;
      glm::vec3 c(0.06f);
      quad(m, {x0, y0, z0}, {x0, y1, z0}, {x0, y1, z1}, {x0, y0, z1}, c);
      quad(m, {x0, y0, z1}, {x0, y1, z1}, {x1, y1, z1}, {x1, y0, z1}, c);
      quad(m, {x0, y0, z0}, {x0, y0, z1}, {x1, y0, z1}, {x1, y0, z0}, c);
      quad(m, {x0, y1, z0}, {x1, y1, z0}, {x1, y1, z1}, {x0, y1, z1}, c);
    }
  }
  // make every normal face outward from the car's centre (winding above is mixed)
  const glm::vec3 centre(0.f, 0.f, 0.7f);
  for (size_t i = 0; i + 2 < m.size(); i += 3) {
    glm::vec3 mid = (m[i].pos + m[i + 1].pos + m[i + 2].pos) / 3.f;
    if (glm::dot(m[i].nrm, mid - centre) < 0.f) {
      for (int k = 0; k < 3; ++k) m[i + k].nrm = -m[i + k].nrm;
    }
  }
  return m;
}

// ---- polyline walking -----------------------------------------------------
struct Walked { glm::vec2 p, t; };

std::vector<Walked> walk(const std::vector<P2>& pts, float step, float phase) {
  std::vector<Walked> out;
  if (pts.size() < 2) return out;
  float next = phase;
  float acc = 0.f;
  for (size_t i = 0; i + 1 < pts.size(); ++i) {
    glm::vec2 a(pts[i].x, pts[i].y), b(pts[i + 1].x, pts[i + 1].y);
    glm::vec2 d = b - a;
    float L = glm::length(d);
    if (L < 1e-5f) continue;
    glm::vec2 t = d / L;
    while (next <= acc + L) {
      out.push_back({a + t * (next - acc), t});
      next += step;
      if (out.size() > 4000) return out;
    }
    acc += L;
  }
  return out;
}

float far_gain(float y) {
  if (y > kFarCut) return 0.f;
  if (y < kFarFadeFrom) return 1.f;
  return std::max(0.15f, 1.f - (y - kFarFadeFrom) / (kFarCut - kFarFadeFrom));
}

}  // namespace

// ---- public colour rules -------------------------------------------------
glm::vec3 lane_rgb(Paint p) {
  // Real paint: yellow paint is yellow, everything else white. No purple.
  return p == Paint::Yellow ? glm::vec3(1.0f, 0.80f, 0.10f) : glm::vec3(0.93f, 0.93f, 0.93f);
}

glm::vec3 edge_rgb(Palette pal) {
  // Red is only road edges / curbs (orange in the dusk overlay palette).
  return pal == Palette::Dusk ? glm::vec3(1.0f, 0.42f, 0.12f) : glm::vec3(1.0f, 0.10f, 0.08f);
}

glm::vec3 track_rgb(const Track& t, int lead_id, Palette pal) {
  bool lead = t.lead || (lead_id >= 0 && t.id == lead_id);
  if (lead) return {0.20f, 1.0f, 0.22f};
  if (pal == Palette::Dusk) return is_cls(t, {"pedestrian", "ped", "person"}) ? glm::vec3(0.75f, 0.45f, 1.0f)
                                                                            : glm::vec3(1.0f, 0.50f, 0.18f);
  if (is_cls(t, {"pedestrian", "ped", "person"})) return {1.0f, 0.55f, 0.95f};
  if (is_cls(t, {"bicycle", "bike", "cyclist", "motorcycle"})) return {1.0f, 0.90f, 0.15f};
  if (is_cls(t, {"obstacle", "parked", "debris", "cone"})) return {1.0f, 0.55f, 0.10f};
  if (is_cls(t, {"truck", "bus", "trailer"})) return {0.95f, 0.25f, 1.0f};
  // A car crossing our path (heading mostly sideways) is yellow, like the
  // cross traffic in the refs; the rest are blue.
  if (std::fabs(std::cos(t.yaw)) > 0.8f) return {1.0f, 0.88f, 0.20f};
  return {0.12f, 0.24f, 1.0f};
}

bool lens_named(const std::string& name, Lens& out) {
  if (name == "chase") { out = Lens{{0.f, -32.f, 10.f}, {0.f, 40.f, 0.f}, 40.f}; return true; }
  if (name == "low") { out = Lens{{0.f, -10.5f, 3.3f}, {0.f, 18.f, 0.4f}, 40.f}; return true; }
  if (name == "high") { out = Lens{{0.f, -16.f, 22.f}, {0.f, 12.f, 0.f}, 50.f}; return true; }
  if (name == "top") { out = Lens{{0.f, 8.f, 70.f}, {0.f, 8.1f, 0.f}, 45.f}; return true; }
  if (name == "side") { out = Lens{{-14.f, -16.f, 7.5f}, {1.5f, 14.f, 0.f}, 34.f}; return true; }
  if (name == "dash") { out = Lens{{0.f, -22.9f, 8.6f}, {0.f, 21.8f, 0.f}, 45.f}; return true; }
  return false;
}

bool palette_named(const std::string& name, Palette& out) {
  if (name == "neon") { out = Palette::Neon; return true; }
  if (name == "parking") { out = Palette::Parking; return true; }
  if (name == "dusk") { out = Palette::Dusk; return true; }
  return false;
}

// ---- renderer -------------------------------------------------------------
struct Renderer::Impl {
  GLuint p_caps = 0, p_boxes = 0, p_ground = 0, p_sprite = 0, p_hud = 0, p_mesh = 0, p_bg = 0;
  GLuint vao_caps = 0, vao_boxes = 0, vao_ground = 0, vao_sprite = 0, vao_hud = 0, vao_mesh = 0, vao_bg = 0;
  GLuint vb_corner = 0, vb_caps = 0, vb_boxes = 0, vb_ground = 0, vb_sprite = 0, vb_hud = 0, vb_mesh = 0;
  size_t cap_caps = 0, cap_boxes = 0, cap_ground = 0, cap_sprite = 0, cap_hud = 0;
  GLsizei mesh_n = 0;
  GLuint fbo_ms = 0, rb_color = 0, rb_depth = 0, fbo_res = 0, tex_res = 0;
  int tw = 0, th = 0, tmsaa = 0;
  GLuint tex_bg = 0;
  const Image* bg_src = nullptr;
  int bg_w = 0, bg_h = 0;
  GLuint q[2] = {0, 0};
  int q_frame = 0;
  bool q_pending[2] = {false, false};
  size_t buf_bytes = 0;
  std::vector<Capsule> caps;
  std::vector<BoxStroke> boxes;
  std::vector<GVert> ground;
  std::vector<Sprite> sprites;
  std::vector<Hud> hud;
};

Renderer::~Renderer() {
  if (!p_) return;
  // GL objects die with the context; the context owner tears that down.
  delete p_;
}

static void attrib(GLuint idx, int n, size_t stride, size_t off, bool inst) {
  glEnableVertexAttribArray(idx);
  glVertexAttribPointer(idx, n, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(stride), reinterpret_cast<void*>(off));
  glVertexAttribDivisor(idx, inst ? 1 : 0);
}

bool Renderer::init(std::string* err) {
  p_ = new Impl();
  Impl& I = *p_;
  if (!(I.p_caps = program(kCapsuleVS, kCapsuleFS, err))) return false;
  if (!(I.p_boxes = program(kBoxVS, kBoxFS, err))) return false;
  if (!(I.p_ground = program(kGroundVS, kGroundFS, err))) return false;
  if (!(I.p_sprite = program(kSpriteVS, kSpriteFS, err))) return false;
  if (!(I.p_hud = program(kHudVS, kHudFS, err))) return false;
  if (!(I.p_mesh = program(kMeshVS, kMeshFS, err))) return false;
  if (!(I.p_bg = program(kBgVS, kBgFS, err))) return false;

  const float corners[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
  glGenBuffers(1, &I.vb_corner);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_corner);
  glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
  I.buf_bytes += sizeof(corners);

  auto corner_attr = [&]() {
    glBindBuffer(GL_ARRAY_BUFFER, I.vb_corner);
    attrib(0, 2, 2 * sizeof(float), 0, false);
  };
  glGenVertexArrays(1, &I.vao_caps);
  glBindVertexArray(I.vao_caps);
  corner_attr();
  glGenBuffers(1, &I.vb_caps);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_caps);
  attrib(1, 3, sizeof(Capsule), offsetof(Capsule, p0), true);
  attrib(2, 3, sizeof(Capsule), offsetof(Capsule, p1), true);
  attrib(3, 2, sizeof(Capsule), offsetof(Capsule, rad), true);
  attrib(4, 4, sizeof(Capsule), offsetof(Capsule, color), true);
  attrib(5, 2, sizeof(Capsule), offsetof(Capsule, glow), true);

  glGenVertexArrays(1, &I.vao_boxes);
  glBindVertexArray(I.vao_boxes);
  corner_attr();
  glGenBuffers(1, &I.vb_boxes);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_boxes);
  for (int i = 0; i < 8; ++i) {
    attrib(1 + i, 3, sizeof(BoxStroke), offsetof(BoxStroke, v) + i * sizeof(glm::vec3), true);
  }
  attrib(9, 4, sizeof(BoxStroke), offsetof(BoxStroke, color), true);
  attrib(10, 2, sizeof(BoxStroke), offsetof(BoxStroke, rad), true);
  attrib(11, 2, sizeof(BoxStroke), offsetof(BoxStroke, glow), true);

  glGenVertexArrays(1, &I.vao_sprite);
  glBindVertexArray(I.vao_sprite);
  corner_attr();
  glGenBuffers(1, &I.vb_sprite);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_sprite);
  attrib(1, 3, sizeof(Sprite), offsetof(Sprite, pos), true);
  attrib(2, 2, sizeof(Sprite), offsetof(Sprite, size_m), true);
  attrib(3, 1, sizeof(Sprite), offsetof(Sprite, min_px), true);
  attrib(4, 1, sizeof(Sprite), offsetof(Sprite, type), true);
  attrib(5, 4, sizeof(Sprite), offsetof(Sprite, color), true);

  glGenVertexArrays(1, &I.vao_hud);
  glBindVertexArray(I.vao_hud);
  corner_attr();
  glGenBuffers(1, &I.vb_hud);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_hud);
  attrib(1, 4, sizeof(Hud), offsetof(Hud, rect), true);
  attrib(2, 4, sizeof(Hud), offsetof(Hud, color), true);
  attrib(3, 2, sizeof(Hud), offsetof(Hud, kind), true);

  glGenVertexArrays(1, &I.vao_bg);
  glBindVertexArray(I.vao_bg);
  corner_attr();

  glGenVertexArrays(1, &I.vao_ground);
  glBindVertexArray(I.vao_ground);
  glGenBuffers(1, &I.vb_ground);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_ground);
  attrib(0, 3, sizeof(GVert), offsetof(GVert, pos), false);
  attrib(1, 4, sizeof(GVert), offsetof(GVert, color), false);
  attrib(2, 2, sizeof(GVert), offsetof(GVert, uv), false);
  attrib(3, 1, sizeof(GVert), offsetof(GVert, mode), false);

  std::vector<MVert> ego = build_ego();
  glGenVertexArrays(1, &I.vao_mesh);
  glBindVertexArray(I.vao_mesh);
  glGenBuffers(1, &I.vb_mesh);
  glBindBuffer(GL_ARRAY_BUFFER, I.vb_mesh);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(ego.size() * sizeof(MVert)), ego.data(), GL_STATIC_DRAW);
  I.buf_bytes += ego.size() * sizeof(MVert);
  attrib(0, 3, sizeof(MVert), offsetof(MVert, pos), false);
  attrib(1, 3, sizeof(MVert), offsetof(MVert, nrm), false);
  attrib(2, 3, sizeof(MVert), offsetof(MVert, col), false);
  attrib(3, 1, sizeof(MVert), offsetof(MVert, emissive), false);
  I.mesh_n = static_cast<GLsizei>(ego.size());
  glBindVertexArray(0);

  glGenQueries(2, I.q);
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    if (err) *err = "GL error during init: " + std::to_string(e);
    return false;
  }
  return true;
}


size_t estimate_target_bytes(int w, int h, int msaa) {
  int ww = std::max(1, w), hh = std::max(1, h);
  int s = std::max(1, msaa);
  size_t px = static_cast<size_t>(ww) * static_cast<size_t>(hh);
  return px * 4 * static_cast<size_t>(s) + px * 4 * static_cast<size_t>(s) + px * 4;
}

void cap_render_targets(int& w, int& h, int& msaa, size_t extra, size_t budget) {
  w = std::clamp(w, 16, 8192);
  h = std::clamp(h, 16, 8192);
  msaa = std::clamp(msaa, 0, 8);
  auto fits = [&](int ww, int hh, int s) {
    return estimate_target_bytes(ww, hh, std::max(1, s)) + extra <= budget;
  };
  if (fits(w, h, msaa)) return;
  while (msaa > 1 && !fits(w, h, msaa)) --msaa;
  if (fits(w, h, msaa)) return;
  msaa = std::min(msaa, 1);
  const double denom = 4.0 * (2.0 * static_cast<double>(std::max(1, msaa)) + 1.0);
  if (budget <= extra || denom <= 0.0) {
    w = 16;
    h = 16;
    return;
  }
  const double area_budget = static_cast<double>(budget - extra) / denom;
  const double area = static_cast<double>(w) * static_cast<double>(h);
  if (area > area_budget && area > 0.0) {
    const double scale = std::sqrt(area_budget / area);
    w = std::max(16, static_cast<int>(std::floor(w * scale / 2.0) * 2.0));
    h = std::max(16, static_cast<int>(std::floor(h * scale / 2.0) * 2.0));
  }
  while (!fits(w, h, msaa) && (w > 16 || h > 16)) {
    w = std::max(16, w - 16);
    h = std::max(16, h - 16);
  }
}

bool Renderer::ensure_targets(int w, int h, int msaa) {
  Impl& I = *p_;
  if (I.fbo_ms && I.tw == w && I.th == h && I.tmsaa == msaa) return true;
  if (I.fbo_ms) {
    glDeleteFramebuffers(1, &I.fbo_ms);
    glDeleteFramebuffers(1, &I.fbo_res);
    glDeleteRenderbuffers(1, &I.rb_color);
    glDeleteRenderbuffers(1, &I.rb_depth);
    glDeleteTextures(1, &I.tex_res);
  }
  GLint max_s = 0;
  glGetIntegerv(GL_MAX_SAMPLES, &max_s);
  int s = std::max(0, std::min(msaa, static_cast<int>(max_s)));
  glGenRenderbuffers(1, &I.rb_color);
  glBindRenderbuffer(GL_RENDERBUFFER, I.rb_color);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, s, GL_RGBA8, w, h);
  glGenRenderbuffers(1, &I.rb_depth);
  glBindRenderbuffer(GL_RENDERBUFFER, I.rb_depth);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, s, GL_DEPTH_COMPONENT24, w, h);
  glGenFramebuffers(1, &I.fbo_ms);
  glBindFramebuffer(GL_FRAMEBUFFER, I.fbo_ms);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, I.rb_color);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, I.rb_depth);
  bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  glGenTextures(1, &I.tex_res);
  glBindTexture(GL_TEXTURE_2D, I.tex_res);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glGenFramebuffers(1, &I.fbo_res);
  glBindFramebuffer(GL_FRAMEBUFFER, I.fbo_res);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, I.tex_res, 0);
  ok = ok && glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  I.tw = w;
  I.th = h;
  I.tmsaa = s;
  return ok;
}

void Renderer::upload_background(const Image* bg) {
  Impl& I = *p_;
  if (!bg || bg->w <= 0) return;
  if (I.bg_src == bg && I.bg_w == bg->w && I.bg_h == bg->h) return;
  if (!I.tex_bg) glGenTextures(1, &I.tex_bg);
  glBindTexture(GL_TEXTURE_2D, I.tex_bg);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, bg->w, bg->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg->rgba.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  I.bg_src = bg;
  I.bg_w = bg->w;
  I.bg_h = bg->h;
}

size_t Renderer::gl_bytes() const {
  if (!p_) return 0;
  const Impl& I = *p_;
  size_t px = static_cast<size_t>(I.tw) * I.th;
  size_t s = static_cast<size_t>(std::max(1, I.tmsaa));
  size_t bytes = px * 4 * s + px * 4 * s /*depth24+pad*/ + px * 4 /*resolve*/;
  bytes += static_cast<size_t>(I.bg_w) * I.bg_h * 4;
  bytes += I.buf_bytes;
  return bytes;
}

void Renderer::target_size(int* w, int* h, int* msaa) const {
  if (!p_) {
    if (w) *w = 0;
    if (h) *h = 0;
    if (msaa) *msaa = 0;
    return;
  }
  if (w) *w = p_->tw;
  if (h) *h = p_->th;
  if (msaa) *msaa = p_->tmsaa;
}

template <class T>
static void upload(GLuint vb, const std::vector<T>& v, size_t& cap, size_t& total) {
  glBindBuffer(GL_ARRAY_BUFFER, vb);
  size_t need = v.size() * sizeof(T);
  if (need > cap) {
    size_t nc = std::max<size_t>(need * 3 / 2, 4096);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(nc), nullptr, GL_DYNAMIC_DRAW);
    total += nc - cap;
    cap = nc;
  }
  if (need) glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(need), v.data());
}

// ---- scene -> instances ---------------------------------------------------
static void add_box(std::vector<Capsule>& caps, std::vector<BoxStroke>& boxes, const Track& t, glm::vec3 col) {
  glm::vec3 d = dims_of(t);
  float c = std::cos(t.yaw), s = std::sin(t.yaw);
  glm::vec2 fwd(c, s), side(-s, c);
  glm::vec2 ctr(t.x, t.y);
  glm::vec3 v[8];
  for (int i = 0; i < 8; ++i) {
    float fl = (i & 1) ? 0.5f : -0.5f, sl = (i & 2) ? 0.5f : -0.5f;
    glm::vec2 p = ctr + fwd * (fl * d.x) + side * (sl * d.y);
    v[i] = glm::vec3(p, (i & 4) ? d.z : 0.03f);
  }
  const int E[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7}, {7, 6}, {6, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  float g = far_gain(t.y);
  if (g <= 0.f) return;
  if (kBoxUniformStroke) {
    BoxStroke b;
    for (int i = 0; i < 8; ++i) b.v[i] = v[i];
    b.color = {col, 1.0f * g};
    b.rad = {0.055f, 1.9f};
    b.glow = {10.f, 0.5f * g};
    boxes.push_back(b);
  } else {
    for (auto& e : E) caps.push_back({v[e[0]], v[e[1]], {0.055f, 1.9f}, {col, 1.0f * g}, {10.f, 0.5f * g}});
  }
  // centre dot on the end face toward the ego (the refs' filled disc)
  glm::vec2 f1 = ctr + fwd * (0.5f * d.x), f2 = ctr - fwd * (0.5f * d.x);
  glm::vec2 face = glm::length(f1) < glm::length(f2) ? f1 : f2;
  glm::vec3 dot(face, d.z * 0.45f);
  float r = std::min(0.22f * std::min(d.y, d.z), 0.38f);
  caps.push_back({dot, dot, {r, 2.2f}, {col, 1.1f * g}, {6.f, 0.6f * g}});
}

static void ground_quad(std::vector<GVert>& g, glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d, glm::vec4 col,
                        float mode = 0.f, float z = 0.02f) {
  GVert A{{a, z}, col, {-1, -1}, mode}, B{{b, z}, col, {1, -1}, mode}, C{{c, z}, col, {1, 1}, mode},
      D{{d, z}, col, {-1, 1}, mode};
  g.push_back(A); g.push_back(B); g.push_back(C);
  g.push_back(A); g.push_back(C); g.push_back(D);
}

static void lane_span_at(const Scene& s, float y, float* lo, float* hi) {
  *lo = 1e9f; *hi = -1e9f;
  for (const auto& l : s.lanes) {
    for (size_t i = 0; i + 1 < l.pts.size(); ++i) {
      const P2 &a = l.pts[i], &b = l.pts[i + 1];
      if ((a.y - y) * (b.y - y) <= 0.f && std::fabs(b.y - a.y) > 1e-4f) {
        float x = a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y);
        *lo = std::min(*lo, x);
        *hi = std::max(*hi, x);
      }
    }
  }
  if (*lo > *hi) { *lo = -5.5f; *hi = 5.5f; }
}

static void seg7(std::vector<Hud>& h, int digit, float x, float y, float W, float H, glm::vec4 col) {
  static const unsigned char M[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
  float t = W * 0.2f;
  unsigned char m = M[digit % 10];
  auto r = [&](float rx, float ry, float rw, float rh) { h.push_back({{rx, ry, rw, rh}, col, {0.f, t * 0.5f}}); };
  if (m & 0x01) r(x + t * 0.5f, y, W - t, t);
  if (m & 0x02) r(x + W - t, y + t * 0.5f, t, H * 0.5f - t * 0.5f);
  if (m & 0x04) r(x + W - t, y + H * 0.5f, t, H * 0.5f - t * 0.5f);
  if (m & 0x08) r(x + t * 0.5f, y + H - t, W - t, t);
  if (m & 0x10) r(x, y + H * 0.5f, t, H * 0.5f - t * 0.5f);
  if (m & 0x20) r(x, y + t * 0.5f, t, H * 0.5f - t * 0.5f);
  if (m & 0x40) r(x + t * 0.5f, y + H * 0.5f - t * 0.5f, W - t, t);
}

bool Renderer::render(const Scene& s, const RenderOptions& o, Image* out, bool blit_to_default, int fb_w,
                      int fb_h) {
  if (!p_) return false;
  Impl& I = *p_;
  auto t0 = std::chrono::steady_clock::now();
  int w = o.w, h = o.h, msaa = o.msaa;
  size_t extra = I.buf_bytes;
  if (o.background && o.background->w > 0 && o.background->h > 0) {
    extra += static_cast<size_t>(o.background->w) * static_cast<size_t>(o.background->h) * 4;
  }
  cap_render_targets(w, h, msaa, extra);
  if (!ensure_targets(w, h, msaa)) return false;
  upload_background(o.background);

  // GPU timer from two frames ago (non-blocking read).
  int qi = I.q_frame & 1;
  if (I.q_pending[qi]) {
    GLint avail = 0;
    glGetQueryObjectiv(I.q[qi], GL_QUERY_RESULT_AVAILABLE, &avail);
    if (avail) {
      GLuint64 ns = 0;
      glGetQueryObjectui64v(I.q[qi], GL_QUERY_RESULT, &ns);
      stats_.gpu_ms = static_cast<double>(ns) / 1e6;
    }
    I.q_pending[qi] = false;
  }

  const Palette pal = o.palette;
  I.caps.clear();
  I.boxes.clear();
  I.ground.clear();
  I.sprites.clear();
  I.hud.clear();

  // lanes: dashes near, beads far (perspective shrinks the dash)
  for (const auto& l : s.lanes) {
    glm::vec3 col = lane_rgb(l.paint);
    float k = l.predicted ? 0.45f : 1.0f;
    for (const auto& w : walk(l.pts, kLaneStep, 0.4f)) {
      float g = far_gain(w.p.y) * k;
      if (g <= 0.f) continue;
      glm::vec3 a(w.p - w.t * kLaneHalf, 0.04f), b(w.p + w.t * kLaneHalf, 0.04f);
      I.caps.push_back({a, b, {kLaneRad, 1.1f}, {col, g}, {5.f, 0.45f * g}});
    }
  }
  // road edges / curbs: red dots
  glm::vec3 ecol = edge_rgb(pal);
  for (const auto& e : s.edges) {
    float k = e.predicted ? 0.6f : 1.0f;
    for (const auto& w : walk(e.pts, kEdgeStep, 0.2f)) {
      float g = far_gain(w.p.y) * k;
      if (g <= 0.f) continue;
      glm::vec3 p(w.p, 0.05f);
      I.caps.push_back({p, p, {kEdgeRad, 1.3f}, {ecol, g}, {5.f, 0.55f * g}});
    }
  }
  // planned path: blue beads
  if (s.path.size() >= 2) {
    float gain = std::clamp(s.path_conf, 0.45f, 1.0f);
    glm::vec3 pcol(0.20f, 0.45f, 1.0f);
    for (const auto& w : walk(s.path, kPathStep, 2.0f)) {
      float g = far_gain(w.p.y) * gain;
      glm::vec3 p(w.p, 0.06f);
      I.caps.push_back({p, p, {kPathRad, 1.6f}, {pcol, g}, {7.f, 0.6f * g}});
    }
  }
  // boxes
  for (const auto& t : s.tracks) add_box(I.caps, I.boxes, t, track_rgb(t, s.lead_id, pal));

  // signs
  bool crosswalk_done = false;
  for (const auto& g : s.signs) {
    if (g.cls == "traffic_light") {
      glm::vec4 col(0.0f);  // a = lit lamp index+1 (1 red top, 2 yellow, 3 green); 0 = none
      if (g.state == "red") col = {1.0f, 0.12f, 0.10f, 1.f};
      else if (g.state == "yellow") col = {1.0f, 0.78f, 0.10f, 2.f};
      else if (g.state == "green") col = {0.15f, 1.0f, 0.35f, 3.f};
      I.sprites.push_back({{g.x, g.y, kLightZ}, {1.25f, 1.9f}, 11.f, 0.f, col});
    } else if (g.cls == "stop_sign") {
      I.sprites.push_back({{g.x, g.y, kSignZ}, {0.9f, 0.9f}, 7.f, 1.f, glm::vec4(1.f)});
      if (!crosswalk_done) {
        crosswalk_done = true;
        float lo, hi;
        lane_span_at(s, g.y, &lo, &hi);
        lo -= 0.6f; hi += 0.6f;
        float y0 = g.y - 0.5f;
        // stop bar
        ground_quad(I.ground, {lo, y0 - 0.5f}, {hi, y0 - 0.5f}, {hi, y0}, {lo, y0}, {0.95f, 0.95f, 0.95f, 0.9f});
        // zebra
        for (float x = lo; x + 0.5f <= hi; x += 1.1f) {
          ground_quad(I.ground, {x, y0 + 0.8f}, {x + 0.55f, y0 + 0.8f}, {x + 0.55f, y0 + 3.8f}, {x, y0 + 3.8f},
                      {0.72f, 0.72f, 0.74f, 0.75f});
        }
      }
    } else if (g.cls == "pole") {
      glm::vec3 a(g.x, g.y, 0.f), b(g.x, g.y, 3.0f);
      I.caps.push_back({a, b, {0.05f, 1.f}, {glm::vec3(0.6f), 0.8f}, {0.f, 0.f}});
    }
  }

  // ego glow (white disc on the ground) + headlight wash ahead
  if (o.draw_ego) {
    ground_quad(I.ground, {-2.9f, -4.0f}, {2.9f, -4.0f}, {2.9f, 4.2f}, {-2.9f, 4.2f}, {1.f, 1.f, 1.f, 0.85f}, 1.f, 0.015f);
    ground_quad(I.ground, {-2.6f, 1.5f}, {2.6f, 1.5f}, {2.6f, 13.f}, {-2.6f, 13.f}, {0.85f, 0.88f, 1.f, 0.22f}, 1.f, 0.012f);
  }

  // HUD: speed digits + bar, engaged ring
  if (o.draw_hud) {
    float W = static_cast<float>(w), H = static_cast<float>(h);
    float u = H / 720.f;
    int mph = static_cast<int>(std::lround(std::max(0.f, s.speed_mps) * 2.23694f));
    mph = std::min(mph, 199);
    glm::vec4 white(0.95f, 0.95f, 0.95f, 0.95f);
    float dw = 22.f * u, dh = 40.f * u, gap = 8.f * u;
    int nd = mph >= 100 ? 3 : (mph >= 10 ? 2 : 1);
    float total = nd * dw + (nd - 1) * gap;
    float x = W * 0.5f - total * 0.5f, y = H - 92.f * u;
    int digits[3] = {mph / 100, (mph / 10) % 10, mph % 10};
    for (int i = 3 - nd; i < 3; ++i) { seg7(I.hud, digits[i], x, y, dw, dh, white); x += dw + gap; }
    // speed bar: track + fill (0..90 mph), set-speed tick
    float bw = 240.f * u, bh = 6.f * u, bx = W * 0.5f - bw * 0.5f, by = H - 40.f * u;
    I.hud.push_back({{bx, by, bw, bh}, {0.35f, 0.35f, 0.38f, 0.8f}, {0.f, 3.f * u}});
    float f = std::clamp(s.speed_mps * 2.23694f / 90.f, 0.f, 1.f);
    if (f > 0.f) I.hud.push_back({{bx, by, bw * f, bh}, {0.25f, 0.55f, 1.0f, 0.95f}, {0.f, 3.f * u}});
    if (s.set_mps >= 0.f) {
      float fs = std::clamp(s.set_mps * 2.23694f / 90.f, 0.f, 1.f);
      I.hud.push_back({{bx + bw * fs - 2.f * u, by - 6.f * u, 4.f * u, bh + 12.f * u}, white, {0.f, 1.f}});
    }
    // engaged: blue steering ring left of the digits; grey when not engaged
    glm::vec4 ring = s.engaged ? glm::vec4(0.25f, 0.55f, 1.0f, 1.f) : glm::vec4(0.5f, 0.5f, 0.5f, 0.6f);
    float rs = 34.f * u;
    I.hud.push_back({{W * 0.5f - total * 0.5f - rs - 26.f * u, y + dh * 0.5f - rs * 0.5f, rs, rs}, ring, {1.f, 3.5f * u}});
    if (s.aeb == "brake") {
      I.hud.push_back({{W * 0.5f - 60.f * u, 14.f * u, 120.f * u, 26.f * u}, {1.f, 0.15f, 0.1f, 0.95f}, {0.f, 6.f * u}});
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  stats_.cpu_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  stats_.capsules = static_cast<int>(I.caps.size() + I.boxes.size());
  stats_.sprites = static_cast<int>(I.sprites.size());
  stats_.ground_verts = static_cast<int>(I.ground.size());

  upload(I.vb_caps, I.caps, I.cap_caps, I.buf_bytes);
  upload(I.vb_boxes, I.boxes, I.cap_boxes, I.buf_bytes);
  upload(I.vb_ground, I.ground, I.cap_ground, I.buf_bytes);
  upload(I.vb_sprite, I.sprites, I.cap_sprite, I.buf_bytes);
  upload(I.vb_hud, I.hud, I.cap_hud, I.buf_bytes);

  glBeginQuery(GL_TIME_ELAPSED, I.q[qi]);
  glBindFramebuffer(GL_FRAMEBUFFER, I.fbo_ms);
  glViewport(0, 0, w, h);
  glDisable(GL_CULL_FACE);
  glDepthMask(GL_TRUE);
  glClearDepth(1.0);
  // background
  glm::vec3 top(0.f), bot(0.f);
  if (pal == Palette::Parking) { top = {0.05f, 0.06f, 0.07f}; bot = {0.17f, 0.18f, 0.19f}; }
  if (pal == Palette::Dusk) { top = {0.16f, 0.20f, 0.33f}; bot = {0.03f, 0.03f, 0.04f}; }
  glClearColor(top.r, top.g, top.b, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  if (o.background || pal != Palette::Neon) {
    glUseProgram(I.p_bg);
    glUniform1i(glGetUniformLocation(I.p_bg, "u_mode"), o.background ? 1 : 0);
    glUniform3fv(glGetUniformLocation(I.p_bg, "u_top"), 1, glm::value_ptr(top));
    glUniform3fv(glGetUniformLocation(I.p_bg, "u_bot"), 1, glm::value_ptr(bot));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, I.tex_bg);
    glUniform1i(glGetUniformLocation(I.p_bg, "u_tex"), 0);
    glBindVertexArray(I.vao_bg);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }

  const float aspect = static_cast<float>(w) / static_cast<float>(h);
  const float near_z = 0.3f;
  glm::mat4 view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 0, 1));
  if (std::fabs(glm::normalize(o.lens.look - o.lens.eye).z) > 0.99f)
    view = glm::lookAt(o.lens.eye, o.lens.look, glm::vec3(0, 1, 0));
  glm::mat4 proj = glm::perspective(glm::radians(o.lens.vfov_deg), aspect, near_z, 600.f);
  glm::mat4 vp = proj * view;
  glm::vec2 vpx(static_cast<float>(w), static_cast<float>(h));

  // ego mesh first: writes depth so lines and glow under/behind it are hidden
  if (o.draw_ego) {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(I.p_mesh);
    glUniformMatrix4fv(glGetUniformLocation(I.p_mesh, "u_vp_m"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform3fv(glGetUniformLocation(I.p_mesh, "u_eye"), 1, glm::value_ptr(o.lens.eye));
    glm::vec3 tint = pal == Palette::Parking ? glm::vec3(0.25f, 0.55f, 1.0f) : glm::vec3(1.0f);
    glUniform3fv(glGetUniformLocation(I.p_mesh, "u_tint"), 1, glm::value_ptr(tint));
    glBindVertexArray(I.vao_mesh);
    glDrawArrays(GL_TRIANGLES, 0, I.mesh_n);
  }
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  // premultiplied; alpha 0 = additive

  if (!I.ground.empty()) {
    glUseProgram(I.p_ground);
    glUniformMatrix4fv(glGetUniformLocation(I.p_ground, "u_vp_m"), 1, GL_FALSE, glm::value_ptr(vp));
    glBindVertexArray(I.vao_ground);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(I.ground.size()));
  }
  auto cap_uniforms = [&](GLuint prog) {
    glUseProgram(prog);
    glUniformMatrix4fv(glGetUniformLocation(prog, "u_view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog, "u_proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform2fv(glGetUniformLocation(prog, "u_vp"), 1, glm::value_ptr(vpx));
    glUniform1f(glGetUniformLocation(prog, "u_near"), near_z + 0.05f);
  };
  if (!I.caps.empty()) {
    cap_uniforms(I.p_caps);
    glBindVertexArray(I.vao_caps);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(I.caps.size()));
  }
  if (!I.boxes.empty()) {
    cap_uniforms(I.p_boxes);
    glBindVertexArray(I.vao_boxes);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(I.boxes.size()));
  }
  if (!I.sprites.empty()) {
    glUseProgram(I.p_sprite);
    glUniformMatrix4fv(glGetUniformLocation(I.p_sprite, "u_view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(I.p_sprite, "u_proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform2fv(glGetUniformLocation(I.p_sprite, "u_vp"), 1, glm::value_ptr(vpx));
    glBindVertexArray(I.vao_sprite);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(I.sprites.size()));
  }
  glDisable(GL_DEPTH_TEST);
  if (!I.hud.empty()) {
    glUseProgram(I.p_hud);
    glUniform2fv(glGetUniformLocation(I.p_hud, "u_vp"), 1, glm::value_ptr(vpx));
    glBindVertexArray(I.vao_hud);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(I.hud.size()));
  }
  glDepthMask(GL_TRUE);
  glBindVertexArray(0);

  // resolve MSAA
  glBindFramebuffer(GL_READ_FRAMEBUFFER, I.fbo_ms);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, I.fbo_res);
  glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  glEndQuery(GL_TIME_ELAPSED);
  I.q_pending[qi] = true;
  I.q_frame++;

  if (blit_to_default) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, I.fbo_res);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, w, h, 0, 0, fb_w > 0 ? fb_w : w, fb_h > 0 ? fb_h : h, GL_COLOR_BUFFER_BIT,
                      GL_LINEAR);
  }
  if (out) {
    out->w = w;
    out->h = h;
    out->rgba.resize(static_cast<size_t>(w) * h * 4);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, I.fbo_res);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, out->rgba.data());
    // GL is bottom-up; flip to top-down rows.
    const size_t row = static_cast<size_t>(w) * 4;
    std::vector<uint8_t> tmp(row);
    for (int y = 0; y < h / 2; ++y) {
      uint8_t* a = out->rgba.data() + y * row;
      uint8_t* b = out->rgba.data() + (h - 1 - y) * row;
      std::copy(a, a + row, tmp.data());
      std::copy(b, b + row, a);
      std::copy(tmp.data(), tmp.data() + row, b);
    }
    for (size_t i = 3; i < out->rgba.size(); i += 4) out->rgba[i] = 255;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  return glGetError() == GL_NO_ERROR;
}

}  // namespace gvd
