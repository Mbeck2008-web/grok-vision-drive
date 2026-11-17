// Scene message (schema v2, docs/specs/viz_cpp.md). Perception output only.
#pragma once
#include <chrono>
#include <string>
#include <vector>

namespace gvd {

constexpr int kSchemaVersion = 2;

// Caps are enforced on both the Python publisher and this parser.
constexpr int kMaxTracks = 64;
constexpr int kMaxSigns = 16;
constexpr int kMaxLanes = 16;
constexpr int kMaxEdges = 8;
constexpr int kMaxPoints = 120;
constexpr int kMaxDots = 1024;

// Live mode: no valid datagram for this long blanks the scene (empty road + ego).
constexpr std::chrono::seconds kStaleTimeout{1};

inline bool scene_is_stale(std::chrono::steady_clock::time_point last_rx,
                           std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) {
  return now - last_rx > kStaleTimeout;
}

struct P2 { float x = 0, y = 0; };  // ego frame: x right, y forward (m)

enum class Paint { White, Yellow };

struct Lane {
  std::vector<P2> pts;
  Paint paint = Paint::White;
  bool predicted = false;
};

struct Edge {
  std::vector<P2> pts;
  bool predicted = false;
};

struct Track {
  int id = -1;
  std::string cls = "vehicle";
  float x = 0, y = 0, yaw = 1.5707963f;  // yaw from +x; pi/2 = same heading as ego
  float l = 0, w = 0, h = 0;              // 0 = class prior
  float v = 0;
  bool lead = false;
};

struct Sign {
  std::string cls;   // traffic_light | stop_sign | pole
  float x = 0, y = 0;
  std::string state; // red | yellow | green | unknown
};

// One bead from model / occ_out point output. Not a polyline sample.
// Coordinates are world metres when Scene::has_ego_pose, otherwise ego frame.
enum class DotCls { White, Yellow, Curb, Edge, Path };

struct Dot {
  int id = -1;
  float x = 0, y = 0;
  DotCls cls = DotCls::White;
  float prob = 1.f;    // [0,1] — per-dot alpha; required on the wire
  bool observed = true;  // required on the wire; missing drops the dot
  int i = -1, j = -1, k = -1;  // optional cell ijk for identity
};

struct Scene {
  int v = kSchemaVersion;
  long long seq = 0;
  double t = 0;
  bool engaged = false;
  float speed_mps = 0;
  float set_mps = -1;  // <0 = none
  // Optional world pose of the ego. Set only when x, y and yaw are all
  // present and finite — missing fields are not invented. When set, *dots*
  // are in the world frame and are transformed so the ego stays screen-locked;
  // tracks, signs, lanes and path stay in the ego frame. Live ego x/y/yaw is
  // not wired up yet (UNPROVEN).
  bool has_ego_pose = false;
  float ego_x = 0, ego_y = 0, ego_yaw = 1.5707963f;
  float yaw_rate_rps = 0;
  int lead_id = -1;
  std::string aeb = "off";
  std::vector<Lane> lanes;
  std::vector<Edge> edges;
  std::vector<Dot> dots;
  std::vector<Track> tracks;
  std::vector<Sign> signs;
  std::vector<P2> path;
  float path_conf = 1.0f;
};

// Live loop: pick the scene to draw. A stale feed (no valid datagram for
// more than kStaleTimeout) blanks to an empty scene (empty road + ego).
inline Scene scene_to_show(const Scene& cur,
                           std::chrono::steady_clock::time_point last_rx,
                           std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) {
  return scene_is_stale(last_rx, now) ? Scene{} : cur;
}

// Parse one JSON scene message. Returns false (and fills err) on bad JSON,
// a schema version this build does not speak, or a value that fails a
// strict enum (aeb). Unknown keys are ignored: the parser only reads the
// documented v2 fields, so a stray GT key can never be drawn. Caps above
// are applied here (extra elements / points dropped). Invalid enums on a
// lane, edge, track or sign drop that element, not the whole message.
bool parse_scene(const std::string& text, Scene& out, std::string* err = nullptr);

// Read a JSONL file of scene messages (offline mode). Bad lines are skipped
// and counted in *bad.
std::vector<Scene> load_scene_file(const std::string& path, int* bad = nullptr);

}  // namespace gvd
