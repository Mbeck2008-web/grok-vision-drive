// Scene message (schema v1, docs/specs/viz_cpp.md). Perception output only.
#pragma once
#include <string>
#include <vector>

namespace gvd {

constexpr int kSchemaVersion = 1;

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
  std::string cls;   // traffic_light | stop_sign | pole | ...
  float x = 0, y = 0;
  std::string state; // red | yellow | green | unknown
};

struct Scene {
  int v = kSchemaVersion;
  long long seq = 0;
  double t = 0;
  bool engaged = false;
  float speed_mps = 0;
  float set_mps = -1;  // <0 = none
  int lead_id = -1;
  std::string aeb = "off";
  std::vector<Lane> lanes;
  std::vector<Edge> edges;
  std::vector<Track> tracks;
  std::vector<Sign> signs;
  std::vector<P2> path;
  float path_conf = 1.0f;
};

// Parse one JSON scene message. Returns false (and fills err) on bad JSON or
// a schema version this build does not speak. Unknown keys are ignored: the
// parser only reads the fields above, so a stray GT key can never be drawn.
bool parse_scene(const std::string& text, Scene& out, std::string* err = nullptr);

// Read a JSONL file of scene messages (offline mode). Bad lines are skipped
// and counted in *bad.
std::vector<Scene> load_scene_file(const std::string& path, int* bad = nullptr);

}  // namespace gvd
