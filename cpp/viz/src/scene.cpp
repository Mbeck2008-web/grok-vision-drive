#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace gvd {
using nlohmann::json;

static bool one_of(const std::string& s, std::initializer_list<const char*> xs) {
  for (const char* x : xs) if (s == x) return true;
  return false;
}

static bool known_track_cls(const std::string& s) {
  return one_of(s, {"vehicle", "car", "pedestrian", "ped", "person", "bicycle", "bike", "cyclist",
                    "motorcycle", "truck", "bus", "trailer", "obstacle", "parked", "debris", "cone"});
}

static bool known_sign_cls(const std::string& s) {
  return one_of(s, {"traffic_light", "stop_sign", "pole"});
}

static float num(const json& j, const char* k, float dflt) {
  auto it = j.find(k);
  if (it == j.end() || !it->is_number()) return dflt;
  double d = it->get<double>();
  return std::isfinite(d) ? static_cast<float>(d) : dflt;
}

static std::string str(const json& j, const char* k, const char* dflt) {
  auto it = j.find(k);
  if (it == j.end() || !it->is_string()) return dflt;
  return it->get<std::string>();
}

static bool boolean(const json& j, const char* k, bool dflt) {
  auto it = j.find(k);
  if (it == j.end() || !it->is_boolean()) return dflt;
  return it->get<bool>();
}

// Strict integer: JSON integer only, no floats, no strings, no bools, fits int.
static bool read_int(const json& j, const char* k, int& out) {
  auto it = j.find(k);
  if (it == j.end() || !it->is_number_integer()) return false;
  std::int64_t v = it->get<std::int64_t>();
  if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false;
  out = static_cast<int>(v);
  return true;
}

static bool read_int64(const json& j, const char* k, long long& out) {
  auto it = j.find(k);
  if (it == j.end() || !it->is_number_integer()) return false;
  out = it->get<long long>();
  return true;
}

// kind: missing -> detected; "detected"/"predicted" ok; anything else invalid.
static bool read_kind(const json& j, bool& predicted) {
  auto it = j.find("kind");
  if (it == j.end()) {
    predicted = false;
    return true;
  }
  if (!it->is_string()) return false;
  const std::string k = it->get<std::string>();
  if (k == "predicted") {
    predicted = true;
    return true;
  }
  if (k == "detected") {
    predicted = false;
    return true;
  }
  return false;
}

static std::vector<P2> pts_of(const json& j) {
  std::vector<P2> out;
  auto it = j.find("pts");
  if (it == j.end() || !it->is_array()) return out;
  out.reserve(std::min(it->size(), static_cast<size_t>(kMaxPoints)));
  for (const auto& p : *it) {
    if (!p.is_array() || p.size() < 2 || !p[0].is_number() || !p[1].is_number()) continue;
    double x = p[0].get<double>(), y = p[1].get<double>();
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    out.push_back({static_cast<float>(x), static_cast<float>(y)});
  }
  if (out.size() > static_cast<size_t>(kMaxPoints)) {
    P2 last = out.back();
    out.resize(static_cast<size_t>(kMaxPoints - 1));
    out.push_back(last);
  }
  return out;
}

bool parse_scene(const std::string& text, Scene& out, std::string* err) {
  json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) {
    if (err) *err = "not a JSON object";
    return false;
  }
  if (str(j, "type", "") != "gvd_scene") {
    if (err) *err = "type is not gvd_scene";
    return false;
  }
  auto vit = j.find("v");
  if (vit == j.end() || !vit->is_number_integer() || vit->get<int>() != kSchemaVersion) {
    if (err) *err = "unsupported schema version";
    return false;
  }
  Scene s;
  s.v = kSchemaVersion;
  read_int64(j, "seq", s.seq);
  if (auto it = j.find("t"); it != j.end() && it->is_number()) {
    double t = it->get<double>();
    if (std::isfinite(t)) s.t = t;
  }
  if (auto it = j.find("ego"); it != j.end() && it->is_object()) {
    s.engaged = boolean(*it, "engaged", false);
    s.speed_mps = num(*it, "speed_mps", 0.f);
    s.set_mps = num(*it, "set_mps", -1.f);
  }
  read_int(j, "lead_id", s.lead_id);
  if (auto it = j.find("aeb"); it != j.end()) {
    if (!it->is_string()) {
      if (err) *err = "aeb must be a string";
      return false;
    }
    s.aeb = it->get<std::string>();
    if (!one_of(s.aeb, {"off", "warn", "brake"})) {
      if (err) *err = "aeb must be off|warn|brake";
      return false;
    }
  } else {
    s.aeb = "off";
  }
  {
    float pc = num(j, "path_conf", 1.f);
    s.path_conf = std::clamp(pc, 0.f, 1.f);
  }
  if (auto it = j.find("lanes"); it != j.end() && it->is_array()) {
    for (const auto& l : *it) {
      if (s.lanes.size() >= static_cast<size_t>(kMaxLanes)) break;
      if (!l.is_object()) continue;
      Lane lane;
      lane.pts = pts_of(l);
      auto pit = l.find("paint");
      if (pit == l.end()) {
        lane.paint = Paint::White;
      } else if (!pit->is_string()) {
        continue;
      } else {
        const std::string ps = pit->get<std::string>();
        if (ps == "yellow") lane.paint = Paint::Yellow;
        else if (ps == "white") lane.paint = Paint::White;
        else continue;
      }
      if (!read_kind(l, lane.predicted)) continue;
      if (lane.pts.size() >= 2) s.lanes.push_back(std::move(lane));
    }
  }
  if (auto it = j.find("edges"); it != j.end() && it->is_array()) {
    for (const auto& e : *it) {
      if (s.edges.size() >= static_cast<size_t>(kMaxEdges)) break;
      if (!e.is_object()) continue;
      Edge edge;
      edge.pts = pts_of(e);
      if (!read_kind(e, edge.predicted)) continue;
      if (edge.pts.size() >= 2) s.edges.push_back(std::move(edge));
    }
  }
  if (auto it = j.find("tracks"); it != j.end() && it->is_array()) {
    for (const auto& t : *it) {
      if (s.tracks.size() >= static_cast<size_t>(kMaxTracks)) break;
      if (!t.is_object()) continue;
      Track tr;
      if (t.contains("id") && !read_int(t, "id", tr.id)) continue;
      if (t.contains("cls")) {
        if (!t["cls"].is_string()) continue;
        tr.cls = t["cls"].get<std::string>();
        if (!known_track_cls(tr.cls)) continue;
      } else {
        tr.cls = "vehicle";
      }
      tr.x = num(t, "x", NAN);
      tr.y = num(t, "y", NAN);
      if (!std::isfinite(tr.x) || !std::isfinite(tr.y)) continue;
      tr.yaw = num(t, "yaw", 1.5707963f);
      tr.l = num(t, "l", 0.f);
      tr.w = num(t, "w", 0.f);
      tr.h = num(t, "h", 0.f);
      tr.v = num(t, "v", 0.f);
      tr.lead = boolean(t, "lead", false);
      s.tracks.push_back(std::move(tr));
    }
  }
  if (auto it = j.find("signs"); it != j.end() && it->is_array()) {
    for (const auto& g : *it) {
      if (s.signs.size() >= static_cast<size_t>(kMaxSigns)) break;
      if (!g.is_object()) continue;
      Sign sg;
      sg.cls = str(g, "cls", "");
      if (!known_sign_cls(sg.cls)) continue;
      sg.x = num(g, "x", NAN);
      sg.y = num(g, "y", NAN);
      if (g.contains("state")) {
        if (!g["state"].is_string()) continue;
        sg.state = g["state"].get<std::string>();
        if (!one_of(sg.state, {"red", "yellow", "green", "unknown"})) continue;
      } else {
        sg.state = "unknown";
      }
      if (!std::isfinite(sg.x) || !std::isfinite(sg.y)) continue;
      s.signs.push_back(std::move(sg));
    }
  }
  if (auto it = j.find("path"); it != j.end() && it->is_object()) s.path = pts_of(*it);
  out = std::move(s);
  return true;
}

std::vector<Scene> load_scene_file(const std::string& path, int* bad) {
  std::vector<Scene> out;
  int nbad = 0;
  std::ifstream f(path, std::ios::binary);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find_first_not_of(" \t") == std::string::npos) continue;
    Scene s;
    if (parse_scene(line, s)) out.push_back(std::move(s));
    else ++nbad;
  }
  if (bad) *bad = nbad;
  return out;
}

}  // namespace gvd
