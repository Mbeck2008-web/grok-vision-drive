#include "scene.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace gvd {
using nlohmann::json;

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

static std::vector<P2> pts_of(const json& j) {
  std::vector<P2> out;
  auto it = j.find("pts");
  if (it == j.end() || !it->is_array()) return out;
  out.reserve(it->size());
  for (const auto& p : *it) {
    if (!p.is_array() || p.size() < 2 || !p[0].is_number() || !p[1].is_number()) continue;
    double x = p[0].get<double>(), y = p[1].get<double>();
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    out.push_back({static_cast<float>(x), static_cast<float>(y)});
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
  if (auto it = j.find("seq"); it != j.end() && it->is_number_integer()) s.seq = it->get<long long>();
  if (auto it = j.find("t"); it != j.end() && it->is_number()) s.t = it->get<double>();
  if (auto it = j.find("ego"); it != j.end() && it->is_object()) {
    s.engaged = boolean(*it, "engaged", false);
    s.speed_mps = num(*it, "speed_mps", 0.f);
    s.set_mps = num(*it, "set_mps", -1.f);
  }
  if (auto it = j.find("lead_id"); it != j.end() && it->is_number_integer()) s.lead_id = it->get<int>();
  s.aeb = str(j, "aeb", "off");
  s.path_conf = num(j, "path_conf", 1.f);
  if (auto it = j.find("lanes"); it != j.end() && it->is_array()) {
    for (const auto& l : *it) {
      if (!l.is_object()) continue;
      Lane lane;
      lane.pts = pts_of(l);
      lane.paint = str(l, "paint", "white") == "yellow" ? Paint::Yellow : Paint::White;
      lane.predicted = str(l, "kind", "detected") == "predicted";
      if (lane.pts.size() >= 2) s.lanes.push_back(std::move(lane));
    }
  }
  if (auto it = j.find("edges"); it != j.end() && it->is_array()) {
    for (const auto& e : *it) {
      if (!e.is_object()) continue;
      Edge edge;
      edge.pts = pts_of(e);
      edge.predicted = str(e, "kind", "detected") == "predicted";
      if (edge.pts.size() >= 2) s.edges.push_back(std::move(edge));
    }
  }
  if (auto it = j.find("tracks"); it != j.end() && it->is_array()) {
    for (const auto& t : *it) {
      if (!t.is_object()) continue;
      Track tr;
      if (auto id = t.find("id"); id != t.end() && id->is_number_integer()) tr.id = id->get<int>();
      tr.cls = str(t, "cls", "vehicle");
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
      if (!g.is_object()) continue;
      Sign sg;
      sg.cls = str(g, "cls", "");
      sg.x = num(g, "x", NAN);
      sg.y = num(g, "y", NAN);
      sg.state = str(g, "state", "unknown");
      if (!std::isfinite(sg.x) || !std::isfinite(sg.y) || sg.cls.empty()) continue;
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
