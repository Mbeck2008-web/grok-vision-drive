// Live-dot fade and identity. Header-only so test_render.cpp can assert
// alphas, match results and fade timing without going through the GPU.
// Production has no mutation flags; the constants below are the spec.
#pragma once

#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gvd {

// Fade length is independent of target alpha and must stay in [150, 250] ms.
constexpr float kDotFadeSec = 0.20f;
// NN radius after ego-motion compensation (or in world frame).
constexpr float kDotMatchRadius = 0.75f;
constexpr float kInferredScale = 0.45f;
constexpr float kDotAliveEps = 0.004f;
// First-frame marker and the gap / repeat policy live in step_live_dots.
constexpr double kDotFirstT = -1e8;

struct LiveDot {
  int id = -1;
  int i = -1, j = -1, k = -1;
  float wx = 0, wy = 0;
  DotCls cls = DotCls::White;
  float target = 0;  // desired alpha (prob, times kInferredScale if unobserved)
  float fade = 0;    // 0..1 progress; duration is kDotFadeSec regardless of target
  float alpha = 0;   // fade * target
};

inline float target_of(const Dot& d) {
  float a = std::clamp(d.prob, 0.f, 1.f);
  if (!d.observed) a *= kInferredScale;
  return a;
}

// Warp an ego-frame point into the current ego frame after the ego has moved
// forward by speed*dt (along +y) and yawed CCW by yaw_rate*dt.
inline P2 ego_motion_warp(float x, float y, float speed_mps, float yaw_rate_rps, float dt) {
  const float s = speed_mps * dt;
  const float w = yaw_rate_rps * dt;
  const float c = std::cos(w), si = std::sin(w);
  const float xt = x, yt = y - s;
  return {xt * c + yt * si, -xt * si + yt * c};
}

// Inverse of world_to_ego: ego-frame (ex, ey) -> world, given a complete pose.
inline P2 ego_to_world(float ex, float ey, const Scene& s) {
  if (!s.has_ego_pose) return {ex, ey};
  const float c = std::cos(s.ego_yaw), si = std::sin(s.ego_yaw);
  const float dx = ex * si + ey * c;
  const float dy = -ex * c + ey * si;
  return {s.ego_x + dx, s.ego_y + dy};
}

inline P2 world_to_ego_pose(float wx, float wy, const Scene& s) {
  if (!s.has_ego_pose) return {wx, wy};
  const float dx = wx - s.ego_x, dy = wy - s.ego_y;
  const float c = std::cos(s.ego_yaw), si = std::sin(s.ego_yaw);
  return {dx * si - dy * c, dx * c + dy * si};
}

// Advance live dots to `now`.
//
// * First call (last_t < kDotFirstT): snap fade=1, alpha=target.
// * Still frames (now == 0, the gallery / ctest convention): reset and snap.
//   Existing mid-fade live state from a previous still must not leak.
// * Repeated / ~0 / negative dt on a live clock (now > 0): rematch, hold fade,
//   do not pop. New unmatched dots start at fade 0 (visible on the next tick).
// * Gap > 0.5 s: rematch, hold fade (no pop). Motion is not applied for that
//   interval (we don't invent a coast).
// * Otherwise: fade_step = dt / kDotFadeSec and, when the incoming points are
//   in the ego frame, previous live positions are ego-motion-compensated
//   before nearest-neighbour matching.
//
// Matching order: stable id, then cell ijk, then NN within kDotMatchRadius.
inline void step_live_dots(std::vector<LiveDot>& live, const std::vector<Dot>& incoming, double now,
                           double& last_t, float speed_mps, float yaw_rate_rps, bool world_frame) {
  const bool first = last_t < kDotFirstT;
  const bool still = now <= 1e-12;  // gallery / ctest fixtures all use t=0
  const double raw_dt = now - last_t;
  if (first || still) {
    live.clear();
    last_t = now;
    for (const auto& d : incoming) {
      LiveDot L;
      L.id = d.id;
      L.i = d.i;
      L.j = d.j;
      L.k = d.k;
      L.wx = d.x;
      L.wy = d.y;
      L.cls = d.cls;
      L.target = target_of(d);
      L.fade = 1.f;
      L.alpha = L.target;
      if (L.alpha > kDotAliveEps || L.target > kDotAliveEps) live.push_back(L);
    }
    return;
  }
  double dt_fade = 0.0;
  double dt_motion = 0.0;
  if (raw_dt > 1e-6 && raw_dt <= 0.5) {
    dt_fade = raw_dt;
    dt_motion = raw_dt;
  }
  last_t = now;

  const float fade_step = (kDotFadeSec > 1e-4f) ? static_cast<float>(dt_fade / kDotFadeSec) : 1.f;

  std::vector<char> used(incoming.size(), 0);
  std::vector<LiveDot> next;
  next.reserve(incoming.size() + live.size());

  auto predict = [&](const LiveDot& L) -> P2 {
    if (world_frame || dt_motion <= 0.0) return {L.wx, L.wy};
    return ego_motion_warp(L.wx, L.wy, speed_mps, yaw_rate_rps, static_cast<float>(dt_motion));
  };

  auto match = [&](const LiveDot& L) -> int {
    if (L.id >= 0) {
      for (size_t i = 0; i < incoming.size(); ++i) {
        if (used[i]) continue;
        if (incoming[i].id == L.id) return static_cast<int>(i);
      }
    }
    if (L.i >= 0) {
      for (size_t i = 0; i < incoming.size(); ++i) {
        if (used[i]) continue;
        const Dot& d = incoming[i];
        if (d.i == L.i && d.j == L.j && d.k == L.k && d.i >= 0) return static_cast<int>(i);
      }
    }
    const P2 p = predict(L);
    float best_d = kDotMatchRadius;
    int best = -1;
    for (size_t i = 0; i < incoming.size(); ++i) {
      if (used[i]) continue;
      const Dot& d = incoming[i];
      if (L.id >= 0 && d.id >= 0 && d.id != L.id) continue;
      if (L.i >= 0 && d.i >= 0 && (d.i != L.i || d.j != L.j || d.k != L.k)) continue;
      const float dx = d.x - p.x, dy = d.y - p.y;
      const float dist = std::sqrt(dx * dx + dy * dy);
      if (dist < best_d) {
        best_d = dist;
        best = static_cast<int>(i);
      }
    }
    return best;
  };

  for (auto& L : live) {
    const int best = match(L);
    if (best >= 0) {
      const Dot& d = incoming[static_cast<size_t>(best)];
      used[static_cast<size_t>(best)] = 1;
      L.id = d.id;
      L.i = d.i;
      L.j = d.j;
      L.k = d.k;
      L.wx = d.x;
      L.wy = d.y;
      L.cls = d.cls;
      L.target = target_of(d);
      if (first) L.fade = 1.f;
      else L.fade = std::min(1.f, L.fade + fade_step);
      L.alpha = L.fade * L.target;
      next.push_back(L);
    } else if (!first) {
      L.fade = std::max(0.f, L.fade - fade_step);
      L.alpha = L.fade * L.target;
      if (L.fade > kDotAliveEps) next.push_back(L);
    }
  }
  for (size_t i = 0; i < incoming.size(); ++i) {
    if (used[i]) continue;
    const Dot& d = incoming[i];
    LiveDot L;
    L.id = d.id;
    L.i = d.i;
    L.j = d.j;
    L.k = d.k;
    L.wx = d.x;
    L.wy = d.y;
    L.cls = d.cls;
    L.target = target_of(d);
    if (first) {
      L.fade = 1.f;
    } else {
      L.fade = std::min(1.f, fade_step);
    }
    L.alpha = L.fade * L.target;
    if (L.alpha > kDotAliveEps || L.target > kDotAliveEps) next.push_back(L);
  }
  live.swap(next);
}

}  // namespace gvd
