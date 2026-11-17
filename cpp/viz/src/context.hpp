// GL 3.3 core context: headless EGL (Linux/CI) or a GLFW window (live / Windows).
#pragma once
#include <string>

struct GLFWwindow;

namespace gvd {

// Visibility for a GLFW window used as the headless/offscreen fallback.
// Must stay false: a visible first-run window was the original Windows bug.
// init_headless passes this to init_window after glfwInit (GLFW 3.4 drops
// pre-init hints, and glfwInit resets them).
inline bool headless_glfw_visible() { return false; }

class GlContext {
 public:
  ~GlContext();
  // Offscreen context. EGL surfaceless when built with EGL, else a hidden GLFW
  // window (GLFW_VISIBLE is set after glfwInit; GLFW 3.4 drops pre-init hints).
  bool init_headless(std::string* err);
  // Window (live mode, or hidden offscreen fallback). Needs GLFW.
  // `visible` is applied with glfwWindowHint after glfwInit (init resets hints).
  bool init_window(int w, int h, const char* title, bool vsync, std::string* err,
                   bool visible = true);
  bool should_close() const;
  void swap_and_poll();
  void framebuffer_size(int* w, int* h) const;
  bool key_down(int glfw_key) const;
  std::string renderer_string() const;
  GLFWwindow* window() const { return win_; }

 private:
  void* egl_display_ = nullptr;
  void* egl_context_ = nullptr;
  GLFWwindow* win_ = nullptr;
  [[maybe_unused]] bool glfw_inited_ = false;
};

}  // namespace gvd
