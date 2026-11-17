// GL 3.3 core context: headless EGL (Linux/CI) or a GLFW window (live / Windows).
#pragma once
#include <string>

struct GLFWwindow;

namespace gvd {

class GlContext {
 public:
  ~GlContext();
  // Offscreen context. EGL surfaceless when built with EGL, else a hidden GLFW window.
  bool init_headless(std::string* err);
  // Visible window (live mode). Needs GLFW.
  bool init_window(int w, int h, const char* title, bool vsync, std::string* err);
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
