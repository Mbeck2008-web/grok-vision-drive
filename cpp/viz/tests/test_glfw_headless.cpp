// ctest: GlContext::init_headless via the GLFW fallback (no EGL).
// This is the Windows / no-EGL path. CI's main render test uses EGL first,
// so it never executes this branch; this binary is compiled without EGL
// so the fallback is what runs. SKIP_RETURN_CODE 77 if no display.
#include <cstdio>
#include <string>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "context.hpp"

int main() {
  if (gvd::headless_glfw_visible()) {
    std::printf("FAIL headless_glfw_visible() is true (must stay false)\n");
    return 1;
  }
  gvd::GlContext ctx;
  std::string err;
  if (!ctx.init_headless(&err)) {
    std::printf("SKIP glfw init_headless (no display?): %s\n", err.c_str());
    return 77;
  }
  if (ctx.window() == nullptr) {
    std::printf("FAIL init_headless succeeded but window() is null\n");
    return 1;
  }
  const int visible = glfwGetWindowAttrib(ctx.window(), GLFW_VISIBLE);
  if (visible != GLFW_FALSE) {
    std::printf("FAIL init_headless window is visible (GLFW_VISIBLE=%d); "
                "must pass visible=headless_glfw_visible() after glfwInit\n",
                visible);
    return 1;
  }
  std::printf("PASS glfw init_headless: window hidden (GLFW_VISIBLE=0)\n");
  return 0;
}
