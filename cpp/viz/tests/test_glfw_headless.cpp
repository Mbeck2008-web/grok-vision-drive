// ctest: GlContext::init_headless via the GLFW fallback (no EGL).
// This is the Windows / no-EGL path. CI's main render test uses EGL first,
// so it never executes this branch; this binary is compiled without EGL
// so the fallback is what runs. SKIP_RETURN_CODE 77 if no display, unless
// GVD_REQUIRE_LIVE is set (CI): then an init_headless failure is FAIL (1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "context.hpp"

static bool require_live() {
  const char* v = std::getenv("GVD_REQUIRE_LIVE");
  if (!v || !*v) return false;
  return std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 &&
         std::strcmp(v, "no") != 0 && std::strcmp(v, "off") != 0;
}

int main() {
  if (gvd::headless_glfw_visible()) {
    std::printf("FAIL headless_glfw_visible() is true (must stay false)\n");
    return 1;
  }
  gvd::GlContext ctx;
  std::string err;
  if (!ctx.init_headless(&err)) {
    if (require_live()) {
      std::printf("FAIL glfw init_headless (GVD_REQUIRE_LIVE=1): %s\n", err.c_str());
      return 1;
    }
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
  std::printf("PASS glfw init_headless: window hidden (GLFW_VISIBLE=0) "
              "GVD_REQUIRE_LIVE=%s\n",
              require_live() ? "1" : "0");
  return 0;
}
