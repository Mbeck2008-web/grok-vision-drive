#include "context.hpp"

#include <glad/gl.h>
#if GVD_VIZ_HAVE_EGL
#include <glad/egl.h>
#endif
#if GVD_VIZ_HAVE_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

namespace gvd {

GlContext::~GlContext() {
#if GVD_VIZ_HAVE_EGL
  if (egl_display_) {
    EGLDisplay d = static_cast<EGLDisplay>(egl_display_);
    eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (egl_context_) eglDestroyContext(d, static_cast<EGLContext>(egl_context_));
    eglTerminate(d);
  }
#endif
#if GVD_VIZ_HAVE_GLFW
  if (win_) glfwDestroyWindow(win_);
  if (glfw_inited_) glfwTerminate();
#endif
}

#if GVD_VIZ_HAVE_EGL
static bool egl_headless(void** out_dpy, void** out_ctx, std::string* err) {
  if (!gladLoaderLoadEGL(nullptr)) {
    if (err) *err = "libEGL not found";
    return false;
  }
  EGLDisplay dpy = EGL_NO_DISPLAY;
  if (eglGetPlatformDisplayEXT) {
    dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
  }
  if (dpy == EGL_NO_DISPLAY) dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLint maj = 0, min = 0;
  if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &maj, &min)) {
    if (err) *err = "eglInitialize failed";
    return false;
  }
  gladLoaderLoadEGL(dpy);
  if (!eglBindAPI(EGL_OPENGL_API)) {
    if (err) *err = "eglBindAPI(OpenGL) failed";
    return false;
  }
  const EGLint cfg_attr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                             EGL_BLUE_SIZE, 8, EGL_NONE};
  EGLConfig cfg = nullptr;
  EGLint n = 0;
  if (!eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n) || n < 1) {
    // Surfaceless Mesa may expose no RGB-sized config; a config is optional
    // with EGL_KHR_no_config_context, try without one.
    cfg = nullptr;
  }
  const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3,
                             EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
  EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
  if (ctx == EGL_NO_CONTEXT) {
    if (err) *err = "eglCreateContext(GL 3.3 core) failed";
    eglTerminate(dpy);
    return false;
  }
  if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
    if (err) *err = "eglMakeCurrent(surfaceless) failed";
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    return false;
  }
  if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(eglGetProcAddress))) {
    if (err) *err = "gladLoadGL failed";
    return false;
  }
  *out_dpy = dpy;
  *out_ctx = ctx;
  return true;
}
#endif

bool GlContext::init_headless(std::string* err) {
#if GVD_VIZ_HAVE_EGL
  if (egl_headless(&egl_display_, &egl_context_, err)) return true;
#endif
#if GVD_VIZ_HAVE_GLFW
  // Do not set hints here: GLFW 3.4 ignores pre-init hints, and glfwInit
  // resets them. init_window sets GLFW_VISIBLE after a successful init.
  return init_window(64, 64, "gvd_viz offscreen", false, err, /*visible=*/false);
#else
  if (err && err->empty()) *err = "no headless backend (build with EGL or GLFW)";
  return false;
#endif
}

bool GlContext::init_window(int w, int h, const char* title, bool vsync, std::string* err,
                               bool visible) {
#if GVD_VIZ_HAVE_GLFW
  if (!glfw_inited_) {
    if (!glfwInit()) {
      if (err) *err = "glfwInit failed";
      return false;
    }
    glfw_inited_ = true;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  // After glfwInit only: 3.4 drops hints set before init, and init resets them.
  glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
  win_ = glfwCreateWindow(w, h, title, nullptr, nullptr);
  if (!win_) {
    if (err) *err = "glfwCreateWindow(GL 3.3 core) failed";
    return false;
  }
  glfwMakeContextCurrent(win_);
  glfwSwapInterval(vsync ? 1 : 0);
  if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress))) {
    if (err) *err = "gladLoadGL failed";
    return false;
  }
  return true;
#else
  (void)w; (void)h; (void)title; (void)vsync; (void)visible;
  if (err) *err = "built without GLFW (no window mode)";
  return false;
#endif
}

bool GlContext::should_close() const {
#if GVD_VIZ_HAVE_GLFW
  return win_ ? glfwWindowShouldClose(win_) != 0 : false;
#else
  return false;
#endif
}

void GlContext::swap_and_poll() {
#if GVD_VIZ_HAVE_GLFW
  if (win_) {
    glfwSwapBuffers(win_);
    glfwPollEvents();
  }
#endif
}

void GlContext::framebuffer_size(int* w, int* h) const {
  *w = 0; *h = 0;
#if GVD_VIZ_HAVE_GLFW
  if (win_) glfwGetFramebufferSize(win_, w, h);
#endif
}

bool GlContext::key_down(int key) const {
#if GVD_VIZ_HAVE_GLFW
  return win_ && glfwGetKey(win_, key) == GLFW_PRESS;
#else
  (void)key;
  return false;
#endif
}

std::string GlContext::renderer_string() const {
  const char* r = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
  const char* v = reinterpret_cast<const char*>(glGetString(GL_VERSION));
  return std::string(r ? r : "?") + " | " + (v ? v : "?");
}

}  // namespace gvd
