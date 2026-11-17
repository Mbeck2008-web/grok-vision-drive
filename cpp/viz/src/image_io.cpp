#include "image_io.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

namespace gvd {
bool write_png(const std::string& path, const Image& img) {
  if (img.w <= 0 || img.h <= 0) return false;
  return stbi_write_png(path.c_str(), img.w, img.h, 4, img.rgba.data(), img.w * 4) != 0;
}
bool read_image(const std::string& path, Image& out) {
  int w = 0, h = 0, n = 0;
  unsigned char* p = stbi_load(path.c_str(), &w, &h, &n, 4);
  if (!p) return false;
  out.w = w;
  out.h = h;
  out.rgba.assign(p, p + static_cast<size_t>(w) * h * 4);
  stbi_image_free(p);
  return true;
}
}  // namespace gvd
