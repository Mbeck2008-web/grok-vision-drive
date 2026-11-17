#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gvd {
struct Image {
  int w = 0, h = 0;              // RGBA8, top row first
  std::vector<uint8_t> rgba;
};
bool write_png(const std::string& path, const Image& img);
bool read_image(const std::string& path, Image& out);  // png/jpg -> RGBA8
}  // namespace gvd
