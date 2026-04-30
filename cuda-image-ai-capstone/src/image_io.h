// image_io.h — Image load / save / directory-listing declarations.
#ifndef CAPSTONE_SRC_IMAGE_IO_H_
#define CAPSTONE_SRC_IMAGE_IO_H_

#include <cstdint>
#include <string>
#include <vector>

struct ImageData {
  std::vector<uint8_t> pixels;
  int width    = 0;
  int height   = 0;
  int channels = 0;
  std::string path;

  size_t NumBytes() const {
    return static_cast<size_t>(width) * height * channels;
  }
};

bool LoadImage(const std::string& path, ImageData* img);
bool SaveImage(const std::string& path, const ImageData& img);
std::vector<std::string> ListImages(const std::string& dir_path);

#endif  // CAPSTONE_SRC_IMAGE_IO_H_
