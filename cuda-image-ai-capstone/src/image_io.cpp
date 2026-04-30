// image_io.cpp — stb_image / stb_image_write implementation + POSIX dir listing.
// STB_*_IMPLEMENTATION macros must appear in exactly ONE translation unit.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

bool LoadImage(const std::string& path, ImageData* img) {
  int w = 0, h = 0, c = 0;
  uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 0);
  if (!data) {
    fprintf(stderr, "LoadImage: cannot load '%s': %s\n",
            path.c_str(), stbi_failure_reason());
    return false;
  }
  img->width    = w;
  img->height   = h;
  img->channels = c;
  img->path     = path;
  img->pixels.assign(data, data + static_cast<size_t>(w) * h * c);
  stbi_image_free(data);
  return true;
}

bool SaveImage(const std::string& path, const ImageData& img) {
  const int ok = stbi_write_png(
      path.c_str(), img.width, img.height, img.channels,
      img.pixels.data(), img.width * img.channels);
  if (!ok) {
    fprintf(stderr, "SaveImage: cannot write '%s'\n", path.c_str());
    return false;
  }
  return true;
}

std::vector<std::string> ListImages(const std::string& dir_path) {
  std::vector<std::string> paths;
  DIR* dir = opendir(dir_path.c_str());
  if (!dir) {
    fprintf(stderr, "ListImages: cannot open '%s': %s\n",
            dir_path.c_str(), strerror(errno));
    return paths;
  }
  auto HasExt = [](const std::string& name, const std::string& ext) -> bool {
    if (name.size() < ext.size()) return false;
    std::string tail = name.substr(name.size() - ext.size());
    std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);
    return tail == ext;
  };
  struct dirent* e = nullptr;
  while ((e = readdir(dir)) != nullptr) {
    const std::string n(e->d_name);
    if (HasExt(n, ".png") || HasExt(n, ".jpg") || HasExt(n, ".jpeg") ||
        HasExt(n, ".ppm") || HasExt(n, ".bmp") || HasExt(n, ".tga")) {
      paths.push_back(dir_path + "/" + n);
    }
  }
  closedir(dir);
  std::sort(paths.begin(), paths.end());
  return paths;
}
