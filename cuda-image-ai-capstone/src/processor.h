// processor.h — CapstoneProcessor: multi-stream GPU batch image processor.
#ifndef CAPSTONE_SRC_PROCESSOR_H_
#define CAPSTONE_SRC_PROCESSOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "image_io.h"

class CapstoneProcessor {
 public:
  enum class FilterType {
    kGrayscale,
    kGaussianBlur,
    kSobelEdge,
    kHistogramEqualize,
    kPipeline,
    kSepGaussBlur,
    kUnsharpMask,
    kCanny,
    kMotionBlur,
    kAutoLevels,
    kEmboss,
    kVignette,
    kBilateral,
    kAll,
  };

  struct Stats {
    int    images_processed = 0;
    double total_ms         = 0.0;
    double avg_ms           = 0.0;
    size_t total_bytes_in   = 0;
  };

  explicit CapstoneProcessor(int num_streams = 4);
  ~CapstoneProcessor();

  CapstoneProcessor(const CapstoneProcessor&)            = delete;
  CapstoneProcessor& operator=(const CapstoneProcessor&) = delete;

  void AllocateForBatch(const std::vector<ImageData>& images);

  Stats ProcessBatch(const std::vector<ImageData>& inputs,
                     FilterType filter,
                     std::vector<ImageData>* outputs);

  static std::string FilterName(FilterType f);
  static FilterType  ParseFilter(const std::string& name);
  static std::vector<FilterType> AllFilters();

  struct StreamCtx {
    cudaStream_t stream      = nullptr;
    uint8_t* d_in            = nullptr;
    uint8_t* d_out           = nullptr;
    uint8_t* d_work          = nullptr;
    float*   d_float_a       = nullptr;
    float*   d_float_b       = nullptr;
    float*   d_gx            = nullptr;
    float*   d_gy            = nullptr;
    float*   d_mag           = nullptr;
    float*   d_angle         = nullptr;
    float*   d_nms           = nullptr;
    int*     d_histogram     = nullptr;
    int*     d_minmax        = nullptr;
    uint8_t* h_pin_in        = nullptr;
    uint8_t* h_pin_out       = nullptr;
    size_t   buf_bytes       = 0;
    size_t   float_elems     = 0;
  };

 private:
  int                      num_streams_;
  std::vector<StreamCtx>   ctxs_;

  void FreeBuffers();
  static void ToGray(const ImageData& in, StreamCtx* c);

  void DispatchGrayscale (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchBlur      (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchEdge      (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchEqualize  (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchPipeline  (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchSepGauss  (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchUnsharp   (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchCanny     (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchMotionBlur(const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchAutoLevels(const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchEmboss    (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchVignette  (const ImageData& in, ImageData* out, StreamCtx* c);
  void DispatchBilateral (const ImageData& in, ImageData* out, StreamCtx* c);
};

#endif  // CAPSTONE_SRC_PROCESSOR_H_
