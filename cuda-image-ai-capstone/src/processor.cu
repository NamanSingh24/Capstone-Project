// processor.cpp — CapstoneProcessor implementation.

#include "processor.h"
#include "kernels.cuh"
#include "cuda_utils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace std::chrono;

static dim3 Grid2D(int w, int h) {
    return dim3((w + BLOCK_W - 1) / BLOCK_W, (h + BLOCK_H - 1) / BLOCK_H);
}
static dim3 Block2D() { return dim3(BLOCK_W, BLOCK_H); }
static dim3 Grid1D(int n, int bs = 256) { return dim3((n + bs - 1) / bs); }
static dim3 Block1D(int bs = 256) { return dim3(bs); }

void CapstoneProcessor::ToGray(const ImageData& in, StreamCtx* c) {
    int w = in.width, h = in.height;
    if (in.channels == 1) {
        CUDA_CHECK(cudaMemcpyAsync(c->d_work, c->d_in,
                                  static_cast<size_t>(w) * h,
                                  cudaMemcpyDeviceToDevice, c->stream));
    } else {
        GrayscaleKernel<<<Grid2D(w, h), Block2D(), 0, c->stream>>>(
            c->d_in, c->d_work, w, h, in.channels);
        CUDA_CHECK(cudaGetLastError());
    }
}

CapstoneProcessor::CapstoneProcessor(int num_streams) : num_streams_(num_streams) {
    ctxs_.resize(num_streams_);
    for (auto& c : ctxs_)
        CUDA_CHECK(cudaStreamCreate(&c.stream));
}

CapstoneProcessor::~CapstoneProcessor() {
    FreeBuffers();
    for (auto& c : ctxs_)
        if (c.stream) { cudaStreamDestroy(c.stream); c.stream = nullptr; }
}

void CapstoneProcessor::FreeBuffers() {
    for (auto& c : ctxs_) {
        auto cf = [](auto*& p) { if (p) { cudaFree(p);     p = nullptr; } };
        auto hf = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        cf(c.d_in); cf(c.d_out); cf(c.d_work);
        cf(c.d_float_a); cf(c.d_float_b);
        cf(c.d_gx); cf(c.d_gy); cf(c.d_mag); cf(c.d_angle); cf(c.d_nms);
        cf(c.d_histogram); cf(c.d_minmax);
        hf(c.h_pin_in); hf(c.h_pin_out);
        c.buf_bytes = 0; c.float_elems = 0;
    }
}

void CapstoneProcessor::AllocateForBatch(const std::vector<ImageData>& images) {
    size_t max_bytes = 0, max_elems = 0;
    for (const auto& img : images) {
        max_bytes = std::max(max_bytes, img.NumBytes());
        max_elems = std::max(max_elems, static_cast<size_t>(img.width) * img.height);
    }
    if (max_bytes == 0) return;
    if (!ctxs_.empty() &&
        ctxs_[0].buf_bytes   >= max_bytes &&
        ctxs_[0].float_elems >= max_elems) return;

    FreeBuffers();
    const size_t fb = max_elems * sizeof(float);
    for (auto& c : ctxs_) {
        CUDA_CHECK(cudaMalloc(&c.d_in,        max_bytes));
        CUDA_CHECK(cudaMalloc(&c.d_out,       max_bytes));
        CUDA_CHECK(cudaMalloc(&c.d_work,      max_elems));
        CUDA_CHECK(cudaMalloc(&c.d_float_a,   fb));
        CUDA_CHECK(cudaMalloc(&c.d_float_b,   fb));
        CUDA_CHECK(cudaMalloc(&c.d_gx,        fb));
        CUDA_CHECK(cudaMalloc(&c.d_gy,        fb));
        CUDA_CHECK(cudaMalloc(&c.d_mag,       fb));
        CUDA_CHECK(cudaMalloc(&c.d_angle,     fb));
        CUDA_CHECK(cudaMalloc(&c.d_nms,       fb));
        CUDA_CHECK(cudaMalloc(&c.d_histogram, 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&c.d_minmax,      2 * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&c.h_pin_in,  max_bytes));
        CUDA_CHECK(cudaMallocHost(&c.h_pin_out, max_bytes));
        c.buf_bytes = max_bytes; c.float_elems = max_elems;
    }
}

CapstoneProcessor::Stats CapstoneProcessor::ProcessBatch(
    const std::vector<ImageData>& inputs,
    FilterType filter,
    std::vector<ImageData>* outputs) {

    Stats stats;
    const int N = static_cast<int>(inputs.size());
    outputs->resize(N);
    const auto t0 = high_resolution_clock::now();

    for (int base = 0; base < N; base += num_streams_) {
        const int count = std::min(num_streams_, N - base);

        for (int i = 0; i < count; ++i) {
            const int idx = base + i;
            StreamCtx& c  = ctxs_[i];
            const ImageData& in = inputs[idx];
            ImageData& out = (*outputs)[idx];

            out.width    = in.width;
            out.height   = in.height;
            out.channels = (filter == FilterType::kVignette) ? in.channels : 1;
            out.path     = in.path;
            out.pixels.resize(out.NumBytes());

            const size_t in_bytes = in.NumBytes();
            std::memcpy(c.h_pin_in, in.pixels.data(), in_bytes);
            CUDA_CHECK(cudaMemcpyAsync(c.d_in, c.h_pin_in, in_bytes,
                                      cudaMemcpyHostToDevice, c.stream));

            switch (filter) {
                case FilterType::kGrayscale:         DispatchGrayscale (in, &out, &c); break;
                case FilterType::kGaussianBlur:      DispatchBlur      (in, &out, &c); break;
                case FilterType::kSobelEdge:         DispatchEdge      (in, &out, &c); break;
                case FilterType::kHistogramEqualize: DispatchEqualize  (in, &out, &c); break;
                case FilterType::kPipeline:          DispatchPipeline  (in, &out, &c); break;
                case FilterType::kSepGaussBlur:      DispatchSepGauss  (in, &out, &c); break;
                case FilterType::kUnsharpMask:       DispatchUnsharp   (in, &out, &c); break;
                case FilterType::kCanny:             DispatchCanny     (in, &out, &c); break;
                case FilterType::kMotionBlur:        DispatchMotionBlur(in, &out, &c); break;
                case FilterType::kAutoLevels:        DispatchAutoLevels(in, &out, &c); break;
                case FilterType::kEmboss:            DispatchEmboss    (in, &out, &c); break;
                case FilterType::kVignette:          DispatchVignette  (in, &out, &c); break;
                case FilterType::kBilateral:         DispatchBilateral (in, &out, &c); break;
                case FilterType::kAll:               DispatchPipeline  (in, &out, &c); break;
            }

            CUDA_CHECK(cudaMemcpyAsync(c.h_pin_out, c.d_out, out.NumBytes(),
                                      cudaMemcpyDeviceToHost, c.stream));
        }

        for (int i = 0; i < count; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(ctxs_[i].stream));
            ImageData& out = (*outputs)[base + i];
            std::memcpy(out.pixels.data(), ctxs_[i].h_pin_out, out.NumBytes());
            stats.total_bytes_in += inputs[base + i].NumBytes();
        }
    }

    const auto t1 = high_resolution_clock::now();
    stats.images_processed = N;
    stats.total_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    stats.avg_ms   = (N > 0) ? stats.total_ms / N : 0.0;
    return stats;
}

std::string CapstoneProcessor::FilterName(FilterType f) {
    switch (f) {
        case FilterType::kGrayscale:         return "grayscale";
        case FilterType::kGaussianBlur:      return "gaussian";
        case FilterType::kSobelEdge:         return "sobel";
        case FilterType::kHistogramEqualize: return "histoeq";
        case FilterType::kPipeline:          return "pipeline";
        case FilterType::kSepGaussBlur:      return "sepgauss";
        case FilterType::kUnsharpMask:       return "unsharp";
        case FilterType::kCanny:             return "canny";
        case FilterType::kMotionBlur:        return "motionblur";
        case FilterType::kAutoLevels:        return "autolevels";
        case FilterType::kEmboss:            return "emboss";
        case FilterType::kVignette:          return "vignette";
        case FilterType::kBilateral:         return "bilateral";
        case FilterType::kAll:               return "all";
    }
    return "unknown";
}

CapstoneProcessor::FilterType CapstoneProcessor::ParseFilter(const std::string& name) {
    if (name == "grayscale")  return FilterType::kGrayscale;
    if (name == "gaussian")   return FilterType::kGaussianBlur;
    if (name == "sobel")      return FilterType::kSobelEdge;
    if (name == "histoeq")    return FilterType::kHistogramEqualize;
    if (name == "pipeline")   return FilterType::kPipeline;
    if (name == "sepgauss")   return FilterType::kSepGaussBlur;
    if (name == "unsharp")    return FilterType::kUnsharpMask;
    if (name == "canny")      return FilterType::kCanny;
    if (name == "motionblur") return FilterType::kMotionBlur;
    if (name == "autolevels") return FilterType::kAutoLevels;
    if (name == "emboss")     return FilterType::kEmboss;
    if (name == "vignette")   return FilterType::kVignette;
    if (name == "bilateral")  return FilterType::kBilateral;
    if (name == "all")        return FilterType::kAll;
    throw std::invalid_argument("Unknown filter: " + name);
}

std::vector<CapstoneProcessor::FilterType> CapstoneProcessor::AllFilters() {
    return {
        FilterType::kGrayscale, FilterType::kGaussianBlur,
        FilterType::kSobelEdge, FilterType::kHistogramEqualize,
        FilterType::kPipeline,  FilterType::kSepGaussBlur,
        FilterType::kUnsharpMask, FilterType::kCanny,
        FilterType::kMotionBlur,  FilterType::kAutoLevels,
        FilterType::kEmboss,      FilterType::kVignette,
        FilterType::kBilateral,
    };
}

// ============================================================================
// Dispatch methods
// ============================================================================

void CapstoneProcessor::DispatchGrayscale(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    if (in.channels == 1) {
        CUDA_CHECK(cudaMemcpyAsync(c->d_out, c->d_in, static_cast<size_t>(w)*h,
                                  cudaMemcpyDeviceToDevice, c->stream));
    } else {
        GrayscaleKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(
            c->d_in, c->d_out, w, h, in.channels);
        CUDA_CHECK(cudaGetLastError());
    }
}

void CapstoneProcessor::DispatchBlur(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    ToGray(in, c);
    GaussianBlurKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchEdge(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    ToGray(in, c);
    SobelEdgeKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchEqualize(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height, np = w*h;
    ToGray(in, c);
    CUDA_CHECK(cudaMemsetAsync(c->d_histogram, 0, 256*sizeof(int), c->stream));
    HistogramKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(c->d_work, c->d_histogram, np);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    int h_hist[256];
    CUDA_CHECK(cudaMemcpy(h_hist, c->d_histogram, 256*sizeof(int), cudaMemcpyDeviceToHost));

    uint8_t lut[256] = {};
    long cdf_min = -1, running = 0;
    for (int i = 0; i < 256; ++i) if (cdf_min < 0 && h_hist[i] > 0) cdf_min = h_hist[i];
    long denom = std::max(1L, (long)np - cdf_min);
    for (int i = 0; i < 256; ++i) {
        running += h_hist[i];
        if (running == 0) continue;
        long v = (running - cdf_min) * 255 / denom;
        lut[i] = (uint8_t)std::max(0L, std::min(255L, v));
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(c->d_float_a), lut, 256,
                               cudaMemcpyHostToDevice, c->stream));
    ApplyLutKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(
        c->d_work, c->d_out, reinterpret_cast<const uint8_t*>(c->d_float_a), np);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchPipeline(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    ToGray(in, c);
    GaussianBlurKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(c->d_work, c->d_out, static_cast<size_t>(w)*h,
                               cudaMemcpyDeviceToDevice, c->stream));
    SobelEdgeKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchSepGauss(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height, np = w*h;
    ToGray(in, c);
    Uint8ToFloatKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(c->d_work, c->d_float_a, np);
    CUDA_CHECK(cudaGetLastError());
    SepGaussHorizontalKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_float_a, c->d_float_b, w, h);
    CUDA_CHECK(cudaGetLastError());
    SepGaussVerticalKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_float_b, c->d_float_a, w, h);
    CUDA_CHECK(cudaGetLastError());
    FloatToUint8Kernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(c->d_float_a, c->d_out, np);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchUnsharp(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height, np = w*h;
    ToGray(in, c);
    GaussianBlurKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    Uint8ToFloatKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(c->d_out, c->d_float_a, np);
    CUDA_CHECK(cudaGetLastError());
    UnsharpMaskKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(c->d_work, c->d_float_a, c->d_out, np, 1.5f);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchCanny(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height, np = w*h;
    ToGray(in, c);
    GaussianBlurKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(c->d_work, c->d_out, static_cast<size_t>(np),
                               cudaMemcpyDeviceToDevice, c->stream));
    GradientKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(
        c->d_work, c->d_gx, c->d_gy, c->d_mag, c->d_angle, w, h);
    CUDA_CHECK(cudaGetLastError());
    NMSKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_mag, c->d_angle, c->d_nms, w, h);
    CUDA_CHECK(cudaGetLastError());
    DoubleThresholdKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(c->d_nms, c->d_out, np, 80.f, 30.f);
    CUDA_CHECK(cudaGetLastError());
    for (int i = 0; i < 4; ++i) {
        HysteresisKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_out, w, h);
        CUDA_CHECK(cudaGetLastError());
    }
    // Suppress remaining weak (128) edges via LUT
    uint8_t lut[256] = {}; lut[255] = 255;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(c->d_float_a), lut, 256,
                               cudaMemcpyHostToDevice, c->stream));
    ApplyLutKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(
        c->d_out, c->d_work, reinterpret_cast<const uint8_t*>(c->d_float_a), np);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(c->d_out, c->d_work, static_cast<size_t>(np),
                               cudaMemcpyDeviceToDevice, c->stream));
}

void CapstoneProcessor::DispatchMotionBlur(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    const int radius = 10;
    ToGray(in, c);
    const size_t smem = static_cast<size_t>(BLOCK_H) * (BLOCK_W + 2*radius);
    MotionBlurKernel<<<Grid2D(w,h), Block2D(), smem, c->stream>>>(c->d_work, c->d_out, w, h, radius);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchAutoLevels(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height, np = w*h;
    ToGray(in, c);
    int h_init[2] = {255, 0};
    CUDA_CHECK(cudaMemcpyAsync(c->d_minmax, h_init, 2*sizeof(int),
                               cudaMemcpyHostToDevice, c->stream));
    MinMaxReductionKernel<<<Grid1D(np,256), Block1D(256), 0, c->stream>>>(c->d_work, c->d_minmax, np);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    int h_mm[2];
    CUDA_CHECK(cudaMemcpy(h_mm, c->d_minmax, 2*sizeof(int), cudaMemcpyDeviceToHost));
    AutoLevelsApplyKernel<<<Grid1D(np), Block1D(), 0, c->stream>>>(
        c->d_work, c->d_out, np, (uint8_t)h_mm[0], (uint8_t)h_mm[1]);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchEmboss(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    ToGray(in, c);
    EmbossKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(c->d_work, c->d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchVignette(const ImageData& in, ImageData* out, StreamCtx* c) {
    out->channels = in.channels;
    int w = in.width, h = in.height;
    VignetteKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(
        c->d_in, c->d_out, w, h, in.channels, 0.75f);
    CUDA_CHECK(cudaGetLastError());
}

void CapstoneProcessor::DispatchBilateral(const ImageData& in, ImageData*, StreamCtx* c) {
    int w = in.width, h = in.height;
    ToGray(in, c);
    BilateralFilterKernel<<<Grid2D(w,h), Block2D(), 0, c->stream>>>(
        c->d_work, c->d_out, w, h, 3.0f, 30.0f);
    CUDA_CHECK(cudaGetLastError());
}
