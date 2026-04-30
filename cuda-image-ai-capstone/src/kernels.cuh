// kernels.cuh — All GPU kernel declarations.
#ifndef CAPSTONE_SRC_KERNELS_CUH_
#define CAPSTONE_SRC_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_W 16
#define BLOCK_H 16

// 1. RGB → grayscale (BT.709)
__global__ void GrayscaleKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height, int channels);

// 2. 5×5 Gaussian blur — shared-memory tiled, single-channel
__global__ void GaussianBlurKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height);

// 3. 3×3 Sobel edge — shared-memory tiled, single-channel
__global__ void SobelEdgeKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height);

// 4. Per-channel histogram (shared-memory staging + atomic)
__global__ void HistogramKernel(
    const uint8_t* __restrict__ input,
    int*           __restrict__ histogram,
    int num_pixels);

// 5. Apply a 256-entry LUT to each pixel
__global__ void ApplyLutKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    const uint8_t* __restrict__ lut,
    int num_pixels);

// 6. Brightness / contrast: out = clamp(alpha*in + beta, 0, 255)
__global__ void BrightnessContrastKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int num_pixels, float alpha, float beta);

// 7. Separable Gaussian — horizontal pass (float buffers)
__global__ void SepGaussHorizontalKernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int width, int height);

// 8. Separable Gaussian — vertical pass (float buffers)
__global__ void SepGaussVerticalKernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int width, int height);

// 9. Unsharp mask: out = clamp(orig + amount*(orig - blurred), 0, 255)
__global__ void UnsharpMaskKernel(
    const uint8_t* __restrict__ original,
    const float*   __restrict__ blurred,
    uint8_t*       __restrict__ output,
    int num_pixels, float amount);

// 10. Canny — compute Gx, Gy, magnitude, angle from grayscale
__global__ void GradientKernel(
    const uint8_t* __restrict__ gray,
    float*         __restrict__ gx,
    float*         __restrict__ gy,
    float*         __restrict__ magnitude,
    float*         __restrict__ angle,
    int width, int height);

// 11. Canny — non-maximum suppression
__global__ void NMSKernel(
    const float*   __restrict__ magnitude,
    const float*   __restrict__ angle,
    float*         __restrict__ nms_out,
    int width, int height);

// 12. Canny — double threshold (strong=255, weak=128, none=0)
__global__ void DoubleThresholdKernel(
    const float*   __restrict__ nms_in,
    uint8_t*       __restrict__ output,
    int num_pixels,
    float strong_thresh, float weak_thresh);

// 13. Canny — hysteresis: promote weak pixels adjacent to strong (run 3-4x)
__global__ void HysteresisKernel(
    uint8_t* __restrict__ edges,
    int width, int height);

// 14. Horizontal motion blur (dynamic shared memory)
__global__ void MotionBlurKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height, int radius);

// 15. Auto-levels pass 1 — warp-shuffle min/max reduction
__global__ void MinMaxReductionKernel(
    const uint8_t* __restrict__ input,
    int*           __restrict__ d_block_results,
    int num_pixels);

// 16. Auto-levels pass 2 — linear stretch
__global__ void AutoLevelsApplyKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int num_pixels,
    uint8_t global_min, uint8_t global_max);

// 17. Emboss (directional relief)
__global__ void EmbossKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height);

// 18. Vignette (radial darkening)
__global__ void VignetteKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height, int channels, float strength);

// 19. Bilateral filter (edge-preserving smoothing)
__global__ void BilateralFilterKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width, int height,
    float sigma_s, float sigma_r);

// 20. uint8 → float cast (identity scale, range [0,255])
__global__ void Uint8ToFloatKernel(
    const uint8_t* __restrict__ input,
    float*         __restrict__ output,
    int num_pixels);

// 21. float → uint8 cast (clamp to [0,255])
__global__ void FloatToUint8Kernel(
    const float*   __restrict__ input,
    uint8_t*       __restrict__ output,
    int num_pixels);

#endif  // CAPSTONE_SRC_KERNELS_CUH_
