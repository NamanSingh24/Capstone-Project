// kernels.cu — GPU kernel implementations.
// Covers: grayscale, tiled Gaussian, Sobel, histogram+LUT, brightness/contrast,
// separable Gaussian (H+V), unsharp mask, Canny pipeline (gradient→NMS→
// double-threshold→hysteresis), motion blur, warp-shuffle auto-levels,
// emboss, vignette, bilateral filter, uint8↔float cast utilities.

#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// Constant memory: 5×5 Gaussian weights (sigma≈1, sum=273)
// ---------------------------------------------------------------------------
__constant__ float kGauss2D[25] = {
    1/273.f,  4/273.f,  7/273.f,  4/273.f,  1/273.f,
    4/273.f, 16/273.f, 26/273.f, 16/273.f,  4/273.f,
    7/273.f, 26/273.f, 41/273.f, 26/273.f,  7/273.f,
    4/273.f, 16/273.f, 26/273.f, 16/273.f,  4/273.f,
    1/273.f,  4/273.f,  7/273.f,  4/273.f,  1/273.f
};

// Separable 1-D Gaussian tap weights (radius=2, sigma≈1)
__constant__ float kGauss1D[5] = {
    1/16.f, 4/16.f, 6/16.f, 4/16.f, 1/16.f
};

#define GAUSS_R  2
#define GAUSS_SW (BLOCK_W + 2*GAUSS_R)
#define GAUSS_SH (BLOCK_H + 2*GAUSS_R)
#define SOBEL_R  1
#define SOBEL_SW (BLOCK_W + 2*SOBEL_R)
#define SOBEL_SH (BLOCK_H + 2*SOBEL_R)

// ============================================================================
// 1. GRAYSCALE
// ============================================================================
__global__ void GrayscaleKernel(const uint8_t* __restrict__ in,
                                 uint8_t*       __restrict__ out,
                                 int w, int h, int ch) {
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = (y*w + x)*ch;
    float r = in[i], g = in[i+1], b = in[i+2];
    out[y*w+x] = (uint8_t)(0.2126f*r + 0.7152f*g + 0.0722f*b);
}

// ============================================================================
// 2. TILED 2-D GAUSSIAN BLUR (single channel)
// ============================================================================
__global__ void GaussianBlurKernel(const uint8_t* __restrict__ in,
                                    uint8_t*       __restrict__ out,
                                    int w, int h) {
    __shared__ uint8_t s[GAUSS_SH][GAUSS_SW];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = (int)blockIdx.x*BLOCK_W - GAUSS_R;
    int y0 = (int)blockIdx.y*BLOCK_H - GAUSS_R;
    for (int dy = ty; dy < GAUSS_SH; dy += BLOCK_H)
        for (int dx = tx; dx < GAUSS_SW; dx += BLOCK_W) {
            int gx = min(max(x0+dx,0),w-1);
            int gy = min(max(y0+dy,0),h-1);
            s[dy][dx] = in[gy*w+gx];
        }
    __syncthreads();
    int ox = blockIdx.x*BLOCK_W+tx, oy = blockIdx.y*BLOCK_H+ty;
    if (ox >= w || oy >= h) return;
    float sum = 0;
    #pragma unroll
    for (int ky = 0; ky < 5; ++ky)
        #pragma unroll
        for (int kx = 0; kx < 5; ++kx)
            sum += s[ty+ky][tx+kx] * kGauss2D[ky*5+kx];
    out[oy*w+ox] = (uint8_t)fminf(255,fmaxf(0,sum));
}

// ============================================================================
// 3. SOBEL EDGE (tiled, single channel)
// ============================================================================
__global__ void SobelEdgeKernel(const uint8_t* __restrict__ in,
                                 uint8_t*       __restrict__ out,
                                 int w, int h) {
    __shared__ uint8_t s[SOBEL_SH][SOBEL_SW];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = (int)blockIdx.x*BLOCK_W - SOBEL_R;
    int y0 = (int)blockIdx.y*BLOCK_H - SOBEL_R;
    for (int dy = ty; dy < SOBEL_SH; dy += BLOCK_H)
        for (int dx = tx; dx < SOBEL_SW; dx += BLOCK_W) {
            int gx = min(max(x0+dx,0),w-1);
            int gy = min(max(y0+dy,0),h-1);
            s[dy][dx] = in[gy*w+gx];
        }
    __syncthreads();
    int ox = blockIdx.x*BLOCK_W+tx, oy = blockIdx.y*BLOCK_H+ty;
    if (ox >= w || oy >= h) return;
    float gx = -s[ty][tx]   +s[ty][tx+2]
              -2*s[ty+1][tx] +2*s[ty+1][tx+2]
              -s[ty+2][tx]   +s[ty+2][tx+2];
    float gy = -s[ty][tx]   -2*s[ty][tx+1]   -s[ty][tx+2]
              + s[ty+2][tx] +2*s[ty+2][tx+1] + s[ty+2][tx+2];
    out[oy*w+ox] = (uint8_t)fminf(255,sqrtf(gx*gx+gy*gy));
}

// ============================================================================
// 4. HISTOGRAM (shared-memory staging)
// ============================================================================
__global__ void HistogramKernel(const uint8_t* __restrict__ in,
                                 int*           __restrict__ hist,
                                 int np) {
    __shared__ int sh[256];
    if (threadIdx.x < 256) sh[threadIdx.x] = 0;
    __syncthreads();
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for (int i = idx; i < np; i += stride) atomicAdd(&sh[in[i]], 1);
    __syncthreads();
    if (threadIdx.x < 256) atomicAdd(&hist[threadIdx.x], sh[threadIdx.x]);
}

// ============================================================================
// 5. LUT APPLICATION
// ============================================================================
__global__ void ApplyLutKernel(const uint8_t* __restrict__ in,
                                uint8_t*       __restrict__ out,
                                const uint8_t* __restrict__ lut, int np) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < np) out[i] = lut[in[i]];
}

// ============================================================================
// 6. BRIGHTNESS / CONTRAST
// ============================================================================
__global__ void BrightnessContrastKernel(const uint8_t* __restrict__ in,
                                          uint8_t*       __restrict__ out,
                                          int np, float alpha, float beta) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= np) return;
    out[i] = (uint8_t)fminf(255,fmaxf(0, alpha*in[i]+beta));
}

// ============================================================================
// 7. SEPARABLE GAUSSIAN — HORIZONTAL PASS
// ============================================================================
__global__ void SepGaussHorizontalKernel(const float* __restrict__ in,
                                          float*       __restrict__ out,
                                          int w, int h) {
    __shared__ float s[BLOCK_H][BLOCK_W + 2*GAUSS_R];
    int tx = threadIdx.x, ty = threadIdx.y;
    int ox = blockIdx.x*BLOCK_W + tx;
    int oy = blockIdx.y*BLOCK_H + ty;
    int x0 = (int)blockIdx.x*BLOCK_W - GAUSS_R;
    int gx = min(max(x0+tx,0),w-1);
    s[ty][tx] = (oy < h) ? in[oy*w+gx] : 0.f;
    if (tx < 2*GAUSS_R) {
        gx = min(max(x0+BLOCK_W+tx,0),w-1);
        s[ty][BLOCK_W+tx] = (oy < h) ? in[oy*w+gx] : 0.f;
    }
    __syncthreads();
    if (ox >= w || oy >= h) return;
    float sum = 0;
    #pragma unroll
    for (int k = 0; k < 5; ++k) sum += s[ty][tx+k]*kGauss1D[k];
    out[oy*w+ox] = sum;
}

// ============================================================================
// 8. SEPARABLE GAUSSIAN — VERTICAL PASS
// ============================================================================
__global__ void SepGaussVerticalKernel(const float* __restrict__ in,
                                        float*       __restrict__ out,
                                        int w, int h) {
    __shared__ float s[BLOCK_H + 2*GAUSS_R][BLOCK_W];
    int tx = threadIdx.x, ty = threadIdx.y;
    int ox = blockIdx.x*BLOCK_W + tx;
    int oy = blockIdx.y*BLOCK_H + ty;
    int y0 = (int)blockIdx.y*BLOCK_H - GAUSS_R;
    int gy = min(max(y0+ty,0),h-1);
    s[ty][tx] = (ox < w) ? in[gy*w+ox] : 0.f;
    if (ty < 2*GAUSS_R) {
        gy = min(max(y0+BLOCK_H+ty,0),h-1);
        s[BLOCK_H+ty][tx] = (ox < w) ? in[gy*w+ox] : 0.f;
    }
    __syncthreads();
    if (ox >= w || oy >= h) return;
    float sum = 0;
    #pragma unroll
    for (int k = 0; k < 5; ++k) sum += s[ty+k][tx]*kGauss1D[k];
    out[oy*w+ox] = sum;
}

// ============================================================================
// 9. UNSHARP MASK
// ============================================================================
__global__ void UnsharpMaskKernel(const uint8_t* __restrict__ orig,
                                   const float*   __restrict__ blurred,
                                   uint8_t*       __restrict__ out,
                                   int np, float amount) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= np) return;
    float v = orig[i] + amount*(orig[i] - blurred[i]);
    out[i] = (uint8_t)fminf(255,fmaxf(0,v));
}

// ============================================================================
// 10. CANNY — GRADIENT
// ============================================================================
__global__ void GradientKernel(const uint8_t* __restrict__ gray,
                                float* __restrict__ gx,
                                float* __restrict__ gy,
                                float* __restrict__ mag,
                                float* __restrict__ angle,
                                int w, int h) {
    __shared__ uint8_t s[SOBEL_SH][SOBEL_SW];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = (int)blockIdx.x*BLOCK_W - SOBEL_R;
    int y0 = (int)blockIdx.y*BLOCK_H - SOBEL_R;
    for (int dy = ty; dy < SOBEL_SH; dy += BLOCK_H)
        for (int dx = tx; dx < SOBEL_SW; dx += BLOCK_W) {
            int cx = min(max(x0+dx,0),w-1);
            int cy = min(max(y0+dy,0),h-1);
            s[dy][dx] = gray[cy*w+cx];
        }
    __syncthreads();
    int ox = blockIdx.x*BLOCK_W+tx, oy = blockIdx.y*BLOCK_H+ty;
    if (ox >= w || oy >= h) return;
    int idx = oy*w+ox;
    float gxv = -s[ty][tx]   +s[ty][tx+2]
               -2*s[ty+1][tx]+2*s[ty+1][tx+2]
               -s[ty+2][tx]  +s[ty+2][tx+2];
    float gyv = -s[ty][tx]  -2*s[ty][tx+1]  -s[ty][tx+2]
               + s[ty+2][tx]+2*s[ty+2][tx+1]+s[ty+2][tx+2];
    gx[idx]    = gxv;
    gy[idx]    = gyv;
    mag[idx]   = sqrtf(gxv*gxv + gyv*gyv);
    float a = atan2f(gyv, gxv);
    if (a < 0) a += 3.14159265f;
    angle[idx] = a;
}

// ============================================================================
// 11. CANNY — NON-MAXIMUM SUPPRESSION
// ============================================================================
__global__ void NMSKernel(const float* __restrict__ mag,
                           const float* __restrict__ angle,
                           float*       __restrict__ nms,
                           int w, int h) {
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) {
        if (x < w && y < h) nms[y*w+x] = 0.f;
        return;
    }
    int idx = y*w+x;
    float a = angle[idx], m = mag[idx], m1, m2;
    if (a < 0.3927f || a >= 2.7489f) {
        m1 = mag[idx-1]; m2 = mag[idx+1];
    } else if (a < 1.1781f) {
        m1 = mag[(y-1)*w+(x+1)]; m2 = mag[(y+1)*w+(x-1)];
    } else if (a < 1.9635f) {
        m1 = mag[(y-1)*w+x]; m2 = mag[(y+1)*w+x];
    } else {
        m1 = mag[(y-1)*w+(x-1)]; m2 = mag[(y+1)*w+(x+1)];
    }
    nms[idx] = (m >= m1 && m >= m2) ? m : 0.f;
}

// ============================================================================
// 12. CANNY — DOUBLE THRESHOLD
// ============================================================================
__global__ void DoubleThresholdKernel(const float*   __restrict__ nms,
                                       uint8_t*       __restrict__ out,
                                       int np,
                                       float strong_t, float weak_t) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= np) return;
    float v = nms[i];
    out[i] = (v >= strong_t) ? 255 : (v >= weak_t) ? 128 : 0;
}

// ============================================================================
// 13. CANNY — HYSTERESIS (one iteration, in-place)
// ============================================================================
__global__ void HysteresisKernel(uint8_t* __restrict__ edges,
                                  int w, int h) {
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) return;
    int idx = y*w+x;
    if (edges[idx] != 128) return;
    bool promote = false;
    for (int dy = -1; dy <= 1 && !promote; ++dy)
        for (int dx = -1; dx <= 1 && !promote; ++dx)
            if (edges[(y+dy)*w+(x+dx)] == 255) promote = true;
    if (promote) edges[idx] = 255;
}

// ============================================================================
// 14. MOTION BLUR (horizontal box, dynamic shared memory)
// ============================================================================
__global__ void MotionBlurKernel(const uint8_t* __restrict__ in,
                                  uint8_t*       __restrict__ out,
                                  int w, int h, int radius) {
    extern __shared__ uint8_t smem[];
    int shw = BLOCK_W + 2*radius;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ox = blockIdx.x*BLOCK_W + tx;
    int oy = blockIdx.y*BLOCK_H + ty;
    for (int dx = tx; dx < shw; dx += BLOCK_W) {
        int gx = min(max((int)blockIdx.x*BLOCK_W - radius + dx, 0), w-1);
        int gy = min(max(oy,0),h-1);
        smem[ty*shw+dx] = in[gy*w+gx];
    }
    __syncthreads();
    if (ox >= w || oy >= h) return;
    float sum = 0;
    int diam = 2*radius+1;
    for (int k = 0; k < diam; ++k) sum += smem[ty*shw + tx+k];
    out[oy*w+ox] = (uint8_t)(sum / diam);
}

// ============================================================================
// 15. AUTO-LEVELS — warp-shuffle min/max reduction
// ============================================================================
__global__ void MinMaxReductionKernel(const uint8_t* __restrict__ in,
                                       int*           __restrict__ res,
                                       int np) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int mn = 255, mx = 0;
    if (idx < np) { mn = in[idx]; mx = in[idx]; }
    for (int off = 16; off > 0; off >>= 1) {
        int om = __shfl_down_sync(0xffffffff, mn, off);
        int ox = __shfl_down_sync(0xffffffff, mx, off);
        mn = min(mn, om); mx = max(mx, ox);
    }
    __shared__ int smn[32], smx[32];
    if (lane == 0) { smn[warp] = mn; smx[warp] = mx; }
    __syncthreads();
    int nwarps = (blockDim.x+31)/32;
    if (warp == 0) {
        mn = (lane < nwarps) ? smn[lane] : 255;
        mx = (lane < nwarps) ? smx[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            int om = __shfl_down_sync(0xffffffff, mn, off);
            int ox = __shfl_down_sync(0xffffffff, mx, off);
            mn = min(mn, om); mx = max(mx, ox);
        }
        if (lane == 0) {
            atomicMin(&res[0], mn);
            atomicMax(&res[1], mx);
        }
    }
}

__global__ void AutoLevelsApplyKernel(const uint8_t* __restrict__ in,
                                       uint8_t*       __restrict__ out,
                                       int np,
                                       uint8_t gmin, uint8_t gmax) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= np) return;
    int range = (int)gmax - gmin;
    if (range <= 0) { out[i] = in[i]; return; }
    out[i] = (uint8_t)(((int)in[i] - gmin) * 255 / range);
}

// ============================================================================
// 16. EMBOSS
// ============================================================================
__global__ void EmbossKernel(const uint8_t* __restrict__ in,
                              uint8_t*       __restrict__ out,
                              int w, int h) {
    __shared__ uint8_t s[SOBEL_SH][SOBEL_SW];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = (int)blockIdx.x*BLOCK_W - SOBEL_R;
    int y0 = (int)blockIdx.y*BLOCK_H - SOBEL_R;
    for (int dy = ty; dy < SOBEL_SH; dy += BLOCK_H)
        for (int dx = tx; dx < SOBEL_SW; dx += BLOCK_W) {
            int gx = min(max(x0+dx,0),w-1);
            int gy = min(max(y0+dy,0),h-1);
            s[dy][dx] = in[gy*w+gx];
        }
    __syncthreads();
    int ox = blockIdx.x*BLOCK_W+tx, oy = blockIdx.y*BLOCK_H+ty;
    if (ox >= w || oy >= h) return;
    float v = -2.f*s[ty][tx]   - 1.f*s[ty][tx+1]
              -1.f*s[ty+1][tx] + 1.f*s[ty+1][tx+1] + 1.f*s[ty+1][tx+2]
              +1.f*s[ty+2][tx+1] + 2.f*s[ty+2][tx+2] + 128.f;
    out[oy*w+ox] = (uint8_t)fminf(255,fmaxf(0,v));
}

// ============================================================================
// 17. VIGNETTE
// ============================================================================
__global__ void VignetteKernel(const uint8_t* __restrict__ in,
                                uint8_t*       __restrict__ out,
                                int w, int h, int ch, float strength) {
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    if (x >= w || y >= h) return;
    float cx = (x - w*0.5f) / (w*0.5f);
    float cy = (y - h*0.5f) / (h*0.5f);
    float d  = sqrtf(cx*cx + cy*cy) / 1.4142f;
    float f  = fmaxf(0.f, fminf(1.f, 1.f - strength * d * d));
    int base = (y*w+x)*ch;
    for (int c = 0; c < ch; ++c)
        out[base+c] = (uint8_t)(in[base+c] * f);
}

// ============================================================================
// 18. BILATERAL FILTER (edge-preserving)
// ============================================================================
__global__ void BilateralFilterKernel(const uint8_t* __restrict__ in,
                                       uint8_t*       __restrict__ out,
                                       int w, int h,
                                       float ss, float sr) {
    int x = blockIdx.x*BLOCK_W + threadIdx.x;
    int y = blockIdx.y*BLOCK_H + threadIdx.y;
    if (x >= w || y >= h) return;
    float center = (float)in[y*w+x];
    float sum_w = 0.f, sum_v = 0.f;
    int R = 3;
    float inv_ss2 = -1.f/(2.f*ss*ss);
    float inv_sr2 = -1.f/(2.f*sr*sr);
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int nx = min(max(x+dx,0),w-1);
            int ny = min(max(y+dy,0),h-1);
            float v  = (float)in[ny*w+nx];
            float ws = expf((dx*dx+dy*dy)*inv_ss2);
            float wr = expf((v-center)*(v-center)*inv_sr2);
            float wc = ws*wr;
            sum_w += wc;
            sum_v += v * wc;
        }
    }
    out[y*w+x] = (uint8_t)fminf(255.f, fmaxf(0.f, sum_v/sum_w));
}

// ============================================================================
// 19–20. UINT8 ↔ FLOAT CAST UTILITIES
// ============================================================================
__global__ void Uint8ToFloatKernel(const uint8_t* __restrict__ in,
                                    float*         __restrict__ out,
                                    int np) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < np) out[i] = static_cast<float>(in[i]);
}

__global__ void FloatToUint8Kernel(const float*   __restrict__ in,
                                    uint8_t*       __restrict__ out,
                                    int np) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < np) out[i] = (uint8_t)fminf(255.f, fmaxf(0.f, in[i]));
}
