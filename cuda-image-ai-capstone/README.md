# GPU Capstone — CUDA Image Filter Pipeline

**Course:** CUDA at Scale for the Enterprise — Capstone Project  
**Framework:** NVIDIA CUDA C++

## Overview

A GPU-accelerated batch image processing pipeline implementing **13 filters** across **21 CUDA kernels**, demonstrating every major GPU programming technique from the specialisation: shared-memory tiling, constant memory, warp-shuffle reductions, dynamic shared memory, multi-stream concurrency, and async DMA via pinned memory.

## Filters

| CLI name | Description | Key GPU technique |
|---|---|---|
| `grayscale` | BT.709 luma conversion | Per-pixel parallelism |
| `gaussian` | 5×5 Gaussian blur | Shared-memory tiling, constant memory |
| `sobel` | Sobel edge detection | Shared-memory tiling |
| `histoeq` | Histogram equalisation | Shared-memory atomic histogram |
| `pipeline` | Gray → Blur → Sobel chain | Multi-kernel chaining |
| `sepgauss` | Separable 1-D Gaussian (H+V) | Two-pass convolution (~2× faster) |
| `unsharp` | Unsharp masking (sharpen) | Float intermediate buffer |
| `canny` | Full Canny edge detector | 7-kernel GPU pipeline |
| `motionblur` | Horizontal box blur | Dynamic shared memory |
| `autolevels` | Contrast auto-stretch | **Warp-shuffle min/max reduction** |
| `emboss` | Directional relief | Convolution kernel |
| `vignette` | Radial darkening | Analytic per-pixel computation |
| `bilateral` | Edge-preserving smooth | Range + spatial Gaussian weights |

## Requirements

| Dependency | Notes |
|---|---|
| NVIDIA GPU (compute 6.0+) | Pascal or newer |
| CUDA Toolkit 11.x – 12.x | Provides `nvcc` |
| GCC / Clang with C++14 | Host compilation |
| GNU Make | Build system |
| curl | Auto-downloads stb headers |

## Build

```bash
# Clone the repo
git clone <repo-url>
cd cuda-image-ai-capstone

# Build (adjust GPU_ARCH for your GPU)
make GPU_ARCH=sm_75        # Turing / RTX 20xx
make GPU_ARCH=sm_86        # Ampere / RTX 30xx
make GPU_ARCH=sm_89        # Ada / RTX 40xx
```

The `stb_image` headers are downloaded automatically on first build.

## Run

```bash
# Place PNG/JPG images in data/
# Run a single filter
./capstone --input data --output output --filter canny

# Run all 13 filters (saves one PNG per filter per image + writes a log)
./capstone --input data --output output --logs logs --filter all --streams 4

# Quick demo target
make run_demo
```

**Output** files are named `<stem>_<filter>.png` in the output directory.  
**Log** files are written to `logs/run_YYYYMMDD_HHMMSS.log`.

## Project Structure

```
cuda-image-ai-capstone/
├── Makefile
├── README.md
├── .gitignore
├── data/           ← place input images here
└── src/
    ├── cuda_utils.h      CUDA_CHECK macro + GpuTimer
    ├── image_io.h/.cpp   Image load/save (stb_image)
    ├── kernels.cuh/.cu   21 CUDA kernel implementations
    ├── processor.h/.cpp  Multi-stream batch engine
    └── main.cpp          CLI entry point
```

## GPU Techniques Demonstrated

- **Shared-memory tiling** — GaussianBlurKernel, SobelEdgeKernel load a halo region into `__shared__` memory, eliminating redundant global reads
- **Constant memory** — 5×5 Gaussian weights in `__constant__` for broadcast-efficient access
- **Warp-shuffle reduction** — MinMaxReductionKernel uses `__shfl_down_sync` for in-warp min/max, no atomics until the final block write
- **Dynamic shared memory** — MotionBlurKernel allocates smem at runtime (`extern __shared__`) to support arbitrary blur radii
- **Multi-stream execution** — ProcessBatch assigns images round-robin across N CUDA streams, overlapping H2D, compute, and D2H
- **Pinned host memory** — `cudaMallocHost` enables async DMA at full PCIe bandwidth
- **Full Canny pipeline** — 7 kernels in one stream: grayscale → gaussian → gradient → NMS → double-threshold → hysteresis × 4 → LUT suppress

## Data Sources

Sample images from:
- [USC SIPI Image Database](https://sipi.usc.edu/database/database.php)
- [Creative Commons Search](https://search.creativecommons.org)
