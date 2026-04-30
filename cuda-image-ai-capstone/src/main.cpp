// main.cpp — GPU Capstone: CUDA Image Filter Pipeline
//
// Usage:
//   ./capstone --input <dir> --output <dir> [--filter <name>] [--streams <n>]
//
// Filters: grayscale gaussian sobel histoeq pipeline sepgauss unsharp
//          canny motionblur autolevels emboss vignette bilateral all

#include "processor.h"
#include "image_io.h"
#include "cuda_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using FilterType = CapstoneProcessor::FilterType;

static std::string Stem(const std::string& path) {
    size_t slash = path.rfind('/');
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.rfind('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

static void PrintUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --input <dir> --output <dir> [--filter <name>] [--streams <n>]\n"
        "Filters: grayscale gaussian sobel histoeq pipeline sepgauss unsharp\n"
        "         canny motionblur autolevels emboss vignette bilateral all\n"
        "Default: filter=canny  streams=4\n", prog);
}

// Opens (or appends to) logs/run_TIMESTAMP.log and returns the file stream.
static FILE* OpenLog(const std::string& log_dir) {
    time_t t = time(nullptr);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", localtime(&t));
    std::string path = log_dir + "/run_" + ts + ".log";
    FILE* f = fopen(path.c_str(), "w");
    if (f) fprintf(f, "=== capstone run %s ===\n", ts);
    return f;
}

static void Log(FILE* f, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (f) {
        va_start(ap, fmt);
        vfprintf(f, fmt, ap);
        va_end(ap);
    }
}

static bool RunFilter(CapstoneProcessor& proc,
                      const std::vector<ImageData>& inputs,
                      FilterType filter,
                      const std::string& out_dir,
                      int num_streams,
                      FILE* log) {
    const std::string fname = CapstoneProcessor::FilterName(filter);
    Log(log, "[capstone] Running filter '%s' on %d image(s)...\n",
        fname.c_str(), (int)inputs.size());

    std::vector<ImageData> outputs;
    auto stats = proc.ProcessBatch(inputs, filter, &outputs);

    Log(log, "  images_processed : %d\n",  stats.images_processed);
    Log(log, "  total_wall_ms    : %.2f\n", stats.total_ms);
    Log(log, "  avg_per_image_ms : %.2f\n", stats.avg_ms);
    Log(log, "  total_input_MB   : %.2f\n",
        (double)stats.total_bytes_in / (1024.0*1024.0));

    bool ok = true;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const ImageData& out = outputs[i];
        std::string path = out_dir + "/" + Stem(inputs[i].path) + "_" + fname + ".png";
        if (!SaveImage(path, out)) {
            Log(log, "  [ERROR] failed to save %s\n", path.c_str());
            ok = false;
        } else {
            Log(log, "  saved: %s  (%dx%dx%d)\n",
                path.c_str(), out.width, out.height, out.channels);
        }
    }
    return ok;
}

int main(int argc, char** argv) {
    std::string in_dir   = "";
    std::string out_dir  = "";
    std::string log_dir  = "logs";
    std::string filter_s = "canny";
    int num_streams      = 4;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i],"--input")   && i+1<argc) in_dir    = argv[++i];
        else if (!strcmp(argv[i],"--output")  && i+1<argc) out_dir   = argv[++i];
        else if (!strcmp(argv[i],"--filter")  && i+1<argc) filter_s  = argv[++i];
        else if (!strcmp(argv[i],"--streams") && i+1<argc) num_streams = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--logs")    && i+1<argc) log_dir   = argv[++i];
        else if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h"))
            { PrintUsage(argv[0]); return 0; }
        else { fprintf(stderr,"Unknown arg: %s\n",argv[i]); PrintUsage(argv[0]); return 1; }
    }

    if (in_dir.empty() || out_dir.empty()) { PrintUsage(argv[0]); return 1; }
    num_streams = std::max(1, std::min(num_streams, 16));

    // Ensure output and log dirs exist
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\" \"%s\"", out_dir.c_str(), log_dir.c_str());
    (void)system(cmd);

    FILE* log = OpenLog(log_dir);

    // GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    Log(log, "[capstone] GPU: %s  (SM %d.%d, %.0f MB VRAM)\n",
        prop.name, prop.major, prop.minor,
        (double)prop.totalGlobalMem / (1024.0*1024.0));
    Log(log, "[capstone] streams=%d  filter=%s\n", num_streams, filter_s.c_str());

    // Load images
    auto paths = ListImages(in_dir);
    if (paths.empty()) {
        Log(log, "[capstone] ERROR: no images found in '%s'\n", in_dir.c_str());
        if (log) fclose(log);
        return 1;
    }
    Log(log, "[capstone] Loading %d image(s) from '%s'\n",
        (int)paths.size(), in_dir.c_str());

    std::vector<ImageData> inputs;
    for (const auto& p : paths) {
        ImageData img;
        if (LoadImage(p, &img)) {
            Log(log, "  loaded: %s  (%dx%dx%d)\n",
                p.c_str(), img.width, img.height, img.channels);
            inputs.push_back(std::move(img));
        } else {
            Log(log, "  [WARN] skipped (load failed): %s\n", p.c_str());
        }
    }
    if (inputs.empty()) {
        Log(log, "[capstone] ERROR: no images could be loaded\n");
        if (log) fclose(log);
        return 1;
    }

    // Parse filter
    FilterType filter;
    try {
        filter = CapstoneProcessor::ParseFilter(filter_s);
    } catch (const std::exception& e) {
        Log(log, "[capstone] ERROR: %s\n", e.what());
        if (log) fclose(log);
        return 1;
    }

    // Allocate processor
    CapstoneProcessor proc(num_streams);
    proc.AllocateForBatch(inputs);

    bool ok = true;
    if (filter == FilterType::kAll) {
        for (FilterType f : CapstoneProcessor::AllFilters())
            if (!RunFilter(proc, inputs, f, out_dir, num_streams, log)) ok = false;
    } else {
        ok = RunFilter(proc, inputs, filter, out_dir, num_streams, log);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    Log(log, "[capstone] Done.%s\n", ok ? "" : " (some outputs failed)");

    if (log) fclose(log);
    return ok ? 0 : 1;
}
