#include "../include/pipeline.h"
#include "../include/timer.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --batch=<int>    Batch size (default: 2048)\n"
        << "  --dim=<int>      Feature dimension (default: 1024)\n"
        << "  --warmup=<int>   Warmup iterations on GPU (default: 3)\n"
        << "  --iters=<int>    Timed GPU iterations (default: 20)\n"
        << "  --seed=<int>     Random seed (default: 42)\n"
        << "  --help           Show help\n";
}

int parse_positive_int(const std::string& text, const char* name) {
    int value = 0;
    try {
        value = std::stoi(text);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + text);
    }
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return value;
}

PipelineConfig parse_args(int argc, char** argv) {
    PipelineConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg.rfind("--batch=", 0) == 0) {
            config.batch = parse_positive_int(arg.substr(8), "batch");
        } else if (arg.rfind("--dim=", 0) == 0) {
            config.dim = parse_positive_int(arg.substr(6), "dim");
        } else if (arg.rfind("--warmup=", 0) == 0) {
            config.warmup = parse_positive_int(arg.substr(9), "warmup");
        } else if (arg.rfind("--iters=", 0) == 0) {
            config.iters = parse_positive_int(arg.substr(8), "iters");
        } else if (arg.rfind("--seed=", 0) == 0) {
            config.seed = parse_positive_int(arg.substr(7), "seed");
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return config;
}

double million_elements_per_second(int batch, int dim, double ms) {
    const double elements = static_cast<double>(batch) * dim;
    return elements / (ms / 1000.0) / 1e6;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const PipelineConfig config = parse_args(argc, argv);
        const std::size_t total = static_cast<std::size_t>(config.batch) * config.dim;

        std::cout << "================ CUDA + C++ Interview Lab ================\n";
        std::cout << "Config: batch=" << config.batch
                  << ", dim=" << config.dim
                  << ", warmup=" << config.warmup
                  << ", iters=" << config.iters
                  << ", seed=" << config.seed << "\n\n";

        std::vector<float> logits(total);
        std::vector<float> bias(config.dim);

        std::mt19937 rng(config.seed);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto& value : logits) {
            value = dist(rng);
        }
        for (auto& value : bias) {
            value = dist(rng);
        }

        CpuTimer cpu_timer;
        const PipelineOutputs cpu_outputs = run_cpu_pipeline(logits, bias, config);
        const double cpu_ms = cpu_timer.elapsed_ms();

        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            std::cout << "GPU runtime not available in current environment.\n";
            std::cout << "CPU baseline finished: " << std::fixed << std::setprecision(3)
                      << cpu_ms << " ms\n";
            std::cout << "Tip: run on a CUDA-enabled machine for full GPU benchmark.\n";
            return 0;
        }

        const GpuRunStats gpu_stats = run_gpu_pipeline_v1(logits, bias, config);
        const double gpu_ms = gpu_stats.avg_ms;

        const float normalized_diff = max_abs_diff(cpu_outputs.normalized, gpu_stats.outputs.normalized);
        const float max_values_diff = max_abs_diff(cpu_outputs.max_values, gpu_stats.outputs.max_values);
        const int argmax_mismatch = count_mismatched_indices(cpu_outputs.argmax, gpu_stats.outputs.argmax);

        const double cpu_meps = million_elements_per_second(config.batch, config.dim, cpu_ms);
        const double gpu_meps = million_elements_per_second(config.batch, config.dim, gpu_ms);

        std::cout << std::left
                  << std::setw(16) << "Backend"
                  << std::setw(14) << "Time(ms)"
                  << std::setw(20) << "Throughput(M elem/s)" << "\n";
        std::cout << std::string(50, '-') << "\n";
        std::cout << std::setw(16) << "CPU"
                  << std::setw(14) << std::fixed << std::setprecision(3) << cpu_ms
                  << std::setw(20) << std::fixed << std::setprecision(2) << cpu_meps << "\n";
        std::cout << std::setw(16) << "GPU(V1)"
                  << std::setw(14) << std::fixed << std::setprecision(3) << gpu_ms
                  << std::setw(20) << std::fixed << std::setprecision(2) << gpu_meps << "\n\n";

        std::cout << "Correctness:\n";
        std::cout << "  max_abs_diff(normalized) = " << std::scientific << normalized_diff << "\n";
        std::cout << "  max_abs_diff(max_values) = " << std::scientific << max_values_diff << "\n";
        std::cout << "  argmax_mismatch_count    = " << std::fixed << argmax_mismatch << "\n";

        const bool pass = normalized_diff < 1e-4f && max_values_diff < 1e-4f && argmax_mismatch == 0;
        std::cout << "\nResult: " << (pass ? "PASS ✅" : "CHECK ⚠️") << "\n";

        if (!pass) {
            std::cout << "Hint: reduce precision/order differences are expected, but mismatch should be small.\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Tip: run with --help for options.\n";
        return 1;
    }
}
