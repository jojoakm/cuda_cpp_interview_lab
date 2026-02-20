#pragma once

#include <vector>

struct PipelineConfig {
    int batch = 2048;
    int dim = 1024;
    int warmup = 3;
    int iters = 20;
    int seed = 42;
};

struct PipelineOutputs {
    std::vector<float> normalized;
    std::vector<int> argmax;
    std::vector<float> max_values;
};

struct GpuRunStats {
    PipelineOutputs outputs;
    float avg_ms = 0.0f;
};

PipelineOutputs run_cpu_pipeline(
    const std::vector<float>& logits,
    const std::vector<float>& bias,
    const PipelineConfig& config
);

GpuRunStats run_gpu_pipeline_v1(
    const std::vector<float>& logits,
    const std::vector<float>& bias,
    const PipelineConfig& config
);

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b);
int count_mismatched_indices(const std::vector<int>& a, const std::vector<int>& b);

