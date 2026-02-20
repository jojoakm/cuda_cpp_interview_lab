#include "../include/pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

PipelineOutputs run_cpu_pipeline(
    const std::vector<float>& logits,
    const std::vector<float>& bias,
    const PipelineConfig& config
) {
    const std::size_t total = static_cast<std::size_t>(config.batch) * config.dim;
    if (logits.size() != total) {
        throw std::invalid_argument("logits size does not match batch * dim");
    }
    if (bias.size() != static_cast<std::size_t>(config.dim)) {
        throw std::invalid_argument("bias size does not match dim");
    }

    PipelineOutputs outputs;
    outputs.normalized.resize(total);
    outputs.argmax.resize(config.batch);
    outputs.max_values.resize(config.batch);

    constexpr float kEps = 1e-6f;

    for (int row = 0; row < config.batch; ++row) {
        float row_sum = 0.0f;
        const int row_offset = row * config.dim;

        for (int col = 0; col < config.dim; ++col) {
            const int idx = row_offset + col;
            const float value = std::max(0.0f, logits[idx] + bias[col]);
            outputs.normalized[idx] = value;
            row_sum += value;
        }

        const float inv = 1.0f / std::max(row_sum, kEps);
        float best_value = -std::numeric_limits<float>::infinity();
        int best_index = 0;

        for (int col = 0; col < config.dim; ++col) {
            const int idx = row_offset + col;
            const float normalized = outputs.normalized[idx] * inv;
            outputs.normalized[idx] = normalized;

            if (normalized > best_value) {
                best_value = normalized;
                best_index = col;
            }
        }

        outputs.argmax[row] = best_index;
        outputs.max_values[row] = best_value;
    }

    return outputs;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("float vector sizes do not match");
    }

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int count_mismatched_indices(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("int vector sizes do not match");
    }

    int mismatches = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

