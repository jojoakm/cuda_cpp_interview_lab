#include "../include/pipeline.h"
#include "../include/cuda_utils.h"

#include <algorithm>
#include <cfloat>
#include <stdexcept>

namespace {

constexpr int kThreads = 256;
constexpr float kEps = 1e-6f;

__global__ void add_bias_relu_kernel(
    const float* logits,
    const float* bias,
    float* activated,
    int batch,
    int dim
) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || col >= dim) {
        return;
    }

    const int idx = row * dim + col;
    const float value = logits[idx] + bias[col];
    activated[idx] = value > 0.0f ? value : 0.0f;
}

__global__ void row_sum_kernel(
    const float* activated,
    float* row_sums,
    int batch,
    int dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= batch) {
        return;
    }

    extern __shared__ float sdata[];
    float local_sum = 0.0f;

    const int row_offset = row * dim;
    for (int col = tid; col < dim; col += blockDim.x) {
        local_sum += activated[row_offset + col];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sums[row] = sdata[0] > kEps ? sdata[0] : kEps;
    }
}

__global__ void normalize_kernel(
    float* activated,
    const float* row_sums,
    int batch,
    int dim
) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || col >= dim) {
        return;
    }

    const int idx = row * dim + col;
    activated[idx] /= row_sums[row];
}

__global__ void row_argmax_kernel(
    const float* normalized,
    int* argmax_indices,
    float* max_values,
    int batch,
    int dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= batch) {
        return;
    }

    extern __shared__ unsigned char shared_raw[];
    float* s_values = reinterpret_cast<float*>(shared_raw);
    int* s_indices = reinterpret_cast<int*>(s_values + blockDim.x);

    const int row_offset = row * dim;
    float best_value = -FLT_MAX;
    int best_index = 0;

    for (int col = tid; col < dim; col += blockDim.x) {
        const float value = normalized[row_offset + col];
        if (value > best_value) {
            best_value = value;
            best_index = col;
        }
    }

    s_values[tid] = best_value;
    s_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_values[tid + stride] > s_values[tid]) {
            s_values[tid] = s_values[tid + stride];
            s_indices[tid] = s_indices[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        argmax_indices[row] = s_indices[0];
        max_values[row] = s_values[0];
    }
}

void launch_pipeline_v1(
    const DeviceBuffer<float>& d_logits,
    const DeviceBuffer<float>& d_bias,
    DeviceBuffer<float>& d_activated,
    DeviceBuffer<float>& d_row_sums,
    DeviceBuffer<int>& d_argmax,
    DeviceBuffer<float>& d_max_values,
    int batch,
    int dim
) {
    const dim3 block(kThreads);
    const dim3 grid_2d((dim + kThreads - 1) / kThreads, batch);
    const dim3 grid_rows(batch);

    add_bias_relu_kernel<<<grid_2d, block>>>(d_logits.data(), d_bias.data(), d_activated.data(), batch, dim);
    CUDA_CHECK(cudaGetLastError());

    row_sum_kernel<<<grid_rows, block, kThreads * sizeof(float)>>>(d_activated.data(), d_row_sums.data(), batch, dim);
    CUDA_CHECK(cudaGetLastError());

    normalize_kernel<<<grid_2d, block>>>(d_activated.data(), d_row_sums.data(), batch, dim);
    CUDA_CHECK(cudaGetLastError());

    row_argmax_kernel<<<grid_rows, block, kThreads * (sizeof(float) + sizeof(int))>>>(
        d_activated.data(), d_argmax.data(), d_max_values.data(), batch, dim
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

GpuRunStats run_gpu_pipeline_v1(
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
    if (config.batch <= 0 || config.dim <= 0 || config.iters <= 0) {
        throw std::invalid_argument("batch/dim/iters must be positive");
    }

    DeviceBuffer<float> d_logits(total);
    DeviceBuffer<float> d_bias(config.dim);
    DeviceBuffer<float> d_activated(total);
    DeviceBuffer<float> d_row_sums(config.batch);
    DeviceBuffer<int> d_argmax(config.batch);
    DeviceBuffer<float> d_max_values(config.batch);

    CUDA_CHECK(cudaMemcpy(
        d_logits.data(),
        logits.data(),
        total * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_bias.data(),
        bias.data(),
        static_cast<std::size_t>(config.dim) * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    for (int i = 0; i < config.warmup; ++i) {
        launch_pipeline_v1(
            d_logits, d_bias, d_activated, d_row_sums, d_argmax, d_max_values, config.batch, config.dim
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < config.iters; ++i) {
        launch_pipeline_v1(
            d_logits, d_bias, d_activated, d_row_sums, d_argmax, d_max_values, config.batch, config.dim
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    GpuRunStats stats;
    stats.avg_ms = elapsed_ms / config.iters;
    stats.outputs.normalized.resize(total);
    stats.outputs.argmax.resize(config.batch);
    stats.outputs.max_values.resize(config.batch);

    CUDA_CHECK(cudaMemcpy(
        stats.outputs.normalized.data(),
        d_activated.data(),
        total * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        stats.outputs.argmax.data(),
        d_argmax.data(),
        static_cast<std::size_t>(config.batch) * sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        stats.outputs.max_values.data(),
        d_max_values.data(),
        static_cast<std::size_t>(config.batch) * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return stats;
}
