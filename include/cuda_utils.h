#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

inline void cuda_check(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error(
            "CUDA error at " + std::string(file) + ":" + std::to_string(line) + " -> " +
            cudaGetErrorString(code)
        );
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        *this = std::move(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    [[nodiscard]] T* data() {
        return ptr_;
    }

    [[nodiscard]] const T* data() const {
        return ptr_;
    }

    [[nodiscard]] std::size_t size() const {
        return count_;
    }

    [[nodiscard]] bool empty() const {
        return ptr_ == nullptr;
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

