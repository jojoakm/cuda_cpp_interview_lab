#pragma once

#include <chrono>

class CpuTimer {
public:
    CpuTimer() : start_(clock::now()) {}

    void reset() {
        start_ = clock::now();
    }

    [[nodiscard]] double elapsed_ms() const {
        const auto now = clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
};

