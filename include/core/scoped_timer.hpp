#pragma once

/// @file scoped_timer.hpp
/// @brief RAII scoped timers for zero-overhead profiling
///
/// ScopedTimer always measures and accumulates elapsed time.
/// OptionalScopedTimer only measures when given a non-null pointer,
/// providing true zero overhead when profiling is disabled.

#include <chrono>

namespace drifter {

/// @brief RAII timer that accumulates elapsed time (ms) into a double reference
class ScopedTimer {
  public:
  explicit ScopedTimer(double &accumulator) : accumulator_(accumulator), start_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    auto end      = std::chrono::steady_clock::now();
    accumulator_ += std::chrono::duration<double, std::milli>(end - start_).count();
  }

  ScopedTimer(const ScopedTimer &)            = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

  private:
  double &accumulator_;
  std::chrono::steady_clock::time_point start_;
};

/// @brief RAII timer that accumulates elapsed time (ms) only if pointer is
/// non-null. True zero overhead when accumulator is nullptr (no chrono calls).
class OptionalScopedTimer {
  public:
  explicit OptionalScopedTimer(double* accumulator) : accumulator_(accumulator) {
    if (accumulator_) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~OptionalScopedTimer() {
    if (accumulator_) {
      auto end       = std::chrono::steady_clock::now();
      *accumulator_ += std::chrono::duration<double, std::milli>(end - start_).count();
    }
  }

  OptionalScopedTimer(const OptionalScopedTimer &)            = delete;
  OptionalScopedTimer &operator=(const OptionalScopedTimer &) = delete;

  private:
  double* accumulator_;
  std::chrono::steady_clock::time_point start_{};
};

} // namespace drifter
