// Common types, logging and timing utilities for the DREAMPlaceFPGA C++20 port.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#define DPFPGA_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
#define DPFPGA_PARALLEL_FOR
#endif

namespace dpfpga {

/// Floating point type used by every numerical kernel. The original code base
/// supports float32/float64 selected at run time; here it is a compile-time
/// choice (define DPFPGA_REAL=float to change it).
#ifdef DPFPGA_REAL
using Real = DPFPGA_REAL;
#else
using Real = double;
#endif

/// Resource / fence-region identifiers used throughout the placer.
enum Region : int { kLUT = 0, kFF = 1, kDSP = 2, kRAM = 3, kIO = 4 };
inline constexpr int kNumMovableRegions = 4;  // LUT, FF, DSP, RAM
inline constexpr int kNumRegions = 5;         // + IO

/// Site type identifiers stored in the site map.
enum SiteType : int { kSiteEmpty = 0, kSiteSlice = 1, kSiteDsp = 2, kSiteRam = 3, kSiteIo = 4 };

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
enum class LogLevel { kInfo, kWarn, kError };

inline void log_line(LogLevel level, const std::string& msg) {
  const char* tag = level == LogLevel::kInfo ? "INFO   " : level == LogLevel::kWarn ? "WARNING" : "ERROR  ";
  std::cout << "[" << tag << "] DREAMPlaceFPGA - " << msg << std::endl;
}

template <class... Args>
void info(std::format_string<Args...> fmt, Args&&... args) {
  log_line(LogLevel::kInfo, std::format(fmt, std::forward<Args>(args)...));
}
template <class... Args>
void warn(std::format_string<Args...> fmt, Args&&... args) {
  log_line(LogLevel::kWarn, std::format(fmt, std::forward<Args>(args)...));
}
template <class... Args>
void error(std::format_string<Args...> fmt, Args&&... args) {
  log_line(LogLevel::kError, std::format(fmt, std::forward<Args>(args)...));
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
class Stopwatch {
 public:
  Stopwatch() : start_(Clock::now()) {}
  void reset() { start_ = Clock::now(); }
  [[nodiscard]] double seconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

/// Error type thrown on unrecoverable input / consistency problems.
struct PlacerError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace dpfpga
