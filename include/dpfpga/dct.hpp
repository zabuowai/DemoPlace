// Discrete cosine/sine transforms used to solve the Poisson equation of the
// electrostatic density model (port of the ops/dct package).
//
// Conventions (all *unnormalised* unless noted, index i = spatial, u = frequency):
//   dct1d   : y_u = sum_i x_i cos(pi (2i+1) u / 2N)
//   idct1d  : x_i = sum_u y_u cos(pi (2i+1) u / 2N)          (full weight on u = 0)
//   idxst1d : x_i = sum_u y_u sin(pi (2i+1) u / 2N)
// The 2D operators below act on row-major M x N arrays, axis 0 has length M.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "dpfpga/common.hpp"

namespace dpfpga {

/// Iterative radix-2 complex FFT of a fixed power-of-two length.
class Fft {
 public:
  explicit Fft(size_t n);
  [[nodiscard]] size_t size() const { return n_; }
  /// In-place transform; `inverse` selects exp(+j...) and performs no scaling.
  void transform(std::complex<Real>* a, bool inverse) const;

 private:
  size_t n_;
  std::vector<size_t> rev_;
  std::vector<std::complex<Real>> twiddle_;  ///< exp(-2 pi j k / n), k < n/2
};

/// 1D transforms for one length. Uses the FFT when N is a power of two and a
/// direct O(N^2) table otherwise.
class Transform1D {
 public:
  explicit Transform1D(size_t n);
  [[nodiscard]] size_t size() const { return n_; }
  void dct(const Real* x, Real* y) const;
  void idct(const Real* y, Real* x) const;
  void idxst(const Real* y, Real* x) const;

 private:
  size_t n_;
  bool pow2_;
  std::vector<std::complex<Real>> fwd_twiddle_;  ///< exp(-j pi k / 2N)
  std::vector<std::complex<Real>> inv_twiddle_;  ///< exp(+j pi k / 2N)
  std::vector<Real> table_;                      ///< cos table for the non power-of-two path
  Fft fft_;
};

/// 2D transforms on row-major (M x N) data.
class Transform2D {
 public:
  Transform2D(size_t m, size_t n);
  [[nodiscard]] size_t rows() const { return m_; }
  [[nodiscard]] size_t cols() const { return n_; }

  /// 2D DCT-II scaled by 4/(M N):   out = 4/(MN) * sum x cos cos.
  void dct2(const Real* in, Real* out) const;
  /// out[m][n] = sum_{u,v} in[u][v] cos(..u..) cos(..v..).
  void idct2(const Real* in, Real* out) const;
  /// out[m][n] = sum_{u,v} in[u][v] sin(..u..) cos(..v..)   (sine along axis 0).
  void idxst_idct(const Real* in, Real* out) const;
  /// out[m][n] = sum_{u,v} in[u][v] cos(..u..) sin(..v..)   (sine along axis 1).
  void idct_idxst(const Real* in, Real* out) const;

 private:
  enum class Kind { kDct, kIdct, kIdxst };
  void apply(Kind row_kind, Kind col_kind, const Real* in, Real* out) const;

  size_t m_, n_;
  Transform1D axis0_, axis1_;
};

}  // namespace dpfpga
