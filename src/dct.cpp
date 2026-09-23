#include "dpfpga/dct.hpp"

#include <cmath>
#include <numbers>

namespace dpfpga {

namespace {
using Complex = std::complex<Real>;
constexpr Real kPi = std::numbers::pi_v<Real>;
bool is_pow2(size_t n) { return n >= 2 && (n & (n - 1)) == 0; }
}  // namespace

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------
Fft::Fft(size_t n) : n_(n) {
  if (!is_pow2(n)) {
    n_ = 0;  // unused placeholder for non power-of-two lengths
    return;
  }
  rev_.resize(n);
  size_t bits = 0;
  while ((size_t{1} << bits) < n) ++bits;
  for (size_t i = 0; i < n; ++i) {
    size_t r = 0;
    for (size_t b = 0; b < bits; ++b) {
      if (i & (size_t{1} << b)) r |= size_t{1} << (bits - 1 - b);
    }
    rev_[i] = r;
  }
  twiddle_.resize(n / 2);
  for (size_t k = 0; k < n / 2; ++k) {
    const double a = -2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n);
    twiddle_[k] = Complex(static_cast<Real>(std::cos(a)), static_cast<Real>(std::sin(a)));
  }
}

void Fft::transform(Complex* a, bool inverse) const {
  for (size_t i = 0; i < n_; ++i) {
    if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
  }
  for (size_t len = 2; len <= n_; len <<= 1) {
    const size_t half = len / 2, step = n_ / len;
    for (size_t base = 0; base < n_; base += len) {
      for (size_t k = 0; k < half; ++k) {
        Complex w = twiddle_[k * step];
        if (inverse) w = std::conj(w);
        const Complex t = a[base + k + half] * w;
        a[base + k + half] = a[base + k] - t;
        a[base + k] += t;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 1D transforms
// ---------------------------------------------------------------------------
Transform1D::Transform1D(size_t n) : n_(n), pow2_(is_pow2(n)), fft_(n) {
  if (pow2_) {
    fwd_twiddle_.resize(n);
    inv_twiddle_.resize(n);
    for (size_t k = 0; k < n; ++k) {
      const double a = std::numbers::pi * static_cast<double>(k) / (2.0 * static_cast<double>(n));
      fwd_twiddle_[k] = Complex(static_cast<Real>(std::cos(a)), static_cast<Real>(-std::sin(a)));
      inv_twiddle_[k] = Complex(static_cast<Real>(std::cos(a)), static_cast<Real>(std::sin(a)));
    }
  } else {
    table_.resize(n * n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t u = 0; u < n; ++u) {
        table_[i * n + u] = static_cast<Real>(
            std::cos(std::numbers::pi * static_cast<double>((2 * i + 1) * u) / (2.0 * static_cast<double>(n))));
      }
    }
  }
}

void Transform1D::dct(const Real* x, Real* y) const {
  if (!pow2_) {
    for (size_t u = 0; u < n_; ++u) {
      Real s = 0;
      for (size_t i = 0; i < n_; ++i) s += x[i] * table_[i * n_ + u];
      y[u] = s;
    }
    return;
  }
  thread_local std::vector<Complex> buf;
  buf.resize(n_);
  // Makhoul's reordering: even samples ascending, odd samples descending.
  for (size_t i = 0; i < n_ / 2; ++i) {
    buf[i] = Complex(x[2 * i], 0);
    buf[n_ - 1 - i] = Complex(x[2 * i + 1], 0);
  }
  fft_.transform(buf.data(), false);
  for (size_t k = 0; k < n_; ++k) y[k] = (buf[k] * fwd_twiddle_[k]).real();
}

void Transform1D::idct(const Real* y, Real* x) const {
  if (!pow2_) {
    for (size_t i = 0; i < n_; ++i) {
      Real s = 0;
      for (size_t u = 0; u < n_; ++u) s += y[u] * table_[i * n_ + u];
      x[i] = s;
    }
    return;
  }
  thread_local std::vector<Complex> buf;
  buf.resize(n_);
  for (size_t k = 0; k < n_; ++k) buf[k] = y[k] * inv_twiddle_[k];
  fft_.transform(buf.data(), true);
  for (size_t m = 0; m < n_ / 2; ++m) {
    x[2 * m] = buf[m].real();
    x[n_ - 1 - 2 * m] = buf[m + n_ / 2].real();
  }
}

void Transform1D::idxst(const Real* y, Real* x) const {
  if (!pow2_) {
    for (size_t i = 0; i < n_; ++i) {
      Real s = 0;
      for (size_t u = 1; u < n_; ++u) {
        s += y[u] * static_cast<Real>(std::sin(std::numbers::pi * static_cast<double>((2 * i + 1) * u) /
                                               (2.0 * static_cast<double>(n_))));
      }
      x[i] = s;
    }
    return;
  }
  // sin(pi (2i+1) u / 2N) = (-1)^i cos(pi (2i+1) (N-u) / 2N): flip the spectrum and reuse idct.
  thread_local std::vector<Real> z;
  z.resize(n_);
  z[0] = 0;
  for (size_t w = 1; w < n_; ++w) z[w] = y[n_ - w];
  idct(z.data(), x);
  for (size_t i = 1; i < n_; i += 2) x[i] = -x[i];
}

// ---------------------------------------------------------------------------
// 2D transforms
// ---------------------------------------------------------------------------
Transform2D::Transform2D(size_t m, size_t n) : m_(m), n_(n), axis0_(m), axis1_(n) {}

void Transform2D::apply(Kind row_kind, Kind col_kind, const Real* in, Real* out) const {
  auto run = [](const Transform1D& t, Kind k, const Real* src, Real* dst) {
    switch (k) {
      case Kind::kDct: t.dct(src, dst); break;
      case Kind::kIdct: t.idct(src, dst); break;
      case Kind::kIdxst: t.idxst(src, dst); break;
    }
  };
  // Axis 1 (contiguous rows) -> out. Each row is an independent 1D transform writing
  // its own disjoint slice of `out`, so this parallelizes with no reduction needed.
  const auto m = static_cast<long long>(m_);
  const auto n = static_cast<long long>(n_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long r = 0; r < m; ++r) {
    run(axis1_, col_kind, in + static_cast<size_t>(r) * n_, out + static_cast<size_t>(r) * n_);
  }
  // Axis 0 (strided columns): likewise independent per column. col_in/col_out are
  // thread_local so each worker thread gets its own scratch buffer.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long c = 0; c < n; ++c) {
    thread_local std::vector<Real> col_in, col_out;
    col_in.resize(m_);
    col_out.resize(m_);
    for (size_t r = 0; r < m_; ++r) col_in[r] = out[r * n_ + static_cast<size_t>(c)];
    run(axis0_, row_kind, col_in.data(), col_out.data());
    for (size_t r = 0; r < m_; ++r) out[r * n_ + static_cast<size_t>(c)] = col_out[r];
  }
}

void Transform2D::dct2(const Real* in, Real* out) const {
  apply(Kind::kDct, Kind::kDct, in, out);
  const Real scale = Real(4) / (static_cast<Real>(m_) * static_cast<Real>(n_));
  for (size_t i = 0; i < m_ * n_; ++i) out[i] *= scale;
}

void Transform2D::idct2(const Real* in, Real* out) const { apply(Kind::kIdct, Kind::kIdct, in, out); }

void Transform2D::idxst_idct(const Real* in, Real* out) const { apply(Kind::kIdxst, Kind::kIdct, in, out); }

void Transform2D::idct_idxst(const Real* in, Real* out) const { apply(Kind::kIdct, Kind::kIdxst, in, out); }

}  // namespace dpfpga
