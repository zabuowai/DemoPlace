// Verifies the fast transforms against direct O(N^4) evaluation of their definitions.
#include <cmath>
#include <numbers>
#include <random>

#include "dpfpga/dct.hpp"

using namespace dpfpga;

static double max_err(const std::vector<Real>& a, const std::vector<Real>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(double(a[i]) - double(b[i])));
  return e;
}

static int check(size_t M, size_t N) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(0, 10);
  std::vector<Real> x(M * N), got(M * N), ref(M * N);
  for (auto& v : x) v = static_cast<Real>(dist(rng));
  Transform2D t(M, N);
  const double pi = std::numbers::pi;
  auto c = [&](size_t i, size_t u, size_t L) { return std::cos(pi * double((2 * i + 1) * u) / (2.0 * double(L))); };
  auto s = [&](size_t i, size_t u, size_t L) { return std::sin(pi * double((2 * i + 1) * u) / (2.0 * double(L))); };
  int fails = 0;
  auto report = [&](const char* name, double err) {
    const bool ok = err < 1e-9 * std::max<double>(1, double(M * N));
    std::printf("  %-11s %zux%zu  max|err| = %.3e  %s\n", name, M, N, err, ok ? "OK" : "FAIL");
    fails += !ok;
  };
  // dct2: 4/(MN) sum x cos cos
  t.dct2(x.data(), got.data());
  for (size_t u = 0; u < M; ++u)
    for (size_t v = 0; v < N; ++v) {
      double a = 0;
      for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) a += x[m * N + n] * c(m, u, M) * c(n, v, N);
      ref[u * N + v] = static_cast<Real>(4.0 / double(M * N) * a);
    }
  report("dct2", max_err(got, ref));
  auto run_inv = [&](const char* name, auto&& op, auto&& fu, auto&& fv) {
    op(x.data(), got.data());
    for (size_t m = 0; m < M; ++m)
      for (size_t n = 0; n < N; ++n) {
        double a = 0;
        for (size_t u = 0; u < M; ++u)
          for (size_t v = 0; v < N; ++v) a += x[u * N + v] * fu(m, u) * fv(n, v);
        ref[m * N + n] = static_cast<Real>(a);
      }
    report(name, max_err(got, ref));
  };
  run_inv("idct2", [&](auto* i, auto* o) { t.idct2(i, o); }, [&](size_t m, size_t u) { return c(m, u, M); },
          [&](size_t n, size_t v) { return c(n, v, N); });
  run_inv("idxst_idct", [&](auto* i, auto* o) { t.idxst_idct(i, o); }, [&](size_t m, size_t u) { return s(m, u, M); },
          [&](size_t n, size_t v) { return c(n, v, N); });
  run_inv("idct_idxst", [&](auto* i, auto* o) { t.idct_idxst(i, o); }, [&](size_t m, size_t u) { return c(m, u, M); },
          [&](size_t n, size_t v) { return s(n, v, N); });
  return fails;
}

int main() {
  int fails = 0;
  for (auto [m, n] : {std::pair<size_t, size_t>{8, 16}, {4, 8}, {32, 32}, {6, 10}}) fails += check(m, n);
  std::printf(fails ? "DCT TESTS FAILED\n" : "all DCT tests passed\n");
  return fails ? 1 : 0;
}
