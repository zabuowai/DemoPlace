// Checks the min-cost assignment against brute force on random small instances.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "dpfpga/legalize_dsp_ram.hpp"

using namespace dpfpga;

int main() {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> pos(0, 30), wt(0.5, 2.0);
  int fails = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const int nb = 1 + trial % 6, ns = nb + trial % 3;
    std::vector<double> b(2 * nb), s(2 * ns), w(nb);
    for (auto& v : b) v = pos(rng);
    for (auto& v : s) v = pos(rng);
    for (auto& v : w) v = wt(rng);
    auto cost = [&](int i, int j) {
      return (std::abs(b[2 * i] - s[2 * j]) + std::abs(b[2 * i + 1] - s[2 * j + 1])) * w[i] * 100.0;
    };
    // brute force over all injective maps
    std::vector<int> perm(ns);
    std::iota(perm.begin(), perm.end(), 0);
    double best = 1e300;
    std::sort(perm.begin(), perm.end());
    do {
      double c = 0;
      for (int i = 0; i < nb; ++i) c += cost(i, perm[i]);
      best = std::min(best, c);
    } while (std::next_permutation(perm.begin(), perm.end()));
    auto got = assign_blocks_to_sites(b, s, w, 1e9, 10.0, 100.0);  // all arcs present -> must be optimal
    double c = 0;
    std::vector<int> used(ns, 0);
    bool valid = true;
    for (int i = 0; i < nb; ++i) {
      c += cost(i, got[i]);
      valid &= (++used[got[i]] == 1);
    }
    if (!valid || std::abs(c - best) > 1e-6) ++fails;
  }
  std::printf(fails == 0 ? "assignment tests passed\n" : "assignment: %d mismatches\n", fails);
  return fails ? 1 : 0;
}
