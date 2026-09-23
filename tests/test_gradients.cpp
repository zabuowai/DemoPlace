// Finite-difference checks of the wirelength gradient and a consistency check of the density force.
#include <cmath>
#include <numeric>

#include "dpfpga/global_placer.hpp"

using namespace dpfpga;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  Params p;
  p.aux_input = argv[1];
  p.gp_noise_ratio = 0;
  PlaceDB db;
  db.read(p);
  db.initialize(p);
  NonLinearPlacer placer(p, db);
  auto& pos = placer.pos();
  PlaceData& d = placer.data();
  const int n = db.num_nodes();

  // Spread the movable cells over the layout so the gradients are non-trivial.
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> ux(5, db.xh - 5), uy(5, db.yh - 5);
  for (int i = 0; i < db.num_movable_nodes; ++i) {
    pos[static_cast<size_t>(i)] = static_cast<Real>(ux(rng));
    pos[static_cast<size_t>(n) + static_cast<size_t>(i)] = static_cast<Real>(uy(rng));
  }

  // ---- weighted-average wirelength: analytic vs central finite difference ----------------------
  std::vector<Real> pin(2 * static_cast<size_t>(db.num_pins())), gpin(pin.size()), grad(pos.size(), 0);
  const Real inv_gamma = Real(1) / Real(4.0);
  ops::pin_pos(d, pos.data(), pin.data());
  (void)ops::weighted_average_wirelength(d, pin.data(), inv_gamma, gpin.data());
  ops::pin_pos_grad(d, gpin.data(), grad.data());
  double worst = 0, gmax = 0;
  for (int t = 0; t < 40; ++t) {
    const int i = (t * 79) % db.num_movable_nodes;
    for (int axis = 0; axis < 2; ++axis) {
      const size_t k = static_cast<size_t>(axis) * static_cast<size_t>(n) + static_cast<size_t>(i);
      const Real h = Real(1e-2), saved = pos[k];
      pos[k] = saved + h;
      ops::pin_pos(d, pos.data(), pin.data());
      const double fp = ops::weighted_average_wirelength(d, pin.data(), inv_gamma, nullptr);
      pos[k] = saved - h;
      ops::pin_pos(d, pos.data(), pin.data());
      const double fm = ops::weighted_average_wirelength(d, pin.data(), inv_gamma, nullptr);
      pos[k] = saved;
      const double fd = (fp - fm) / (2.0 * h);
      worst = std::max(worst, std::abs(fd - grad[k]));
      gmax = std::max(gmax, std::abs(double(grad[k])));
    }
  }
  std::printf("WA wirelength gradient: worst abs error %.3e (max |grad| %.3e, ratio %.3e) %s\n", worst, gmax, worst / gmax,
              worst / gmax < 1e-5 ? "OK" : "FAIL");
  int fails = worst / gmax < 1e-5 ? 0 : 1;

  // ---- electric force: it is an approximation of -dE/dx, check it correlates strongly ------------
  for (int r = 0; r < 2; ++r) {  // LUT and FF regions
    RegionDensity& region = *placer.regions()[static_cast<size_t>(r)];
    const Real e0 = region.energy(pos.data(), true);
    std::vector<Real> g(pos.size(), 0);
    region.gradient(1, pos.data(), g.data());
    double dot = 0, na = 0, nb = 0;
    int used = 0;
    for (int t = 0; t < 60 && t < region.num_movable(); ++t) {
      const int id = region.node_ids()[static_cast<size_t>((t * 37) % region.num_movable())];
      for (int axis = 0; axis < 2; ++axis) {
        const size_t k = static_cast<size_t>(axis) * static_cast<size_t>(n) + static_cast<size_t>(id);
        const Real h = Real(0.05), saved = pos[k];
        pos[k] = saved + h;
        const double fp = region.energy(pos.data(), false);
        pos[k] = saved - h;
        const double fm = region.energy(pos.data(), false);
        pos[k] = saved;
        const double fd = (fp - fm) / (2.0 * h);
        dot += fd * g[k];
        na += fd * fd;
        nb += static_cast<double>(g[k]) * g[k];
        ++used;
      }
    }
    const double cosine = dot / std::sqrt(na * nb + 1e-300);
    std::printf("region %d electric energy %.6g, force-vs-FD cosine similarity %.4f over %d components %s\n", r,
                static_cast<double>(e0), cosine, used, cosine > 0.9 ? "OK" : "FAIL");
    fails += cosine > 0.9 ? 0 : 1;
  }
  return fails;
}
