#include "dpfpga/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dpfpga::ops {

void pin_pos(const PlaceData& d, const Real* pos, Real* out) {
  const int num_nodes = d.db.num_nodes();
  const int num_pins = d.db.num_pins();
  const Real* x = pos;
  const Real* y = pos + num_nodes;
  DPFPGA_PARALLEL_FOR
  for (int p = 0; p < num_pins; ++p) {
    const auto node = static_cast<size_t>(d.db.pin2node_map[static_cast<size_t>(p)]);
    out[p] = d.pin_offset_x[static_cast<size_t>(p)] + x[node];
    out[num_pins + p] = d.pin_offset_y[static_cast<size_t>(p)] + y[node];
  }
}

void pin_pos_grad(const PlaceData& d, const Real* grad_pin, Real* grad_pos) {
  const int num_nodes = d.db.num_nodes();
  const int num_pins = d.db.num_pins();
  const int n_phys = d.db.num_physical_nodes;
  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < n_phys; ++i) {
    Real gx = 0, gy = 0;
    for (int j = d.db.flat_node2pin_start_map[static_cast<size_t>(i)]; j < d.db.flat_node2pin_start_map[static_cast<size_t>(i) + 1]; ++j) {
      const auto pin = static_cast<size_t>(d.db.flat_node2pin_map[static_cast<size_t>(j)]);
      gx += grad_pin[pin];
      gy += grad_pin[static_cast<size_t>(num_pins) + pin];
    }
    grad_pos[i] += gx;
    grad_pos[num_nodes + i] += gy;
  }
}

Real hpwl(const PlaceData& d, const Real* pin_pos) {
  const int num_pins = d.db.num_pins();
  const Real* x = pin_pos;
  const Real* y = pin_pos + num_pins;
  const int num_nets = d.db.num_nets();
  double total = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(static)
#endif
  for (int n = 0; n < num_nets; ++n) {
    if (!d.net_mask[static_cast<size_t>(n)]) continue;
    Real max_x = std::numeric_limits<Real>::lowest(), min_x = std::numeric_limits<Real>::max();
    Real max_y = max_x, min_y = min_x;
    for (int j = d.db.flat_net2pin_start_map[static_cast<size_t>(n)]; j < d.db.flat_net2pin_start_map[static_cast<size_t>(n) + 1]; ++j) {
      const auto p = static_cast<size_t>(d.db.flat_net2pin_map[static_cast<size_t>(j)]);
      min_x = std::min(min_x, x[p]);
      max_x = std::max(max_x, x[p]);
      min_y = std::min(min_y, y[p]);
      max_y = std::max(max_y, y[p]);
    }
    const Real w = d.db.net_weights[static_cast<size_t>(n)];
    total += static_cast<double>((max_x - min_x) * w * d.db.x_wirelen_wt + (max_y - min_y) * w * d.db.y_wirelen_wt);
  }
  return static_cast<Real>(total);
}

Real weighted_average_wirelength(const PlaceData& d, const Real* pin_pos, Real inv_gamma, Real* grad_pin) {
  const int num_pins = d.db.num_pins();
  const Real* x = pin_pos;
  const Real* y = pin_pos + num_pins;
  const int num_nets = d.db.num_nets();
  if (grad_pin) std::fill(grad_pin, grad_pin + 2 * static_cast<size_t>(num_pins), Real(0));
  double total = 0;
#ifdef _OPENMP
  // schedule(runtime): the actual policy is chosen once per run in place_fpga() from
  // deterministic_flag. Net cost varies a lot with pin count, so schedule(dynamic) balances
  // load better than schedule(static) -- but which thread's partial sum a given net lands in
  // then depends on runtime timing, and float addition isn't associative, so two runs of the
  // identical config can differ by an ULP here and (in this chaotic optimizer) diverge
  // completely after enough iterations. schedule(static) fixes the iteration-to-thread
  // mapping regardless of timing, trading some load-balance for exact reproducibility.
#pragma omp parallel for reduction(+ : total) schedule(runtime)
#endif
  for (int n = 0; n < num_nets; ++n) {
    if (!d.net_mask[static_cast<size_t>(n)]) continue;
    const int begin = d.db.flat_net2pin_start_map[static_cast<size_t>(n)];
    const int end = d.db.flat_net2pin_start_map[static_cast<size_t>(n) + 1];
    Real x_max = std::numeric_limits<Real>::lowest(), x_min = std::numeric_limits<Real>::max();
    Real y_max = x_max, y_min = x_min;
    for (int j = begin; j < end; ++j) {
      const auto p = static_cast<size_t>(d.db.flat_net2pin_map[static_cast<size_t>(j)]);
      x_max = std::max(x[p], x_max);
      x_min = std::min(x[p], x_min);
      y_max = std::max(y[p], y_max);
      y_min = std::min(y[p], y_min);
    }
    Real xexp_x = 0, xexp_nx = 0, exp_x = 0, exp_nx = 0;
    Real yexp_y = 0, yexp_ny = 0, exp_y = 0, exp_ny = 0;
    for (int j = begin; j < end; ++j) {
      const auto p = static_cast<size_t>(d.db.flat_net2pin_map[static_cast<size_t>(j)]);
      const Real ex = std::exp((x[p] - x_max) * inv_gamma), enx = std::exp((x_min - x[p]) * inv_gamma);
      xexp_x += x[p] * ex;
      xexp_nx += x[p] * enx;
      exp_x += ex;
      exp_nx += enx;
      const Real ey = std::exp((y[p] - y_max) * inv_gamma), eny = std::exp((y_min - y[p]) * inv_gamma);
      yexp_y += y[p] * ey;
      yexp_ny += y[p] * eny;
      exp_y += ey;
      exp_ny += eny;
    }
    const Real w = d.db.net_weights[static_cast<size_t>(n)];
    total += static_cast<double>((xexp_x / exp_x - xexp_nx / exp_nx + yexp_y / exp_y - yexp_ny / exp_ny) * w);
    if (!grad_pin) continue;

    const Real b_x = inv_gamma / exp_x, a_x = (Real(1) - b_x * xexp_x) / exp_x;
    const Real b_nx = -inv_gamma / exp_nx, a_nx = (Real(1) - b_nx * xexp_nx) / exp_nx;
    const Real b_y = inv_gamma / exp_y, a_y = (Real(1) - b_y * yexp_y) / exp_y;
    const Real b_ny = -inv_gamma / exp_ny, a_ny = (Real(1) - b_ny * yexp_ny) / exp_ny;
    for (int j = begin; j < end; ++j) {
      const auto p = static_cast<size_t>(d.db.flat_net2pin_map[static_cast<size_t>(j)]);
      if (d.pin_mask_fixed[p]) continue;  // no gradient for fixed IO pins
      const Real ex = std::exp((x[p] - x_max) * inv_gamma), enx = std::exp((x_min - x[p]) * inv_gamma);
      const Real ey = std::exp((y[p] - y_max) * inv_gamma), eny = std::exp((y_min - y[p]) * inv_gamma);
      grad_pin[p] = ((a_x + b_x * x[p]) * ex - (a_nx + b_nx * x[p]) * enx) * w;
      grad_pin[static_cast<size_t>(num_pins) + p] = ((a_y + b_y * y[p]) * ey - (a_ny + b_ny * y[p]) * eny) * w;
    }
  }
  return static_cast<Real>(total);
}

std::vector<Real> precond_wl(const PlaceData& d) {
  std::vector<Real> out(static_cast<size_t>(d.db.num_nodes()), Real(0));
  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < d.db.num_physical_nodes; ++i) {
    for (int j = d.db.flat_node2pin_start_map[static_cast<size_t>(i)]; j < d.db.flat_node2pin_start_map[static_cast<size_t>(i) + 1]; ++j) {
      const int net = d.db.pin2net_map[static_cast<size_t>(d.db.flat_node2pin_map[static_cast<size_t>(j)])];
      const int degree = d.db.flat_net2pin_start_map[static_cast<size_t>(net) + 1] - d.db.flat_net2pin_start_map[static_cast<size_t>(net)];
      if (degree > 1) out[static_cast<size_t>(i)] += d.db.net_weights[static_cast<size_t>(net)] / (static_cast<Real>(degree) - Real(1));
    }
  }
  return out;
}

void move_boundary(const PlaceData& d, Real* pos) {
  const int n = d.db.num_nodes();
  const int n_movable = d.db.num_movable_nodes;
  const int n_filler = d.db.num_filler_nodes;
  Real* x = pos;
  Real* y = pos + n;
  const Real xl = static_cast<Real>(d.db.xl), yl = static_cast<Real>(d.db.yl);
  const Real xh = static_cast<Real>(d.db.xh), yh = static_cast<Real>(d.db.yh);
  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < n; ++i) {
    if (i < n_movable || i >= n - n_filler) {
      const auto k = static_cast<size_t>(i);
      x[i] = std::min(std::max(x[i], xl), xh - d.db.node_size_x[k]);
      y[i] = std::min(std::max(y[i], yl), yh - d.db.node_size_y[k]);
    }
  }
}

std::array<std::vector<Real>, 4> demand_maps(const PlaceDB& db) {
  const int nbx = db.num_bins_x, nby = db.num_bins_y;
  const auto cells = static_cast<size_t>(nbx) * static_cast<size_t>(nby);
  std::array<std::vector<Real>, 4> cap;  // capacity of slice, (unused FF alias), DSP, RAM
  for (auto& c : cap) c.assign(cells, Real(0));

  const Real bin_w = static_cast<Real>(db.width()) / static_cast<Real>(nbx);
  const Real bin_h = static_cast<Real>(db.height()) / static_cast<Real>(nby);
  auto overlap = [](int k, Real bin, Real lo, Real len) {
    const Real b0 = static_cast<Real>(k) * bin, b1 = b0 + bin;
    return std::max(Real(0), std::min(lo + len, b1) - std::max(lo, b0));
  };
  for (int s = 0; s < db.width() * db.height(); ++s) {
    const int t = db.site_type_map[static_cast<size_t>(s)];
    int which;
    switch (t) {
      case kSiteSlice: which = 0; break;
      case kSiteDsp: which = 2; break;
      case kSiteRam: which = 3; break;
      default: continue;
    }
    const int rw = s / db.height();
    const int cl = s % db.height();
    const Real node_x = db.resource_size_x[static_cast<size_t>(which == 0 ? 1 : which)];
    const Real node_y = db.resource_size_y[static_cast<size_t>(which == 0 ? 1 : which)];
    const Real col = std::round(static_cast<Real>(cl) / node_y) * node_y;
    const int i_lo = static_cast<int>(static_cast<Real>(rw) / bin_w);
    const int j_lo = static_cast<int>(col / bin_h);
    const int i_hi = std::min(static_cast<int>((static_cast<Real>(rw) + node_x) / bin_w), nbx - 1);
    const int j_hi = std::min(static_cast<int>((col + node_y) / bin_h), nby - 1);
    auto& map = cap[static_cast<size_t>(which)];
    for (int i = i_lo; i <= i_hi; ++i) {
      const Real w = overlap(i, bin_w, static_cast<Real>(rw), node_x);
      for (int j = j_lo; j <= j_hi; ++j) {
        map[static_cast<size_t>(i) * static_cast<size_t>(nby) + static_cast<size_t>(j)] += w * overlap(j, bin_h, col, node_y);
      }
    }
  }
  cap[1] = cap[0];  // FFs share the slice capacity map

  const Real bin_area = static_cast<Real>((db.xh - db.xl) / nbx * ((db.yh - db.yl) / nby));
  std::array<std::vector<Real>, 4> fixed;
  for (size_t r = 0; r < 4; ++r) {
    fixed[r].resize(cells);
    for (size_t k = 0; k < cells; ++k) fixed[r][k] = bin_area - cap[r][k];
  }
  return fixed;
}

}  // namespace dpfpga::ops
