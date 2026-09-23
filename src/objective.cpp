#include "dpfpga/objective.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dpfpga {

namespace {

double l1_norm(const std::vector<Real>& v) {
  double s = 0;
  for (Real x : v) s += std::abs(static_cast<double>(x));
  return s;
}
double l2_norm_sq_diff(const std::vector<Real>& a, const std::vector<Real>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    s += d * d;
  }
  return s;
}

}  // namespace

std::string Metrics::to_string() const {
  std::string s = std::format("iter: {:4d}, HPWL {:.6E}, Overflow [", iteration, static_cast<double>(hpwl));
  for (size_t i = 0; i < overflow.size(); ++i) {
    if (i) s += ", ";
    s += std::format("{:.3E}", static_cast<double>(overflow[i]));
  }
  return s + "]";
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------
PlaceObjective::PlaceObjective(PlaceData& data, const Params& params, RegionArray& regions)
    : data_(data), params_(params), regions_(regions) {
  const PlaceDB& db = data.db;
  precond_wl_ = ops::precond_wl(data);
  pin_pos_.resize(2 * static_cast<size_t>(db.num_pins()));
  grad_pin_.resize(2 * static_cast<size_t>(db.num_pins()));

  // Wirelength gamma schedule (elfPlace): gamma = 10^(k*overflow + b) * base, blended over resource types.
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    base_gamma_[r] = static_cast<Real>(0.5 * params.gamma * (db.bin_size_x + db.bin_size_y));
    gamma_k_[r] = static_cast<Real>(2.0 / (1.0 - db.target_overflow[r]));
    gamma_b_[r] = Real(1) - gamma_k_[r];
    double wt = 0;
    for (int i = 0; i < db.num_physical_nodes; ++i) {
      if (db.node2fence_region_map[static_cast<size_t>(i)] == static_cast<int>(r)) wt += precond_wl_[static_cast<size_t>(i)];
    }
    gamma_wt_[r] = static_cast<Real>(wt);
  }
  update_gamma(Vec4{1, 1, 1, 1});
}

void PlaceObjective::update_gamma(const Vec4& overflow) {
  double total_gamma = 0, total_wt = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    const double g = static_cast<double>(base_gamma_[r]) * std::pow(10.0, static_cast<double>(overflow[r] * gamma_k_[r] + gamma_b_[r]));
    total_gamma += g * static_cast<double>(gamma_wt_[r]);
    total_wt += static_cast<double>(gamma_wt_[r]);
  }
  gamma_ = total_wt > 0 ? static_cast<Real>(total_gamma / total_wt) : base_gamma_[0];
}

Real PlaceObjective::wirelength_grad_l1(const std::vector<Real>& pos) {
  ops::pin_pos(data_, pos.data(), pin_pos_.data());
  (void)ops::weighted_average_wirelength(data_, pin_pos_.data(), Real(1) / gamma_, grad_pin_.data());
  std::vector<Real> g(pos.size(), Real(0));
  ops::pin_pos_grad(data_, grad_pin_.data(), g.data());
  return static_cast<Real>(l1_norm(g));
}

Real PlaceObjective::obj_and_grad(const std::vector<Real>& pos, std::vector<Real>& grad) {
  const int n = data_.db.num_nodes();
  ops::pin_pos(data_, pos.data(), pin_pos_.data());
  const Real wl = ops::weighted_average_wirelength(data_, pin_pos_.data(), Real(1) / gamma_, grad_pin_.data());
  grad.assign(pos.size(), Real(0));
  ops::pin_pos_grad(data_, grad_pin_.data(), grad.data());
  (void)n;

  double density_term = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    RegionDensity& region = *regions_[r];
    const Real e = region.energy(pos.data(), true);
    if (!initialized_ && init_density_[r] == 0 && e != 0) {
      // First evaluation without initialize_density_weight(): record the reference density.
      init_density_[r] = e;
      weight_grad_precond_[r] = Real(1) / e;
      quad_penalty_coeff_[r] = density_quad_coeff_ / 2 * weight_grad_precond_[r];
    }
    const Real q = quad_penalty_coeff_[r];
    density_term += static_cast<double>(density_weight_u_[r]) * static_cast<double>(e * (1 + q * e));
    region.gradient(density_weight_u_[r] * (1 + 2 * q * e), pos.data(), grad.data());
  }
  precondition(grad);
  return wl + static_cast<Real>(density_term);
}

void PlaceObjective::precondition(std::vector<Real>& grad) const {
  const PlaceDB& db = data_.db;
  const int n = db.num_nodes();
  const int n_mov = db.num_movable_nodes;

  std::vector<Real> node_area(data_.node_areas);
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    for (int id : regions_[r]->node_ids()) node_area[static_cast<size_t>(id)] *= density_weight_[r];
  }
  std::array<bool, kNumMovableRegions> blocked{};
  int blocked_count = 0;
  if (update_mask) {
    for (size_t r = 0; r < kNumMovableRegions; ++r) {
      blocked[r] = !(*update_mask)[r];
      blocked_count += blocked[r];
    }
  }
  const bool zero_blocked = update_mask && blocked_count < kNumMovableRegions;  // mirrors the original condition

  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < n; ++i) {
    const auto k = static_cast<size_t>(i);
    const Real precond = std::max(precond_wl_[k] + node_area[k], Real(1));
    grad[k] /= precond;
    grad[static_cast<size_t>(n) + k] /= precond;
  }
  if (zero_blocked) {
    for (size_t r = 0; r < kNumMovableRegions; ++r) {
      if (!blocked[r]) continue;
      for (int id : regions_[r]->node_ids()) {
        grad[static_cast<size_t>(id)] = 0;
        grad[static_cast<size_t>(n) + static_cast<size_t>(id)] = 0;
      }
    }
  }
  (void)n_mov;
}

Metrics PlaceObjective::evaluate(const std::vector<Real>& pos) const {
  Metrics m;
  ops::pin_pos(data_, pos.data(), pin_pos_.data());
  m.hpwl = ops::hpwl(data_, pin_pos_.data());
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    RegionDensity& region = *regions_[r];
    const auto [cost, max_density] = region.overflow(pos.data());
    m.overflow[r] = cost / std::max(static_cast<Real>(data_.total_movable_node_area_fence_region[r]), Real(1));
    m.max_density[r] = max_density;
    m.density[r] = region.energy(pos.data(), false);
  }
  return m;
}

void PlaceObjective::initialize_density_weight(const std::vector<Real>& pos) {
  const Real wl_norm = wirelength_grad_l1(pos);

  std::array<double, kNumMovableRegions> grad_norm{};
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    std::vector<Real> g(pos.size(), Real(0));
    init_density_[r] = regions_[r]->energy(pos.data(), true);
    regions_[r]->gradient(Real(1), pos.data(), g.data());
    grad_norm[r] = l1_norm(g);
  }
  Vec4 s{};
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    weight_grad_precond_[r] = init_density_[r] != 0 ? Real(1) / init_density_[r] : init_density_[r];
    Real u = init_density_[r] * weight_grad_precond_[r];
    u += Real(0.5) * density_quad_coeff_ * u * u;
    density_weight_u_[r] = u;
    s[r] = 1 + density_quad_coeff_ * init_density_[r] * weight_grad_precond_[r];
    quad_penalty_coeff_[r] = density_quad_coeff_ / 2 * weight_grad_precond_[r];
  }
  double density_grad_norm = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) density_grad_norm += static_cast<double>(density_weight_u_[r]) * s[r] * grad_norm[r];

  const Real scale = static_cast<Real>(params_.density_weight * static_cast<double>(wl_norm) / density_grad_norm);
  double u_norm = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    density_weight_u_[r] *= scale;
    u_norm += static_cast<double>(density_weight_u_[r]) * density_weight_u_[r];
  }
  step_size_ = (step_size_inc_low_ - 1) * static_cast<Real>(std::sqrt(u_norm));
  for (size_t r = 0; r < kNumMovableRegions; ++r) density_weight_[r] = density_weight_u_[r] * s[r];
  initialized_ = true;
}

void PlaceObjective::reset_density_weight(const std::vector<Real>& pos, Real ratio) {
  const Real wl_norm = wirelength_grad_l1(pos);
  Vec4 upd{};
  std::array<double, kNumMovableRegions> grad_norm{};
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    std::vector<Real> g(pos.size(), Real(0));
    upd[r] = regions_[r]->energy(pos.data(), true);
    regions_[r]->gradient(Real(1), pos.data(), g.data());
    grad_norm[r] = l1_norm(g);
  }
  Vec4 s{};
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    Real u = upd[r] * weight_grad_precond_[r];
    u += Real(0.5) * density_quad_coeff_ * u * u;
    density_weight_u_[r] = u;
    s[r] = 1 + density_quad_coeff_ * upd[r] * weight_grad_precond_[r];
  }
  double density_grad_norm = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) density_grad_norm += static_cast<double>(density_weight_u_[r]) * s[r] * grad_norm[r];
  const Real scale = static_cast<Real>(static_cast<double>(ratio) * static_cast<double>(wl_norm) / density_grad_norm);
  double u_norm = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    density_weight_u_[r] *= scale;
    u_norm += static_cast<double>(density_weight_u_[r]) * density_weight_u_[r];
  }
  step_size_ = (step_size_inc_low_ - 1) * static_cast<Real>(std::sqrt(u_norm));
  for (size_t r = 0; r < kNumMovableRegions; ++r) density_weight_[r] = density_weight_u_[r] * s[r];
}

void PlaceObjective::update_density_weight(const Metrics& cur) {
  Vec4 norm{}, grad{};
  double grad_sq = 0, norm_sq = 0;
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    norm[r] = cur.density[r] * weight_grad_precond_[r];
    grad[r] = norm[r] + density_quad_coeff_ / 2 * norm[r] * norm[r];
    if (std::isinf(grad[r])) grad[r] = 0;
    grad_sq += static_cast<double>(grad[r]) * grad[r];
    norm_sq += static_cast<double>(norm[r]) * norm[r];
  }
  const double grad_norm = std::sqrt(grad_sq);
  if (grad_norm > 0) {
    for (auto& g : grad) g = static_cast<Real>(g / grad_norm);
  }
  Vec4 s{};
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    density_weight_u_[r] += step_size_ * grad[r];
    s[r] = 1 + density_quad_coeff_ * norm[r];
  }
  Real rate = static_cast<Real>(std::max(std::log(static_cast<double>(density_quad_coeff_) * std::sqrt(norm_sq)), 0.0));
  rate = rate / (1 + rate);
  rate = rate * (step_size_inc_high_ - step_size_inc_low_) + step_size_inc_low_;
  step_size_ *= rate;

  if (!update_mask) {
    std::array<bool, kNumMovableRegions> upd{}, lock{};
    for (size_t r = 0; r < kNumMovableRegions; ++r) {
      upd[r] = cur.overflow[r] >= static_cast<Real>(data_.db.target_overflow[r]);
      lock[r] = cur.overflow[r] < static_cast<Real>(data_.db.target_overflow[r]);
    }
    update_mask = upd;
    lock_mask = lock;
  }
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    if ((*update_mask)[r]) density_weight_[r] = density_weight_u_[r] * s[r];
  }
}

Real PlaceObjective::estimate_initial_learning_rate(std::vector<Real>& pos) {
  std::vector<Real> g_k, g_k_1;
  obj_and_grad(pos, g_k);
  const int n = data_.db.num_nodes();
  Real lr = static_cast<Real>(0.001 * std::min(data_.db.xh - data_.db.xl, data_.db.yh - data_.db.yl) *
                              (n - data_.db.num_terminals));
  lr /= static_cast<Real>(l1_norm(g_k));
  std::vector<Real> x1(pos);
  for (size_t i = 0; i < x1.size(); ++i) x1[i] -= lr * g_k[i];
  obj_and_grad(x1, g_k_1);
  return static_cast<Real>(std::sqrt(l2_norm_sq_diff(pos, x1) / l2_norm_sq_diff(g_k, g_k_1)));
}

// ---------------------------------------------------------------------------
// Nesterov optimizer
// ---------------------------------------------------------------------------
void NesterovOptimizer::step(std::vector<Real>& v_k) {
  const size_t len = v_k.size();
  if (!initialized_) {
    u_k_ = v_k;
    obj_k_ = obj_and_grad_(v_k, g_k_);
    a_k_ = 1;
    v_k_1_.resize(len);
    for (size_t i = 0; i < len; ++i) v_k_1_[i] = v_k[i] - lr_ * g_k_[i];
    obj_k_1_ = obj_and_grad_(v_k_1_, g_k_1_);
    alpha_k_ = static_cast<Real>(std::sqrt(l2_norm_sq_diff(v_k, v_k_1_) / l2_norm_sq_diff(g_k_, g_k_1_)));
    initialized_ = true;
  }
  v_kp1_.resize(len);
  u_kp1_.resize(len);

  const Real a_kp1 = (1 + std::sqrt(4 * a_k_ * a_k_ + 1)) / 2;
  const Real coef = (a_k_ - 1) / a_kp1;
  constexpr int kMaxBacktrack = 10;
  int backtrack = 0;
  Real f_kp1 = 0;
  while (true) {
    for (size_t i = 0; i < len; ++i) {
      u_kp1_[i] = v_k[i] - alpha_k_ * g_k_[i];
      v_kp1_[i] = u_kp1_[i] + coef * (u_kp1_[i] - u_k_[i]);
    }
    constraint_(v_kp1_);
    f_kp1 = obj_and_grad_(v_kp1_, g_kp1_);
    const Real alpha_kp1 = static_cast<Real>(std::sqrt(l2_norm_sq_diff(v_kp1_, v_k) / l2_norm_sq_diff(g_kp1_, g_k_)));
    ++backtrack;
    const bool accept = alpha_kp1 > Real(0.95) * alpha_k_ || backtrack >= kMaxBacktrack;
    alpha_k_ = alpha_kp1;
    if (accept) break;
  }
  // Shift the history: k-1 <- k, k <- k+1.
  obj_k_1_ = obj_k_;
  u_k_ = u_kp1_;
  v_k = v_kp1_;
  g_k_1_ = g_k_;
  g_k_ = g_kp1_;
  obj_k_ = f_kp1;
  a_k_ = a_kp1;
}

}  // namespace dpfpga
