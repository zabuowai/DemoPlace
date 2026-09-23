// Placement objective: wirelength + density penalty with the elfPlace-style weight schedules
// (port of PlaceObj.py) and Nesterov's accelerated gradient optimizer
// (port of NesterovAcceleratedGradientOptimizer.py).
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "dpfpga/density.hpp"
#include "dpfpga/ops.hpp"
#include "dpfpga/params.hpp"
#include "dpfpga/place_data.hpp"

namespace dpfpga {

using RegionArray = std::array<std::unique_ptr<RegionDensity>, kNumMovableRegions>;
using Vec4 = std::array<Real, kNumMovableRegions>;

/// Metrics evaluated at one optimisation step.
struct Metrics {
  int iteration = 0;
  Real hpwl = 0;
  Real objective = 0;
  Vec4 overflow{};
  Vec4 max_density{};
  Vec4 density{};  ///< electric energy per region
  [[nodiscard]] std::string to_string() const;
};

class PlaceObjective {
 public:
  PlaceObjective(PlaceData& data, const Params& params, RegionArray& regions);

  /// Objective value and preconditioned gradient (length 2 * num_nodes) at `pos`.
  Real obj_and_grad(const std::vector<Real>& pos, std::vector<Real>& grad);

  /// Metrics that need no gradient: HPWL, overflow, electric energies, max densities.
  [[nodiscard]] Metrics evaluate(const std::vector<Real>& pos) const;

  // ---- schedules ----------------------------------------------------------------
  void initialize_density_weight(const std::vector<Real>& pos);
  void reset_density_weight(const std::vector<Real>& pos, Real ratio);
  void update_density_weight(const Metrics& cur);
  void update_gamma(const Vec4& overflow);
  /// Estimate the initial step size by moving a small step along the gradient.
  Real estimate_initial_learning_rate(std::vector<Real>& pos);

  [[nodiscard]] bool density_weight_initialized() const { return initialized_; }
  [[nodiscard]] Real gamma() const { return gamma_; }
  [[nodiscard]] const Vec4& density_weight() const { return density_weight_; }
  [[nodiscard]] const std::vector<Real>& precond_wl() const { return precond_wl_; }

  /// Per-region "allowed to move" flags (set on the first density-weight update).
  std::optional<std::array<bool, kNumMovableRegions>> update_mask;
  std::optional<std::array<bool, kNumMovableRegions>> lock_mask;

 private:
  /// L1 norm of the gradient of the wirelength alone.
  Real wirelength_grad_l1(const std::vector<Real>& pos);
  void precondition(std::vector<Real>& grad) const;

  PlaceData& data_;
  const Params& params_;
  RegionArray& regions_;
  std::vector<Real> precond_wl_;

  Real gamma_ = 0;
  std::array<Real, 4> base_gamma_{}, gamma_k_{}, gamma_b_{}, gamma_wt_{};

  Vec4 density_weight_{};          ///< committed weight per region (u * s)
  Vec4 density_weight_u_{};
  Vec4 init_density_{};
  Vec4 weight_grad_precond_{};
  Vec4 quad_penalty_coeff_{};
  Real density_quad_coeff_ = 1000;
  Real step_size_inc_low_ = Real(1.05), step_size_inc_high_ = Real(1.06);
  Real step_size_ = 0;
  bool initialized_ = false;

  mutable std::vector<Real> pin_pos_, grad_pin_;
};

/// Nesterov's accelerated gradient method with backtracking line search (e-place, Algorithm 2).
class NesterovOptimizer {
 public:
  using ObjGrad = std::function<Real(const std::vector<Real>&, std::vector<Real>&)>;
  using Constraint = std::function<void(std::vector<Real>&)>;

  NesterovOptimizer(ObjGrad obj_and_grad, Constraint constraint) : obj_and_grad_(std::move(obj_and_grad)), constraint_(std::move(constraint)) {}

  void set_lr(Real lr) { lr_ = lr; }
  [[nodiscard]] Real lr() const { return lr_; }
  /// Forget all history (equivalent to loading the initial optimizer state).
  void reset() { initialized_ = false; }
  /// One iteration; `pos` (the reference solution v_k) is updated in place.
  void step(std::vector<Real>& pos);
  /// Objective at the previous reference solution (obj_{k-1}).
  [[nodiscard]] Real previous_objective() const { return obj_k_1_; }

 private:
  ObjGrad obj_and_grad_;
  Constraint constraint_;
  Real lr_ = 0;
  bool initialized_ = false;
  std::vector<Real> u_k_, g_k_, v_k_1_, g_k_1_, v_kp1_, g_kp1_, u_kp1_;
  Real obj_k_ = 0, obj_k_1_ = 0, a_k_ = 1, alpha_k_ = 0;
};

}  // namespace dpfpga
