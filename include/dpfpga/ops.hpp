// Basic placement operators: pin positions, wirelength, preconditioner, bounds, capacity maps.
// All position vectors use the layout [x_0 .. x_{n-1}, y_0 .. y_{n-1}].
#pragma once

#include <array>
#include <vector>

#include "dpfpga/place_data.hpp"

namespace dpfpga::ops {

/// pin_pos = [pin_x (P), pin_y (P)] from node lower-left positions and pin offsets.
void pin_pos(const PlaceData& d, const Real* pos, Real* pin_pos);

/// Accumulate per-pin gradients into per-node gradients (x block then y block of `grad_pos`).
void pin_pos_grad(const PlaceData& d, const Real* grad_pin, Real* grad_pos);

/// Half-perimeter wirelength with the (x, y) direction weights of the database.
[[nodiscard]] Real hpwl(const PlaceData& d, const Real* pin_pos);

/// Weighted-average wirelength. If `grad_pin` is non-null it receives the gradient
/// w.r.t. every pin position (net weights applied, fixed-node pins zeroed).
[[nodiscard]] Real weighted_average_wirelength(const PlaceData& d, const Real* pin_pos, Real inv_gamma,
                                               Real* grad_pin);

/// Wirelength preconditioner: sum over the pins of each node of w_net / (degree - 1).
[[nodiscard]] std::vector<Real> precond_wl(const PlaceData& d);

/// Clamp movable and filler nodes into the layout region.
void move_boundary(const PlaceData& d, Real* pos);

/// Fixed-demand maps (bin area minus available capacity) for LUT, FF, DSP and RAM regions.
[[nodiscard]] std::array<std::vector<Real>, 4> demand_maps(const PlaceDB& db);

}  // namespace dpfpga::ops
