// Electrostatic density model for one resource region
// (port of ops/electric_potential/electric_potential.py + electric_overflow.py).
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "dpfpga/dct.hpp"
#include "dpfpga/place_data.hpp"

namespace dpfpga {

class RegionDensity {
 public:
  /// `fixed_demand` is the per-bin area already occupied / unavailable for this region.
  RegionDensity(const PlaceData& data, int region, std::vector<Real> fixed_demand);

  /// Re-read node sizes from the database (after routability area inflation).
  void reset();
  /// After DSP/RAM legalization the corresponding electric fields are frozen.
  void lock();
  [[nodiscard]] bool locked() const { return locked_; }
  [[nodiscard]] int region() const { return region_; }
  /// True if this region has no movable nodes and no fillers.
  [[nodiscard]] bool empty() const { return num_movable_ == 0 && num_filler_ == 0; }

  /// Electric energy sum(phi * rho). If `need_field` the field maps are prepared for gradient().
  [[nodiscard]] Real energy(const Real* pos, bool need_field);
  /// grad_pos += upstream * d(energy)/d(pos). Requires energy(pos, true) to have been called with the same pos.
  void gradient(Real upstream, const Real* pos, Real* grad_pos) const;
  /// Density overflow area and maximum bin density (fillers excluded).
  [[nodiscard]] std::pair<Real, Real> overflow(const Real* pos) const;

  /// Global node ids handled by this region: movable nodes first, then fillers.
  [[nodiscard]] const std::vector<int>& node_ids() const { return ids_; }
  [[nodiscard]] int num_movable() const { return num_movable_; }
  [[nodiscard]] int num_filler() const { return num_filler_; }

 private:
  void accumulate(const Real* pos, int begin, int end, Real stretch_ratio, std::vector<Real>& map) const;
  void accumulate_one(const Real* pos, int i, int n_all, Real xl, Real yl, Real xh, Real yh, Real inv_bx, Real inv_by,
                      Real target_half_x, Real target_half_y, Real* map_ptr) const;

  const PlaceData& data_;
  int region_;
  std::vector<Real> fixed_demand_;
  std::vector<int> ids_;
  std::vector<Real> size_x_, size_y_;
  int num_movable_ = 0, num_filler_ = 0;
  bool locked_ = false;

  int nbx_, nby_;
  Real bin_size_x_, bin_size_y_;
  Transform2D dct_;
  std::vector<Real> inv_w_, wu_inv_w_, wv_inv_w_;  ///< spectral factors
  std::vector<Real> auv_;                          ///< DCT of the density map
  std::vector<Real> field_x_, field_y_;
  bool fillers_in_map_ = true;
};

}  // namespace dpfpga
