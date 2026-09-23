// Routability and clustering-compatibility operators
// (ports of ops/rudy, ops/pin_utilization, ops/clustering_compatibility, ops/adjust_node_area).
#pragma once

#include <optional>
#include <vector>

#include "dpfpga/place_data.hpp"

namespace dpfpga {

/// RUDY routing-demand estimate; returns the max of horizontal and vertical utilisation per routing bin.
class RudyMap {
 public:
  RudyMap(const PlaceData& data, const Params& params);
  [[nodiscard]] std::vector<Real> compute(const Real* pin_pos) const;

 private:
  const PlaceData& data_;
  int nbx_, nby_;
  Real bin_x_, bin_y_;
};

/// Pin-density map (pins per unit area relative to capacity).
class PinUtilizationMap {
 public:
  PinUtilizationMap(const PlaceData& data, const Params& params);
  void reset();  ///< recompute stretched sizes after node sizes changed
  [[nodiscard]] std::vector<Real> compute(const Real* pos) const;

 private:
  const PlaceData& data_;
  Real stretch_ratio_, unit_pin_capacity_;
  int nbx_, nby_;
  Real bin_x_, bin_y_;
  std::vector<Real> half_x_, half_y_;
};

/// Resource areas of LUTs / FFs that respect the clustering constraints of a slice.
/// Both return a vector of length num_nodes with non-zero entries at LUT (resp. FF) indices.
[[nodiscard]] std::vector<Real> lut_compatibility_areas(const PlaceData& data, const Real* pos);
[[nodiscard]] std::vector<Real> ff_compatibility_areas(const PlaceData& data, const Real* pos);

struct AreaAdjustFlags {
  bool adjust_area = false, resource = false, route = false, pin = false;
};

/// Inflates cell areas according to routing / pin / resource maps.
class AdjustNodeArea {
 public:
  AdjustNodeArea(PlaceData& data, const Params& params);
  /// Any of the maps may be absent. Node sizes / pin offsets in `data` are updated when inflation happens.
  AreaAdjustFlags run(const Real* pos, const std::vector<Real>* resource_areas,
                      const std::vector<Real>* route_utilization, const std::vector<Real>* pin_utilization);

 private:
  [[nodiscard]] std::vector<Real> node_area_from_map(const Real* pos, const std::vector<Real>& util) const;

  PlaceData& data_;
  const Params& params_;
  double total_place_area_, total_whitespace_area_;
  Real min_route_rate_, min_pin_rate_;
  int nbx_, nby_;
  Real bin_x_, bin_y_;
};

}  // namespace dpfpga
