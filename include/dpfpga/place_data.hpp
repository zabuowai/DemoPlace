// Mutable working data shared by the placement operators
// (port of PlaceDataCollectionFPGA in BasicPlace.py).
#pragma once

#include <array>
#include <vector>

#include "dpfpga/placedb.hpp"

namespace dpfpga {

struct PlaceData {
  PlaceData(PlaceDB& db, const Params& params);

  PlaceDB& db;

  // Working copies that the routability optimisation may modify.
  std::vector<Real> pin_offset_x, pin_offset_y;
  std::vector<Real> original_node_size_x, original_node_size_y;
  std::vector<Real> original_pin_offset_x, original_pin_offset_y;

  std::vector<Real> pin_weights;          ///< number of pins per physical node (FFs use ffPinWeight)
  std::vector<int> lut_indices;           ///< indices of LUT nodes
  std::vector<int> flop_lut_indices;      ///< indices of LUT and FF nodes
  std::vector<char> flop_lut_mask, dsp_ram_mask;
  std::vector<char> net_mask;             ///< nets with 2 <= degree < ignore_net_degree
  std::vector<char> pin_mask_fixed;       ///< pins of fixed nodes get no wirelength gradient
  std::vector<Real> node_areas;           ///< size_x * size_y for every node (incl. fillers)
  std::array<double, 4> total_movable_node_area_fence_region{};

  /// Recompute node_areas and the per-resource movable area after node sizes changed.
  void refresh_areas();
};

}  // namespace dpfpga
