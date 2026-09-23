#include "dpfpga/place_data.hpp"

namespace dpfpga {

PlaceData::PlaceData(PlaceDB& database, const Params& params) : db(database) {
  pin_offset_x = db.pin_offset_x;
  pin_offset_y = db.pin_offset_y;
  original_node_size_x = db.node_size_x;
  original_node_size_y = db.node_size_y;
  original_pin_offset_x = pin_offset_x;
  original_pin_offset_y = pin_offset_y;

  const auto n_phys = static_cast<size_t>(db.num_physical_nodes);
  pin_weights.resize(n_phys);
  for (size_t i = 0; i < n_phys; ++i) {
    pin_weights[i] = static_cast<Real>(db.flat_node2pin_start_map[i + 1] - db.flat_node2pin_start_map[i]);
  }
  flop_lut_mask.assign(n_phys, 0);
  dsp_ram_mask.assign(n_phys, 0);
  for (size_t i = 0; i < n_phys; ++i) {
    if (db.flop_mask[i]) pin_weights[i] = static_cast<Real>(params.ff_pin_weight);
    flop_lut_mask[i] = db.flop_mask[i] || db.lut_mask[i];
    dsp_ram_mask[i] = db.dsp_mask[i] || db.ram_mask[i];
    if (db.lut_mask[i]) lut_indices.push_back(static_cast<int>(i));
    if (flop_lut_mask[i]) flop_lut_indices.push_back(static_cast<int>(i));
  }

  net_mask.resize(static_cast<size_t>(db.num_nets()));
  for (int n = 0; n < db.num_nets(); ++n) {
    const int degree = db.flat_net2pin_start_map[static_cast<size_t>(n) + 1] - db.flat_net2pin_start_map[static_cast<size_t>(n)];
    net_mask[static_cast<size_t>(n)] = (degree >= 2 && degree < params.ignore_net_degree);
  }
  pin_mask_fixed.resize(static_cast<size_t>(db.num_pins()));
  for (int p = 0; p < db.num_pins(); ++p) {
    pin_mask_fixed[static_cast<size_t>(p)] = db.pin2node_map[static_cast<size_t>(p)] >= db.num_movable_nodes;
  }
  refresh_areas();
}

void PlaceData::refresh_areas() {
  node_areas.resize(db.node_size_x.size());
  for (size_t i = 0; i < node_areas.size(); ++i) node_areas[i] = db.node_size_x[i] * db.node_size_y[i];
  total_movable_node_area_fence_region.fill(0.0);
  for (int i = 0; i < db.num_physical_nodes; ++i) {
    const int r = db.node2fence_region_map[static_cast<size_t>(i)];
    if (r < kNumMovableRegions) total_movable_node_area_fence_region[static_cast<size_t>(r)] += node_areas[static_cast<size_t>(i)];
  }
}

}  // namespace dpfpga
