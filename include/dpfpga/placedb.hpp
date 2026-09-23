// FPGA placement database (port of PlaceDB.py + the C++ place_io module).
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "dpfpga/common.hpp"
#include "dpfpga/params.hpp"

namespace dpfpga {

/// Flat, index based database. Node order is: movable nodes, fixed IO nodes
/// ("terminals"), then filler nodes (added by initialize()).
struct PlaceDB {
  // ---- counts ----------------------------------------------------------------
  int num_movable_nodes = 0;
  int num_terminals = 0;
  int num_physical_nodes = 0;  ///< movable + terminals
  int num_filler_nodes = 0;
  std::array<int, 5> node_count{};  ///< LUT, FF, DSP, RAM, IO

  [[nodiscard]] int num_nodes() const { return num_physical_nodes + num_filler_nodes; }
  [[nodiscard]] int num_nets() const { return static_cast<int>(net_names.size()); }
  [[nodiscard]] int num_pins() const { return static_cast<int>(pin2node_map.size()); }
  [[nodiscard]] int num_tnets() const { return static_cast<int>(tnet2net_map.size()); }

  // ---- nodes -----------------------------------------------------------------
  std::vector<std::string> node_names, node_types;
  std::unordered_map<std::string, int> node_name2id;
  std::vector<Real> node_size_x, node_size_y;  ///< physical nodes followed by fillers
  std::vector<Real> node_x, node_y;            ///< physical nodes, lower-left corner
  std::vector<int> node_z;
  std::vector<int> node2fence_region_map;  ///< Region id for every physical node
  std::vector<int> lut_type, cluster_lut_type;
  std::vector<int> flop_indices;
  std::vector<int> node2outpin_map;  ///< output pin of each node (0 if none)
  std::vector<int> node2pincount_map, net2pincount_map;

  // ---- pins / nets -------------------------------------------------------------
  std::vector<std::string> pin_names;
  std::vector<Real> pin_offset_x, pin_offset_y;
  std::vector<Real> lg_pin_offset_x, lg_pin_offset_y;
  std::vector<int> pin_type_ids;       ///< 0 out, 1 in, 2 clk, 3 ce, 4 sr
  std::vector<int> pin2nodetype_map;   ///< region of the owning node
  std::vector<int> pin2net_map, pin2node_map;
  std::vector<int> flat_node2pin_map, flat_node2pin_start_map;
  std::vector<std::string> net_names;
  std::unordered_map<std::string, int> net_name2id;
  std::vector<int> flat_net2pin_map, flat_net2pin_start_map;
  std::vector<Real> net_weights;

  // ---- timing nets (built for the legalizer; timing-driven mode not ported) -------
  std::vector<int> tnet2net_map, net2tnet_start_map, flat_tnet2pin_map, snkpin2tnet_map;
  std::vector<Real> tnet_weights, tnet_criticality;

  // ---- flop control sets ---------------------------------------------------------
  std::vector<int> flat_ctrlsets;  ///< (flop node, ck/sr id, ce id) triplets
  std::vector<int> flop2ctrlset_map;
  int num_cksr = 0, num_ce = 0;

  // ---- device ---------------------------------------------------------------------
  int num_sites_x = 0, num_sites_y = 0;
  std::vector<int> site_type_map;       ///< [x * num_sites_y + y]
  std::vector<Real> lg_site_xy;         ///< [(x * num_sites_y + y) * 2 + {0,1}]
  std::vector<Real> dsp_site_xy, ram_site_xy;  ///< flattened (x, y) pairs
  std::vector<Real> flat_region_boxes;  ///< (xl, yl, xh, yh) per column box
  std::vector<int> flat_region_boxes_start;
  std::vector<int> spiral_accessor;     ///< flattened (dx, dy) pairs
  double xl = 0, yl = 0, xh = 0, yh = 0;
  double row_height = 0, site_width = 0;
  [[nodiscard]] int width() const { return num_sites_x; }
  [[nodiscard]] int height() const { return num_sites_y; }

  // ---- placement parameters derived in initialize() --------------------------------
  Real x_wirelen_wt = 0.7, y_wirelen_wt = 1.2;
  Real inst_dem_stddev_trunc = 2.5;
  Real gp_inst_stddev = 0, inst_dem_stddev_x = 0, inst_dem_stddev_y = 0;
  Real nbr_dist_end = 0;
  std::array<Real, 4> resource_size_x{}, resource_size_y{};
  std::array<double, 4> filler_size_x{}, filler_size_y{};
  std::array<double, 4> target_overflow{};
  std::array<double, 4> overflow_inst_density_stretch_ratio{};
  int num_bins_x = 512, num_bins_y = 512;
  double bin_size_x = 0, bin_size_y = 0;
  int num_routing_grids_x = 0, num_routing_grids_y = 0;
  double routing_grid_xl = 0, routing_grid_yl = 0, routing_grid_xh = 0, routing_grid_yh = 0;
  double unit_horizontal_capacity = 0, unit_vertical_capacity = 0;
  double total_movable_node_area = 0, total_fixed_node_area = 0, total_space_area = 0;
  double total_filler_node_area = 0;

  // fence regions (columns of a given resource): region_boxes[r] is a list of boxes
  std::vector<std::vector<std::array<Real, 4>>> region_boxes;
  std::vector<int> filler_start_map;              ///< size 5
  std::array<int, 4> num_filler_nodes_fence_region{};
  std::array<int, 4> num_movable_nodes_fence_region{};
  std::array<double, 4> total_movable_node_area_fence_region{};

  // ---- masks -----------------------------------------------------------------------
  std::vector<char> lut_mask, flop_mask, dsp_mask, ram_mask;  ///< over physical nodes

  // ---- I/O ---------------------------------------------------------------------------
  /// Parse Bookshelf files referenced by the .aux file and build all maps.
  void read(const Params& params);
  /// Compute filler counts, gamma parameters, etc. (PlaceDB.initialize()).
  void initialize(const Params& params);
  /// Read an additional .pl file to override fixed instance locations.
  void read_pl(const std::string& pl_file);
  /// Intermediate global placement result: "name x y z area".
  void write_gp(const std::string& file) const;
  /// Final solution: "name x y z" with integer coordinates.
  void write_final(const std::string& file) const;
};

}  // namespace dpfpga
