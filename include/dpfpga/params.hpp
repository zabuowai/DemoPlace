// User parameters (port of Params.py / paramsFPGA.json).
#pragma once

#include <string>
#include <vector>

#include "dpfpga/common.hpp"

namespace dpfpga {

/// One global-placement stage: {"iteration", "learning_rate", "optimizer", ...}.
struct StageParams {
  int num_bins_x = 512;
  int num_bins_y = 512;
  int iteration = 2000;  ///< maximum number of Lgamma iterations
  double learning_rate = 0.01;
  double learning_rate_decay = 1.0;
  std::string wirelength = "weighted_average";
  std::string optimizer = "nesterov";
  int llambda_density_weight_iteration = 1;
  int lsub_iteration = 1;
  int routability_lsub_iteration = -1;  ///< -1: same as lsub_iteration
};

struct Params {
  // --- inputs / outputs -----------------------------------------------------
  std::string aux_input;
  std::string io_pl;  ///< optional .pl with fixed IO placement (overrides the one in .aux)
  std::string result_dir = "results";
  std::string place_sol;         ///< used when global_place_flag == 0 and legalize_flag == 0
  std::string global_place_sol;  ///< used when global_place_flag == 0 and legalize_flag == 1

  // --- execution --------------------------------------------------------------
  int gpu = 0;  ///< accepted for compatibility; this build is CPU only
  int num_threads = 8;
  int random_seed = 1000;
  std::string dtype = "float32";  ///< accepted for compatibility; see Real in common.hpp
  int deterministic_flag = 1;
  int plot_flag = 0;

  // --- global placement -------------------------------------------------------
  int num_bins_x = 512;
  int num_bins_y = 512;
  std::vector<StageParams> global_place_stages;
  double target_density = 1.0;
  double density_weight = 8e-5;
  double scale_factor = 0.0;
  int ignore_net_degree = 3000;
  double gp_noise_ratio = 0.025;
  int enable_fillers = 1;
  int global_place_flag = 1;
  int legalize_flag = 1;
  double stop_overflow = 0.1;
  double re_place_ref_hpwl = 350000;
  double re_place_lower_pcof = 0.95;
  double re_place_upper_pcof = 1.05;
  double gamma = 5.0;
  int random_center_init_flag = 1;
  int sort_nets_by_degree = 0;

  // --- routability ------------------------------------------------------------
  int routability_opt_flag = 0;
  int route_num_bins_x = 512;
  int route_num_bins_y = 512;
  double node_area_adjust_overflow = 0.15;
  int max_num_area_adjust = 3;
  int adjust_resource_area_flag = 1;
  int adjust_route_area_flag = 1;
  int adjust_pin_area_flag = 1;
  double area_adjust_stop_ratio = 0.01;
  double route_area_adjust_stop_ratio = 0.01;
  double pin_area_adjust_stop_ratio = 0.05;
  double unit_horizontal_capacity = 209;
  double unit_vertical_capacity = 239;
  double unit_pin_capacity = 50;
  double max_route_opt_adjust_rate = 2.0;
  double route_opt_adjust_exponent = 2.0;
  double pin_stretch_ratio = 1.414213562;
  double max_pin_opt_adjust_rate = 1.5;
  double ff_pin_weight = 3.0;
  double inflation_ratio = 1.0;

  // --- timing-driven mode (parsed, not yet supported by this port) -------------
  int timing_driven_flag = 0;
  double timing_iteration_overflow = 0.15;
  int max_num_timing_iteration = 20;
  int timing_interval = 5;
  double criticality_exponent = 9.0;
  double beta_ratio = 1.1;
  int enable_timing_preclustering = 0;
  double lg_alpha = 0.2;
  double lg_beta = 0.1;

  // --- misc outputs (parsed, not ported) ---------------------------------------
  int enable_if = 0;
  int enable_site_routing = 0;
  int write_tcl_flag = 0;
  int write_io_placement_flag = 0;

  /// Load parameters from a JSON file; unknown keys produce a warning.
  void load(const std::string& json_path);

  /// Name derived from the .aux file ("design" if none was given).
  [[nodiscard]] std::string design_name() const;
  [[nodiscard]] std::string solution_file_suffix() const { return "pl"; }
};

}  // namespace dpfpga
