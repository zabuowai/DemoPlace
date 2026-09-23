#include "dpfpga/params.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_map>

#include "dpfpga/json.hpp"

namespace dpfpga {

namespace {

StageParams parse_stage(const json::Value& v) {
  StageParams s;
  for (const auto& [key, val] : v.as_object()) {
    if (key == "num_bins_x") s.num_bins_x = val.as_int();
    else if (key == "num_bins_y") s.num_bins_y = val.as_int();
    else if (key == "iteration") s.iteration = val.as_int();
    else if (key == "learning_rate") s.learning_rate = val.as_number();
    else if (key == "learning_rate_decay") s.learning_rate_decay = val.as_number();
    else if (key == "wirelength") s.wirelength = val.as_string();
    else if (key == "optimizer") s.optimizer = val.as_string();
    else if (key == "Llambda_density_weight_iteration") s.llambda_density_weight_iteration = val.as_int();
    else if (key == "Lsub_iteration") s.lsub_iteration = val.as_int();
    else if (key == "routability_Lsub_iteration") s.routability_lsub_iteration = val.as_int();
    else warn("unknown global_place_stages key '{}' ignored", key);
  }
  if (s.routability_lsub_iteration < 0) s.routability_lsub_iteration = s.lsub_iteration;
  return s;
}

}  // namespace

void Params::load(const std::string& json_path) {
  std::ifstream in(json_path, std::ios::binary);
  if (!in) throw PlacerError("cannot open parameter file " + json_path);
  std::stringstream ss;
  ss << in.rdbuf();
  const json::Value root = json::parse(ss.str());
  if (!root.is_object()) throw PlacerError("parameter file must contain a JSON object");

  using Setter = std::function<void(const json::Value&)>;
  std::unordered_map<std::string, Setter> table;
  auto num = [&](const char* k, auto& field) {
    table[k] = [&field](const json::Value& v) { field = static_cast<std::remove_reference_t<decltype(field)>>(v.as_number()); };
  };
  auto str = [&](const char* k, std::string& field) {
    table[k] = [&field](const json::Value& v) { field = v.as_string(); };
  };

  str("aux_input", aux_input);
  str("io_pl", io_pl);
  str("result_dir", result_dir);
  str("place_sol", place_sol);
  str("global_place_sol", global_place_sol);
  str("dtype", dtype);
  num("gpu", gpu);
  num("num_threads", num_threads);
  num("random_seed", random_seed);
  num("deterministic_flag", deterministic_flag);
  num("plot_flag", plot_flag);
  num("num_bins_x", num_bins_x);
  num("num_bins_y", num_bins_y);
  num("target_density", target_density);
  num("density_weight", density_weight);
  num("scale_factor", scale_factor);
  num("ignore_net_degree", ignore_net_degree);
  num("gp_noise_ratio", gp_noise_ratio);
  num("enable_fillers", enable_fillers);
  num("global_place_flag", global_place_flag);
  num("legalize_flag", legalize_flag);
  num("stop_overflow", stop_overflow);
  num("RePlAce_ref_hpwl", re_place_ref_hpwl);
  num("RePlAce_LOWER_PCOF", re_place_lower_pcof);
  num("RePlAce_UPPER_PCOF", re_place_upper_pcof);
  num("gamma", gamma);
  num("random_center_init_flag", random_center_init_flag);
  num("sort_nets_by_degree", sort_nets_by_degree);
  num("routability_opt_flag", routability_opt_flag);
  num("route_num_bins_x", route_num_bins_x);
  num("route_num_bins_y", route_num_bins_y);
  num("node_area_adjust_overflow", node_area_adjust_overflow);
  num("max_num_area_adjust", max_num_area_adjust);
  num("adjust_resource_area_flag", adjust_resource_area_flag);
  num("adjust_route_area_flag", adjust_route_area_flag);
  num("adjust_pin_area_flag", adjust_pin_area_flag);
  num("area_adjust_stop_ratio", area_adjust_stop_ratio);
  num("route_area_adjust_stop_ratio", route_area_adjust_stop_ratio);
  num("pin_area_adjust_stop_ratio", pin_area_adjust_stop_ratio);
  num("unit_horizontal_capacity", unit_horizontal_capacity);
  num("unit_vertical_capacity", unit_vertical_capacity);
  num("unit_pin_capacity", unit_pin_capacity);
  num("max_route_opt_adjust_rate", max_route_opt_adjust_rate);
  num("route_opt_adjust_exponent", route_opt_adjust_exponent);
  num("pin_stretch_ratio", pin_stretch_ratio);
  num("max_pin_opt_adjust_rate", max_pin_opt_adjust_rate);
  num("ffPinWeight", ff_pin_weight);
  num("inflation_ratio", inflation_ratio);
  num("timing_driven_flag", timing_driven_flag);
  num("timing_iteration_overflow", timing_iteration_overflow);
  num("max_num_timing_iteration", max_num_timing_iteration);
  num("timing_interval", timing_interval);
  num("criticality_exponent", criticality_exponent);
  num("beta_ratio", beta_ratio);
  num("enableTimingPreclustering", enable_timing_preclustering);
  num("lg_alpha", lg_alpha);
  num("lg_beta", lg_beta);
  num("enable_if", enable_if);
  num("enable_site_routing", enable_site_routing);
  num("write_tcl_flag", write_tcl_flag);
  num("write_io_placement_flag", write_io_placement_flag);
  table["global_place_stages"] = [this](const json::Value& v) {
    global_place_stages.clear();
    if (v.is_array()) {
      for (const auto& e : v.as_array()) global_place_stages.push_back(parse_stage(e));
    } else if (v.is_object()) {
      global_place_stages.push_back(parse_stage(v));
    }
  };
  // Keys that exist in paramsFPGA.json but have no effect in a CPU-only, Bookshelf-only port.
  for (const char* k : {"scl_file", "instance_file", "pin_file", "net_file", "routing_file", "util_file",
                        "pickle_file", "load_pickle", "detailed_place_engine", "detailed_place_command",
                        "dump_global_place_solution_flag", "dump_legalize_solution_flag", "detailed_place_flag",
                        "interchange_device", "interchange_netlist"}) {
    table[k] = [](const json::Value&) {};
  }

  for (const auto& [key, val] : root.as_object()) {
    auto it = table.find(key);
    if (it == table.end()) {
      warn("unknown parameter '{}' ignored", key);
      continue;
    }
    it->second(val);
  }

  if (global_place_flag && global_place_stages.empty()) {
    throw PlacerError("global_place_stages is required when global_place_flag is set");
  }
  if (gpu) {
    warn("gpu=1 requested but this build is CPU-only; running on the CPU");
    gpu = 0;
  }
  if (num_threads < 1) num_threads = 1;
}

std::string Params::design_name() const {
  if (aux_input.empty()) return "design";
  std::string name = std::filesystem::path(aux_input).filename().string();
  for (const char* ext : {".aux", ".AUX"}) {
    if (auto p = name.find(ext); p != std::string::npos) name.erase(p, 4);
  }
  return name;
}

}  // namespace dpfpga
