// Nonlinear placement engine (port of BasicPlace.py + NonLinearPlace.py).
#pragma once

#include <memory>
#include <random>
#include <vector>

#include "dpfpga/objective.hpp"
#include "dpfpga/place_data.hpp"

namespace dpfpga {

class NonLinearPlacer {
 public:
  NonLinearPlacer(const Params& params, PlaceDB& db);

  /// Run global placement and (if enabled) legalization; results are written back to the database.
  std::vector<Metrics> run();

  // Access for tests / tools.
  [[nodiscard]] std::vector<Real>& pos() { return pos_; }
  [[nodiscard]] PlaceData& data() { return data_; }
  [[nodiscard]] RegionArray& regions() { return regions_; }

  /// Build the initial positions: movable cells around the centroid of the fixed IOs, fillers spread over their regions.
  void init_positions();
  /// (Re)create the per-resource density operators from the current node sizes.
  void build_regions();

 private:
  void run_global_placement_stage(const StageParams& stage, int& iteration, std::vector<std::vector<std::vector<Metrics>>>& all_metrics);
  void run_legalization(PlaceObjective& model, int& iteration, std::vector<Metrics>& log);
  void load_solution(const std::string& file);
  void apply_solution(const std::vector<int>& node_z);
  void print(const Metrics& m) const;

  const Params& params_;
  PlaceDB& db_;
  PlaceData data_;
  std::vector<Real> pos_;
  RegionArray regions_;
  std::unique_ptr<PlaceObjective> model_;
  std::vector<Real> precond_wl_;
  std::vector<int> node_z_;
  std::mt19937_64 rng_;
  int block_legal_iter_ = 0;  ///< iterations since DSP/RAM were legalized and locked
};

}  // namespace dpfpga
