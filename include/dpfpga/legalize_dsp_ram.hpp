// DSP / RAM legalization at the end of global placement
// (port of ops/dsp_ram_legalization: min-cost bipartite assignment).
#pragma once

#include <vector>

#include "dpfpga/place_data.hpp"

namespace dpfpga {

struct MoveStats {
  double max_move = 0;
  double avg_move = 0;
};

/// Assign every movable instance of `region` (kDSP or kRAM) to a distinct legal site, minimising
/// sum(|dx|+|dy| * wirelength_precond). `pos` (lower-left corners, layout [x..., y...]) is updated.
MoveStats legalize_dsp_ram(const PlaceData& data, std::vector<Real>& pos, int region,
                           const std::vector<Real>& precond_wl);

/// Generic min-cost assignment of `num_blocks` blocks to `num_sites` sites. Exposed for testing.
/// `blocks_xy` / `sites_xy` are (x, y) pairs; returns the chosen site of each block.
[[nodiscard]] std::vector<int> assign_blocks_to_sites(const std::vector<double>& blocks_xy,
                                                      const std::vector<double>& sites_xy,
                                                      const std::vector<double>& block_weight,
                                                      double dist_init, double dist_incr, double cost_scale);

}  // namespace dpfpga
