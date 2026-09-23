// LUT / FF packing and legalization (direct legalization + rip-up / greedy slot assignment).
#pragma once

#include <memory>
#include <vector>

#include "dpfpga/place_data.hpp"

namespace dpfpga {

class LutFfLegalizer {
 public:
  LutFfLegalizer(PlaceData& data, const Params& params);
  ~LutFfLegalizer();
  LutFfLegalizer(const LutFfLegalizer&) = delete;
  LutFfLegalizer& operator=(const LutFfLegalizer&) = delete;

  /// Legalize all LUTs and FFs. `pos` holds lower-left corners ([x..., y...]); LUT/FF entries are replaced
  /// by their slice coordinates and `node_z` receives the BEL index inside the slice.
  void run(std::vector<Real>& pos, const std::vector<Real>& precond_wl, std::vector<int>& node_z);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dpfpga
