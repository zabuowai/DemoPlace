// Temporary stub: replaced by the ported legalizer.
#include "dpfpga/legalize_lut_ff.hpp"
namespace dpfpga {
struct LutFfLegalizer::Impl {};
LutFfLegalizer::LutFfLegalizer(PlaceData&, const Params&) {}
LutFfLegalizer::~LutFfLegalizer() = default;
void LutFfLegalizer::run(std::vector<Real>&, const std::vector<Real>&, std::vector<int>&) { throw PlacerError("LUT/FF legalizer not built"); }
}  // namespace dpfpga
