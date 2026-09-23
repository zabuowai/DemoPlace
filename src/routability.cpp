#include "dpfpga/routability.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace dpfpga {

namespace {

/// RISA wiring distribution weight for a net with `pins` pins.
Real net_wiring_weight(int pins) {
  switch (pins) {
    case 1: case 2: case 3: return Real(1.0);
    case 4: return Real(1.0828);
    case 5: return Real(1.1536);
    case 6: return Real(1.2206);
    case 7: return Real(1.2823);
    case 8: return Real(1.3385);
    case 9: return Real(1.3991);
    case 10: return Real(1.4493);
    default: break;
  }
  if (pins <= 15) return Real(1.6899);
  if (pins <= 20) return Real(1.8924);
  if (pins <= 25) return Real(2.0743);
  if (pins <= 30) return Real(2.2334);
  if (pins <= 35) return Real(2.3892);
  if (pins <= 40) return Real(2.5356);
  if (pins <= 45) return Real(2.6625);
  return Real(2.7933);
}

inline Real overlap_1d(Real lo0, Real hi0, Real lo1, Real hi1) {
  return std::max(std::min(hi0, hi1) - std::max(lo0, lo1), Real(0));
}

Real gaussian_auc(Real mu, Real sigma, Real x_lo, Real x_hi, Real inv_sqrt2) {
  const Real a = inv_sqrt2 / sigma;
  return Real(0.5) * (std::erfc((mu - x_hi) * a) - std::erfc((mu - x_lo) * a));
}

Real smooth_ceil(Real val, Real threshold) {
  const Real r = std::fmod(val, Real(1));
  return val - r + std::min(r / threshold, Real(1));
}

}  // namespace

// ---------------------------------------------------------------------------
// RUDY
// ---------------------------------------------------------------------------
RudyMap::RudyMap(const PlaceData& data, const Params&)
    : data_(data),
      nbx_(data.db.num_routing_grids_x),
      nby_(data.db.num_routing_grids_y),
      bin_x_(static_cast<Real>((data.db.routing_grid_xh - data.db.routing_grid_xl) / nbx_)),
      bin_y_(static_cast<Real>((data.db.routing_grid_yh - data.db.routing_grid_yl) / nby_)) {}

std::vector<Real> RudyMap::compute(const Real* pin_pos) const {
  const PlaceDB& db = data_.db;
  const int num_pins = db.num_pins();
  const Real* px = pin_pos;
  const Real* py = pin_pos + num_pins;
  const auto cells = static_cast<size_t>(nbx_) * static_cast<size_t>(nby_);
  std::vector<Real> horizontal(cells, Real(0)), vertical(cells, Real(0));
  const Real xl = static_cast<Real>(db.routing_grid_xl), yl = static_cast<Real>(db.routing_grid_yl);
  const Real inv_bx = Real(1) / bin_x_, inv_by = Real(1) / bin_y_;
  constexpr Real eps = std::numeric_limits<Real>::epsilon();

  for (int n = 0; n < db.num_nets(); ++n) {
    const int begin = db.flat_net2pin_start_map[static_cast<size_t>(n)];
    const int end = db.flat_net2pin_start_map[static_cast<size_t>(n) + 1];
    Real x_max = std::numeric_limits<Real>::lowest(), x_min = std::numeric_limits<Real>::max();
    Real y_max = x_max, y_min = x_min;
    for (int j = begin; j < end; ++j) {
      const auto p = static_cast<size_t>(db.flat_net2pin_map[static_cast<size_t>(j)]);
      x_max = std::max(px[p], x_max);
      x_min = std::min(px[p], x_min);
      y_max = std::max(py[p], y_max);
      y_min = std::min(py[p], y_min);
    }
    const int bxl = std::max(static_cast<int>((x_min - xl) * inv_bx), 0);
    const int bxh = std::min(static_cast<int>((x_max - xl) * inv_bx) + 1, nbx_);
    const int byl = std::max(static_cast<int>((y_min - yl) * inv_by), 0);
    const int byh = std::min(static_cast<int>((y_max - yl) * inv_by) + 1, nby_);
    const Real wt = net_wiring_weight(end - begin) * db.net_weights[static_cast<size_t>(n)];
    for (int x = bxl; x < bxh; ++x) {
      const Real bx0 = xl + static_cast<Real>(x) * bin_x_;
      const Real ox = overlap_1d(x_min, x_max, bx0, bx0 + bin_x_);
      for (int y = byl; y < byh; ++y) {
        const Real by0 = yl + static_cast<Real>(y) * bin_y_;
        const Real overlap = ox * overlap_1d(y_min, y_max, by0, by0 + bin_y_) * wt;
        const size_t idx = static_cast<size_t>(x) * static_cast<size_t>(nby_) + static_cast<size_t>(y);
        horizontal[idx] += overlap / (y_max - y_min + eps);
        vertical[idx] += overlap / (x_max - x_min + eps);
      }
    }
  }
  const Real bin_area = bin_x_ * bin_y_;
  std::vector<Real> util(cells);
  for (size_t k = 0; k < cells; ++k) {
    const Real h = std::abs(horizontal[k] / (bin_area * static_cast<Real>(db.unit_horizontal_capacity)));
    const Real v = std::abs(vertical[k] / (bin_area * static_cast<Real>(db.unit_vertical_capacity)));
    util[k] = std::max(h, v);
  }
  return util;
}

// ---------------------------------------------------------------------------
// Pin utilisation
// ---------------------------------------------------------------------------
PinUtilizationMap::PinUtilizationMap(const PlaceData& data, const Params& params)
    : data_(data),
      stretch_ratio_(static_cast<Real>(params.pin_stretch_ratio)),
      unit_pin_capacity_(static_cast<Real>(params.unit_pin_capacity)),
      nbx_(data.db.num_routing_grids_x),
      nby_(data.db.num_routing_grids_y),
      bin_x_(static_cast<Real>((data.db.routing_grid_xh - data.db.routing_grid_xl) / nbx_)),
      bin_y_(static_cast<Real>((data.db.routing_grid_yh - data.db.routing_grid_yl) / nby_)) {
  reset();
}

void PinUtilizationMap::reset() {
  const auto n = static_cast<size_t>(data_.db.num_physical_nodes);
  half_x_.resize(n);
  half_y_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    half_x_[i] = Real(0.5) * std::max(data_.db.node_size_x[i], bin_x_ * stretch_ratio_);
    half_y_[i] = Real(0.5) * std::max(data_.db.node_size_y[i], bin_y_ * stretch_ratio_);
  }
}

std::vector<Real> PinUtilizationMap::compute(const Real* pos) const {
  const PlaceDB& db = data_.db;
  const int n_all = db.num_nodes();
  const auto cells = static_cast<size_t>(nbx_) * static_cast<size_t>(nby_);
  std::vector<Real> map(cells, Real(0));
  const Real xl = static_cast<Real>(db.routing_grid_xl), yl = static_cast<Real>(db.routing_grid_yl);
  const Real inv_bx = Real(1) / bin_x_, inv_by = Real(1) / bin_y_;
  for (int i = 0; i < db.num_physical_nodes; ++i) {
    const auto k = static_cast<size_t>(i);
    const Real cx = pos[k] + db.node_size_x[k] / 2, cy = pos[static_cast<size_t>(n_all) + k] + db.node_size_y[k] / 2;
    const Real x_min = cx - half_x_[k], x_max = cx + half_x_[k];
    const Real y_min = cy - half_y_[k], y_max = cy + half_y_[k];
    const int bxl = std::max(static_cast<int>((x_min - xl) * inv_bx), 0);
    const int bxh = std::min(static_cast<int>((x_max - xl) * inv_bx) + 1, nbx_);
    const int byl = std::max(static_cast<int>((y_min - yl) * inv_by), 0);
    const int byh = std::min(static_cast<int>((y_max - yl) * inv_by) + 1, nby_);
    const Real density = data_.pin_weights[k] / (half_x_[k] * half_y_[k] * 4);
    for (int x = bxl; x < bxh; ++x) {
      const Real bx0 = xl + static_cast<Real>(x) * bin_x_;
      const Real ox = overlap_1d(x_min, x_max, bx0, bx0 + bin_x_);
      for (int y = byl; y < byh; ++y) {
        const Real by0 = yl + static_cast<Real>(y) * bin_y_;
        map[static_cast<size_t>(x) * static_cast<size_t>(nby_) + static_cast<size_t>(y)] +=
            ox * overlap_1d(y_min, y_max, by0, by0 + bin_y_) * density;
      }
    }
  }
  const Real scale = Real(1) / (bin_x_ * bin_y_ * unit_pin_capacity_);
  for (auto& v : map) v *= scale;
  return map;
}

// ---------------------------------------------------------------------------
// Clustering compatibility
// ---------------------------------------------------------------------------
std::vector<Real> lut_compatibility_areas(const PlaceData& data, const Real* pos) {
  const PlaceDB& db = data.db;
  const int n_all = db.num_nodes();
  const Real sx = db.inst_dem_stddev_x, sy = db.inst_dem_stddev_y;
  const int nbx = static_cast<int>(std::ceil((db.xh - db.xl) / sx));
  const int nby = static_cast<int>(std::ceil((db.yh - db.yl) / sy));
  // Six LUT classes (cluster types 0..5) are required by the area formulas below.
  const int nl = std::max(6, *std::max_element(db.cluster_lut_type.begin(), db.cluster_lut_type.end()) + 1);
  const int ext = std::max(static_cast<int>(std::lround(db.inst_dem_stddev_trunc - Real(0.5))), 0);
  const Real inv_sqrt2 = Real(1) / std::sqrt(Real(2));
  const Real inv_sx = Real(1) / sx, inv_sy = Real(1) / sy;
  const auto stride_x = static_cast<size_t>(nby) * static_cast<size_t>(nl);

  std::vector<Real> dem(static_cast<size_t>(nbx) * stride_x, Real(0));
  std::vector<Real> demand_x(static_cast<size_t>(2 * ext + 1)), demand_y(static_cast<size_t>(2 * ext + 1));

  for (int idx : data.lut_indices) {
    const auto k = static_cast<size_t>(idx);
    const Real nx = pos[k] + Real(0.5) * db.node_size_x[k];
    const Real ny = pos[static_cast<size_t>(n_all) + k] + Real(0.5) * db.node_size_y[k];
    const int type = db.cluster_lut_type[k];
    const int bin_x = static_cast<int>(nx * inv_sx), bin_y = static_cast<int>(ny * inv_sy);
    const int xl = std::max(bin_x - ext, 0), xh = std::min(bin_x + ext + 1, nbx);
    const int yl = std::max(bin_y - ext, 0), yh = std::min(bin_y + ext + 1, nby);
    const Real gx = gaussian_auc(nx, sx, static_cast<Real>(xl) * sx, static_cast<Real>(xh) * sx, inv_sqrt2);
    const Real gy = gaussian_auc(ny, sy, static_cast<Real>(yl) * sy, static_cast<Real>(yh) * sy, inv_sqrt2);
    const Real sf = Real(1) / (gx * gy);
    for (int x = xl; x < xh; ++x) demand_x[static_cast<size_t>(x - xl)] = gaussian_auc(nx, sx, static_cast<Real>(x) * sx, static_cast<Real>(x + 1) * sx, inv_sqrt2);
    for (int y = yl; y < yh; ++y) demand_y[static_cast<size_t>(y - yl)] = gaussian_auc(ny, sy, static_cast<Real>(y) * sy, static_cast<Real>(y + 1) * sy, inv_sqrt2);
    for (int x = xl; x < xh; ++x) {
      for (int y = yl; y < yh; ++y) {
        dem[static_cast<size_t>(x) * stride_x + static_cast<size_t>(y) * static_cast<size_t>(nl) + static_cast<size_t>(type)] +=
            sf * demand_x[static_cast<size_t>(x - xl)] * demand_y[static_cast<size_t>(y - yl)];
      }
    }
  }

  std::vector<Real> area(dem);  // window aggregation is read from `dem`, written to `area`
  const Real bin_area = sx * sy;
  const int total_bins = nbx * nby;
  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < total_bins; ++i) {
    const int bx = i / nby, by = i % nby;
    const int xl = std::max(bx - ext, 0), xh = std::min(bx + ext + 1, nbx);
    const int yl = std::max(by - ext, 0), yh = std::min(by + ext + 1, nby);
    Real* a = &area[static_cast<size_t>(bx) * stride_x + static_cast<size_t>(by) * static_cast<size_t>(nl)];
    for (int x = xl; x < xh; ++x) {
      for (int y = yl; y < yh; ++y) {
        if (x != bx && y != by) {
          const Real* src = &dem[static_cast<size_t>(x) * stride_x + static_cast<size_t>(y) * static_cast<size_t>(nl)];
          for (int l = 0; l < nl; ++l) a[l] += src[l];
        }
      }
    }
    // Instance areas derived from the window demand distribution (LUT pairing rules).
    const Real win_area = static_cast<Real>((xh - xl) * (yh - yl)) * bin_area;
    Real total_dem = 0;
    for (int l = 0; l < nl; ++l) total_dem += a[l];
    Real space = std::max(win_area - total_dem, Real(0));
    const Real total_area = total_dem + space;
    space += space;
    const Real sum23 = a[2] + a[3], sum45 = a[4] + a[5], sum3 = a[3], sum0 = a[0];
    a[0] = (total_dem + sum45 + space) / total_area;
    a[1] = (total_dem + sum3 + sum45 + space) / total_area;
    a[2] = (total_dem + sum23 + sum45 + space) / total_area;
    a[3] = (Real(2) * total_dem - sum0 + space) / total_area;
    a[4] = Real(2);
    a[5] = a[4];
  }
  dem.clear();
  dem.shrink_to_fit();

  std::vector<Real> result(static_cast<size_t>(n_all), Real(0));
  for (int idx : data.lut_indices) {
    const auto k = static_cast<size_t>(idx);
    const Real nx = pos[k] + Real(0.5) * db.node_size_x[k];
    const Real ny = pos[static_cast<size_t>(n_all) + k] + Real(0.5) * db.node_size_y[k];
    const int bin_x = std::clamp(static_cast<int>(nx * inv_sx), 0, nbx - 1);
    const int bin_y = std::clamp(static_cast<int>(ny * inv_sy), 0, nby - 1);
    result[k] = area[static_cast<size_t>(bin_x) * stride_x + static_cast<size_t>(bin_y) * static_cast<size_t>(nl) +
                     static_cast<size_t>(db.cluster_lut_type[k])] / Real(16);
  }
  return result;
}

std::vector<Real> ff_compatibility_areas(const PlaceData& data, const Real* pos) {
  const PlaceDB& db = data.db;
  const int n_all = db.num_nodes();
  const Real sx = db.inst_dem_stddev_x, sy = db.inst_dem_stddev_y;
  const int nbx = static_cast<int>(std::ceil((db.xh - db.xl) / sx));
  const int nby = static_cast<int>(std::ceil((db.yh - db.yl) / sy));
  const int nck = std::max(db.num_cksr, 1), nce = std::max(db.num_ce, 1);
  const int ext = std::max(static_cast<int>(std::lround(db.inst_dem_stddev_trunc - Real(0.5))), 0);
  const Real inv_sqrt2 = Real(1) / std::sqrt(Real(2));
  const Real inv_sx = Real(1) / sx, inv_sy = Real(1) / sy;
  const auto per_bin = static_cast<size_t>(nck) * static_cast<size_t>(nce);
  const size_t stride_x = static_cast<size_t>(nby) * per_bin;

  std::vector<Real> dem(static_cast<size_t>(nbx) * stride_x, Real(0));
  std::vector<Real> demand_x(static_cast<size_t>(2 * ext + 1)), demand_y(static_cast<size_t>(2 * ext + 1));
  for (size_t i = 0; i < db.flop_indices.size(); ++i) {
    const auto k = static_cast<size_t>(db.flop_indices[i]);
    const Real nx = pos[k] + Real(0.5) * db.node_size_x[k];
    const Real ny = pos[static_cast<size_t>(n_all) + k] + Real(0.5) * db.node_size_y[k];
    const auto cksr = static_cast<size_t>(db.flat_ctrlsets[i * 3 + 1]);
    const auto ce = static_cast<size_t>(db.flat_ctrlsets[i * 3 + 2]);
    const int bin_x = static_cast<int>(nx * inv_sx), bin_y = static_cast<int>(ny * inv_sy);
    const int xl = std::max(bin_x - ext, 0), xh = std::min(bin_x + ext + 1, nbx);
    const int yl = std::max(bin_y - ext, 0), yh = std::min(bin_y + ext + 1, nby);
    const Real gx = gaussian_auc(nx, sx, static_cast<Real>(xl) * sx, static_cast<Real>(xh) * sx, inv_sqrt2);
    const Real gy = gaussian_auc(ny, sy, static_cast<Real>(yl) * sy, static_cast<Real>(yh) * sy, inv_sqrt2);
    const Real sf = Real(1) / (gx * gy);
    for (int x = xl; x < xh; ++x) demand_x[static_cast<size_t>(x - xl)] = gaussian_auc(nx, sx, static_cast<Real>(x) * sx, static_cast<Real>(x + 1) * sx, inv_sqrt2);
    for (int y = yl; y < yh; ++y) demand_y[static_cast<size_t>(y - yl)] = gaussian_auc(ny, sy, static_cast<Real>(y) * sy, static_cast<Real>(y + 1) * sy, inv_sqrt2);
    for (int x = xl; x < xh; ++x) {
      for (int y = yl; y < yh; ++y) {
        dem[static_cast<size_t>(x) * stride_x + static_cast<size_t>(y) * per_bin + cksr * static_cast<size_t>(nce) + ce] +=
            sf * demand_x[static_cast<size_t>(x - xl)] * demand_y[static_cast<size_t>(y - yl)];
      }
    }
  }

  std::vector<Real> area(dem);
  const Real half_slice = Real(16) / Real(2);
  const int total_bins = nbx * nby;
  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < total_bins; ++i) {
    const int bx = i / nby, by = i % nby;
    const int xl = std::max(bx - ext, 0), xh = std::min(bx + ext + 1, nbx);
    const int yl = std::max(by - ext, 0), yh = std::min(by + ext + 1, nby);
    Real* a = &area[static_cast<size_t>(bx) * stride_x + static_cast<size_t>(by) * per_bin];
    for (int x = xl; x < xh; ++x) {
      for (int y = yl; y < yh; ++y) {
        if (x != bx && y != by) {
          const Real* src = &dem[static_cast<size_t>(x) * stride_x + static_cast<size_t>(y) * per_bin];
          for (size_t c = 0; c < per_bin; ++c) a[c] += src[c];
        }
      }
    }
    for (int ck = 0; ck < nck; ++ck) {
      Real total_q = 0;
      Real* row = a + static_cast<size_t>(ck) * static_cast<size_t>(nce);
      for (int ce = 0; ce < nce; ++ce) {
        if (row[ce] > Real(0)) total_q += smooth_ceil(row[ce] * Real(0.25), Real(0.25));
      }
      const Real sf = half_slice * smooth_ceil(total_q * Real(0.5), Real(0.5)) / total_q;
      for (int ce = 0; ce < nce; ++ce) {
        if (row[ce] > Real(0)) row[ce] = sf * smooth_ceil(row[ce] * Real(0.25), Real(0.25)) / row[ce];
      }
    }
  }
  dem.clear();
  dem.shrink_to_fit();

  std::vector<Real> result(static_cast<size_t>(n_all), Real(0));
  for (size_t i = 0; i < db.flop_indices.size(); ++i) {
    const auto k = static_cast<size_t>(db.flop_indices[i]);
    const Real nx = pos[k] + Real(0.5) * db.node_size_x[k];
    const Real ny = pos[static_cast<size_t>(n_all) + k] + Real(0.5) * db.node_size_y[k];
    const auto cksr = static_cast<size_t>(db.flat_ctrlsets[i * 3 + 1]);
    const auto ce = static_cast<size_t>(db.flat_ctrlsets[i * 3 + 2]);
    const int bin_x = std::clamp(static_cast<int>(nx * inv_sx), 0, nbx - 1);
    const int bin_y = std::clamp(static_cast<int>(ny * inv_sy), 0, nby - 1);
    result[k] = area[static_cast<size_t>(bin_x) * stride_x + static_cast<size_t>(bin_y) * per_bin + cksr * static_cast<size_t>(nce) + ce] / Real(16);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Area adjustment
// ---------------------------------------------------------------------------
AdjustNodeArea::AdjustNodeArea(PlaceData& data, const Params& params)
    : data_(data),
      params_(params),
      min_route_rate_(static_cast<Real>(1.0 / params.max_route_opt_adjust_rate)),
      min_pin_rate_(static_cast<Real>(1.0 / params.max_pin_opt_adjust_rate)),
      nbx_(data.db.num_routing_grids_x),
      nby_(data.db.num_routing_grids_y),
      bin_x_(static_cast<Real>((data.db.routing_grid_xh - data.db.routing_grid_xl) / nbx_)),
      bin_y_(static_cast<Real>((data.db.routing_grid_yh - data.db.routing_grid_yl) / nby_)) {
  const PlaceDB& db = data.db;
  double movable = 0;
  for (int i = 0; i < db.num_movable_nodes; ++i) {
    if (data.flop_lut_mask[static_cast<size_t>(i)]) movable += static_cast<double>(db.node_size_x[static_cast<size_t>(i)]) * db.node_size_y[static_cast<size_t>(i)];
  }
  double filler = 0;
  for (int f = 0; f < db.filler_start_map[2]; ++f) {
    const auto k = static_cast<size_t>(db.num_physical_nodes + f);
    filler += static_cast<double>(db.node_size_x[k]) * db.node_size_y[k];
  }
  total_place_area_ = movable + filler;
  total_whitespace_area_ = total_place_area_ - movable;
}

std::vector<Real> AdjustNodeArea::node_area_from_map(const Real* pos, const std::vector<Real>& util) const {
  const PlaceDB& db = data_.db;
  const int n_all = db.num_nodes();
  std::vector<Real> out(static_cast<size_t>(db.num_movable_nodes), Real(0));
  const Real xl = static_cast<Real>(db.routing_grid_xl), yl = static_cast<Real>(db.routing_grid_yl);
  const Real inv_bx = Real(1) / bin_x_, inv_by = Real(1) / bin_y_;
  for (int i : data_.flop_lut_indices) {
    const auto k = static_cast<size_t>(i);
    const Real x_min = pos[k], x_max = x_min + db.node_size_x[k];
    const Real y_min = pos[static_cast<size_t>(n_all) + k], y_max = y_min + db.node_size_y[k];
    const int bxl = std::max(static_cast<int>((x_min - xl) * inv_bx), 0);
    const int bxh = std::min(static_cast<int>((x_max - xl) * inv_bx) + 1, nbx_);
    const int byl = std::max(static_cast<int>((y_min - yl) * inv_by), 0);
    const int byh = std::min(static_cast<int>((y_max - yl) * inv_by) + 1, nby_);
    Real area = 0;
    for (int x = bxl; x < bxh; ++x) {
      const Real bx0 = xl + static_cast<Real>(x) * bin_x_;
      const Real ox = overlap_1d(x_min, x_max, bx0, bx0 + bin_x_);
      for (int y = byl; y < byh; ++y) {
        const Real by0 = yl + static_cast<Real>(y) * bin_y_;
        area += ox * overlap_1d(y_min, y_max, by0, by0 + bin_y_) * util[static_cast<size_t>(x) * static_cast<size_t>(nby_) + static_cast<size_t>(y)];
      }
    }
    out[k] = area;
  }
  return out;
}

AreaAdjustFlags AdjustNodeArea::run(const Real* pos, const std::vector<Real>* resource_areas,
                                    const std::vector<Real>* route_utilization,
                                    const std::vector<Real>* pin_utilization) {
  PlaceDB& db = data_.db;
  AreaAdjustFlags flags{true, resource_areas != nullptr, route_utilization != nullptr, pin_utilization != nullptr};
  if (!(flags.resource || flags.pin || flags.route)) return {};

  const int n_mov = db.num_movable_nodes;
  const int n_phys = db.num_physical_nodes;
  const int n_lut_fillers = db.filler_start_map[1];
  const int n_slice_fillers = db.filler_start_map[2];
  const int n_flop_fillers = n_slice_fillers - n_lut_fillers;

  std::vector<Real> old_area(static_cast<size_t>(n_mov));
  double old_lut_sum = 0, old_flop_sum = 0;
  for (int i = 0; i < n_mov; ++i) {
    const auto k = static_cast<size_t>(i);
    old_area[k] = db.node_size_x[k] * db.node_size_y[k];
    if (db.lut_mask[k]) old_lut_sum += old_area[k];
    if (db.flop_mask[k]) old_flop_sum += old_area[k];
  }
  double old_filler_lut_sum = 0, old_filler_flop_sum = 0;
  for (int f = 0; f < n_slice_fillers; ++f) {
    const auto k = static_cast<size_t>(n_phys + f);
    const double a = static_cast<double>(db.node_size_x[k]) * db.node_size_y[k];
    (f < n_lut_fillers ? old_filler_lut_sum : old_filler_flop_sum) += a;
  }
  const double old_filler_sum = old_filler_lut_sum + old_filler_flop_sum;

  std::vector<Real> route_area, pin_area;
  if (flags.route) {
    std::vector<Real> clamped(route_utilization->size());
    for (size_t k = 0; k < clamped.size(); ++k) {
      clamped[k] = std::clamp(static_cast<Real>(std::pow((*route_utilization)[k], static_cast<Real>(params_.route_opt_adjust_exponent))),
                              min_route_rate_, static_cast<Real>(params_.max_route_opt_adjust_rate));
    }
    route_area = node_area_from_map(pos, clamped);
    for (auto& v : route_area) v *= static_cast<Real>(params_.inflation_ratio);
  }
  if (flags.pin) {
    std::vector<Real> clamped(pin_utilization->size());
    for (size_t k = 0; k < clamped.size(); ++k) {
      clamped[k] = std::clamp((*pin_utilization)[k], min_pin_rate_, static_cast<Real>(params_.max_pin_opt_adjust_rate));
    }
    pin_area = node_area_from_map(pos, clamped);
    for (int i = 0; i < n_mov; ++i) {
      const auto k = static_cast<size_t>(i);
      pin_area[k] *= data_.pin_weights[k] / (db.node_size_x[k] * db.node_size_y[k] * static_cast<Real>(params_.unit_pin_capacity));
    }
  }

  std::vector<Real> increment(static_cast<size_t>(n_mov), Real(0));
  for (int i = 0; i < n_mov; ++i) {
    const auto k = static_cast<size_t>(i);
    Real target = 0;
    if (flags.resource) target = std::max(target, (*resource_areas)[k]);
    if (flags.route) target = std::max(target, route_area[k]);
    if (flags.pin) target = std::max(target, pin_area[k]);
    // With no map at all the increment is zero; otherwise it is relu(target - old).
    increment[k] = std::max(target - old_area[k], Real(0));
  }
  double inc_lut_sum = 0, inc_flop_sum = 0;
  for (int i = 0; i < n_mov; ++i) {
    const auto k = static_cast<size_t>(i);
    if (db.lut_mask[k]) inc_lut_sum += increment[k];
    if (db.flop_mask[k]) inc_flop_sum += increment[k];
  }
  auto scale_for = [&](double capacity_left, double inc_sum) {
    if (inc_sum == 0) return 1.0;
    return std::max(std::min(capacity_left / inc_sum, 1.0), 0.0);
  };
  const double scale_lut = scale_for(total_place_area_ / 2.0 - old_lut_sum, inc_lut_sum);
  const double scale_flop = scale_for(total_place_area_ / 2.0 - old_flop_sum, inc_flop_sum);

  std::vector<Real> new_area(old_area);
  for (int i = 0; i < n_mov; ++i) {
    const auto k = static_cast<size_t>(i);
    if (db.lut_mask[k]) new_area[k] += static_cast<Real>(increment[k] * scale_lut);
    if (db.flop_mask[k]) new_area[k] += static_cast<Real>(increment[k] * scale_flop);
  }
  const double inc_sum = inc_lut_sum * scale_lut + inc_flop_sum * scale_flop;
  const double old_movable_sum = old_lut_sum + old_flop_sum;
  const double new_movable_sum = old_movable_sum + inc_sum;
  const double inc_ratio = inc_sum / old_movable_sum;

  if (inc_sum > 0) {
    info("area_increment = {:.6E}, area_increment / movable = {:.6g}, area_adjust_stop_ratio = {:.6g}", inc_sum, inc_ratio,
         params_.area_adjust_stop_ratio);
    info("area_increment / total_place_area = {:.6g}, area_increment / filler = {:.6g}, area_increment / total_whitespace_area = {:.6g}",
         inc_sum / total_place_area_, inc_sum / old_filler_sum, inc_sum / total_whitespace_area_);
  }

  auto relu_sum = [&](const std::vector<Real>& target) {
    double s = 0;
    for (int i = 0; i < n_mov; ++i) {
      const auto k = static_cast<size_t>(i);
      if (data_.flop_lut_mask[k]) s += std::max(target[k] - old_area[k], Real(0));
    }
    return s / old_movable_sum;
  };
  if (flags.resource) {
    const double r = relu_sum(*resource_areas);
    flags.resource = r > params_.route_area_adjust_stop_ratio;
    info("resource_area_increment_ratio = {:.6g}, resource_area_adjust_stop_ratio = {:.6g}", r, params_.route_area_adjust_stop_ratio);
  }
  if (flags.route) {
    const double r = relu_sum(route_area);
    flags.route = r > params_.route_area_adjust_stop_ratio;
    info("route_area_increment_ratio = {:.6g}, route_area_adjust_stop_ratio = {:.6g}", r, params_.route_area_adjust_stop_ratio);
  }
  if (flags.pin) {
    const double r = relu_sum(pin_area);
    flags.pin = r > params_.pin_area_adjust_stop_ratio;
    info("pin_area_increment_ratio = {:.6g}, pin_area_adjust_stop_ratio = {:.6g}", r, params_.pin_area_adjust_stop_ratio);
  }
  flags.adjust_area = (inc_ratio > params_.area_adjust_stop_ratio) && (flags.resource || flags.route || flags.pin);
  if (!flags.adjust_area) return flags;

  // Inflate movable nodes (area ratio -> linear ratio).
  std::vector<Real> ratio(static_cast<size_t>(n_mov));
  double ratio_sum = 0;
  Real ratio_max = 0;
  for (int i = 0; i < n_mov; ++i) {
    const auto k = static_cast<size_t>(i);
    const Real r = new_area[k] / old_area[k];
    ratio_sum += static_cast<double>(r);
    ratio_max = std::max(ratio_max, r);
    ratio[k] = std::sqrt(r);
    db.node_size_x[k] *= ratio[k];
    db.node_size_y[k] *= ratio[k];
  }
  info("inflation ratio for movable nodes: avg/max {:.6g}/{:.6g}", ratio_sum / std::max(n_mov, 1), static_cast<double>(ratio_max));

  if (new_movable_sum + old_filler_sum > total_place_area_) {
    const double new_lut_sum = old_lut_sum + inc_lut_sum * scale_lut;
    const double new_flop_sum = old_flop_sum + inc_flop_sum * scale_flop;
    if (n_lut_fillers > 0) {
      const Real len = static_cast<Real>(std::sqrt(std::max(total_place_area_ / 2 - new_lut_sum, 0.0) / n_lut_fillers));
      for (int f = 0; f < n_lut_fillers; ++f) {
        db.node_size_x[static_cast<size_t>(n_phys + f)] = len;
        db.node_size_y[static_cast<size_t>(n_phys + f)] = len;
      }
    }
    if (n_flop_fillers > 0) {
      const Real len = static_cast<Real>(std::sqrt(std::max(total_place_area_ / 2 - new_flop_sum, 0.0) / n_flop_fillers));
      for (int f = n_lut_fillers; f < n_slice_fillers; ++f) {
        db.node_size_x[static_cast<size_t>(n_phys + f)] = len;
        db.node_size_y[static_cast<size_t>(n_phys + f)] = len;
      }
    }
  }
  info("new total movable nodes area {:.3E}, total_place_area {:.3E}", new_movable_sum, total_place_area_);

  // Pin offsets follow the inflated node sizes (uses the updated sizes, as in the original kernel).
  for (int i = 0; i < n_mov; ++i) {
    const auto k = static_cast<size_t>(i);
    const Real r = (ratio[k] - 1) / 2;
    for (int j = db.flat_node2pin_start_map[k]; j < db.flat_node2pin_start_map[k + 1]; ++j) {
      const auto pin = static_cast<size_t>(db.flat_node2pin_map[static_cast<size_t>(j)]);
      data_.pin_offset_x[pin] += r * db.node_size_x[k];
      data_.pin_offset_y[pin] += r * db.node_size_y[k];
    }
  }
  return flags;
}

}  // namespace dpfpga
