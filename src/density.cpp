#include "dpfpga/density.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace dpfpga {

namespace {

/// Length of the overlap of [lo, hi] with bin k.
inline Real bin_overlap(Real hi, Real lo, int k, Real bin_size) {
  const Real b0 = static_cast<Real>(k) * bin_size;
  return std::max(Real(0), std::min(hi, b0 + bin_size) - std::max(lo, b0));
}

}  // namespace

/// Splats one cell's density onto `map_ptr` (nbx_ x nby_, row-major). Shared by the serial
/// and parallel-reduction paths of accumulate() below.
void RegionDensity::accumulate_one(const Real* pos, int i, int n_all, Real xl, Real yl, Real xh, Real yh, Real inv_bx,
                                   Real inv_by, Real target_half_x, Real target_half_y, Real* map_ptr) const {
  const auto li = static_cast<size_t>(i);
  const auto g = static_cast<size_t>(ids_[li]);
  const Real off_x = Real(0.5) * size_x_[li], off_y = Real(0.5) * size_y_[li];
  const Real cx = pos[g] + off_x, cy = pos[static_cast<size_t>(n_all) + g] + off_y;
  const Real ratio = size_x_[li] * size_y_[li] * Real(0.25);

  const Real reg_x = std::min(cx - xl, xh - cx);
  const Real half_x = std::max(off_x, std::min(target_half_x, reg_x));
  const Real x_lo = cx - half_x, x_hi = cx + half_x;
  const int bxl = std::max(static_cast<int>(x_lo * inv_bx), 0);
  const int bxh = std::min(static_cast<int>(x_hi * inv_bx) + 1, nbx_);

  const Real reg_y = std::min(cy - yl, yh - cy);
  const Real half_y = std::max(off_y, std::min(target_half_y, reg_y));
  const Real y_lo = cy - half_y, y_hi = cy + half_y;
  const int byl = std::max(static_cast<int>(y_lo * inv_by), 0);
  const int byh = std::min(static_cast<int>(y_hi * inv_by) + 1, nby_);

  const Real density = ratio / (half_x * half_y);
  for (int k = bxl; k < bxh; ++k) {
    const Real px = bin_overlap(x_hi, x_lo, k, bin_size_x_) * density;
    for (int h = byl; h < byh; ++h) {
      map_ptr[static_cast<size_t>(k) * static_cast<size_t>(nby_) + static_cast<size_t>(h)] +=
          px * bin_overlap(y_hi, y_lo, h, bin_size_y_);
    }
  }
}

RegionDensity::RegionDensity(const PlaceData& data, int region, std::vector<Real> fixed_demand)
    : data_(data),
      region_(region),
      fixed_demand_(std::move(fixed_demand)),
      nbx_(data.db.num_bins_x),
      nby_(data.db.num_bins_y),
      bin_size_x_(static_cast<Real>((data.db.xh - data.db.xl) / nbx_)),
      bin_size_y_(static_cast<Real>((data.db.yh - data.db.yl) / nby_)),
      dct_(static_cast<size_t>(nbx_), static_cast<size_t>(nby_)) {
  const PlaceDB& db = data.db;
  for (int i = 0; i < db.num_movable_nodes; ++i) {
    if (db.node2fence_region_map[static_cast<size_t>(i)] == region) ids_.push_back(i);
  }
  num_movable_ = static_cast<int>(ids_.size());
  const int f0 = db.filler_start_map[static_cast<size_t>(region)];
  const int f1 = db.filler_start_map[static_cast<size_t>(region) + 1];
  for (int f = f0; f < f1; ++f) ids_.push_back(db.num_physical_nodes + f);
  num_filler_ = f1 - f0;
  reset();

  // Spectral factors. The aspect ratio scales wv because a bin need not be square.
  const size_t M = static_cast<size_t>(nbx_), N = static_cast<size_t>(nby_);
  const double ar = static_cast<double>(bin_size_x_) / bin_size_y_ * db.x_wirelen_wt / db.y_wirelen_wt;
  inv_w_.resize(M * N);
  wu_inv_w_.resize(M * N);
  wv_inv_w_.resize(M * N);
  for (size_t u = 0; u < M; ++u) {
    const double wu = static_cast<double>(u) * 2.0 * std::numbers::pi / static_cast<double>(M);
    for (size_t v = 0; v < N; ++v) {
      const double wv = static_cast<double>(v) * 2.0 * std::numbers::pi / static_cast<double>(N) * ar;
      const double w2 = (u == 0 && v == 0) ? 1.0 : wu * wu + wv * wv;
      const double inv = (u == 0 && v == 0) ? 0.0 : 1.0 / w2;
      inv_w_[u * N + v] = static_cast<Real>(inv);
      wu_inv_w_[u * N + v] = static_cast<Real>(wu * inv);
      wv_inv_w_[u * N + v] = static_cast<Real>(wv * inv);
    }
  }
  auv_.resize(M * N);
}

void RegionDensity::reset() {
  size_x_.resize(ids_.size());
  size_y_.resize(ids_.size());
  for (size_t i = 0; i < ids_.size(); ++i) {
    size_x_[i] = data_.db.node_size_x[static_cast<size_t>(ids_[i])];
    size_y_[i] = data_.db.node_size_y[static_cast<size_t>(ids_[i])];
  }
  // If inflation shrank the fillers to zero size they are dropped from the density map.
  fillers_in_map_ = !(!size_x_.empty() && (size_x_.back() == 0 || size_y_.back() == 0));
}

void RegionDensity::lock() {
  if (region_ > 1) locked_ = true;
}

void RegionDensity::accumulate(const Real* pos, int begin, int end, Real stretch_ratio, std::vector<Real>& map) const {
  const int n_all = data_.db.num_nodes();
  const Real xl = static_cast<Real>(data_.db.xl), yl = static_cast<Real>(data_.db.yl);
  const Real xh = static_cast<Real>(data_.db.xh), yh = static_cast<Real>(data_.db.yh);
  const Real inv_bx = Real(1) / bin_size_x_, inv_by = Real(1) / bin_size_y_;
  const Real target_half_x = Real(0.5) * stretch_ratio * bin_size_x_;
  const Real target_half_y = Real(0.5) * stretch_ratio * bin_size_y_;
  // Scatter-add: bins written by different i can overlap, so parallelizing needs a
  // reduction rather than a plain parallel for. The reduction gives every thread a private
  // copy of `map` (nbx_*nby_ reals), so it only pays off for the big filler-dominated calls
  // (LUT/FF regions can have hundreds of thousands of ids); skip it for small ranges such as
  // the movable-only pass in overflow(), where the reduction's own overhead would dominate.
  Real* map_ptr = map.data();
  const std::size_t cells = map.size();
#ifdef _OPENMP
  if (static_cast<std::size_t>(end - begin) > cells / 4) {
#pragma omp parallel for reduction(+ : map_ptr[0:cells]) schedule(static)
    for (int i = begin; i < end; ++i) {
      accumulate_one(pos, i, n_all, xl, yl, xh, yh, inv_bx, inv_by, target_half_x, target_half_y, map_ptr);
    }
    return;
  }
#endif
  for (int i = begin; i < end; ++i) {
    accumulate_one(pos, i, n_all, xl, yl, xh, yh, inv_bx, inv_by, target_half_x, target_half_y, map_ptr);
  }
}

Real RegionDensity::energy(const Real* pos, bool need_field) {
  if (empty() || locked_) return Real(0);
  const size_t cells = static_cast<size_t>(nbx_) * static_cast<size_t>(nby_);

  // NOTE: like the original, the density (energy) mode always uses the stretch ratio of region 0.
  const Real stretch = static_cast<Real>(data_.db.overflow_inst_density_stretch_ratio[0]);
  std::vector<Real> map(fixed_demand_);
  accumulate(pos, 0, num_movable_, stretch, map);
  if (num_filler_ > 0 && fillers_in_map_) accumulate(pos, num_movable_, num_movable_ + num_filler_, stretch, map);

  const Real inv_bin_area = Real(1) / (bin_size_x_ * bin_size_y_);
  for (auto& v : map) v *= inv_bin_area;

  dct_.dct2(map.data(), auv_.data());
  std::vector<Real> tmp(cells), potential(cells);
  for (size_t k = 0; k < cells; ++k) tmp[k] = auv_[k] * inv_w_[k];
  dct_.idct2(tmp.data(), potential.data());
  double e = 0;
  for (size_t k = 0; k < cells; ++k) e += static_cast<double>(potential[k]) * static_cast<double>(map[k]);

  if (need_field) {
    field_x_.resize(cells);
    field_y_.resize(cells);
    for (size_t k = 0; k < cells; ++k) tmp[k] = auv_[k] * wu_inv_w_[k];
    dct_.idxst_idct(tmp.data(), field_x_.data());
    for (size_t k = 0; k < cells; ++k) tmp[k] = auv_[k] * wv_inv_w_[k];
    dct_.idct_idxst(tmp.data(), field_y_.data());
  }
  return static_cast<Real>(e);
}

void RegionDensity::gradient(Real upstream, const Real* pos, Real* grad_pos) const {
  if (empty() || locked_) return;
  const int n_all = data_.db.num_nodes();
  const Real inv_bx = Real(1) / bin_size_x_, inv_by = Real(1) / bin_size_y_;
  const int count = num_movable_ + num_filler_;
  DPFPGA_PARALLEL_FOR
  for (int i = 0; i < count; ++i) {
    const auto li = static_cast<size_t>(i);
    const auto g = static_cast<size_t>(ids_[li]);
    const Real x_lo = pos[g], x_hi = x_lo + size_x_[li];
    const Real y_lo = pos[static_cast<size_t>(n_all) + g], y_hi = y_lo + size_y_[li];
    const int bxl = std::max(static_cast<int>(x_lo * inv_bx), 0);
    const int bxh = std::min(static_cast<int>(x_hi * inv_bx), nbx_ - 1);
    const int byl = std::max(static_cast<int>(y_lo * inv_by), 0);
    const int byh = std::min(static_cast<int>(y_hi * inv_by), nby_ - 1);
    Real gx = 0, gy = 0;
    for (int k = bxl; k <= bxh; ++k) {
      const Real px = bin_overlap(x_hi, x_lo, k, bin_size_x_);
      for (int h = byl; h <= byh; ++h) {
        const Real area = px * bin_overlap(y_hi, y_lo, h, bin_size_y_);
        const size_t idx = static_cast<size_t>(k) * static_cast<size_t>(nby_) + static_cast<size_t>(h);
        gx += area * field_x_[idx];
        gy += area * field_y_[idx];
      }
    }
    // Every node id is unique, so the accumulation below is race free.
    grad_pos[g] += -gx * upstream;
    grad_pos[static_cast<size_t>(n_all) + g] += -gy * upstream;
  }
}

std::pair<Real, Real> RegionDensity::overflow(const Real* pos) const {
  if (empty() || locked_) return {Real(0), Real(0)};
  std::vector<Real> map(fixed_demand_);
  const Real stretch = static_cast<Real>(data_.db.overflow_inst_density_stretch_ratio[static_cast<size_t>(region_)]);
  accumulate(pos, 0, num_movable_, stretch, map);
  const Real bin_area = bin_size_x_ * bin_size_y_;
  double cost = 0;
  Real max_density = std::numeric_limits<Real>::lowest();
  for (Real v : map) {
    cost += static_cast<double>(std::max(v - bin_area, Real(0)));
    max_density = std::max(max_density, v);
  }
  return {static_cast<Real>(cost), max_density / bin_area};
}

}  // namespace dpfpga
