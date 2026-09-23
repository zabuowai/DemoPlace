#include "dpfpga/global_placer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "dpfpga/legalize_dsp_ram.hpp"
#include "dpfpga/legalize_lut_ff.hpp"
#include "dpfpga/routability.hpp"

namespace dpfpga {

namespace {

using MetricLog = std::vector<std::vector<std::vector<Metrics>>>;  // [Lgamma][Llambda][Lsub]

Real max_of(const Vec4& v) { return *std::max_element(v.begin(), v.end()); }

bool all_below_target(const Vec4& overflow, const PlaceDB& db) {
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    if (!(overflow[r] < static_cast<Real>(db.target_overflow[r]))) return false;
  }
  return true;
}

}  // namespace

NonLinearPlacer::NonLinearPlacer(const Params& params, PlaceDB& db)
    // The original ties BasicPlaceFPGA's np.random.seed/torch.manual_seed to params.random_seed,
    // then immediately overwrites both with a hardcoded 0 ("Settings to ensure reproducibility")
    // before any placement randomness is drawn, so random_seed has no effect on its GP result.
    // Matched here for behavior parity: this port's own RNG is independent of random_seed too.
    : params_(params), db_(db), data_(db, params), rng_(static_cast<std::mt19937_64::result_type>(0)) {
  init_positions();
  build_regions();
  node_z_.assign(static_cast<size_t>(db.num_movable_nodes), 0);
}

void NonLinearPlacer::init_positions() {
  const int n = db_.num_nodes();
  const int n_phys = db_.num_physical_nodes;
  const int n_mov = db_.num_movable_nodes;
  pos_.assign(2 * static_cast<size_t>(n), Real(0));

  // Centroid of the fixed pins (or the layout centre if the design has no IOs).
  double init_x = 0, init_y = 0;
  if (db_.num_terminals > 0) {
    long long num_pins = 0;
    for (int node = n_mov; node < n_phys; ++node) {
      for (int j = db_.flat_node2pin_start_map[static_cast<size_t>(node)]; j < db_.flat_node2pin_start_map[static_cast<size_t>(node) + 1]; ++j) {
        const auto pin = static_cast<size_t>(db_.flat_node2pin_map[static_cast<size_t>(j)]);
        init_x += static_cast<double>(db_.node_x[static_cast<size_t>(node)] + db_.pin_offset_x[pin]);
        init_y += static_cast<double>(db_.node_y[static_cast<size_t>(node)] + db_.pin_offset_y[pin]);
      }
      num_pins += db_.node2pincount_map[static_cast<size_t>(node)];
    }
    if (num_pins > 0) {
      init_x /= static_cast<double>(num_pins);
      init_y /= static_cast<double>(num_pins);
    }
  } else {
    init_x = 0.5 * (db_.xh - db_.xl);
    init_y = 0.5 * (db_.yh - db_.yl);
  }

  for (int i = 0; i < n_phys; ++i) {
    pos_[static_cast<size_t>(i)] = db_.node_x[static_cast<size_t>(i)];
    pos_[static_cast<size_t>(n) + static_cast<size_t>(i)] = db_.node_y[static_cast<size_t>(i)];
  }
  if (params_.global_place_flag && params_.random_center_init_flag) {
    const double sigma = std::min(db_.xh - db_.xl, db_.yh - db_.yl) * 0.001;
    std::normal_distribution<double> nx(init_x, sigma), ny(init_y, sigma);
    for (int i = 0; i < n_mov; ++i) pos_[static_cast<size_t>(i)] = static_cast<Real>(nx(rng_));
    for (int i = 0; i < n_mov; ++i) pos_[static_cast<size_t>(n) + static_cast<size_t>(i)] = static_cast<Real>(ny(rng_));
  }
  for (int i = 0; i < n_mov; ++i) {  // centre -> lower-left corner
    pos_[static_cast<size_t>(i)] -= Real(0.5) * db_.node_size_x[static_cast<size_t>(i)];
    pos_[static_cast<size_t>(n) + static_cast<size_t>(i)] -= Real(0.5) * db_.node_size_y[static_cast<size_t>(i)];
  }

  // Fillers are spread uniformly over the columns of their resource type.
  if (db_.num_filler_nodes > 0) {
    for (int r = 0; r < kNumMovableRegions; ++r) {
      const int f0 = db_.filler_start_map[static_cast<size_t>(r)], f1 = db_.filler_start_map[static_cast<size_t>(r) + 1];
      const auto& boxes = db_.region_boxes[static_cast<size_t>(r)];
      if (f1 <= f0 || boxes.empty()) continue;
      double total_area = 0;
      for (const auto& b : boxes) total_area += static_cast<double>(b[2] - b[0]) * static_cast<double>(b[3] - b[1]);
      std::vector<int> counts(boxes.size());
      int assigned = 0;
      for (size_t j = 0; j + 1 < boxes.size(); ++j) {
        const double area = static_cast<double>(boxes[j][2] - boxes[j][0]) * static_cast<double>(boxes[j][3] - boxes[j][1]);
        counts[j] = static_cast<int>(std::nearbyint((f1 - f0) * area / total_area));
        assigned += counts[j];
      }
      counts.back() = (f1 - f0) - assigned;
      int cursor = db_.num_physical_nodes + f0;
      for (size_t j = 0; j < boxes.size(); ++j) {
        std::uniform_real_distribution<double> ux(boxes[j][0], std::max<double>(boxes[j][2] - db_.filler_size_x[static_cast<size_t>(r)], boxes[j][0]));
        std::uniform_real_distribution<double> uy(boxes[j][1], std::max<double>(boxes[j][3] - db_.filler_size_y[static_cast<size_t>(r)], boxes[j][1]));
        for (int k = 0; k < counts[j]; ++k, ++cursor) {
          pos_[static_cast<size_t>(cursor)] = static_cast<Real>(ux(rng_));
          pos_[static_cast<size_t>(n) + static_cast<size_t>(cursor)] = static_cast<Real>(uy(rng_));
        }
      }
    }
  }
}

void NonLinearPlacer::build_regions() {
  auto fixed = ops::demand_maps(db_);
  for (size_t r = 0; r < kNumMovableRegions; ++r) {
    regions_[r] = std::make_unique<RegionDensity>(data_, static_cast<int>(r), std::move(fixed[r]));
  }
}

void NonLinearPlacer::print(const Metrics& m) const { info("{}", m.to_string()); }

std::vector<Metrics> NonLinearPlacer::run() {
  std::vector<Metrics> log;
  int iteration = 0;
  MetricLog all_metrics;

  if (params_.timing_driven_flag) warn("timing-driven placement is not supported by this port; running wirelength-driven placement");
  if (params_.plot_flag) warn("plot_flag is ignored (plotting is not ported)");

  if (params_.global_place_flag) {
    for (const auto& stage : params_.global_place_stages) run_global_placement_stage(stage, iteration, all_metrics);
    if (params_.routability_opt_flag) {
      // Node sizes / pin offsets were inflated during global placement: restore them for legalization.
      db_.node_size_x = data_.original_node_size_x;
      db_.node_size_y = data_.original_node_size_y;
      data_.pin_offset_x = data_.original_pin_offset_x;
      data_.pin_offset_y = data_.original_pin_offset_y;
      data_.refresh_areas();
    }
  } else {
    const std::string& file = params_.legalize_flag ? params_.global_place_sol : params_.place_sol;
    if (file.empty()) throw PlacerError("global_place_flag=0 requires global_place_sol (or place_sol) to be set");
    load_solution(file);
    build_regions();
    model_ = std::make_unique<PlaceObjective>(data_, params_, regions_);
    precond_wl_ = model_->precond_wl();
    Metrics m = model_->evaluate(pos_);
    m.iteration = iteration;
    print(m);
    log.push_back(m);
  }

  if (params_.legalize_flag) run_legalization(*model_, iteration, log);
  apply_solution(node_z_);
  return log;
}

void NonLinearPlacer::run_global_placement_stage(const StageParams& stage, int& iteration, MetricLog& all_metrics) {
  Stopwatch stage_timer;
  build_regions();
  model_ = std::make_unique<PlaceObjective>(data_, params_, regions_);
  PlaceObjective& model = *model_;
  precond_wl_ = model.precond_wl();

  if (stage.optimizer != "nesterov") {
    throw PlacerError("optimizer '" + stage.optimizer + "' is not ported; only \"nesterov\" is supported");
  }
  if (stage.wirelength != "weighted_average") warn("wirelength model '{}' is not ported; using weighted_average", stage.wirelength);
  info("use nesterov optimizer");

  NesterovOptimizer optimizer(
      [&](const std::vector<Real>& p, std::vector<Real>& g) { return model.obj_and_grad(p, g); },
      [&](std::vector<Real>& p) { ops::move_boundary(data_, p.data()); });

  const int n = db_.num_nodes();
  auto initialize_learning_rate = [&] { optimizer.set_lr(model.estimate_initial_learning_rate(pos_)); };

  if (!model.density_weight_initialized()) model.initialize_density_weight(pos_);

  if (iteration == 0 && params_.gp_noise_ratio > 0.0) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
      if (i >= db_.num_movable_nodes && i < n - db_.num_filler_nodes) continue;  // no noise on fixed cells
      const auto k = static_cast<size_t>(i);
      pos_[k] += static_cast<Real>((uni(rng_) - 0.5) * static_cast<double>(db_.node_size_x[k]) * params_.gp_noise_ratio);
      pos_[static_cast<size_t>(n) + k] += static_cast<Real>((uni(rng_) - 0.5) * static_cast<double>(db_.node_size_y[k]) * params_.gp_noise_ratio);
    }
    initialize_learning_rate();
  }

  std::optional<Metrics> best_metric;
  std::vector<Real> best_pos;

  int lsub_iteration = stage.lsub_iteration;
  int llambda_flat_iteration = 0;
  int num_area_adjust = 0;
  bool adjust_resource_flag = params_.adjust_resource_area_flag != 0;
  bool adjust_route_flag = params_.adjust_route_area_flag != 0;
  bool adjust_pin_flag = params_.adjust_pin_area_flag != 0;
  std::vector<Real> resource_areas;  // kept between rounds (the original re-uses the stale value)
  bool have_resource_areas = false;

  std::optional<RudyMap> rudy;
  std::optional<PinUtilizationMap> pin_util;
  std::optional<AdjustNodeArea> adjust_op;
  if (params_.routability_opt_flag) {
    rudy.emplace(data_, params_);
    pin_util.emplace(data_, params_);
    adjust_op.emplace(data_, params_);
  }
  std::vector<Real> pin_pos_buf(2 * static_cast<size_t>(db_.num_pins()));

  // ---- stopping criteria -------------------------------------------------------------------------
  auto lgamma_stop = [&](int lgamma_step) {
    if (all_metrics.size() <= 1) return false;
    const Metrics& cur = all_metrics[all_metrics.size() - 1].back().back();
    const Metrics& prev = all_metrics[all_metrics.size() - 2].back().back();
    const bool blocks_ready = db_.num_movable_nodes_fence_region[2] == 0 || block_legal_iter_ >= 5;
    if (lgamma_step > 100 &&
        ((all_below_target(cur.overflow, db_) && cur.hpwl > prev.hpwl) || max_of(cur.max_density) < Real(1.0)) &&
        blocks_ready) {
      info("Lgamma stopping criteria: {} > 100 and (( OVFL: {:g}; {:g}; {:g}; {:g} and HPWL {:g} > {:g} ) or {:g} < 1.0) and DSP/RAM block legal iter {} >= 5",
           lgamma_step, cur.overflow[0], cur.overflow[1], cur.overflow[2], cur.overflow[3], cur.hpwl, prev.hpwl,
           max_of(cur.max_density), block_legal_iter_);
      return true;
    }
    return false;
  };
  auto llambda_stop = [&](const std::vector<std::vector<Metrics>>& metrics) {
    if (metrics.size() <= 1) return false;
    const Metrics& cur = metrics[metrics.size() - 1].back();
    const Metrics& prev = metrics[metrics.size() - 2].back();
    const bool blocks_ready = db_.num_movable_nodes_fence_region[2] == 0 || block_legal_iter_ >= 5;
    if ((all_below_target(cur.overflow, db_) && cur.hpwl > prev.hpwl && blocks_ready) || cur.max_density[3] < Real(1.0)) {
      info("Llambda stopping criteria: OVFL/HPWL condition met (HPWL {:g} > {:g})", cur.hpwl, prev.hpwl);
      return true;
    }
    return false;
  };
  auto lsub_stop = [&](const std::vector<Metrics>& metrics) {
    const int window = std::max(std::min(lsub_iteration / 2, 3), 1);
    if (static_cast<int>(metrics.size()) < window * 2) return false;
    double cur = 0, prev = 0;
    for (int i = 0; i < window; ++i) {
      cur += static_cast<double>(metrics[metrics.size() - 1 - static_cast<size_t>(i)].objective);
      prev += static_cast<double>(metrics[metrics.size() - 1 - static_cast<size_t>(window) - static_cast<size_t>(i)].objective);
    }
    cur /= window;
    prev /= window;
    return cur >= prev * 0.999;
  };

  // ---- one descent step -----------------------------------------------------------------------------
  auto one_descent_step = [&](std::vector<Metrics>& lsub_metrics) {
    ops::move_boundary(data_, pos_.data());
    Stopwatch timer;
    Metrics cur = model.evaluate(pos_);
    cur.iteration = iteration;
    info("{}, time {:.3f}ms", cur.to_string(), timer.seconds() * 1000);

    if (model.update_mask) {
      const std::vector<Real> backup = pos_;
      optimizer.step(pos_);
      for (size_t r = 0; r < kNumMovableRegions; ++r) {
        if (!(*model.update_mask)[r] && db_.num_movable_nodes_fence_region[r] > 0) {
          // Frozen region: undo the movement of its cells and fillers.
          for (int id : regions_[r]->node_ids()) {
            pos_[static_cast<size_t>(id)] = backup[static_cast<size_t>(id)];
            pos_[static_cast<size_t>(n) + static_cast<size_t>(id)] = backup[static_cast<size_t>(n) + static_cast<size_t>(id)];
          }
        }
      }
    } else {
      optimizer.step(pos_);
    }
    cur.objective = optimizer.previous_objective();
    if (!best_metric || (best_metric->overflow[0] > cur.overflow[0] && best_metric->overflow[1] > cur.overflow[1])) {
      best_metric = cur;
      best_pos = pos_;
    }
    lsub_metrics.push_back(cur);
  };

  // ---- main loops ---------------------------------------------------------------------------------------
  Stopwatch optimization_timer;
  for (int lgamma_step = 0; lgamma_step < stage.iteration; ++lgamma_step) {
    all_metrics.emplace_back();
    for (int llambda_step = 0; llambda_step < stage.llambda_density_weight_iteration; ++llambda_step) {
      all_metrics.back().emplace_back();
      auto& lsub_metrics = all_metrics.back().back();
      for (int lsub_step = 0; lsub_step < lsub_iteration; ++lsub_step) {
        one_descent_step(lsub_metrics);
        ++iteration;
        if (model.lock_mask && (*model.lock_mask)[2] && (*model.lock_mask)[3]) ++block_legal_iter_;
        if (lsub_stop(lsub_metrics)) break;
      }
      ++llambda_flat_iteration;
      auto& llambda_metrics = all_metrics.back();
      if (llambda_flat_iteration > 1) model.update_density_weight(llambda_metrics.back().back());
      if (llambda_stop(llambda_metrics)) break;

      const Metrics& last = llambda_metrics.back().back();

      // Routability: inflate cells according to congestion / pin / resource maps.
      if (params_.routability_opt_flag && num_area_adjust < params_.max_num_area_adjust &&
          std::max(last.overflow[0], last.overflow[1]) < static_cast<Real>(params_.node_area_adjust_overflow)) {
        std::optional<std::vector<Real>> route_map, pin_map;
        if (adjust_route_flag) {
          ops::pin_pos(data_, pos_.data(), pin_pos_buf.data());
          route_map = rudy->compute(pin_pos_buf.data());
        }
        if (adjust_pin_flag) pin_map = pin_util->compute(pos_.data());
        if (adjust_resource_flag) {
          resource_areas = lut_compatibility_areas(data_, pos_.data());
          const std::vector<Real> ff = ff_compatibility_areas(data_, pos_.data());
          for (size_t i = 0; i < resource_areas.size(); ++i) resource_areas[i] += ff[i];
          have_resource_areas = true;
        }
        const AreaAdjustFlags flags = adjust_op->run(pos_.data(), have_resource_areas ? &resource_areas : nullptr,
                                                     route_map ? &*route_map : nullptr, pin_map ? &*pin_map : nullptr);
        adjust_resource_flag = flags.resource;
        adjust_route_flag = flags.route;
        adjust_pin_flag = flags.pin;
        info("routability optimization round {}: adjust area flags = ({}, {}, {}, {})", num_area_adjust,
             static_cast<int>(flags.adjust_area), static_cast<int>(flags.resource), static_cast<int>(flags.route),
             static_cast<int>(flags.pin));
        if (flags.adjust_area) {
          ++num_area_adjust;
          data_.refresh_areas();
          for (auto& region : regions_) region->reset();
          pin_util->reset();
          Metrics m = model.evaluate(pos_);
          model.update_gamma(m.overflow);
          model.reset_density_weight(pos_, Real(0.1));
          optimizer.reset();
          initialize_learning_rate();
          lsub_iteration = stage.routability_lsub_iteration;
          break;
        }
      }

      // DSP / RAM legalization once every resource is spread well enough.
      if (std::max(db_.num_movable_nodes_fence_region[2], db_.num_movable_nodes_fence_region[3]) > 0 &&
          all_below_target(last.overflow, db_)) {
        if (model.lock_mask && (*model.lock_mask)[2] && (*model.lock_mask)[3]) break;
        const MoveStats dsp = legalize_dsp_ram(data_, pos_, kDSP, precond_wl_);
        info("Legalized DSPs with maxMov = {:g} and avgMov = {:g}", dsp.max_move, dsp.avg_move);
        const MoveStats ram = legalize_dsp_ram(data_, pos_, kRAM, precond_wl_);
        info("Legalized RAMs with maxMov = {:g} and avgMov = {:g}", ram.max_move, ram.avg_move);

        if (!model.lock_mask) model.lock_mask = std::array<bool, kNumMovableRegions>{};
        (*model.lock_mask)[2] = (*model.lock_mask)[3] = true;
        std::array<bool, kNumMovableRegions> upd{};
        for (size_t r = 0; r < kNumMovableRegions; ++r) upd[r] = !(*model.lock_mask)[r];
        model.update_mask = upd;
        for (auto& region : regions_) region->lock();

        Metrics m = model.evaluate(pos_);
        model.update_gamma(m.overflow);
        model.reset_density_weight(pos_, Real(1.0));
        optimizer.reset();
        initialize_learning_rate();
        lsub_iteration = stage.routability_lsub_iteration;
        break;
      }
    }

    model.update_gamma(all_metrics.back().back().back().overflow);
    if (lgamma_stop(lgamma_step)) break;
  }
  info("optimization took {:.3f} seconds", optimization_timer.seconds());

  // Roll back if the last step diverged compared to the best recorded position.
  const Metrics& last_metric = all_metrics.back().back().back();
  Real target_max = 0;
  for (double t : db_.target_overflow) target_max = std::max(target_max, static_cast<Real>(t));
  if (best_metric && max_of(last_metric.overflow) > std::max(target_max, max_of(best_metric->overflow)) &&
      last_metric.hpwl > best_metric->hpwl) {
    pos_ = best_pos;
    error("possible DIVERGENCE detected, roll back to the best position recorded");
    info("{}", best_metric->to_string());
  }
  info("global placement stage took {:.3f} seconds", stage_timer.seconds());
}

void NonLinearPlacer::run_legalization(PlaceObjective& model, int& iteration, std::vector<Metrics>& log) {
  Stopwatch timer;
  LutFfLegalizer legalizer(data_, params_);
  legalizer.run(pos_, model.precond_wl(), node_z_);
  info("legalization takes {:.3f} seconds", timer.seconds());
  Metrics m;
  m.iteration = iteration++;
  std::vector<Real> pin_pos_buf(2 * static_cast<size_t>(db_.num_pins()));
  ops::pin_pos(data_, pos_.data(), pin_pos_buf.data());
  m.hpwl = ops::hpwl(data_, pin_pos_buf.data());
  info("iter: {:4d}, HPWL {:.6E}", m.iteration, static_cast<double>(m.hpwl));
  log.push_back(m);
}

void NonLinearPlacer::load_solution(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw PlacerError("cannot open solution file " + file);
  std::string line;
  const int n = db_.num_nodes();
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string name;
    double x, y;
    int z;
    if (!(ss >> name >> x >> y >> z)) continue;
    auto it = db_.node_name2id.find(name);
    if (it == db_.node_name2id.end()) continue;
    const auto id = static_cast<size_t>(it->second);
    pos_[id] = static_cast<Real>(x);
    pos_[static_cast<size_t>(n) + id] = static_cast<Real>(y);
    if (it->second < db_.num_movable_nodes) node_z_[id] = z;
  }
  info("read solution from {}", file);
}

void NonLinearPlacer::apply_solution(const std::vector<int>& node_z) {
  const int n = db_.num_nodes();
  for (int i = 0; i < db_.num_movable_nodes; ++i) {
    const auto k = static_cast<size_t>(i);
    db_.node_x[k] = pos_[k];
    db_.node_y[k] = pos_[static_cast<size_t>(n) + k];
    db_.node_z[k] = node_z[k];
  }
}

}  // namespace dpfpga
