#include "dpfpga/legalize_dsp_ram.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace dpfpga {

namespace {

/// Successive-shortest-path min-cost flow specialised for unit-capacity bipartite assignment.
class MinCostFlow {
 public:
  explicit MinCostFlow(int nodes) : graph_(static_cast<size_t>(nodes)) {}

  int add_edge(int from, int to, int cap, double cost) {
    graph_[static_cast<size_t>(from)].push_back({to, cap, cost, static_cast<int>(graph_[static_cast<size_t>(to)].size())});
    graph_[static_cast<size_t>(to)].push_back({from, 0, -cost, static_cast<int>(graph_[static_cast<size_t>(from)].size()) - 1});
    return static_cast<int>(graph_[static_cast<size_t>(from)].size()) - 1;
  }

  /// Push up to `max_flow` units from s to t; returns the flow actually pushed.
  int run(int s, int t, int max_flow) {
    const size_t n = graph_.size();
    std::vector<double> potential(n, 0.0), dist(n);
    std::vector<int> prev_node(n), prev_edge(n);
    int flow = 0;
    using Item = std::pair<double, int>;
    while (flow < max_flow) {
      std::fill(dist.begin(), dist.end(), std::numeric_limits<double>::infinity());
      std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;
      dist[static_cast<size_t>(s)] = 0;
      pq.emplace(0.0, s);
      while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();
        if (d > dist[static_cast<size_t>(v)]) continue;
        for (size_t e = 0; e < graph_[static_cast<size_t>(v)].size(); ++e) {
          const Edge& ed = graph_[static_cast<size_t>(v)][e];
          if (ed.cap <= 0) continue;
          const double nd = d + ed.cost + potential[static_cast<size_t>(v)] - potential[static_cast<size_t>(ed.to)];
          if (nd < dist[static_cast<size_t>(ed.to)] - 1e-12) {
            dist[static_cast<size_t>(ed.to)] = nd;
            prev_node[static_cast<size_t>(ed.to)] = v;
            prev_edge[static_cast<size_t>(ed.to)] = static_cast<int>(e);
            pq.emplace(nd, ed.to);
          }
        }
      }
      if (!std::isfinite(dist[static_cast<size_t>(t)])) break;  // no augmenting path left
      for (size_t v = 0; v < n; ++v) {
        if (std::isfinite(dist[v])) potential[v] += dist[v];
      }
      for (int v = t; v != s; v = prev_node[static_cast<size_t>(v)]) {
        Edge& e = graph_[static_cast<size_t>(prev_node[static_cast<size_t>(v)])][static_cast<size_t>(prev_edge[static_cast<size_t>(v)])];
        e.cap -= 1;
        graph_[static_cast<size_t>(v)][static_cast<size_t>(e.rev)].cap += 1;
      }
      ++flow;
    }
    return flow;
  }

  /// Residual capacity of an edge (0 means saturated).
  [[nodiscard]] int cap(int from, int idx) const { return graph_[static_cast<size_t>(from)][static_cast<size_t>(idx)].cap; }

 private:
  struct Edge {
    int to, cap;
    double cost;
    int rev;
  };
  std::vector<std::vector<Edge>> graph_;
};

}  // namespace

std::vector<int> assign_blocks_to_sites(const std::vector<double>& blocks_xy, const std::vector<double>& sites_xy,
                                        const std::vector<double>& block_weight, double dist_init, double dist_incr,
                                        double cost_scale) {
  const int nb = static_cast<int>(blocks_xy.size() / 2);
  const int ns = static_cast<int>(sites_xy.size() / 2);
  if (nb == 0) return {};
  if (ns < nb) throw PlacerError(std::format("cannot legalize {} instances onto only {} sites", nb, ns));

  // Arcs are added incrementally in growing distance windows [dist_min, dist_max).
  struct Arc {
    int block, site;
    double cost;
  };
  std::vector<Arc> arcs;
  double dist_min = 0, dist_max = dist_init;
  while (true) {
    for (int b = 0; b < nb; ++b) {
      for (int s = 0; s < ns; ++s) {
        const double d = std::abs(blocks_xy[static_cast<size_t>(b) * 2] - sites_xy[static_cast<size_t>(s) * 2]) +
                         std::abs(blocks_xy[static_cast<size_t>(b) * 2 + 1] - sites_xy[static_cast<size_t>(s) * 2 + 1]);
        if (d >= dist_min && d < dist_max) arcs.push_back({b, s, d * block_weight[static_cast<size_t>(b)] * cost_scale});
      }
    }
    // Node layout: 0 = source, 1 = sink, 2.. blocks, then sites.
    MinCostFlow mcf(2 + nb + ns);
    for (int b = 0; b < nb; ++b) mcf.add_edge(0, 2 + b, 1, 0.0);
    for (int s = 0; s < ns; ++s) mcf.add_edge(2 + nb + s, 1, 1, 0.0);
    std::vector<int> arc_edge(arcs.size());
    std::vector<int> per_block_count(static_cast<size_t>(nb), 1);  // edge 0 of a block node is its reverse arc from the source
    for (size_t a = 0; a < arcs.size(); ++a) {
      arc_edge[a] = mcf.add_edge(2 + arcs[a].block, 2 + nb + arcs[a].site, 1, arcs[a].cost);
    }
    if (mcf.run(0, 1, nb) != nb) {  // infeasible: enlarge the search window
      dist_min = dist_max;
      dist_max += dist_incr;
      continue;
    }
    std::vector<int> result(static_cast<size_t>(nb), -1);
    for (size_t a = 0; a < arcs.size(); ++a) {
      if (mcf.cap(2 + arcs[a].block, arc_edge[a]) == 0) result[static_cast<size_t>(arcs[a].block)] = arcs[a].site;
    }
    return result;
  }
}

MoveStats legalize_dsp_ram(const PlaceData& data, std::vector<Real>& pos, int region,
                           const std::vector<Real>& precond_wl) {
  constexpr double kMaxDistInit = 10.0, kMaxDistIncr = 10.0, kFlowCostScale = 100.0;
  const PlaceDB& db = data.db;
  const int n_all = db.num_nodes();
  const auto& mask = region == kDSP ? db.dsp_mask : db.ram_mask;
  const auto& sites = region == kDSP ? db.dsp_site_xy : db.ram_site_xy;

  std::vector<int> ids;
  for (int i = 0; i < db.num_physical_nodes; ++i) {
    if (mask[static_cast<size_t>(i)]) ids.push_back(i);
  }
  std::vector<double> blocks, site_xy(sites.begin(), sites.end()), weight;
  for (int id : ids) {
    blocks.push_back(static_cast<double>(pos[static_cast<size_t>(id)]));
    blocks.push_back(static_cast<double>(pos[static_cast<size_t>(n_all) + static_cast<size_t>(id)]));
    weight.push_back(static_cast<double>(precond_wl[static_cast<size_t>(id)]));
  }
  const std::vector<int> assignment = assign_blocks_to_sites(blocks, site_xy, weight, kMaxDistInit, kMaxDistIncr, kFlowCostScale);

  MoveStats stats;
  for (size_t b = 0; b < ids.size(); ++b) {
    const auto s = static_cast<size_t>(assignment[b]);
    const double mov = std::abs(blocks[b * 2] - site_xy[s * 2]) + std::abs(blocks[b * 2 + 1] - site_xy[s * 2 + 1]);
    stats.avg_move += mov;
    stats.max_move = std::max(stats.max_move, mov);
    pos[static_cast<size_t>(ids[b])] = static_cast<Real>(site_xy[s * 2]);
    pos[static_cast<size_t>(n_all) + static_cast<size_t>(ids[b])] = static_cast<Real>(site_xy[s * 2 + 1]);
  }
  if (!ids.empty()) stats.avg_move /= static_cast<double>(ids.size());
  return stats;
}

}  // namespace dpfpga
