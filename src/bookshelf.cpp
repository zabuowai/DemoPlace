// Bookshelf (ISPD'2016 FPGA flavour) reader and PlaceDB construction.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "dpfpga/placedb.hpp"

namespace dpfpga {

namespace {

namespace fs = std::filesystem;

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw PlacerError("cannot open " + p.string());
  in.seekg(0, std::ios::end);
  std::string s(static_cast<size_t>(in.tellg()), '\0');
  in.seekg(0);
  in.read(s.data(), static_cast<std::streamsize>(s.size()));
  return s;
}

bool iequals(std::string_view a, std::string_view b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return std::tolower(x) == std::tolower(y); });
}

/// Iterates over the non-empty, non-comment lines of a text buffer, splitting them into tokens.
class LineReader {
 public:
  explicit LineReader(std::string_view text) : text_(text) {}

  /// Fill `tokens` with the next meaningful line; returns false at end of input.
  bool next(std::vector<std::string_view>& tokens) {
    while (pos_ < text_.size()) {
      size_t eol = text_.find('\n', pos_);
      if (eol == std::string_view::npos) eol = text_.size();
      std::string_view line = text_.substr(pos_, eol - pos_);
      pos_ = eol + 1;
      tokens.clear();
      size_t i = 0;
      while (i < line.size()) {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= line.size() || line[i] == '#') break;
        size_t j = i;
        while (j < line.size() && !std::isspace(static_cast<unsigned char>(line[j]))) ++j;
        tokens.push_back(line.substr(i, j - i));
        i = j;
      }
      if (!tokens.empty()) return true;
    }
    return false;
  }

 private:
  std::string_view text_;
  size_t pos_ = 0;
};

int to_int(std::string_view s) { return static_cast<int>(std::strtol(std::string(s).c_str(), nullptr, 10)); }
double to_double(std::string_view s) { return std::strtod(std::string(s).c_str(), nullptr); }

/// Library cell: pin name -> pin type id (0 output, 1 input, 2 clock, 3 control).
using LibCell = std::unordered_map<std::string, int>;

struct FileList {
  std::string lib, scl, nodes, pl, nets;
};

FileList parse_aux(const fs::path& aux) {
  const std::string text = slurp(aux);
  LineReader reader(text);
  std::vector<std::string_view> tok;
  FileList files;
  while (reader.next(tok)) {
    for (auto t : tok) {
      auto dot = t.rfind('.');
      if (dot == std::string_view::npos) continue;
      std::string ext(t.substr(dot + 1));
      for (auto& c : ext) c = static_cast<char>(std::tolower(c));
      std::string name(t);
      if (ext == "lib") files.lib = name;
      else if (ext == "scl") files.scl = name;
      else if (ext == "nodes") files.nodes = name;
      else if (ext == "pl") files.pl = name;
      else if (ext == "nets") files.nets = name;
    }
  }
  return files;
}

std::unordered_map<std::string, LibCell> parse_lib(const fs::path& file) {
  const std::string text = slurp(file);
  LineReader reader(text);
  std::vector<std::string_view> tok;
  std::unordered_map<std::string, LibCell> cells;
  LibCell* cur = nullptr;
  while (reader.next(tok)) {
    if (iequals(tok[0], "CELL") && tok.size() >= 2) {
      cur = &cells[std::string(tok[1])];
    } else if (iequals(tok[0], "END")) {
      cur = nullptr;
    } else if (iequals(tok[0], "PIN") && cur && tok.size() >= 3) {
      int type = iequals(tok[2], "OUTPUT") ? 0 : 1;
      if (type == 1 && tok.size() >= 4) {
        if (iequals(tok[3], "CLOCK")) type = 2;
        else if (iequals(tok[3], "CTRL")) type = 3;
      }
      cur->emplace(std::string(tok[1]), type);  // first definition wins (matches std::map::insert)
    }
  }
  return cells;
}

}  // namespace

void PlaceDB::read(const Params& params) {
  const fs::path aux(params.aux_input);
  if (params.aux_input.empty()) throw PlacerError("aux_input is required (Bookshelf input)");
  const fs::path dir = aux.parent_path();
  const FileList files = parse_aux(aux);
  if (files.lib.empty() || files.scl.empty() || files.nodes.empty() || files.nets.empty()) {
    throw PlacerError("the .aux file must list .lib, .scl, .nodes and .nets files");
  }
  info("reading {}", params.aux_input);
  Stopwatch timer;

  // ---- .lib ---------------------------------------------------------------------
  const auto lib = parse_lib(dir / files.lib);

  // ---- .scl ---------------------------------------------------------------------
  {
    const std::string text = slurp(dir / files.scl);
    LineReader reader(text);
    std::vector<std::string_view> tok;
    bool in_sitemap = false;
    while (reader.next(tok)) {
      if (iequals(tok[0], "SITEMAP") && tok.size() >= 3) {
        num_sites_x = to_int(tok[1]);
        num_sites_y = to_int(tok[2]);
        site_type_map.assign(static_cast<size_t>(num_sites_x) * num_sites_y, kSiteEmpty);
        in_sitemap = true;
      } else if (iequals(tok[0], "END") && tok.size() >= 2 && iequals(tok[1], "SITEMAP")) {
        in_sitemap = false;
      } else if (in_sitemap && tok.size() >= 3) {
        const int x = to_int(tok[0]), y = to_int(tok[1]);
        int type = kSiteEmpty;
        if (iequals(tok[2], "SLICE")) type = kSiteSlice;
        else if (iequals(tok[2], "DSP")) type = kSiteDsp;
        else if (iequals(tok[2], "BRAM")) type = kSiteRam;
        else if (iequals(tok[2], "IO")) type = kSiteIo;
        if (x < 0 || x >= num_sites_x || y < 0 || y >= num_sites_y) {
          throw PlacerError(std::format("site ({}, {}) outside the {}x{} site map", x, y, num_sites_x, num_sites_y));
        }
        site_type_map[static_cast<size_t>(x) * num_sites_y + y] = type;
      }
    }
    if (num_sites_x <= 0) throw PlacerError("no SITEMAP found in " + files.scl);
    xl = yl = 0;
    xh = num_sites_x;
    yh = num_sites_y;
  }

  // ---- .nodes -------------------------------------------------------------------
  std::vector<std::string> fixed_names, fixed_types;
  std::vector<int> mov_lut_type, mov_cluster_lut_type, mov_region;
  std::vector<Real> mov_sx, mov_sy;
  std::array<int, 5> counts{};
  {
    const double sqrt0625 = std::sqrt(0.0625), sqrt0125 = std::sqrt(0.125);
    const std::string text = slurp(dir / files.nodes);
    LineReader reader(text);
    std::vector<std::string_view> tok;
    auto add_movable = [&](std::string_view name, std::string_view type, int region, double sx, double sy, int lt,
                           int clt) {
      node_name2id.emplace(std::string(name), static_cast<int>(node_names.size()));
      if (region == kFF) flop_indices.push_back(static_cast<int>(node_names.size()));
      node_names.emplace_back(name);
      node_types.emplace_back(type);
      mov_region.push_back(region);
      mov_sx.push_back(static_cast<Real>(sx));
      mov_sy.push_back(static_cast<Real>(sy));
      mov_lut_type.push_back(lt);
      mov_cluster_lut_type.push_back(clt);
      ++counts[static_cast<size_t>(region)];
    };
    while (reader.next(tok)) {
      if (tok.size() < 2) continue;
      const std::string_view name = tok[0], type = tok[1];
      if (iequals(type, "FDRE")) {
        add_movable(name, type, kFF, sqrt0625, sqrt0625, 0, 0);
      } else if (iequals(type, "LUT0") || iequals(type, "GND") || iequals(type, "VCC")) {
        add_movable(name, type, kLUT, sqrt0625, sqrt0625, 0, 0);
      } else if (iequals(type, "LUT1")) {
        add_movable(name, type, kLUT, sqrt0625, sqrt0625, 1, 0);
      } else if (iequals(type, "LUT2")) {
        add_movable(name, type, kLUT, sqrt0625, sqrt0625, 2, 1);
      } else if (iequals(type, "LUT3")) {
        add_movable(name, type, kLUT, sqrt0625, sqrt0625, 3, 2);
      } else if (iequals(type, "LUT4")) {
        add_movable(name, type, kLUT, sqrt0125, sqrt0125, 4, 3);
      } else if (iequals(type, "LUT5")) {
        add_movable(name, type, kLUT, sqrt0125, sqrt0125, 5, 4);
      } else if (iequals(type, "LUT6") || iequals(type, "LUT6_2")) {  // LUT6_2 is treated like LUT6
        add_movable(name, type, kLUT, sqrt0125, sqrt0125, 6, 5);
      } else if (type.find("DSP") != std::string_view::npos) {
        add_movable(name, type, kDSP, 1.0, 2.5, 0, 0);
      } else if (type.find("RAM") != std::string_view::npos) {
        add_movable(name, type, kRAM, 1.0, 5.0, 0, 0);
      } else if (type.find("BUF") != std::string_view::npos) {
        fixed_names.emplace_back(name);
        fixed_types.emplace_back(type);
      } else {
        warn("unknown component type in .nodes file: {}, {}", name, type);
      }
    }
  }
  num_movable_nodes = static_cast<int>(node_names.size());
  num_terminals = static_cast<int>(fixed_names.size());
  num_physical_nodes = num_movable_nodes + num_terminals;
  node_count = {counts[kLUT], counts[kFF], counts[kDSP], counts[kRAM], num_terminals};
  for (int i = 0; i < num_terminals; ++i) {
    node_name2id.emplace(fixed_names[static_cast<size_t>(i)], num_movable_nodes + i);
    node_names.push_back(fixed_names[static_cast<size_t>(i)]);
    node_types.push_back(fixed_types[static_cast<size_t>(i)]);
  }
  node_size_x = mov_sx;
  node_size_y = mov_sy;
  node_size_x.insert(node_size_x.end(), static_cast<size_t>(num_terminals), Real(1));
  node_size_y.insert(node_size_y.end(), static_cast<size_t>(num_terminals), Real(1));
  node2fence_region_map = mov_region;
  node2fence_region_map.insert(node2fence_region_map.end(), static_cast<size_t>(num_terminals), kIO);
  lut_type = mov_lut_type;
  lut_type.insert(lut_type.end(), static_cast<size_t>(num_terminals), 0);
  cluster_lut_type = mov_cluster_lut_type;
  cluster_lut_type.insert(cluster_lut_type.end(), static_cast<size_t>(num_terminals), 0);
  node_x.assign(static_cast<size_t>(num_physical_nodes), Real(0));
  node_y.assign(static_cast<size_t>(num_physical_nodes), Real(0));
  node_z.assign(static_cast<size_t>(num_physical_nodes), 0);

  // ---- .pl (fixed instances) ------------------------------------------------------
  auto load_pl = [&](const fs::path& file) {
    const std::string text = slurp(file);
    LineReader reader(text);
    std::vector<std::string_view> tok;
    while (reader.next(tok)) {
      if (tok.size() < 4 || iequals(tok[0], "UCLA")) continue;
      auto it = node_name2id.find(std::string(tok[0]));
      if (it == node_name2id.end()) continue;
      node_x[static_cast<size_t>(it->second)] = static_cast<Real>(to_double(tok[1]));
      node_y[static_cast<size_t>(it->second)] = static_cast<Real>(to_double(tok[2]));
      node_z[static_cast<size_t>(it->second)] = to_int(tok[3]);
    }
  };
  if (!files.pl.empty()) load_pl(dir / files.pl);
  if (!params.io_pl.empty()) {
    info("reading {}", params.io_pl);
    load_pl(params.io_pl);
  }

  // ---- .nets ----------------------------------------------------------------------
  node2pincount_map.assign(static_cast<size_t>(num_physical_nodes), 0);
  node2outpin_map.assign(static_cast<size_t>(num_physical_nodes), 0);
  flat_net2pin_start_map.push_back(0);
  net2tnet_start_map.push_back(0);
  {
    const std::string text = slurp(dir / files.nets);
    LineReader reader(text);
    std::vector<std::string_view> tok;
    std::vector<int> source_pins, sink_pins;
    int cur_net = -1;
    while (reader.next(tok)) {
      if (iequals(tok[0], "net") && tok.size() >= 2) {
        cur_net = static_cast<int>(net_names.size());
        net_name2id.emplace(std::string(tok[1]), cur_net);
        net_names.emplace_back(tok[1]);
        source_pins.clear();
        sink_pins.clear();
        continue;
      }
      if (iequals(tok[0], "endnet")) {
        const int net_pin_count = static_cast<int>(flat_net2pin_map.size()) - flat_net2pin_start_map.back();
        net2pincount_map.push_back(net_pin_count);
        flat_net2pin_start_map.push_back(static_cast<int>(flat_net2pin_map.size()));
        for (int src : source_pins) {
          for (int snk : sink_pins) {
            if (net_pin_count > 3000) break;
            flat_tnet2pin_map.push_back(src);
            flat_tnet2pin_map.push_back(snk);
            snkpin2tnet_map[static_cast<size_t>(snk)] = static_cast<int>(tnet2net_map.size());
            tnet2net_map.push_back(cur_net);
          }
        }
        net2tnet_start_map.push_back(static_cast<int>(tnet2net_map.size()));
        cur_net = -1;
        continue;
      }
      if (cur_net < 0 || tok.size() < 2) continue;

      // a pin line: <instance> <pin name>
      const std::string inst(tok[0]);
      const int pin_id = static_cast<int>(pin_names.size());
      auto it = node_name2id.find(inst);
      if (it == node_name2id.end()) {
        throw PlacerError(std::format("net {} connects to instance {} which is not in the .nodes file",
                                      net_names[static_cast<size_t>(cur_net)], inst));
      }
      const int node_id = it->second;
      const bool fixed = node_id >= num_movable_nodes;
      pin_names.emplace_back(tok[1]);
      pin2net_map.push_back(cur_net);
      snkpin2tnet_map.push_back(-1);
      pin2node_map.push_back(node_id);
      pin2nodetype_map.push_back(fixed ? kIO : node2fence_region_map[static_cast<size_t>(node_id)]);
      pin_offset_x.push_back(fixed ? Real(0.5) : Real(0.5) * node_size_x[static_cast<size_t>(node_id)]);
      pin_offset_y.push_back(fixed ? Real(0.5) : Real(0.5) * node_size_y[static_cast<size_t>(node_id)]);

      const std::string& type = node_types[static_cast<size_t>(node_id)];
      auto cell = lib.find(type);
      if (cell == lib.end()) throw PlacerError("instance type " + type + " is not defined in the .lib file");
      int pin_type = -1;
      if (auto p = cell->second.find(std::string(tok[1])); p != cell->second.end()) pin_type = p->second;
      if (pin_type == -1) {
        warn("net {} connects to instance {} pin {}, which is not a valid pin of {} in the .lib file",
             net_names[static_cast<size_t>(cur_net)], inst, tok[1], type);
      }
      const bool is_io = fixed;
      switch (pin_type) {
        case 0:
          if (!is_io) source_pins.push_back(pin_id);
          break;
        case 1:
          if (!is_io) sink_pins.push_back(pin_id);
          break;
        case 2:
          if (!is_io) sink_pins.push_back(pin_id);
          break;
        case 3:
          if (!is_io) sink_pins.push_back(pin_id);
          if (tok[1].find("CE") == std::string_view::npos) pin_type = 4;  // set/reset control
          break;
        default: break;
      }
      pin_type_ids.push_back(pin_type);
      ++node2pincount_map[static_cast<size_t>(node_id)];
      if (pin_type == 0) node2outpin_map[static_cast<size_t>(node_id)] = pin_id;
      flat_net2pin_map.push_back(pin_id);
    }
  }

  // ---- derived maps -----------------------------------------------------------------
  flat_node2pin_start_map.assign(static_cast<size_t>(num_physical_nodes) + 1, 0);
  for (int n : pin2node_map) ++flat_node2pin_start_map[static_cast<size_t>(n) + 1];
  std::partial_sum(flat_node2pin_start_map.begin(), flat_node2pin_start_map.end(), flat_node2pin_start_map.begin());
  flat_node2pin_map.assign(pin2node_map.size(), 0);
  {
    std::vector<int> cursor(flat_node2pin_start_map.begin(), flat_node2pin_start_map.end() - 1);
    for (int p = 0; p < num_pins(); ++p) {
      flat_node2pin_map[static_cast<size_t>(cursor[static_cast<size_t>(pin2node_map[static_cast<size_t>(p)])]++)] = p;
    }
  }
  net_weights.assign(net_names.size(), Real(1));
  tnet_weights.assign(tnet2net_map.size(), Real(0));
  tnet_criticality.assign(tnet2net_map.size(), Real(0));

  // Legalization pin offsets: only DSP/RAM pins keep an offset.
  lg_pin_offset_x = pin_offset_x;
  lg_pin_offset_y = pin_offset_y;
  for (size_t p = 0; p < pin_offset_x.size(); ++p) {
    if (pin2nodetype_map[p] < 2 || pin2nodetype_map[p] > 3) lg_pin_offset_x[p] = lg_pin_offset_y[p] = 0;
  }

  // Flop control sets: identical (clock, set/reset) and CE nets share an id.
  {
    std::unordered_map<int, std::unordered_map<int, int>> cksr;
    std::unordered_map<int, int> ce_map;
    int n_cksr = 0, n_ce = 0;
    for (int f : flop_indices) {
      int ck = -1, sr = -1, ce = -1;
      for (int j = flat_node2pin_start_map[static_cast<size_t>(f)]; j < flat_node2pin_start_map[static_cast<size_t>(f) + 1]; ++j) {
        const int pin = flat_node2pin_map[static_cast<size_t>(j)];
        switch (pin_type_ids[static_cast<size_t>(pin)]) {
          case 2: ck = pin2net_map[static_cast<size_t>(pin)]; break;
          case 3: ce = pin2net_map[static_cast<size_t>(pin)]; break;
          case 4: sr = pin2net_map[static_cast<size_t>(pin)]; break;
          default: break;
        }
      }
      auto& sr_map = cksr[ck];
      auto sr_it = sr_map.find(sr);
      const int cksr_id = sr_it == sr_map.end() ? (sr_map[sr] = n_cksr++) : sr_it->second;
      auto ce_it = ce_map.find(ce);
      const int ce_id = ce_it == ce_map.end() ? (ce_map[ce] = n_ce++) : ce_it->second;
      flat_ctrlsets.push_back(f);
      flat_ctrlsets.push_back(cksr_id);
      flat_ctrlsets.push_back(ce_id);
    }
    num_cksr = n_cksr;
    num_ce = n_ce;
  }
  flop2ctrlset_map.assign(static_cast<size_t>(num_physical_nodes), 0);
  for (size_t i = 0; i < flop_indices.size(); ++i) flop2ctrlset_map[static_cast<size_t>(flop_indices[i])] = static_cast<int>(i);

  // Site geometry.
  lg_site_xy.resize(site_type_map.size() * 2);
  std::vector<std::unordered_set<int>> cols(5);
  for (int i = 0; i < num_sites_x; ++i) {
    for (int j = 0; j < num_sites_y; ++j) {
      const size_t idx = static_cast<size_t>(i) * num_sites_y + j;
      const int t = site_type_map[idx];
      const Real off = (t == kSiteSlice) ? Real(0.5) : Real(0);
      lg_site_xy[idx * 2] = i + off;
      lg_site_xy[idx * 2 + 1] = j + off;
      switch (t) {
        case kSiteSlice:
          cols[kLUT].insert(i);
          cols[kFF].insert(i);
          break;
        case kSiteDsp:
          dsp_site_xy.push_back(static_cast<Real>(i));
          dsp_site_xy.push_back(static_cast<Real>(std::round(j / 2.5) * 2.5));
          cols[kDSP].insert(i);
          break;
        case kSiteRam:
          ram_site_xy.push_back(static_cast<Real>(i));
          ram_site_xy.push_back(static_cast<Real>(std::round(j / 5.0) * 5.0));
          cols[kRAM].insert(i);
          break;
        case kSiteIo: cols[kIO].insert(i); break;
        default: break;
      }
    }
  }
  flat_region_boxes_start.push_back(0);
  for (int r = 0; r < kNumRegions; ++r) {
    std::vector<int> sorted(cols[static_cast<size_t>(r)].begin(), cols[static_cast<size_t>(r)].end());
    std::sort(sorted.begin(), sorted.end());  // the original iterates an unordered_set; sorted is deterministic
    for (int c : sorted) {
      flat_region_boxes.insert(flat_region_boxes.end(),
                               {static_cast<Real>(c), Real(0), static_cast<Real>(c + 1), static_cast<Real>(num_sites_y)});
    }
    flat_region_boxes_start.push_back(static_cast<int>(flat_region_boxes.size() / 4));
  }

  row_height = height();
  site_width = width();
  num_routing_grids_x = width();
  num_routing_grids_y = height();
  routing_grid_xl = xl;
  routing_grid_yl = yl;
  routing_grid_xh = xh;
  routing_grid_yh = yh;

  // Spiral accessor used by the legalizer to enumerate neighbouring sites by increasing distance.
  {
    const int rad = std::max(num_sites_x, num_sites_y);
    spiral_accessor.reserve(static_cast<size_t>(2 * rad * (rad + 1) + 1) * 2);
    spiral_accessor.insert(spiral_accessor.end(), {0, 0});
    for (int r = 1; r <= rad; ++r) {
      for (int x = r, y = 0; y < r; --x, ++y) spiral_accessor.insert(spiral_accessor.end(), {x, y});
      for (int x = 0, y = r; y > 0; --x, --y) spiral_accessor.insert(spiral_accessor.end(), {x, y});
      for (int x = -r, y = 0; y > -r; ++x, --y) spiral_accessor.insert(spiral_accessor.end(), {x, y});
      for (int x = 0, y = -r; y < 0; ++x, ++y) spiral_accessor.insert(spiral_accessor.end(), {x, y});
    }
  }

  unit_horizontal_capacity = 0.95 * params.unit_horizontal_capacity;
  unit_vertical_capacity = 0.95 * params.unit_vertical_capacity;

  lut_mask.assign(static_cast<size_t>(num_physical_nodes), 0);
  flop_mask = dsp_mask = ram_mask = lut_mask;
  for (int i = 0; i < num_physical_nodes; ++i) {
    switch (node2fence_region_map[static_cast<size_t>(i)]) {
      case kLUT: lut_mask[static_cast<size_t>(i)] = 1; break;
      case kFF: flop_mask[static_cast<size_t>(i)] = 1; break;
      case kDSP: dsp_mask[static_cast<size_t>(i)] = 1; break;
      case kRAM: ram_mask[static_cast<size_t>(i)] = 1; break;
      default: break;
    }
  }

  info("read {} movable + {} fixed instances, {} nets, {} pins, {}x{} sites in {:.2f} s", num_movable_nodes,
       num_terminals, num_nets(), num_pins(), num_sites_x, num_sites_y, timer.seconds());
}

void PlaceDB::read_pl(const std::string& pl_file) {
  const std::string text = slurp(pl_file);
  LineReader reader(text);
  std::vector<std::string_view> tok;
  while (reader.next(tok)) {
    if (tok.size() < 4 || iequals(tok[0], "UCLA")) continue;
    auto it = node_name2id.find(std::string(tok[0]));
    if (it == node_name2id.end()) continue;
    node_x[static_cast<size_t>(it->second)] = static_cast<Real>(to_double(tok[1]));
    node_y[static_cast<size_t>(it->second)] = static_cast<Real>(to_double(tok[2]));
    node_z[static_cast<size_t>(it->second)] = to_int(tok[3]);
  }
}

void PlaceDB::initialize(const Params& params) {
  resource_size_x = {1, 1, 1, 1};
  resource_size_y = {1, 1, Real(2.5), Real(5.0)};
  x_wirelen_wt = Real(0.7);
  y_wirelen_wt = Real(1.2);
  inst_dem_stddev_trunc = Real(2.5);

  gp_inst_stddev = static_cast<Real>(std::sqrt(2.5e-4 * num_nodes()) / (2.0 * inst_dem_stddev_trunc));
  inst_dem_stddev_x = inst_dem_stddev_y = gp_inst_stddev;
  nbr_dist_end = static_cast<Real>(1.2 * gp_inst_stddev * inst_dem_stddev_trunc);

  // Filler / stretching parameters per resource (0 LUT, 1 FF, 2 DSP, 3 RAM).
  const double s = std::sqrt(0.125);
  filler_size_x = {s, s, 1.0, 1.0};
  filler_size_y = {s, s, 2.5, 5.0};
  target_overflow = {0.1, 0.1, 0.2, 0.2};
  overflow_inst_density_stretch_ratio = {std::sqrt(2.0), std::sqrt(2.0), 0.0, 0.0};

  // The number of density bins is fixed at 512 x 512 by the original implementation.
  num_bins_x = 512;
  num_bins_y = 512;
  bin_size_x = static_cast<double>(width()) / num_bins_x;
  bin_size_y = static_cast<double>(height()) / num_bins_y;
  (void)params;

  // Total areas (only used for reporting).
  total_movable_node_area = 0;
  for (int i = 0; i < num_movable_nodes; ++i) {
    const int r = node2fence_region_map[static_cast<size_t>(i)];
    if (r < 2) {
      total_movable_node_area += filler_size_x[0] * filler_size_y[0];
    } else {
      total_movable_node_area += static_cast<double>(node_size_x[static_cast<size_t>(i)]) * node_size_y[static_cast<size_t>(i)];
    }
  }
  total_fixed_node_area = num_terminals;
  total_space_area = static_cast<double>(width()) * height();

  // Fence regions.
  region_boxes.assign(kNumMovableRegions, {});
  for (int r = 0; r < kNumMovableRegions; ++r) {
    for (int b = flat_region_boxes_start[static_cast<size_t>(r)]; b < flat_region_boxes_start[static_cast<size_t>(r) + 1]; ++b) {
      const Real* p = &flat_region_boxes[static_cast<size_t>(b) * 4];
      region_boxes[static_cast<size_t>(r)].push_back({p[0], p[1], p[2], p[3]});
    }
  }

  // Filler counts: fill the gap between placeable area and the area of the movable nodes.
  num_filler_nodes = 0;
  total_filler_node_area = 0;
  std::vector<Real> filler_sx, filler_sy;
  for (int r = 0; r < kNumMovableRegions; ++r) {
    double movable_area = 0;
    int count = 0;
    for (int i = 0; i < num_movable_nodes; ++i) {
      if (node2fence_region_map[static_cast<size_t>(i)] == r) {
        movable_area += static_cast<double>(node_size_x[static_cast<size_t>(i)]) * node_size_y[static_cast<size_t>(i)];
        ++count;
      }
    }
    double placeable_area = 0;
    for (const auto& b : region_boxes[static_cast<size_t>(r)]) placeable_area += double(b[2] - b[0]) * double(b[3] - b[1]);
    int num_filler = 0;
    if (count > 0) {
      const double filler_area = std::max(placeable_area - movable_area, 0.0);
      num_filler = static_cast<int>(std::floor(filler_area / (filler_size_x[static_cast<size_t>(r)] * filler_size_y[static_cast<size_t>(r)])));
      info("Region:{:2d} #movable_nodes = {:8d} movable_node_area ={:10.1f}, placeable_area ={:10.1f}, filler_node_area ={:10.1f}, #fillers ={:8d}, filler sizes ={:.4g}x{:.4g}",
           r, count, movable_area, placeable_area, filler_area, num_filler, filler_size_x[static_cast<size_t>(r)], filler_size_y[static_cast<size_t>(r)]);
    }
    num_movable_nodes_fence_region[static_cast<size_t>(r)] = count;
    total_movable_node_area_fence_region[static_cast<size_t>(r)] = movable_area;
    num_filler_nodes_fence_region[static_cast<size_t>(r)] = num_filler;
    num_filler_nodes += num_filler;
    filler_sx.insert(filler_sx.end(), static_cast<size_t>(num_filler), static_cast<Real>(filler_size_x[static_cast<size_t>(r)]));
    filler_sy.insert(filler_sy.end(), static_cast<size_t>(num_filler), static_cast<Real>(filler_size_y[static_cast<size_t>(r)]));
    total_filler_node_area += num_filler * filler_size_x[static_cast<size_t>(r)] * filler_size_y[static_cast<size_t>(r)];
  }

  filler_start_map.assign(kNumMovableRegions + 1, 0);
  if (params.enable_fillers) {
    for (int r = 0; r < kNumMovableRegions; ++r) {
      filler_start_map[static_cast<size_t>(r) + 1] = filler_start_map[static_cast<size_t>(r)] + num_filler_nodes_fence_region[static_cast<size_t>(r)];
    }
    node_size_x.insert(node_size_x.end(), filler_sx.begin(), filler_sx.end());
    node_size_y.insert(node_size_y.end(), filler_sy.begin(), filler_sy.end());
  } else {
    total_filler_node_area = 0;
    num_filler_nodes = 0;
    num_filler_nodes_fence_region.fill(0);
  }
}

void PlaceDB::write_gp(const std::string& file) const {
  std::ofstream out(file);
  if (!out) throw PlacerError("cannot write " + file);
  for (int i = 0; i < num_physical_nodes; ++i) {
    const auto k = static_cast<size_t>(i);
    out << '\n'
        << std::format("{} {:.6E} {:.6E} {:g} {:.6E}", node_names[k], static_cast<double>(node_x[k]),
                       static_cast<double>(node_y[k]), static_cast<double>(node_z[k]),
                       static_cast<double>(node_size_x[k]) * static_cast<double>(node_size_y[k]));
  }
  info("write placement solution to {}", file);
}

void PlaceDB::write_final(const std::string& file) const {
  std::ofstream out(file);
  if (!out) throw PlacerError("cannot write " + file);
  for (int i = 0; i < num_physical_nodes; ++i) {
    const auto k = static_cast<size_t>(i);
    // The original formats coordinates with %d, i.e. truncates DSP/RAM half-site offsets.
    out << node_names[k] << ' ' << static_cast<long long>(node_x[k]) << ' ' << static_cast<long long>(node_y[k]) << ' '
        << node_z[k] << '\n';
  }
  info("writing to {}", file);
}

}  // namespace dpfpga
