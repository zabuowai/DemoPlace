// Command line driver: dreamplacefpga <config.json> [<config2.json> ...]
// Port of the __main__ / placeFPGA() flow in Placer.py.
#include <filesystem>
#include <fstream>

#include "dpfpga/global_placer.hpp"

using namespace dpfpga;
namespace fs = std::filesystem;

static void place_fpga(const Params& params) {
  Stopwatch total;
  PlaceDB db;
  db.read(params);
  db.initialize(params);

  if (params.write_io_placement_flag || params.write_tcl_flag || params.enable_if) {
    warn("write_io_placement_flag / write_tcl_flag / enable_if (Vivado Tcl and FPGA Interchange output) are not ported");
  }

  NonLinearPlacer placer(params, db);
  placer.run();
  info("Placement completed in {:.2f} seconds", total.seconds());

  const fs::path dir = fs::path(params.result_dir) / params.design_name();
  fs::create_directories(dir);
  if (params.global_place_flag && params.legalize_flag == 0) {
    // Only global placement was run: write the .gp.pl; legalization/detailed placement need an external engine.
    db.write_gp((dir / (params.design_name() + ".gp.pl")).string());
    warn("legalize_flag=0: run legalization and detailed placement with an external engine (e.g. elfPlace)");
  } else if (params.legalize_flag) {
    db.write_final((dir / (params.design_name() + ".final." + params.solution_file_suffix())).string());
    info("Detailed Placement not run");
  }
  info("Completed Placement in {:.3f} seconds", total.seconds());
}

int main(int argc, char** argv) {
  if (argc < 2) {
    error("Input parameters required in json format");
    return 1;
  }
  try {
    std::vector<Params> all;
    for (int i = 1; i < argc; ++i) {
      Params p;
      p.load(argv[i]);
      all.push_back(std::move(p));
    }
    info("Parameters[{}] loaded", all.size());
#ifdef _OPENMP
    omp_set_num_threads(all.front().num_threads);
#endif
    for (const auto& p : all) place_fpga(p);
  } catch (const std::exception& e) {
    error("{}", e.what());
    return 2;
  }
  return 0;
}
