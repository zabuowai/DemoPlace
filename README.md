# dpfpga — a C++20 port of DREAMPlaceFPGA's global placer

`dpfpga` is a standalone, CPU-only, dependency-light re-implementation of the **global
placement** engine from [DREAMPlaceFPGA](https://github.com/rachelselinar/DREAMPlaceFPGA)
— an open-source, GPU-accelerated analytical placer for heterogeneous FPGAs based on the
[ePlace](https://ieeexplore.ieee.org/document/6800810)/[elfPlace](https://ieeexplore.ieee.org/document/8942075)
electrostatic placement algorithm. Where the original is a PyTorch application with custom
C++/CUDA extension ops, `dpfpga` is plain C++20 + OpenMP: no Python, no PyTorch, no CUDA.

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a full explanation of the theory,
the algorithms, and how data flows through the placer, with diagrams.

## Status

| Stage | Status |
|---|---|
| Bookshelf (ISPD'2016) parsing | ✅ done |
| Global placement (wirelength + electrostatic density, Nesterov) | ✅ done |
| Routability-driven area inflation | ✅ done |
| DSP/RAM legalization (min-cost flow) | ✅ done |
| LUT/FF packing + legalization | ❌ stub — throws if invoked (`src/legalize_lut_ff_stub.cpp`) |
| Detailed placement | ❌ not ported |
| Timing-driven placement | ❌ not ported (parsed and warned about, ignored) |

Run with `"legalize_flag": 0` to stop after global placement and write a `.gp.pl`
solution; hand that off to an external legalizer/detailed-placer (the original project
bundles one, `elfPlace_LG_DP`) for a fully legal placement.

**Known limitations** (see [`docs/ARCHITECTURE.md#known-limitations`](docs/ARCHITECTURE.md#known-limitations)
for detail): results are not bit-reproducible run-to-run because the OpenMP reductions in
the hot loops don't fix a summation order (`deterministic_flag` is parsed but not yet
wired up to a deterministic accumulation path), and the RNG draws don't correspond
seed-for-seed with the original's PyTorch/NumPy RNGs (different algorithms), so HPWL
curves diverge from the original after enough iterations even at nominally "the same"
seed. Early-iteration behavior and the underlying model match the original closely —
see the architecture doc for the validation evidence.

## Requirements

- A C++20 compiler with OpenMP support. Developed and tested with **g++ 15.2** on Ubuntu
  25.10/26.04; anything g++ 11+ / clang 14+ with `-fopenmp` should work.
- No other dependencies — no Boost, no CMake, no Python. (If you also want to compare
  against the *original* DREAMPlaceFPGA, that's a much heavier PyTorch/CMake/Boost stack;
  see `docs/ARCHITECTURE.md` for notes, it is not needed to build or run this port.)

## Build

From the repository root:

```bash
g++ -std=c++20 -O3 -march=native -fopenmp -Wno-maybe-uninitialized \
    -Iinclude apps/dreamplacefpga.cpp src/*.cpp -o dpfpga
```

This produces a single `dpfpga` binary. There is no build system beyond this one
command — `-march=native` and `-O3` matter for performance (the density model does a lot
of DCT/FFT work per iteration), `-fopenmp` enables the multi-threaded hot loops.

## Run the example (FPGA-example1)

The repository bundles the ISPD'2016 `FPGA-example1` sample benchmark under
`benchmarks/sample_ispd2016_benchmarks/FPGA-example1/` (Bookshelf format: `.aux`, `.lib`,
`.scl`, `.nodes`, `.pl`, `.nets`). Run it with the provided config:

```bash
./dpfpga examples/FPGA-example1.json
```

You should see per-iteration log lines like:

```
[INFO   ] DREAMPlaceFPGA - read 3264 movable + 72 fixed instances, 3346 nets, 15575 pins, 168x480 sites in 0.05 s
[INFO   ] DREAMPlaceFPGA - Region: 0 #movable_nodes = ... #fillers =  535899, ...
[INFO   ] DREAMPlaceFPGA - use nesterov optimizer
[INFO   ] DREAMPlaceFPGA - iter:    0, HPWL 4.6E+03, Overflow [1.000E+00, 1.000E+00, ...], time 30-100ms
...
[INFO   ] DREAMPlaceFPGA - Lgamma stopping criteria: ...
[INFO   ] DREAMPlaceFPGA - write placement solution to results/design/design.gp.pl
```

This runs 1000-2000 Nesterov iterations of global placement (a few minutes on a modern
multi-core CPU) and writes the global-placement solution to
`results/design/design.gp.pl`. Since `legalize_flag` is `0` in the bundled config, it
stops there with a warning that legalization/detailed-placement need an external engine.

To also try the DSP/RAM-free variant benchmark (see
[`docs/ARCHITECTURE.md#a-dsp-ram-free-variant-benchmark`](docs/ARCHITECTURE.md#a-dsp-ram-free-variant-benchmark)
for why this exists):

```bash
./dpfpga examples/FPGA-example1-noDSPRAM.json
```

### Config file reference

`examples/FPGA-example1.json` sets the fields you're most likely to want to change:

| Field | Meaning |
|---|---|
| `aux_input` | path to the Bookshelf `.aux` file |
| `num_threads` | OpenMP thread count |
| `global_place_stages[].iteration` | max Nesterov iterations for that stage |
| `target_density` | target placement density (1.0 = fill exactly) |
| `routability_opt_flag` | enable congestion-driven cell-area inflation |
| `legalize_flag` | `0` = stop after GP; `1` = also try `LutFfLegalizer` (currently a stub — will throw) |
| `random_seed` | seeds this port's own RNG (init-position noise); does **not** need to match the original's `random_seed`, whose global-placement result is seed-independent — see the architecture doc |

The full field list lives in [`include/dpfpga/params.hpp`](include/dpfpga/params.hpp).

## Run the tests

Three independent, self-contained tests validate the pieces that are easiest to get
subtly wrong: the DCT/FFT-based Poisson solver, the min-cost bipartite assignment used
for DSP/RAM legalization, and the wirelength/density gradients (checked against finite
differences, so this one needs a real benchmark to run on).

```bash
# DCT/IDCT/IDXST vs. brute-force O(N^4) reference evaluation of their definitions
g++ -std=c++20 -O2 -Iinclude tests/test_dct.cpp src/dct.cpp -o test_dct
./test_dct

# min-cost bipartite assignment vs. brute-force over all injective maps (200 random trials)
g++ -std=c++20 -O2 -Iinclude tests/test_assignment.cpp src/legalize_dsp_ram.cpp -o test_assign
./test_assign

# finite-difference gradient checks against a real benchmark
g++ -std=c++20 -O2 -fopenmp -Iinclude tests/test_gradients.cpp src/*.cpp -o test_gradients
./test_gradients benchmarks/sample_ispd2016_benchmarks/FPGA-example1/design.aux
```

All three print `OK`/`FAIL` per check and exit non-zero on any failure.

## Repository layout

```
include/dpfpga/     public headers (Params, PlaceDB, PlaceObjective, RegionDensity, ...)
src/                implementation (bookshelf parser, density model, DCT, Nesterov, ...)
apps/               the dpfpga CLI driver (main())
tests/              the three tests described above, plus a scratch debug harness
examples/           ready-to-run JSON configs
benchmarks/         bundled ISPD'2016 Bookshelf sample designs
scripts/            benchmark-derivation utilities (e.g. stripping DSP/RAM instances)
docs/               detailed architecture, theory and algorithm documentation
```

## Provenance

This is a from-scratch C++ re-implementation guided by the public
[DREAMPlaceFPGA](https://github.com/rachelselinar/DREAMPlaceFPGA) source (Rajarathnam,
Alawieh, Jiang, Iyer, Pan — UT Austin) and its underlying
[ePlace](https://ieeexplore.ieee.org/document/6800810)/[elfPlace](https://ieeexplore.ieee.org/document/8942075)
papers. The bundled `FPGA-example1` benchmark is from the
[ISPD'2016 FPGA placement contest](http://www.ispd.cc/contests/16/FAQ.html). See
`docs/ARCHITECTURE.md` for the algorithm-level references.
