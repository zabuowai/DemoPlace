#!/usr/bin/env python3
"""Derive a DSP/RAM-free variant of an ISPD'2016 Bookshelf benchmark.

Removes every .nodes instance whose type contains "DSP" or "RAM" (matching the
substring check in src/bookshelf.cpp's PlaceDB::read), then removes those
instances' pins from .nets, dropping any net left with fewer than 2 pins.
.lib, .scl, .pl, .wts, .aux are copied unchanged (design.pl only lists fixed
IO instances; DSP/RAM are movable, so removing them never touches .pl).

Usage:
    python3 scripts/derive_no_dsp_ram_benchmark.py \\
        benchmarks/sample_ispd2016_benchmarks/FPGA-example1 \\
        benchmarks/sample_ispd2016_benchmarks/FPGA-example1-noDSPRAM
"""
import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    # ---- design.nodes: drop DSP/RAM instances --------------------------------
    removed = set()
    kept_lines = []
    with open(src / "design.nodes") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                kept_lines.append(line)
                continue
            parts = stripped.split()
            if len(parts) >= 2 and ("DSP" in parts[1] or "RAM" in parts[1]):
                removed.add(parts[0])
                continue
            kept_lines.append(line)
    with open(dst / "design.nodes", "w") as f:
        f.writelines(kept_lines)
    print(f"design.nodes: removed {len(removed)} DSP/RAM instances")

    # ---- design.nets: drop pins on removed instances, drop under-degree nets --
    nets_in = 0
    nets_out = 0
    pins_dropped = 0
    out_lines = []
    with open(src / "design.nets") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("net "):
            nets_in += 1
            header_parts = stripped.split()
            net_name = header_parts[1]
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != "endnet":
                pin_line = lines[i]
                pin_parts = pin_line.strip().split()
                if pin_parts and pin_parts[0] in removed:
                    pins_dropped += 1
                else:
                    body.append(pin_line)
                i += 1
            i += 1  # skip the endnet line itself
            if len(body) >= 2:
                nets_out += 1
                out_lines.append(f"net {net_name} {len(body)}\n")
                out_lines.extend(body)
                out_lines.append("endnet\n")
            # else: drop the whole net (fewer than 2 pins remain)
        else:
            out_lines.append(line)
            i += 1

    with open(dst / "design.nets", "w") as f:
        f.writelines(out_lines)
    print(f"design.nets: {nets_in} nets in, {nets_out} nets out, "
          f"{pins_dropped} pins dropped on removed instances, "
          f"{nets_in - nets_out} nets dropped entirely")

    # ---- everything else: copy unchanged --------------------------------------
    for name in ("design.lib", "design.scl", "design.pl", "design.wts", "design.aux"):
        shutil.copy(src / name, dst / name)
    print("copied design.lib, design.scl, design.pl, design.wts, design.aux unchanged")


if __name__ == "__main__":
    main()
