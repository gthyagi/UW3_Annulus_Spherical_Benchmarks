#!/usr/bin/env python3
"""Run annulus Kramer field plots for Figure 1-style density cases."""

from __future__ import annotations

import subprocess
from pathlib import Path


PYTHON = Path(
    "/Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python"
)
SCRIPT = Path(
    "/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/annulus/kramer_field_plots.py"
)
REPO_ROOT = Path("/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks")
OUTPUT_ROOT = Path("/Volumes/seagate4_1/output/annulus/kramer/latest")

INV_LC = 64
CASES = ("case2", "case1")
REQUESTED_NK_CASES = [
    (2, 2),
    (8, 8),
    (32, 8),
]
NK_CASES = list(dict.fromkeys(REQUESTED_NK_CASES))


for case in CASES:
    print(f"\n=== {case} ===")
    for n, k in NK_CASES:
        if case in ("case2", "case4"):
            case_dir = (
                f"{case}_inv_lc_{INV_LC}_n_{n}_k_{k}_"
                "vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_8_bc_natural_nitsche"
            )
        else:
            case_dir = (
                f"{case}_inv_lc_{INV_LC}_n_{n}_"
                "vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_8_bc_natural_nitsche"
            )
        if not (OUTPUT_ROOT / case_dir).is_dir():
            print(f"Skipping missing output directory: {OUTPUT_ROOT / case_dir}", flush=True)
            continue

        cmd = [str(PYTHON), str(SCRIPT), "-dirname", case_dir]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
