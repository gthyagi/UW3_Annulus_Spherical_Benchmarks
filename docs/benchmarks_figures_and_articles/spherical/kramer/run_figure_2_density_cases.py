#!/usr/bin/env python3
"""Run Kramer field plots for Figure 2-style density cases."""

from __future__ import annotations

import subprocess
from pathlib import Path


PYTHON = Path(
    "/Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python"
)
SCRIPT = Path(
    "/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks/benchmarks/spherical/kramer_field_plots.py"
)
REPO_ROOT = Path("/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks")

SMOOTH_CASES = [
    "case2_inv_lc_32_l_2_m_1_k_3_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-08_ncpus_192_bc_natural_nitsche",
    "case2_inv_lc_32_l_4_m_2_k_5_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-08_ncpus_192_bc_natural_nitsche",
    "case2_inv_lc_32_l_8_m_4_k_9_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-08_ncpus_192_bc_natural_nitsche",
]

DELTA_CASES = [
    "case1_inv_lc_32_l_2_m_1_k_3_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-08_ncpus_192_bc_natural_nitsche",
    "case1_inv_lc_32_l_4_m_2_k_5_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-08_ncpus_192_bc_natural_nitsche",
    "case1_inv_lc_32_l_8_m_4_k_9_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-08_ncpus_192_bc_natural_nitsche",
]


for density_type, case_dirs in (("smooth rho", SMOOTH_CASES), ("delta rho", DELTA_CASES)):
    print(f"\n=== {density_type} ===")
    for case_dir in case_dirs:
        cmd = [str(PYTHON), str(SCRIPT), case_dir]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
