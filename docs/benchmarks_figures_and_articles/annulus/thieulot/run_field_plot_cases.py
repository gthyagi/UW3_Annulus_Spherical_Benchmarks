#!/usr/bin/env python3
"""Run annulus Thieulot field plots for selected benchmark output folders."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

PYTHON = Path(
    "/Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python"
)
REPO_ROOT = Path("/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks")
PLOT_SCRIPT = REPO_ROOT / "benchmarks/annulus/thieulot_field_plots.py"
OUTPUT_ROOT = Path("/Volumes/seagate4_1/output/annulus/thieulot/latest")

DIRNAMES = [
    "model_inv_lc_64_k_0_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_4_bc_natural_vel_penalty_2.5e+08",
    "model_inv_lc_64_k_1_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_4_bc_essential",
    "model_inv_lc_64_k_2_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_4_bc_essential",
    "model_inv_lc_64_k_4_vdeg_2_pdeg_1_pcont_true_stokes_tol_1e-09_ncpus_4_bc_essential",
]


def patched_script_text(dirname: str) -> str:
    """Return thieulot_field_plots.py with only the dirname assignment changed."""

    script_text = PLOT_SCRIPT.read_text()
    patched_text, count = re.subn(
        r'^dirname = ".*"$',
        f'dirname = "{dirname}"',
        script_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Could not patch dirname assignment in {PLOT_SCRIPT}")
    return patched_text


def main() -> None:
    for dirname in DIRNAMES:
        output_dir = OUTPUT_ROOT / dirname
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Missing output directory: {output_dir}")

        with tempfile.TemporaryDirectory(prefix="annulus_thieulot_plot_") as tmpdir:
            tmp_script = Path(tmpdir) / "thieulot_field_plots.py"
            tmp_script.write_text(patched_script_text(dirname))

            cmd = [str(PYTHON), str(tmp_script)]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
