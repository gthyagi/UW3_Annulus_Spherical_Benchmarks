import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path("/Users/tgol0006/uw_folder/UW3_Annulus_Spherical_Benchmarks")
SCRIPT = REPO_ROOT / "benchmarks" / "sphere" / "ex_stokes_thieulot.py"
OUTPUT_ROOT = REPO_ROOT / "output" / "sphere" / "thieulot" / "latest"


def _read_pressure_points(case_dir: Path):
    mesh_h5 = case_dir / "output.mesh.00000.h5"
    pressure_h5 = case_dir / "output.mesh.Pressure.00000.h5"

    with h5py.File(mesh_h5, "r") as h5f:
        points = np.asarray(h5f["geometry/vertices"], dtype=np.float64)

    with h5py.File(pressure_h5, "r") as h5f:
        for candidate in (
            "vertex_fields/Pressure_Pressure",
            "vertex_fields/Pressure",
            "fields/Pressure",
            "Pressure",
        ):
            if candidate in h5f:
                pressure = np.asarray(h5f[candidate], dtype=np.float64).reshape(-1)
                break
        else:
            raise KeyError("Pressure dataset not found in checkpoint file.")

    return points, pressure


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is required")
def test_pressure_dirichlet_bc_preserves_zero_boundary_pressure():
    case_name = (
        "case_inv_lc_4_m_-1_vdeg_2_pdeg_1_pcont_true_"
        "vel_penalty_1e+08_stokes_tol_1e-06_ncpus_2_bc_essential_p_bc_true"
    )
    case_dir = OUTPUT_ROOT / case_name

    if case_dir.exists():
        shutil.rmtree(case_dir)

    command = [
        "mpirun",
        "-np",
        "2",
        sys.executable,
        str(SCRIPT),
        "-uw_cellsize",
        "1/4",
        "-uw_bc_type",
        "essential",
        "-uw_p_bc",
        "True",
        "-uw_stokes_tol",
        "1e-6",
    ]

    subprocess.run(
        command,
        cwd=SCRIPT.parent,
        check=True,
        timeout=300,
    )

    points, pressure = _read_pressure_points(case_dir)
    radius = np.linalg.norm(points, axis=1)

    for boundary_radius in (0.5, 1.0):
        mask = np.isclose(radius, boundary_radius, rtol=0.0, atol=1.0e-6)
        boundary_values = pressure[mask]

        assert boundary_values.size > 0
        assert np.max(np.abs(boundary_values)) < 1.0e-8
