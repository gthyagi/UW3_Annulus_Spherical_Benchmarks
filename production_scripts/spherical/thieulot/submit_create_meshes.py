#!/usr/bin/env python3
"""Submit spherical Thieulot mesh-generation jobs via the common Gadi PBS script."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submit_utils import repo_root_from, slug, submit_job  # noqa: E402

CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]

def main() -> None:
    repo_root = repo_root_from(__file__)

    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    mesh_script = repo_root / "benchmarks" / "spherical" / "create_spherical_mesh.py"

    for cellsize in CELL_SIZES:
        submit_job(
            common_pbs=common_pbs,
            script=mesh_script,
            job_name=f"th_sph_mesh_{slug(cellsize)}",
            args=[
                "-uw_benchmark",
                "thieulot",
                "-uw_radius_inner",
                "0.5",
                "-uw_radius_internal",
                "None",
                "-uw_radius_outer",
                "1.0",
                "-uw_cellsize",
                cellsize,
            ],
            walltime="12:00:00",
            ncpus=1,
            mem="256GB",
            queue="hugemembw",
        )


if __name__ == "__main__":
    main()
