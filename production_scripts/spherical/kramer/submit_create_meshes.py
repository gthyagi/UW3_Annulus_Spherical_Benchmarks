#!/usr/bin/env python3
"""Submit spherical Kramer mesh jobs using the common PBS script."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submit_utils import repo_root_from, slug, submit_job  # noqa: E402

CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]
CASES = [
    {"name": "smooth", "radius_inner": "1.22", "radius_internal": "None", "radius_outer": "2.22"},
    {"name": "delta", "radius_inner": "1.22", "radius_internal": "2.0", "radius_outer": "2.22"},
]


def main() -> None:
    repo_root = repo_root_from(__file__)

    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    mesh_script = repo_root / "benchmarks" / "spherical" / "create_spherical_mesh.py"

    for cellsize in CELL_SIZES:
        for case in CASES:
            submit_job(
                common_pbs=common_pbs,
                script=mesh_script,
                job_name=f"kr_sph_mesh_{case['name']}_{slug(cellsize)}",
                args=[
                    "-uw_benchmark",
                    "kramer",
                    "-uw_radius_inner",
                    case["radius_inner"],
                    "-uw_radius_internal",
                    case["radius_internal"],
                    "-uw_radius_outer",
                    case["radius_outer"],
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
