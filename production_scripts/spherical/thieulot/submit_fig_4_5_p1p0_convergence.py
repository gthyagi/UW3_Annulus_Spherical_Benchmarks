#!/usr/bin/env python3
"""Submit spherical Thieulot Fig. 4-5 P1/P0 convergence jobs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submit_utils import repo_root_from, slug, submit_job  # noqa: E402


MS = [-1, 3]
CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]


def main() -> None:
    repo_root = repo_root_from(__file__)
    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    bench_script = repo_root / "benchmarks" / "spherical" / "ex_stokes_thieulot.py"

    for m in MS:
        for cellsize in CELL_SIZES:
            args = [
                "-uw_run_on_gadi",
                "True",
                "-uw_vdegree",
                "1",
                "-uw_pdegree",
                "0",
                "-uw_pcont",
                "False",
                "-uw_bc_type",
                "essential",
                "-uw_stokes_tol",
                "1e-9",
                "-uw_m",
                str(m),
                "-uw_cellsize",
                cellsize,
            ]
            submit_job(
                common_pbs=common_pbs,
                script=bench_script,
                job_name=f"th_s_p1p0_m{slug(m)}_cs{slug(cellsize)}",
                args=args,
                walltime="12:00:00",
                ncpus=1,
                mem="8gb",
            )


if __name__ == "__main__":
    main()
