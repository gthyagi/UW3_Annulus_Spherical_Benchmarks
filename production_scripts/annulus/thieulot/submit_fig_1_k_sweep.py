#!/usr/bin/env python3
"""Submit annulus Thieulot Fig. 1 fixed-cellsize jobs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submit_utils import repo_root_from, slug, submit_job  # noqa: E402


KS = [0, 2]
CELLSIZE = "1/64"


def main() -> None:
    repo_root = repo_root_from(__file__)
    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    bench_script = repo_root / "benchmarks" / "annulus" / "ex_stokes_thieulot.py"

    for k in KS:
        args = [
            "-run_on_gadi",
            "True",
            "-uw_vdegree",
            "2",
            "-uw_pdegree",
            "1",
            "-uw_pcont",
            "True",
            "-uw_bc_type",
            "essential",
            "-uw_stokes_tol",
            "1e-9",
            "-uw_k",
            str(k),
            "-uw_cellsize",
            CELLSIZE,
        ]
        submit_job(
            common_pbs=common_pbs,
            script=bench_script,
            job_name=f"th_a_f1_k{k}_cs{slug(CELLSIZE)}",
            args=args,
            walltime="08:00:00",
            ncpus=16,
            mem="64gb",
        )


if __name__ == "__main__":
    main()
