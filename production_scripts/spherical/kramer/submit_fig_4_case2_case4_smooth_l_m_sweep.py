#!/usr/bin/env python3
"""Submit spherical Kramer Fig. 4 smooth sweeps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submit_utils import repo_root_from, slug, submit_job  # noqa: E402


CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]
LM_PAIRS = [(2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]
CASES = [
    ("case2", ["-uw_freeslip_type", "nitsche"]),
    ("case4", []),
]


def main() -> None:
    repo_root = repo_root_from(__file__)
    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    bench_script = repo_root / "benchmarks" / "spherical" / "ex_stokes_kramer.py"

    for case_name, extra_args in CASES:
        for l, m in LM_PAIRS:
            for cellsize in CELL_SIZES:
                args = [
                    "-uw_run_on_gadi",
                    "True",
                    "-uw_case",
                    case_name,
                    "-uw_l",
                    str(l),
                    "-uw_m",
                    str(m),
                    "-uw_vdegree",
                    "2",
                    "-uw_pdegree",
                    "1",
                    "-uw_pcont",
                    "True",
                    "-uw_stokes_tol",
                    "1e-8",
                    *extra_args,
                    "-uw_cellsize",
                    cellsize,
                ]
                submit_job(
                    common_pbs=common_pbs,
                    script=bench_script,
                    job_name=f"kr_s_{case_name}_l{l}_m{slug(m)}_cs{slug(cellsize)}",
                    args=args,
                    walltime="24:00:00",
                    ncpus=16,
                    mem="64gb",
                )


if __name__ == "__main__":
    main()
