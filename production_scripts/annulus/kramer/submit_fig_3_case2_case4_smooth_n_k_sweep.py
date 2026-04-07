#!/usr/bin/env python3
"""Submit annulus Kramer Fig. 3 smooth sweeps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submit_utils import repo_root_from, slug, submit_job  # noqa: E402


CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256"]
KS = [2, 8]
NS = [2, 8, 32]
CASES = [
    ("case2", ["-uw_freeslip_type", "nitsche"]),
    ("case4", []),
]


def main() -> None:
    repo_root = repo_root_from(__file__)
    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    bench_script = repo_root / "benchmarks" / "annulus" / "ex_stokes_kramer.py"

    for case_name, extra_args in CASES:
        for k in KS:
            for n in NS:
                for cellsize in CELL_SIZES:
                    args = [
                        "-run_on_gadi",
                        "True",
                        "-uw_case",
                        case_name,
                        "-uw_k",
                        str(k),
                        "-uw_vdegree",
                        "2",
                        "-uw_pdegree",
                        "1",
                        "-uw_pcont",
                        "True",
                        "-uw_stokes_tol",
                        "1e-9",
                        *extra_args,
                        "-uw_n",
                        str(n),
                        "-uw_cellsize",
                        cellsize,
                    ]
                    submit_job(
                        common_pbs=common_pbs,
                        script=bench_script,
                        job_name=f"kr_a_{case_name}_k{k}_n{n}_cs{slug(cellsize)}",
                        args=args,
                        walltime="24:00:00",
                        ncpus=16,
                        mem="64gb",
                    )


if __name__ == "__main__":
    main()
