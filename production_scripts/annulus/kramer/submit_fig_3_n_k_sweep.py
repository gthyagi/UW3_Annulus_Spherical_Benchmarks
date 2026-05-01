#!/usr/bin/env python3
"""Submit annulus Kramer Fig. 3 n/k sweeps for cases 1-4."""

import base64
import json
from pathlib import Path
import subprocess


CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256"]
NS = [2, 8, 32]
KS = [2, 8]
CASES = ["case1", "case2", "case3", "case4"]

RESOURCES_BY_CELLSIZE = {
    "1/8": {"ncpus": 2, "mem": "8gb", "walltime": "06:00:00"},
    "1/16": {"ncpus": 2, "mem": "8gb", "walltime": "06:00:00"},
    "1/32": {"ncpus": 2, "mem": "8gb", "walltime": "06:00:00"},
    "1/64": {"ncpus": 4, "mem": "16gb", "walltime": "06:00:00"},
    "1/128": {"ncpus": 8, "mem": "32gb", "walltime": "06:00:00"},
    "1/256": {"ncpus": 16, "mem": "64gb", "walltime": "06:00:00"},
}

# Case1: Free-slip boundaries and delta function density perturbation
# Case2: Free-slip boundaries and smooth density distribution
# Case3: Zero-slip boundaries and delta function density perturbation
# Case4: Zero-slip boundaries and smooth density distribution
FREE_SLIP_CASES = {"case1", "case2"}
SMOOTH_CASES = {"case2", "case4"}


def slug(value: int | str) -> str:
    return str(value).replace("/", "_")


def case_extra_args(case_name: str) -> list[str]:
    if case_name in FREE_SLIP_CASES:
        return ["-uw_freeslip_type", "nitsche"]
    return []


def k_values_for_case(case_name: str) -> list[int | None]:
    if case_name in SMOOTH_CASES:
        return KS
    return [None]


def resources_for_cellsize(cellsize: str) -> dict[str, int | str]:
    return dict(RESOURCES_BY_CELLSIZE[cellsize])


def submit_job(
    common_pbs: Path,
    script: Path,
    job_name: str,
    args: list[str],
    walltime: str,
    ncpus: int,
    mem: str,
) -> None:
    args_json_b64 = base64.b64encode(json.dumps(args).encode()).decode()

    cmd = [
        "qsub",
        "-P",
        "m18",
        "-N",
        job_name,
        "-q",
        "normal",
        "-l",
        f"walltime={walltime}",
        "-l",
        f"ncpus={ncpus}",
        "-l",
        f"mem={mem}",
        "-v",
        f"SCRIPT={script},ARGS_JSON_B64={args_json_b64}",
        str(common_pbs),
    ]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    bench_script = repo_root / "benchmarks" / "annulus" / "ex_stokes_kramer.py"

    for case_name in CASES:
        extra_args = case_extra_args(case_name)
        for k in k_values_for_case(case_name):
            for n in NS:
                for cellsize in CELL_SIZES:
                    resources = resources_for_cellsize(cellsize)
                    args = [
                        "-uw_run_on_gadi",
                        "True",
                        "-uw_case",
                        case_name,
                    ]
                    job_name = f"kr_a_{case_name}_n{n}_cs{slug(cellsize)}"

                    if k is not None:
                        args.extend(["-uw_k", str(k)])
                        job_name = f"kr_a_{case_name}_k{k}_n{n}_cs{slug(cellsize)}"

                    args.extend(
                        [
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
                    )

                    submit_job(
                        common_pbs=common_pbs,
                        script=bench_script,
                        job_name=job_name,
                        args=args,
                        walltime=resources["walltime"],
                        ncpus=resources["ncpus"],
                        mem=resources["mem"],
                    )


if __name__ == "__main__":
    main()
