#!/usr/bin/env python3
"""Submit spherical Kramer Fig. 4 l/m sweeps for cases 1-4."""

import base64
import json
from pathlib import Path
import subprocess


CELL_SIZES = ["1/8", "1/16", "1/32", "1/64"]  # "1/96"
LM_PAIRS = [(2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]
CASES = ["case1", "case2", "case3", "case4"]
METRICS_FROM_CHECKPOINT_ONLY = False

RESOURCES_BY_CELLSIZE = {
    "1/8": {"ncpus": 8, "mem": "32gb", "walltime": "01:00:00"},
    "1/16": {"ncpus": 32, "mem": "128gb", "walltime": "01:00:00"},
    "1/32": {"ncpus": 192, "mem": "768gb", "walltime": "01:00:00"},
    "1/64": {"ncpus": 1440, "mem": "5760gb", "walltime": "01:30:00"},
    "1/96": {"ncpus": 3840, "mem": "14976gb", "walltime": "02:00:00"},
}

# Case1: Free-slip boundaries and delta function density perturbation
# Case2: Free-slip boundaries and smooth density distribution
# Case3: Zero-slip boundaries and delta function density perturbation
# Case4: Zero-slip boundaries and smooth density distribution
FREE_SLIP_CASES = {"case1", "case2"}
DELTA_FN_CASES = {"case1", "case3"}


def slug(value: int | str) -> str:
    return str(value).replace("/", "_")


def case_density(case_name: str) -> str:
    if case_name in DELTA_FN_CASES:
        return "delta-fn"
    return "smooth"


def case_extra_args(case_name: str) -> list[str]:
    if case_name in FREE_SLIP_CASES:
        return ["-uw_freeslip_type", "nitsche"]
    return []


def resources_for_case(case_name: str, cellsize: str) -> dict[str, int | str]:
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
    bench_script = repo_root / "benchmarks" / "spherical" / "ex_stokes_kramer.py"

    for case_name in CASES:
        extra_args = case_extra_args(case_name)
        for l, m in LM_PAIRS:
            for cellsize in CELL_SIZES:
                resources = resources_for_case(case_name, cellsize)
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
                    "-uw_metrics_from_checkpoint_only",
                    str(METRICS_FROM_CHECKPOINT_ONLY),
                ]
                submit_job(
                    common_pbs=common_pbs,
                    script=bench_script,
                    job_name=f"kr_s_{case_name}_l{l}_m{slug(m)}_cs{slug(cellsize)}",
                    args=args,
                    walltime=resources["walltime"],
                    ncpus=resources["ncpus"],
                    mem=resources["mem"],
                )


if __name__ == "__main__":
    main()
