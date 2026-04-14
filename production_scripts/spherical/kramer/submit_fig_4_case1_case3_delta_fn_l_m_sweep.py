#!/usr/bin/env python3
"""Submit spherical Kramer Fig. 4 delta-function sweeps."""

import base64
import json
from pathlib import Path
import subprocess


CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/96"]
RESOURCES_BY_CELLSIZE = {
    "1/8": {"ncpus": 8, "mem": "32gb", "walltime": "01:00:00"},
    "1/16": {"ncpus": 32, "mem": "128gb", "walltime": "06:00:00"},
    "1/32": {"ncpus": 144, "mem": "576gb", "walltime": "06:00:00"},
    "1/64": {"ncpus": 1056, "mem": "4118gb", "walltime": "06:00:00"},
    "1/96": {"ncpus": 3840, "mem": "14976gb", "walltime": "05:00:00"},
}
LM_PAIRS = [(2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]
CASES = [
    ("case1", ["-uw_freeslip_type", "nitsche"]),
    ("case3", []),
]


def slug(value: int | str) -> str:
    return str(value).replace("/", "_")


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

    for case_name, extra_args in CASES:
        for l, m in LM_PAIRS:
            for cellsize in CELL_SIZES:
                resources = RESOURCES_BY_CELLSIZE[cellsize]
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
                    walltime=resources["walltime"],
                    ncpus=resources["ncpus"],
                    mem=resources["mem"],
                )


if __name__ == "__main__":
    main()
