#!/usr/bin/env python3
"""Submit spherical Thieulot Fig. 4-5 P1/P0 convergence jobs."""

import base64
import json
from pathlib import Path
import subprocess


MS = [-1, 3]
CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]
RESOURCES_BY_CELLSIZE = {
    "1/8": {"ncpus": 1, "mem": "4gb", "walltime": "06:00:00"},
    "1/16": {"ncpus": 2, "mem": "8gb", "walltime": "06:00:00"},
    "1/32": {"ncpus": 16, "mem": "64gb", "walltime": "06:00:00"},
    "1/64": {"ncpus": 144, "mem": "576gb", "walltime": "06:00:00"},
    "1/128": {"ncpus": 1152, "mem": "4608gb", "walltime": "12:00:00"},
}


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
    bench_script = repo_root / "benchmarks" / "spherical" / "ex_stokes_thieulot.py"

    for m in MS:
        for cellsize in CELL_SIZES:
            resources = RESOURCES_BY_CELLSIZE[cellsize]
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
                walltime=resources["walltime"],
                ncpus=resources["ncpus"],
                mem=resources["mem"],
            )


if __name__ == "__main__":
    main()
