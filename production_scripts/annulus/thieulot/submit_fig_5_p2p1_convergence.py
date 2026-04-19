#!/usr/bin/env python3
"""Submit annulus Thieulot Fig. 5 P2/P1 convergence jobs."""

import base64
import json
from pathlib import Path
import subprocess


KS = [1, 4, 8]
CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256", "1/512"]
NCPUS_BY_CELLSIZE = {
    "1/8": 2,
    "1/16": 2,
    "1/32": 2,
    "1/64": 4,
    "1/128": 8,
    "1/256": 16,
    "1/512": 32,
}
MEM_BY_CELLSIZE = {
    "1/8": "8gb",
    "1/16": "8gb",
    "1/32": "8gb",
    "1/64": "16gb",
    "1/128": "32gb",
    "1/256": "64gb",
    "1/512": "128gb",
}


def slug(value: str) -> str:
    return value.replace("/", "_")


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
    bench_script = repo_root / "benchmarks" / "annulus" / "ex_stokes_thieulot.py"

    for k in KS:
        for cellsize in CELL_SIZES:
            ncpus = NCPUS_BY_CELLSIZE[cellsize]
            mem = MEM_BY_CELLSIZE[cellsize]
            args = [
                "-uw_run_on_gadi",
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
                cellsize,
            ]
            submit_job(
                common_pbs=common_pbs,
                script=bench_script,
                job_name=f"th_a_p2p1_k{k}_cs{slug(cellsize)}",
                args=args,
                walltime="06:00:00",
                ncpus=ncpus,
                mem=mem,
            )


if __name__ == "__main__":
    main()
