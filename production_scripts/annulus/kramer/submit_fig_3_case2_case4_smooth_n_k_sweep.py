#!/usr/bin/env python3
"""Submit annulus Kramer Fig. 3 smooth sweeps."""

import base64
import json
from pathlib import Path
import subprocess


CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128", "1/256"]
KS = [2, 8]
NS = [2, 8, 32]
CASES = [
    ("case2", ["-uw_freeslip_type", "nitsche"]),
    ("case4", []),
]
NCPUS_BY_CELLSIZE = {
    "1/8": 2,
    "1/16": 2,
    "1/32": 2,
    "1/64": 8,
    "1/128": 8,
    "1/256": 16,
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
    bench_script = repo_root / "benchmarks" / "annulus" / "ex_stokes_kramer.py"

    for case_name, extra_args in CASES:
        for k in KS:
            for n in NS:
                for cellsize in CELL_SIZES:
                    ncpus = NCPUS_BY_CELLSIZE[cellsize]
                    args = [
                        "-uw_run_on_gadi",
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
                        walltime="06:00:00",
                        ncpus=ncpus,
                        mem="64gb",
                    )


if __name__ == "__main__":
    main()
