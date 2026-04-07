#!/usr/bin/env python3
"""Submit spherical Thieulot mesh-generation jobs via the common Gadi PBS script."""

import base64
import json
from pathlib import Path
import subprocess

CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]

def slug(value: str) -> str:
    return value.replace("/", "_")


def submit_job(common_pbs: Path, mesh_script: Path, cellsize: str) -> None:
    args = [
        "-uw_benchmark",
        "thieulot",
        "-uw_radius_inner",
        "0.5",
        "-uw_radius_internal",
        "None",
        "-uw_radius_outer",
        "1.0",
        "-uw_cellsize",
        cellsize,
    ]
    args_json_b64 = base64.b64encode(json.dumps(args).encode()).decode()

    cmd = [
        "qsub",
        "-P",
        "m18",
        "-N",
        f"th_sph_mesh_{slug(cellsize)}",
        "-q",
        "hugemembw",
        "-l",
        "walltime=12:00:00",
        "-l",
        "mem=256GB",
        "-l",
        "jobfs=1GB",
        "-l",
        "ncpus=1",
        "-v",
        f"SCRIPT={mesh_script},ARGS_JSON_B64={args_json_b64}",
        str(common_pbs),
    ]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    mesh_script = repo_root / "benchmarks" / "spherical" / "create_spherical_mesh.py"

    for cellsize in CELL_SIZES:
        submit_job(common_pbs, mesh_script, cellsize)


if __name__ == "__main__":
    main()
