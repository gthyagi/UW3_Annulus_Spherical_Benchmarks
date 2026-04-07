#!/usr/bin/env python3
"""Submit spherical Kramer mesh jobs using the common PBS script."""

import base64
import json
from pathlib import Path
import subprocess

CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]
CASES = [
    {"name": "smooth", "radius_inner": "1.22", "radius_internal": "None", "radius_outer": "2.22"},
    {"name": "delta", "radius_inner": "1.22", "radius_internal": "2.0", "radius_outer": "2.22"},
]


def slug(value: str) -> str:
    return value.replace("/", "_")


def submit_job(common_pbs: Path, mesh_script: Path, case: dict[str, str], cellsize: str) -> None:
    args = [
        "-uw_benchmark",
        "kramer",
        "-uw_radius_inner",
        case["radius_inner"],
        "-uw_radius_internal",
        case["radius_internal"],
        "-uw_radius_outer",
        case["radius_outer"],
        "-uw_cellsize",
        cellsize,
    ]
    args_json_b64 = base64.b64encode(json.dumps(args).encode()).decode()

    cmd = [
        "qsub",
        "-P",
        "m18",
        "-N",
        f"kr_sph_mesh_{case['name']}_{slug(cellsize)}",
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
        for case in CASES:
            submit_job(common_pbs, mesh_script, case, cellsize)


if __name__ == "__main__":
    main()
