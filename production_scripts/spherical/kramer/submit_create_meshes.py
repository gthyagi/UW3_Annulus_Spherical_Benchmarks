#!/usr/bin/env python3
"""Submit spherical Kramer mesh jobs using common PBS script."""

from pathlib import Path
import subprocess

CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]

CASES = [
    {"name": "smooth", "radius_inner": "1.22", "radius_internal": "None", "radius_outer": "2.22"},
    {"name": "delta",  "radius_inner": "1.22", "radius_internal": "2.0",  "radius_outer": "2.22"},
]


def submit(common_pbs, mesh_script, case, cellsize):
    job_name = f"kr_sph_mesh_{case['name']}_{cellsize.replace('/', '_')}"

    args = (
        f"-uw_radius_inner {case['radius_inner']} "
        f"-uw_radius_internal {case['radius_internal']} "
        f"-uw_radius_outer {case['radius_outer']} "
        f"-uw_cellsize {cellsize}"
    )

    cmd = [
        "qsub",
        "-P", "m18",
        "-N", job_name,
        "-q", "hugemembw",
        "-l", "walltime=12:00:00",
        "-l", "mem=256GB",
        "-l", "jobfs=1GB",
        "-l", "ncpus=1",
        "-v", f"SCRIPT={mesh_script},ARGS={args}",
        str(common_pbs),
    ]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]

    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    mesh_script = repo_root / "benchmarks" / "spherical" / "create_spherical_mesh.py"

    for cellsize in CELL_SIZES:
        for case in CASES:
            submit(common_pbs, mesh_script, case, cellsize)


if __name__ == "__main__":
    main()
