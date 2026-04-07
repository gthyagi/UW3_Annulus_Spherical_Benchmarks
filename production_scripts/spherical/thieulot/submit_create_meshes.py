#!/usr/bin/env python3
"""Submit spherical Thieulot mesh-generation jobs via the common Gadi PBS script."""

from pathlib import Path
import subprocess

CELL_SIZES = ["1/8", "1/16", "1/32", "1/64", "1/128"]


def submit_job(common_pbs: Path, mesh_script: Path, cellsize: str) -> None:
    job_name = f"th_sph_mesh_{cellsize.replace('/', '_')}"

    args = (
        "-uw_radius_inner 0.5 "
        "-uw_radius_outer 1.0 "
        "-uw_radius_internal None "
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


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]

    common_pbs = repo_root / "production_scripts" / "gadi_pbs_job.sh"
    mesh_script = repo_root / "benchmarks" / "spherical" / "create_spherical_mesh.py"

    if not common_pbs.is_file():
        raise FileNotFoundError(f"Missing PBS script: {common_pbs}")
    if not mesh_script.is_file():
        raise FileNotFoundError(f"Missing mesh script: {mesh_script}")

    for cellsize in CELL_SIZES:
        submit_job(common_pbs, mesh_script, cellsize)


if __name__ == "__main__":
    main()
