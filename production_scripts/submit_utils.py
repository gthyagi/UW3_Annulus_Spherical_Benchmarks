#!/usr/bin/env python3
"""Shared helpers for submitting benchmark jobs through the common Gadi PBS script."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


def repo_root_from(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    for parent in resolved.parents:
        if parent.name == "production_scripts":
            return parent.parent
    raise ValueError(f"Could not locate production_scripts above: {resolved}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def slug(value: object) -> str:
    return str(value).replace("/", "_").replace("-", "neg").replace(".", "p")


def submit_job(
    *,
    common_pbs: Path,
    script: Path,
    job_name: str,
    args: list[str],
    walltime: str,
    ncpus: int,
    mem: str,
    queue: str = "normal",
    project: str = "m18",
) -> None:
    require_file(common_pbs, "PBS launcher")
    require_file(script, "benchmark script")

    cmd = [
        "qsub",
        "-P",
        project,
        "-N",
        job_name,
        "-q",
        queue,
        "-l",
        f"walltime={walltime}",
        "-l",
        f"ncpus={ncpus}",
        "-l",
        f"mem={mem}",
        "-v",
        f"SCRIPT={script},ARGS={' '.join(args)}",
        str(common_pbs),
    ]

    print("Submitting:", shlex.join(cmd))
    subprocess.run(cmd, check=True)
