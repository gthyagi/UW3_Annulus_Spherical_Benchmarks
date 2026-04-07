#!/bin/bash
#
# Underworld3 PBS job script (Gadi)
#
# Usage:
#   qsub gadi_pbs_job.sh
#   qsub -v SCRIPT=/path/to/script.py gadi_pbs_job.sh
#
# Notes:
#   - Runs from submission directory (#PBS -l wd)
#   - Uses pixi + PETSc + HDF5 environment via gadi_install_user.sh
#   - SCRIPT can be relative (to submission dir) or absolute
#

#PBS -P m18
#PBS -N uw3_job
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=8
#PBS -l mem=16gb
#PBS -l storage=scratch/n69+gdata/n69+scratch/m18+gdata/m18
#PBS -l wd

# ============================================================
# USER CONFIG
# ============================================================

INSTALL_SCRIPT=/home/565/tg7098/UW3_Annulus_Spherical_Benchmarks/production_scripts/gadi_install_user.sh
SCRIPT=${SCRIPT:-gadi_test_stokes.py}
ARGS=${ARGS:-}
ARGS_JSON_B64=${ARGS_JSON_B64:-}

# ============================================================
# ENV
# ============================================================

source "${INSTALL_SCRIPT}"

# ============================================================
# RUN
# ============================================================

if [[ -n "${ARGS_JSON_B64}" ]]; then
    mapfile -t RUN_ARGS < <(
        python3 - <<'PY'
import base64
import json
import os

for arg in json.loads(base64.b64decode(os.environ["ARGS_JSON_B64"]).decode()):
    print(arg)
PY
    )
elif [[ -n "${ARGS}" ]]; then
    # shellcheck disable=SC2206
    RUN_ARGS=(${ARGS})
else
    RUN_ARGS=()
fi

mpiexec -n "${PBS_NCPUS}" python3 "${SCRIPT}" "${RUN_ARGS[@]}"
