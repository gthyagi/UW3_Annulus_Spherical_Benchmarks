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

#PBS -P n69
#PBS -N uw3_job
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=4
#PBS -l mem=16gb
#PBS -l storage=scratch/n69+gdata/n69+scratch/m18+gdata/m18
#PBS -l wd

# ============================================================
# USER CONFIG
# ============================================================

INSTALL_SCRIPT=/home/565/tg7098/UW3_Annulus_Spherical_Benchmarks/production_scripts/gadi_install_user.sh
SCRIPT=${SCRIPT:-gadi_test_stokes.py}

# ============================================================
# ENV
# ============================================================

source "${INSTALL_SCRIPT}"

# ============================================================
# RUN
# ============================================================

mpiexec -n "${PBS_NCPUS}" python3 "${SCRIPT}"