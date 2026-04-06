#!/bin/bash
# Spherical Thieulot supplemental convergence sweep:
# UW P3/P3 spherical-shell runs for m = -1 and m = 3.
# This is not a paper element pair, but is provided as an additional UW study
# alongside the Fig. 4-5 convergence scripts.

#PBS -P m18
#PBS -N th_sph_p3p3
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -l storage=scratch/n69+gdata/n69+scratch/m18+gdata/m18
#PBS -l wd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-${REPO_ROOT}/production_scripts/gadi_install_user.sh}"
BENCH_SCRIPT="${BENCH_SCRIPT:-${REPO_ROOT}/benchmarks/spherical/ex_stokes_thieulot.py}"
# Fall back to a single rank for local dry runs outside PBS.
NCPUS="${PBS_NCPUS:-1}"

# Fail early if the expected shared environment or benchmark entry point is missing.
require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "${INSTALL_SCRIPT}"
require_file "${BENCH_SCRIPT}"
source "${INSTALL_SCRIPT}"
cd "${REPO_ROOT}"

ms=(-1 3)
cellsizes=("1/8" "1/16" "1/32" "1/64" "1/128")

common_args=(
    -uw_run_on_gadi True
    -uw_vdegree 3
    -uw_pdegree 3
    -uw_pcont True
    -uw_bc_type essential
    -uw_stokes_tol 1e-9
)

run_case() {
    echo
    echo "[$(date)] Running: mpiexec -n ${NCPUS} python3 ${BENCH_SCRIPT} $*"
    mpiexec -n "${NCPUS}" python3 "${BENCH_SCRIPT}" "$@"
}

echo
echo "=== Spherical Thieulot supplemental: P3/P3 convergence sweep ==="

for m in "${ms[@]}"; do
    for cellsize in "${cellsizes[@]}"; do
        run_case "${common_args[@]}" -uw_m "${m}" -uw_cellsize "${cellsize}"
    done
done

echo
echo "Completed spherical Thieulot P3/P3 convergence sweep."
