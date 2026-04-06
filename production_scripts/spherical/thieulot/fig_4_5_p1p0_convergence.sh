#!/bin/bash
# Thieulot (2017), Figs. 4-5:
# UW P1/P0 spherical-shell convergence for m = -1 and m = 3.
# These runs provide both the velocity (Fig. 4) and pressure (Fig. 5) L2 curves.

#PBS -P m18
#PBS -N th_sph_p1p0
#PBS -q normal
#PBS -l walltime=12:00:00
#PBS -l ncpus=1
#PBS -l mem=8gb
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
    -uw_vdegree 1
    -uw_pdegree 0
    -uw_pcont False
    -uw_bc_type essential
    -uw_stokes_tol 1e-9
)

run_case() {
    echo
    echo "[$(date)] Running: python3 ${BENCH_SCRIPT} $*"
    python3 "${BENCH_SCRIPT}" "$@"
}

echo
echo "=== Thieulot spherical Figs. 4-5: P1/P0 convergence sweep ==="

for m in "${ms[@]}"; do
    for cellsize in "${cellsizes[@]}"; do
        run_case "${common_args[@]}" -uw_m "${m}" -uw_cellsize "${cellsize}"
    done
done

echo
echo "Completed spherical Thieulot P1/P0 convergence sweep."
