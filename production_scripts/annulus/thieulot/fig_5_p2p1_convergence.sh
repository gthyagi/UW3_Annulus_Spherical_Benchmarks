#!/bin/bash
# Thieulot & Puckett (2018), Fig. 5:
# P2/P1 annulus convergence for k = 1, 4, 8.

#PBS -P m18
#PBS -N th_p2p1_conv
#PBS -q normal
#PBS -l walltime=12:00:00
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -l storage=gdata/m18+scratch/m18
#PBS -l wd

set -euo pipefail

INSTALL_SCRIPT="${INSTALL_SCRIPT:-/g/data/m18/software/uw3-pixi/gadi_install_shared.sh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${REPO_ROOT}/benchmarks/annulus/ex_stokes_thieulot.py"
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

ks=(1 4 8)
cellsizes=("1/8" "1/16" "1/32" "1/64" "1/128" "1/256" "1/512")

common_args=(
    -run_on_gadi True
    -uw_vdegree 2
    -uw_pdegree 1
    -uw_pcont True
    -uw_bc_type essential
    -uw_stokes_tol 1e-9
)

run_case() {
    echo
    echo "[$(date)] Running: mpiexec -n ${NCPUS} python3 ${BENCH_SCRIPT} $*"
    mpiexec -n "${NCPUS}" -x LD_PRELOAD=libmpi.so python3 "${BENCH_SCRIPT}" "$@"
}

echo
echo "=== Thieulot Fig. 5: P2/P1 convergence sweep ==="

for k in "${ks[@]}"; do
    for cellsize in "${cellsizes[@]}"; do
        run_case "${common_args[@]}" -uw_k "${k}" -uw_cellsize "${cellsize}"
    done
done

echo
echo "Completed P2/P1 annulus convergence sweep."
