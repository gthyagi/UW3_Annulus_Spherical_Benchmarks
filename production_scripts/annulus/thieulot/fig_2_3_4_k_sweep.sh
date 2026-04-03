#!/bin/bash
# Thieulot & Puckett (2018), Figs. 2-4:
# P2/P1 at fixed cellsize = 1/128.
# Run only the missing fixed-cellsize cases here.
# Reuse `fig_5_p2p1_convergence.sh` at cellsize = 1/128 for k = 1, 4, 8.

#PBS -P m18
#PBS -N th_fig234_p2p1
#PBS -q normal
#PBS -l walltime=08:00:00
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

ks=(2 3)
cellsize="1/128"

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
echo "=== Thieulot annulus: P2/P1, cellsize=${cellsize}, k=2,3 ==="

for k in "${ks[@]}"; do
    run_case "${common_args[@]}" -uw_k "${k}" -uw_cellsize "${cellsize}"
done

echo
echo "Completed P2/P1 fixed-cellsize sweep for k = 2, 3."
