#!/bin/bash
# Kramer et al. (2021), Fig. 3:
# case1 (free-slip) and case3 (zero-slip)
# delta-function annulus cases using the paper's P2/P1 setup.

#PBS -P m18
#PBS -N kr_fig3_all
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -l storage=gdata/m18+scratch/m18
#PBS -l wd

set -euo pipefail

INSTALL_SCRIPT="${INSTALL_SCRIPT:-/g/data/m18/software/uw3-pixi/gadi_install_shared.sh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_SCRIPT="${REPO_ROOT}/benchmarks/annulus/ex_stokes_kramer.py"
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

ns=(2 8 32)
cellsizes=("1/8" "1/16" "1/32" "1/64" "1/128" "1/256")

run_case() {
    echo
    echo "[$(date)] Running: $*"
    mpiexec -n "${NCPUS}" -x LD_PRELOAD=libmpi.so \
        python3 "${BENCH_SCRIPT}" "$@"
}

run_all() {
    local case_name="$1"
    shift
    local extra_args=("$@")

    echo
    echo "=== case: ${case_name} ==="

    for n in "${ns[@]}"; do
        for cellsize in "${cellsizes[@]}"; do
            run_case \
                -run_on_gadi True \
                -uw_case "${case_name}" \
                -uw_vdegree 2 \
                -uw_pdegree 1 \
                -uw_pcont True \
                -uw_stokes_tol 1e-9 \
                "${extra_args[@]}" \
                -uw_n "${n}" \
                -uw_cellsize "${cellsize}"
        done
    done
}

# case1: free-slip
run_all case1 -uw_freeslip_type nitsche

# case3: zero-slip
run_all case3

echo
echo "Completed all Kramer Fig. 3 sweeps."
