#!/bin/bash
# Kramer et al. (2021), Fig. 3:
# - case2: free-slip (Nitsche), smooth forcing
# - case4: zero-slip, smooth forcing
# k = 2, 8; n = 2, 8, 32; cellsize = 1/8 ... 1/256
# using the paper's P2/P1 setup.

#PBS -P m18
#PBS -N kr_fig3_c2_c4
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

ks=(2 8)
ns=(2 8 32)
cellsizes=("1/8" "1/16" "1/32" "1/64" "1/128" "1/256")

run_case() {
    echo
    echo "[$(date)] Running: $*"
    mpiexec -n "${NCPUS}" python3 "${BENCH_SCRIPT}" "$@"
}

run_sweep() {
    local case_name="$1"
    shift
    local extra_args=("$@")

    echo
    echo "=== case: ${case_name} ==="

    for k in "${ks[@]}"; do
        for n in "${ns[@]}"; do
            for cellsize in "${cellsizes[@]}"; do
                run_case \
                    -run_on_gadi True \
                    -uw_case "${case_name}" \
                    -uw_k "${k}" \
                    -uw_vdegree 2 \
                    -uw_pdegree 1 \
                    -uw_pcont True \
                    -uw_stokes_tol 1e-9 \
                    "${extra_args[@]}" \
                    -uw_n "${n}" \
                    -uw_cellsize "${cellsize}"
            done
        done
    done
}

# case2: free-slip
run_sweep case2 -uw_freeslip_type nitsche

# case4: zero-slip
run_sweep case4

echo
echo "Completed Kramer Fig. 3 case2 and case4 sweeps."
