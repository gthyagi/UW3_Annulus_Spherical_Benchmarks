#!/bin/bash
# Kramer et al. (2021), Fig. 4:
# - case1: free-slip, delta-function forcing
# - case3: zero-slip, delta-function forcing
# Spherical P2/P1 runs for the figure's (l,m) series:
# (2,1), (2,2), (4,2), (4,4), (8,4), (8,8)
# and five refinement levels approximated by cellsize = 1/8 ... 1/128.

#PBS -P m18
#PBS -N kr_sph_f4_dlt
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
BENCH_SCRIPT="${REPO_ROOT}/benchmarks/spherical/ex_stokes_kramer.py"
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

cellsizes=("1/8" "1/16" "1/32" "1/64" "1/128")

# (l m)
lm_pairs=(
    "2 1"
    "2 2"
    "4 2"
    "4 4"
    "8 4"
    "8 8"
)

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

    for lm in "${lm_pairs[@]}"; do
        read -r l m <<< "${lm}"
        for cellsize in "${cellsizes[@]}"; do
            run_case \
                -uw_run_on_gadi True \
                -uw_case "${case_name}" \
                -uw_l "${l}" \
                -uw_m "${m}" \
                -uw_vdegree 2 \
                -uw_pdegree 1 \
                -uw_pcont True \
                -uw_stokes_tol 1e-8 \
                "${extra_args[@]}" \
                -uw_cellsize "${cellsize}"
        done
    done
}

# case1: free-slip
run_sweep case1 -uw_freeslip_type nitsche

# case3: zero-slip
run_sweep case3

echo
echo "Completed spherical Kramer Fig. 4 delta-function sweeps."
