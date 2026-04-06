#!/bin/bash
# Prebuild spherical Thieulot shell meshes on Gadi
# for r_i = 0.5, r_o = 1.0 and cellsize = 1/8 ... 1/128.

#PBS -P m18
#PBS -N th_sph_mesh
#PBS -q hugemembw
#PBS -l walltime=48:00:00
#PBS -l mem=256GB
#PBS -l jobfs=1GB
#PBS -l ncpus=1
#PBS -l software=underworld3
#PBS -l wd
#PBS -l storage=scratch/m18+gdata/m18+scratch/n69+gdata/n69

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-${REPO_ROOT}/production_scripts/gadi_install_user.sh}"
MESH_SCRIPT="${REPO_ROOT}/benchmarks/spherical/create_spherical_mesh.py"

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "${INSTALL_SCRIPT}"
require_file "${MESH_SCRIPT}"
source "${INSTALL_SCRIPT}"
cd "${REPO_ROOT}"

cellsizes=("1/8" "1/16" "1/32" "1/64" "1/128")

run_mesh() {
    echo
    echo "[$(date)] Running: $*"
    mpiexec -n 1 python3 "${MESH_SCRIPT}" "$@"
}

for cellsize in "${cellsizes[@]}"; do
    run_mesh \
        -uw_radius_inner 0.5 \
        -uw_radius_outer 1.0 \
        -uw_radius_internal None \
        -uw_cellsize "${cellsize}"
done

echo
echo "Completed spherical Thieulot mesh prebuild."
