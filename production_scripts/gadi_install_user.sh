#!/bin/bash
#
# Underworld3 per-user install script for NCI Gadi (pixi-based) - modified by TG
#
# Installs UW3 to /g/data/n69/$USER/uw3-pixi/ using pixi for Python
# package management. Each user manages their own install.
#
# Gadi modules provide OpenMPI and HDF5; pixi handles pure Python
# dependencies. mpi4py, PETSc, and h5py are built against the system MPI/HDF5.
#
# Usage:
#   source gadi_install_user.sh
#   source gadi_install_user.sh install
#
# NOTE: This script is designed to be sourced, NOT executed directly.
# Do NOT add 'set -e' here — it would cause your shell to close on any
# error since the script runs in your current shell.

usage="
Usage:
  source <this_script_name>
      Activate existing install

  source <this_script_name> install
      Install / rebuild environment
"

while getopts ':h' option; do
  case "$option" in
    h)
      echo "$usage"
      (return 0 2>/dev/null) && return 0 || exit 0
      ;;
    \?)
      echo "Error: Incorrect options"
      echo "$usage"
      (return 0 2>/dev/null) && return 0 || exit 0
      ;;
  esac
done

# ============================================================
# CONFIGURATION
# ============================================================

export UW3_BRANCH=development
export UW3_REPO="https://github.com/gthyagi/underworld3.git"
export INSTALL_NAME=underworld3

# Persistent source / PETSc location
export BASE_PATH=/g/data/n69/${USER}/uw3-pixi
export UW3_PATH=${BASE_PATH}/${INSTALL_NAME}

# Pixi binary / cache / detached envs on scratch
export PIXI_HOME="/scratch/n69/${USER}/.pixi"
export PIXI_CACHE_DIR="/scratch/n69/${USER}/.pixi-cache"
export PIXI_ENV_ROOT="/scratch/n69/${USER}/pixi-envs"

# ============================================================
# DERIVED PATHS
# ============================================================

export PIXI_MANIFEST="${UW3_PATH}/pixi.toml"
export PETSC_DIR="${UW3_PATH}/petsc-custom/petsc"
export PETSC_ARCH=petsc-4-uw-openmpi

export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_io=ompio
export CDIR=$PWD

# ============================================================
# HELPERS
# ============================================================

get_active_pixi_prefix() {
    python3 - <<'PY'
import sys
from pathlib import Path
print(Path(sys.executable).resolve().parents[1])
PY
}

configure_pixi_detached_envs() {
    mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}" "${PIXI_ENV_ROOT}"
    cd "${UW3_PATH}" || return 1
    pixi config set detached-environments "${PIXI_ENV_ROOT}" >/dev/null
    cd "${CDIR}" || return 1
}

activate_hpc_env() {
    if [ "${PIXI_ENVIRONMENT_NAME:-}" != "hpc" ]; then
        eval "$(pixi shell-hook -e hpc --manifest-path "${PIXI_MANIFEST}")"
    fi
}

# ============================================================
# ENVIRONMENT ACTIVATION
# ============================================================

load_env() {
    module purge
    module load openmpi/4.1.7 hdf5/1.12.2p gmsh/4.13.1 cmake/3.31.6

    export MPI_DIR
    MPI_DIR="$(dirname "$(dirname "$(which mpicc)")")"

    export PATH="${PIXI_HOME}/bin:${PATH}"

    if command -v pixi &>/dev/null && [ -d "${UW3_PATH}" ] && [ -f "${PIXI_MANIFEST}" ]; then
        configure_pixi_detached_envs
        activate_hpc_env
    fi

    export LD_LIBRARY_PATH="${HDF5_DIR}/lib:${LD_LIBRARY_PATH}"

    if [ -d "${PETSC_DIR}/${PETSC_ARCH}" ]; then
        export PYTHONPATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PYTHONPATH}"
    fi

    export PYTHONNOUSERSITE=1
    export OPENBLAS_NUM_THREADS=1
    export OMPI_MCA_io=ompio

    echo "==> Environment ready"
    echo "    MPI_DIR:       ${MPI_DIR}"
    echo "    HDF5_DIR:      ${HDF5_DIR}"
    echo "    UW3_PATH:      ${UW3_PATH}"
    echo "    PETSC_DIR:     ${PETSC_DIR}"
    echo "    PETSC_ARCH:    ${PETSC_ARCH}"
    echo "    PIXI_HOME:     ${PIXI_HOME}"
    echo "    PIXI_CACHE_DIR:${PIXI_CACHE_DIR}"
    echo "    PIXI_ENV_ROOT: ${PIXI_ENV_ROOT}"
}

# ============================================================
# INSTALLATION FUNCTIONS
# ============================================================

setup_pixi() {
    if command -v pixi &>/dev/null; then
        echo "==> pixi already installed: $(pixi --version)"
        return 0
    fi
    echo "==> Installing pixi to ${PIXI_HOME}..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${PIXI_HOME}/bin:${PATH}"
    echo "==> pixi installed: $(pixi --version)"
}

clone_uw3() {
    if [ ! -d "${UW3_PATH}" ]; then
        echo "==> Cloning Underworld3 (branch: ${UW3_BRANCH}) to ${UW3_PATH}..."
        mkdir -p "${BASE_PATH}"
        git clone --branch "${UW3_BRANCH}" --depth 1 "${UW3_REPO}" "${UW3_PATH}"
    else
        echo "==> Underworld3 source already present at ${UW3_PATH}"
    fi
}

install_pixi_env() {
    echo "==> Installing pixi hpc environment on scratch (~3 min)..."
    configure_pixi_detached_envs
    pixi install -e hpc --manifest-path "${PIXI_MANIFEST}"
    activate_hpc_env
    echo "==> pixi hpc environment ready"
}

install_mpi4py() {
    echo "==> Building mpi4py from source against Gadi OpenMPI..."
    pip install --no-binary :all: --no-cache-dir --force-reinstall "mpi4py>=4,<5"
    echo "==> mpi4py installed"
}

install_petsc() {
    echo "==> Building PETSc with AMR tools (~1 hour)..."
    bash "${UW3_PATH}/petsc-custom/build-petsc.sh"
    export PYTHONPATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PYTHONPATH}"
    echo "==> PETSc installed"
}

install_h5py() {
    echo "==> Building h5py against Gadi HDF5 module..."

    local _conda_lib
    _conda_lib="$(get_active_pixi_prefix)/lib"

    local _hidden=()
    local _f
    for _f in "${_conda_lib}"/libhdf5*.so*; do
        [ -f "${_f}" ] && [[ "${_f}" != *.h5build ]] || continue
        mv "${_f}" "${_f}.h5build"
        _hidden+=("${_f}")
    done
    [ ${#_hidden[@]} -gt 0 ] && echo "  Hid ${#_hidden[@]} pixi HDF5 lib(s) for clean build"

    (
        unset LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
        export LDFLAGS="-L${HDF5_DIR}/lib -Wl,--disable-new-dtags,-rpath,${HDF5_DIR}/lib"
        export LD_LIBRARY_PATH="${HDF5_DIR}/lib:${MPI_DIR}/lib"

        python -m pip install --upgrade "setuptools>=77" wheel packaging

        CC=mpicc \
        HDF5_MPI="ON" \
        HDF5_DIR="${HDF5_DIR}" \
        HDF5_VERSION="1.12.2" \
        CFLAGS="-I${HDF5_DIR}/include -include ${HDF5_DIR}/include/hdf5.h -include ${HDF5_DIR}/include/H5FDmpio.h" \
        python -m pip install --no-build-isolation --no-binary=h5py --no-cache-dir --force-reinstall --no-deps h5py
    )
    local _rc=$?

    for _f in "${_hidden[@]}"; do
        mv "${_f}.h5build" "${_f}"
    done
    [ ${#_hidden[@]} -gt 0 ] && echo "  Restored ${#_hidden[@]} pixi HDF5 lib(s)"

    [ $_rc -ne 0 ] && { echo "ERROR: h5py build failed (rc=$_rc)"; return $_rc; }
    echo "==> h5py installed"
}

install_uw3() {
    echo "==> Installing Underworld3..."
    cd "${UW3_PATH}"
    pip install --no-build-isolation -e .
    cd "${CDIR}"
    echo "==> Underworld3 installed"
}

check_petsc_exists() {
    python3 -c "from petsc4py import PETSc" 2>/dev/null
}

check_uw3_exists() {
    python3 -c "import underworld3" 2>/dev/null
}

verify_install() {
    echo "==> Verifying installation..."
    python3 -c "
from mpi4py import MPI
print(f'mpi4py OK   — MPI version: {MPI.Get_version()}')
from petsc4py import PETSc
print(f'petsc4py OK — PETSc version: {PETSc.Sys.getVersion()}')
import h5py
print(f'h5py OK     — HDF5 version: {h5py.version.hdf5_version}')
import underworld3 as uw
print(f'underworld3 OK — version: {uw.__version__}')
"
    echo ""
    echo "==> Single-process MPI import check:"
    python3 -c "from mpi4py import MPI; print(f'mpi4py MPI import OK (rank 0 of 1)')"
    echo "==> All checks passed"
    echo ""
    echo "    NOTE: Multi-rank MPI tests must be run from a compute node (PBS job)."
    echo "    Example: mpirun -n 4 python3 -c \"from mpi4py import MPI; print(MPI.COMM_WORLD.rank)\""
}

# ============================================================
# ENTRY POINT
# ============================================================

load_env

if [ "${1}" = "install" ]; then
    echo ""
    echo "Starting user installation..."
    echo "  BASE_PATH:      ${BASE_PATH}"
    echo "  UW3_PATH:       ${UW3_PATH}"
    echo "  UW3_BRANCH:     ${UW3_BRANCH}"
    echo "  PIXI_HOME:      ${PIXI_HOME}"
    echo "  PIXI_CACHE_DIR: ${PIXI_CACHE_DIR}"
    echo "  PIXI_ENV_ROOT:  ${PIXI_ENV_ROOT}"
    echo ""
    setup_pixi
    clone_uw3
    install_pixi_env
    install_mpi4py
    if ! check_petsc_exists; then
        install_petsc
    else
        echo "==> PETSc already installed, skipping"
    fi
    install_h5py
    if ! check_uw3_exists; then
        install_uw3
    else
        echo "==> Underworld3 already installed, skipping"
    fi
    verify_install
    echo ""
    echo "=========================================="
    echo "User installation complete!"
    echo "To activate: source $(realpath "${BASH_SOURCE[0]}")"
    echo "=========================================="
fi
