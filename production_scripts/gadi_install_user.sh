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

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: this script requires Bash"
    echo "Start Bash first, then source this script again."
    return 1
fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: this script must be sourced, not executed"
    echo "Use: source ${BASH_SOURCE[0]} [install]"
    exit 1
fi

usage="
Usage:
  source <this_script_name>
      Activate existing install

  source <this_script_name> install
      Install / rebuild environment

  UW3_BRANCH=<branch> source <this_script_name> install
      Select the branch used when a new UW3 source clone is required
"

OPTIND=1
while getopts ':h' option; do
  case "$option" in
    h)
      echo "$usage"
      return 0
      ;;
    \?)
      echo "Error: Incorrect options"
      echo "$usage"
      return 0
      ;;
  esac
done

# ============================================================
# CONFIGURATION
# ============================================================

export UW3_BRANCH="${UW3_BRANCH:-development}"
export UW3_REPO="https://github.com/gthyagi/underworld3.git"
export INSTALL_NAME=underworld3

# Persistent source and PETSc location
export BASE_PATH="/g/data/n69/${USER}/uw3-pixi"
export UW3_PATH="${BASE_PATH}/${INSTALL_NAME}"

# Persistent Pixi executable
export PIXI_HOME="${BASE_PATH}/.pixi"

# Rebuildable cache and detached environments on scratch
export PIXI_CACHE_DIR="/scratch/n69/${USER}/.pixi-cache"
export PIXI_ENV_ROOT="/scratch/n69/${USER}/pixi-envs"

# ============================================================
# DERIVED PATHS
# ============================================================

export PIXI_MANIFEST="${UW3_PATH}/pixi.toml"
export PETSC_DIR="${UW3_PATH}/petsc-custom/petsc"
export PETSC_ARCH=""

export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_io=ompio

# ============================================================
# HELPERS
# ============================================================

prepend_colon_path() {
    local _var_name="$1"
    local _path="$2"
    local _current="${!_var_name:-}"
    local _entry
    local _filtered=""
    local -a _entries=()

    IFS=':' read -r -a _entries <<< "${_current}"
    for _entry in "${_entries[@]}"; do
        [ -n "${_entry}" ] || continue
        [ "${_entry}" = "${_path}" ] && continue
        _filtered="${_filtered:+${_filtered}:}${_entry}"
    done

    if [ -n "${_filtered}" ]; then
        printf -v "${_var_name}" '%s:%s' "${_path}" "${_filtered}"
    else
        printf -v "${_var_name}" '%s' "${_path}"
    fi
    export "${_var_name?}"
}

resolve_petsc_arch() {
    local _version_file="${UW3_PATH}/petsc-custom/.petsc-version"
    local _petsc_version="324"

    if [ -f "${_version_file}" ]; then
        _petsc_version="$(tr -d '[:space:]' < "${_version_file}")"
    fi
    if ! [[ "${_petsc_version}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid PETSc version '${_petsc_version}' in ${_version_file}"
        return 1
    fi

    export PETSC_ARCH="petsc-${_petsc_version}-uw-openmpi"
}

refresh_petsc_pythonpath() {
    local _entry
    local _filtered=""
    local -a _entries=()

    IFS=':' read -r -a _entries <<< "${PYTHONPATH:-}"
    for _entry in "${_entries[@]}"; do
        [ -n "${_entry}" ] || continue
        case "${_entry}" in
            "${PETSC_DIR}"/petsc-*-uw-openmpi/lib) continue ;;
        esac
        _filtered="${_filtered:+${_filtered}:}${_entry}"
    done

    if [ -d "${PETSC_DIR}/${PETSC_ARCH}/lib" ]; then
        _filtered="${PETSC_DIR}/${PETSC_ARCH}/lib${_filtered:+:${_filtered}}"
    fi
    export PYTHONPATH="${_filtered}"
}

configure_runtime_libraries() {
    if [ -n "${CONDA_PREFIX:-}" ]; then
        prepend_colon_path LD_LIBRARY_PATH "${CONDA_PREFIX}/lib"
    fi
    prepend_colon_path LD_LIBRARY_PATH "${MPI_DIR}/lib"
    prepend_colon_path LD_LIBRARY_PATH "${HDF5_DIR}/lib"

    if [ -n "${CONDA_PREFIX:-}" ]; then
        local _pixi_libstdcxx="${CONDA_PREFIX}/lib/libstdc++.so.6"
        if [ ! -f "${_pixi_libstdcxx}" ]; then
            echo "ERROR: Pixi C++ runtime not found: ${_pixi_libstdcxx}"
            return 1
        fi
        case ":${LD_PRELOAD:-}:" in
            *":${_pixi_libstdcxx}:"*) ;;
            *) export LD_PRELOAD="${_pixi_libstdcxx}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
        esac
    fi
}

configure_pixi_detached_envs() {
    mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}" "${PIXI_ENV_ROOT}"
    (
        cd "${UW3_PATH}" || exit 1
        "${PIXI_HOME}/bin/pixi" config set detached-environments "${PIXI_ENV_ROOT}" >/dev/null
    )
}

activate_hpc_env() {
    if [ "${PIXI_ENVIRONMENT_NAME:-}" = "hpc" ]; then
        if [ "${PIXI_PROJECT_ROOT:-}" != "${UW3_PATH}" ]; then
            echo "ERROR: an hpc environment from another Pixi project is already active"
            echo "Start a fresh shell before sourcing this script."
            return 1
        fi
        if [ ! -x "${CONDA_PREFIX:-}/bin/python3" ]; then
            echo "ERROR: the active hpc environment is incomplete or has been purged"
            echo "Start a fresh shell and run: source gadi_install_user.sh install"
            return 1
        fi
        return 0
    fi

    eval "$("${PIXI_HOME}/bin/pixi" shell-hook -e hpc --manifest-path "${PIXI_MANIFEST}")" || return 1
    [ "${PIXI_ENVIRONMENT_NAME:-}" = "hpc" ] || return 1
    [ "${PIXI_PROJECT_ROOT:-}" = "${UW3_PATH}" ] || return 1
    [ -x "${CONDA_PREFIX:-}/bin/python3" ] || return 1
}

# ============================================================
# ENVIRONMENT ACTIVATION
# ============================================================

load_env() {
    local _mpicc

    module purge || return 1
    module load openmpi/4.1.7 hdf5/1.12.2p gmsh/4.13.1 cmake/3.31.6 || return 1

    _mpicc="$(command -v mpicc)" || {
        echo "ERROR: mpicc was not provided by the Gadi OpenMPI module"
        return 1
    }
    export MPI_DIR
    MPI_DIR="$(dirname "$(dirname "${_mpicc}")")"

    export PATH="${PIXI_HOME}/bin:${PATH}"

    if [ -x "${PIXI_HOME}/bin/pixi" ] && [ -d "${UW3_PATH}" ] && [ -f "${PIXI_MANIFEST}" ]; then
        configure_pixi_detached_envs || return 1
        activate_hpc_env || return 1
    fi

    configure_runtime_libraries || return 1
    resolve_petsc_arch || return 1
    refresh_petsc_pythonpath || return 1

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
    if [ -d "${UW3_PATH}/.git" ]; then
        echo "    UW3_BRANCH:    $(git -C "${UW3_PATH}" branch --show-current)"
    fi
}

# ============================================================
# INSTALLATION FUNCTIONS
# ============================================================

setup_pixi() {
    if [ -x "${PIXI_HOME}/bin/pixi" ] && "${PIXI_HOME}/bin/pixi" --version &>/dev/null; then
        echo "==> pixi already installed: $("${PIXI_HOME}/bin/pixi" --version)"
        return 0
    fi
    echo "==> Installing pixi to ${PIXI_HOME}..."
    mkdir -p "${PIXI_HOME}" || return 1
    local _installer
    _installer="$(mktemp "${TMPDIR:-/tmp}/pixi-installer.XXXXXX")" || return 1
    if ! curl -fsSL https://pixi.sh/install.sh -o "${_installer}"; then
        rm -f "${_installer}"
        return 1
    fi
    if ! bash "${_installer}"; then
        rm -f "${_installer}"
        return 1
    fi
    rm -f "${_installer}"
    export PATH="${PIXI_HOME}/bin:${PATH}"
    [ -x "${PIXI_HOME}/bin/pixi" ] || return 1
    echo "==> pixi installed: $("${PIXI_HOME}/bin/pixi" --version)"
}

clone_uw3() {
    if [ ! -d "${UW3_PATH}" ]; then
        echo "==> Cloning Underworld3 (branch: ${UW3_BRANCH}) to ${UW3_PATH}..."
        mkdir -p "${BASE_PATH}"
        git clone --branch "${UW3_BRANCH}" --depth 1 "${UW3_REPO}" "${UW3_PATH}" || return 1
    elif [ ! -d "${UW3_PATH}/.git" ]; then
        echo "ERROR: ${UW3_PATH} exists but is not a Git checkout"
        return 1
    else
        local _active_branch
        _active_branch="$(git -C "${UW3_PATH}" branch --show-current)" || return 1
        if [ -z "${_active_branch}" ]; then
            echo "ERROR: ${UW3_PATH} is in detached-HEAD state"
            echo "Switch to the UW3 branch you want to install, then rerun this script."
            return 1
        fi
        echo "==> Underworld3 source already present at ${UW3_PATH}"
        echo "    Using checked-out branch: ${_active_branch}"
        export UW3_BRANCH="${_active_branch}"
    fi
    resolve_petsc_arch || return 1
}

install_pixi_env() {
    echo "==> Installing pixi hpc environment on scratch (~3 min)..."
    configure_pixi_detached_envs || return 1
    "${PIXI_HOME}/bin/pixi" install -e hpc --manifest-path "${PIXI_MANIFEST}" || return 1
    activate_hpc_env || return 1
    configure_runtime_libraries || return 1
    resolve_petsc_arch || return 1
    refresh_petsc_pythonpath || return 1
    echo "==> pixi hpc environment ready"
}

install_mpi4py() {
    echo "==> Building mpi4py from source against Gadi OpenMPI..."
    MPICC="${MPI_DIR}/bin/mpicc" \
        python3 -m pip install --no-binary :all: --no-cache-dir \
        --force-reinstall --no-deps "mpi4py>=4,<5" || return 1
    python3 -c "from mpi4py import MPI; assert 'open mpi' in MPI.Get_library_version().lower()" || return 1
    echo "==> mpi4py installed"
}

archive_incompatible_petsc() {
    local _arch_dir="${PETSC_DIR}/${PETSC_ARCH}"
    [ -d "${_arch_dir}" ] || return 0

    local _backup
    _backup="${_arch_dir}.incompatible-$(date +%Y%m%d-%H%M%S)"
    echo "==> Preserving incompatible PETSc build at ${_backup}"
    mv "${_arch_dir}" "${_backup}" || return 1
    refresh_petsc_pythonpath || return 1
}

install_petsc() {
    echo "==> Building PETSc with AMR tools (~1 hour)..."
    bash "${UW3_PATH}/petsc-custom/build-petsc.sh" || return 1
    [ -d "${PETSC_DIR}/${PETSC_ARCH}/lib" ] || {
        echo "ERROR: expected PETSc architecture was not created: ${PETSC_ARCH}"
        return 1
    }
    refresh_petsc_pythonpath || return 1
    check_petsc_exists || return 1
    echo "==> PETSc installed"
}

install_h5py() {
    echo "==> Building h5py against Gadi HDF5 module..."

    local _conda_lib
    _conda_lib="${CONDA_PREFIX}/lib"

    local _hidden=()
    local _f
    for _f in "${_conda_lib}"/libhdf5*.so*; do
        [ -f "${_f}" ] && [[ "${_f}" != *.h5build ]] || continue
        mv "${_f}" "${_f}.h5build"
        _hidden+=("${_f}")
    done
    [ "${#_hidden[@]}" -gt 0 ] && echo "  Hid ${#_hidden[@]} pixi HDF5 lib(s) for clean build"

    (
        unset LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
        export LDFLAGS="-L${HDF5_DIR}/lib -Wl,--disable-new-dtags,-rpath,${HDF5_DIR}/lib"
        export LD_LIBRARY_PATH="${HDF5_DIR}/lib:${MPI_DIR}/lib:${_conda_lib}"

        # h5py >=3.15 requires setuptools >=77 to build, while UW3 pins the
        # runtime environment to setuptools 75. Build isolation satisfies the
        # h5py build requirements without modifying the locked Pixi packages.
        CC="${MPI_DIR}/bin/mpicc" \
        HDF5_MPI="ON" \
        HDF5_VERSION="1.12.2" \
        CFLAGS="-I${HDF5_DIR}/include -include ${HDF5_DIR}/include/hdf5.h -include ${HDF5_DIR}/include/H5FDmpio.h" \
        python3 -m pip install --no-binary=h5py --no-cache-dir \
        --force-reinstall --no-deps "h5py>=3.12,<4"
    )
    local _rc=$?

    for _f in "${_hidden[@]}"; do
        mv "${_f}.h5build" "${_f}"
    done
    [ "${#_hidden[@]}" -gt 0 ] && echo "  Restored ${#_hidden[@]} pixi HDF5 lib(s)"

    [ "${_rc}" -ne 0 ] && { echo "ERROR: h5py build failed (rc=${_rc})"; return "${_rc}"; }
    python3 -c "import h5py; assert h5py.get_config().mpi, 'h5py lacks MPI support'" || return 1
    echo "==> h5py installed"
}

install_uw3() {
    echo "==> Installing Underworld3..."
    (
        cd "${UW3_PATH}" || exit 1
        python3 -m pip install --no-build-isolation --no-deps -e .
    ) || return 1
    echo "==> Underworld3 installed"
}

check_petsc_exists() {
    EXPECTED_PETSC_LIB="${PETSC_DIR}/${PETSC_ARCH}/lib" python3 -c "
import os
from pathlib import Path
import petsc4py
from petsc4py import PETSc
expected = Path(os.environ['EXPECTED_PETSC_LIB']).resolve()
actual = Path(petsc4py.__file__).resolve()
assert expected in actual.parents, f'wrong petsc4py: {actual}'
" 2>/dev/null
}

verify_install() {
    echo "==> Verifying installation..."
    EXPECTED_UW3_PATH="${UW3_PATH}" python3 -c "
import os
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import h5py
import underworld3 as uw

assert h5py.get_config().mpi, 'h5py lacks MPI support'
expected_uw3 = Path(os.environ['EXPECTED_UW3_PATH']).resolve()
actual_uw3 = Path(uw.__file__).resolve()
assert expected_uw3 in actual_uw3.parents, f'wrong underworld3 source: {actual_uw3}'

cpp_libs = sorted({
    line.split()[-1]
    for line in Path('/proc/self/maps').read_text().splitlines()
    if 'libstdc++.so' in line
})
assert any(os.environ['CONDA_PREFIX'] in path for path in cpp_libs), cpp_libs

print(f'NumPy OK      - version: {np.__version__}')
print(f'mpi4py OK     - MPI version: {MPI.Get_version()}')
print(f'petsc4py OK   - PETSc version: {PETSc.Sys.getVersion()}')
print(f'h5py OK       - HDF5 version: {h5py.version.hdf5_version}, MPI: enabled')
print(f'underworld3 OK - version: {uw.__version__}')
print(f'C++ runtime   - {cpp_libs}')
" || return 1
    echo ""
    echo "==> Single-process MPI import check:"
    python3 -c "from mpi4py import MPI; print(f'mpi4py MPI import OK (rank 0 of 1)')" || return 1
    echo "==> All checks passed"
    echo ""
    echo "    NOTE: Multi-rank MPI tests must be run from a compute node (PBS job)."
    echo "    Example: mpirun -n 4 python3 -c \"from mpi4py import MPI; print(MPI.COMM_WORLD.rank)\""
}

run_install() {
    setup_pixi || return 1
    clone_uw3 || return 1
    install_pixi_env || return 1
    install_mpi4py || return 1

    if ! check_petsc_exists; then
        archive_incompatible_petsc || return 1
        install_petsc || return 1
    else
        echo "==> Compatible PETSc already installed, skipping"
    fi

    install_h5py || return 1
    install_uw3 || return 1
    verify_install || return 1
}

# ============================================================
# ENTRY POINT
# ============================================================

load_env
_load_rc=$?
if [ "${_load_rc}" -ne 0 ]; then
    echo "ERROR: Gadi environment activation failed"
    return "${_load_rc}"
fi

if [ "${1:-}" = "install" ]; then
    echo ""
    echo "Starting user installation..."
    echo "  BASE_PATH:      ${BASE_PATH}"
    echo "  UW3_PATH:       ${UW3_PATH}"
    echo "  UW3_BRANCH:     ${UW3_BRANCH}"
    echo "  PIXI_HOME:      ${PIXI_HOME}"
    echo "  PIXI_CACHE_DIR: ${PIXI_CACHE_DIR}"
    echo "  PIXI_ENV_ROOT:  ${PIXI_ENV_ROOT}"
    echo ""
    run_install
    _install_rc=$?
    if [ "${_install_rc}" -ne 0 ]; then
        echo "ERROR: Underworld3 installation failed"
        return "${_install_rc}"
    fi
    echo ""
    echo "=========================================="
    echo "User installation complete!"
    echo "To activate: source $(realpath "${BASH_SOURCE[0]}")"
    echo "=========================================="
fi
