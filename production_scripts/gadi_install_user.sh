#!/usr/bin/env bash
#
# Underworld3 per-user install / activation script for NCI Gadi (pixi-based)
#
# Usage:
#   source gadi_install_user.sh
#   source gadi_install_user.sh install
#
# This script must be sourced, not executed.

usage="
Usage:
  source gadi_install_user.sh
      Activate an existing install

  source gadi_install_user.sh install
      Install Underworld3 and activate the environment

Options:
  -h    Show this help
"

# -------------------------
# helpers
# -------------------------

is_sourced() { [[ "${BASH_SOURCE[0]}" != "${0}" ]]; }

finish() {
    local rc="${1:-0}"
    if is_sourced; then
        return "${rc}"
    else
        exit "${rc}"
    fi
}

die() {
    echo "ERROR: $*" >&2
    finish 1
}

note() {
    echo "==> $*"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# Must be sourced
is_sourced || {
    echo "ERROR: This script must be sourced, not executed." >&2
    echo "Use: source ${0##*/} [install]" >&2
    exit 1
}

# -------------------------
# args
# -------------------------

INSTALL_MODE=0

while getopts ":h" opt; do
    case "$opt" in
        h) echo "$usage"; finish 0 ;;
        \?) echo "Error: unknown option -$OPTARG" >&2; echo "$usage" >&2; finish 1 ;;
    esac
done
shift $((OPTIND - 1))

case "${1:-}" in
    install) INSTALL_MODE=1 ;;
    "") ;;
    *) echo "Error: unknown argument '${1}'" >&2; echo "$usage" >&2; finish 1 ;;
esac

# -------------------------
# config
# -------------------------

export UW3_BRANCH="${UW3_BRANCH:-development}"
export UW3_REPO="${UW3_REPO:-https://github.com/gthyagi/underworld3.git}"
export INSTALL_NAME="${INSTALL_NAME:-underworld3}"

export BASE_PATH="/g/data/m18/${USER}/uw3-pixi"
export PIXI_HOME="${HOME}/.pixi"
export UW3_PATH="${BASE_PATH}/${INSTALL_NAME}"

export PIXI_MANIFEST="${UW3_PATH}/pixi.toml"
export PETSC_DIR="${UW3_PATH}/petsc-custom/petsc"
export PETSC_ARCH="${PETSC_ARCH:-petsc-4-uw-openmpi}"

export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_io=ompio
export PYTHONNOUSERSITE=1

ORIG_PWD="${PWD}"

# -------------------------
# environment
# -------------------------

load_modules() {
    module purge || die "module purge failed"
    module load openmpi/4.1.7 hdf5/1.12.2p gmsh/4.13.1 cmake/3.31.6 || die "module load failed"

    export MPI_DIR
    MPI_DIR="$(dirname "$(dirname "$(which mpicc)")")"
    export PATH="${PIXI_HOME}/bin:${PATH}"
}

activate_existing_env() {
    if command -v pixi >/dev/null 2>&1 && [[ -f "${PIXI_MANIFEST}" ]]; then
        eval "$(pixi shell-hook -e gadi --manifest-path "${PIXI_MANIFEST}")" \
            || die "failed to activate pixi env"
        export LD_LIBRARY_PATH="${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}"
    fi

    if [[ -d "${PETSC_DIR}/${PETSC_ARCH}" ]]; then
        export PYTHONPATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PYTHONPATH:-}"
    fi
}

load_env() {
    load_modules
    activate_existing_env

    note "Environment ready"
    echo "    MPI_DIR:    ${MPI_DIR}"
    echo "    HDF5_DIR:   ${HDF5_DIR}"
    echo "    UW3_PATH:   ${UW3_PATH}"
    echo "    PETSC_DIR:  ${PETSC_DIR}"
    echo "    PETSC_ARCH: ${PETSC_ARCH}"
}

# -------------------------
# install steps
# -------------------------

setup_pixi() {
    if command -v pixi >/dev/null 2>&1; then
        note "pixi already installed: $(pixi --version)"
        return 0
    fi

    require_cmd curl
    note "Installing pixi ..."
    curl -fsSL https://pixi.sh/install.sh | bash || die "pixi install failed"
    export PATH="${PIXI_HOME}/bin:${PATH}"
    command -v pixi >/dev/null 2>&1 || die "pixi installed but not found in PATH"
    note "pixi installed: $(pixi --version)"
}

ensure_uw3_checkout() {
    mkdir -p "${BASE_PATH}" || die "cannot create ${BASE_PATH}"
    require_cmd git

    if [[ -d "${UW3_PATH}/.git" && -f "${PIXI_MANIFEST}" ]]; then
        note "Valid UW3 checkout found at ${UW3_PATH}"
        return 0
    fi

    if [[ -e "${UW3_PATH}" && ! -d "${UW3_PATH}/.git" ]]; then
        die "${UW3_PATH} exists but is not a valid git checkout"
    fi

    if [[ -d "${UW3_PATH}/.git" && ! -f "${PIXI_MANIFEST}" ]]; then
        die "UW3 checkout exists but pixi.toml is missing: ${UW3_PATH}"
    fi

    note "Cloning Underworld3 (${UW3_BRANCH}) ..."
    git clone --branch "${UW3_BRANCH}" --depth 1 "${UW3_REPO}" "${UW3_PATH}" \
        || die "git clone failed"

    [[ -f "${PIXI_MANIFEST}" ]] || die "clone completed but ${PIXI_MANIFEST} not found"
}

install_pixi_env() {
    [[ -f "${PIXI_MANIFEST}" ]] || die "Missing pixi manifest: ${PIXI_MANIFEST}"

    note "Installing pixi gadi environment ..."
    pixi install -e gadi --manifest-path "${PIXI_MANIFEST}" || die "pixi install failed"
    eval "$(pixi shell-hook -e gadi --manifest-path "${PIXI_MANIFEST}")" \
        || die "pixi activation failed"

    export LD_LIBRARY_PATH="${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}"

    command -v python >/dev/null 2>&1 || die "python not found after pixi activation"
    python -m pip --version >/dev/null 2>&1 || die "pip not available in pixi environment"

    note "pixi gadi environment ready"
}

install_mpi4py() {
    note "Building mpi4py from source ..."
    python -m pip install --no-binary :all: --no-cache-dir --force-reinstall "mpi4py>=4,<5" \
        || die "mpi4py install failed"
}

install_petsc() {
    local build_script="${UW3_PATH}/petsc-custom/build-petsc.sh"
    [[ -f "${build_script}" ]] || die "Missing PETSc build script: ${build_script}"

    note "Building PETSc ..."
    bash "${build_script}" || die "PETSc build failed"
    export PYTHONPATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PYTHONPATH:-}"
}

install_h5py() {
    note "Building h5py against Gadi HDF5 ..."

    local conda_lib="${UW3_PATH}/.pixi/envs/gadi/lib"
    local hidden=()
    local f

    if [[ -d "${conda_lib}" ]]; then
        for f in "${conda_lib}"/libhdf5*.so*; do
            [[ -f "${f}" && "${f}" != *.h5build ]] || continue
            mv "${f}" "${f}.h5build" || die "failed to hide ${f}"
            hidden+=("${f}")
        done
    fi

    (
        unset LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
        export LDFLAGS="-L${HDF5_DIR}/lib -Wl,--disable-new-dtags,-rpath,${HDF5_DIR}/lib"
        export LD_LIBRARY_PATH="${HDF5_DIR}/lib:${MPI_DIR}/lib"
        CC=mpicc \
        HDF5_MPI=ON \
        HDF5_DIR="${HDF5_DIR}" \
        HDF5_VERSION="1.12.2" \
        CFLAGS="-I${HDF5_DIR}/include -include ${HDF5_DIR}/include/hdf5.h -include ${HDF5_DIR}/include/H5FDmpio.h" \
        python -m pip install --no-binary=h5py --no-cache-dir --force-reinstall --no-deps h5py
    )
    local rc=$?

    for f in "${hidden[@]}"; do
        mv "${f}.h5build" "${f}" || die "failed to restore ${f}"
    done

    [[ ${rc} -eq 0 ]] || die "h5py build failed"
}

install_uw3() {
    note "Installing Underworld3 ..."
    cd "${UW3_PATH}" || die "cannot cd to ${UW3_PATH}"
    python -m pip install --no-build-isolation -e . || {
        cd "${ORIG_PWD}" || true
        die "Underworld3 install failed"
    }
    cd "${ORIG_PWD}" || die "cannot return to original directory"
}

# -------------------------
# checks
# -------------------------

check_petsc() {
    python - <<'PY' >/dev/null 2>&1
from petsc4py import PETSc
print(PETSc.Sys.getVersion())
PY
}

check_uw3() {
    python - <<'PY' >/dev/null 2>&1
import underworld3 as uw
print(uw.__version__)
PY
}

verify_install() {
    note "Verifying installation ..."
    python - <<'PY' || die "verification failed"
from mpi4py import MPI
print(f"mpi4py OK      - MPI version: {MPI.Get_version()}")

from petsc4py import PETSc
print(f"petsc4py OK    - PETSc version: {PETSc.Sys.getVersion()}")

import h5py
print(f"h5py OK        - HDF5 version: {h5py.version.hdf5_version}")

import underworld3 as uw
print(f"underworld3 OK - version: {uw.__version__}")
PY

    echo
    echo "NOTE: Multi-rank MPI tests must be run from a compute node."
    echo 'Example: mpirun -n 4 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"'
    note "All checks passed"
}

# -------------------------
# main
# -------------------------

load_env

if [[ "${INSTALL_MODE}" -eq 1 ]]; then
    echo
    echo "Starting user installation..."
    echo "  BASE_PATH:  ${BASE_PATH}"
    echo "  UW3_PATH:   ${UW3_PATH}"
    echo "  UW3_BRANCH: ${UW3_BRANCH}"
    echo

    setup_pixi
    ensure_uw3_checkout
    install_pixi_env
    install_mpi4py

    if check_petsc; then
        note "PETSc already installed, skipping"
    else
        install_petsc
    fi

    install_h5py

    if check_uw3; then
        note "Underworld3 already installed, skipping"
    else
        install_uw3
    fi

    verify_install

    echo
    echo "=========================================="
    echo "User installation complete"
    echo "To activate later:"
    echo "  source ${BASH_SOURCE[0]}"
    echo "=========================================="
fi