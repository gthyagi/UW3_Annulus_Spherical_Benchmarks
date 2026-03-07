#!/usr/bin/env bash
# Activate Underworld3 pixi environment.
# - Execute: opens a pixi subshell
#     ./activate_uw_pixi_env.sh [env_name]
# - Source: activates in current shell
#     source ./activate_uw_pixi_env.sh [env_name]

set -euo pipefail

UW_ROOT="/Users/tgol0006/uw_folder/uw3_git_gthyagi_latest_bdint/underworld3"
HOME_DIR="/Users/tgol0006"
CONFIG_FILE="${UW_ROOT}/.pixi-env"
DEFAULT_ENV="runtime"

if command -v pixi >/dev/null 2>&1; then
    PIXI_BIN="pixi"
elif [[ -x "${HOME}/.pixi/bin/pixi" ]]; then
    PIXI_BIN="${HOME}/.pixi/bin/pixi"
else
    echo "pixi was not found in PATH or ${HOME}/.pixi/bin/pixi"
    return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${UW_ROOT}" ]]; then
    echo "Underworld3 root not found: ${UW_ROOT}"
    return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${HOME_DIR}" ]]; then
    echo "Home directory not found: ${HOME_DIR}"
    return 1 2>/dev/null || exit 1
fi

ENV_NAME="${1:-}"
if [[ -z "${ENV_NAME}" ]]; then
    if [[ -f "${CONFIG_FILE}" ]]; then
        ENV_NAME="$(tr -d '[:space:]' < "${CONFIG_FILE}")"
    else
        ENV_NAME="${DEFAULT_ENV}"
    fi
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # On macOS with MPICH+OFI, explicitly prefer tcp provider for stable local MPI performance.
    export FI_PROVIDER="${FI_PROVIDER:-tcp}"
    cd "${HOME_DIR}"
    echo "Starting pixi shell '${ENV_NAME}' from ${UW_ROOT}"
    exec "${PIXI_BIN}" shell -m "${UW_ROOT}" -e "${ENV_NAME}"
else
    # On macOS with MPICH+OFI, explicitly prefer tcp provider for stable local MPI performance.
    export FI_PROVIDER="${FI_PROVIDER:-tcp}"
    eval "$(
        cd "${UW_ROOT}"
        "${PIXI_BIN}" shell-hook -e "${ENV_NAME}"
    )"
    cd "${HOME_DIR}"
    echo "Activated pixi environment '${ENV_NAME}' from ${UW_ROOT} and moved to ${HOME_DIR}"
fi
