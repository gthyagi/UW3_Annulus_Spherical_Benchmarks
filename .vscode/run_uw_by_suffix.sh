#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python_file> [args...]"
    exit 2
fi

FILE="$1"
shift || true

if [[ ! -f "$FILE" ]]; then
    echo "File not found: $FILE"
    exit 2
fi

BASE_NAME="$(basename "$FILE")"

run_latest() {
    local uw_root="/Users/tgol0006/uw_folder/uw3_git_gthyagi_openmpi/underworld3"
    local env_name="runtime"
    local pixi_bin

    if command -v pixi >/dev/null 2>&1; then
        pixi_bin="pixi"
    elif [[ -x "${HOME}/.pixi/bin/pixi" ]]; then
        pixi_bin="${HOME}/.pixi/bin/pixi"
    else
        echo "pixi not found in PATH or ${HOME}/.pixi/bin/pixi"
        exit 1
    fi

    if [[ -f "${uw_root}/.pixi-env" ]]; then
        env_name="$(tr -d '[:space:]' < "${uw_root}/.pixi-env")"
        [[ -n "${env_name}" ]] || env_name="runtime"
    fi

    # Keep local MPI behavior stable on macOS unless user overrides it.
    export FI_PROVIDER="${FI_PROVIDER:-tcp}"

    exec "${pixi_bin}" run -m "${uw_root}" -e "${env_name}" python "$FILE" "$@"
}

run_legacy() {
    local env_script="/Users/tgol0006/uw3_venv_cp_nci_21125.sh"
    local quoted_env_script
    local quoted_file
    local quoted_args=""
    local arg

    if [[ ! -f "${env_script}" ]]; then
        echo "Legacy env script not found: ${env_script}"
        exit 1
    fi

    quoted_env_script="$(printf '%q' "${env_script}")"
    quoted_file="$(printf '%q' "$FILE")"

    for arg in "$@"; do
        quoted_args+=" $(printf '%q' "$arg")"
    done

    exec /bin/bash -lc "set +u; source ${quoted_env_script}; python ${quoted_file}${quoted_args}"
}

case "${BASE_NAME}" in
    *_latest.py)
        run_latest "$@"
        ;;
    *_legacy.py)
        run_legacy "$@"
        ;;
    *)
        echo "Expected file ending with _latest.py or _legacy.py"
        echo "Got: ${BASE_NAME}"
        exit 2
        ;;
esac
