#!/usr/bin/env bash
set -e

# Legacy UW3 env setup script.
ENV_SCRIPT="/Users/tgol0006/uw3_venv_cp_nci_21125.sh"
TARGET_HOME="/Users/tgol0006"
SHELL_BIN="/bin/zsh"

# Detect if script is sourced (bash/zsh).
is_sourced=0
if [[ -n "${ZSH_VERSION-}" ]]; then
  [[ "${ZSH_EVAL_CONTEXT-}" == *:file ]] && is_sourced=1
elif [[ "${BASH_SOURCE[0]-$0}" != "$0" ]]; then
  is_sourced=1
fi

# Source mode: activate in current shell.
if [[ "${is_sourced}" -eq 1 ]]; then
  set +u
  source "${ENV_SCRIPT}"
  cd "${TARGET_HOME}"
  return 0
fi

# Exec mode: open a new shell with env activated.
exec "${SHELL_BIN}" -ic "set +u; source '${ENV_SCRIPT}'; cd '${TARGET_HOME}'; exec '${SHELL_BIN}' -i"
