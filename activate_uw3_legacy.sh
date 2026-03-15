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
  rehash 2>/dev/null || hash -r 2>/dev/null || true
  cd "${TARGET_HOME}"
  return 0
fi

# Exec mode: open a fresh interactive zsh and apply the legacy UW3 env after
# the user's normal shell config so the intended mpirun/python pair wins.
TMP_ZDOTDIR="$(mktemp -d "${TMPDIR:-/tmp}/uw3-legacy-zdotdir.XXXXXX")"

cat > "${TMP_ZDOTDIR}/.zshenv" <<EOF
if [[ -f "/Users/tgol0006/.zshenv" ]]; then
  source "/Users/tgol0006/.zshenv"
fi
EOF

cat > "${TMP_ZDOTDIR}/.zshrc" <<EOF
if [[ -f "/Users/tgol0006/.zshrc" ]]; then
  source "/Users/tgol0006/.zshrc"
fi
set +u
source "${ENV_SCRIPT}"
rehash 2>/dev/null || hash -r 2>/dev/null || true
cd "${TARGET_HOME}"
TRAPEXIT() {
  rm -rf "${TMP_ZDOTDIR}"
}
EOF

export ZDOTDIR="${TMP_ZDOTDIR}"
exec "${SHELL_BIN}" -i
