#!/usr/bin/env bash
# FLaas — One-time shell hook setup.
#
# Run once after cloning:
#   bash federated_learning/.scripts/setup_autoactivate.sh
#
# Effect: every new terminal opened inside this repository will
# automatically source activate_env.sh (creates the venv on first run,
# then just activates it on subsequent runs).

HOOK_TAG="# >>> FLaas auto-activate <<<"

# Detect shell config file
if [ -n "$ZSH_VERSION" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
    HOOK_BODY='
'"$HOOK_TAG"'
_flaas_activate() {
    local _root; _root="$(git rev-parse --show-toplevel 2>/dev/null)"
    [[ -f "$_root/federated_learning/.scripts/activate_env.sh" ]] && \
        source "$_root/federated_learning/.scripts/activate_env.sh"
}
autoload -Uz add-zsh-hook
add-zsh-hook chpwd _flaas_activate   # fires on every cd
_flaas_activate                       # fires on new terminal
'"$HOOK_TAG"
else
    SHELL_RC="$HOME/.bashrc"
    HOOK_BODY='
'"$HOOK_TAG"'
_flaas_activate() {
    local _root; _root="$(git rev-parse --show-toplevel 2>/dev/null)"
    [[ -f "$_root/federated_learning/.scripts/activate_env.sh" ]] && \
        source "$_root/federated_learning/.scripts/activate_env.sh"
}
export PROMPT_COMMAND="_flaas_activate; $PROMPT_COMMAND"
_flaas_activate                       # fires on new terminal
'"$HOOK_TAG"
fi

# Idempotent: skip if hook already present
if grep -qF "$HOOK_TAG" "$SHELL_RC" 2>/dev/null; then
    echo "[FLaas] Hook already installed in $SHELL_RC — nothing to do."
    exit 0
fi

printf '%s\n' "$HOOK_BODY" >> "$SHELL_RC"
echo "[FLaas] Hook added to $SHELL_RC"
echo "[FLaas] Reload now:  source $SHELL_RC"
