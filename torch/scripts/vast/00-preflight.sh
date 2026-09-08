#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

require_command git
require_command jq
require_command rsync
require_command ssh
require_command ssh-keygen
require_command uv
require_command vastai
[[ -f "$HOME/.ssh/id_ed25519.pub" ]] || die "missing ~/.ssh/id_ed25519.pub"

git -C "$REPO_DIR" check-ignore -q .env || die ".env is not ignored by Git"
if git -C "$REPO_DIR" ls-files --error-unmatch .env >/dev/null 2>&1; then
  die ".env is tracked by Git"
fi

remote_url="$(git -C "$REPO_DIR" remote get-url origin)"
public_url="${VAST_REPO_URL:-https://github.com/wraitii/llm-kv-sidechannel.git}"
commit="$(git -C "$REPO_DIR" rev-parse HEAD)"
dirty="$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)"
[[ -z "$dirty" ]] || printf 'warning: tracked worktree changes are present\n' >&2

user_status="$(vast show user --raw)"
ssh_keys="$(vast show ssh-keys --raw)"

printf 'vast_cli=%s\n' "$(vastai --version)"
printf 'vast_authenticated=%s\n' "$(jq -r '((.id // .user_id // .username) != null)' <<<"$user_status")"
printf 'vast_has_credit=%s\n' "$(jq -r '((.credit // .balance // 0) > 0)' <<<"$user_status")"
printf 'vast_ssh_key_count=%s\n' "$(jq -r 'length' <<<"$ssh_keys")"
printf 'vast_has_local_ssh_key=%s\n' \
  "$(jq -r --arg expected "$(< "$HOME/.ssh/id_ed25519.pub")" \
    'map(.public_key == $expected) | any' <<<"$ssh_keys")"
printf 'git_commit=%s\n' "$commit"
printf 'git_remote=%s\n' "$remote_url"
printf 'artifact_dir=%s\n' "$VAST_ARTIFACTS_DIR"
printf 'local_ssh_fingerprint=%s\n' "$(ssh-keygen -lf "$HOME/.ssh/id_ed25519.pub" | awk '{print $2}')"

remote_refs="$(git ls-remote "$public_url")"
remote_has_commit="$(awk -v commit="$commit" '$1 == commit { found=1 } END { print found ? "true" : "false" }' <<<"$remote_refs")"
printf 'public_clone_check=ok\n'
printf 'remote_has_local_commit=%s\n' "$remote_has_commit"
[[ "$remote_has_commit" == true ]] || \
  printf 'warning: push commit %s before bootstrapping an instance\n' "$commit" >&2
