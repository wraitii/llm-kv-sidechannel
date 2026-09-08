#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ $# -ge 1 && $# -le 2 ]] || die "usage: $0 INSTANCE_ID [GIT_COMMIT]"
instance_id="$1"
commit="${2:-$(git -C "$REPO_DIR" rev-parse HEAD)}"
require_integer INSTANCE_ID "$instance_id"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "GIT_COMMIT must be a full 40-character SHA"
repo_url="${VAST_REPO_URL:-https://github.com/wraitii/llm-kv-sidechannel.git}"
remote_repo="${VAST_REMOTE_REPO:-/workspace/llm-kv-sidechannel}"

git ls-remote "$repo_url" | awk -v commit="$commit" \
  '$1 == commit { found=1 } END { exit !found }' || \
  die "commit $commit is not advertised by the public remote; push it first"

read -r ssh_host ssh_port <<<"$(vast_ssh_parts "$instance_id")"
printf 'Bootstrapping instance %s at commit %s\n' "$instance_id" "$commit"
ssh -p "$ssh_port" "$ssh_host" bash -s -- "$repo_url" "$remote_repo" "$commit" <<'REMOTE'
set -euo pipefail
repo_url="$1"
repo_dir="$2"
commit="$3"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git
if [[ ! -d "$repo_dir/.git" ]]; then
  git clone "$repo_url" "$repo_dir"
fi
git -C "$repo_dir" fetch origin "$commit"
git -C "$repo_dir" checkout --detach "$commit"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$repo_dir/torch"
uv sync --locked --extra dev --extra data

git rev-parse HEAD
git status --short
uv run --locked python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
REMOTE

printf 'Bootstrap complete. Run the checklist hardware and soundness checks next.\n'
