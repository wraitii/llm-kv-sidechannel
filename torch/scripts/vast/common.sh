#!/usr/bin/env bash

set -euo pipefail

VAST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_DIR="$(cd "$VAST_SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$TORCH_DIR/.." && pwd)"
VAST_ARTIFACTS_DIR="${VAST_ARTIFACTS_DIR:-$REPO_DIR/artifacts/vast}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_integer() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a nonnegative integer: $2"
}

load_vast_env() {
  local env_file="${VAST_ENV_FILE:-$REPO_DIR/.env}"
  [[ -f "$env_file" ]] || die "missing $env_file (expected VAST_API_KEY)"
  set -a
  # This is a user-owned secrets file. Keep commands and substitutions out of it.
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  [[ -n "${VAST_API_KEY:-}" ]] || die "VAST_API_KEY is not set in $env_file"
}

vast() {
  require_command vastai
  load_vast_env
  vastai "$@"
}

vast_ssh_parts() {
  local instance_id="$1"
  local ssh_url
  ssh_url="$(vast ssh-url "$instance_id")"
  python3 -c 'import re, sys, urllib.parse
value = sys.stdin.read().strip()
if value.startswith("ssh://"):
    parsed = urllib.parse.urlparse(value)
    user = parsed.username or "root"
    print(f"{user}@{parsed.hostname} {parsed.port or 22}")
else:
    match = re.search(r"ssh\s+(?:-p\s+(\d+)\s+)?([^ ]+)", value)
    if not match:
        raise SystemExit("could not parse Vast SSH URL")
    print(f"{match.group(2)} {match.group(1) or 22}")
' <<<"$ssh_url"
}

confirm_exact() {
  local expected="$1"
  local prompt="$2"
  local answer
  [[ -t 0 ]] || die "confirmation requires an interactive terminal"
  printf '%s\nType %q to continue: ' "$prompt" "$expected" >&2
  IFS= read -r answer
  [[ "$answer" == "$expected" ]] || die "confirmation did not match"
}
