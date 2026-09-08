#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ $# -eq 1 ]] || die "usage: $0 INSTANCE_ID"
instance_id="$1"
require_integer INSTANCE_ID "$instance_id"
instance_dir="$VAST_ARTIFACTS_DIR/instances/$instance_id"
marker="$(find "$instance_dir" -mindepth 2 -maxdepth 2 \
  -name RECOVERY_COMPLETE -type f -print -quit 2>/dev/null || true)"
[[ -n "$marker" ]] || die \
  "no local RECOVERY_COMPLETE marker for instance $instance_id; recover files first"

vast show instance "$instance_id" --raw | jq '{id, label, actual_status, dph_total, disk_space}'
printf 'Recovery marker: %s\n' "$marker"
confirm_exact "DESTROY $instance_id" \
  "Destruction permanently deletes the instance disk. This cannot be undone."
vast destroy instance "$instance_id" -y --raw
