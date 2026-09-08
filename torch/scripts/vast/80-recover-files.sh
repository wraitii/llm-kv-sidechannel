#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ $# -eq 4 ]] || die \
  "usage: $0 INSTANCE_ID RUN_ID CONFIG_RELATIVE_PATH DATASET_RELATIVE_PATH"
instance_id="$1"
run_id="$2"
config_path="$3"
dataset_path="$4"
require_integer INSTANCE_ID "$instance_id"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe RUN_ID: $run_id"
[[ "$config_path" != /* && "$config_path" != *..* ]] || die "config path must be safe and relative"
[[ "$dataset_path" != /* && "$dataset_path" != *..* ]] || die "dataset path must be safe and relative"

remote_repo="${VAST_REMOTE_REPO:-/workspace/llm-kv-sidechannel}"
destination="$VAST_ARTIFACTS_DIR/instances/$instance_id/$run_id"
read -r ssh_host ssh_port <<<"$(vast_ssh_parts "$instance_id")"

mkdir -p "$destination/outputs"
rsync -avP -e "ssh -p $ssh_port" \
  "$ssh_host:$remote_repo/torch/outputs/$run_id/" "$destination/outputs/"
rsync -avP -e "ssh -p $ssh_port" \
  "$ssh_host:$remote_repo/torch/$config_path" "$destination/"
rsync -avP -e "ssh -p $ssh_port" \
  "$ssh_host:$remote_repo/torch/$dataset_path/manifest.json" "$destination/"

# A second checksum pass transfers any mismatch rather than merely trusting size
# and modification time. Only mark recovery complete after all three succeed.
rsync -avc -e "ssh -p $ssh_port" \
  "$ssh_host:$remote_repo/torch/outputs/$run_id/" "$destination/outputs/"
printf 'instance_id=%s\nrun_id=%s\nrecovered_at=%s\n' \
  "$instance_id" "$run_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$destination/RECOVERY_COMPLETE"
printf 'Recovered and checksum-verified: %s\n' "$destination"
