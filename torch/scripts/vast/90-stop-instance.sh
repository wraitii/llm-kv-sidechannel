#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ $# -eq 1 ]] || die "usage: $0 INSTANCE_ID"
instance_id="$1"
require_integer INSTANCE_ID "$instance_id"
vast show instance "$instance_id" --raw | jq '{id, label, actual_status, dph_total}'
confirm_exact "stop $instance_id" \
  "Stopping ends compute billing but preserves the disk and its storage charges."
vast stop instance "$instance_id" --raw
