#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ $# -eq 1 ]] || die "usage: $0 INSTANCE_ID"
instance_id="$1"
require_integer INSTANCE_ID "$instance_id"

vast show instance "$instance_id" --raw | jq '{
  id, label, actual_status, intended_status, gpu_name, gpu_ram,
  dph_total, disk_space, image_uuid, docker_image, start_date,
  ssh_host, ssh_port, direct_port_start, direct_port_end
}'
printf 'SSH: '
vast ssh-url "$instance_id"
