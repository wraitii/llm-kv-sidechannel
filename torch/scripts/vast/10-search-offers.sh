#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

storage_gb="${VAST_STORAGE_GB:-100}"
max_hourly="${VAST_MAX_HOURLY:-0.80}"
min_reliability="${VAST_MIN_RELIABILITY:-0.99}"
template_hash="${VAST_TEMPLATE_HASH:-$VAST_DEFAULT_TEMPLATE_HASH}"
limit="${VAST_SEARCH_LIMIT:-30}"
require_integer VAST_STORAGE_GB "$storage_gb"
require_integer VAST_SEARCH_LIMIT "$limit"

# The selected template requires amd64/arm64 and CUDA >=12.8. This project is
# amd64-only and its locked Torch stack requires a CUDA 13-capable driver, so
# these constraints are deliberately stricter than the template minimum.
template="$(vast search templates "hash_id=$template_hash" --raw)"
[[ "$(jq 'length' <<<"$template")" -ge 1 ]] || \
  die "template $template_hash was not found"
template_image="$(jq -r '.[0] | .image + ":" + .tag' <<<"$template")"
template_filters="$(jq -r '.[0].extra_filters // "{}"' <<<"$template")"
jq -e 'type == "object"' <<<"$template_filters" >/dev/null || \
  die "template $template_hash has invalid compatibility filters"

query="gpu_name=RTX_5090 num_gpus=1 verified=true reliability>=$min_reliability rentable=true direct_port_count>=1 cpu_arch=amd64 cuda_vers>=13.0 disk_space>=$storage_gb cpu_ram>=32 dph_total<=$max_hourly"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$VAST_ARTIFACTS_DIR/searches"
output="$output_dir/rtx5090-$timestamp.json"
filtered="$output_dir/.rtx5090-$timestamp.filtered.json"
mkdir -p "$output_dir"

vast search offers "$query" --type on-demand --storage "$storage_gb" \
  --order 'dph_total' --limit "$limit" --raw >"$output"
# Vast applies some query filters before adding the requested storage price.
# Reapply the all-in hourly cap to the returned values.
jq --argjson maximum "$max_hourly" --argjson filters "$template_filters" '
  def satisfies($offer; $filters):
    all($filters | to_entries[];
      . as $field |
      all($field.value | to_entries[];
        . as $condition |
        ($offer[$field.key]) as $actual |
        if $condition.key == "eq" then $actual == $condition.value
        elif $condition.key == "gte" then $actual >= $condition.value
        elif $condition.key == "lte" then $actual <= $condition.value
        elif $condition.key == "gt" then $actual > $condition.value
        elif $condition.key == "lt" then $actual < $condition.value
        elif $condition.key == "in" then $condition.value | index($actual) != null
        else false
        end));
  [.[] | select(.dph_total <= $maximum) | select(satisfies(.; $filters))]
' "$output" >"$filtered"
mv "$filtered" "$output"

jq -r '
  (["OFFER", "$/HR", "REL", "CUDA", "VRAM_GB", "CPU", "RAM_GB", "DISK_GB", "DISK_MB/S", "DOWN_MB/S", "UP_MB/S", "LOCATION"] | @tsv),
  (.[] | [
    .id, .dph_total, .reliability, .cuda_max_good,
    ((.gpu_ram / 1000) | floor), .cpu_cores_effective,
    ((.cpu_ram / 1000) | floor), (.disk_space | floor),
    (.disk_bw | floor), (.inet_down | floor), (.inet_up | floor), .geolocation
  ] | @tsv)
' "$output" | column -t -s $'\t'

printf '\nTemplate: %s (%s)\n' "$template_hash" "$template_image"
printf 'Template filters: %s\n' "$template_filters"
printf '\nRaw offers: %s\n' "$output"
printf 'No offer was selected or rented. Re-run immediately before creation.\n'
