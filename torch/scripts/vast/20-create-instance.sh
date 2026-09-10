#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ $# -ge 1 && $# -le 3 ]] || die "usage: $0 OFFER_ID [LABEL] [DISK_GB]"
offer_id="$1"
label="${2:-llmpr-qwen5090}"
disk_gb="${3:-${VAST_STORAGE_GB:-100}}"
template_hash="${VAST_TEMPLATE_HASH:-$VAST_DEFAULT_TEMPLATE_HASH}"
max_hourly="${VAST_MAX_HOURLY:-0.80}"
require_integer OFFER_ID "$offer_id"
require_integer DISK_GB "$disk_gb"

offer="$(vast search offers "id=$offer_id rentable=true" --type on-demand \
  --storage "$disk_gb" --limit 1 --raw)"
# Some Vast API deployments currently return no rows when the documented `id`
# search field is used, even while the same offer appears in an unfiltered
# search. Fall back to selecting the requested ID locally from live offers.
if [[ "$(jq 'length' <<<"$offer")" == 0 ]]; then
  offer="$(vast search offers "rentable=true" --type on-demand \
    --storage "$disk_gb" --limit 10000 --raw | \
    jq --argjson offer_id "$offer_id" '[.[] | select(.id == $offer_id)]')"
fi
[[ "$(jq 'length' <<<"$offer")" == 1 ]] || die "offer $offer_id is no longer rentable"
price="$(jq -r '.[0].dph_total' <<<"$offer")"
template="$(vast search templates "hash_id=$template_hash" --raw)"
[[ "$(jq 'length' <<<"$template")" -ge 1 ]] || \
  die "template $template_hash was not found"
template_image="$(jq -r '.[0] | .image + ":" + .tag' <<<"$template")"
jq -r '.[0] | {
  offer_id: .id, machine_id, gpu_name,
  vram_gb: ((.gpu_ram / 1000) | floor), reliability,
  cuda_max: .cuda_max_good, driver: .driver_version,
  total_per_hour: .dph_total, storage_gb: $storage,
  disk_available_gb: .disk_space, location: .geolocation
}' --argjson storage "$disk_gb" <<<"$offer"

awk -v price="$price" -v maximum="$max_hourly" \
  'BEGIN { exit !(price <= maximum) }' || die "offer price $price exceeds guardrail $max_hourly"
confirm_exact "rent $offer_id" \
  "This creates a billable on-demand instance using template $template_hash ($template_image) with ${disk_gb} GB disk."

result="$(vast create instance "$offer_id" --template_hash "$template_hash" \
  --disk "$disk_gb" --label "$label" --cancel-unavail --raw)"
printf '%s\n' "$result"
instance_id="$(python3 -c 'import ast, json, sys
text = sys.stdin.read().strip()
try:
    value = json.loads(text)
except json.JSONDecodeError:
    value = ast.literal_eval(text)
print(value.get("new_contract", ""))
' <<<"$result")"
require_integer INSTANCE_ID "$instance_id"
mkdir -p "$VAST_ARTIFACTS_DIR/instances/$instance_id"
jq -n --argjson instance_id "$instance_id" --argjson offer_id "$offer_id" \
  --arg template_hash "$template_hash" --arg template_image "$template_image" \
  --arg label "$label" --argjson disk_gb "$disk_gb" \
  --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{instance_id: $instance_id, offer_id: $offer_id,
    template_hash: $template_hash, template_image: $template_image,
    label: $label, disk_gb: $disk_gb, created_at: $created_at}' \
  >"$VAST_ARTIFACTS_DIR/instances/$instance_id/instance.json"
printf 'Instance %s created. Inspect it with 30-show-instance.sh %s\n' \
  "$instance_id" "$instance_id"
