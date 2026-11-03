#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

echo "$SCRIPT_DIR"
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
  echo "Error: Missing ${SCRIPT_DIR}/.env file." >&2
  exit 1
fi

# Load static variables from .env
set -a
source "${SCRIPT_DIR}/.env"
set +a

REPO="undp-data/geo-undp-impact-evaluation"

echo "=== 2. Finding latest Ymd.N tag in ACR ($ACR_NAME) ==="
LATEST_VERSION_TAG=$(az acr repository show-tags \
  --name "$ACR_NAME" \
  --repository "$REPO" \
  --orderby time_desc \
  --output tsv | grep -v '^latest$' | head -n 1 || true)

if [ -z "$LATEST_VERSION_TAG" ]; then
  echo "Warning: No Ymd.N tag found yet, falling back to 'latest'"
  LATEST_VERSION_TAG="latest"
fi

IMAGE="${ACR_NAME}.azurecr.io/${REPO}:${LATEST_VERSION_TAG}"
echo "Deploying versioned image: $IMAGE"

echo "=== 3. Updating Container App ($APP_NAME) ==="
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$IMAGE"

echo "============================================================"
echo "Successfully updated $APP_NAME to $IMAGE!"
echo "============================================================"
