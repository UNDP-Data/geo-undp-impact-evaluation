#!/bin/bash
set -e

echo "=== 1. Loading Configuration ==="
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found."
    exit 1
fi

set -a
source .env
set +a

REPO="undp-data/geo-undp-impact-evaluation"

echo "=== 2. Finding latest Ymd.N tag in ACR ($ACR_NAME) ==="
LATEST_VERSION_TAG=$(az acr repository show-tags \
  --name "$ACR_NAME" \
  --repository "$REPO" \
  --orderby time_desc \
  --query "[?@!='latest'] |" \
  --output tsv)

IMAGE="${ACR_NAME}.azurecr.io/${REPO}:${LATEST_VERSION_TAG}"
echo "Deploying versioned image: $IMAGE"

echo "=== 3. Updating Container App ($APP_NAME) ==="
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$IMAGE"

echo "============================================================"
echo "Successfully updated $APP_NAME to version $LATEST_VERSION_TAG!"
echo "============================================================"
