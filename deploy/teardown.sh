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

echo "============================================================"
echo "⚠️  WARNING: You are about to delete the following resources:"
echo " - Container App: $APP_NAME"
echo " - Container Environment: $ENV_NAME"
echo " (Your Resource Group and ACR will NOT be deleted)"
echo "============================================================"
read -p "Are you sure you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Teardown aborted."
    exit 1
fi

echo "=== 2. Deleting the Container App ($APP_NAME) ==="
# --yes bypasses the az cli confirmation prompt since we already asked above
az containerapp delete \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --yes

echo "=== 3. Deleting the Container Apps Environment ($ENV_NAME) ==="
az containerapp env delete \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --yes

echo "============================================================"
echo "Teardown Successful!"
echo "Your app and environment have been completely removed."
echo "============================================================"