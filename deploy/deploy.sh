#!/bin/bash
set -e

echo "=== 1. Loading Configuration ==="
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Please create one with your variables."
    exit 1
fi

# Automatically export all variables loaded from the .env file
set -a
source .env
set +a

echo "=== 2. Updating Azure Container Apps CLI extension ==="
az extension add --name containerapp --upgrade

echo "=== 3. Creating the Container Apps Environment ($ENV_NAME) ==="
# This takes 1-2 minutes. It automatically sets up the Log Analytics workspace.
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "=== 4. Deploying the Container App from YAML ==="
if [ ! -f "$YAML_FILE" ]; then
    echo "ERROR: $YAML_FILE not found in the current directory."
    exit 1
fi

# The CLI will automatically configure ACR pull permissions using your active Azure login
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --registry-server "$ACR_NAME.azurecr.io" \
  --yaml "$YAML_FILE"

echo "============================================================"
echo "Deployment Successful!"
echo "Fetching your live application URL..."
echo "============================================================"

URL=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "https://$URL"