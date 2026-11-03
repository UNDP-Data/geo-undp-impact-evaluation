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

#echo "=== 2. Updating Azure Container Apps CLI extension ==="
#az extension add --name containerapp --upgrade
#
#echo "=== 3. Creating the Container Apps Environment ($ENV_NAME) ==="
#az containerapp env create \
#  --name "$ENV_NAME" \
#  --resource-group "$RESOURCE_GROUP" \
#  --location "$LOCATION"

echo "=== 4. Deploying the Container App from YAML ==="


az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "${ACR_NAME}.azurecr.io/undp-data/geo-undp-impact-evaluation:latest" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --ingress external \
  --target-port 8501 \
  --cpu 4.0 \
  --memory 8.0 \
  --min-replicas 1 \
  --max-replicas 1

echo "============================================================"
echo "Deployment Successful!"
echo "============================================================"

URL=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "Live URL: https://$URL"
echo ""
echo "⚠️ ACTION REQUIRED: Earth Engine credentials were not deployed."
echo "Please go to the Azure Portal -> $APP_NAME -> Containers -> Environment variables,"
echo "and manually add GEE and ACLED  env vars"