#!/usr/bin/env bash
# Pull every country built on the Hetzner server (all user workspaces) into this
# repository's shared config/ data/ outputs/ tree, then stage the files the
# dashboard needs so that `git clone` + `git pull` delivers a populated app.
# Rasters (*.tif) and secrets are never included. Run from the repo root.
set -euo pipefail
SERVER=${SERVER:-root@62.238.42.192}
TMP=/tmp/undp_ws; rm -rf "$TMP"; mkdir -p "$TMP"

echo "== 1. pulling workspaces from $SERVER =="
rsync -az --exclude '*.tif' --exclude '.env' --exclude '*.json.bak' \
      "$SERVER:/home/undp/.undp_dashboard/workspaces/" "$TMP/"
rsync -az --exclude '*.tif' "$SERVER:/opt/undp-dashboard/outputs/" "$TMP/_server_shared_outputs/"
rsync -az --exclude '*.tif' --exclude '*.env' "$SERVER:/opt/undp-dashboard/data/" "$TMP/_server_shared_data/"

echo "== 2. merging into the shared tree (never overwrites an existing shared country) =="
merge_country () {   # $1 = config yaml, $2 = data root, $3 = outputs root
  local cfg=$1 code name
  name=$(basename "$cfg" .yaml)
  code=$(python3 -c "import yaml,sys;print(str(yaml.safe_load(open(sys.argv[1]))['country_code']).lower())" "$cfg")
  [ -f "config/$name.yaml" ] || { cp "$cfg" config/; echo "   + config/$name.yaml"; }
  [ -d "data/$code" ]    || { [ -d "$2/$code" ] && cp -R "$2/$code" "data/$code" && echo "   + data/$code"; }
  [ -d "outputs/$code" ] || { [ -d "$3/$code" ] && cp -R "$3/$code" "outputs/$code" && echo "   + outputs/$code"; }
}
# shared tree on the server first (mmr, pse, and anything run as shared), then each user's workspace
for cfg in "$TMP"/_server_shared_outputs/../_server_shared_data/../config/*.yaml; do :; done 2>/dev/null
for d in "$TMP"/*/; do
  [ "$(basename "$d")" = _server_shared_outputs ] && continue
  [ "$(basename "$d")" = _server_shared_data ] && continue
  for cfg in "$d"config/*.yaml; do [ -f "$cfg" ] && merge_country "$cfg" "$d/data" "$d/outputs"; done
done
for code_dir in "$TMP"/_server_shared_outputs/*/; do
  code=$(basename "$code_dir")
  [ -d "outputs/$code" ] || { cp -R "$code_dir" "outputs/$code"; echo "   + outputs/$code (server shared)"; }
  [ -d "data/$code" ] || { [ -d "$TMP/_server_shared_data/$code" ] && cp -R "$TMP/_server_shared_data/$code" "data/$code" && echo "   + data/$code (server shared)"; }
done

echo "== 3. staging what the dashboard reads (rasters and secrets excluded) =="
git add -f config/*.yaml
for c in data/*/; do
  c=${c%/}
  [ -d "$c/boundaries" ] && git add -f "$c"/boundaries/*.geojson 2>/dev/null || true
  [ -d "$c/external" ]   && git add -f "$c"/external/*.csv "$c"/external/*.geojson 2>/dev/null || true
  ls "$c"/gee_panel_*.csv >/dev/null 2>&1 && git add -f "$c"/gee_panel_*.csv || true
done
for o in outputs/*/; do
  o=${o%/}
  [ -f "$o/panel_data.csv" ] && git add -f "$o/panel_data.csv"
  [ -d "$o/figures" ]  && git add -f "$o"/figures/*.png "$o"/figures/*.csv "$o"/figures/*/*.png 2>/dev/null || true
  [ -d "$o/results" ]  && git add -f "$o"/results/*.csv 2>/dev/null || true
  [ -d "$o/showcase" ] && git add -f "$o"/showcase/* 2>/dev/null || true
  [ -d "$o/external" ] && git add -f "$o"/external/*.csv 2>/dev/null || true
done
echo
git status --short | awk '{print $2}' | cut -d/ -f1-2 | sort | uniq -c | sort -rn | head -30
echo
echo "Staged. Size of staged content:"; git diff --cached --stat | tail -1
echo "Next: git commit -m 'Publish country content for deployment' && git push"
