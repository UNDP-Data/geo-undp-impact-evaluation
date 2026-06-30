#!/bin/bash
# =============================================================================
# UNDP Dashboard - one-shot server setup for a fresh Ubuntu 24.04 box (Hetzner)
# Run as root on the new server:  bash setup_server.sh
# =============================================================================
set -e

REPO="https://github.com/hgrueso/undp-impact-evaluation.git"
APP_DIR="/opt/undp-dashboard"
APP_USER="undp"

echo "=== 1. System packages (incl. geospatial native libs) ==="
apt-get update
apt-get install -y python3 python3-venv python3-pip git \
    gdal-bin libgdal-dev libgeos-dev libproj-dev proj-data proj-bin \
    build-essential nginx apache2-utils ufw

echo "=== 2. App user + clone repo ==="
id -u $APP_USER >/dev/null 2>&1 || useradd -m -s /bin/bash $APP_USER
rm -rf "$APP_DIR"
git clone "$REPO" "$APP_DIR"
chown -R $APP_USER:$APP_USER "$APP_DIR"

echo "=== 3. Python venv + dependencies ==="
sudo -u $APP_USER bash -c "cd $APP_DIR && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt"

echo ""
echo "=== 4. ACTION NEEDED: place your secret files, then re-run step 5 ==="
echo "  Copy these two files into $APP_DIR (see the deploy guide for the scp command):"
echo "    - gee_service_account.json"
echo "    - pilot_credentials.txt"
echo "  Then create $APP_DIR/.env from pilot_credentials.txt:"
echo "    sudo -u $APP_USER cp $APP_DIR/pilot_credentials.txt $APP_DIR/.env"
echo ""

echo "=== 5. systemd service (keeps the app running, restarts on reboot) ==="
cat > /etc/systemd/system/undp-dashboard.service << SERVICE
[Unit]
Description=UNDP Early Recovery Dashboard
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE
systemctl daemon-reload
systemctl enable undp-dashboard

echo "=== 6. nginx reverse proxy + password protection ==="
read -p "Set a username for the dashboard login: " DASH_USER
htpasswd -c /etc/nginx/.htpasswd "$DASH_USER"
cat > /etc/nginx/sites-available/undp << NGINX
server {
    listen 80;
    server_name _;
    location / {
        auth_basic "UNDP Dashboard";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 86400;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/undp /etc/nginx/sites-enabled/undp
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

echo "=== 7. firewall ==="
ufw allow OpenSSH
ufw allow 'Nginx HTTP'
ufw --force enable

echo ""
echo "============================================================"
echo "Base setup done. Now:"
echo "  1) Copy secret files into $APP_DIR (step 4)"
echo "  2) Create .env  (step 4)"
echo "  3) systemctl start undp-dashboard"
echo "  4) Visit  http://YOUR_SERVER_IP  and log in"
echo "============================================================"
