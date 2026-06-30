# Deploy the UNDP Dashboard to a Hetzner server

You need: a credit card, your two secret files (gee_service_account.json,
pilot_credentials.txt), ~45 min. Everything else is copy-paste.

## 1. Create the server (Hetzner Cloud)
1. Sign up at https://console.hetzner.cloud  → New Project → "undp".
2. Add Server:
   - Location: Falkenstein or Helsinki (cheapest, EU).
   - Image: **Ubuntu 24.04**.
   - Type: **CX33** (4 vCPU / 8 GB, ~€6.49/mo). If runs feel slow later, resize up.
   - SSH key: add yours if you have one, OR choose "password" and Hetzner emails it.
   - Create & Buy. Note the server's **IP address**.

## 2. Connect to the server
From your Mac Terminal (replace with your IP):
```
ssh root@YOUR_SERVER_IP
```
(accept the fingerprint; enter the password Hetzner gave you if prompted)

## 3. Run the setup script
On the server, paste:
```
curl -fsSL https://raw.githubusercontent.com/hgrueso/undp-impact-evaluation/main/deploy/setup_server.sh -o setup_server.sh
bash setup_server.sh
```
It will pause at step 6 to ask for a dashboard **username** and **password** —
this is what your colleagues will type to log in. Pick something and share it
with them privately.

## 4. Upload the two secret files
The script can't fetch your secrets (they're not in the public repo). From a
SECOND Terminal window ON YOUR MAC (not the server), run — adjust the local path:
```
cd ~/Dropbox/Documents/Laboral/AImpact\ Lab/Clients/UNDP/data_analysis
scp gee_service_account.json root@YOUR_SERVER_IP:/opt/undp-dashboard/
scp pilot_credentials.txt    root@YOUR_SERVER_IP:/opt/undp-dashboard/
```
Then back ON THE SERVER:
```
cp /opt/undp-dashboard/pilot_credentials.txt /opt/undp-dashboard/.env
chown undp:undp /opt/undp-dashboard/gee_service_account.json /opt/undp-dashboard/pilot_credentials.txt /opt/undp-dashboard/.env
```

## 5. Start it
On the server:
```
systemctl start undp-dashboard
systemctl status undp-dashboard --no-pager
```
Look for "active (running)". Then open in your browser:
```
http://YOUR_SERVER_IP
```
Log in with the username/password from step 3. You should see the dashboard.

## 6. Test
- View existing countries (instant).
- Setup & Run → pick a country → Run Full Pipeline. Watch it run server-side.
- Open the site in a second browser/incognito at the same time and try to run a
  pipeline → you should see the "a pipeline is running, please wait" message.
  That confirms the one-at-a-time queue works.

## Useful commands (on the server)
```
systemctl restart undp-dashboard     # restart after changes
journalctl -u undp-dashboard -f      # live logs / errors
cd /opt/undp-dashboard && git pull   # update to latest code, then restart
```

## Notes
- The login is a simple shared password (fine for a ~20-person pilot).
- Only one pipeline runs at a time (the queue lock). Viewing is unlimited/simultaneous.
- A proper https domain can be added later (Let's Encrypt) — http works for the pilot.
- Cost stops when you delete the server; you can also power it off to pause billing
  partially (you still pay for the disk while it exists).
