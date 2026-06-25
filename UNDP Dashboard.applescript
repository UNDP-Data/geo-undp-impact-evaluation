-- Compile this once into a clickable .app (see LAUNCHER_SETUP.md).
-- It runs the dashboard in the background with NO Terminal window.
tell application "Finder" to set repoFolder to (container of (path to me)) as alias
set repoPath to POSIX path of repoFolder
do shell script "cd " & quoted form of repoPath & " && nohup bash run_dashboard.sh > /tmp/undp_dashboard.log 2>&1 &"
