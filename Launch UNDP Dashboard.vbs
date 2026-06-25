' UNDP Early Recovery Dashboard - Windows launcher
' Double-click this file. First run shows a setup window; later runs are silent.
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = scriptDir
sentinel = scriptDir & "\.venv\.deps_installed"
If fso.FileExists(sentinel) Then
    style = 0   ' already set up -> run hidden, no console
Else
    style = 1   ' first run -> show window so the user sees install progress
End If
sh.Run "cmd /c """ & scriptDir & "\_undp_bootstrap.bat""", style, False
