' ============================================================================
'  UNDP Early Recovery Dashboard - Windows launcher (no console window)
'  Double-click this file. First run sets up (a few minutes); later runs are
'  instant. The browser opens automatically. No terminal is ever shown.
'  Place this file in the same folder as app.py.
' ============================================================================
Option Explicit
Dim sh, fso, Q, scriptDir, homeDir, credDir, homeEnv, venvPy, sentinel
Dim pyCmd, rc, credFile, ts, line, ef

Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
Q = Chr(34)

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = scriptDir
homeDir  = sh.ExpandEnvironmentStrings("%USERPROFILE%")
credDir  = homeDir & "\.undp_dashboard"
homeEnv  = credDir & "\.env"
venvPy   = scriptDir & "\.venv\Scripts\python.exe"
sentinel = scriptDir & "\.venv\.deps_installed"

' --- Create a branded desktop shortcut on first run (so it looks like a real app) ---
Dim iconPath, deskLnk, lnk
iconPath = scriptDir & "\icon.ico"
deskLnk  = sh.SpecialFolders("Desktop") & "\UNDP Dashboard.lnk"
If fso.FileExists(iconPath) And Not fso.FileExists(deskLnk) Then
    Set lnk = sh.CreateShortcut(deskLnk)
    lnk.TargetPath = "wscript.exe"
    lnk.Arguments  = Chr(34) & WScript.ScriptFullName & Chr(34)
    lnk.WorkingDirectory = scriptDir
    lnk.IconLocation = iconPath
    lnk.Description = "UNDP Early Recovery Dashboard"
    lnk.Save
End If

' --- Seed credentials from a sibling pilot_credentials.txt (if present) ---
'     pilot_credentials.txt is gitignored and distributed privately, NEVER committed.
credFile = scriptDir & "\pilot_credentials.txt"
If fso.FileExists(credFile) And Not fso.FileExists(homeEnv) Then
    If Not fso.FolderExists(credDir) Then fso.CreateFolder(credDir)
    Set ts = fso.OpenTextFile(credFile, 1)
    Set ef = fso.CreateTextFile(homeEnv, True)
    Do Until ts.AtEndOfStream
        line = Trim(ts.ReadLine)
        If line <> "" And Left(line,1) <> "#" Then ef.WriteLine line
    Loop
    ts.Close : ef.Close
End If

' --- Detect an existing Python (do NOT force a reinstall) ---
pyCmd = ""
If PyOk("py")      Then pyCmd = "py"
If pyCmd = "" Then If PyOk("python")  Then pyCmd = "python"
If pyCmd = "" Then If PyOk("python3") Then pyCmd = "python3"
If pyCmd = "" Then
    MsgBox "Python 3.10 or newer is required." & vbCrLf & vbCrLf & _
           "Install it from https://www.python.org/downloads/ and tick " & _
           "'Add python.exe to PATH', then double-click this launcher again.", _
           vbExclamation, "UNDP Dashboard"
    WScript.Quit
End If

' --- First-time setup (hidden), with a friendly heads-up dialog ---
If Not fso.FileExists(sentinel) Then
    MsgBox "First-time setup will take a few minutes. Click OK, then please wait." & vbCrLf & _
           "The dashboard will open in your browser automatically when ready.", _
           vbInformation, "UNDP Dashboard"
    If Not fso.FileExists(venvPy) Then
        sh.Run Q & pyCmd & Q & " -m venv .venv", 0, True
    End If
    sh.Run Q & venvPy & Q & " -m pip install --upgrade pip", 0, True
    rc = sh.Run(Q & venvPy & Q & " -m pip install -r requirements.txt", 0, True)
    If rc <> 0 Then
        MsgBox "Setup failed while installing components. Please check your " & _
               "internet connection and try again.", vbCritical, "UNDP Dashboard"
        WScript.Quit
    End If
    Set ef = fso.CreateTextFile(sentinel, True) : ef.WriteLine "done" : ef.Close
End If

' --- Run the app hidden; browser opens automatically ---
sh.Run Q & venvPy & Q & " -m streamlit run app.py", 0, False

Function PyOk(cmd)
    On Error Resume Next
    Dim r
    r = sh.Run(Q & cmd & Q & " --version", 0, True)
    PyOk = (Err.Number = 0) And (r = 0)
    Err.Clear
    On Error GoTo 0
End Function
