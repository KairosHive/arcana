; Inno Setup script for the Arcana desktop installer.
;
;   1. build the app first:  .venv-pack\Scripts\pyinstaller installer\arcana.spec --noconfirm
;   2. then compile this:    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\arcana.iss
;
; Output lands in installer\out\Arcana-Setup-<version>.exe.
;
;   2b. ISCC.exe location depends on how Inno was installed. winget installs
;       per-user by default, which puts it under LOCALAPPDATA rather than the
;       Program Files path most documentation assumes:
;         %LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe   (winget / per-user)
;         C:\Program Files (x86)\Inno Setup 6\ISCC.exe     (machine-wide)
;
; Inno Setup is not installed by default. Get it with:
;   winget install --id JRSoftware.InnoSetup -e
;
; Why an installer at all, when PyInstaller already produces a runnable folder:
; the folder has no Start Menu entry, no uninstaller, no Add/Remove Programs
; row, and no way to tell a user "just run this". It is also 927 MB of loose
; files to copy around; compressed into one file it is roughly half that.

#define AppName        "Arcana"
#define AppVersion     "0.2.0"
#define AppPublisher   "Antoine Bellemare"
#define AppExeName     "Arcana.exe"
; Relative to this .iss file, so the script works from any working directory.
#define DistDir        "..\dist\Arcana"

[Setup]
; A stable GUID. Changing it makes Windows treat a new build as a different
; product, which would leave the old one installed alongside -- 927 MB twice.
AppId={{7A1C4E62-9B3D-4F58-A0E1-6C2D8B5F3A47}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=out
OutputBaseFilename=Arcana-Setup-{#AppVersion}
; No SetupIconFile: it requires a .ico and the project has none yet, so the
; wizard uses Inno's default. Add arcana/assets/arcana.ico and set both this and
; the spec's `icon=` when there is artwork. UninstallDisplayIcon does accept an
; .exe, so Add/Remove Programs still shows the app's own icon.
UninstallDisplayIcon={app}\{#AppExeName}
; AppVersion drives the wizard and Add/Remove Programs, but NOT the Windows file
; properties of the setup binary itself -- without these the compiled .exe shows
; a blank File version, which reads as an unfinished build.
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
WizardStyle=modern
DisableProgramGroupPage=yes
; The app has no licence file yet; add LicenseFile= here when it does.

; Torch ships x64 binaries only, so refuse the install rather than let it fail
; at the first import.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Let the user choose between "everyone" (Program Files, needs admin) and "just
; me" (no UAC prompt at all). Arcana keeps all its state in the per-user data
; directory either way, so a per-user install is genuinely equivalent -- and it
; is the only option for someone without admin on a work machine.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; LZMA2/max roughly halves 927 MB. It is slow to compile (several minutes) and
; costs nothing at install time beyond decompression.
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Everything PyInstaller produced, including _internal (which holds the bundled
; ModFlows source and the Dash/Plotly assets).
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; nowait + postinstall: the wizard closes immediately instead of sitting there
; while a Dash server runs. skipifsilent keeps unattended installs unattended.
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; PyInstaller writes __pycache__ directories beside the bundled modules at
; runtime. They are created after install, so Inno does not know about them and
; would otherwise leave {app} behind as a non-empty directory.
Type: filesandordirs; Name: "{app}\_internal\modflows\src\__pycache__"

[Code]
// Deliberately NOT deleted on uninstall: %LOCALAPPDATA%\Arcana. It holds the
// user's indexed datasets, downloaded encoders and the ModFlows checkpoint --
// gigabytes that are expensive to rebuild and that they may well want when they
// reinstall. Tell them where it is instead of silently destroying it.
// NB: no line in this block may START with '#'. The Inno *preprocessor* runs
// before the Pascal compiler and reads a leading '#' as a directive, so a
// continuation line beginning "#13#10 + ..." aborts the build with "Unknown
// preprocessor directive". Keep the newline constants mid-line.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir, Msg: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\Arcana');
    if DirExists(DataDir) then
    begin
      Msg := 'Your Arcana data has been kept at:' + #13#10 + #13#10;
      Msg := Msg + DataDir + #13#10 + #13#10;
      Msg := Msg + 'That folder holds your indexed datasets, downloaded ';
      Msg := Msg + 'encoders and saved results. Delete it by hand if you ';
      Msg := Msg + 'want the space back.';
      MsgBox(Msg, mbInformation, MB_OK);
    end;
  end;
end;
