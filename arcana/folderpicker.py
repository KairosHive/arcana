# folderpicker.py — a native "choose a folder" dialog
#
# A browser cannot hand a server a filesystem path: the File System Access API
# gives the *page* a handle, not a path, and a folder input gives file contents.
# Arcana runs on the user's own machine, so the dialog is opened server-side and
# the chosen path is filled into the text field.
#
# Two rules make this safe:
#
#   * It runs in a SUBPROCESS. A GUI dialog on a Flask worker thread is a
#     well-known way to hang a server -- tkinter in particular is not safe to
#     drive from a non-main thread -- and a subprocess can simply be killed.
#   * It always times out, so a dialog the user never answers cannot wedge the
#     app.
#
# Each platform gets a mechanism that survives freezing, because in a PyInstaller
# build sys.executable is the app itself and "python -c ..." is not available.

from __future__ import annotations

import os
import subprocess
import sys

# The Dash callback that opens this blocks until it returns, so the timeout is
# also how long the UI can sit on "Updating...". Five minutes was far too long
# to wait for a dialog that might not be visible; ninety seconds is enough to
# find a folder and short enough that a mistake is recoverable.
TIMEOUT = 90


class PickerUnavailable(RuntimeError):
    """No native dialog on this platform; the caller should fall back to typing."""


def _run(cmd: list[str], **kw) -> str:
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out = subprocess.run(
        cmd, capture_output=True, text=True, timeout=TIMEOUT,
        creationflags=flags, **kw,
    )
    return (out.stdout or "").strip()


# The dialog script, passed via -EncodedCommand rather than -Command so it can
# be written as readable multi-line PowerShell instead of one long
# semicolon-separated string with nested quoting.
#
# Two things had to be got right, in order of discovery:
#
#   1. The owner form must be SHOWN. The first version created one, set TopMost
#      and passed it to ShowDialog() without ever calling Show(). An unshown
#      form owns nothing, so with the console hidden the dialog opened behind
#      everything with no taskbar button -- the app said "Updating..." until it
#      timed out, and each further click stacked another invisible dialog.
#
#   2. TopMost alone is not enough. Windows refuses to let a background process
#      steal the foreground, so the dialog appeared above the terminal that
#      spawned it but still below the browser the user was actually looking at.
#      The documented way around the foreground lock is to attach this thread's
#      input queue to the current foreground window's thread, which makes the
#      two threads share focus state, and only then call SetForegroundWindow.
#      The attachment is released immediately afterwards.
#   3. FolderBrowserDialog is the OLD dialog. WinForms on .NET Framework --
#      which is what Windows PowerShell 5.1 runs on, and there is no pwsh on a
#      stock Windows -- calls SHBrowseForFolder, the cramped tree from Windows
#      95: no sidebar, no address bar, no search, no typing a path, no network
#      shortcuts. .NET Core's FolderBrowserDialog switched to the modern dialog
#      years ago, but that is not available here.
#
#      So this drives IFileOpenDialog directly, with FOS_PICKFOLDERS. That is
#      the dialog Explorer itself uses: Quick Access and This PC in the
#      sidebar, an address bar, search, and a path you can paste into. It needs
#      the COM vtable declared in exact order, which is why the interface below
#      lists methods it never calls -- each one is a slot, and a missing slot
#      would silently call the wrong function.
#
#      FolderBrowserDialog stays as a fallback. If the COM route ever throws on
#      some future Windows, a cramped dialog beats no dialog.
_WIN_SCRIPT = r'''
Add-Type -AssemblyName System.Windows.Forms | Out-Null
Add-Type -AssemblyName System.Drawing | Out-Null
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Fg {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr pid);
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool attach);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
  public static void Steal(IntPtr target) {
    uint other = GetWindowThreadProcessId(GetForegroundWindow(), IntPtr.Zero);
    uint mine  = GetCurrentThreadId();
    if (other != mine) AttachThreadInput(other, mine, true);
    BringWindowToTop(target);
    SetForegroundWindow(target);
    if (other != mine) AttachThreadInput(other, mine, false);
  }
}

// Explorer's folder dialog. Every method below is a vtable slot and must stay
// in this order even though most are never called -- COM dispatches by
// position, so a missing entry calls the wrong function rather than failing.
[ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IShellItem {
  void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
  void GetParent(out IShellItem ppsi);
  void GetDisplayName(uint sigdnName, out IntPtr ppszName);
  void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
  void Compare(IShellItem psi, uint hint, out int piOrder);
}

[ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IFileDialog {
  [PreserveSig] int Show(IntPtr parent);
  void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
  void SetFileTypeIndex(uint iFileType);
  void GetFileTypeIndex(out uint piFileType);
  void Advise(IntPtr pfde, out uint pdwCookie);
  void Unadvise(uint dwCookie);
  void SetOptions(uint fos);
  void GetOptions(out uint pfos);
  void SetDefaultFolder(IShellItem psi);
  void SetFolder(IShellItem psi);
  void GetFolder(out IShellItem ppsi);
  void GetCurrentSelection(out IShellItem ppsi);
  void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
  void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
  void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
  void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
  void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
  void GetResult(out IShellItem ppsi);
  void AddPlace(IShellItem psi, int fdap);
  void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
  void Close(int hr);
  void SetClientGuid(ref Guid guid);
  void ClearClientData();
  void SetFilter(IntPtr pFilter);
}

public static class ArcanaPicker {
  [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
  static extern void SHCreateItemFromParsingName(
      [MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc,
      ref Guid riid, [MarshalAs(UnmanagedType.Interface)] out object ppv);

  [DllImport("ole32.dll")] static extern void CoTaskMemFree(IntPtr p);

  public static string Pick(IntPtr owner, string title, string initial) {
    Guid clsid = new Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7");
    IFileDialog dlg = (IFileDialog)Activator.CreateInstance(Type.GetTypeFromCLSID(clsid));
    try {
      uint opts;
      dlg.GetOptions(out opts);
      // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
      dlg.SetOptions(opts | 0x20 | 0x40 | 0x800);
      if (!string.IsNullOrEmpty(title)) dlg.SetTitle(title);
      if (!string.IsNullOrEmpty(initial) && System.IO.Directory.Exists(initial)) {
        try {
          Guid iid = new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe");
          object si;
          SHCreateItemFromParsingName(initial, IntPtr.Zero, ref iid, out si);
          dlg.SetFolder((IShellItem)si);
        } catch { /* a bad starting folder must not stop the dialog */ }
      }
      if (dlg.Show(owner) != 0) return null;          // cancelled
      IShellItem res;
      dlg.GetResult(out res);
      IntPtr p;
      res.GetDisplayName(0x80058000, out p);          // SIGDN_FILESYSPATH
      string path = Marshal.PtrToStringUni(p);
      CoTaskMemFree(p);
      return path;
    } finally {
      Marshal.ReleaseComObject(dlg);
    }
  }
}
"@

$t = New-Object System.Windows.Forms.Form
$t.StartPosition = 'Manual'
$t.Location = New-Object System.Drawing.Point(-32000, -32000)
$t.Size = New-Object System.Drawing.Size(1, 1)
$t.ShowInTaskbar = $false
$t.TopMost = $true
$t.Show()
$t.Activate()
[System.Windows.Forms.Application]::DoEvents()
[Fg]::Steal($t.Handle)

$picked = $null
try {
  $picked = [ArcanaPicker]::Pick($t.Handle, 'Choose the folder to index', '__INITIAL__')
} catch {
  # Modern dialog unavailable: fall back to the old tree rather than nothing.
  $d = New-Object System.Windows.Forms.FolderBrowserDialog
  $d.Description = 'Choose the folder to index'
  $d.ShowNewFolderButton = $false
  __SELECTED__
  if ($d.ShowDialog($t) -eq [System.Windows.Forms.DialogResult]::OK) { $picked = $d.SelectedPath }
  $d.Dispose()
}
if ($picked) { Write-Output $picked }
$t.Close()
$t.Dispose()
'''


def _pick_windows(initial: str | None) -> str:
    # PowerShell + WinForms: present on every supported Windows, and needs no
    # Python, so it works from a frozen build.
    import base64

    if initial:
        start = initial.replace("'", "''")
        seed = (f"if (Test-Path '{start}') {{ $d.SelectedPath = '{start}' }}")
    else:
        start, seed = "", ""
    script = (_WIN_SCRIPT
              .replace("__SELECTED__", seed)
              .replace("__INITIAL__", start))
    # -EncodedCommand takes UTF-16LE base64, which sidesteps every quoting and
    # newline question between here and PowerShell's parser.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return _run(["powershell", "-NoProfile", "-STA", "-NonInteractive",
                 "-WindowStyle", "Hidden", "-EncodedCommand", encoded])


def _pick_macos(initial: str | None) -> str:
    where = f' default location POSIX file "{initial}"' if initial and os.path.isdir(initial) else ""
    script = (f'POSIX path of (choose folder with prompt '
              f'"Choose the folder to index"{where})')
    return _run(["osascript", "-e", script])


def _pick_linux(initial: str | None) -> str:
    for tool, args in (
        ("zenity", ["--file-selection", "--directory",
                    "--title=Choose the folder to index"]),
        ("kdialog", ["--getexistingdirectory", initial or os.path.expanduser("~")]),
    ):
        try:
            return _run([tool, *args])
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    raise PickerUnavailable("install zenity or kdialog for a folder dialog")


def available() -> bool:
    """Whether a dialog can plausibly be shown. Never raises."""
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        return True
    from shutil import which
    return bool(which("zenity") or which("kdialog"))


def pick_folder(initial: str | None = None) -> str | None:
    """
    Show a native folder chooser and return the path, or None if cancelled.

    Raises PickerUnavailable when the platform has no dialog to offer, and
    TimeoutError if the user never answers.
    """
    initial = initial if (initial and os.path.isdir(initial)) else None
    try:
        if sys.platform == "win32":
            path = _pick_windows(initial)
        elif sys.platform == "darwin":
            path = _pick_macos(initial)
        else:
            path = _pick_linux(initial)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError("the folder dialog was left open") from e
    except FileNotFoundError as e:
        raise PickerUnavailable(str(e)) from e

    path = (path or "").strip().strip('"')
    if not path:
        return None                       # cancelled
    return path if os.path.isdir(path) else None


if __name__ == "__main__":              # manual check: python -m arcana.folderpicker
    print(pick_folder() or "(cancelled)")
