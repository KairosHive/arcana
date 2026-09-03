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


def _pick_windows(initial: str | None) -> str:
    # PowerShell + WinForms: present on every supported Windows, and needs no
    # Python, so it works from a frozen build.
    #
    # The owner form has to be SHOWN. The first version created one, set
    # TopMost on it and passed it to ShowDialog() without ever calling Show().
    # An unshown form owns nothing and cannot take the foreground, and because
    # the host console is hidden (-WindowStyle Hidden, CREATE_NO_WINDOW) the
    # dialog opened behind the browser with no taskbar button -- invisible and
    # unreachable. The app just said "Updating..." until the timeout, and every
    # further click stacked another invisible dialog.
    #
    # So: show a 1x1 owner off-screen, keep it out of the taskbar, activate it,
    # and let it drag the dialog to the front.
    start = (initial or "").replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null;"
        "Add-Type -AssemblyName System.Drawing | Out-Null;"
        "$t = New-Object System.Windows.Forms.Form;"
        "$t.StartPosition = 'Manual';"
        "$t.Location = New-Object System.Drawing.Point(-32000, -32000);"
        "$t.Size = New-Object System.Drawing.Size(1, 1);"
        "$t.ShowInTaskbar = $false;"
        "$t.TopMost = $true;"
        "$t.Show();"
        "$t.Activate();"
        "[System.Windows.Forms.Application]::DoEvents();"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$d.Description = 'Choose the folder to index';"
        "$d.ShowNewFolderButton = $false;"
        f"if ('{start}' -ne '' -and (Test-Path '{start}')) {{ $d.SelectedPath = '{start}' }};"
        "if ($d.ShowDialog($t) -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ Write-Output $d.SelectedPath };"
        "$d.Dispose();"
        "$t.Close();"
        "$t.Dispose()"
    )
    return _run(["powershell", "-NoProfile", "-STA", "-NonInteractive",
                 "-WindowStyle", "Hidden", "-Command", script])


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
