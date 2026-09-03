# PyInstaller spec for the Arcana desktop build.
#
#   .venv-pack\Scripts\pyinstaller installer\arcana.spec --noconfirm
#
# Build from an environment with the CPU-only torch: the CUDA build is 4.4 GB
# against 421 MB, and searching never needs a GPU. Indexing and story mode are
# slower without one, which is the documented trade for a shippable download.
#
# onedir, not onefile, on purpose: onefile unpacks a multi-gigabyte archive to a
# temp directory on every launch. onedir starts immediately and is what an
# installer wants anyway.

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

BLOCK_CIPHER = None
ROOT = os.path.abspath(os.getcwd())

# ── data files ────────────────────────────────────────────────────────────────
datas = [
    # Dash serves assets/custom.css from the package directory; without it every
    # input renders white-on-white. db.py reads assets/labels_*.txt for cluster
    # naming. Neither is a .py file, so nothing collects them automatically.
    (os.path.join(ROOT, "arcana", "assets"), os.path.join("arcana", "assets")),
]

# ModFlows is not a package we import normally: color_transfer adds its
# directory to sys.path and does `import src.encoder`. PyInstaller cannot see
# through that, so the source ships as plain files next to the executable --
# 120 KB, and color_transfer._candidate_dirs() already looks in
# <exe dir>/modflows. The 229 MB checkpoint deliberately does NOT ship; it is
# downloaded once into the user's writable data directory on first use, which
# is also the only place it could survive a reinstall.
_modflows_src = os.path.join(ROOT, "modflows", "src")
if os.path.isdir(_modflows_src):
    datas.append((_modflows_src, os.path.join("modflows", "src")))
else:
    print("[spec] WARNING: modflows/src not found -- the built app will offer "
          "LAB colour transfer only.")
for pkg in ("dash", "dash_daq", "plotly", "librosa", "usearch"):
    try:
        datas += collect_data_files(pkg)
    except Exception as e:                                   # pragma: no cover
        print(f"[spec] WARNING: no data files collected for {pkg}: {e}")

# ── imports PyInstaller's static analysis cannot see ──────────────────────────
# Everything here is imported inside a function, behind a try/except, or by
# string. Verified by walking the AST of arcana/*.py for non-module-scope
# imports; see docs/hardening-audit.md.
hiddenimports = [
    # our own modules, several of which are only imported lazily
    "arcana.arcana", "arcana.db", "arcana.bundle", "arcana.legacy",
    "arcana.relocate", "arcana.paths", "arcana.cvio",
    "arcana.palette", "arcana.style", "arcana.color_transfer", "arcana.lab_transfer",
    # Added after the first draft of this spec. models/jobs/ui_* are reached
    # from arcana.py at module scope, but refselect, folderpicker and envcheck
    # are imported inside functions and would otherwise be dropped -- taking
    # the colour-range panel and the Browse button with them.
    "arcana.models", "arcana.jobs", "arcana.envcheck", "arcana.folderpicker",
    "arcana.ui_datasets", "arcana.ui_style", "arcana.refselect",
    # scientific bits reached only from inside functions
    "sklearn.decomposition", "sklearn.manifold", "sklearn.cluster", "sklearn.metrics",
    "scipy.special.cython_special",
    "ot",                       # POT, used for palette EMD
    # media
    "PIL.PngImagePlugin", "PIL.WebPImagePlugin", "PIL.JpegImagePlugin",
    "soundfile", "librosa", "numba",
    # modflows/src/encoder.py needs these, and it is shipped as data rather
    # than analysed, so nothing else pulls them in.
    "torchvision", "torchvision.transforms", "torchvision.transforms.v2",
    "einops",
    # models
    "usearch.index",
    "transformers", "transformers.models.clip", "transformers.models.clap",
    "diffusers", "diffusers.pipelines.stable_diffusion",
    # dash component libraries register themselves at import
    "dash_daq",
]
hiddenimports += collect_submodules("dash")

# ── what we deliberately leave out ───────────────────────────────────────────
excludes = [
    "tkinter",          # nothing in the app uses it
    "pytest", "pyflakes", "mypy", "ruff",
    "IPython", "jupyter", "notebook",
    # NOT torch.distributed: torch/utils/data/dataloader.py imports it
    # unconditionally, so excluding it fails at `import torch` with
    # ModuleNotFoundError. Same reasoning for torch.testing -- leave torch
    # alone and let its own PyInstaller hook decide what is reachable.
]

a = Analysis(
    [os.path.join(ROOT, "installer", "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=BLOCK_CIPHER)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Arcana",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts some numpy/torch DLLs
    # Keep the console for the spike: a windowed build hides the traceback that
    # tells you what is missing. Flip to console=False once it launches clean.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Arcana",
)
