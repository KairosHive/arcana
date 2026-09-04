# Installing Arcana

Three routes. Pick one:

- **[The installer](#the-installer)** — you want to use Arcana on Windows.
- **[uv](#uv)** — you want to work on Arcana.
- **[pip](#pip)** — you want Arcana inside an environment you already manage.

All three run the same code. They differ in what they assume you already have,
and whether they can use a GPU.

Everything runs locally. No account, no upload, no network beyond downloading
the models on first use.

---

## The installer

Download `Arcana-Setup-<version>.exe` and run it. That is the whole procedure —
Python and PyTorch are inside it.

**Windows will warn you.** The build is not code-signed, so SmartScreen shows
"Windows protected your PC". Choose **More info → Run anyway**. A signing
certificate is a purchase we have not made; the warning means unrecognised, not
unsafe.

By default it installs **just for you** and needs no administrator rights. The
wizard offers "for everyone" if you prefer, which does need them.

Arcana keeps its data in `%LOCALAPPDATA%\Arcana`, separate from the program.
**Uninstalling does not delete it** — your indexes, downloaded models and saved
results stay, and a later version picks them up again. Delete the folder by hand
if you want the space back.

Installing a newer version upgrades in place. There is no need to uninstall
first.

### One thing to know

The packaged build ships **CPU-only PyTorch**, because the CUDA build is 4.4 GB
against 421 MB and would take the download from 226 MB to well over 2 GB.

Search never needs a GPU. Indexing works without one — the fast encoder is only
about twice as quick with a card. If you own an NVIDIA GPU and index large
collections with the largest encoder, install from source instead; that is
roughly thirty times faster there.

---

## uv

The development route, and what the project is built with.

```bash
git clone https://github.com/KairosHive/arcana.git
cd arcana
uv sync
uv run python -m arcana.arcana
```

`uv sync` reads `uv.lock` and reproduces the exact environment the project was
tested against, including a CUDA build of PyTorch on machines that can use one.

The development server runs on **port 8051**, deliberately not the 8050 the
packaged app uses, so you can run both side by side and compare.

Useful while working:

```bash
uv run python -m pytest tests -q        # the test suite
uv run python -m arcana.envcheck        # has the environment drifted from the lock?
```

`envcheck` is worth knowing about: installing a tool with `uv pip install` can
quietly re-resolve everything, and this tells you what moved.

---

## pip

For putting Arcana into an environment you already control.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install .
arcana
```

That installs the package, its dependencies and four commands: `arcana`,
`arcana-build-latent`, `arcana-relocate`, `arcana-migrate`.

Use `pip install -e .` instead if you want your edits to take effect without
reinstalling.

### Getting a GPU build with pip

`pyproject.toml` pins PyTorch to NVIDIA's package index, but **that pin is a uv
feature and pip ignores it**. A plain `pip install .` therefore gets whatever
PyTorch PyPI serves by default, which on Windows and Linux is the CPU build.

For GPU support, install PyTorch first from the index that has it:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install .
```

Check it worked:

```bash
python -c "from arcana import gpu; print(gpu.describe())"
```

That prints the card and precision it will use, or the reason it fell back to
the CPU. It is a real capability check, not just "is a driver present" — a card
too old for the shipped kernels is reported rather than crashing halfway
through an index.

---

## Building the installer yourself

Only needed if you want to produce a `.exe`. Two extra pieces:

**1. A packaging environment with CPU-only PyTorch.** The lock file pins the
CUDA build, which is 4.4 GB and does not belong in a download, so this
environment matches the lock in everything except torch:

```bash
uv venv .venv-pack
.venv-pack\Scripts\python -m pip install pyinstaller
# every locked package except torch, which comes from PyPI as CPU-only
grep -vE "^(torch|torchvision|torchaudio)([=<>~ ]|$)" requirements-lock.txt > pack-reqs.txt
uv pip install --python .venv-pack\Scripts\python.exe -r pack-reqs.txt --no-deps
uv pip install --python .venv-pack\Scripts\python.exe torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
.venv-pack\Scripts\python -m arcana.envcheck      # should say it matches
```

**2. Inno Setup**, for turning the built folder into one installer:

```bash
winget install --id JRSoftware.InnoSetup -e
```

Then build:

```bash
.venv-pack\Scripts\pyinstaller installer\arcana.spec --noconfirm
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installer\arcana.iss
```

The result is `installer\out\Arcana-Setup-<version>.exe`, around 226 MB.

Two traps that cost time the first time round:

- **winget installs Inno per-user**, so `ISCC.exe` is under `%LOCALAPPDATA%\Programs\Inno Setup 6`, not the `C:\Program Files (x86)` path most documentation assumes.
- **Inno's preprocessor reads a leading `#` as a directive**, so a Pascal continuation line starting `#13#10 + ...` aborts the compile with "Unknown preprocessor directive". Keep newline constants mid-line.

The version comes from `arcana/__init__.py` alone. `arcana.spec` stamps the
executable and generates `installer/version.iss` from it, so the installer can
never claim a version different from the code inside it.

---

## Requirements

Python **3.10 or newer** (the code uses `X | None` and `dict[int, str]`).

Models download themselves on first use, into your data directory:

| | size |
|---|---|
| CLIP ViT-B/32 (default) | 605 MB |
| CLIP ViT-L/14 | 1.7 GB |
| CLIP ViT-H/14 | 3.9 GB |
| CLAP (audio) | 614 MB |
| ModFlows colour transfer | 229 MB |

Only what you actually use is fetched.

---

## Two machines

Code goes through git. Datasets do not — a single index can be hundreds of
megabytes, so they are deliberately not tracked.

To share datasets between machines, point both at one folder:

```bash
setx ARCANA_DATA_DIR "D:\arcana-data"        # Windows, new shells
export ARCANA_DATA_DIR=/mnt/shared/arcana    # macOS / Linux
```

That works for a source checkout and the installed app alike, so both see the
same datasets.

Otherwise just re-index on the second machine. It is quicker than it used to
be: a few hundred photographs take seconds.

---

## If something goes wrong

**"No module named cv2" after `pip install .`** — dependencies did not install.
Check the network and re-run; `pip install .` needs to reach PyPI.

**The app says it is using the CPU but you have a card** — run
`python -c "from arcana import gpu; print(gpu.describe())"`. It states the
reason: no driver, a card older than the shipped kernels, or a packaged build,
which is always CPU-only.

**A dataset says "unfinished"** — indexing wrote the index but never the 2-D
map, usually an interrupted run. Index that folder again.

**A dataset says "files missing"** — the pictures moved. `arcana-relocate
--name <dataset>` finds them again by content, without re-indexing.
