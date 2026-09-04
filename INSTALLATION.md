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

### Two builds: CPU and GPU

`Arcana-Setup-<version>.exe` ships **CPU-only PyTorch** (310 MB against 4,312 MB
for the CUDA build) so the download stays near 200 MB. It runs everywhere and
needs no graphics card.

`Arcana-Setup-<version>-GPU.exe` is the same application with CUDA PyTorch. It is
a much larger download, and only worth it if you have an NVIDIA card.

They are the same product to Windows — same AppId — so **installing one over
the other upgrades in place**. You will not end up with both.

Which to pick:

| | CPU | GPU |
|---|---|---|
| Prompt search | instant | instant |
| Indexing, ViT-B/32 | fine (decode-bound either way) | slightly faster |
| Indexing, ViT-H/14 | 715 ms an image | 7 ms — **100x** |
| Style features | 11.7 /s | 45 /s — 3.8x |
| Inject Poetry | 37.5 s a scene | 1.3 s — **29x** |

Search never needs a GPU. Take the CPU build unless you index large collections
with the largest encoder, or use Inject Poetry.

The app tells you which situation you are in: if it finds an NVIDIA card it
cannot use, the Datasets panel and Inject Poetry both say so, rather than
leaving you to wonder why things are slow.

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

### The GPU build

Same spec and same .iss, pointed at an environment with CUDA PyTorch. Build that
environment from the lock, which already pins `torch==2.9.1+cu128`. The CUDA
wheels are not on PyPI, so torch comes from NVIDIA's index first and everything
else from the lock afterwards:

```bash
uv venv .venv-pack-gpu
uv pip install --python .venv-pack-gpu\Scripts\python.exe torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
grep -vE "^(torch|torchvision|torchaudio)([=<>~ ]|$)" requirements-lock.txt > gpu-reqs.txt
uv pip install --python .venv-pack-gpu\Scripts\python.exe -r gpu-reqs.txt --no-deps
uv pip install --python .venv-pack-gpu\Scripts\python.exe pyinstaller
.venv-pack-gpu\Scripts\python -m arcana.envcheck
```

Then build and compile it:

```bash
$env:ARCANA_BUILD_VARIANT="GPU"; .venv-pack-gpu\Scripts\pyinstaller installer\arcana.spec --noconfirm --distpath dist --workpath build-gpu
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" /DGpu installer\arcana.iss
```

`ARCANA_BUILD_VARIANT=GPU` stages to `dist\Arcana-GPU`, and `/DGpu` points the
installer at that folder and appends `-GPU` to the setup filename — so the two
builds never overwrite each other's output.

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
