<p align="center">
  <img src="arcana/assets/arcana-mark.png" alt="" width="120">
</p>

<h1 align="center">Arcana</h1>

<p align="center">
  Explore, search and tell stories with large image and audio collections,<br>
  by the way they look and sound rather than by filename.
</p>

---

Arcana reads your files **where they are**. Nothing is copied, moved or
uploaded — it builds a searchable index beside them and leaves the originals
alone. Everything runs on your own machine.

## What it does

**Prompt search** — type *"a quiet street at dusk"* and get the pictures that
look like it, from a collection that was never tagged. Works on audio too:
*"distant thunder"* against a folder of field recordings.

**A map of the collection** — every file placed by similarity, clustered and
named, so you can see the shape of what you have and click your way around it.

**Moodboard** — collect pictures, then find more like them by colour palette or
texture, and transfer the colours of one onto another.

**Story mode** — give it a sequence of scenes and it assembles a visual
narrative from your own images.

## Install

Three ways, depending on who you are — see **[INSTALLATION.md](INSTALLATION.md)**
for the details.

| | for | command |
|---|---|---|
| **Installer** | using Arcana on Windows | run `Arcana-Setup-<version>.exe` |
| **uv** | developing on it | `uv sync && uv run python -m arcana.arcana` |
| **pip** | adding it to an environment you already have | `pip install .` then `arcana` |

The installer needs nothing else. The other two need Python 3.10+.

## First run

Arcana opens on the **Datasets** tab, which walks you through four steps:

1. **Point at a folder** — everything inside, including subfolders
2. **Check what it found** — media type and name are detected from the folder
3. **Choose the quality** — a stronger encoder understands prompts better but
   takes longer to index
4. **Name the groups, then go** — optional extras, then Start indexing

The encoder downloads itself on first use (605 MB for the fast one). Indexing
10,000 photographs takes about seven minutes on a CPU, three with a GPU.

Once a dataset is built it appears in the **Dataset** menu at the top of every
tab, and the other three tabs come alive.

## Where things live

Your media is never touched. Everything Arcana creates goes in one place:

| | |
|---|---|
| Windows | `%LOCALAPPDATA%\Arcana` |
| macOS | `~/Library/Application Support/Arcana` |
| Linux | `~/.local/share/arcana` |
| anywhere | set `ARCANA_DATA_DIR` |

That folder holds the indexes, the downloaded models and anything you save. It
survives uninstalling, and pointing two machines at the same one lets them
share datasets.

## Command line

The GUI covers everything, but each piece is also a command:

```bash
arcana                                   # the app
arcana-build-latent --path ./photos --name holiday
arcana-relocate --name holiday           # after moving your files
arcana-migrate                           # older datasets to the portable format
```

`arcana-build-latent --help` lists the indexing options — cluster count, label
vocabulary, which features to extract.

## Requires

Python 3.10 or newer. A GPU is optional: search never needs one, and indexing
works without one — it is simply slower with the larger encoders.

## Licence

Not yet declared. `pyproject.toml` sets no licence and the repository has no
LICENSE file, which means default copyright applies and others have no explicit
right to use or redistribute this. Worth deciding before publishing widely.
