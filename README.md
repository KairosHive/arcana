<p align="center">
  <img src="arcana/assets/arcana-mark.png" alt="" width="120">
</p>

<h1 align="center">Arcana</h1>

<p align="center">
  Search, explore and tell stories with large image and audio collections —<br>
  by what they look and sound like, not by what they are called.
</p>

---

## What it is

A photo library is usually searchable only by the things written down about it:
filenames, folders, dates, whatever you were disciplined enough to tag. The
pictures themselves are opaque. Ten thousand images means ten thousand things
you can only find by remembering where you put them.

Arcana makes the contents searchable. It runs every file through a
**vision–language model** — CLIP for images, CLAP for audio — which turns each
one into a few hundred numbers describing what is *in* it. Those numbers live in
a shared space with the same model's reading of English, so a sentence and a
photograph can be compared directly.

Two useful things follow.

**You can search by description.** Type *"a quiet street at dusk"* and get the
pictures that look like that, from a collection nobody ever tagged. Type
*"distant thunder"* at a folder of field recordings and get the right sounds.
Nothing was labelled; the model recognises the content.

**The collection gets a shape.** Every file has a position, so the whole library
can be laid out as a map where similar things sit together. Related pictures
form visible clusters, each named automatically, and you navigate by looking
rather than by scrolling a list of filenames.

On top of that sit two tools that use the same representation: a **moodboard**
that finds images by colour palette or texture and transfers the colours of one
onto another, and a **story mode** that assembles a visual narrative from a
sequence of scene descriptions.

It is meant for people with more images than they can remember — photographers,
researchers, artists, anyone with an archive.

## Demo — search mode

![image](https://github.com/user-attachments/assets/dd46c1b2-d8db-4417-b173-a9872e01a927)

## Demo — story mode

![image](https://github.com/user-attachments/assets/9977b27d-501e-49ac-9ebc-60bb8d42a467)

## Install

Three ways, depending on who you are — see **[INSTALLATION.md](INSTALLATION.md)**
for the details.

| | for | command |
|---|---|---|
| **Installer** | using Arcana on Windows | run `Arcana-Setup-<version>.exe` |
| **uv** | developing on it | `uv sync && uv run python -m arcana.arcana` |
| **pip** | adding it to an environment you already have | `pip install .` then `arcana` |

The installer needs nothing else. The other two need Python 3.10+.

There is also `Arcana-Setup-<version>-GPU.exe`, the same application built with
CUDA PyTorch. It is a much larger download and only worth it if you have an
NVIDIA card — search never needs one. It matters most for indexing with the
largest encoder and for Inject Poetry, which is about 29× faster on a GPU. Both
installers share an identity, so installing one over the other swaps the build
in place. If you take the CPU one and Arcana finds a card it cannot use, it says
so rather than leaving you to wonder why things are slow.

## First run

Arcana opens on the **Datasets** tab, which walks you through five steps:

1. **Point at a folder** — everything inside, including subfolders
2. **Check what it found** — media type and name are detected from the folder
3. **Choose the quality** — a stronger encoder understands prompts better but
   takes longer to index
4. **How to name the groups** — how many clusters, and where their names come
   from: Arcana's word list, or your own subfolder names
5. **Extras, then go** — colour palette, style and thumbnails, then Start
   indexing

The encoder downloads itself on first use (605 MB for the fast one). Indexing
10,000 photographs takes about seven minutes on a CPU, three with a GPU.

Once a dataset is built it appears in the **Dataset** menu at the top of every
tab, and the other three tabs come alive.

## Your files stay put

Arcana reads your media where it lives. Nothing is copied, moved or uploaded —
it builds an index beside your collection and leaves the originals alone.
Everything runs on your own machine, including the models.

What Arcana creates goes in one place:

| | |
|---|---|
| Windows | `%LOCALAPPDATA%\Arcana` |
| macOS | `~/Library/Application Support/Arcana` |
| Linux | `~/.local/share/arcana` |
| anywhere | set `ARCANA_DATA_DIR` |

That folder holds the indexes, the downloaded models and anything you save. It
survives uninstalling, and pointing two machines at the same one lets them share
datasets.

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
right to use or redistribute this.

[LICENSING.md](LICENSING.md) records the plan — AGPL-3.0-or-later, third-party
notices, and one blocking item: the built installers currently bundle source
from a repository that has no licence at all. Do not distribute a build until
that is resolved. [COLOR_TRANSFER_PLAN.md](COLOR_TRANSFER_PLAN.md) is the rework
that resolves it permanently.
