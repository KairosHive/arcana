# Roadmap: generative video through ComfyUI

Arcana turns a photograph, a pair of photographs or a clip into a new video
made by a local video model, and files the result like any other output. It
has three modes.

- **Animate** (`video animate`). One picture becomes a video. Loops, masked
  motion and trajectories are controls of this mode.
- **Bridge** (`video bridge`). A start and an end picture become a transition
  conditioned on both. It sits beside the native DINOv2 morph.
- **Transform** (`video transform`). A clip or a section is restyled with
  temporal conditioning while its motion is kept.

Batches are a way of running a mode over many sources. `video animate-batch`
comes first, and the bridge and transform batches reuse it.

ComfyUI runs the models. Arcana keeps the handles, the plans, the jobs, the
provenance, the boards, the film and the search. ComfyUI is reached through its
HTTP and WebSocket API, and the agent reaches every step through Arcana's ops
and the existing MCP `exec` tool.

This document integrates the ComfyUI implementation roadmap of 11 November 2026
with the code on `main` at f60dce3 and with this machine. `VIDEO_ROADMAP.md`
holds the rest of the video work, and section 3.6 there now points here.

---

## 1. Where Arcana stands (11 November 2026)

### The branch

The video work is on `main`. `claude/video-modality` was merged into `main`
at f60dce3 after the Camera (ICM) commit was merged into it, and 761 tests
pass. The rebase and merge part of the original Phase 0 is done. New work
starts from `main`.

### This machine

| Item | State |
|---|---|
| GPU | RTX 4090 Laptop, 16 GB, driver 610.60 |
| Free disk on C: | 72 GB |
| ComfyUI | Installed at `Documents\ComfyUI` with ComfyUI-Manager, ComfyUI-GGUF and x-flux. A second, bare checkout is at `Documents\Github\ComfyUI`, and a venv at `comfy-venv`. Neither runs on 127.0.0.1:8188 today. |
| Video weights in ComfyUI | None. The install holds Flux.1 dev (GGUF), Flux.2 Klein 4B, SD 1.5 with two ControlNets, AnimateDiff v1.4 motion modules, T5-XXL fp8 and Qwen3 4B. |

Every model choice is therefore made for a 16 GB card. The original roadmap
places Wan 2.2 TI2V-5B in the 24 GB class and LTX at 32 GB and more in its
standard configuration. On this card both need fp8 or GGUF weights and
offloading, and the GGUF loader is already installed. A Wan 2.2 14B pair
(high and low noise) with its text encoder and VAE takes 30 GB or more on
disk at fp8, so the survey downloads at most two families at a time within
the 72 GB.

### What the code offers, checked file by file

| Module | What it does today | What generation uses it for |
|---|---|---|
| `ops/registry.py` | Op names match `^[a-z]+( [a-z][a-z-]*){0,2}$`, so `video animate-batch` is valid. Summaries stay under 90 characters, media fields declare `modalities`, and `needs` flags are checked through `FLAG_PROBES`. | The `video` noun is free. A `comfy` flag probe refuses and hides the generation ops while ComfyUI is unreachable, as `gpu` does today. |
| `mcp_server.py` | One `exec` tool runs op lines. | The agent's whole path. No ComfyUI tools are exposed to the agent. |
| `ops/effects.py` | `run_effect(ctx, label, compute, record, background=)` keeps the compute outside the transaction and records inside it. `job list/show/wait/cancel` exist. | A single render is one effect. |
| `jobs.py` | One worker thread, in memory, one job at a time, cooperative cancel through `Progress`. Jobs vanish on restart. | A batch is one job that walks its items. A parent job waiting on child jobs would deadlock this worker. |
| `state.py` | `state.STORE`, transactions with undo per actor, `write_json` (atomic), and the signature behind `/api/state` and the `/api/events` stream. `state/film.json` is the pattern for a durable document the page follows live. | The generation journal is a set of state documents. The page's batch monitor updates from the stream as each item lands. |
| `ledger.py` | `out:` handles with `kind` in (`derived`, `generated`), `op`, `args`, `sources`, `derived_from`. `record` has a fixed set of fields. `alchemy film make`, `colour video` and `alchemy make` already record videos and pictures. | Generated videos are `kind="generated"`, the rule `generate.py` follows so nothing made by a model passes for a photograph. `record` gains a `meta` dict for the generation fields. |
| `ops/archive.py` | `resolve_item` takes dataset handles, `out:` handles and, for the person and the terminal, paths. The agent names items by handle. Its messages speak of pictures. | Source resolution for every mode. |
| `modality.py` | IMAGE, AUDIO, VIDEO, `name@video` datasets, `file#t=a,b` section refs, `of_handle`, `of_path`. | `of_handle` returns None for `out:` handles, so an `out:` video is invisible to modality checks, and an `out:` id made only of digits reads as a dataset called `out`. Phase 1 fixes both. |
| `video.py`, `media.py` | ffmpeg probe, frames, sound, an H.264 proxy for files a browser cannot play, `/film-video` for anything under `output/`. | Validation of what ComfyUI returns, posters, playback. |
| `alchemy/film.py`, `ops/film.py` | Clip keys on the picture lane, the sound mix, carried looks and tracked masks, the open film as an op-driven state document. `is_clip` reads modality from the handle or the path. | `film add out:xxxx` currently treats a generated video as a still, because `is_clip` misses `out:` handles. Phase 1 fixes it through the same `modality` change. |
| `ops/sequence.py` | Cuts clips and stills into one mp4 with a CMX 3600 edit list. The mp4 is left out of the ledger. | Generated clips cut together. Recording the sequence in the ledger joins Phase 4. |
| `ops/board.py` | `board add` and `board move` accept image, audio and video. | The destination of a batch is a board. |
| `ops/sets.py` | Sets of handles, built for photos. | A source of a batch. |
| `activity.py`, `ActivityStrip.svelte` | Each finished op with its actor, Show and Follow, pushed over `/api/events`. | Notices. Durable batch detail comes from the journal. |
| `omni.py`, `omni_worker.py` | EmbeddingGemma 2 in a worker process with its own uv environment, started on demand. | The pattern for the planner's vision-language model, which can be stopped to give its memory back before a render. |
| `item describe`, `captions.py` | CLIP ViT-H scores against Arcana's vocabularies (material, surface, condition) and names colours from the pixels. | The planner's fallback when no vision-language model is installed. It still gives each picture its own words. |
| `capabilities.py` | Features per modality and the tabs that hold them. | Generation features enter the table with their modalities. |

Arcana has no vision-language model and no SQLite store today.

### Decisions already taken that this plan respects

- Diffusion stays off clips in the film (10 November 2026). Animate and Bridge
  make new clips from pictures, which leaves that decision as it is.
  Transform is the mode that would lift it, and it ships after the flicker
  measurement of `VIDEO_ROADMAP.md` 3.9, phase 9.
- `VIDEO_ROADMAP.md` 3.6 planned Wan through diffusers. This plan moves the
  generative video models to ComfyUI. The DINOv2 and FLUX.2 VAE token morphs,
  carried looks, colour on video and the pattern engine stay native.
- LoRA looks are out of the menu pending tests. Generation adds no LoRA
  controls until they return.

---

## 2. Architecture

```text
   page / CLI / agent (MCP exec)
                 |
          ops registry  (video ...)
                 |
   +-------------+---------------+-------------------+
   |             |               |                   |
 sources     Creative         journal            GPU lease
 handles,    Director         state/generation/  Arcana's models
 sets,       (VLM worker,     batches and        out before a
 boards      CLIP fallback)   items              render, ComfyUI's
   |             |               |               out before Arcana's
   +-------------+---------------+-------------------+
                 |
        GenerationRequest  (mode, sources, plan, preset, seed)
                 |
        preset router  ->  workflow manifest + API graph template
                 |
        ComfyUIBackend  upload -> /prompt -> /ws -> /history -> /view
                 |
        ingest  probe -> checksum -> move into output/ -> ledger (tx) -> board (tx)
                 |
        film, sequence, boards, generated@video dataset
```

### Packages

```text
arcana/generation/
  contracts.py       requests, sources, outputs, preset capabilities
  presets.py         preset catalogue, local availability
  director.py        plans for animate, bridge and transform
  prompts.py         plan -> prompt per model family
  journal.py         batches and items as state documents
  ingest.py          validate, publish, record, add to a board
  gpu_lease.py       who holds the card
  backends/base.py   the backend protocol
  backends/comfyui.py
  workflows/<preset>.json        ComfyUI API-format graph
  workflows/<preset>.manifest.json  named bindings, hashes, needs
arcana/ops/video_gen.py   the ops, with no model logic
arcana/plan_worker.py     the VLM in its own process, as omni_worker.py
```

A manifest binds names to nodes (`prompt -> node 6 input text`,
`image -> node 52 input image`, `seed`, `frames`, `width`, `height`,
`output -> node 58`). Op code reads names. Each manifest carries the
template's sha256, the custom nodes and model files it needs, the sizes,
frame counts and fps it accepts, and its tested status. Every generation
stores the graph it actually submitted.

### The ComfyUI backend

- Endpoint from `settings` (`comfy.url`, default `http://127.0.0.1:8188`).
  Loopback only unless the person sets another address.
- Health through `/system_stats`, nodes and model files through
  `/object_info` and `/models/<folder>`. A missing node or file is refused
  with its name and the folder it belongs in.
- Sources go through `/upload/image` under a `arcana/<generation id>` subfolder.
  A video section is first cut to a file with `video.py` at the preset's fps
  and size. A section ref's `#t=` part is stripped before any file access.
- Submission through `/prompt` with a `client_id` of Arcana's own, progress
  through `/ws` filtered by `prompt_id`, with `/history/<prompt_id>` polled
  as the fallback. Outputs are read from the history's output node and
  fetched through `/view`.
- Cancel of a queued prompt goes through `POST /queue {"delete": [id]}`,
  which touches that prompt alone. `POST /interrupt` stops whatever runs, so
  Arcana sends it only when the running prompt is its own, read from
  `/queue`, or when Arcana launched the ComfyUI process itself.
- Arcana accepts graph templates from its own `workflows/` folder. Agents and
  the page choose presets and set bound values.

### The GPU lease

The card has 16 GB, and Arcana's own models (the alchemy pipelines, the
omni worker, CLIP indexing, the planner) and ComfyUI's share it. A small
lease decides who holds it.

- Before a render, Arcana unloads its cached pipelines and stops the omni and
  planner workers.
- After a batch, and before an Arcana op that needs the card, Arcana calls
  ComfyUI's `POST /free {"unload_models": true, "free_memory": true}`.
- Renders run one at a time. A second concurrent render is allowed only after
  a benchmark shows it is faster.

---

## 3. The Creative Director

The person or the agent gives the direction for a whole batch. A planner then
reads each picture and writes a structured plan, and a compiler turns the plan
into the prompt for the chosen model family.

### Global controls

- Direction: `naturalistic`, `surreal`, `abstract`, `custom`.
- Intensity from 0 to 1, shown as almost still, subtle, moderate, strong.
- Camera: `locked` (default), `gentle`, `free`.
- Coherence: each picture on its own, or a shared pace and theme.
- A brief, and things to preserve (colour, framing, identity, architecture).
- Seconds, quality, variations and seed, separate from the plan.
- Loop, as an Animate option with a measured seam.

### A plan

```json
{
  "source": "forest:42",
  "direction": "naturalistic",
  "scene": "A tree reflected in still water",
  "motion_targets": [
    {"subject": "small branches", "action": "subtle irregular swaying"},
    {"subject": "water reflection", "action": "soft slow ripples"}
  ],
  "preserve": ["tree trunk", "bankline", "camera position", "photographic colour"],
  "camera": "locked",
  "intensity": 0.22,
  "prompt": "A quiet static-camera scene. Small leaves sway almost imperceptibly, and gentle ripples travel across the reflected trees. Preserve framing and original light.",
  "planner": "qwen3-vl-8b@<sha>",
  "edited": false
}
```

### How it fits Arcana

- The planner model runs in `plan_worker.py`, in its own uv environment and
  process, as the omni encoder does. Stopping the process frees its memory
  before ComfyUI renders.
- With no planner model installed, the plan is built from `item describe`'s
  CLIP words and `captions.py`'s colour names, through a template per
  direction. Each picture still gets its own subjects, and the review screen
  says the plan came from words.
- Plans are state documents (`state/generation/plans/<id>.json`), so an
  agent's plan shows on the person's review screen live and edits are undone
  like any write. A plan keeps the brief, the scene summary, the planner and
  its version, every edit and the final prompt.
- The planner runs locally. A hosted planner is an option the person turns on
  in settings, and it sends pictures only after that. The MCP `pixels`
  setting governs what reaches the agent, and the local renderer reads
  originals through the backend.
- ComfyUI's own prompt enhancers stay off when a plan supplies the prompt.
- A batch renders after the person approves its plans, unless the person turns
  on auto-approve for the agent.

The Bridge variant reads both endpoints and proposes an event between them.
The Transform variant reads key frames and the clip's motion descriptors
(`motion.py`), and says what changes and what stays.

---

## 4. Phase 0. Survey and benchmark on this machine

The branch audit is done (section 1). What remains is choosing models.

**Work.**

- Survey current open-weight video models with ComfyUI workflows for each
  mode. The original roadmap names Wan 2.2 TI2V-5B, Wan 2.2 I2V-14B and its
  first-last-frame workflow, and LTX for I2V, first-last-frame and IC-LoRA
  video-to-video. `VIDEO_ROADMAP.md` named LTX-2.3. The survey confirms the
  versions current today and their licence terms from the model cards.
- Measure on the 16 GB card, with fp8 or GGUF weights and offloading where
  needed. Peak memory, first load, warm render, size, fps and length.
- A benchmark set of 24 to 30 pictures from the person's own datasets, chosen
  by handle and kept as a set (`set save bench-generative ...`), covering
  vegetation, water, cloud and fog, portraits, architecture and texture, in
  landscape and portrait shapes. Ten to fifteen pairs for Bridge.
- Two directions (naturalistic, surreal) and two to three seeds per picture.
- Measures. First-frame similarity to the source, movement outside the
  motion targets (flow residual, from `motion.py`), flicker with camera
  movement removed (the measure `VIDEO_ROADMAP.md` used for carried looks),
  endpoint similarity for Bridge, loop seam. The person scores aesthetic
  quality, fidelity, naturalism or surprise, continuity and prompt adherence,
  blind to the model.
- Licence matrix for every model and custom node, including the weights'
  terms for users in the EU.
- Choice of the ComfyUI install Arcana uses (section 9, decision 1).

**Delivers.** `docs/generative-video/MODEL_SURVEY.md`,
`MODEL_BENCHMARK.md`, `LICENSES.md`, the benchmark runner, the first
presets, and the download steps the person runs.

**Gate.** One ComfyUI API-format I2V graph renders on this card from a
script. Every default preset has a licence entry and a measured memory peak.

---

## 5. Phases 1 to 8

### Phase 1. The backend and one picture to one video

**Work.**

- `generation/contracts.py`, `presets.py`, the manifest loader and its
  checks, `backends/comfyui.py`, `ingest.py`, `gpu_lease.py`.
- `video animate HANDLE --preset P --seconds S --seed N [--prompt TEXT]`
  as an effect with `needs={"comfy", "gpu"}`, modalities image, and
  `--background`.
- `video presets` and `video status` (reads, no GPU), listing what is
  installed locally.
- Ledger. `record(..., meta={...})`, with `kind="generated"` and `meta`
  holding mode, generation id, preset, model files and hashes, the submitted
  graph's sha256 (the graph saved beside the video), seed, prompt, sources,
  source range, codec, size, fps, length and sound.
- `modality.of_handle` resolves `out:` handles through the ledger's path, so
  `film add`, `board add`, the registry's modality check and `is_clip` treat a
  generated video as a video. Messages in `resolve_item` speak of items.
- A fake ComfyUI server in `tests/` that answers upload, prompt, ws, history,
  view, queue and free.

**Gate.** `video animate japan:12 --preset nature-subtle` returns an `out:`
mp4 that plays in the page, its ledger entry names the graph and the source,
`film add` places it as a clip, and a stopped ComfyUI is refused with its
address. The suite passes without ComfyUI.

### Phase 2. Plans

**Work.** `plan_worker.py` with the chosen VLM, the CLIP fallback,
`video plan HANDLE` and `video plan-batch` (handles, `--set`, `--board`),
`video plan show`, `video plan edit`, plan documents in state, the prompt
compiler per family, settings for local and hosted planners.

**Gate.** Ten varied pictures under one brief give ten distinct plans, a
manual edit reaches the prompt word for word, the fallback works with no VLM,
and a hosted planner receives no picture until the person turns it on.

### Phase 3. Batches (the first release)

**Work.**

- Source snapshot at submission. Handles, order, paths and file fingerprints
  from a list, a set or a board. The page's collection (paths) works for the
  person.
- `video animate-batch` as one job in the existing worker. It walks the items
  and writes each item's state to `state/generation/batches/<id>.json` with
  `write_json`. States `planned`, `approved`, `queued`, `staging`,
  `rendering`, `retrieving`, `validating`, `publishing`, `completed`,
  `failed`, `cancelled`, `recovery_needed`.
- Each item keeps its ComfyUI `prompt_id`, attempts, timestamps, error,
  output path and `out:` handle.
- Ingest per item. Probe and checksum the file in a temporary name, move it
  into `output/generated/video/<date>/`, then in one transaction record the
  ledger entry and add it to the destination board. A stable generation id
  in `meta` makes a second ingest find the first.
- On start, Arcana reads unfinished batches, asks `/history` and `/queue` for
  each `prompt_id`, ingests what finished and marks the rest
  `recovery_needed`. It submits nothing again without the person's retry.
- `video batch show`, `video batch retry` (failed items), `video batch
  cancel` (between items, plus the safe cancel of section 2).
- Undo. Undoing a batch's write takes the ledger entry and the board card
  back and leaves the video file in `output/`, as every derived file does
  today. It cancels nothing in ComfyUI.
- The page. Animate selected and Animate collection in Prompt Search, the
  collection and the Moodboard. A panel with preset, direction, intensity,
  camera, coherence, brief, seconds, variations, loop and destination. A
  review screen with one card per source (thumbnail, targets, prompt,
  preserve, camera, preset, edit, regenerate, exclude). A batch monitor that
  follows the batch document live, plays each video as it lands, retries and
  cancels.

**Gate.** Twenty pictures from a set become twenty planned videos on a new
board, or documented per-item failures. Killing Arcana mid-render and
restarting leaves no duplicate and no orphan. Native Alchemy, the film and the
morphs pass their tests unchanged.

### Phase 4. Generated video everywhere

Most of this exists. Boards take video, `/film-video` serves `output/`, the
proxy plays any codec, the film takes clips once Phase 1 fixes `out:`
modality.

**Work.** Send to Film from the result viewer (`film add`), posters for
`out:` videos, `sequence make` recorded in the ledger, a
`generated@video` dataset over `output/generated/video` refreshed with
`dataset extend` after a batch (sections, motion and the shared space come
with it).

**Gate.** A generated clip opens, goes on a board and into the film, cuts into
a sequence, and is found by `search text` after the refresh. Tests cover
silent video and the section refs.

### Phase 5. Bridge

**Work.** First-last-frame presets from the benchmark, `video bridge A B`,
the bridge batch over consecutive pairs of a set, the Transition Director,
Bridge selected pair when two pictures are selected, endpoint similarity
measured on every render. In the film, a bridge between two stills on the
picture lane enters as a clip key between them, in place of the native
morph for that passage.

**Gate.** Two pictures give a plan, a video recorded with both sources, and a
film where the bridge plays between them. The DINOv2 morph stays available
for the same pair.

### Phase 6. Motion guidance, loops and comparison

**Work.** Region and trajectory controls where a preset supports them, fed
from Arcana's masks (`alchemy mask presets`, SAM, the word masks of 3.4).
Loop presets or a loop post-process, with the seam measured and shown.
Variant grids per source, seed comparison, rejection of a result.

**Gate.** One preset honours a keep-still mask in measured flow. Loops show
their seam score, and controls appear only on presets that support them.

### Phase 7. Transform

This is `VIDEO_ROADMAP.md` phase 9 through ComfyUI.

**Work.** Refresh the V2V survey (LTX IC-LoRA, Wan VACE or what supersedes
them), section staging, `video transform`, references and masks, windowing
with overlaps once measured, sound kept or replaced, the Transformation
Director, Transform clip in the page, and the transform batch.

**Gate.** A section is transformed with its motion kept, and the flicker
measure with the clip's own movement removed stays near the source's (the
carried ControlNet look reached 23.4 against the clip's 8.7, and per-frame
diffusion 66.2). Passing this gate is what lifts the decision to keep
diffusion off clips.

### Phase 8. Agent, polish, packaging

Examples in every op's help and in `orient`, preset comparisons, retry and
variants in one action, model download guidance, diagnostics in `video
status`, privacy and licence pages, a packaged-install smoke test. Arcana
runs fully when ComfyUI is absent.

One pull request per phase, with Phase 3 split into the journal and the page
if it grows.

---

## 6. Ops

```bash
arcana video presets
arcana video status
arcana video plan forest:42 --direction naturalistic --intensity 0.25 --camera locked
arcana video plan-batch --set forest-select --direction surreal --brief "slow dreamlike changes"
arcana video animate forest:42 --preset nature-subtle --seconds 5 --seed 42 --background
arcana video animate-batch --set forest-select --preset nature-subtle --destination "Forest animated" --background
arcana video batch show <batch-id>
arcana video batch retry <batch-id>
arcana video bridge forest:42 forest:56 --preset bridge-dream --seconds 6 --background
arcana video transform rushes@video:12 --preset v2v-preserve-motion --brief "luminous organic structures" --background
arcana job show <job-id>
```

All modes call one `GenerationService.submit(request)` and one ingest.
`video models` from the original roadmap folds into `video presets`, which
lists models per preset with their local state.

---

## 7. Tests

**In the suite, no GPU, no ComfyUI.**

- Registry. Names, summaries, modalities, the `comfy` flag, help on CLI and
  MCP. The example runner lists the generation ops as slow.
- Manifests. Binding, hash, missing node and missing model messages.
- The fake server. Upload, prompt, ws progress, history fallback, view,
  queue delete, free, a dropped connection.
- Sources. Dataset handles, `out:` pictures and videos, sets, boards, section
  refs, unknown handles, paths from the agent refused, path traversal in
  output names.
- Plans. Invalid planner output, missing targets, manual edits kept, prompt
  compilation stable, the CLIP fallback.
- Journal. Progress, cancel, retry, restart after submit, restart after the
  file and before the ledger, ingest twice, the one-worker batch.
- Ledger. Old entries read unchanged, `meta`, two sources for Bridge, undo.
- `modality.of_handle` on `out:` handles, including an id of digits.

**On the GPU, by hand or scheduled.** A real render that plays in Arcana,
two seeds and two directions recorded apart, a 20-picture batch with a killed
ComfyUI and a killed Arcana, a Bridge with endpoint scores, a Transform with
the flicker measure, a loop seam, a Unicode file name, a full disk.

**Release blockers.** A duplicate `out:` after restart, a picture sent to a
hosted service without consent, an original modified, an output Arcana cannot
open, a deadlock, another user's ComfyUI job cancelled, a default model
without a licence entry, a native Alchemy or morph test failing.

---

## 8. Out of scope for now

Text-to-video, graph editing inside Arcana, paid model APIs, cloud or
distributed rendering, downloads the person has not approved, a second asset
database, replacing the native morphs, and V2V of any length before Phase 7.

---

## 9. Decisions to take

| Decision | Recommendation | Settled by |
|---|---|---|
| 1. Which ComfyUI Arcana talks to | Arcana launches the person's chosen install (`Documents\ComfyUI` here) on demand on a port of its own, so its renders are its own and interrupt is safe. Attaching to a running ComfyUI stays possible, with cancel limited to queued prompts. | Phase 0, by how the person uses ComfyUI outside Arcana |
| 2. Journal store | State documents written with `write_json`, as the open film is. SQLite only if a batch outgrows them. | Phase 3, at a 200-item batch |
| 3. Planner model | A local Qwen-VL class model in a worker, with the CLIP words as fallback | Phase 0 survey and Phase 2 gate, by memory beside the renderer |
| 4. Destination of a batch | A new board, with the `generated@video` dataset for search | Phase 3 use |
| 5. Default Animate preset on 16 GB | Decided by the benchmark between a quantized Wan 2.2 and an LTX build that fits | Phase 0 benchmark |
| 6. Bridge inside the film | A clip key between the two stills, chosen per passage | Phase 5 use |

`VIDEO_ROADMAP.md`'s row "Generative video model, Wan 2.1 VACE through
diffusers" is replaced by decision 5 and Phase 7 here.

---

## Limitations

- The model facts in this plan (sizes, memory classes, licences) come from
  the original roadmap and from `VIDEO_ROADMAP.md`. Phase 0 checks each one
  against the model cards and on this card.
- Memory figures for 16 GB are untested. Some candidates may only run with
  offloading slow enough to change the choice of default.
- The 72 GB of free disk holds about two model families at once.
- ComfyUI's routes are read from its documentation. The fake server in the
  tests follows that reading, and the Phase 0 gate checks it against the
  installed version.
- The CLIP fallback names materials, surfaces and colours. It gives the plan
  subjects to move, and it reads less of a scene's structure than a
  vision-language model.
