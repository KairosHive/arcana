# Execution plan: generative video through ComfyUI

This is the working brief for the agent that builds `GENERATIVE_VIDEO_ROADMAP.md`.
The roadmap holds the design and the reasons. This file holds the order of
work, the files, the checks, and the points where the person decides.

## 0. Before writing code

**Read.**

1. `GENERATIVE_VIDEO_ROADMAP.md`, whole. Section 1 lists what the code offers
   with the defects already found. Section 2 holds the architecture, the GPU
   tiers and the lease.
2. `VIDEO_ROADMAP.md`, sections 3.4, 3.6, 3.9 and the Status part on
   carried looks (the flicker measure used in Phase 7).
3. The person's writing rules in their global `CLAUDE.md` apply to every
   document and reply. No em dashes, no colons or semicolons inside
   sentences, affirmative phrasing, facts in every sentence, limitations
   once in a section called Limitations.

**Start from `main`.** Make a branch per phase (`generative/phase-0`,
`generative/phase-1`, ...). Run the suite once before changing anything.

```bash
uv run pytest -q -p no:cacheprovider --deselect tests/test_board_page.py::test_a_real_print_follows_the_pages_own_print_css
```

It passed with 761 tests on 11 November 2026.

**Targets.** A laptop RTX 4090 with 16 GB (sm_89, the development machine)
and a desktop RTX 3090 with 24 GB (sm_86, no fp8 compute). The software
adapts to any NVIDIA card through the tiers. Arcana without a usable GPU or
without ComfyUI keeps working, and the generation ops are refused there.

## 1. Rules of this repository

- **Ops.** One `@op(name, kind, Args, Result, doc=, examples=, modalities=,
  needs=)` per op (`arcana/ops/registry.py`). Names match
  `^[a-z]+( [a-z][a-z-]*){0,2}$` (line 34). The first sentence of `doc` is
  under 90 characters. Any argument naming media declares `modalities`. A
  new module is imported in `arcana/ops/__init__.py`. Each op has examples,
  and examples that need a GPU, ComfyUI or a model go in the `SLOW` set of
  `tests/test_ops_registry.py`.
- **Flags.** `needs` names flags from `FLAG_PROBES` (`registry.py:138`).
  Add a `comfy` probe that answers from a cached health check and never
  blocks longer than one second.
- **Writes.** A write calls `ctx.tx.touch(path)` before it writes, so undo per
  actor works. Durable documents go through `state.write_json`. The ledger
  is written with `ledger.record(tx, ...)` inside the transaction.
- **Effects.** Long work uses `run_effect(ctx, label, compute, record,
  background=)` (`ops/effects.py:40`). Compute runs outside the transaction,
  record runs inside. `jobs.py` has one in-memory worker, so a job never
  waits on another job.
- **Agent.** The agent names media by handle (`dataset:key`, `out:id`).
  `archive.resolve_item` (`ops/archive.py:391`) refuses its paths. The agent
  reaches every feature through these ops and the MCP `exec` tool.
- **Page.** Svelte 5 in `frontend/`. `rpc(op, args)` posts to `/api/rpc`.
  State the page follows lives on the server and moves the state signature,
  which `/api/events` pushes. `frontend/src/tabs/alchemy/film.svelte.ts`
  is the pattern for a page view of a server document.
- **Checking the page.** Use the `arcana-sandbox` launch config (port 8061,
  `tests/sandbox_app.py`) with `ARCANA_DATA_DIR` pointing at a temporary
  folder. The person's own Arcana runs on port 8051 and stays untouched.
  `preview_stop` can leave the child Python running. Stop it by its command
  line before starting again.
- **Shell.** In a worktree session, compound shell commands are refused.
  Run commands one at a time.
- **Downloads.** Each model download is approved by the person first, with
  its name, source, size and licence. Free disk on the laptop was 72 GB.
- **ComfyUI.** The person's install is `C:\Users\skite\Documents\ComfyUI`
  (Manager, GGUF and x-flux nodes, no video weights). Arcana launches it on
  a port of its own (8189) while it is free, and leaves any ComfyUI on 8188
  alone. Custom nodes are installed only with the person's approval.
- **Commits.** End messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit and push
  when the person asks. One pull request per phase.

## 2. Phase 0. Survey and benchmark

**Steps.**

1. Read the current model cards and ComfyUI workflow pages for image-to-video,
   first-last-frame and video-to-video. Start from Wan 2.2 (TI2V-5B, I2V-14B,
   FLF2V), LTX (I2V, first-last-frame, IC-LoRA) and anything newer that runs
   locally in ComfyUI. Record version, licence, EU terms, disk size per
   quantization, and the custom nodes each workflow needs.
2. Do the same for local vision-language models for the planner (Qwen-VL
   class and current competitors). Record size, licence and whether it emits
   JSON reliably.
3. Write `docs/generative-video/MODEL_SURVEY.md` and `LICENSES.md`.
4. **Stop and ask the person.** Propose a shortlist of at most two video
   families and two planner models, with the downloads each needs and their
   total size. Download only what the person approves.
5. Write `scripts/bench_generative.py`. It starts ComfyUI, submits each
   candidate graph for each benchmark picture, seed and direction, and
   records peak VRAM (from `/system_stats` and nvidia-smi), first-load time,
   warm render time, output size, fps and length. It writes a JSON and a
   folder of videos. It takes `--tier` and runs unchanged on the 3090.
6. Build the benchmark set with the person. Ask them to pick or approve 24 to
   30 pictures from their datasets across vegetation, water, cloud and fog,
   portraits, architecture and texture, plus 10 to 15 pairs for Bridge. Save
   them with `set save bench-generative ...`.
7. On the laptop, measure each candidate in its best configuration for 16 GB.
   Try GGUF levels, fp8, step-distilled LoRAs (4 to 8 steps), faster
   attention kernels, block swapping and torch.compile, one at a time against
   the same model without them. Keep what improves time without a visible
   loss in quality.
8. Compute the automatic measures. First-frame similarity to the source
   (DINOv2 cosine), flow residual outside the motion targets (`motion.py`),
   flicker with camera motion removed (the measure in `VIDEO_ROADMAP.md`'s
   carried-looks status), endpoint similarity for pairs, loop seam.
9. Make a blind review page for the person. Videos side by side with model
   names hidden, scored on aesthetic quality, fidelity, naturalism or
   surprise, continuity and prompt adherence. **Stop and ask the person to
   score.**
10. Run the planner candidates on twenty pictures and show the person the
    plans side by side.
11. Write `docs/generative-video/MODEL_BENCHMARK.md` with the tables, the
    chosen 16 GB variant per preset, the drafted 24 GB variant per preset,
    the chosen planner per tier, and the command for the desktop run.

**Done when** one API-format I2V graph renders from the script on the laptop,
each default preset has a licence entry and a measured 16 GB peak, the 24 GB
variants exist as manifests marked untested, and the planner per tier is
chosen.

## 3. Phase 1. Backend and one picture to one video

**Files.**

- `arcana/generation/contracts.py`. `Source`, `CreativeSpec`,
  `GenerationRequest`, `GeneratedAsset`, with roles and counts checked per
  mode (animate one image, bridge two images, transform one video).
- `arcana/generation/presets.py`. Loads `workflows/*.manifest.json`, checks
  the template hash, picks the variant for the card (roadmap section 2, GPU
  tiers), reports local availability through `/object_info` and
  `/models/<folder>`.
- `arcana/generation/backends/base.py` and `backends/comfyui.py`. Health,
  launch on 8189, upload, prompt, ws progress with history polling as the
  fallback, view, queue delete, owned interrupt, free.
- `arcana/generation/gpu_lease.py`. Frees Arcana's cached pipelines and stops
  the omni and planner workers before a render. Calls ComfyUI's `/free`
  before an Arcana op that needs the card.
- `arcana/generation/ingest.py`. Probe with `video.py`, checksum, move into
  `output/generated/video/<date>/`, ledger record, optional board add.
- `arcana/ops/video_gen.py`. `video presets`, `video status`,
  `video animate`.
- `arcana/ledger.py`. `record(..., meta=None)`, stored as `entry["meta"]`.
  Old entries read unchanged.
- `arcana/modality.py`. `of_handle` (line 117) resolves `out:` handles to
  the modality of the ledger entry's path, and it checks the `out:` prefix
  before reading a dataset id, so an `out:` id made of digits stays a ledger
  handle. This also fixes
  `film.is_clip` (`alchemy/film.py:128`) for generated videos.
- `arcana/ops/archive.py`. `resolve_item` messages speak of items.
- `arcana/capabilities.py`. The generation features with their modalities.
- `tests/fake_comfy.py` and `tests/test_generation_backend.py`. A small
  server answering the routes above, a dropped connection and a missing
  node.

**Done when** `video animate <handle> --preset <p>` returns an `out:` mp4
that plays in the sandbox page, the ledger entry holds the meta of roadmap
Phase 1, `film add out:<id>` places it as a clip, a stopped ComfyUI is
refused with its address, and the suite passes on a machine without ComfyUI.

## 4. Phase 2. The Creative Director

**Files.** `arcana/plan_worker.py` (the vision-language model in its own uv
environment and process, as `omni_worker.py`), `arcana/generation/director.py`,
`arcana/generation/prompts.py`, plan documents under
`state/generation/plans/`, ops `video plan`, `video plan-batch`,
`video plan show`, `video plan edit`, settings for the planner variant and an
opt-in for a hosted planner. The fallback builds a plan from
`item describe` and `captions.py`.

**Done when** ten varied pictures under one brief give ten distinct plans
from the vision-language model, a manual edit reaches the prompt word for
word, invalid model output is retried once and then falls back, the fallback
runs with the model absent, and a hosted planner receives nothing until the
person turns it on.

## 5. Phase 3. Batches

Follow roadmap Phase 3 point by point. The journal lives under
`state/generation/batches/`. The page work goes in a new Generate panel
reached from Prompt Search, the collection and the Moodboard, with the review
screen and the batch monitor following the batch document live.

**Done when** twenty pictures from a set become twenty videos on a new board
(or documented per-item failures), killing Arcana and ComfyUI mid-batch and
restarting leaves no duplicate and no orphan, and native Alchemy, film and
morph tests pass unchanged. Check the page in the sandbox and show the person
a screenshot of the review screen and the monitor.

## 6. Phases 4 to 8

Follow the roadmap for each. Phase 4 adds Send to Film, posters, the
sequence ledger record and the `generated@video` dataset. Phase 5 adds
Bridge. Phase 6 adds masks, loops and comparison. Phase 7 adds Transform and
must pass the flicker gate before diffusion on clips returns. Phase 8 is
polish and packaging.

**Desktop check.** At the end of Phase 1 and of Phase 3, ask the person to
run on the 3090:

```bash
uv run python scripts/bench_generative.py --tier 24 --quick
```

and to send back the JSON. Mark the 24 GB variants tested from it.

## 7. Reporting

At the end of each phase, add to a Status section at the end of
`GENERATIVE_VIDEO_ROADMAP.md` (create it in Phase 0)
with what was built and measured, in the person's writing rules, and report
to the person in a few plain sentences with the numbers.
