# Agent mode: an MCP server for Arcana

Not yet executed. Goal: ask an agent for a moodboard on a theme and get one
back. It should be sourced from your own archive across several angles,
assembled, unified with Arcana's own colour and style tools, and laid out. Arcana
itself takes on no API key, no language model and no network dependency.

**Verified against the repo 20 October 2026** (at `b2fa984`). Revised the same
day: COLOR_TRANSFER_PLAN's B0 and B1 moved in as P4, and the three open
decisions settled (see the end).

**Superseded on architecture, 7 November 2026,** by [OPS_PLAN.md](OPS_PLAN.md).
The design decisions here stay in force (A5 to A8, board version 2, the skills,
the eval briefs, P4's registry). The tool list, with one MCP tool per
capability, is replaced by one op vocabulary reached through one `exec` tool,
one `arcana` command and the web UI alike. Read OPS_PLAN.md first.

---

## Verdict

Worth doing, and Arcana is unusually well placed for it.

A moodboard is two jobs. The first is **breadth**: turn a theme into the many
visual angles it can be approached from, then find each one. The second is
**judgment**: cut sixty candidates to fifteen that hold together. Arcana covers
the retrieval part of breadth very well and nothing else. Today the person
breaks the theme into angles in their head, runs one query at a time, and
curates by hand. A language model is good at exactly the missing parts and has
no access to the pictures. MCP connects the two.

It is cheaper than it sounds:

- every image is already a vector, so search by example, diversity, dedupe and
  "how strong is this match" are arithmetic on data already on disk, with no new
  models
- palette, style, colour transfer, style transfer and sd-turbo already exist as
  Python functions
- boards already persist as JSON in the data dir, where both the app and a
  separate process can see them

The MCP server itself is a thin file and costs little. The real cost is what the
server needs underneath it:

1. **A headless core.** Prompt search, the marks filter and dedupe live inside
   `arcana.py`, which builds the Dash app on import (`app = dash.Dash(` at
   `arcana.py:1232`).
2. **A richer board.** Today a board is a flat list of paths. It has no
   sections, roles, notes or palette, and no record of how a derived image was
   made.
3. **Evidence.** An agent can't curate without knowing how good a match is, and
   `search()` returns a distance that nothing reads (ROADMAP 10).

The real risks are about the product, not the code: the privacy promise in the
README, and boards that are competent but generic. Both are covered below.

---

## Why MCP, and not an agent inside the app

| shape | what it means | verdict |
|---|---|---|
| **MCP server** | Arcana exposes tools; an MCP client (Claude Code, Claude Desktop, or a client for a local model) runs the agent | **first** |
| Agent in Dash, Anthropic API | chat panel; Arcana runs its own tool loop against the API | later, on the same tools |
| Agent in Dash, Claude Agent SDK | same, with Claude Code's harness as a library | later, on the same tools |
| Fixed pipeline, no agent | theme → N queries → top-k → grid | the baseline to beat; it has no judgment |

MCP comes first for three reasons:

- **Arcana stays model-free.** There is no key, no billing and no SDK pinned
  into the installer. The ROADMAP's case against cloud GPUs applies just as much
  to cloud language models. MCP leaves the choice with the person, including a
  fully local model through an MCP-capable local client.
- **The host already exists.** Claude Code and Claude Desktop already handle the
  loop, context, approvals and skills.
- **Nothing is wasted later.** An in-app chat panel would call the same tools,
  so the server becomes its backend.

---

## What there is to wrap

| capability | where | agent-ready? |
|---|---|---|
| prompt search (CLIP / CLAP text) | `arcana.py:1115 search` | ❌ inside the Dash module |
| encoder matched to the index | `arcana.py:1058 text_model_for_dim`, `:1083 load_clip_text` | ❌ same |
| stars / colour / date filter | `arcana.py:665 marks_mask` | ❌ same |
| near-duplicate grouping | `arcana.py:298 _cosine_group` | ❌ same |
| dataset loading | `arcana.py:610 load_data`, `:629 load_index` | ❌ same |
| sd-turbo generation | `arcana.py:4680 _poetry_job` | ❌ same |
| palette search | `db.py:1128 search_by_palette` | ✅ |
| style search | `db.py:1223 search_by_style`, `:1333 search_combined` | ✅ |
| cluster names | `label` / `cluster_id` in the latent frame | ✅ |
| a picture's palette, with lightness / saturation filters | `refselect.py:134 palette_strip` | ✅ |
| colour transfer | `color_transfer.py:319 transfer_colors`; the method choice is in the callback at `arcana.py:4510` | ⚠️ the function is ready, but method dispatch isn't (P4) |
| style transfer, ControlNet | `style_transfer.py:352 transfer`, `:491 controlnet_transfer` | ✅ |
| CLIP description of a picture | `style_transfer.py:168 describe` | ✅ |
| boards | `boards.py` | ⚠️ headless, but the schema is flat |
| background work with progress | `jobs.py` `MANAGER` | ✅ |
| content fingerprint | `bundle.py:65 fingerprint` | ✅ |
| search by example image | none | ❌ missing (ROADMAP 2) |
| negative phrases | none | ❌ missing (ROADMAP 2) |
| calibrated match strength | none | ❌ missing (ROADMAP 10) |
| a board rendered as one image | none | ❌ missing |

---

## Architecture

```
   Claude Code / Claude Desktop / any MCP client
         │  stdio, JSON-RPC
   ┌─────┴───────────────────────────────────────────┐
   │ arcana/mcp_server.py   tool schemas, contact     │
   │                        sheets, id ↔ path, guards │
   └─────┬───────────────────────────────────────────┘
         │
   ┌─────┴───────────────────────────────────────────┐
   │ arcana/core.py   datasets, search, filters,      │ ◄── arcana.py (Dash)
   │                  dedupe, more-like, scoring      │     imports these instead
   └─────┬───────────────────────────────────────────┘     of defining them
         │
   db.py · palette.py · style.py · refselect.py · color_transfer.py
   style_transfer.py · boards.py · jobs.py · compose.py (new)

   shared state is the data dir:   boards/*.json    output/
```

### A1. Headless core first

Move `search`, `text_model_for_dim`, `load_clip_text`, `load_clap`,
`marks_mask`, `_cosine_group`, `load_data` and `load_index` into
`arcana/core.py`, with a small per-process cache of loaded datasets.
`arcana.py` imports them back.

`load_data` and `load_index` also call `register_dataset_roots`, which feeds the
media-serving allowlist. That is a UI concern, so it stays in a thin wrapper in
`arcana.py`.

This is a pure refactor that pays for itself: those functions become testable
without Dash. Everything else depends on it.

### A2. stdio, in its own process

The server is an `arcana-mcp` entry point, and the client starts and stops it.
It works whether or not the app is open, and a crash in one doesn't take down
the other.

The cost: with both running, the CLIP text tower is loaded twice. Measure it
before optimising. The alternative is to serve the tools over streamable HTTP
from inside the app process, sharing models and the job queue; that is P6.

### A3. The data dir is the bus

The agent writes boards through `boards.py`, and the app notices. For a live
view, a `dcc.Interval` compares the boards folder's mtimes every couple of
seconds and shows a toast: *"Monsoon melancholy" updated by the agent — open*.
That needs no sockets and no IPC.

### A4. One GPU job at a time, across processes

`jobs.py` serialises work within one process, but two processes could put
sd-turbo and ModFlows on the card at the same time. A lock file in the data dir
(`gpu.lock`) around GPU-heavy calls, held by whichever process is running,
prevents the OOM. CPU-only work needs no lock.

### A5. Show the model pixels on a budget

An image costs roughly *width × height ÷ 750* tokens, whether it holds one
picture or forty. So what matters is how big each picture is shown, set by
purpose:

| purpose | size | tokens per picture |
|---|---|---|
| survey a cluster, scan candidates | 160 px tile | ~35 |
| compare finalists | 320 px tile | ~135 |
| inspect one picture | 1024 px | ~1,400 |
| check a rendered board | 1568 px long edge | ~2,200–3,300 |

Results come back as **contact sheets**: one JPEG of numbered tiles plus a
compact text table, for example
`#7  japan:1423  p 0.92  ★★★★  [Forest]  DSC04481.jpg`. The numbers make
references unambiguous, a grid is what curation needs, and a hundred candidates
cost about 3,500 tokens of image.

### A6. Handles, not paths

Tools speak `dataset:key` (for example `japan:1423`) and the server maps that to
a path. Handles are shorter and stable within an index. Every path a tool touches
still goes through the existing `paths.is_within` media-root guard.

### A7. Never touch an original

Every write goes into the data dir (`boards/`, `output/`). Tools carry MCP
annotations: `readOnlyHint` on everything that only reads. The server has no
delete tool at all.

The agent can edit only boards it authored (`"author": "agent"`). To change a
board the person made, it saves a copy.

### A8. Privacy is a setting, not a footnote

The README promises *"Nothing is copied, moved or uploaded."* With a cloud
model, the theme, the queries, the text tables and the contact-sheet thumbnails
go to the model provider. Search, palettes, transfers and the originals stay on
the machine. There are two modes:

- **blind** (the default): no images leave the machine. Tools return cluster
  names, palettes as hex, match strengths and `describe()` text. Curation gets
  worse, but the promise holds. While blind, every result says so in one line,
  so the person is never unsure which mode is on.
- **pixels:** thumbnails at the sizes in A5.

The setting lives in `agent.json` in the data dir (`{"pixels": true}`). It is
per machine, never committed, and only the person changes it: by editing that
file, or from a toggle in Arcana (P2). **No tool can change it**, so the model
can't decide for itself to start sending pictures. Pointing an MCP-capable local
client at the server with pixels on keeps everything on the machine; the server
doesn't need to know which client it is.

The README's *Your files stay put* section gets a paragraph saying exactly this.

---

## The board, version 2

A board is the deliverable, so it has to hold everything a moodboard is. Today a
board is `{name, items: [path, …], reference, transfer, dataset}`. Version 2:

```jsonc
{
  "schema": 2,
  "name": "Monsoon melancholy",
  "author": "agent",                          // or "person"; see A7
  "brief": "the words that were asked for",
  "concept": "one paragraph: what the board argues",
  "palette": [{"hex": "#1f2a36", "weight": 0.22, "role": "ground"}],
  "sections": [
    {"id": "s1", "title": "Water as weight", "rationale": "…",
     "queries": ["heavy rain on a dark street", "-umbrella"]}
  ],
  "items": ["D:/photos/…/IMG_4412.jpg", "…/output/color_transfer/….png"],
  "details": {
    "D:/photos/…/IMG_4412.jpg": {
      "fp": "9c1e…",                                        // bundle.fingerprint
      "section": "s1", "role": "hero",                      // hero | support | texture | detail | counterpoint
      "note": "why it is here",
      "source": {"kind": "search", "dataset": "japan", "query": "…", "match": 0.91}},
    "…/output/color_transfer/….png": {
      "derived_from": "D:/photos/…/IMG_2210.jpg",
      "op": {"kind": "color_transfer", "method": "modflows", "reference": "…", "strength": 0.5},
      "section": "s2", "role": "support"}
  },
  "layout": {"kind": "editorial", "order": ["…"]}
}
```

- **Backward compatible in both directions.** `items` stays exactly what it was
  in version 1, a list of paths, and everything new about an item goes in
  `details`, keyed by path. That matters because two Arcanas can share one data
  dir: the installed app and a development checkout on the same machine, or two
  machines on one `ARCANA_DATA_DIR`. An older version reading a version 2 board
  sees the paths it expects and ignores the rest. Turning items into objects
  would have crashed it on every board the agent wrote.
- **The app keeps what it doesn't show.** Re-saving a board from the Moodboard
  tab keeps the `details` of every picture still on it, and every top-level
  field it doesn't set, so a person tidying an agent's board doesn't wipe the
  agent's notes.
- **Derived images are always marked.** A colour-transferred or generated
  picture must never pass for a photograph from the archive. The UI badges
  anything with `derived_from` or `"kind": "generated"`.
- **`fp` brings ROADMAP 7 forward.** A content fingerprint on each item is what
  lets a board survive a folder move.

---

## Tools

A small set of coarse tools, each returning something the agent can decide on.
Fourteen in all.

**Orient**

| tool | does |
|---|---|
| `list_datasets()` | for each dataset: name, modality, item count, encoder, which features exist (palette, style, marks), date range |
| `survey(dataset)` | cluster names and sizes, plus a contact sheet of exemplars from each cluster, so the agent sees what the archive holds before planning angles |

**Gather**

| tool | does |
|---|---|
| `search(dataset, facets, filters, k, diversity)` | several facets in one call. Each facet is a set of phrasings plus negatives (`"fog"`, `"mist over water"`, `-"people"`), scored as a weighted sum of cosines (ROADMAP 2). MMR diversity over the image vectors stops twenty versions of one photo filling the results. Each picture gets p, CLIP's zero-shot confidence that it shows the phrases rather than something generic, and a facet with nothing good is flagged *weak* (see P1 for why not a percentile). |
| `more_like(ids, mode, k)` | search by example. `semantic` reads the stored vectors, so it needs no vision model and no GPU. `palette`, `style` and `combined` call the existing `db.search_by_*`. |
| `find_by_palette(hex, k)` | the agent picks a palette ("ink blue, rust, bone") and finds pictures in it. The swatches are rendered as an image and passed to `search_by_palette` as the query. |
| `inspect(ids)` | one or a few pictures shown large, with palette, marks, cluster and `describe()` tags |

**Curate**

| tool | does |
|---|---|
| `board_open(name, brief)` | create a board, or open one the agent authored |
| `board_edit(board, ops)` | one tool for a list of operations: add, remove, move, annotate, set sections, set palette, set concept |
| `board_view(board)` | the board as a contact sheet grouped by section, plus metrics: diversity (mean pairwise CLIP distance), near-duplicate pairs, palette spread in LAB, balance across sections, share of derived pictures |

**Complete** (these can run long, so each returns a job id)

| tool | does |
|---|---|
| `harmonize(board, items, toward, method, strength)` | colour-transfers the chosen items toward a reference item or the board palette; the results join the board as derived items |
| `restyle(source, target, method, …)` | style transfer or ControlNet |
| `generate(prompt, init, strength)` | sd-turbo, only for an angle that `search` flagged *weak*; the result is always marked `"kind": "generated"`. The server refuses a third generated picture on one board, and refuses to make one the hero. |
| `job(id)` | status, progress and result of a running job |

**Present**

| tool | does |
|---|---|
| `render_board(board, layout, width)` | a PIL compositor (`compose.py`, new): editorial, masonry or grid layout, with a palette strip, section titles and notes. Generated pictures carry a small mark. Writes a PNG to `output/boards/` and returns a preview so the agent can look at what it made and revise. |

The server also exposes an MCP **prompt**, `moodboard(theme, dataset)`, so
clients without skills still get the workflow.

Fast tools should finish in seconds. Claude Code waits only a bounded time for
an MCP call (`MCP_TOOL_TIMEOUT`), and an sd-turbo scene takes 37 s on a CPU, so
anything that might be slow goes through `jobs.py`.

The rough shape of one tool:

```python
from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

mcp = FastMCP("arcana")

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search(dataset: str, facets: list[Facet], k: int = 24,
           diversity: float = 0.3, filters: Filters | None = None):
    """Written for the model: what a facet is, and how CLIP reads a phrase."""
    hits = core.search_facets(dataset, facets, k=k, diversity=diversity, filters=filters)
    return [Image(data=sheets.contact_sheet(hits, tile=160), format="jpeg"),
            sheets.table(hits)]
```

Tool descriptions are prompts. They teach the model how CLIP reads a query, and
that drives half the quality of the results.

---

## Skills: where the design knowledge lives

The server knows *how* to do things; the skills know *what good looks like*.
They're packaged as a Claude Code plugin inside the repo: a plugin manifest, the
MCP config and the skill folders. Installing that one plugin brings both the
server and the knowledge.

### `moodboard`

The workflow, with budgets:

1. Restate the brief as a one-paragraph concept. If it's ambiguous, ask one
   question, then go.
2. `survey` the dataset, so the plan is based on what's actually there.
3. Break the theme into 4–7 angles, using these lenses: subject · place · light
   and atmosphere · material and texture · colour · scale (wide / mid / detail)
   · human presence or absence · motion · symbol · time (hour, season, era).
   Add one **counterpoint**, which sharpens the theme by contrast. Post the
   angles in one short message and carry on unless told otherwise.
4. Translate each angle into what CLIP can see: replace the abstract with a
   visual equivalent ("melancholy" → "empty bench in rain, overcast, muted").
   Use two or three phrasings per angle, negatives for subjects that keep
   intruding, and photographic vocabulary (long exposure, overhead, macro,
   backlit).
5. Gather wide: 30–60 candidates. Say plainly where the archive is thin.
6. Cut to 12–24: one picture per idea, no near-duplicates, a mix of scales, and
   one hero that nothing else competes with.
7. Take the palette from the selection rather than imposing one on it.
8. Harmonize sparingly: outliers only, never the hero, at most a third of the
   board, strength 0.3–0.6, and keep the originals. Generate only to fill an
   angle the archive came back weak on, at most two per board, never the hero,
   and say so in the hand-back.
9. Render the board, look at it, critique it against the metrics and the
   `board-design` skill, and revise once.
10. Hand back the rendered board, its name in Arcana, and what was missing from
    the archive.

### `board-design`

High-level design:

- hierarchy: one hero, a few supports, textures and details for rhythm
- negative space and density
- value structure before hue (the squint test)
- palette proportion: dominant, secondary, accent
- temperature balance and contrast of scale
- the path the eye takes, and how many pictures is too many

It also covers how to critique a rendered board in those terms.

### `harmonize`

Which transfer method fits which job: LAB for quick, safe nudges, MKL when LAB
leaves a colour cast, IDT to match a whole distribution, ModFlows for quality,
texture transfer for grain, and ControlNet only when you mean it. When not to
transfer at all. The method list itself comes from the P4 registry, so the skill
explains judgment, not inventory.

### Evaluating the skills

Measure the skills the way the ROADMAP measures everything else. Write five
fixed briefs against `japan` and `abstract`, and run each with and without the
skills. Score angle coverage, the board metrics above, and a person's ranking.
The `skill-creator` skill has a harness for this.

---

## Phases

Each phase ends with something usable.

### P0: Headless core and board version 2 (about 2 days)

- `arcana/core.py` as in A1; `arcana.py` imports from it, and
  `register_dataset_roots` stays in a UI-side wrapper
- board schema 2 in `boards.py`, with a read path for version 1, and `fp` from
  `bundle.fingerprint`
- tests: `tests/test_core.py` (search against a tiny fixture index, the marks
  filter, dedupe); `tests/test_boards.py` extended to cover version 1 → 2
- **done when** importing `arcana.core` leaves `dash` out of `sys.modules`, and
  the app behaves exactly as before

**Done, 20 October 2026.** Importing `arcana.core` loads neither dash nor torch
nor transformers (tested in a fresh interpreter). The app was launched from the
checkout and a prompt search on japan ran through the moved code with no errors.
All 327 tests pass. One of them loads a full CLIP model, and on a machine
already short of memory it crashed until memory was freed. It only touches
`db.py`. What building it changed:

- Boards keep `items` as plain paths and add `details` beside them (see *The
  board, version 2*), because the installed app shares the data dir.
- Fingerprinting is opt-in (`fingerprints=True`). It reads every file whole,
  and the Moodboard tab's save should stay instant.
- `core.open_dataset()` caches by file mtime for the server. The app's own
  `load_data` stays uncached, so its behaviour is unchanged.

### P1: Read-only server (about 2 days)

- `mcp` added as an optional extra (`[agent]`) in `pyproject.toml`, with an
  `arcana-mcp` script
- tools: `list_datasets`, `survey`, `search` (facets, negatives, filters, MMR,
  calibrated match strength), `more_like`, `find_by_palette`, `inspect`
- contact-sheet packer, handle ↔ path mapping, blind and pixels modes (A8)
- `.mcp.json` at the repo root, so Claude Code in this repo offers the server.
  It runs `uv run --extra agent arcana-mcp`, so the optional dependency is
  there without becoming part of every install.
- **done when**, in Claude Code, *"what does my japan dataset have about water,
  apart from the sea?"* returns a contact sheet with match strengths, and the
  MCP Inspector shows clean schemas

**Built, 20 October 2026; the acceptance check in Claude Code is still to do.**
The server runs from `.mcp.json` and answers over a real stdio pipe. Tested: all
six tools, a tool that prints mid-call, and errors that reach the model in
words. On the real japan and midjourney datasets it lists, surveys and searches.
Contact sheets are untested on real pictures, because every dataset's files
have moved off this machine (Relocate fixes that). What building it changed:

- **MCP SDK 2.x.** `FastMCP` is now `MCPServer`. The stdio transport keeps fd 1
  off the wire by itself, which beats routing print() by hand because it also
  catches native writes. Expected failures must be raised as `ToolError`, or
  the model sees only "Error executing tool search".
- **Match strength is CLIP zero-shot, not a z-score.** A z-score against the
  dataset punishes common subjects. On the 9,359-image set, "a spreadsheet of
  numbers" got the highest z of every query tried. Weighing the query against
  a bank of generic descriptions separates cleanly instead: subjects present
  scored 0.52–1.00, absent ones 0.00–0.16 (`core.confidence`). A weak facet
  still points out a single picture that scores well, like the one cat in an
  archive of temples.
- **Moved files are reported.** `list_datasets` says when a dataset's files
  are gone, and each result line marks a missing file.
- **Datasets without a map are offered.** An index alone is enough to search,
  so midjourney (9,359 images, never laid out) is searchable here although the
  app hides it.
- **Audio datasets are named `name@audio`,** so a picture handle stays
  `dataset:key`. Audio has no distractor bank yet, so its facets come back
  "unrated", with z instead of p.

### P2: Boards through the agent (about 2 days)

- `board_open`, `board_edit`, and `board_view` with metrics
- in the app: a boards-folder watcher and toast; sections, roles and notes shown
  when a board loads; a badge on derived pictures; the pixels toggle (A8)
- **done when** a board the agent builds shows up in the Moodboard tab within
  seconds, and editing it there keeps the agent's notes

### P3: Render and the moodboard skill (about 3 days)

- `compose.py` and `render_board`
- `moodboard` and `board-design` skills, the `moodboard` MCP prompt, and the
  plugin packaging
- the five-brief eval set, then iterate
- **done when** *"moodboard for 'monsoon melancholy' from japan"* returns a
  rendered board with at least 4 angles, no near-duplicate pairs, a palette
  strip and a reason for each section, and it's a board you'd keep

### P4: Colour transfer as one table, plus the classical tier (about 2 days)

Moved here from COLOR_TRANSFER_PLAN (where it was B0 and B1), for two reasons.
P5's `harmonize` tool needs it. And it fills the gap the CPU build has today:
two methods, only one of them instant. It touches only the colour-transfer
code, so it can run alongside P1–P3 once P0 has landed.

**Where it stands.** There are two methods, ModFlows and LAB, and their names
are hardcoded: the dropdown at `arcana.py:2156`, then `method == "lab"`
branches at `4089` (which controls to grey out), `4284` (the quality note),
`4523–4525` (availability), `4569` (dispatch) and `4604` (filename suffix). The
agent would be a third place to copy that chain into.

**The registry.** `arcana/transfer_methods.py`, following `models.py`, which is
already the house pattern: a frozen dataclass, a tuple ordered cheapest first,
`BY_ID`, `get()`, `available()`, `default_for()`.

```python
@dataclass(frozen=True)
class TransferMethod:
    id: str
    label: str              # what a person sees
    licence: str            # SPDX or short name; THIRD_PARTY.md is generated from it
    needs_download_mb: int  # 0 means nothing to fetch
    needs_gpu: bool         # False means usable on the CPU build
    cpu_ms: float           # measured at 1024px, not guessed
    gpu_ms: float
    quality: str
    params: tuple[str, ...] # which controls apply, e.g. ("strength",) or ("strength", "quality")
    blurb: str
```

One `transfer(method_id, content, reference, strength, **kw)` beside it is the
only dispatch, for the UI and the agent alike. Then:

- the dropdown is built from the registry
- control gating reads `params` instead of `method == "lab"`
- the filename suffix and PNG metadata read `id` and `label`
- the availability checks collapse to one lookup
- the agent's `harmonize` description is generated from `label`, `blurb` and the
  timings, so the model's choices stay current as methods are added

`cpu_ms` and `gpu_ms` get **measured**, the way the `models.py` timings were, so
the UI can say "about 20 seconds" instead of spinning silently.

**The classical tier.** Two methods from papers, both short, both instant on
CPU, both written here so neither carries any obligation.

- **MKL, linear Monge–Kantorovitch.** Pitié and Kokaram, CVMP 2007. Closed form:
  map the content's 3×3 colour covariance onto the reference's through the
  symmetric square root. About 20 lines of numpy.
- **IDT, N-dimensional PDF transfer.** Pitié, Kokaram and Dahyot, ICCV 2005 and
  CVIU 2007. Iterative: random rotation, a 1-D histogram match along each axis,
  rotate back, repeat. About 60 lines, plus the grain-suppression post-process
  from the CVIU paper.

`pengbo-learn/python-color-transfer` (MIT) implements both and is a fair
reference to check numerical agreement against. Read it to verify behaviour;
write our own.

A local experiment with per-channel histogram matching and a skewness-corrected
Reinhard was dropped in October 2026 in favour of these. IDT is per-channel
histogram matching done properly, along rotated axes so the channels stay
consistent. MKL matches the full covariance where the skewness tweak was a
heuristic.

- tests: each method returns the content unchanged at strength 0 and when the
  reference is the content itself; MKL's output covariance matches the
  reference's within tolerance; IDT's marginals match after its iterations;
  every registry id dispatches; the dropdown options equal the registry
- **done when** the dropdown offers ModFlows, LAB, MKL and IDT from the
  registry, no `method == "lab"` literal is left in `arcana.py`, and MKL and IDT
  have measured CPU timings at 1024 px recorded in the registry

### P5: Completion tools (about 3 days)

- prerequisite: P4. The agent picks methods from the same table as the dropdown.
- `harmonize`, `restyle`, `generate` and `job`; the cross-process GPU lock (A4);
  the generation step of `_poetry_job` moved into a headless function
- the `harmonize` skill
- **done when** the agent harmonizes two outliers toward the board palette, the
  results carry the derived badge, and the originals' mtimes are unchanged

### P6: Inside the app (optional)

Only if P1–P5 prove their worth. Add a chat panel that calls the same tools,
through the Claude Agent SDK or the Anthropic API's tool runner. Move the server
onto streamable HTTP inside the app process so the two share models and the job
queue. At this point the API key becomes part of the UI.

Roughly fourteen working days take this to the end of P5. P0–P1 on their own
already give you an assistant that knows your archive.

---

## Other uses once the server exists

Ordered by value for the effort. Most are ROADMAP items that the agent makes
easier to reach.

| use | needs | ROADMAP |
|---|---|---|
| **Name clusters by looking.** The agent views each cluster's exemplars and names it, which beats the nearest word in a 100-word list. | `survey`, plus a write-back for cluster names | 4, 5 |
| **Write the vocabulary.** A word list tailored to this archive, for per-image tags. | `survey`, plus a write-back for tags | 5 |
| **Ask about the archive.** *"Four-star, foggy, June"*, in plain language. | P1 only | none |
| **A story from a narrative.** The agent writes the scene lines, picks frames that flow into each other, and runs Inject Poetry. | P5 generation | sequence a selection |
| **Culling.** Duplicate stacks across the whole archive, with suggested keepers that defer to ON1 stars. | dedupe over the whole index | 6 |
| **A soundtrack for a board.** CLAP search of an audio dataset, driven by the board's concept. | `search` already branches on modality | stories with sound |
| Dataset admin by chat (index, extend, relocate) | none | low value; the GUI already does this well |

---

## Risks

- **Generic boards.** Language models reach for the obvious: fog, neon, rain.
  The counters are surveying first, the counterpoint rule, and showing the
  angles before gathering.
- **CLIP's limits.** CLIP can't see "nostalgia". The translation step and the
  weak-match flag turn a silent failure into an honest *"your archive has little
  for this"*.
- **Memory with both processes running.** Measure in P1; the ViT-H text tower
  is the large one. If it hurts, P6's in-process transport fixes it.
- **Two processes using the GPU.** See A4.
- **Licensing.** The `mcp` SDK is MIT-licensed, which fits the AGPL plan in
  LICENSING.md.

## Decisions

Settled 20 October 2026.

1. **Blind by default; pixels is the person's switch.** The setting is
   `agent.json` in the data dir, per machine and never committed. The person
   flips it, by hand or from Arcana's toggle; no tool can. While blind, every
   result says so. Details in A8.
2. **Generated pictures are allowed on boards, on a short leash.** Only for an
   angle the archive came back weak on, at most two per board, never the hero,
   always `"kind": "generated"` with a badge in the app and a mark on the
   rendered board. The server enforces the count and the hero rule; the skill
   explains them.
3. **Claude Code in this repo hosts v1,** through the project `.mcp.json` from
   P1. Claude Desktop can register the same command later. In P3 the plugin
   packaging makes this a single install.
