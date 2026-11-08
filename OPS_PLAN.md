# Arcana ops: one vocabulary, many thin surfaces

P0 done, P1 next. Goal: every action Arcana can perform is a named op, declared
once, and the web UI, the command line and an MCP agent all reach the same ops
through thin surfaces generated from that one declaration. An agent can then
prompt the latent space, build libraries, assemble moodboards from varied
queries, change the colours of a picture, run ControlNet and IP-Adapter
transformations, and lay a board out as HTML, with the same vocabulary a person
types at a prompt.

**Verified against the repo 7 November 2026** at `b2fa984`, plus the
uncommitted work on the `agent-mcp` branch (`core.py`, `boards.py` schema 2,
`sheets.py`, `lut.py`, `mcp_server.py` with six tools, 327 tests passing).

This plan succeeds [AGENT_PLAN.md](AGENT_PLAN.md) on architecture. It keeps
that plan's decisions on privacy (A8, blind by default), originals (A7), handles
(A6), contact sheets (A5), board schema 2, the skills, the eval briefs and the
colour-transfer registry (P4). It replaces its shape of "one MCP tool per
capability, each with its own argument model" with ops, and it folds the nine
console scripts into one command.

---

## The rule

Every capability has one declaration. From it come the JSON Schema, the
command-line flags, `--help`, shell completion, the machine-readable op list
and the typed client. If the UI can do something the CLI or an agent cannot,
or if a capability's shape lives in two places, the design is wrong.

Arcana today fails that rule in four places, and the plan below is the list of
repairs.

| where | today | after |
|---|---|---|
| console scripts | nine entry points in `pyproject.toml`, each with its own `argparse` (`db.py` alone holds four) | one `arcana` command whose subcommands are the ops |
| the Dash app | 43 callbacks in `arcana.py` (5,700 lines) call `boards.save`, `transfer_colors`, `style_transfer.transfer` directly, and the colour method is a `method == "lab"` chain in six places | callbacks call `ops.call()`, and the method list comes from the registry |
| the MCP server | six tools, each a Pydantic signature, each a `do_*` function | one tool, `exec`, carrying text lines to the same ops |
| jobs and the GPU | `jobs.MANAGER` per process, so the app and the agent's server could both put a model on the card | one process owns the models, and every surface reaches it |

Seven pieces already fit the rule and stay as they are. `core.py` (search, filters,
dedupe, calibrated strength, headless), `boards.py` (schema 2, author, details),
`sheets.py` (contact sheets), `models.py` (the encoder catalogue, which is the
house pattern for a registry), `jobs.py` (progress and cancellation),
`paths.py` (the media-root guard), and the `do_*` functions in `mcp_server.py`,
which become the bodies of the first read ops.

---

## The design, applied to Arcana

### An op

```python
# arcana/ops/registry.py
@dataclass(frozen=True)
class Op:
    name: str                     # "board add": noun, then verb
    kind: Literal["read", "write", "effect"]
    args: type[Args]              # Pydantic model, extra="forbid"
    result: type[Result]          # declared result shape
    run: Callable[[Args, Context], Result]
    doc: str                      # one paragraph, written for a reader who may be a model
    examples: tuple[str, ...]     # complete command lines, run by the tests
    needs: frozenset[str] = frozenset()        # capability flags, see "Modes are flags"
    actors: frozenset[str] = frozenset({"person", "cli", "agent"})
```

Everything an op needs to be described, validated, invoked and tested is in
that one object. `register()` refuses a second op with the same name, an op
without a doc, an args model without `extra="forbid"`, and an example that
fails to parse.

```python
class BoardAddArgs(Args):
    board: str = positional(0, description="Board id (slug).")
    handles: list[str] = positional(1, min_length=1,
                                    description="Pictures, as dataset:key.")
    section: str | None = Field(None, description="Section id to place them in.")
    role: Role = Field("support", description="hero, support, texture, detail or counterpoint.")
    note: str | None = Field(None, description="Why they are here.")

register(Op("board add", "write", BoardAddArgs, BoardResult, _board_add,
            doc="Put pictures on a board. ...",
            examples=("board add monsoon japan:1423 japan:88 --role hero",)))
```

Parsing that model is the only validation. The web UI, the CLI, the MCP tool and
the tests all produce a dict and hand it to the model. A field constraint is
written once, in the `Field`, and reaches `--help`, the JSON Schema and the
refusal message from there.

### Kinds

- A **read** changes nothing. It may load a dataset or an encoder, which is
  cached, and it may render a contact sheet into the response.
- A **write** runs as a transaction on the single owner of state. It records
  one labelled undo entry, and a shared step after it bumps the state version
  and notifies clients. A write never broadcasts on its own.
- An **effect** does I/O or runs a process. Indexing a folder, a colour
  transfer, a ControlNet pass, rendering a board to disk, downloading a model.
  Long effects return a job id, and the job's completion is itself recorded
  through a write (the output ledger below), so clients learn of it the same
  way they learn of every other change.

### What "state" is in Arcana

Arcana's mutable state is small, and it already lives in the data dir.

| state | today | owner after |
|---|---|---|
| boards | `boards/*.json`, written by `boards.save` from the app and planned from the agent | `state.Store`, which calls `boards.py` |
| agent settings | `agent.json` (`pixels`) | `state.Store`, read fresh, written by person actors only |
| jobs | `jobs.MANAGER`, in memory, per process | `state.Store`, in the one server process |
| derived pictures | files in `output/`, remembered only by the board that references them | an **output ledger**, `output/ledger.json`: id, path, op and args that made it, source handles, when |
| datasets | index and latent pickles, bundles | files, written by effects, listed by reads. A dataset is immutable once finished, so it needs no transaction. |

The ledger is new and small. It is what lets a derived picture have a handle
(`out:3f9c…`) that `board add`, `item show` and `colour transfer` can take like
any other, and it is what the Moodboard tab reads to badge a picture as derived.

### The owner of state, `arcana/state.py`

```python
class Store:
    version: int                              # bumped by every write
    def transaction(self, actor: str, label: str) -> Transaction
    def undo(self, actor: str) -> str | None   # label of what was undone
    def redo(self, actor: str) -> str | None
    def subscribe(self, fn)                   # called with (version, labels) after each commit
```

A transaction snapshots each board it touches before the first change. Commit
pushes one entry, holding those snapshots and the label, onto the actor's undo
stack. Rollback restores them. Each actor has its own stack, so an agent cannot
undo a person's edit, and a person's undo in the Moodboard tab leaves the
agent's work alone. Redo stacks clear on the next commit by the same actor.

Writes happen on one thread. Reads take no lock, since boards are read from disk
through `boards.load`, which is atomic per file.

### Actors

Three classes, named in every call.

| actor | who | may |
|---|---|---|
| `person` | the web UI | everything, including `settings set` and `board delete` |
| `cli` | a terminal on this machine | everything a person may, since the person is typing |
| `agent` | an MCP client | ops whose `actors` include `agent`. Today that leaves out `settings set`, `board delete` and `dataset delete`. |

A board records its `author`. A write by an agent actor on a board authored by a
person is refused, and the refusal names the op that makes a copy
(`board copy monsoon --id monsoon-agent`). This is A7, enforced in one place,
`state.Store`, rather than in each tool.

### The grammar, text lines

One line is one op. The op name is the longest registered prefix of the line,
so `board add monsoon japan:1` resolves to `board add`. The rest is parsed with
`shlex` and mapped onto the args model by these rules, generated from the
model's fields.

| field type | on the line |
|---|---|
| scalar | `--k 12`, `--strength 0.5` |
| bool | `--full-res` and `--no-full-res` |
| `list[scalar]` | the flag repeated, `--phrase "rain" --phrase "wet asphalt"` |
| enum / Literal | `--mode palette`, with the choices in `--help` and in completion |
| positional | `positional(n)` fields, in order, with a trailing list taking the rest |
| nested model or `list[model]` | JSON in one argument, `--facet '{"name": …}'`, repeatable |
| the whole args model | `--args '{…}'` or `--args @request.json` |

Flag names are the field names in kebab case. Field names stay the same across
the text grammar, the JSON Schema and the Python model, so a refusal that names
`strength` names the same thing on every surface.

Handles make paths rare on the line. A dataset is `japan`, an audio dataset
`field@audio`, a picture `japan:1423`, a derived picture `out:3f9c1e`, a board
its slug. Where a path is unavoidable (`dataset create --folder`), it is quoted,
and forward slashes work on Windows.

### `call(op, args, actor)` and `exec(lines, actor)`

```python
def call(op: str, args: dict, actor: str, ctx: Context | None = None) -> Result
def exec(lines: list[str], actor: str, ids: dict[str, str] | None = None) -> Batch
```

`call` is the one server-side entry point. It looks the op up, parses the args
model, checks the op's `needs` against the running flags and its `actors`
against the caller, then runs it. For a write it opens the transaction and
commits or rolls back.

`exec` parses each line into `(op, args)`, then runs the list as **one atomic
batch and one undo step**. If any step is refused, every step rolls back and the
batch result says which line failed and why. Callers may pass their own ids, so
a batch can refer to what it creates.

```
board create "Monsoon melancholy" --id monsoon --brief "…"
board section add monsoon --id s1 --title "Water as weight"
board add monsoon japan:1423 japan:88 --section s1 --role hero
board render monsoon --format html
```

A read in a batch runs in order with the writes around it, so an agent can add
pictures and view the result in one call.

### Errors teach

A refusal is a `Refusal(op, reason, help)`. The reason comes from Pydantic's
message rewritten in Arcana's words ("`k` must be between 1 and 48, got 200"),
from an op's own `ValueError` ("No dataset called `nope`. Known datasets:
japan, abstract"), or from policy ("`board delete` is for the person. Ask
them"). `help` is that op's generated help text. On the CLI the help prints
under the reason. On MCP the refusal goes back as tool content with
`isError: true`, so the model reads it and corrects itself. Protocol errors are
reserved for a broken transport.

### Results

Each op declares a result model. Three renderers read it.

| renderer | used by | what it does |
|---|---|---|
| text | CLI, MCP | one compact block per result. Hit lines follow `sheets.py`'s existing form, `#7  japan:1423  p 0.92  ★★★★  [Forest]  DSC04481.jpg`. |
| json | the web UI, tests, `arcana --json <op>` on the CLI (before the op, so it never collides with an op's own flags) | the model dumped, with image attachments as captions and paths |
| attachments | all | a result may carry images (a contact sheet, a rendered board). The CLI writes them to `output/sheets/` and prints the path. MCP sends them as image content in pixels mode and as a line saying so in blind mode. The web UI gets a `/thumb` URL. |

The pixels gate stays exactly where A8 put it, in `agent.json`, read on every
call, and applied in the attachments renderer for agent actors.

### Transports

```
  Dash callbacks ──────► ops.call()  ◄────── tests
                             ▲
  POST /exec  (text lines)   │   GET /ops  (op list)   GET /ops/schema.json
  POST /rpc   (json)         │   GET /state/version
        ▲                    │
        │          ┌─────────┴──────────┐
  arcana (CLI) ────┤   arcana.server    │  lives on app.server (Flask), the same
  arcana-mcp  ─────┤   one process,     │  process as the Dash app, so models,
                   │   owns the models  │  jobs and the GPU have one owner
                   └────────────────────┘
```

- **In the app process.** `arcana/server.py` is a Flask blueprint registered on
  `app.server`, which already serves `/thumb` and `/preview`. The routes are
  `/exec` (text lines), `/rpc` (one op as JSON), `/ops`, `/ops/schema.json`
  and `/state/version`. Everything binds to `127.0.0.1`.
- **Discovery and a token.** On start the server writes `server.json` to the
  data dir with its port, pid and a random token, and removes it on exit. The
  CLI and the MCP server read it and send the token in a header. Anything able
  to read the data dir is already able to edit the boards in it, so the token
  adds no new trust and shuts out other software on the machine.
- **Headless.** `arcana --headless` starts the same process with the ops routes
  and without the Dash layout. One code path, one flag, which is how the design
  wants modes handled.
- **In-process fallback.** With no `server.json`, the CLI and the MCP server run
  ops in their own process. A read works at once. An effect that needs the GPU
  runs in that process too, which is what the installed app does when nothing
  else is open. The A4 lock file then has nothing left to guard, because when
  two processes could want the card, one of them is the app, and everything
  else reaches it over `/exec`.
- **Change notification.** `Store.subscribe` bumps `version`. The Dash app
  polls `/state/version` with a `dcc.Interval` every two seconds and reloads
  the Moodboard tab's board list when it moves. This replaces the mtime watcher
  of A3 with the shared step every write already runs.

### The CLI client

`arcana` is one console script, `arcana.cli:main`. It knows nothing about ops.

1. With arguments, it joins them into a line, finds the server, posts the line
   to `/exec` and prints the reply. Without a server it imports `arcana.ops`
   and runs the line in-process.
2. Without arguments it starts the app, as it does today.
3. `arcana --help`, `arcana <op> --help`, `arcana op list` and
   `arcana complete "<partial line>"` are all answered by the server-side
   grammar layer, so the client ships no help text.
4. Completion scripts for bash, zsh and PowerShell are generated from the op
   list by `arcana completion <shell>`.

The eight other console scripts (`arcana-build-latent`, `arcana-extend`,
`arcana-finish`, `arcana-marks`, `arcana-migrate`, `arcana-relocate`,
`arcana-mcp`, `arcana-lut`) stay for one release as aliases, each a two-line
shim that maps its old flags onto one op line and prints the new spelling.
`arcana-mcp` stays for good, since MCP clients launch it by name.

### MCP

`arcana-mcp` exposes **one tool**, `exec(commands: list[str], actor: str =
"agent")`. Its description is the output of `orient`, so the model's first
sight of Arcana is the same page a person reads. Each command's result comes
back as text, with images in pixels mode. A refusal comes back as `isError`
content with the op's help attached.

The server also exposes two resources, `arcana://orient` and `arcana://ops`,
for clients that read resources before acting, and the `moodboard(theme,
dataset)` prompt from AGENT_PLAN.

It forwards to the running app when `server.json` exists and runs in-process
otherwise. The MCP SDK already in the lock (`mcp` 2.3.0, `MCPServer`) serves it
over stdio, and the fd-1 protection and `ToolError` handling from the current
`mcp_server.py` carry over.

Because the CLI and MCP share the grammar, a person can paste an agent's
command into a terminal and get the same result, and a test written as command
lines covers both.

### Modes are flags

The server computes a set of capability flags at start, and every op declares
what it `needs`.

| flag | true when |
|---|---|
| `gpu` | `gpu.available()` |
| `diffusers` | the diffusers import succeeds (img2img, IP-Adapter, ControlNet, generate) |
| `modflows` | the ModFlows checkpoint is present or downloadable |
| `pixels` | `agent.json` allows thumbnails to leave the machine |
| `ui` | the Dash layout is being served |
| `frozen` | running from the installer build |

`op list` shows every op, and marks a hidden one with the flag that hides it
(`style transfer   needs diffusers`), so an agent on the CPU build learns what
it is missing before it tries. A flag hides an op. There is one code path for
every mode.

### The system describes itself

```
$ arcana orient
Arcana searches your own picture archive by what the pictures show, builds
moodboards from it, and changes the colours or style of a picture. Nothing here
edits or moves an original. Pictures are named by handle (japan:1423).
Learn it in three steps:  orient  ·  op list  ·  <op> --help
Datasets: japan (4,312 images, ViT-H/14), abstract (9,359, ViT-B/32), field@audio
Mode: blind (thumbnails stay on this machine). GPU: RTX 4070, fp16.
Boards: 3. Jobs running: 0.

$ arcana op list
dataset list        read    The datasets on this machine, with size, encoder and features.
dataset survey      read    What a dataset holds: clusters, largest first, with exemplars.
search text         read    Pictures matching phrases, with negatives and the person's marks.
…

$ arcana search text --help
search text <dataset> --phrase … [--phrase …] [--not …] [--k 12] …
  Pictures that match one visual idea said several ways. CLIP sees concrete
  things (objects, light, weather, texture, technique) and misses abstractions,
  so write what a camera would see. Each result has p, …
```

Tests run the examples of every op through `exec`, so help and behaviour move
together.

---

## The op vocabulary, version 1

Names are noun then verb. Kind is read (R), write (W) or effect (E). A trailing
`job` means the op returns a job id and finishes through `job show` or
`job wait`.

### system

| op | kind | does |
|---|---|---|
| `orient` | R | the page above |
| `op list [--all]` | R | every op, hidden ones marked with the flag that hides them |
| `help <op>` | R | the generated help |
| `settings show` | R | `agent.json`, the flags, the data dir |
| `settings set --pixels on` | W | person and cli only |
| `undo`, `redo`, `history` | W/R | the caller's own stack |

### dataset and model

| op | kind | does | today |
|---|---|---|---|
| `dataset list` | R | name, modality, count, encoder, features, marks, date range, health | `core.find_datasets`, `do_list_datasets` |
| `dataset show <ds>` | R | one dataset in full | `describe_dataset` |
| `dataset survey <ds> [--per-cluster 4]` | R | clusters with exemplars | `do_survey` |
| `dataset create --name --folder [--model] [--k] [--features palette,style] [--dims 2]` | E job | index a folder | `db.index_dataset`, `arcana-build-latent` |
| `dataset extend <ds> --folder` | E job | add new files | `db.extend_dataset`, `arcana-extend` |
| `dataset finish <ds> [--k] [--labels] [--dims]` | E job | rework without re-encoding | `db.rework_dataset`, `arcana-finish` |
| `dataset marks <ds>` | E | re-read ON1 and XMP sidecars | `db.refresh_marks`, `arcana-marks` |
| `dataset relocate <ds> --root` | E | find moved files by content | `relocate.main` |
| `dataset migrate` | E | older datasets to bundles | `legacy.main` |
| `dataset delete <ds>` | E | person and cli only | `legacy.delete_dataset` |
| `model list`, `model download <id>` | R / E job | the encoder catalogue | `models.catalogue`, `models.download_model` |

### search and item

| op | kind | does | today |
|---|---|---|---|
| `search text <ds> --phrase … [--negative …] [--k] [--diversity] [--min-stars] [--tag …] [--date-from] [--date-to]` | R | one facet, calibrated p, strong/fair/weak. Diversity defaults to 0, so the hits are the Prompt Search tab's | `do_search`, one `Facet` |
| `search facets <ds> --facet '{json}' …` | R | several facets at once, no picture repeated across them | `do_search` |
| `search like <handles…> [--mode semantic\|palette\|style\|combined] [--k] [--exclude]` | R | by example | `do_more_like`, `db.search_by_*` |
| `search palette <ds> --colour '#1f2a36:0.4' … [--k]` | R | pictures in a chosen palette | `do_find_by_palette` |
| `item show <handles…> [--side 1024]` | R | a close look, with palette, marks, cluster and a CLIP description | `do_inspect` |
| `item sheet <handles…> [--tile 160]` | R | a contact sheet of any handles | `sheets.contact_sheet` |
| `item palette <handle> [--n 8] [--l-min] [--l-max] [--s-min]` | R | the palette strip with lightness and saturation filters | `refselect.palette_strip` |
| `item describe <handle>` | R | CLIP tags | `style_transfer.describe` |

### board

| op | kind | does | today |
|---|---|---|---|
| `board list` | R | slug, name, author, count, updated | `boards.listing` |
| `board show <slug> [--sheet]` | R | the record, grouped by section, with metrics (diversity, near-duplicate pairs, palette spread, section balance, derived share) | new, on `core` arithmetic |
| `board create <name> [--id] [--brief] [--concept]` | W | | `boards.save` |
| `board copy <slug> --id <new>` | W | the agent's way into a person's board | new |
| `board add <slug> <handles…> [--section] [--role] [--note]` | W | | `boards.save` with details |
| `board remove <slug> <handles…>` | W | | |
| `board move <slug> <handle> --section s2 [--before h]` | W | | |
| `board note <slug> <handle> --text` | W | | |
| `board section add\|set\|remove <slug> …` | W | title, rationale, queries | |
| `board palette set <slug> --colour …` and `board palette extract <slug>` | W | impose or derive | `palette.extract_dominant_colors` |
| `board concept set <slug> --text` | W | | |
| `board render <slug> [--format html\|png] [--layout editorial\|masonry\|grid] [--width]` | E | writes `output/boards/<slug>.html`, returns the path and a preview | new, `compose.py` |
| `board open <slug>` | E | opens the rendered board in the browser, person and cli only | |
| `board export <slug> --folder` | E | copies the files | the Moodboard tab's "Export copies" |
| `board delete <slug>` | W | person and cli only | `boards.delete` |

### colour and style

| op | kind | does | today |
|---|---|---|---|
| `colour methods` | R | the registry with licence, download, GPU need and measured timings | P4's `transfer_methods.py` |
| `colour transfer <target> --from <reference> \| --palette … [--method lab\|mkl\|idt\|modflows] [--strength] [--full-res]` | E, job for modflows | a derived picture, in the ledger | `transfer_colors`, `lab_transfer` |
| `colour lut <target> --from <reference> [--n 33]` | E job | a `.cube` LUT | `lut.bake_modflows`, `arcana-lut` |
| `colour harmonize <slug> --items … --toward <handle> \| --palette [--method] [--strength]` | E job | several transfers, results added to the board as derived supports | new, on the op above |
| `style methods` | R | texture, img2img, ip_adapter, controlnet, with needs | the strings at `arcana.py:4088` |
| `style transfer <target> --from <source> --method … [--control depth\|canny\|luminance] [--strength] [--scale] [--steps] [--prompt] [--seed]` | E job | | `style_transfer.transfer`, `controlnet_transfer` |
| `style control-map <handle> --kind depth` | R | the map a ControlNet will read, so the agent can judge it | `style_transfer.control_map` |
| `generate --prompt … [--init <handle>] [--strength] [--seed] [--for-board <slug>]` | E job | sd-turbo, always `kind: generated` in the ledger. With `--for-board`, the server refuses a third generated picture on that board and refuses role hero. | `_poetry_job` at `arcana.py:4445`, moved to `generate.py` |

### job

| op | kind | does |
|---|---|---|
| `job list`, `job show <id>` | R | status, progress, result handles |
| `job wait <id> [--timeout 120]` | R | blocks up to the timeout, then returns the state either way |
| `job cancel <id>` | W | |

That is 55 ops. The agent needs about twenty of them for a moodboard, and the
rest exist so that the rule holds for the Datasets tab and the console scripts.

---

## The web UI on ops

`arcana.py` keeps its layout, its media routes (`/thumb`, `/preview`,
`/palette`, `/audio`) and its callbacks. What changes is the body of each
callback that acts.

- The Moodboard tab's save, load, delete and export call `board …` ops with
  actor `person`. Its board list reloads when `/state/version` moves, which is
  how an agent's board appears there within two seconds.
- The colour and style panels call `colour transfer` and `style transfer` and
  poll `job show`. Their method dropdowns are built from `colour methods` and
  `style methods`, which closes P4.
- Inject Poetry calls `generate` per scene.
- The Datasets tab calls `dataset create`, `extend`, `finish`, `marks`,
  `relocate` and `model download`, and reads progress from `job show`. The
  job manager it uses today is the same object, now owned by `state.Store`.
- Search stays on `core` directly, since it already is a read on the shared
  module, and `search text` is a thin op over the same function.

One test guards the rule. It walks the AST of `arcana.py` and `ui_datasets.py`
and fails on any call into `boards`, `color_transfer`, `style_transfer`,
`lab_transfer`, `lut` or a `db` function that writes. The allowed path into
those modules is `arcana.ops`.

---

## Boards in HTML

`arcana/compose.py` renders a board record to one self-contained HTML file.

- Three layouts, `editorial` (one hero, supports in a ragged grid, details in a
  strip), `masonry` and `grid`. Each is a CSS layout over the same markup, so
  the layout is a class on the root element.
- Pictures are referenced by `file://` path, with thumbnails as data URIs at
  320 px so the page opens on a machine where the paths are gone.
- The palette strip, the concept, each section's title and rationale, each
  picture's note and role, and a mark on every derived or generated picture.
- Styles come from `ui_style.py`'s tokens, so the board and the app agree on
  colour and type.
- Plain HTML and CSS, with one small inline script for the lightbox, and a
  print stylesheet.

`board render --format png` composes the same board with PIL, which is what the
agent looks at in pixels mode to critique its own layout. The HTML is for
people, and for publishing.

The **board-design** skill from AGENT_PLAN teaches what a good board looks like
in these terms, and gains one section on the HTML layouts, when to pick each,
and how to read the metrics `board show` reports against the rendered page.

---

## Skills and the plugin

As in AGENT_PLAN, three skills, `moodboard`, `board-design` and `harmonize`,
packaged with `.mcp.json` as a Claude Code plugin in the repo. The skills now
speak in op lines, so a skill step reads
`exec ["dataset survey japan", "search facets japan --facet …"]`, and every line
in a skill is one the tests run.

The `moodboard` skill's workflow keeps its substance. The one addition is
that it ends with `board render --format html` and hands back the path.

---

## Tests

| file | checks |
|---|---|
| `tests/test_ops_registry.py` | every op has a doc, a kind, a result model and an args model with `extra="forbid"`. Every example parses and runs against the synthetic fixture dataset. Help renders for every op. The JSON Schema generates and round-trips. Names are unique and two words. |
| `tests/test_exec.py` | a batch with a refused third line leaves no change from the first two. Undo is per actor. A refusal carries the op's help. Ids passed in are honoured. |
| `tests/test_grammar.py` | text line to args and back for every field type in the table above, with quoting and Windows paths. |
| `tests/test_server.py` | `/exec`, `/rpc`, `/ops`, `/state/version` over the Flask test client, the token, and that writes bump the version once per batch. |
| `tests/test_cli.py` | the client finds `server.json`, falls back in-process, prints help from the server, and the alias shims map old flags. |
| `tests/test_mcp_server.py` | one tool, `exec`, over a real stdio pipe. A refusal comes back as `isError`. Blind mode sends no image content. The server forwards when a `server.json` is present. |
| `tests/test_parity.py` | the AST walk above. |
| `tests/test_compose.py` | an HTML board validates, references every item, marks derived ones, and renders a PNG the same size as its declared width. |

The existing 327 tests keep passing throughout, since `core.py`, `boards.py`
and `db.py` keep their signatures.

---

## The installer

The MCP SDK is MIT, pure Python and small, and after this plan the installed
app is the natural server for an agent. So the installer ships `mcp` and
`arcana-mcp`, and the first-run page gains one line under "Working with an
agent" with the command to register it, which for Claude Code is
`claude mcp add arcana -- "<install dir>\Arcana.exe" --mcp`. The frozen
executable takes `--mcp` and `--headless` as flags on the same `launcher.py`,
which is one more case of a mode being a flag. `pyproject.toml`'s `agent` extra
then folds into the main dependencies.

---

## Phases

Each phase ends with something usable and keeps every existing test green.

### P0. Registry, grammar, exec, CLI, read ops (about 3 days)

- `arcana/ops/` with `registry.py`, `grammar.py`, `render.py`, and the system,
  dataset (reads only), search and item ops. Their bodies are the `do_*`
  functions moved out of `mcp_server.py`.
- `arcana/cli.py`, in-process only, with help and completion.
- `tests/test_ops_registry.py`, `test_grammar.py`, `test_cli.py`.
- **Done when** `arcana search text japan --phrase "rain on a dark street"`
  prints the same hits the Dash search tab shows, `arcana op list` lists the
  read ops, and every example runs in the tests.

**Done 7 November 2026.** Sixteen read ops across system, dataset, model,
search and item. `search text` returns the Prompt Search tab's cards in the
same order for eight queries over `japan` (ViT-B/32) and `export-quick`
(ViT-H/14), and `test_ops_registry.py` checks the same on the synthetic
dataset. Every example runs as the CLI, as a blind agent and as a seeing agent,
and writes nothing. The six MCP tools now call the ops, so their tests cover
the ops too. 400 tests pass.

What building it settled, beyond the text above:

- `exec(lines)` is `ops.exec_lines`, since `exec` is a Python builtin.
- The CLI's output switch is `arcana --json <op>`, before the op. Inside a
  line the whole args model is `--args`, so `--json` means one thing.
- Flag names for the filters are `--min-stars`, `--tag` (colour tags) and
  `--date-from`/`--date-to`. `from` is a Python keyword, and `--colour` stays
  for hex colours in `search palette`. Negatives are `--negative`.
- `register()` refuses write ops until `state.Store` exists in P1, so nothing
  can write outside a transaction in the meantime.
- `pydantic` is declared in `requirements.txt`. Dash already brought it in.
- A frame with nullable integer ratings holds `pd.NA`, which made the old
  `x == x` test raise inside `facts()`. It uses `pd.notna` now, which fixed
  search on `export-quick` for the MCP tools as well.

### P1. State, boards, server, MCP as one tool (about 3 days)

- `arcana/state.py` with transactions, per-actor undo and the version counter.
  The output ledger.
- The board ops. The Moodboard tab on them, with the version poll.
- `arcana/server.py` on `app.server`, `server.json`, the token, `--headless`.
- `mcp_server.py` rewritten to one `exec` tool that forwards or runs in-process.
  The six tools go.
- `tests/test_exec.py`, `test_server.py`, `test_parity.py` (boards only at this
  point), and `test_mcp_server.py` rewritten.
- **Done when** an agent in Claude Code builds a board from two searches in one
  `exec` call, it appears in the open app within two seconds, a person's edit
  there keeps the agent's notes, and `arcana undo` in a terminal takes back
  only the terminal's own last batch.

### P2. Transfer registry and the completion ops (about 3 days)

- P4 of AGENT_PLAN as written there, `transfer_methods.py` with LAB, MKL, IDT
  and ModFlows, measured timings, and the dropdown built from it.
- `colour transfer`, `colour lut`, `colour harmonize`, `style methods`,
  `style transfer`, `style control-map`, `generate` with the leash, and the job
  ops. `_poetry_job`'s generation moves to `generate.py`.
- The colour, style and Inject Poetry panels on ops.
- **Done when** the agent harmonizes two outliers toward a board palette and
  runs one ControlNet depth transfer through `exec`, both results carry derived
  handles that `board add` accepts, the app badges them, and the originals'
  mtimes are unchanged.

### P3. HTML boards, the skills, the plugin (about 3 days)

- `compose.py`, `board render`, `board open`, `board show` metrics.
- The three skills rewritten in op lines, the `moodboard` prompt, the plugin
  manifest, the five-brief eval set.
- **Done when** "moodboard for monsoon melancholy from japan" in Claude Code
  returns an HTML board with at least four sections, a palette strip, a reason
  per section and no near-duplicate pair, and it is a board the person would
  keep.

### P4. Dataset lifecycle and the single command (about 2 days)

- `dataset create`, `extend`, `finish`, `marks`, `relocate`, `migrate`,
  `delete`, `model download` as effects with jobs. The Datasets tab on them.
- The eight console scripts become alias shims. `INSTALLATION.md` and the
  README's command-line section describe `arcana <op>`.
- `test_parity.py` extended to `ui_datasets.py` and `db` writes.
- **Done when** every row of the README's command-line block has an op
  spelling, and the Datasets tab performs no write outside `ops.call`.

### P5. Installer (about 1 day)

- `mcp` into the main dependencies, `--mcp` and `--headless` on the launcher,
  the spec's hidden imports, the first-run line, both builds rebuilt and
  installed over the current one.
- **Done when** Claude Code registered against the installed `Arcana.exe --mcp`
  lists the datasets, and the GPU build reports its card through `orient`.

About fifteen working days to the end of P4. P0 and P1 alone give one grammar
across the terminal and the agent, and a Moodboard tab that shows what the
agent builds.

---

## Decisions to take

Each has a recommendation. The work starts on the recommendation unless you
say otherwise.

1. **One MCP tool, or one plus a read-only `search`?** One. The grammar is
   small, the instructions carry `orient`, and the first refusal teaches. A
   second tool would start the drift the design exists to prevent. If a model
   stumbles on the grammar in the P1 acceptance check, the fix is a better
   `orient` page, measured on the eval briefs.
2. **Nested facets on the command line.** JSON in a repeated `--facet` flag,
   plus `search text` for the common one-facet case. A bespoke mini-language
   would be a second grammar.
3. **Ship `mcp` in the installer.** Yes, in P5. It changes the installer's
   size by a few megabytes and makes the installed app the agent's server.
4. **Where the plan lives.** This file. AGENT_PLAN.md keeps the design
   decisions it owns and gains a pointer here at the top.

## Risks

- **A 5,700-line module under refactor.** The parity test and the existing
  327 tests are the safety net, and each phase moves one tab at a time.
- **Grammar for a model.** A model that has never seen Arcana has to write
  `--facet '{"name": …}'` from help text. The eval briefs in P3 measure how
  often it gets a refusal on the first try, and the `orient` page is tuned on
  that number.
- **Two processes and the GPU.** Solved by construction while the app is open,
  since every surface reaches its process. Two headless processes started by
  hand could still collide, and `server.json` makes the second one say so and
  forward.
- **Windows quoting.** PowerShell and bash quote differently, and the MCP
  client sends raw strings. `shlex` in POSIX mode is the one parser, forward
  slashes are accepted in paths, and `test_grammar.py` carries the cases that
  bit.
