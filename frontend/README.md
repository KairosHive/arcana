# Arcana's page

The web page is a Svelte 5 application built with Vite. Flask serves the built
bundle from `arcana/web` beside the JSON API (`arcana/api`), the ops routes
and the media routes, so the installed app ships one static page and no
JavaScript toolchain.

## Build

```bash
npm install
npm run build
```

`npm run build` writes `arcana/web`. The Python server, the wheel and the
installer all read from there, so build before `uv run python -m arcana.arcana`,
`pip install .` or the PyInstaller spec. `npm run check` runs the Svelte and
TypeScript checks.

## Develop

Start the Python server with the development token route enabled, then Vite:

```bash
ARCANA_UI_DEV=1 uv run python -m arcana.arcana
```

```bash
npm run dev
```

Vite serves the page on 5173 and forwards `/api`, `/ops`, `/agent` and the
media routes to the Python server on 8051 (`ARCANA_DEV_PORT` changes it). The
page asks `/api/session` for its token, which the server answers only while
`ARCANA_UI_DEV` is set. In the built page the token is written into
`index.html` as it is served.

## Layout

| folder | what |
|---|---|
| `src/styles` | `tokens.css` (colours, type, spacing, radii, control sizes) and `base.css` |
| `src/lib/ui` | the shared controls: Button, Field, Select, Slider, Card, Disclosure, Tabs, Modal, Thumb, Stepper, SplitPane and the rest |
| `src/lib/components` | MediaCard, JobCard, MarksBadge, shared by several tabs |
| `src/lib/api` | the fetch client with the token, and the media URL builders |
| `src/lib/stores` | the shell's state (`app.svelte.ts`): tab, dataset, boards, collection, settings, the state poll, jobs, toasts, the viewer |
| `src/shell` | the header, the context bar, the relocate panel, the picture viewer |
| `src/tabs/datasets` | the five-step wizard, the dataset list, the encoders |
| `src/tabs/search` | Prompt Search and Generate Story: the map, the marks filters, the results |
| `src/tabs/moodboard` | the collection rail, the similarity search, the board page |
| `src/tabs/alchemy` | the step (cast, stage, settings) and the film (timeline, player) |
| `src/tabs/agent` | the terminals |

The colour tokens mirror `arcana/ui_style.py`; `tests/test_ui_style.py` keeps
the two in step and refuses a palette hue spelled out in a component.
