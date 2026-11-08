---
name: metaphysical-moodboard
description: Build a moodboard on a metaphysical or existential theme (disintegration, impermanence, death, emptiness, the void, transformation, memory, the sacred) from the person's Arcana archive, with text fragments from poets, mystics and traditions, laid out as a designed HTML page whose form enacts the theme and prints to PDF. Use when the person asks for a "metaphysical moodboard", a poetic or literary moodboard, a board "with texts" or "with poets", or wants the design of the board itself to embody its subject.
---

# Metaphysical moodboard

A metaphysical board pairs the person's photographs with fragments of text
from writers and traditions that worked on the same theme, and the page's
design carries the theme as well. A board on disintegration comes apart as
it is read. A board on emptiness is mostly empty. Use the `moodboard` skill
for the search and cut, and this one for the texts, the structure and the
page.

The general method, with the reference-scout brief and the table of theme
verbs and page behaviours, lives in the system skill `resonant-moodboard`
(in the intuition-skills repo). This skill holds the Arcana specifics.

Reference build: the `disintegration` board. Its spec is
`example-disintegration.spec.json` in this folder, and its page is
`output/boards/disintegration.page.html` in Arcana's output folder. Copy
the spec as the starting point for a new board, and point its `images` and
`out` fields at this machine's folders.

## 1. Read the theme through its traditions

Before searching pictures, gather texts. Spawn one research agent (or search
yourself) for 18 to 25 fragments across a wide range, covering

- one or two poets who wrote from inside the theme (for disintegration,
  Artaud's *L'Ombilic des limbes* and *Le Pèse-nerfs*, Baudelaire's
  "Une charogne", Rilke),
- contemplative traditions (the Bardo Thödol, the Heart Sutra, Nagarjuna,
  Zhuangzi, Dōgen, Ikkyū, the kusōzu nine-stages paintings, Advaita, Eckhart,
  Ecclesiastes),
- one or two modern voices (Michaux, Celan, Beckett, Lispector, Cioran, Jabès).

Rules for the texts.

- Verify every wording against a source (Wikisource, Gallica, sacred-texts,
  a scholarly edition). Drop or flag anything recalled from memory.
- Prefer public-domain texts and quote them in the original language with a
  translation. Mark your own translations as such in the colophon.
- For authors still in copyright, use a single short line at most, or name
  the work in the colophon as an influence without quoting it.
- Choose fragments that carry concrete images (bone, dust, light, water,
  ash, a name wearing away), because they sit beside photographs.

## 2. Find the structure in the sources

Take the board's sequence from a structure the texts themselves describe,
and make it the section order. Examples

- disintegration, the Bardo Thödol's dissolution of the elements (earth into
  water, water into fire, fire into air, air into consciousness, the clear
  light, the void),
- impermanence, the kusōzu nine stages, or the four seasons of a haiku year,
- transformation, the alchemical nigredo, albedo, citrinitas, rubedo,
- the sacred, approach, threshold, sanctum, departure.

Each stage becomes one spread with two or three photographs and one text.
Translate each stage into what CLIP can see (earth is rock, bone and mud,
fire is rust, embers and red-brown light) and search with `search facets`,
one facet per stage, so no picture repeats.

## 3. Build the board in Arcana

Create the board with one section per stage, the stage's reason and query,
and put the text fragment and its author in the picture notes. Then
`board export <id>` copies the files the page builder embeds.

## 4. Design the page so its form enacts the theme

Write a spec (JSON) and run the builder in this folder:

    python .claude/skills/metaphysical-moodboard/build_page.py <folder>/spec.json [--wireframe]

`build_page.py`'s docstring documents the spec. Positions are millimetres
on a 297 x 210 page, so screen and PDF share one layout. Each spread has a
`decay` from 0 to 1 that raises blur, lowers contrast and saturation, and
fades the text. Words in `[[double brackets]]` are drawn half-erased and
come back on hover. `ghost` gives a text a displaced double. An image with
`trace: true` is a faint memory of an earlier spread that drifts slowly.

Design moves by theme.

- **Disintegration, memory.** Minimal and subliminal. Pale paper, warm grey
  ink, a light serif. Decay rises spread by spread. Each spread carries a
  trace of an earlier image. Edges dissolve (`fade`). The last spread is a
  dark field with one small picture and one line.
- **Emptiness, the void.** Very few elements, very large margins, one
  picture per spread at small scale, text set far from the image.
- **The sacred, ritual.** Symmetry, a central axis, one warm accent colour,
  slow ascent from dark to light.
- **Transformation.** Colour shifts across the spreads in the order of the
  stages, and decay runs in reverse (from blur to sharp).

Constants for every theme.

- One idea per spread, a lot of negative space, nothing decorative that the
  theme does not explain.
- Text sits in the original language with the translation beneath (the
  text item's `tr` field) and a mono citation (author, work, year).
- Subliminal belongs to the images and the deliberate traces. Quotes,
  translations and writers' names stay readable at screen size. Use
  `text_scale` 1.3, quote size 3.9 or more, secondary lines 3.0 or more.
  The person asked for this on the first board.
- A first spread holds the title and an opening line. A last spread is a
  colophon listing every source, translator and picture handle.
- Check the page's screen view at phone width and in print preview.

## 5. Show it in Arcana and as PDF

Write the page into Arcana's own output folder as
`output/boards/<id>.page.html` (the builder's `out` field). For the installed
app that folder is `C:/Users/skite/AppData/Local/Arcana/output`, the folder
`board export` and `board pdf` report, which can differ from the repo's
`arcana/output`. `board page set` refuses a file from anywhere else. The page prints
one spread per A4-landscape page with backgrounds kept, so a browser's
Print to PDF gives the finished document.

Attach it to the board and print it.

```ops
board page set disintegration output/boards/disintegration.page.html
board pdf disintegration --wait 4
board show disintegration
```

`board page set` records the page on the board (agent-made boards only).
Writing straight to `output/boards/<id>.page.html` skips the copy.
`board pdf` prints the page with headless Chrome or Edge into
`output/boards/<id>.pdf`, honouring the page's `@page` size and print CSS.
`--wait` is how long the page's script runs before printing. `board show`
then reports `Page:`. In the Moodboard tab the board gets a "View page"
button, which opens the page with Save as PDF and Print.

The app serves the page in a sandbox with an opaque origin and no network,
so keep everything embedded and use script only for small effects such as
the trace drift. If `exec` says these ops are unknown, the app predates them.
Run the same lines with the `arcana` command in the terminal instead.

Blind mode. Check layout and print with
`build_page.py spec.json --wireframe`, which swaps the pictures for grey
tiles, then screenshot or print that file with headless Chrome. The real
page is for the person's eyes.

## 6. Hand back

Give the board's name in Arcana, the page's path, the PDF's path if made,
the structure and why it fits the theme, the texts used and their sources,
and what the archive was missing.
