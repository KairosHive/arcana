---
name: moodboard
description: Build a moodboard from a person's own picture archive with Arcana. Use when asked for a moodboard, a mood board, a visual reference board, a look book or a set of reference pictures on a theme, from a dataset in Arcana. Covers planning angles, translating abstract themes into what CLIP can see, gathering wide, cutting to a board, taking its palette, harmonizing sparingly and rendering it as HTML.
---

# Moodboard

Arcana's `exec` tool takes op lines. Every line below is one you can send. Put
several in one call and they run as one batch and one undo step. `<op> --help`
gives any op's full shape.

## 1. Restate the brief

Write the brief back as one paragraph that says what the board argues, in
concrete terms. If the brief could mean two very different boards, ask one question,
then go.

## 2. See what the archive holds

```ops
orient
dataset list
dataset survey japan --per-cluster 4
```

Plan from the survey as well as the brief. A brief about the sea from an
archive of temples is a board about the few sea pictures there are, said
plainly.

## 3. Break the theme into angles

Four to seven angles, from lenses such as subject, place, light and atmosphere,
material and texture, colour, scale (wide, mid, detail), human presence or
absence, motion, symbol, time (hour, season, era). Add one counterpoint, an
angle that sharpens the theme by contrast. Post the angles in one short
message and carry on unless told otherwise.

## 4. Translate each angle into what CLIP sees

CLIP ranks by concrete visual things. Replace the abstract with a visual
equivalent ("melancholy" becomes "empty bench in rain, overcast, muted
colours"). Two or three phrasings per angle, photographic vocabulary (long
exposure, overhead, macro, backlit), and negatives for subjects that keep
intruding.

## 5. Gather wide

Search all angles at once, so no picture repeats across them. Thirty to sixty
candidates in all.

```ops
search facets japan --facet '{"name": "water as weight", "phrases": ["heavy rain on a dark street", "flooded pavement at night"], "negatives": ["umbrella"]}' --facet '{"name": "shelter", "phrases": ["empty bus stop at night", "a doorway out of the rain"]}' --k 10
search like japan:1423 --k 8
```

Every facet comes back strong, fair or weak. Weak means the archive has
little for that angle. Say so in the hand-back and in the section's reason,
and present a weak facet's pictures as stand-ins. A weak facet can still hold
one real picture (it names the ones that score well on their own), so look
before using it.

## 6. Cut to a board

Twelve to twenty-four pictures. One picture per idea, no near-duplicates, a
mix of scales, and one hero that nothing else competes with. Make the board,
one section per angle, each with its reason and the queries that found it.

```ops
board create "Monsoon melancholy" --id monsoon --dataset japan --brief "monsoon melancholy" --concept "Rain as a weight the city carries, and the shelters people make under it."
board section add monsoon --id s1 --title "Water as weight" --rationale "Rain as a weight the street carries." --query "heavy rain on a dark street"
board add monsoon japan:1423 --section s1 --role hero --note "The street holds the rain like a held breath."
board add monsoon japan:88 japan:412 --section s1 --role support
board show monsoon
```

`board show` reports the metrics the board-design skill reads, such as heroes,
near-duplicate pairs, diversity, palette spread and the share of derived
pictures.
Fix what it flags before rendering.

## 7. Take the palette from the selection

```ops
board palette extract monsoon --n 5
```

The board's own pictures decide its palette.

## 8. Harmonize sparingly, generate rarely

Harmonize outliers only, keep the hero as it is, keep derived pictures to a
third of the board at most, use strength 0.3 to 0.6, and keep the originals. The harmonize skill says which method fits.

```ops
colour harmonize monsoon japan:412 --method mkl --strength 0.45
```

Generate only to fill an angle the archive came back weak on, at most two per
board and outside the hero role, and say so in the hand-back.

```ops
generate --prompt "rain on a tin roof, close up, grey light" --for-board monsoon --section s1 --role texture --note "The archive has no rain on metal."
```

## 9. Render, look, revise once

```ops
board render monsoon --format png
board render monsoon --layout editorial
```

With pixels on, look at the png and critique it with the board-design skill.
Blind, critique it from `board show` and the metrics. Revise once by removing,
moving and re-noting, then render again.

## A page of its own

When the brief asks for a page designed for the theme, write one
self-contained HTML file (images embedded, no network, system fonts, print
CSS with an `@page` size), save it in your workspace or in `output/boards`,
and attach it. The app's Moodboard tab then shows it under *View page*, with
Save as PDF.

```ops
board page set monsoon output/boards/monsoon.page.html
board pdf monsoon
```

`board render` keeps writing the template layout to `<board>.html` and never
touches the page. `board page clear` detaches it.

## 10. Hand back

The path of the HTML board, its name in Arcana (the person sees it in the
Moodboard tab), and what the archive was missing. Name any derived or
generated picture as such.
