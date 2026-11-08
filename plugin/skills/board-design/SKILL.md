---
name: board-design
description: Judge and improve the design of a moodboard built in Arcana. Use when critiquing a rendered board, choosing a board layout, deciding which picture is the hero, cutting a board down, or reading the metrics `board show` reports. Covers hierarchy, negative space and density, value before hue, palette proportion, temperature and scale contrast, and the path the eye takes.
---

# Board design

A board argues one thing. Design is what makes that argument readable at a
glance, before anyone reads a caption.

## Hierarchy

One hero, a few supports, then textures and details for rhythm. The hero is
the picture the rest answer, and it alone leads in size, contrast and
saturation. Two strong pictures fighting for the eye is the most
common failure, and `board show` names it ("2 heroes").

| role | what it does | share of a board |
|---|---|---|
| hero | sets the tone | exactly one |
| support | carries an angle | about half |
| texture | material, surface, grain | a quarter at most |
| detail | a close look that rewards the second glance | a few |
| counterpoint | sharpens the theme by contrast | one or two |

## Negative space and density

Twelve to twenty-four pictures. Past thirty the eye stops reading and starts
scanning. If a section has more than six, cut it to the best four.

## Value before hue

Squint at the png. The board should still read as light against dark (or the
reverse) in a clear pattern. If everything is mid-grey, cut the mid-tone
pictures that add nothing. Value structure carries the mood before colour does.

## Palette proportion

Dominant, secondary, accent, roughly 60, 30, 10. `board palette extract`
measures the board's own proportions. An accent that covers a third of the
board has become a secondary colour. Harmonize the outliers or cut them.

## Temperature and scale

Warm against cool, wide against close. A board all at one distance is flat, so
check that wide, mid and detail scales all appear.

## The path of the eye

The editorial layout reads hero first, then each section left to right. Put
the strongest support first in its section and the quietest last, so each
section closes on a quiet note.

## Reading the metrics

```ops
board show monsoon
board render monsoon --format png
```

- **near-duplicate pairs**: keep the better one of each pair, always.
- **diversity** (mean distance between pictures, 0 to 1): under 0.3 the board
  repeats itself, and over 0.8 it risks falling apart.
- **palette spread** (mean colour difference between pictures' main colours):
  low means a tight palette, high means the board needs its palette argued.
- **derived share**: above a third, the board is about the edits more than
  the archive.

## Choosing a layout

- **editorial**: one hero across the top, sections by role. For a board that
  argues something. The default.
- **masonry**: pictures at their own proportions in columns. For a board of
  many verticals, or a loose collection.
- **grid**: even squares. For comparing like with like (palettes, textures).

```ops
board render monsoon --layout masonry
```

## Critique in these terms

Say what the board argues, then what weakens it, naming pictures by number:
"#6 competes with the hero, so make it a detail or cut it", "s3 has no wide
shot". Then change it, render again, and stop after one revision unless the
person asks for more.
