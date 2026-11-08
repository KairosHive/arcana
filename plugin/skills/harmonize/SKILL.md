---
name: harmonize
description: Decide whether and how to bring a picture's colours toward a board, a palette or another picture with Arcana's colour and style transfer. Use when harmonizing a moodboard's outliers, matching one photograph's grade to another, baking a LUT, or choosing between LAB, MKL, IDT, ModFlows, texture transfer, IP-Adapter and ControlNet.
---

# Harmonize

A transfer makes a new picture. The original stays as it was, and the result
is marked as derived on every board, so harmonizing is cheap to try and easy
to undo. It is still a choice to use sparingly.

## When to leave a picture as it is

- The picture is the hero. The hero sets the tone, and the board answers it.
- The picture is off-palette on purpose, as a counterpoint.
- More than a third of the board would be derived. The board would then be
  about the edits.
- The difference is one of light. A dark picture among light ones needs a
  different picture, and a colour transfer leaves its darkness as it was.

## Which colour method

`colour methods` lists them with their cost on this machine.

```ops
colour methods
```

| method | use it for |
|---|---|
| lab | a quick, safe nudge. It can leave a cast, since it treats channels one at a time. |
| mkl | the default. It matches the whole colour covariance and fixes LAB's casts, instantly. |
| idt | taking on a palette completely, distribution and all, in about a second. |
| modflows | the most faithful to a reference's mood, for a final picture. Slow on a CPU. |

Strength 0.3 to 0.6 keeps each picture itself. Full strength is for matching
a grade exactly.

```ops
colour transfer japan:412 --reference japan:1423 --method mkl --strength 0.45
colour transfer japan:412 --palette '#1f2a36:0.6' --palette '#b5651d:0.4' --method idt --strength 0.5
colour harmonize monsoon japan:412 japan:77 --method mkl --strength 0.45
```

`colour harmonize` without a reference uses the board's own palette, places
each result beside its original as a derived support, and refuses the hero.

The lightness and saturation filters choose which reference pixels count, so a
bright sky or a black shadow stays out of the transfer:

```ops
colour transfer japan:412 --reference japan:1423 --method mkl --l-min 15 --l-max 85
```

## Beyond colour

```ops
style methods
style transfer japan:412 --source japan:1423 --method texture --strength 0.5
style control-map japan:412 --kind depth
style transfer japan:412 --source japan:1423 --method controlnet --control depth --strength 0.5
```

- **texture**: borrows grain and detail and keeps every pixel. Safe.
- **ip_adapter**: regenerates the picture in the source's look. Strong. The
  subject survives up to about 0.5.
- **controlnet**: IP-Adapter held to a map of the target. Use it only when
  you mean to restyle, and check the map first with `style control-map`.

## A grade to take elsewhere

```ops
colour lut japan:412 --reference japan:1423 --n 33
```

The .cube file grades other photographs or footage the same way in Resolve,
Premiere or Photoshop.
