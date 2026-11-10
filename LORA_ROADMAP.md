# Learned looks (LoRA): what would bring them back

On 9 November 2026 the learned-look method left Alchemy's menu, with LoRA training in Sets & labels.
Everything still runs (the `lora train` op, `generative.lora` for recipes and films, the Archive's
learned looks); only the menu entries are hidden.

## Why it is hidden

- The yellow LoRA (latent-lab, SD 1.5, 2000 steps) changes a whole picture by a mean of 4 to 5 out of
  255 at any strength; its training caption changes it more than the LoRA does.
- The water-reflections LoRA trained in Arcana (120 pictures, SD 1.5, 1000 steps at 1e-4, rank 16) drew
  soft pastel blends: 0.40 cosine to its own set in DINOv2, below the base model's stock water (0.51).
  The IP label made from the same set scored 0.78, against 0.81 for real members.
- The looks kept from that work (textured faces, the material atlas) used no LoRA.

A set's style is carried today by its IP label (Search's sets, Alchemy's Sets & labels): picture in a
material, the Generate tab, and the material editor's swatch, drawn from the place or from the set's
own pictures by FLUX.2 klein.

## What to try with the large set at home

1. **Data.** 500 to 2000 pictures of one coherent look, bursts deduplicated, each captioned on its own
   (Arcana's captioner: material, surface, colour, light). Crops at a third of the frame.
2. **A stronger base.** SD 1.5 is the likely ceiling. Train on SDXL (RealVisXL or the base) and, if
   diffusers supports it by then, on FLUX.2 klein 4B, the editor's own model; a 24 GB card fits both.
3. **A gentler run.** Learning rate 3e-5 (latent-lab's working recipe) against 1e-4; rank 16 against 32;
   up to 6000 steps with a checkpoint every 500, chosen by sheet, not by loss.
4. **The cases a label cannot reach,** tested first:
   - goofi, live: a fast LoRA worn by goofi's Diffusion node, against no LoRA, in real time;
   - words: the trigger in a sentence ("a staircase made of …, in ice style"), against the label;
   - a thing rather than a material (one sculpture, one object, one person), where CLIP is weak.

## How to judge a run

`evals/lora_vs_label.py` repeats the three tests: drawing from nothing, closeness to the set in CLIP and
in DINOv2 with fine detail, and grain at 100% on a picture and a face. Then blind pairs of LoRA against
label-only draws, judged by eye.

A learned look comes back to the menu when a run beats the label alone in DINOv2 closeness to its set,
keeps the set's structure at 100%, and wins the blind pairs, in at least one of the cases above.
