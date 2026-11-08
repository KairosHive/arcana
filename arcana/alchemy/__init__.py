"""Alchemy: pictures from the collection mixed, mapped, recoloured and restyled into new pictures.

One pipeline, five stages: the inputs (a picture [T] and up to three materials with weights, from the
same collection the Moodboard gathers), a method from one of four families (pixels, tokens, colour,
generative; methods.py), where it acts (a mask of presets and prompts; mask.py), the finish (a basic
Lightroom toolkit; finish.py), and recipes that hold any subset of those settings (recipes.py).
pipeline.run() takes one step through all of it. The ops in arcana/ops/alchemy.py are the only way
the tab, the command line and an agent reach it.

The engine is a curated port of latent-lab (RAE and FLUX.2 VAE token spaces, the pixel mapping
modes, the mask models, Real-ESRGAN), with its memory manager: models load on first use and are let
go when memory runs short (memory.py).
"""
