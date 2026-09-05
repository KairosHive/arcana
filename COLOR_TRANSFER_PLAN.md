# Colour transfer: proposed rework

Not yet executed. Goal: every shipped method is either licence-free or under a
licence Arcana can sell under, while quality improves rather than regresses.

This also permanently resolves [A0](LICENSING.md#a0-stop-shipping-unlicensed-code--blocking-),
which currently blocks distributing builds.

**Verified against the repo 5 September 2026.**

---

## Where it stands

Two methods, both reached through a hardcoded dropdown:

| method | where | licence surface |
|---|---|---|
| ModFlows (neural) | `color_transfer.py::transfer_colors` | ⚠️ **unlicensed source**, MIT weights |
| LAB (Reinhard) | `lab_transfer.py::lab_color_transfer_pil` | none — own code |

✅ The method choice is two literals at `arcana.py:1606-1607`, and the branches
are spread across at least six places: `3270`, `3464`, `3700`, `3702`, and the
filename and PNG-metadata writes further down. Adding three more methods to that
shape is not viable — hence B0 first.

---

## B0. A registry, before anything else

Create `arcana/transfer_methods.py` following `models.py` exactly. ✅ That is
already the house pattern: a frozen dataclass, a tuple ordered cheapest first,
`BY_ID`, `get()`, `for_modality()`, `default_for()`.

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
    blurb: str
```

Then the dropdown is built from the registry, parameter gating keys off registry
fields rather than `method == "lab"`, the filename suffix and PNG metadata read
`id` and `label`, and the availability checks collapse to one lookup.

`cpu_ms` and `gpu_ms` get **measured**, the way `models.py` timings were, so the
UI can say "about 20 seconds" instead of spinning silently.

## B1. Classical tier — no licence surface at all

Two methods from papers, both short, both instant on CPU, both written here so
neither carries any obligation.

- **MKL / linear Monge–Kantorovitch.** Pitié and Kokaram, CVMP 2007. Closed
  form: match the content distribution's 3×3 covariance to the reference's via
  the symmetric square root. ~20 lines of numpy.
- **IDT / N-dimensional PDF transfer.** Pitié, Kokaram and Dahyot, ICCV 2005 and
  CVIU 2007. Iterative: random rotation, 1-D histogram match per axis, rotate
  back, repeat. ~60 lines, plus the grain-suppression post-process from the CVIU
  paper.

`pengbo-learn/python-color-transfer` (MIT) implements all three and is a fair
reference to check numerical agreement against — read it to verify behaviour,
write our own.

This gives a good floor that works on any machine with zero download.

## B2. Reimplement the ModFlows runtime, keep the MIT weights

This is what makes A0 permanent rather than a regression.

The inference path is small: of 555 lines upstream, only `encoder.py` (94) and
`neural_ode.py` (149) matter. Write `arcana/modflows_net.py` **from the paper**
(Larchenko et al., AAAI 2025, arXiv 2503.19062), not by copying:

- Encoder is `torchvision.models.efficientnet_b6(num_classes=k_dim)` for the B6
  checkpoint, `efficientnet_b0` for B0. The head output splits into the weights
  and biases of a small MLP parameterising the velocity field, at boundaries
  `input_dim*hidden`, `+hidden`, `+output_dim*hidden`, `+output_dim`.
- Preprocessing resizes to 528 (B6) or 256 (B0) using the corresponding
  `EfficientNet_B*_Weights.IMAGENET1K_V1.transforms()`.
- Integration is a rectified-flow forward pass in RGB; `strength` sets how far
  along the interpolation curve to travel, `steps` the solver.

**The binding constraint:** the MIT checkpoint is a `state_dict` keyed by
torchvision's own parameter names, so the module structure has to produce
matching keys. Build against torchvision's public API and prove it with
`load_state_dict(..., strict=True)`. A test must assert strict loading succeeds
*and* that a fixed input reproduces a stored reference output within tolerance —
otherwise a silently-wrong reimplementation looks like it works.

Then restore bundling: `modflows_net.py` is our own code and ships normally.
Delete `ARCANA_MODFLOWS_DIR`, `modflows_source_dir()` and the `_candidate_dirs()`
search from `color_transfer.py`. ✅ Keep `download_checkpoint()`,
`checkpoint_dir()` and `CHECKPOINT_URL` unchanged — they are correct and the
weights are MIT.

Worth doing anyway: ask Maria Larchenko (Skoltech) to add MIT or Apache-2.0 to
the GitHub repo. Academic authors usually say yes. If she does, this
reimplementation was still worth it, because we then own what we ship.

## B3. ColorFM-O as the quality option

ColorFM, ECCV 2026, arXiv 2607.07119, `github.com/cszn/ColorFM`. **Code is
Apache-2.0.**

Take **ColorFM-O only** — it needs no pretrained checkpoint at all, fitting a
velocity field per image pair. Nothing to download, nothing to license.

**Not ColorFM-L:** its checkpoint is CC-BY-NC-4.0, non-commercial, which poisons
any paid tier. Retraining is not realistic — 237,408 generated triplets.

From the paper: two-layer bias-free MLP, 512 hidden units, Swish; hierarchical
colour coupling with `D_max = 3`; Adam at lr 5e-4, 700 steps, batches of 4096
sampled pixels; midpoint solver, 5 inference steps (metrics saturate there).

**The segmentation dependency has to be replaced.** The paper uses SegFormer-B5
on ADE20K, and NVIDIA's weights are NVIDIA Source Code License-NC —
non-commercial. Do not ship them.

Masks only guide distribution pairing; they do not define hard regional
transforms. The paper reports style similarity dropping only from 0.745 to 0.722
under 40% mask misclassification, so a coarse permissive substitute is fine. In
order of preference:

1. **Coarse regions from CLIP patch tokens.** Arcana already computes CLIP
   embeddings and already ships the encoder, so this adds zero download and zero
   new licence. Try first.
2. Any Apache-2.0 or MIT segmentation checkpoint — verify the licence on the
   *specific checkpoint*, not the architecture.
3. **No masks.** The paper's own ablation (`D_max` 0 vs 3) shows hierarchical
   coupling carries most of the benefit. Ship this as fallback.

**Benchmark before committing.** The published 19.3 s is on a 4090 and is
dominated by SegFormer-B5. The actual optimisation is a tiny MLP over 4096
sampled pixels and should be seconds on CPU. If it is not, the method is GPU-only
and the registry must say so.

## B4. Explicitly rejected

Recorded so nobody re-evaluates them:

- **SA-LUT** (ICCV 2025) — S-Lab License 1.0, non-commercial. Also worst style
  score in the ColorFM benchmark at 0.381.
- **ColorFM-L weights** — CC-BY-NC-4.0.
- **Neural Preset** (CVPR 2023) — code never released, mobile app only.
- **D-LUT** (82.8 s) and **NLUT** (18.9 s) — slower than what is kept, no
  better. NLUT's LICENSE is MIT text with a copy-pasted wrong copyright header.
- **CAP-VSTNet** and **WCT2** — licences fine (MIT), but Lipschitz constants of
  50.3 and 35.3 against 2.8 for the flow methods means visible banding, and both
  need a VGG backbone.

Watch, do not build on: **Hist2Style** (arXiv 2606.01819) — bilateral grids with
locally affine constraints. Too new for a licence or released code, but the most
likely to be both CPU-fast and artifact-free.

## B5. The shipped set

| method | download | licence surface | when |
|---|---|---|---|
| Reinhard LAB | 0 | none | instant, exists today |
| MKL linear | 0 | none | instant, better than Reinhard |
| IDT / PDF transfer | 0 | none | strong classical baseline |
| ModFlows | 229 MB | MIT weights, our code | strongest style match |
| ColorFM-O | 0 | Apache-2.0, our code | best content/style balance |

Three of five need no download and carry no third-party terms.

---

## Ordering

1. **B0** registry — everything else depends on it.
2. **B1** classical tier — cheapest quality win, no risk.
3. **B2** ModFlows reimplementation — closes A0 permanently.
4. **B3** ColorFM-O — largest piece, do last.

B0 and B1 can land immediately and independently of any licensing decision. B2
is what lets the packaged app keep its best method after A0 removes the bundled
source.

## One caveat on sequencing

A0 and B2 are in tension. A0 removes ModFlows from the build **today** and B2
restores it, but B2 is the second-largest item here. Between them, the packaged
app has only Reinhard LAB — which is why B1 is worth landing early even though
it is not on the critical path: MKL and IDT are hours of work and would leave
the interim build with three usable methods instead of one.
