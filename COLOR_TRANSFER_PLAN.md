# Colour transfer: proposed rework

Partly executed. Goal: every shipped method is either licence-free or under a
licence Arcana can sell under, while quality improves rather than regresses.

This also permanently resolves [A0](LICENSING.md#a0-stop-shipping-unlicensed-code--blocking-).

**Verified against the repo 5 September 2026; status updated 20 October 2026.**

| | | status |
|---|---|---|
| B0 | a registry of methods | moved to [AGENT_PLAN.md, P4](AGENT_PLAN.md#p4-colour-transfer-as-one-table-plus-the-classical-tier-about-2-days) |
| B1 | MKL and IDT, the classical tier | moved to [AGENT_PLAN.md, P4](AGENT_PLAN.md#p4-colour-transfer-as-one-table-plus-the-classical-tier-about-2-days) |
| B2 | our own ModFlows runtime | ✅ done in `78d11d3` (`arcana/modflows_net.py`) |
| B3 | ColorFM-O | not started |

---

## B0 and B1. Moved

Both now live in [AGENT_PLAN.md as P4](AGENT_PLAN.md#p4-colour-transfer-as-one-table-plus-the-classical-tier-about-2-days),
with the registry spec, the two methods, their tests and current line numbers.
They moved because the agent's `harmonize` tool needs the registry: a dropdown
and an agent both choosing methods is the third copy of the `method == "lab"`
chain that B0 was written to prevent.

## B2. Reimplement the ModFlows runtime, keep the MIT weights — *done*

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

1. **B0** registry, then **B1** classical tier: now P4 of AGENT_PLAN.md.
2. **B2** ModFlows reimplementation: ✅ done.
3. **B3** ColorFM-O: largest piece, do last. It is one more registry entry, so it
   waits for P4.

The old caveat, that the packaged app would have only Reinhard LAB between A0
and B2, no longer applies: B2 landed, so ModFlows ships as our own code. B1
still matters for a different reason: of the two methods shipped today, only LAB
is instant on a CPU.
