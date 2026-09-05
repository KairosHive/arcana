# Licensing plan

Not yet executed. This records what needs doing, what was verified against the
repo and against live model metadata, and what changed after the plan was
written.

**Verified 5 September 2026.** Claims marked ✅ were checked; ⚠️ marks a gap
found during that check.

---

## A0. Stop shipping unlicensed code — blocking ⚠️

`installer/arcana.spec` lines 77–79 bundle `modflows/src` into the built
application:

```python
_modflows_src = os.path.join(ROOT, "modflows", "src")
if os.path.isdir(_modflows_src):
    datas.append((_modflows_src, os.path.join("modflows", "src")))
```

That source comes from `github.com/maria-larchenko/modflows`, which has **no
LICENSE file**. Default copyright applies; there is no redistribution right.

✅ **Confirmed, and worse than the plan states.** Both installers currently on
disk contain it:

| build | unlicensed `.py` files |
|---|---|
| `dist/Arcana` (CPU, 227 MB installer) | 5 |
| `dist/Arcana-GPU` (1,994 MB installer) | 5 |

`encoder.py`, `inference.py`, `lipschitz_constant.py`, `neural_ode.py`,
`utils.py`, under `_internal/modflows/src/`.

One mitigating detail the plan does not mention: **`modflows/` is not tracked in
git** — `git ls-files modflows` returns nothing. The repository does not
redistribute it. Only the built installers do. So this blocks *distribution of
builds*, not publication of the source.

✅ **The checkpoint is genuinely fine.** `MariaLarchenko/modflows_color_encoder`
reports `license: mit` in live HuggingFace metadata, so the 229 MB weights are
MIT and the runtime download is not a problem. The split is real: unlicensed
code, MIT weights.

**The change:** remove the `_modflows_src` block from `arcana.spec`. Keep every
search path in `color_transfer.py::modflows_source_dir()` — the frozen-build
branches simply will not find anything until B2 lands. The existing "ModFlows is
not installed in this build, use the LAB method" path becomes the shipped
behaviour.

**This is a visible regression:** the packaged app loses its best colour
transfer method until B2. That is a product decision, not a purely technical
one, which is why it has not been done unilaterally.

## A1. Declare the licence

**AGPL-3.0-or-later.**

Arcana is a Dash/Flask server, so it can be wrapped as hosted archive-search
SaaS trivially. AGPL section 13 is the only clause that prevents that, and it
costs local users nothing. Sole copyright ownership means licence exceptions can
be sold later without asking anyone. A permissive grant cannot be withdrawn, and
the project has no users yet, so nothing is closed off by choosing AGPL now.

- `LICENSE` at repo root, verbatim GNU AGPL v3.
- `pyproject.toml` under `[project]`: `license = "AGPL-3.0-or-later"` — the SPDX
  expression string, not the deprecated `{file = ...}` table form — plus the
  OSI classifier.
- SPDX header on each file in `arcana/`.
- README replaces "Not yet declared" with AGPL and a pointer to
  `THIRD_PARTY.md`.

✅ Confirmed no `LICENSE` file exists and `pyproject.toml` declares none.

## A2. Third-party notices

Arcana ships no weights, which keeps redistribution obligations off the project,
but display and pass-through obligations still apply.

✅ **Every licence below was read from live HuggingFace metadata**, not assumed:

| model | licence | obligation |
|---|---|---|
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | MIT | attribution |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | MIT | attribution |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | MIT | attribution |
| `laion/clap-htsat-fused` | Apache-2.0 | attribution |
| `MariaLarchenko/modflows_color_encoder` | MIT | attribution |
| `stabilityai/sd-turbo` | Stability AI Community | see A3 |
| `h94/IP-Adapter` | Apache-2.0 | attribution |
| `lllyasviel/control_v11f1p_sd15_depth` | OpenRAIL | see A3 |
| `lllyasviel/control_v11p_sd15_canny` | OpenRAIL | see A3 |
| `Intel/dpt-hybrid-midas` | Apache-2.0 | attribution |
| ⚠️ `runwayml/stable-diffusion-v1-5` | **CreativeML OpenRAIL-M** | see A3 |

⚠️ **The last row is new.** The plan predates the IP-Adapter and ControlNet work
of 5 September 2026, and SD-1.5 is the base model both of those load
(`style_transfer.py`, `IP_BASE_MODEL`). It carries the same OpenRAIL
restrictions as the ControlNets and must be in the EULA flow-through.

**Bundled binaries.** The `opencv-python` wheel bundles FFmpeg; include the LGPL
text and FFmpeg attribution for the frozen build.

## A3. Obligations that reach the UI and installer

**Stability AI Community License** — `sd-turbo`, used by Inject Poetry and by
the `img2img` style method:

- Display "Powered by Stability AI" prominently — the Inject Poetry panel and
  the About text, not buried in a licence file.
- Ship a copy of the agreement to third parties.
- NOTICE file line: `This Stability AI Model is licensed under the Stability AI
  Community License, Copyright (c) Stability AI Ltd. All Rights Reserved`
- Free for commercial use under $1M annual revenue; above that it needs an
  enterprise licence. Record this so it does not have to be rediscovered.

**CreativeML OpenRAIL-M** — the ControlNets **and SD-1.5**. Attachment A use
restrictions must flow through to the end user; reproduce them in the EULA.

**Installer:** add a licence page to `installer/arcana.iss` presenting the
combined EULA — AGPL for Arcana, plus OpenRAIL restrictions and the Stability
notice for the downloadable models.

## A4. Contribution hygiene

Sole copyright ownership is what makes selling exceptions possible. Protect it
before the first outside PR: `CONTRIBUTING.md` requiring DCO sign-off, plus a
CLA. Reuse whatever goofi-pipe already uses rather than drafting new terms.

## A5. Do not do

- **No proprietary in-process plugin on an AGPL core.** The derivative-work
  argument is a mess. If open core becomes the plan, the core has to move to
  MPL-2.0 first, which is file-level copyleft and is what open core actually
  wants.
- **Do not paywall the GPU installer.** Same source, and AGPL lets anyone
  rebuild it.

---

## Ordering

1. **A0** — unbundle `modflows/src`. Blocking for distributing builds.
2. **A1, A2** — LICENSE, SPDX headers, THIRD_PARTY.md.
3. **A3** — Stability and OpenRAIL obligations in UI and installer.
4. **A4** — CLA/DCO, before the first outside contribution.

The colour-transfer rework that permanently resolves A0 is tracked separately in
[COLOR_TRANSFER_PLAN.md](COLOR_TRANSFER_PLAN.md).
