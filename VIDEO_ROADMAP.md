# Roadmap: modalities and video

Plan written 9 November 2026 against `3ae5e90`. Everything below is proposed work.

Arcana indexes photos with CLIP and sounds with CLAP, one vector per file. This
plan makes modality a declared property of every feature, adds video as a third
modality built on timed sections, and extends Alchemy and Film to moving
pictures. It sits beside [ROADMAP.md](ROADMAP.md) and builds on the ops
vocabulary of [OPS_PLAN.md](OPS_PLAN.md).

## The sequence

| Phase | Delivers | Depends on | Size |
|---|---|---|---|
| 1 | Modality contract, tags on every feature, the audio defects fixed | | medium |
| 2 | Live state, every control an op, the person sees agent work as it happens | | medium |
| 3 | Encoder spaces, EmbeddingGemma 2, cross-modal queries between photo and audio archives | 1 | medium |
| 4 | Video foundation with decoding, proxies, a player, sections and handles | 1 | medium |
| 5 | Video indexing with detected sections at nested levels, three channels, the dataset builder | 3, 4 | large |
| 6 | Video search with granularity, a content and dynamics balance, sections and whole videos | 5 | medium |
| 7 | Film with clips, an audio lane, drops from search, export with sound | 2, 4 | large |
| 8 | Video Alchemy with the pattern and rhythm engine, static and dynamic colour, looks over time, tracked masks | 7 | large |
| 9 | Morphs between videos, diffusion looks with temporal integration, generative video | 8 | large |

Phase 1 is built (see Status at the end). Phases 1 and 2 are independent and
improve the photo and audio app on their own.
Phase 3 gives sound-to-image and image-to-sound queries before any video code
exists.

---

# Part 1. Modality as a declared property

## 1.1 Where modality lives today

A dataset carries a modality, `image` or `audio`. The id suffix `@audio` marks
audio sets, and the same suffix logic is written three times
(`ops/archive.py:78-85`, `api/__init__.py:119`, `aliases.py:22`). Dataset
discovery uses a regex with `(image|audio)` written in two places
(`core.py:647-648`, `ui_datasets.py:233`).

The frontend holds three modality guards, `isAudio` in `SearchTab.svelte:171`,
`isImage` in `MoodboardTab.svelte:62` and an extension check in
`MediaCard.svelte:28`. The backend holds one explicit refusal, in
`api/search.py:472`. Every other photo-only path is gated silently inside
indexing or left open.

An op is declared once in `ops/registry.py` with a `needs` set of capability
flags (`gpu`, `diffusers`, `pixels`, `ui`). `needs` is checked in `call()` and
printed in `--help`, `/ops/list` and the JSON schema. Modality has the same
shape as these flags and is absent from the declaration.

## 1.2 The capability matrix today

Scope says what the feature reads. A **dataset** feature follows the dataset
chosen in the context bar. An **item** feature follows the pictures in the
collection, which can come from several datasets.

| Feature | Tab | Scope | Works with | On audio today | Video arrives |
|---|---|---|---|---|---|
| Text search | Prompt Search | dataset | photo, audio | works, `core.py:364` branches | phase 6 |
| Latent map, hover, click | Prompt, Story | dataset | photo, audio | hover shows waveform or spectrogram | phase 6 |
| Group twins | Prompt, Moodboard | dataset | photo, audio | works on vectors alone | phase 6 |
| Marks filter (stars, colour tags, dates) | Prompt, Story | dataset | photo | bar hidden, marks read for photos only (`db.py:2455`) | later, with video sidecars |
| Spectrogram toggle | Prompt, Story | dataset | audio | shown | phase 6, for clips with sound |
| Story, one search per line | Story | dataset | photo, audio | works | phase 7 |
| Inject Poetry | Story | dataset | photo | hidden in the page, `/api/story/poetry` accepts audio unchecked | |
| Save story | Story | dataset | photo | writes `story.txt` and skips every sound silently (`search.py:403`) | phase 7 |
| Save selected | Prompt | dataset | photo, audio | works | phase 6 |
| Similarity search (palette, style) | Moodboard | dataset | photo | guarded on both sides | phase 8, on keyframes |
| Palette strip on cards | cards | item | photo | hidden | phase 6, on keyframes |
| Collection, boards, export, PDF | Moodboard rail, cards | item | photo | accepted, thumbnails return 404 (`media.py:193`, `Thumb.svelte:20`) | phase 4 |
| Reference and Target roles, colour transfer | Moodboard | item | photo | open | phase 8 |
| Alchemy step (pixels, tokens, colour, generative, mask, finish, recipes) | Alchemy | item | photo | PIL error from `alchemy/inputs.py:84` | phase 8 |
| Texture LoRA, style transfer, ControlNet, IP-Adapter | Alchemy | item | photo | open | phases 8 and 9 |
| Film (picture lane, looks, lanes, templates, render) | Film | item | photo | open | phase 7 |
| Archive tree | Archive | output | pictures, mp4 | sounds saved from Prompt Search are absent (`catalogue.py:48`) | phase 4 |
| Wizard modality switch, encoder list | Datasets | dataset | photo, audio | works | phase 5 |
| Wizard extras (palette, style, thumbnails) | Datasets | dataset | photo | shown, ignored by the backend (`db.py:2402`) | phase 5 |
| Rework, Add new, Delete, Relocate | Datasets | dataset | photo, audio | works | phase 5 |
| Marks refresh | Datasets | dataset | photo | button off (`ui_datasets.py:227`) | later |
| Agent terminal | Agent | any | any | | |

## 1.3 Defects the matrix exposes

These are fixed in phase 1 as the contract lands.

- Feature files carry the dataset name only (`features_{name}_palette.npz`). An
  image set and an audio set with one name share them, and
  `existing_feature_paths` (`db.py:1505`) writes the photo palette into the
  audio bundle. Photo datasets keep that name, so nothing on disk moves. Every
  other modality uses `features_{name}_{modality}_{kind}.npz`, and each reader
  asks for the file by modality.
- Save story on an audio dataset reports success with an empty folder.
- `/api/story/poetry`, the board ops, the Alchemy inputs and the film ops accept
  sounds and fail later with a PIL error or a broken thumbnail.
- The Archive tree counts `PICTURE_EXT` and `VIDEO_EXT`, so saved sounds are
  invisible there.
- The wizard offers palette, style and thumbnails for audio and the backend
  skips them.

## 1.4 The contract

**One vocabulary.** `models.py` holds `IMAGE`, `AUDIO` and `VIDEO`. One helper
builds and splits dataset ids, and one pattern discovers dataset files. The
three copies of the suffix logic and the two regexes call them.

**Ops declare their modalities.** `Op` gains `modalities: frozenset[str]`
beside `needs`. An op that takes a dataset or pictures must declare it, and
`register()` refuses the op otherwise. An op that takes no media (orient, undo,
job list) declares nothing. `call()` resolves the modality of each dataset and
handle argument and raises one `Refusal` when it falls outside the set. The message names the op, the modality
it received and the modalities it accepts. The tag prints in `--help`,
`op list`, `/ops/list` and the schema, so an agent reads the same map as the
page.

**A capability table for the page.** Search, story, map, marks, similar and save
go straight to `core` and `db` by design (`OPS_PLAN.md:470`). A table in
`arcana/capabilities.py` lists every user-facing feature with its id, tab,
scope, modalities and the ops it rests on. Features backed by ops inherit their
modalities from the op, so each fact is written once. `/api/bootstrap` serves
the table, and the routes in `api/search.py` check it through one decorator.

**Two scopes.** A dataset-scoped feature is checked against `current.modality`.
An item-scoped feature is checked against the kind of each item it receives, so
a collection holding photos and sounds enables colour transfer on its photos
and refuses its sounds one by one.

## 1.5 The UI rules

- **Tags.** A `ModalityTag` component renders small chips labelled `photo`,
  `audio` and `video` from the capability table. The label `photo` stands for
  the id `image`. The chips reuse the existing `Chip` tones, with a new
  `--arc-video` token.
- **Tabs.** Each tab button carries one dot per modality it supports. A tab
  with zero features for the current dataset dims and states the reason on
  hover, such as "Alchemy works with photos. This dataset holds sounds."
- **Panels and controls.** Every panel header carries its tags. A control
  outside the current modality stays visible and disabled, with the same
  one-line reason, so the whole map of the app is learnable from any dataset.
  Decorative elements such as the palette strip simply hide.
- **Dataset menu.** The context bar groups datasets by modality and shows the
  chip beside each name, as the Datasets list already does.
- **Cards and thumbnails.** `MediaCard`, `Thumb` and `Viewer` choose their
  renderer from the item kind (picture, waveform, clip strip) through one
  function, replacing the extension checks.
- **Wizard.** The extras step lists only the features of the chosen modality.

## 1.6 Done when

- `arcana op list` prints a modality tag on every op, and the page shows a tag
  on every tab, panel and card action.
- A parity test runs every op and every `api/search.py` route against a
  synthetic photo set and a synthetic audio set from `tests/synth.py`. Each
  call either succeeds or returns the standard refusal.
- The five defects of 1.3 have regression tests.
- Adding a modality means adding one constant, one renderer and tags. Phase 4
  uses this to bring in `video`.

---

# Part 2. Video as a modality

## 2.1 What the survey found

| Family | Examples | What the vector carries | Text query | Role here |
|---|---|---|---|---|
| Image tower on sampled frames, pooled | CLIP ViT-H/14 (installed), SigLIP 2 | content of single frames | yes | content channel in the space of the existing photo archives |
| Video and text dual encoders | InternVideo2, InternVideo-Next, VideoPrism, Perception Encoder, X-CLIP | content, with partial motion sensitivity | yes | candidate dynamics-aware channel, chosen by measurement |
| Omni embedders on a multimodal LLM | EmbeddingGemma 2, Qwen3-VL-Embedding, jina-embeddings-v5-omni, Ovis-Omni-Embedding, Omni-Embed-Nemotron | content of text, image, video and audio in one space | yes | the shared space for cross-modal queries |
| Audio, video and text contrastive | PE-AV (Meta) | content plus sound correspondence | yes | alternative shared space when sound leads |
| Self-supervised video | V-JEPA 2 | motion, 77.3 top-1 on Something-Something v2 with a probe | none | motion query by example |
| Hosted embedding APIs | | | | excluded, media stays on the machine |

**EmbeddingGemma 2 is the model named in the brief.** Google released it on
6 October 2026 under Apache 2.0. It maps text, code, images, video and audio
into one 768-dimension space, with Matryoshka sizes 512, 256 and 128. It has
740M parameters (270M text, 170M vision, 300M audio) and an 8,192-token context.
Video is read as frames at 1 frame per second by default, 140 tokens per frame,
with `max_frames` 32 by default and about 58 frames at the default budget. Audio
is read as 16 kHz mono at 25 tokens per second, about 327 seconds at most. It
loads through `sentence-transformers` with `model.encode({"video": path})`. It
requires bfloat16 or float32, and float16 returns NaN silently. Its model card
reports 50.67 Hit@1 on the video part of MMEB-v2.

**Video retrieval scores sit close together and come from different protocols.**
Qwen3-VL-Embedding reports 53.6 (2B) and 58.7 (8B) on MMEB-v2 video retrieval in
its own README and covers image and video without audio. The MVEB benchmark
(23 tasks) concludes that every family leads somewhere, with LLM-based embedders
ahead on classification and QA and binding models ahead on retrieval.

**Video embeddings are dominated by appearance.** The ViTAL-X preprint measures
cosine similarity above 0.91 between a clip and the same clip in opposite
temporal order across existing video and text models, and finds that retrieval
on MSR-VTT plateaus after 4 to 8 frames. The RTime benchmark reports that models
retrieve videos with the right look and the wrong temporal meaning. At 1 frame
per second a 3-second clip contributes three frames, so short-range motion
(shake, speed, flicker, rhythm) stays outside what EmbeddingGemma 2 receives.
This confirms the concern in the brief. Dynamics need their own channel.

**Shot detection has two practical tools.** PySceneDetect runs
on the CPU and its 0.7 release handles variable frame rate. TransNetV2 is a
small neural detector that scored 87.0% against 65.5% for PySceneDetect in the
CineTrans authors' own test, with the gain concentrated on gradual transitions.

## 2.2 The design in one paragraph

A video dataset is a table of timed sections at several nested scales. Arcana
samples every video twice per second, finds where the footage changes, and cuts
it into sections that follow those changes, so a section lasts as long as what
it shows. Each section is encoded directly as a clip, from its first frame to
its last, and carries one vector per channel (content, dynamics, sound). The
whole file always gets its own vectors. A query picks a scale or searches all of
them, and results come back as sections, whole videos or both.

## 2.3 Sections and granularity

**Ticks.** Each video is sampled every 0.5 seconds. A tick holds the CLIP image
vector of its frame and the motion descriptors of the half second around it
(2.4). Ticks are the signal that sections are detected from. 100 hours of
footage is 720,000 ticks, about 85 minutes of ViT-H/14 forward time on this GPU
at the 7.1 ms per frame recorded in `models.py`, with decoding on top.

**Boundaries.** For each video, agglomerative clustering restricted to temporal
neighbours merges ticks into a tree, up to the whole file. scikit-learn, already
a dependency, provides it. Every point between two ticks gets a boundary
strength, the height in the tree at which its two sides merge. A strong boundary
marks a place where content or motion changes and stays changed. A gradual
change inside one take, such as a slow tilt from water to sky, produces a
boundary at the point of greatest change, so continuous footage is sectioned as
well as edited footage. A hard cut found by the cut detector raises the strength
of the boundary where it falls, so a section edge lands exactly on the cut, and
similar shots on both sides can still join at coarse levels.

**Sections.** A sensitivity threshold on boundary strength gives the sections.
The threshold is relative to each video's own range of strengths, so a uniform
video yields few sections and a varied one yields many. A minimum length,
2 seconds by default, absorbs brief disturbances into their neighbours. The
example in the brief is a 3-minute file with 10 seconds of water reflections,
30 seconds of sky and 12 seconds of someone walking. It comes out as sections of
those lengths, with each boundary placed within half a second.

**Levels.** One tree gives nested sections at several sensitivities. The builder
stores three by default, plus the whole file.

| Level | Holds | In the example |
|---|---|---|
| Whole file | the video as one unit, always stored | the 3-minute file |
| Coarse | the main parts of a video | water, sky, walking |
| Medium | changes within a part | the sky clearing, the walker entering the frame |
| Fine | single gestures and moments, down to the minimum length | one ripple crossing the reflection |

The number of levels and each sensitivity are build settings. A dataset of
single-shot clips can keep the whole file and one level.

**Section embeddings.** Every section at every stored level is encoded directly
by the omni encoder as a clip. Frames are spread uniformly over the section, at
4 per second for short sections and capped at 32 frames, the model's default
budget. A 10-second section is read at about 3 frames per second and a
30-second section at 1 per second. Each section also gets its own sound vector,
dynamics vector and motion summary, and its CLIP vector is the mean of its
ticks. The whole file is encoded the same way from 32 uniformly spaced frames.

**Granularity at query time.** The query box offers Whole files, Coarse, Medium,
Fine and Auto. Auto searches every level and keeps, among nested sections, the
one with the best score. A 10-second shot of water reflection then comes back as
one result at its natural length. A duration filter ("at least 8 seconds",
"5 to 15 seconds") narrows results by length.

**Whole videos.** A video is ranked by the better of its whole-file score and
its best section. Its card shows where in the file the match sits.

**Re-sectioning.** Ticks are stored. Rework re-reads them with new
sensitivities and encodes only the sections whose boundaries moved.

## 2.4 The channels

**Content.** Two encoders can fill this channel, and a dataset can hold both.

- The omni encoder embeds each section as a clip, as 2.3 describes. Text,
  photo, sound and clip queries all land in this index.
- The installed CLIP image tower gives each section the mean of its tick
  vectors. This index shares a space with existing photo archives built on the
  same CLIP, so one query can span `japan` and a video set. The middle frame of
  each section feeds the palette, style, cluster naming and map code unchanged.

**Dynamics.** This channel is built in three layers, each useful alone.

1. **Motion descriptors.** Optical flow on the proxy (256 px, 12 fps) gives a
   vector of about 48 numbers per tick, summarised per section as mean, spread
   and trend. A homography fit separates camera
   motion (pan, tilt, zoom, roll, shake) from subject motion. The vector also
   holds motion energy and its envelope, a direction histogram, where in the
   frame motion happens, periodicity from autocorrelation, cut density,
   brightness flicker and the rate of colour change. The descriptors are
   deterministic, cheap and readable. They follow the pattern of the palette
   and style features, stored as `features_{name}_video_motion.npz`. They give
   facets (static, pan, handheld, slow, fast), sliders, a map arranged by
   motion, and "find clips that move like this one".
2. **Motion words.** A vocabulary of motion phrases ("slow pan left", "locked
   off", "handheld", "pulsing", "fast zoom in") maps to descriptor prototypes.
   A text query is matched to the vocabulary with the text encoder and scored
   against the descriptors. This gives text-to-dynamics search with no extra
   model.
3. **A learned dynamics vector.** Two candidates fill it, and the probe of 2.9
   picks one. A local vision-language model writes one line about camera and
   subject motion per section, seeded with the descriptor values, and the line is
   embedded as text and shown under the result as evidence. Or a motion-sensitive
   encoder (InternVideo-Next, Perception Encoder, V-JEPA 2 for example queries)
   embeds each fine section at 8 frames per second.

**Sound.** The audio track of each section is embedded by the omni
audio encoder, in the same space as the content vectors. The EmbeddingGemma 2
documentation describes video input as frames, so the track is embedded as its
own vector. A clip is then findable by how it sounds, and a sound query reaches
both the picture and the track.

**The query balance.** A slider `b` from 0 (content) to 1 (dynamics) weights
the two channel scores. Each channel score is standardised against the dataset
before mixing, since raw cosines from different indexes have different ranges.

```
score(section) = (1 - b) * z_content(section) + b * z_dynamics(section)
```

Three presets cover the brief, Content (0), Both (0.5) and Dynamics (1). A
query can also give each channel its own phrase, such as "wet street at night"
for content and "slow handheld drift" for dynamics.

## 2.5 Encoder spaces and cross-modal queries

**Spaces.** `ModelInfo.modality` becomes `modalities`, the kinds of input the
encoder accepts, and gains `space`, the id of its vector space. Vectors are
comparable when they share a space. CLIP ViT-H/14 gets space `clip-vit-h-14`
with modalities image and video keyframes. CLAP gets `clap` with audio.
EmbeddingGemma 2 gets `embeddinggemma-2` with image, audio and video. A further
field holds the precision each model accepts, since `gpu.py` selects fp16 for
this machine and EmbeddingGemma 2 needs bfloat16.

**Several indexes per dataset.** A dataset keeps its modality and gains one
index per space. `dataset extend --space embeddinggemma-2` adds an index to an
existing dataset and leaves the others untouched, as Rework does today. The
photo and audio archives already on disk gain the shared space this way.

**Queries across modalities.** A query is a weighted list of terms. A term is a
phrase, a handle (a photo, a sound, a clip span) or a file dropped on the box.
Every term is encoded into the space of the index being searched and the scores
are summed with their weights and signs. This is item 2 of ROADMAP.md (query
algebra) with mixed term kinds. A sound finds photos, a photo finds sounds, and
"this clip, more fog, minus people" is one query.

**Score calibration.** In a shared space, similarities between different
modalities have a lower mean than similarities within one modality. Each term's
scores are standardised across the dataset before the weighted sum, so a text
term and a sound term contribute in proportion to their weights.

**Searching several datasets.** The context bar gains a multi-select limited to
datasets that hold an index in a common space. Results interleave photos,
sounds and clips with their kind chip.

**Prerequisites.** `pyproject.toml:79` pins `transformers<5` because version 5
changed `CLIPModel.get_image_features`. The EmbeddingGemma 2 guides install the
current transformers release, so phase 3 starts by updating the CLIP call sites
and lifting the pin, with `test_models.py` and `test_parity.py` as the guard.
`torchcodec` joins the dependencies for decoding, which `db.py:519-540` already
anticipates for torchaudio 2.9.

## 2.6 Storage

Today a key is one whole file. `encode_into` adds one key per path
(`db.py:715-735`), the bundle matches line i of `items.jsonl` to vector row i,
and byte-identical files collapse by fingerprint (`db.py:2111-2120`).

| File | Holds |
|---|---|
| `ticks_{name}_video.npz` | per tick, its time, its CLIP vector reduced to 64 dimensions, its motion descriptors |
| `tree_{name}_video.npz` | the merge tree and boundary strengths for each video |
| `sections_{name}_video.parquet` | one row per section with key, video id, path, level, parent, `t0`, `t1`, keyframe time, fps, size, `has_audio` |
| `index_{name}_video.{space}.pkl` | one usearch index per space and channel, keyed by section, holding every level and the whole files, with level as a filter |
| `features_{name}_video_motion.npz` | motion summary per section |
| `features_{name}_video_palette.npz`, `_style.npz` | palette and style of keyframes |
| `cache/video/<fingerprint>/` | proxy mp4 (540p, every frame a keyframe, so any frame is reached exactly and at once), thumbnail strip, waveform |

**Handles.** `clips:8120` names a section at any level. `clips:v37` names a
whole video. `clips:v37@12.0-22.5` names a time range, for trims and for spans
marked by hand. Boards, the collection, Alchemy and Film accept all three, and
they store the video fingerprint and the times beside the handle, so their items
survive a re-sectioning. The bundle `Item` gains `t0`, `t1`, the level and the
video id in `extra`, and its fingerprint includes the time range so sections of
one file stay distinct.

**Originals.** Arcana reads videos where they are, as it does photos. Proxies
and strips are a cache that the app rebuilds on demand.

**Size.** With average section lengths of 6, 20 and 60 seconds at the three
levels, 100 hours of footage gives about 84,000 sections. One 768-d float32
index of that size is about 260 MB, and usearch stores half precision at half
that. The reduced tick vectors for the same footage take about 90 MB.

## 2.7 The dataset builder

- **Scan.** The wizard counts videos beside images and sounds (`.mp4`, `.mov`,
  `.mkv`, `.webm`, `.avi`) and reports total duration, since duration drives
  cost.
- **Sectioning step.** A new wizard step shows the levels, their
  sensitivities, the minimum length and the cut detector. The preview computes
  ticks for one sample video and draws its timeline with the sections of each
  level and a thumbnail per section. The sliders regroup the preview live, so
  the sectioning is judged on real footage before the long run.
- **Channels step.** Checkboxes for content (omni, CLIP keyframes), dynamics
  (descriptors, learned vector) and sound, each with a measured time estimate
  in the style of the encoder timings in `models.py`.
- **Indexing.** Decoding goes through one reader that yields frames at a
  requested rate and the audio track as an array. The job is checkpointed per
  video, which also delivers the resumable indexing listed in ROADMAP.md.
- **Add new, Rework, Relocate, Delete.** Add new encodes unseen files. Rework
  re-sections from the stored ticks, encodes the sections that changed, and
  rebuilds the map. Relocate and Delete follow the existing paths plus the
  proxy cache.
- **Ops.** `dataset create --modality video --levels coarse,medium,fine --min-section 2 --channels content,dynamics,sound`,
  `dataset extend --space`, `dataset survey` reporting hours, files and
  sections per level.

## 2.8 The search tabs

- **Result cards.** A clip card shows a thumbnail strip that scrubs on hover,
  plays the proxy between its in and out points, and prints the time range, the
  source file and the motion line. Its actions are Collection, Reference, Add to
  film and Open at this moment.
- **Return mode.** Sections, Videos or Both. Videos mode shows one card per file
  with a heat strip of fine-section scores along its duration, and a click on the strip
  jumps there. Both nests the best sections under each video.
- **Granularity.** Whole files, Coarse, Medium, Fine or Auto beside the query,
  with Auto as the default, and a duration range.
- **Balance.** The Content, Both, Dynamics control of 2.4, and an optional
  second phrase for dynamics.
- **Facets.** Camera motion class, motion energy, duration, has sound, and the
  existing cluster labels.
- **Map.** One point per section at the chosen level, or per video. A
  click plays the clip, hover shows the strip. Arrange-by gains Motion beside
  the CLIP layout, which is item 4 of ROADMAP.md.
- **Evidence.** Each result prints strong, fair or weak per channel, so a clip
  that matches on content and misses on dynamics says so.
- **Ops.** `search text clips "water reflection" --level auto --min-duration 8 --balance 0.2 --return sections`,
  `search like clips field@audio:212`, `item show clips:8120`. In blind
  mode the agent receives handles, time ranges, motion lines and strengths.

## 2.9 Evaluation

Model choices in 2.4 rest on measurements on the person's own footage, stored
in `evals/` beside the moodboard briefs.

- **The sectioning check.** Ten videos are sectioned by hand at the coarse
  level, and the detected boundaries are counted as hits within one second.
- **The dynamics probe.** 200 fine sections are each rendered as four variants,
  reversed, frame-shuffled, frozen on the middle frame, and at double speed. For
  each candidate encoder the probe reports the cosine similarity between the
  original and each variant. An encoder that carries dynamics separates them.
- **The query set.** 40 written queries, half about content and half about
  dynamics, with the top 10 judged by eye for each configuration.
- **The photo check.** The five existing briefs run on `japan` with CLIP ViT-H/14
  and with EmbeddingGemma 2, to place the omni encoder against the current
  default on photo retrieval.
- **Throughput.** Milliseconds per tick and per section for each channel on this machine,
  recorded in `models.py` so the wizard quotes real times.

## 2.10 Done when

- Phase 3. `search like japan field@audio:212` returns photos for a sound, and
  the reverse works. The photo check has run.
- Phase 4. A folder of videos plays in the viewer through proxies, and clip
  handles resolve in boards and the collection.
- Phase 5. A video dataset builds from the wizard and resumes after a cancel.
  A test file made of three joined parts comes out as three coarse sections,
  each with its own vectors. The probe has chosen the learned dynamics vector.
- Phase 6. One query returns sections, whole videos or both. Auto returns each
  match at its natural length, and the duration filter finds a 10-second shot.
  The balance control changes the ranking on the dynamics half of the query
  set.

---

# Part 3. Video in Alchemy and Film

## 3.1 The film takes clips

A film today is a JSON dict with a picture lane, a look lane and number lanes
(`alchemy/film.py`). `Film.frame(t)` produces each frame, and the renderer
pipes raw frames to the ffmpeg binary from imageio-ffmpeg. The film has no
sound, and this renderer is the only video I/O in the code base.

The film model already applies a whole Alchemy step over time. Clips enter it
through one new kind of source.

```json
{
  "pictures": [{"t": 0, "ref": "japan:12"},
               {"t": 4, "ref": "clips@video:8120", "in": 0.0, "out": 18.5, "speed": 1.0, "ease": "linear"}],
  "audio": [{"t": 0, "ref": "field@audio:212", "gain": -6, "fade": [0.5, 2.0]}]
}
```

- **Clip lane.** A clip key holds a handle, in and out points and a speed.
  `Film.frame(t)` decodes the source frame for that time. Looks, number lanes,
  masks and finish then apply to clips exactly as they apply to morphed
  pictures. Picture keys and clip keys share the lane, so a film can pass from
  a still into a moving shot.
- **Audio lane.** Sounds from an audio dataset and the tracks of clips, with
  gain and fades. The renderer muxes the mix into the mp4.
- **Two render qualities.** A preview render reads proxies, and the final
  render reads the originals at full size.

## 3.2 Finding and dropping

- Every clip card has **Add to film**, and cards drag onto the timeline from
  Prompt Search, Story and the Moodboard rail. A docked search drawer in the
  Film tab runs the same query box, so finding and placing happen in one view.
- **Story to rough cut.** Generate Story searches one section per line. A new
  action lays the chosen sections on a film in order, with the story lines as
  markers. Sequencing uses the content distance between consecutive sections
  (the "Sequence a selection" idea in ROADMAP.md) plus motion continuity from
  the descriptors. With an audio dataset in the shared space, each line also
  proposes a sound for the audio lane, which is "Stories with sound" in
  ROADMAP.md.
- **Match cuts.** For two neighbouring clips the timeline proposes the cut
  point where their motion descriptors agree most.

## 3.3 Colour on video

- **Static transfer.** The transform is fitted once per shot on frames pooled
  across the shot, baked into a 3D LUT with `lut.py`, and applied to every
  frame. One transform per shot is temporally stable by construction, and the
  LUT exports as `.cube` for Resolve. All existing methods (LAB, ModFlows,
  harmonize) pass through this path, since each is fitted once and then
  applied as a table.
- **Dynamic transfer.** The reference is a video or an ordered set of photos.
  Its palette is extracted per section with `palette.py`, a LUT is fitted at each
  key time, and the film interpolates between LUTs along the timeline. Three
  alignments are offered, stretch (the reference spans the target), loop, and
  keyed (the person or the agent places palette keys on a lane). A palette lane
  in the film stores the keys.
- **Finish.** The Lightroom-style sliders already animate through lanes and
  apply per frame.

## 3.4 Looks, masks and style on video

- **Temporal class per method.** Each entry in `alchemy/methods.py` gains a
  field stating how it behaves over time. Stable methods (finish, colour, the
  geometric pixel methods) run per frame. Methods that estimate something from
  each frame (guided, analogy, quilt) run with temporal smoothing of their
  estimate. Generative methods run on keyframes and propagate.
- **Grain and noise.** Seeded noise takes a policy, held (one pattern for the
  shot) or animated (a new pattern per frame).
- **Tracked masks.** `mask.py` already uses SAM 2, which propagates a mask
  through a video. A subject or prompt mask is read once and tracked across the
  clip. Geometric layers move through lanes as today.
- **Diffusion looks, by temporal integration.** A diffusion method (texture
  LoRA, img2img, IP-Adapter, ControlNet) run on each frame alone flickers,
  since each frame draws its own noise. Four levels of integration are
  offered, and each method states the ones it supports.
  1. **Keyframe and propagate.** The look is applied to keyframes chosen at
     motion changes, and optical flow carries it to the frames between,
     blended from both neighbouring keys. It reuses every existing method and
     trained LoRA, and it is the default.
  2. **Shared and warped noise.** Every frame starts from one noise field,
     moved along the optical flow, so texture stays attached to what it
     covers.
  3. **Previous-frame conditioning.** Each frame is denoised with the result
     of the frame before it as a condition, as RFDM does for image editing
     models.
  4. **A video model.** The whole section goes through a video diffusion
     model that denoises all its frames together (Wan VACE), which gives the
     strongest consistency at the highest cost.
- **Deflicker.** A last pass compares the processed clip with its source and
  removes brightness and colour changes that the source lacks (Bonneel et
  al.). It applies after any per-frame method.
- **Similarity on video.** Palette and style search in the Moodboard run on
  keyframe features, so "clips with this palette" and "clips with this grain"
  work as they do for photos.

## 3.5 Pattern and rhythm

Procedural editing. A rule places the frames, and the person shapes the rule.
It lives in the Alchemy Video subtab as a new kind of lane, the pattern lane,
with its own editor panel. Looks, colour, masks and finish apply on top of it
like on any other lane.

**The model.** A pattern is a pure function from an output frame number to a
list of layers, each a source and a source frame. It reads only numbers and a
seed, so it renders identically every time, previews without any model, and an
agent can write one as text.

**Sources.** Clips and sections, stills (the hidden images), flat colours
(black and white flashes), and a search, such as "every section matching
water, longest first".

**Voices.** A pattern holds several voices. Each voice has a pulse and an
action.

- **Pulse.** A period and an offset, counted in frames or in beats at a tempo.
  Other pulse forms are a duty cycle (on for 3 frames, off for 5), a Euclidean
  rhythm (5 hits spread over 8 steps), a list written by hand (`x..x.x..`), a
  seeded probability, and a ramp that speeds up or slows down.
- **Polyrhythm.** Voices with different periods run together, such as 3
  against 4 against 5 frames. Phasing is two voices whose periods differ by a
  fraction of a frame, so they drift apart and realign over a known number of
  seconds.
- **Action.** Cut to a source, flash a source for a number of frames, swap two
  sources, hold a frame, reverse, or jump a knob of a look.

**Interlacing in time.** Sources alternate frame by frame in any ratio (A B A
B, or three frames of A to one of B), over any number of sources. The ratio
itself can be keyed on a lane, so one source takes over gradually.

**Interlacing in space.** Sources alternate by line, column, checker cell or
block inside one frame, and the spatial pattern can shift every frame. Line
interlace of two clips with a one-line shift per frame is one preset.

**Source time.** Each source in a pattern has a time policy. Running keeps the
clip advancing underneath while it is hidden. Stepping advances it one frame
each time it shows. The others are freeze, stutter (a short loop), scrub from a
lane and reverse.

**Flashes and hidden images.**

- **Subliminal flash.** A still or a clip shows for one to three frames on a
  voice's pulse.
- **Complementary pair.** An image is added to one frame and subtracted from
  the next. At normal speed the pair averages back to the footage, and the
  image appears when the video is paused or slowed.
- **Rhythmic reveal.** An image is cut into parts, and each part flashes on
  its own voice, so the whole image exists only across the polyrhythm.
- **Flicker.** Brightness or colour strobes at a stated frequency, up to half
  the frame rate.

**Sound.** A voice can follow the onsets of the audio lane, and it can write a
click or a tone to the audio lane, so picture and sound share one rhythm.

**Flash meter.** The pattern editor counts flashes per second as defined in
WCAG 2.3.1 and ITU-R BT.1702 (opposing luminance changes of 10% or more, with
saturated red counted separately). Above three per second it shows a warning
on the timeline and writes a note in the export sidecar. The person decides,
and the meter states the fact.

**Ops.** `film pattern add`, `film pattern voice`, `film pattern preview`. A
pattern is JSON beside the film's lanes.

```json
{"pattern": {"fps": 30, "tempo": 120, "seed": 7,
  "sources": {"A": "clips:8120", "B": "clips:812", "I": "japan:1423"},
  "voices": [
    {"pulse": {"every": 2}, "action": {"cut": ["A", "B"]}, "time": "running"},
    {"pulse": {"euclid": [5, 8], "unit": "beat/4"}, "action": {"flash": "I", "frames": 1}},
    {"pulse": {"every": 3.02}, "action": {"knob": "finish.exposure", "to": 1.5, "frames": 1}}
  ],
  "space": {"kind": "lines", "size": 2, "shift": 1}}}
```

## 3.6 Morphs between videos and generative video

- **Synchronous token morph.** Two clips play together and each output frame
  decodes a mix of their token grids, weighted by a lane. This is the existing
  film morph with grids that change over time, in the DINOv2 and FLUX.2 VAE
  spaces already present.
- **Transition morph.** At a cut, the two clips overlap for a chosen length and
  the mix weight crosses from one to the other.
- **Generated in-between.** A first-and-last-frame video model generates new
  frames between the end of one clip and the start of the next. Wan 2.1 has
  FLF2V and VACE pipelines in diffusers, and VACE also covers video-to-video
  conditioning, inpainting and reference-guided generation.
- **Video-to-video diffusion.** Whole-clip restyling through the same model
  family, registered with `needs={"gpu", "diffusers"}`. This is level 4 of the
  temporal integration in 3.4. LTX-2.3 is the faster alternative with synced
  audio, under a community licence.
- **Motion transfer.** The descriptors of one clip drive the lanes of another
  (zoom, displacement, speed), which gives a readable form of motion transfer
  ahead of any generative model.

## 3.7 Agent parity and live view

**What exists.** The agent runs ops through one MCP tool, forwarded to the
running app. The page polls `/api/state` every two seconds and reloads boards,
collection and settings when a signature moves. FilmTab re-lists films and
ArchiveTab reloads while visible.

**What stays stale today.** Search, story and map results, Moodboard
similarity results, the Alchemy stage and its controls, the recipe list, and
the open film, which lives in localStorage (`film.svelte.ts:34`). The film spec
edits (`/film/add`, `space`, `lane`, `look*`, `meta`) are functions in
`ui_film.py` outside the ops registry, so the agent edits a film only by
saving a whole spec.

**The plan.**

- **Every control is an op.** Film edits become ops (`film clip add`,
  `film clip trim`, `film lane key`, `film look add`, `film audio add`), and
  the page calls them through `/api/rpc`. The Alchemy stage settings and the
  search query become addressable state with `alchemy stage set` and
  `search show`. The parity test of phase 1 gains a check that every control in
  the page maps to an op.
- **View state on the server.** Each tab's working state (the query and its
  results, the stage settings, the open film) is a document in `state.STORE`.
  Tabs render from it, undo covers it, and the signature includes it.
- **Push.** A server-sent events stream at `/api/events` carries the signature
  and the list of changed documents. The two-second poll remains as the
  fallback. Effect ops stream preview frames on the same channel, so a render
  or an Alchemy step appears on the stage as it progresses, in whichever tab
  holds that stage.
- **Follow the agent.** An activity strip lists each agent op with a Show
  button that opens its result in the right tab. A Follow toggle makes the page
  switch to where the agent is working. With Follow off, the person's own view
  stays put and the strip carries the updates. View ops declare `needs={"ui"}`,
  a flag the registry already defines.
- **Blind mode.** The agent receives clip handles, time ranges, motion lines,
  palettes and strengths as text. With pixels allowed, contact sheets show
  keyframe strips.

## 3.8 Export

- mp4 with the audio mix, at preview or full quality.
- An edit decision list in OpenTimelineIO, pointing at the original media with
  the in and out points, so a cut made in Arcana opens in DaVinci Resolve or
  Premiere. Colour work travels as `.cube` LUTs per clip.
- The Archive tree lists videos with sound, sounds, and timeline files, and
  `/film-video` serves every container the scan accepts.

## 3.9 Done when

- Phase 2. An agent's `search text`, `alchemy preview` and film edit each appear
  in the open page within a second, and undo reverses them.
- Phase 7. A film mixes stills and clips with a soundtrack, clips drop in from
  search, a story becomes a rough cut in one action, and the export opens in
  Resolve.
- Phase 8. A pattern of three voices at 3, 4 and 5 frames renders the same
  twice from its seed. Two clips interlace by frame and by line. A still
  flashes for one frame on a Euclidean rhythm, and a complementary pair is
  invisible at speed and visible on pause. The flash meter warns above three
  flashes per second. A static colour transfer on a shot is free of flicker, a
  dynamic transfer follows a reference video's palette over time, and a
  tracked mask holds a subject through a clip.
- Phase 9. Two clips morph into each other over a transition, a generated
  in-between joins two shots, and a diffusion look on a ten-second clip passes
  a flicker measurement at each level of integration it offers.

---

# Status

**Phase 1, built.**

- `arcana/modality.py` holds the vocabulary (ids, labels, extensions, dataset
  ids, the discovery patterns, feature file names). The three copies of the id
  suffix and the two discovery regexes call it.
- Every op that takes a dataset or pictures carries `modalities=`, 38 in all.
  `op list`, `--help`, `/ops/list` and the schema print the tag, and `call()`
  refuses other media with one sentence.
- `arcana/capabilities.py` lists the page's features. `/api/bootstrap` serves
  the table, and `/api/similar`, `/api/story/save` and `/api/story/poetry`
  refuse by it with HTTP 409.
- The page shows tags on tabs, the search bar, Inject Poetry, the similarity
  card, the wizard extras and the Alchemy subtabs. The dataset menu groups by
  modality. Thumbnails of sounds draw their waveform.
- Fixed with tests in `tests/test_modality.py`. A photo set's features stay
  out of an audio set of the same name. Save story and Inject Poetry refuse
  sounds. Alchemy, colour and style ops refuse sounds before any file opens. A
  search saved from an audio dataset counts its sounds in the Archive tree.
- `tests/sandbox_app.py` serves the page on a scratch data folder with a
  synthetic photo set and audio set, on port 8061.

**Phases 4 and 5, first pieces built.**

- `arcana/sections.py` is the section detector of 2.3. It builds the
  neighbour-restricted tree over ticks, gives each boundary a strength, and
  returns nested coarse, medium and fine sections plus the whole file. It also
  picks the frames a section hands an encoder and keeps the best of nested
  sections for Auto granularity. It runs on synthetic tick streams in
  `tests/test_sections.py`.
- `arcana/video.py` reads video through the ffmpeg that imageio-ffmpeg ships.
  It probes a file, yields frames at any rate and size, extracts the sound
  track, and detects hard cuts from a colour signature. In
  `tests/test_video.py` a generated file of three parts comes out as three
  coarse sections at the right times.
- The level thresholds (35%, 12% and 4% of a video's strongest boundary, with
  an absolute floor) are first values. They await real CLIP ticks and the
  sectioning check of 2.9.

**A first video dataset, end to end, on the installed CLIP encoders.**

- `arcana/video_index.py` indexes a folder of videos by section. It writes the
  index, `sections_{name}_video.pkl` and `tree_{name}_video.pkl`. A section's
  vector is the mean of its CLIP ticks.
- A section is addressed as its file plus a time range,
  `walk.mp4#t=12.000,22.500`. Every existing path-based surface (search, map,
  cards, hits) carries it unchanged, and a `<video>` element plays that range.
- `dataset create --modality video` builds one. The wizard has a Video switch.
  Text search takes a level (Auto, Whole files, Coarse, Medium, Fine) and a
  minimum length. `/thumb`, `/preview` and a new `/video` route serve sections.
- Run on real footage with `evals/video_sections.py`. 35 varied videos (camera
  files, phone clips, screen recordings, generated clips, glitch exports) all
  read and sectioned. The noise floor moved from 0.004 to 0.012 after single
  takes were being cut in two.
- Measured with CLIP ViT-B/32 on this machine. Reading and encoding ran at
  about 6 times playback speed on 1080p 60 fps camera files, where decoding is
  the main cost (63 ms a tick against 19 ms to encode). Hardware decoding cut
  one such file from 10.1 s to 6.1 s. Keyframe-only decoding was ruled out,
  since keyframe spacing ranged from 0.5 s to 4 s across the sample.

**Dynamics, layer 1.**

- `arcana/motion.py` measures movement from optical flow. It separates camera
  travel (pan, tilt, zoom, roll) from subject movement with a similarity fit,
  and adds shake, flicker and regularity. `describe()` writes the line shown
  under a result, such as "handheld, subject moving".
- Indexing reads frames at 8 per second in the pass it already makes, takes
  every fourth as a tick, and measures motion on all of them. The flow added
  no measurable time on the test set.
- Each section stores its motion vector in `features_{name}_video_motion.npz`
  and its line in the sections table. Video cards print the line.
- On the 30 test videos the lines matched the footage by kind. Phone clips
  read as handheld with a moving subject, screen recordings as still, and
  generated clips as slow pans and zooms. "Rhythmic" fired on tripod shots of
  water and trees and was tightened.
- Sectioning still reads content alone. Motion joins it once the thresholds
  are checked against footage where only the movement changes.

**Dynamics, layer 2.**

- A query about movement is read for the words it names (still, pan left, pan
  right, tilt up, tilt down, zoom in, zoom out, handheld, fast, slow, calm,
  subject moving, flicker, rhythmic) and scored against the motion numbers.
- Prompt Search on a video dataset has a Match control (Content, Both,
  Dynamics) and a movement box. The page prints the words it understood.
- In the page, "trees" with the movement "handheld" returned four handheld
  sections.

**Phase 3, the shared space.**

- EmbeddingGemma 2 was published against transformers 5.18. The app stays on
  transformers 4, and the encoder runs in a worker process with its own
  environment, built by `uv run --script` from the header of
  `arcana/omni_worker.py`. This replaces the plan of lifting the pin, which
  would have put Alchemy and the diffusion methods on an untested version.
- `arcana/omni.py` starts the worker on first use and hands it decoded arrays.
  `arcana/spaces.py` keeps a dataset's shared-space vectors in
  `space_{name}_{modality}.{space}.npz` and ranks mixed queries with each term
  standardised.
- `dataset embed <dataset>` embeds any dataset and resumes where it stopped.
  `search across <dataset> --text ... --like <handle> --not-text ...` searches
  it with words and with items of any kind.
- On the 30 test videos (92 sections), text to video was right on three
  queries out of three. A man talking outdoors returned the four phone clips,
  tree branches returned the tree shots, and a diagram returned the animated
  explainer.
- Sound to video was weak. Ten seconds of a voice ranked screen recordings
  first and one talking clip fourth.
- Cost on this machine. The worker starts in about 45 s. A section of 32
  frames takes about 4 s, and 92 sections took 402 s. The frame count and
  frame size are the first things to lower.
- Open. The page has no control for the shared space yet, and the photo check
  of 2.9 has not run.

**Transformers 5, tested.**

- A scratch environment with transformers 5.20.2 (and huggingface-hub 1.33,
  tokenizers 0.23) ran the suite. It passed apart from the test that checks
  the environment against the lock, which fails by design there.
- Six of nine real encoder calls broke for one cause. The feature calls return
  an output object in version 5, and the Clap processor renamed one argument.
  `arcana/compat.py` handles both, and every encoder call goes through it.
- With the helper, CLIP and CLAP vectors are identical between 4.57.6 and
  5.20.2 (cosine 1.000000 on an image, a phrase, a sound). Real img2img with
  sd-turbo, texture style transfer, `tokens.mix` and CLIP description all ran
  under version 5. Text-to-image from words failed once on a dropped
  connection to the model hub.
- The remaining paths, run under 5.20.2 on real pictures. Text-to-image,
  img2img, IP-Adapter, ControlNet with depth, canny and luminance maps, the
  depth control map, the texture LoRA (with a LoRA, with a material, and on
  the plain model), a mask through Grounding DINO and SAM 2, verification of
  the four encoders, and a 1.2 GB model download into an empty cache with
  progress reported. All passed.
- The packaged build. A CPU build made from the same spec with the version 5
  package set is 1,054 MB. It started headless in 24 s, listed the datasets
  and the pattern ops, and a CLIP text search through it returned the right
  clip first.
- One difference in behaviour. With huggingface-hub 1.33 a dropped connection
  to the model hub ended the op four times in this session (text-to-image
  once, IP-Adapter once, the download twice), each passing on a retry. Under
  0.36 the same drops were retried silently. Three of the four were checks on
  models already on disk.
- That is handled in `arcana/hub.py`. Every model load goes through
  `hub.pretrained`, which loads from disk first and asks the hub only for a
  model that is missing, with retries on a dropped connection. Downloads retry
  the same way. A test fails if a loader is added without it.
- The pin is lifted. `transformers` is held at 5.20.2, with huggingface-hub
  1.33 and tokenizers 0.23.3, in `pyproject.toml`, `uv.lock` and
  `requirements-lock.txt`. The suite passes on it, and every encoder call runs
  with the network switched off.
- Masks by words were checked on frames with a known subject, on both
  versions. People, face, hat, glasses, beard, a cloaked figure, the floor and
  the sky were each cut correctly, with the same coverage to within a point.
  An earlier reading of 0% came from a test that opened a different photo of
  the same file name.

**A reader bug found by rendering examples.** `video.frames` kept ffmpeg's
messages in a pipe that nothing read. A decoder that printed on many frames
filled it and the read never returned, which held one render for 26 minutes.
The messages go to a temporary file now, in the reader and in the pattern
writer.

**Embedding speed.**

- Decoding was the bottleneck, about 1 s a section at any setting, since each
  section re-read its range of the file. `dataset embed` now decodes each
  video once at 4 frames a second and slices sections from memory.
- Frames and size, on twelve 8-second sections and six queries. 32 frames at
  448 px encoded in 1.24 s a section. 16 at 336 px took 0.55 s and kept the
  top result on five queries of six, and is the default.
- The 92 test sections went from 402 s to 140 s. That is 12 s per minute of
  footage for the shared space, beside 10 s per minute for the CLIP index and
  motion, on a set weighted to 1080p 60 fps camera files.

| Footage | CLIP index and motion | Shared space | Together |
|---|---|---|---|
| 10 clips of 30 s | 50 s | 1 min | 2 min |
| 100 clips of 1 min | 16 min | 20 min | 36 min |
| 100 videos of 5 min | 1 h 20 | 1 h 40 | 3 h |
| 1,000 clips of 1 min | 2 h 40 | 3 h 20 | 6 h |
| 50 hours | 8 h | 10 h | 18 h |

These scale from 11.5 minutes of measured footage. Light files (720p, 24 or
30 fps) decode several times faster. Photos took 0.9 s each at 896 px and
sounds 0.5 s each.

**Sound and photo, on real files.** 240 photos from the Pictures archive
against 200 labelled sounds (ESC-50, four per class).

- Sound to photo worked where the archive held the subject. Bird song returned
  six bird photos, laughter returned smiling people, sea waves and rain
  returned water, church bells returned a piano and a cross.
- Sounds with no matching photo (fire, footsteps, train, crickets, typing)
  returned the same few water and texture photos.
- Photo to sound, with the best photo of each row sent back. The bird photo
  returned bird song, the smiling man returned laughter, the water photo
  returned sea waves and rain, and the engine photo returned engines.
- Text to sound. The right class came first for 13 classes of 50, against a
  chance level of 1.

**Pattern and rhythm, built.**

- `arcana/pattern.py` renders a pattern from a JSON rule. It has pulses
  (every, duty cycle, Euclidean, hand-written, chance, fractional periods that
  drift), three actions (cut, flash, complementary pair), interlacing in space
  (lines, columns, checker, with a per-frame shift), three source time
  policies and the flash meter.
- `pattern make` and `pattern presets` are ops. The Alchemy tab has a Pattern
  subtab whose sources are the first items of the collection.
- In the page, a clip with Euclidean white flashes over drifting line
  interlace rendered 180 frames, and the meter warned at 7 flashes a second.
- Curves were added. A wave with its own period, in frames, moves hue,
  saturation, brightness, inversion, a red and blue split, or a dissolve into
  another source, so it runs against the voices' rhythms.
- A morph is a source like any other. The film engine renders a DINOv2 morph
  through stills to a video (300 frames at 768 px in 48 s), and a pattern
  interlaces it with clips.
- A complementary pair now hides the picture's detail around its own average.
  Adding the average too made the whole frame flicker 15 times a second.
- Two sets of examples are in `arcana/output/pattern-examples` and
  `pattern-examples-2`, each with the rule beside it.
- Open from 3.5. Knob jumps, the rhythmic reveal, sound voices, and the
  pattern as a lane of the film timeline.

**Shared space in the page.** Prompt Search has an Embed button for a dataset
outside the shared space, then a Shared space switch and an option to search
by the collection's items. A text search through it returned the four talking
clips in the sandbox.

**Phase 2, live view.**

- `arcana/activity.py` records each op that ran, with its actor, its
  arguments and the first line of its result. Lookups about Arcana itself and
  refused ops leave no line.
- `/api/events` is a server-sent stream that sends a message the moment the
  activity list or the state signature moves. `/api/activity` returns the new
  entries. The two-second poll stays as the fallback.
- The page has a strip under the header with the last thing an agent or the
  terminal did, a Show button and a Follow switch. Show opens a search in
  Prompt Search with its results, a board change in the Moodboard, and
  something rendered in the Archive.
- In the page, a search sent as an agent appeared in the strip within three
  seconds, and Show ran it.
- Open. The film timeline, the Alchemy stage and the recipe list still keep
  their state in the browser, and film edits are still outside the ops.

**Phase 4, proxies.** A file a browser cannot decode (10-bit, 4:2:2, HEVC,
ProRes, mkv, avi) plays through an H.264 proxy made on first request and kept
in `cache/video`. Files a browser plays are served as they are.

**Phase 5, more.**

- `dataset extend` adds new videos to a video dataset. Existing keys stay, and
  the map is laid out again.
- `dataset embed --sound` embeds each section's sound track (up to 30 s) into
  its own space file, and `search across --listen` searches it. It is tested
  with a stub encoder and has yet to run on real footage.

**Phase 6, more.** Prompt Search returns sections or whole videos. A whole
video's card carries a strip of its finest sections, shaded by score, with the
best one outlined, and a click plays from there. Chips filter by motion
(still, pan, tilt, zoom, handheld, subject, flicker).

**Phase 7, sequences.**

- `arcana/sequence.py` cuts sections, whole videos and stills together in
  order, scaled to fill one frame size, with their sound and silence where a
  clip has none. Clips take in and out points and a speed.
- A CMX 3600 edit list is written beside the render. It names the original
  files and their timecodes, for Resolve or Premiere.
- `sequence make` is the op. Prompt Search has "Cut selected together" and
  Generate Story has "Make a rough cut", with the story lines as notes in the
  edit list.
- In the page, two whole videos (one of them an mkv) cut into a 52 s video.
- Open. This is a cut list, separate from the film's token morph timeline.
  Dragging onto a timeline, an audio lane of chosen sounds, transitions and
  the match-cut suggestion are still to build.

**Phase 8, colour on video.**

- `arcana/video_colour.py` fits a transfer once on frames pooled across the
  clip and applies it to every frame as one 33-point table. Every existing
  method (lab, mkl, idt, modflows) goes through it.
- With several references, or `--keys` on a reference clip, a table is fitted
  at each and the clip passes from one to the next.
- `colour video` is the op. It keeps the sound, writes each table as a
  `.cube`, and reports the frame-to-frame colour change before and after.
- On a real 12.5 s clip (749 frames, mkl toward a talking-head clip) that
  change went from 0.24 to 0.52 levels of 255.
- `colour deflicker` scales each frame to the slow course of the clip's
  colour, and with `--source` puts back the changes the original clip had. On
  a test card with every other frame darkened it cut the frame-to-frame
  change to under a quarter.
- The page. Alchemy has a Colour subtab. A clip of the collection is the
  target, pictures and other clips are the references in order, and the
  controls are the method, the strength and the number of moments to follow a
  reference clip at. The result plays beside the original with the measured
  change between frames, and Steady runs `colour deflicker` on it against the
  original.
- In the page, on a 12.5 s clip recoloured toward a still with Lab, the
  change between frames went from 0.24 to 0.26. Steady on that result, which
  had no added flicker, raised it from 0.26 to 0.34.
- Open. Steady on a clip without flicker should leave it as it is. Tracked
  masks and propagated style are still to build.

**Checked against the revised Part 3.** The built code stands as it is. The
pattern engine addresses frames by number, which the reader and the section
references already allow. One line of the plan changed, the proxy, which now
makes every frame a keyframe so any frame is reached exactly.

**Still to build in phases 4 to 6.** Search and facets by motion, the motion
words of layer 2, The proxy cache (files a browser cannot
decode play nowhere yet), parallel decoding across files, Add new for video
datasets, the portable bundle for sections, collection and board support for
clips, whole-video result cards with a heat strip, the omni encoder (phase 3),
the dynamics channel and the sound channel.

**Open from phase 1.** The film spec accepts any reference inside its JSON,
since the check reads op arguments. A tab dims only when every feature it holds
excludes the chosen dataset, which happens for video datasets and for none of
today's two modalities.

## Adjustments after merging main (10 November 2026)

Main gained four commits while this branch was built. They change the plan in
six places.

**3.1, the clip lane.** Main put kept films on the picture lane as keys with
`in`, `out` and `speed`, and `film.py` states that clips from a video dataset
take the same shape. The separate `clips` list of 3.1 is dropped. A clip is a
picture-lane key whose `ref` is a section reference, and it reuses main's
`span`, the two-point timing (`at`, `plays`) and the morph into the frame at
`in` and out of the frame at `out`. The work left is a frame source for such a
key (`video.frames` through the proxy for preview, the original for the final
render), the audio lane and the mux.

**3.1, pace.** Main's steady spacing measures how much the picture changes
along a passage in the morph's own space. A clip key keeps its length under
that spacing, as a held film does today.

**3.4, looks over time.** Main's lanes take a whole shape between two times
(rise, fall, swell, dip, wave, steps), the amount lane sets how much of the
look covers the morph, and masks glide or cross-fade between looks. Looks over
time on a clip therefore need only the clip frame source. The lane shapes and
the pattern engine's curves describe the same thing, so the pattern lane of
3.5 reads the film's lane shapes and the pattern engine keeps its pulses,
interlaces and flashes.

**3.6, morphs.** `alchemy derive` keeps several versions of one picture and
lays them on the picture lane. The same op applied to a clip's key frames
gives the versions a video morph passes through, which replaces the separate
key-frame step planned in 3.6.

**2.5, spaces.** Sets and IP labels live in CLIP ViT-H/14 (1024 numbers).
Video sections are indexed with the dataset's CLIP, so the search route accepts an IP
label on a video dataset built with ViT-H. That path is untested, and the
capability table lists labels and axes as photo features until it is.

**1.4, the contract.** Main's ops and model loads were brought under this
branch's rules during the merge. `alchemy derive` and `set save` declare
photos. The loaders in `alchemy/editor.py`, `ip_labels.py` and `lora_train.py`
go through `hub.pretrained`, and `captions.py` reads CLIP features through
`compat.features`. The Generate tab and the features Save to a set, IP labels
and axes are in the capability table as photo features. Prompt Search on a
sound dataset keeps + Collection and loses Save selected, which main replaced
with sets.

**Order of work from here.** Clip keys on the picture lane with a frame
source, then the audio lane and mux, then the page controls for colour on
video, then tracked masks and propagated style, then phase 9.

## Clips on the film's picture lane (built)

- A key on the picture lane holds a clip by a video handle, a file or a
  section of a file, such as `{"t": 4, "ref": "clips@video:12", "in": 12.0,
  "out": 22.5, "speed": 1}`. `in` and `out` are seconds of the source file
  and default to the section's range.
- `film.Clip` reads frames from one running decoder when they are asked for
  in order, and starts the decoder again on a jump. A render decodes each
  clip once.
- The film morphs into the clip's frame at `in`, plays the clip's own frames
  and morphs on from its frame at `out`. Looks, lanes and finish apply to the
  clip's frames.
- Sound. A clip is heard while it plays, at its speed, with `mute` and `gain`
  on its key. The `audio` list places sounds or the track of a video with
  `t`, `in`, `out`, `gain` and `fade`. The mix is laid under the rendered
  picture, and `"sound": false` renders a silent film.
- The page. The timeline draws a clip as a segment with its frame at `in`,
  its name and its length. The key panel sets In, Out, Speed, Sound and Gain.
  A clip in the collection is dragged or clicked onto the timeline as a
  picture is.
- `alchemy film save`, `preview` and `make` are tagged photo and video, and
  the op's help describes clip keys and the audio list.
- Measured on a 1080p 60 fps camera clip between two stills (DINOv2, 512 x
  288, 12 fps, 9 s). The render took 70 s. Frames during the clip equal the
  source frames, and the sound sits on the four seconds the clip plays.
- Open. A mask read by a model (subject, words, depth) is read on the clip's
  frames at `in` and `out` and blended between them, so it trails a subject
  that moves. The audio lane has no page control. Sound from a clip inside a
  held film is left out. The render reads originals at the film's size, and
  scrubbing reads the same files.

# Decisions to take

| Decision | Recommendation | Settled by |
|---|---|---|
| Primary content space for video | EmbeddingGemma 2 for every video dataset, CLIP keyframes as an option for compatibility with photo archives | the photo check and query set of 2.9 |
| Section levels | three levels plus the whole file, with a 2-second minimum length | the sectioning preview and the sectioning check |
| Learned dynamics vector | motion lines from a local VLM, since the text doubles as evidence under each result | the dynamics probe |
| Cut detector | PySceneDetect by default, TransNetV2 when a GPU is present | a count of missed dissolves on real footage |
| Follow the agent | off by default, with the activity strip always on | use |
| Generative video model | Wan 2.1 VACE through diffusers | VRAM and quality on this machine, licence terms on the model cards |
| Where pattern editing lives | a pattern lane and panel in the Alchemy Video subtab, beside the film's other lanes | use |
| Default temporal integration for diffusion looks | keyframe and propagate | the flicker measurement of phase 9 |

# Limitations

- The benchmark figures in 2.1 come from each model's authors under different
  protocols and metrics. They rank the models loosely.
- The appearance-bias evidence comes from one preprint (ViTAL-X) and one
  benchmark paper (RTime). The dynamics probe measures it directly on the
  person's footage.
- EmbeddingGemma 2 throughput, VRAM use on this machine and its minimum
  transformers version are unmeasured here. The documentation leaves open
  whether a video's audio track enters its video embedding, and this plan
  embeds the track separately for that reason.
- The motion-word vocabulary and the standardised score mix are designs that
  await the query set. Section detection by constrained clustering awaits the
  sectioning check.
- Licence terms for PE-AV, Qwen3-VL-Embedding and Wan were read from secondary
  sources. Wan 2.2 first-and-last-frame support was found as a ComfyUI workflow
  only. The PySceneDetect 0.7 release date comes from a mirror of its
  repository.
- Line references are valid at `3ae5e90`. Most were collected by a code survey
  and a sample was re-read by hand.
- Synchronous token morphs decode every frame independently, and flicker in the
  RAE decoder is untested.
- Warped-noise integration (level 2 in 3.4) is described from the published
  idea and was absent from the survey results. The flash meter follows the
  WCAG definition and is a guide, since display brightness and viewing distance
  are unknown to it.

# Sources

- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
- [Multimodal embeddings with EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [EmbeddingGemma 2 developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
- [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [jina-embeddings-v5-omni](https://arxiv.org/html/2605.08384v2)
- [Ovis-Embedding](https://arxiv.org/pdf/2609.25165)
- [PE-AV, audiovisual perception with multimodal correspondence learning](https://arxiv.org/abs/2512.19687v1)
- [InternVideo-Next](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_InternVideo-Next_Towards_World-Understanding_Video_Models_CVPR_2026_paper.pdf)
- [V-JEPA 2](https://arxiv.org/pdf/2506.09985)
- [VideoPrism](https://arxiv.org/pdf/2402.13217)
- [VidVec](https://arxiv.org/html/2602.08099v1)
- [MVEB, Massive Video Embedding Benchmark](https://arxiv.org/abs/2606.14958)
- [ViTAL-X, video and text alignment with temporal edits](https://arxiv.org/pdf/2609.00505)
- [RTime, Reversed in Time](https://arxiv.org/pdf/2412.19178)
- [Appearance-Motion Decomposed Alignment for Text-Video Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/37745)
- [CameraBench, camera motions in any video](https://arxiv.org/abs/2504.15376)
- [AutoShot](https://arxiv.org/pdf/2304.06116)
- [CineTrans, with its shot detector comparison](https://arxiv.org/pdf/2508.11484)
- [PySceneDetect migration guide for 0.7](https://www.scenedetect.com/docs/head/api/migration_guide.html)
- [Wan pipelines in diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan)
- [LTX-2](https://en.wikipedia.org/wiki/LTX-2)
- [NLUT, neural LUTs for video style transfer](https://arxiv.org/abs/2303.09170v2)
- [LumiVideo](https://arxiv.org/html/2604.02409v1)
- [Blind video temporal consistency, Bonneel et al.](https://perso.liris.cnrs.fr/nbonneel/consistency/temporalconsistency.pdf)
- [RFDM, previous-frame conditioning for video editing](https://arxiv.org/html/2602.06871v2)
- [Adapting VACE for real-time autoregressive video diffusion](https://arxiv.org/abs/2602.14381)
- [NOVA, sparse keyframe editing with dense motion control](https://arxiv.org/html/2603.02802)
- [WCAG 2.3.1, three flashes or below threshold](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)
