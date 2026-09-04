# Roadmap

Candidate directions for Arcana. Nothing here is committed work — it is a
place to think from.

The ordering favours things that reuse representations already on disk. Arcana
computes a great deal per image and currently reads back a fraction of it, so
several of these are closer to *exposing* work already done than to building
something new.

---

## Already on the list

Three ideas that came first, kept here so the rest can be judged against them.

### Timeline and time filters
Filter and arrange by capture date from file metadata. Pairs with almost
everything below: a lasso region narrowed to a date range, "new since last
sync", colouring the map by date to see how the archive filled up.

### ON1 sidecars: stars and colour tags
Read ON1 sidecar files so existing star ratings and colour labels become
filters. The value is that this metadata already exists and represents
decisions already made — the machine should defer to it rather than re-guess.
Reading it in is the first half; writing marks back out (see **7**) is what
makes Arcana part of a real workflow instead of a side trip.

### Image-to-image style transfer
A sibling to the existing palette transfer: take the look of one image and
apply it to another. `style.py` already extracts VGG gram matrices, and
`color_transfer.py` already handles full-resolution output with quality
presets, so the output path exists.

---

## 1. Rework a dataset without re-encoding it — *small*

`index_dataset` already takes `reuse_index` (`db.py:1695`), and at `db.py:1759`
it skips `build()` entirely and runs only palette/style, `latent_space()` and
`write_bundle()`. The CLI exposes it as `--reuse_index`. **The GUI never passes
it** — zero occurrences in `ui_datasets.py`.

Exposing it turns four jobs that currently cost a full re-encode into a job of
under a minute: adding palette/style after the fact, re-clustering at a
different k, renaming groups from a different vocabulary, and laying out in 3-D.
It also re-extracts the stale palette features flagged by
`palette_features_stale`, and rescues interrupted runs.

Three pieces of UI copy become false and need rewriting: `ui_datasets.py:291`
("cannot be added later without re-reading every file"), the "palette and style
can only be added by re-indexing" tooltip, and the stranded-dataset warning,
which becomes a **Finish this** button.

One caution: `reuse_index` skips the glob, so a dataset whose files moved would
reuse stale paths. Gate the button on `dataset_health()`.

*This is the highest value-per-hour item on the list, and it unblocks several
others.*

## 2. Query algebra: a picture, plus and minus phrases — *medium*

One query box taking a stored image vector plus weighted positive and negative
phrases: `[this photo] -people 0.5 at night`. Search-by-example is missing
entirely today, and exclusion is the commonest fix on a personal archive, where
the right subject keeps arriving with the wrong company.

`search()` already encodes one string and calls `index.search(...)` brute-force,
so replace the single vector with a weighted sum of cosines over the whole
matrix. `text_model_for_dim` already resolves the right text tower from
`index.ndim`, so a modifier is encoded by the encoder that built the index. A
reference already in the dataset is a dictionary lookup — no vision tower, no
GPU. Default negative weight ~0.3–0.5; a full 1.0 subtraction usually returns
nothing.

## 3. Lasso the map into results — *small*

Box or lasso select on the latent map fills the results column with every image
inside the shape, as ordinary cards with the usual switches and buttons.

Plotly puts the lasso in the modebar by default, so people already try it and
nothing happens — **`selectedData` appears zero times in `arcana.py`**.
`update_images` listens only to `clickData` and renders exactly one card. Every
point already carries `custom_data=["path"]`, so the selection event arrives
with every path needed.

Today the map can raise the question *"what is that dense blob I have never
looked at"* and cannot answer it.

## 4. Map controls: arrange by look, colour by anything — *medium*

**Arrange by** swaps t-SNE coordinates for the stored style embedding, so images
group by grain and contrast rather than subject — an arrangement CLIP
structurally cannot produce, because it is trained to ignore exactly that. On
larger datasets this is free: `gram` is already stored PCA-transformed, so
`gram[:, 0:2]` is a finished 2-D layout.

**Colour by** replaces categorical cluster colour with the image's own dominant
colour, a stored scalar (lightness, chroma, contrast from `moments`), or a
user-typed semantic axis (empty↔crowded, cold↔warm) as one dot product per item.

It also fixes a live defect: the map colours by `label`, and a dataset with 9
cluster ids sharing 4 label strings renders two separate blobs identically.
`cluster_id` is written and never read.

## 5. Per-image vocabulary tags and a facet rail — *medium*

The cached label matrix names cluster centroids and is then dropped. Scoring
every image against all 100 words instead gives a tag line per image and a facet
rail (Fog 412, Neon 180, Ruins 96) that filters on click.

Because the image vectors are already on disk, pasting a different word list
re-tags ten thousand photos in seconds — which makes the vocabulary choice in
step 4 of the dataset flow **revisable instead of permanent**.

The facet rail is also the natural home for ON1 stars and colour tags, as
sibling facets that intersect with the machine tags.

## 6. Whole-archive duplicate stacks — *medium*

One pass over the entire index finding every burst, bracket and near-copy,
presented as stacks with one representative large and siblings small.

The grouping code is written and trusted — it runs on every search page — but it
only ever sees the top N results. So it can tell you two of fifty results are
twins and can never tell you 900 of your 8,470 frames are duplicates. Culling
bursts is the most time-consuming job on a large archive.

Where a stack already contains a starred frame, that is the keeper; defer to it.

## 7. Named boards and decisions that survive a move — *medium*

Working on two projects at once is currently impossible: there is one
collection, unnamed, in localStorage keyed by origin, and Clear empties it with
no undo.

Give boards names, allow several, add keep/maybe/reject marks, and key
everything by content fingerprint so decisions survive a folder move or a drive
swap. Export writes a list, not ten thousand copied JPEGs.

With ON1 sidecar writing, a cull done in Arcana shows up in the tools you
already edit in — which is what makes it part of a workflow rather than a
detour.

## 8. A moodboard you can iterate in — *medium*

Finding a picture is a walk: you land near it and then step. Every step
currently costs a full rescan plus four clicks to move the reference.

Cache the ranking the search already computes and discards, put a
**use as reference** button on every card, and split scoring into a vectorised
prefilter and an expensive re-score so the walk stays fast as the archive grows.

A style transfer is only as good as the source you found, and finding it is
currently the slow half.

## 9. Add newly-shot files without re-indexing — *large*

A personal archive grows every week and Arcana has no concept of adding to a
dataset. Today the only route is re-encoding everything — which also re-runs
t-SNE and KMeans, so every cluster moves and gets renamed and the mental map the
latent view exists to build is destroyed.

Adding files should cost the price of the new files. Needs stable cluster
assignment across runs, which is the hard part and why this is *large*.

## 10. Evidence under every result — *medium*

A strip under each card saying why it is there.

`search()` returns `(key, path, distance)` and `update_images` destructures it as
`for (k, p, d)` and **never touches `d`** — so a nonsense query returns twenty
confident-looking photographs indistinguishable from a perfect match.

The moodboard is worse than uninformative: its score is a weighted sum of
`1.0 - dist/max_dist` where `max_dist` is the largest distance *in this query*,
so 0.87 in one search is not 0.87 in another.

---

## Worth keeping in view

- **Colour-presence filter** — pick a colour and tolerance, get every image
  containing at least N% of it, ranked by coverage. The one job the 4096-bin LAB
  histogram can do that nothing else can. Needs **1** first, since existing
  palette files predate the LAB fix.
- **Checkpointed, resumable indexing** — `build()` holds the whole index in RAM
  and pickles once at the end, so a cancel or a sleeping laptop during the
  3%–65% encode window discards hours.
- **Dataset passport** — feature `.npz` files are keyed by bare usearch ids with
  no link to the index that produced them, so re-indexing a name after one file
  changed silently re-points every palette row at a different photograph. Record
  encoder, precision, item count and an id-list hash; refuse to rank on a
  mismatch. *(The `fmt` stamp added for palette features is a first step.)*
- **Stories with sound** — run each scene line through CLAP against an audio
  dataset alongside the image search, so a story is scored as well as
  illustrated. `search()` already branches on modality.
- **Sequence a selection** — nearest-neighbour plus 2-opt over the CLIP distance
  matrix so consecutive frames flow. Story mode already draws the path overlay
  but picks each node independently, so consecutive frames have no relation.
