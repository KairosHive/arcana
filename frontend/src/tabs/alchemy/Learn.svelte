<script lang="ts">
  import { onMount } from 'svelte'
  import { app, setMode } from '../../lib/stores/app.svelte'
  import { openViewer } from '../../lib/stores/viewer.svelte'
  import { thumbUrl } from '../../lib/api/media'
  import { learn, loadAll, loadSets, makeLabel, deleteLabel, trainLora, setShow, makeAxis, deleteAxis, type LabelRow } from '../../lib/stores/learn.svelte'
  import { step, method, chooseMethod, reloadCatalog } from './alchemy.svelte'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Toggle from '../../lib/ui/Toggle.svelte'
  import JobCard from '../../lib/components/JobCard.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'

  /**
   * What Alchemy learns from a set. A set is pictures gathered in Search. An IP
   * label is a named position in IP-Adapter space, measured on a set, on words
   * or on one picture, that the diffusion methods draw from. A LoRA is trained
   * on a set and drawn by "picture in a learned look".
   */
  let { onchange }: { onchange: () => void } = $props()

  onMount(() => { void loadAll() })
  // Every time the sets change (Search saved into one), read them again.
  $effect(() => { void app.signature; void loadSets() })
  // A new label or checkpoint: the knobs' options are read again.
  let seenRev = 0
  $effect(() => { const r = learn.rev; if (r !== seenRev) { seenRev = r; void reloadCatalog() } })

  // ---- sets ----
  let thumbs = $state<Record<string, string[]>>({})
  $effect(() => {
    for (const s of learn.sets) {
      const key = `${s.slug}:${s.count}`
      if (thumbs[key]) continue
      setShow(s.slug).then((r) => { thumbs = { ...thumbs, [key]: r.items } }).catch(() => {})
    }
  })
  let grain = $state(false)
  async function labelOf(slug: string, name: string) {
    await makeLabel({ name: grain ? `${name} grain` : name, set: slug, crops: grain ? 4 : 0 })
  }

  // ---- labels ----
  let words = $state('')
  let wordsN = $state(50)
  const datasetName = () => (app.dataset || '').split('::')[0]
  async function labelFromWords() {
    const w = words.trim()
    if (!w || !app.dataset) return
    const lab = await makeLabel({ name: w, words: w, dataset: datasetName(), n: wordsN })
    if (lab) words = ''
  }
  function origin(l: LabelRow): string {
    const o = l.origin || {}
    if (o.kind === 'set') return `set ${o.set}${o.crops ? `, ${o.crops} crops each` : ''}`
    if (o.kind === 'words') return `"${o.text}", top ${o.n} in ${o.dataset}`
    if (o.kind === 'picture') return 'one picture'
    return ''
  }
  const hasLabelKnob = () => !!method(step.method)?.knobs.some((k) => k.key === 'label')
  function use(slug: string, slot: 'label' | 'label2') {
    if (hasLabelKnob()) chooseMethod(step.method, { ...step.knobs, [slot]: slug, ...(slot === 'label2' && step.knobs.label === 'none' ? { label: slug } : {}) })
    else chooseMethod('generative.controlnet', { label: slug })
    onchange()
  }
  function searchWith(slug: string) {
    learn.searchLabel = slug
    setMode('prompt')
  }

  // ---- concept axes ----
  let axisName = $state('')
  let axisPos = $state(''), axisNeg = $state('')
  const endOptions = $derived([...learn.labels.filter((l) => !l.slug.startsWith('picture-')).map((l) => ({ value: 'label:' + l.slug, label: l.name })),
    ...learn.sets.map((s) => ({ value: 'set:' + s.slug, label: `set: ${s.name}` }))])
  let posPick = $state<string | null>(null), negPick = $state<string | null>(null)
  async function fitAxis() {
    const pos = posPick || (axisPos.trim() ? 'words:' + axisPos.trim() : '')
    const neg = negPick || (axisNeg.trim() ? 'words:' + axisNeg.trim() : '')
    if (!axisName.trim() || !pos || !neg) { return }
    const a = await makeAxis(axisName.trim(), pos, neg, datasetName())
    if (a) { axisName = ''; axisPos = ''; axisNeg = ''; posPick = null; negPick = null }
  }
  const endText = (e: string | null) => (e || '').replace(/^words:/, '“').replace(/^(set|label):/, '') + (e?.startsWith('words:') ? '”' : '')

  // ---- LoRAs ----
  let trainSet = $state<string | null>(null)
  let trainName = $state('')
  let trainSteps = $state(2000)
  const setOptions = $derived(learn.sets.map((s) => ({ value: s.slug, label: `${s.name}  ·  ${s.count}` })))
  $effect(() => { if (trainSet && !trainName) trainName = learn.sets.find((s) => s.slug === trainSet)?.slug ?? '' })
  const training = $derived(!!learn.loraJob && !['done', 'failed', 'cancelled'].includes(learn.loraJob.status))
  function train() {
    if (!trainSet || !trainName.trim()) return
    void trainLora({ name: trainName.trim(), set: trainSet, steps: trainSteps })
  }
  function useLora(name: string) {
    chooseMethod('generative.lora', { ...(step.method === 'generative.lora' ? step.knobs : {}), lora: name })
    onchange()
  }
</script>

<Card title="Learn" subtitle="Sets come from Search. Measure an IP label on one, or train a LoRA, and draw with them in the diffusion methods." pad="sm" scroll style="flex: 1 1 auto; min-height: 0">
  <div class="learn">
    <section>
      <h6 class="caps">Sets</h6>
      {#if !learn.sets.length}
        <p class="hint">No sets yet. In Search, "Save to a set" gathers results into one.</p>
      {/if}
      {#each learn.sets as s (s.slug)}
        {@const items = thumbs[`${s.slug}:${s.count}`] ?? []}
        <div class="row-item">
          <div class="head"><b>{s.name}</b><span class="dim">{s.count} pictures</span></div>
          <div class="strip">
            {#each items.slice(0, 5) as p, i (p)}<button class="mini" onclick={() => openViewer(items, i, s.name)}><img src={thumbUrl(p)} alt="" loading="lazy" /></button>{/each}
          </div>
          <div class="acts">
            <Button size="sm" onclick={() => labelOf(s.slug, s.name)} busy={learn.labelBusy} disabled={learn.labelBusy}>IP label</Button>
          </div>
        </div>
      {/each}
      <Toggle bind:checked={grain} label="Labels from crops (keeps grain)" size="sm" />
    </section>

    <section>
      <h6 class="caps">IP labels</h6>
      {#each learn.labels as l (l.slug)}
        <div class="row-item">
          <div class="head"><b>{l.name}</b><span class="dim">{l.pictures} pictures{l.tightness !== null ? ` · alike ${l.tightness.toFixed(2)}` : ''}</span></div>
          <div class="dim tiny">{origin(l)}</div>
          <div class="strip">
            {#each l.nearest.slice(0, 5) as p, i (p)}<button class="mini" onclick={() => openViewer(l.nearest, i, l.name)}><img src={thumbUrl(p)} alt="" loading="lazy" /></button>{/each}
          </div>
          <div class="acts">
            <Button size="sm" kind="primary" onclick={() => use(l.slug, 'label')} title="Draw with this label in the current diffusion method (ControlNet when the method takes no label)">Use</Button>
            <Button size="sm" onclick={() => use(l.slug, 'label2')} title="The second label, mixed with the first by the mix knob">Mix in</Button>
            <Button size="sm" kind="ghost" onclick={() => searchWith(l.slug)} title="Search a ViT-H dataset for the pictures nearest this label">Search</Button>
            <Button size="sm" kind="ghost" onclick={() => deleteLabel(l.slug)} title="Delete this label">✕</Button>
          </div>
        </div>
      {/each}
      <div class="words">
        <TextInput bind:value={words} placeholder="A label from words, e.g. frost on glass" size="sm" onenter={labelFromWords} />
        <NumberInput bind:value={wordsN} min={5} max={500} size="sm" width="64px" />
        <Button size="sm" onclick={labelFromWords} busy={learn.labelBusy} disabled={!words.trim() || !app.dataset || learn.labelBusy}>Make</Button>
      </div>
      <p class="hint">From words, the label is the centre of the best matches in the dataset chosen in the header.</p>
    </section>

    <section>
      <h6 class="caps">Concept axes</h6>
      <p class="hint">A direction from one place to another, such as wet against dry. It sorts Search and moves a label in Alchemy and Generate.</p>
      {#each learn.axes as a (a.slug)}
        <div class="row-item">
          <div class="head"><b>{a.name}</b><span class="dim">{a.apart} spreads apart</span></div>
          <div class="dim tiny">{endText(a.neg)} → {endText(a.pos)}</div>
          <div class="acts"><Button size="sm" kind="ghost" onclick={() => { learn.searchLabel = 'axis:' + a.slug; setMode('prompt') }}>Sort in Search</Button><Button size="sm" kind="ghost" onclick={() => deleteAxis(a.slug)}>✕</Button></div>
        </div>
      {/each}
      <TextInput bind:value={axisName} placeholder="name, e.g. wet" size="sm" />
      <div class="words"><Select bind:value={posPick} options={endOptions} placeholder="toward…" size="sm" /><TextInput bind:value={axisPos} placeholder="or words" size="sm" /></div>
      <div class="words"><Select bind:value={negPick} options={endOptions} placeholder="away from…" size="sm" /><TextInput bind:value={axisNeg} placeholder="or words" size="sm" /></div>
      <Button size="sm" onclick={fitAxis} busy={learn.labelBusy} disabled={learn.labelBusy || !axisName.trim()}>Fit the axis</Button>
    </section>

    <!-- Learned looks (LoRA training) are out of view until a test shows one beating a label: LORA_ROADMAP.md. -->
    {#if false}
    <Disclosure label="Learned looks" note="a model trained on a set (LoRA)" level={2}>
      <section>
      {#if !learn.gpu}<p class="hint">Training a look needs an NVIDIA GPU. This Arcana runs on the CPU.</p>{/if}
      <div class="train">
        <Select bind:value={trainSet} options={setOptions} placeholder="Train a look on a set…" size="sm" />
        <div class="words">
          <TextInput bind:value={trainName} placeholder="name" size="sm" />
          <NumberInput bind:value={trainSteps} min={200} max={20000} step={100} size="sm" width="78px" suffix="steps" />
          <Button size="sm" kind="primary" onclick={train} disabled={!learn.gpu || !trainSet || !trainName.trim() || training}>Train</Button>
        </div>
        <p class="hint">A checkpoint every 500 steps, each with a sheet at the same seeds. Choose by the sheets: the loss does not say when a LoRA starts to blur. About a step a second on a laptop GPU, so 2000 steps take half an hour.</p>
      </div>
      {#if learn.loraJob}{@const running = learn.loraJob as NonNullable<typeof learn.loraJob>}<JobCard job={running} ondismiss={() => (learn.loraJob = null)} />{/if}
      {#each learn.loras as r (r.name)}
        <Disclosure label={r.name} note="{r.checkpoints.length} checkpoint{r.checkpoints.length === 1 ? '' : 's'}" level={2} open={r === learn.loras[0]}>
          {#each [...r.checkpoints].reverse() as c (c.name)}
            <div class="cp">
              {#if c.sheet}{@const sheet = c.sheet ?? ''}<button class="sheet" onclick={() => openViewer([sheet], 0, c.name)}><img src={thumbUrl(sheet)} alt="" loading="lazy" /></button>{/if}
              <div class="acts"><span class="dim tiny">{c.name.split('/').pop()}</span><span class="grow"></span><Button size="sm" onclick={() => useLora(c.name)}>Use</Button></div>
            </div>
          {/each}
        </Disclosure>
      {/each}
    </section>
    </Disclosure>
    {/if}
  </div>
</Card>

<style>
  .learn { display: flex; flex-direction: column; gap: 14px; }
  section { display: flex; flex-direction: column; gap: 8px; }
  h6 { margin: 0; }
  .row-item { display: flex; flex-direction: column; gap: 4px; padding-bottom: 8px; border-bottom: 1px dashed var(--arc-line); }
  .head { display: flex; justify-content: space-between; gap: 6px; font-size: var(--fs-sm); }
  .head b { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .head span { font-size: var(--fs-2xs); white-space: nowrap; }
  .tiny { font-size: var(--fs-2xs); }
  .strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 3px; }
  .mini { padding: 0; border: 0; border-radius: 3px; overflow: hidden; aspect-ratio: 1; background: var(--arc-surface-2); cursor: zoom-in; }
  .mini img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .acts { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
  .grow { flex: 1; }
  .words { display: flex; gap: 4px; align-items: center; }
  .words :global(.inp), .words > :first-child { flex: 1; min-width: 0; }
  .train { display: flex; flex-direction: column; gap: 6px; }
  .hint { margin: 0; font-size: var(--fs-2xs); color: var(--arc-ink-faint); }
  .cp { display: flex; flex-direction: column; gap: 3px; margin: 6px 0; }
  .sheet { padding: 0; border: 1px solid var(--arc-line); border-radius: 4px; overflow: hidden; cursor: zoom-in; background: none; }
  .sheet img { width: 100%; display: block; }
</style>
