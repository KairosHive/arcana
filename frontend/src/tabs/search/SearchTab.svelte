<script lang="ts">
  import { app, current, type Job } from '../../lib/stores/app.svelte'
  import { get, post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { loadMap, filteredRows, filterActive, EMPTY_FILTER, type MapData, type Filter } from './map'
  import LatentMap from './LatentMap.svelte'
  import MarksBar from './MarksBar.svelte'
  import type { MarksInfo } from './types'
  import MarksPanel from './MarksPanel.svelte'
  import MediaCard from '../../lib/components/MediaCard.svelte'
  import JobCard from '../../lib/components/JobCard.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import TextArea from '../../lib/ui/TextArea.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Toggle from '../../lib/ui/Toggle.svelte'
  import Slider from '../../lib/ui/Slider.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Empty from '../../lib/ui/Empty.svelte'
  import SplitPane from '../../lib/ui/SplitPane.svelte'
  import { addToCollection } from '../../lib/stores/app.svelte'
  import { previewUrl } from '../../lib/api/media'

  /**
   * Prompt Search and Generate Story share this tab: the map on the left with
   * the prompt and the marks filters under it, the results on the right. The
   * mode decides what the prompt box is and what a run returns.
   */
  let { mode }: { mode: 'prompt' | 'story' } = $props()
  const visible = $derived(app.mode === mode)

  type Group = { gid: string; keys: number[]; paths: string[]; marks: any[]; rows: number[]; scores: number[] }
  type Scene = { text: string; path: string; row: number | null; poetryPath?: string }

  // ---- the map and its filter ----
  let map = $state<MapData | null>(null)
  let mapLoading = $state(false)
  let mapError = $state('')
  let marks = $state<MarksInfo>({ has: false })
  let filter = $state<Filter>({ ...EMPTY_FILTER })
  let kept = $state<number[] | null>(null)
  let filterCount = $state('')
  let panelOpen = $state(false)
  let loadedFor = ''

  $effect(() => {
    const ds = app.dataset
    if (!ds) { map = null; marks = { has: false }; loadedFor = ''; return }
    if (ds === loadedFor) return
    loadedFor = ds
    mapLoading = true; mapError = ''
    filter = { ...EMPTY_FILTER }; kept = null; filterCount = ''; panelOpen = false
    groups = []; hits = []; scenes = []
    Promise.all([loadMap(ds), get<MarksInfo>('/api/marks', { dataset: ds })]).then(([m, mk]) => {
      if (loadedFor !== ds) return
      map = m; marks = mk
    }).catch((e) => { mapError = reason(e) }).finally(() => { mapLoading = false })
  })

  $effect(() => {
    const f = filter, ds = app.dataset
    if (!ds || !map) return
    if (!filterActive(f)) { kept = null; filterCount = ''; return }
    filteredRows(ds, f).then((r) => {
      if (r.count === 0) { kept = null; filterCount = 'nothing matches, searching would return an empty page' }
      else { kept = r.kept; filterCount = `searching ${r.count.toLocaleString()} of ${r.total.toLocaleString()}` }
    }).catch(() => {})
  })

  function pickCell(stars: number | 'all', colour: string) {
    filter = { ...filter, stars: stars === 'all' ? filter.stars : stars, colours: colour ? [colour] : filter.colours }
    panelOpen = false
  }

  // ---- search ----
  let text = $state('')
  let k = $state(4)
  let groupTwins = $state(true)
  let thresh = $state(0.08)
  let spectrogram = $state(false)
  let searching = $state(false)
  let groups = $state<Group[]>([])
  let hits = $state<number[]>([])
  let frames = $state<Record<string, number>>({})
  let selected = $state<Record<string, boolean>>({})
  let scenes = $state<Scene[]>([])
  let chunks = $state<string[]>([])
  let clicked = $state<{ path: string; marks: any } | null>(null)
  let saveFolder = $state('')
  let saveStatus = $state(''), saveTone = $state<'ok' | 'warn' | 'bad' | 'dim'>('dim')
  let resultsEl: HTMLDivElement | undefined = $state()

  const body = () => ({ dataset: app.dataset, stars: filter.stars, colours: filter.colours, dateFrom: filter.dateFrom || null, dateTo: filter.dateTo || null })

  async function run() {
    if (!app.dataset) { toast('Choose a dataset above to begin.', 'warn'); return }
    if (!text.trim()) return
    searching = true; clicked = null; saveStatus = ''
    try {
      if (mode === 'prompt') {
        const r = await post('/api/search', { ...body(), text, k, group: groupTwins, thresh })
        groups = r.groups; hits = r.rows
        frames = Object.fromEntries(r.groups.map((g: Group) => [g.gid, 0]))
        selected = Object.fromEntries(r.groups.map((g: Group) => [g.gid, false]))
        scenes = []
      } else {
        const r = await post('/api/story', { ...body(), text })
        scenes = r.scenes; chunks = r.chunks
        groups = []; hits = []
        poetry = null
      }
      resultsEl?.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) {
      toast(reason(e), 'bad')
    } finally {
      searching = false
    }
  }

  async function clickPoint(path: string) {
    try {
      clicked = await post('/api/click', { dataset: app.dataset, path })
      selected = { ...selected, __clicked: false }
      resultsEl?.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) { toast(reason(e), 'bad') }
  }

  function currentPath(g: Group) { return g.paths[frames[g.gid] ?? 0] }
  function selectAll(on: boolean) { const s: Record<string, boolean> = {}; for (const g of groups) s[g.gid] = on; if (clicked) s.__clicked = on; selected = s }
  const nSelected = $derived(Object.values(selected).filter(Boolean).length)

  async function saveSelected() {
    const paths = groups.filter((g) => selected[g.gid]).map(currentPath)
    if (clicked && selected.__clicked) paths.push(clicked.path)
    if (!paths.length) { saveStatus = 'No images selected.'; saveTone = 'warn'; return }
    try {
      const r = await post('/api/save', { paths, folder: saveFolder, prompt: text })
      saveStatus = r.message; saveTone = 'ok'
    } catch (e) { saveStatus = reason(e); saveTone = 'bad' }
  }

  async function saveStory() {
    try {
      const r = await post('/api/story/save', { scenes, chunks, folder: saveFolder, text })
      saveStatus = r.message; saveTone = r.saved ? 'ok' : 'warn'
    } catch (e) { saveStatus = reason(e); saveTone = 'bad' }
  }

  // ---- inject poetry (story + images) ----
  let strength = $state(0.72)
  let poetry = $state<Job | null>(null)
  let poetryDone = $state<{ items: Scene[]; output_dir: string } | null>(null)

  async function injectPoetry() {
    if (!scenes.length) return
    try {
      const r = await post('/api/story/poetry', { scenes, folder: saveFolder, strength })
      poetryDone = null
      watchJob(r.job, (j) => { poetry = j }, (j) => {
        if (j.status === 'done' && j.result) {
          poetryDone = j.result
          scenes = scenes.map((s, i) => ({ ...s, poetryPath: j.result.items[i]?.poetryPath }))
          saveStatus = `Poetry-injected images saved successfully in ${j.result.output_dir}.`; saveTone = 'ok'
        } else if (j.status === 'failed') { saveStatus = `Poetry failed: ${j.error}`; saveTone = 'bad' }
      })
    } catch (e) { toast(reason(e), 'bad') }
  }

  const isAudio = $derived(current.modality === 'audio')
  const mapMessage = $derived(mapError || (app.datasets.length ? 'Choose a dataset above to begin.' : 'No dataset yet. Index a folder on the Datasets tab to begin.'))
</script>

<SplitPane storeKey="search-{mode}" initial={Math.round(window.innerWidth * 0.56)} min={360} max={Math.round(window.innerWidth * 0.8)}>
  {#snippet a()}
    <div class="left">
      <div class="map-wrap">
        <LatentMap data={map} loading={mapLoading} message={mapMessage} {kept} {hits} story={mode === 'story' ? scenes : []} {spectrogram} onclick={clickPoint} {visible} />
        {#if panelOpen && marks.has && marks.matrix}
          <MarksPanel info={marks} onpick={pickCell} onclose={() => (panelOpen = false)} />
        {/if}
      </div>
      <div class="controls">
        <div class="row nowrap">
          <div class="grow">
            {#if mode === 'prompt'}
              <TextInput bind:value={text} placeholder="Describe what you are looking for…" onenter={run} disabled={!app.dataset} />
            {:else}
              <TextArea bind:value={text} placeholder="One scene per line." rows={3} disabled={!app.dataset} />
            {/if}
          </div>
          {#if mode === 'prompt'}<NumberInput bind:value={k} min={1} max={1000} suffix="results" />{/if}
          <Button kind="primary" onclick={run} busy={searching} disabled={!app.dataset || !text.trim()}>{mode === 'prompt' ? 'Search' : 'Generate Story'}</Button>
        </div>
        {#if marks.has}
          <MarksBar info={marks} bind:filter count={filterCount} bind:panelOpen />
        {/if}
      </div>
    </div>
  {/snippet}
  {#snippet b()}
    <div class="right">
      <div class="toolbar">
        {#if mode === 'prompt'}
          <Toggle bind:checked={groupTwins} label="Group twins" size="sm" />
          <div class="thresh" title="Near-duplicates within this cosine distance fold into one card"><span class="caps">distance ≤</span><NumberInput bind:value={thresh} min={0} max={0.5} step={0.01} size="sm" width="60px" /></div>
        {/if}
        {#if isAudio}<Toggle bind:checked={spectrogram} label="Spectrogram" size="sm" />{/if}
        {#if mode === 'story' && !isAudio}
          <div class="poetry">
            <Button size="sm" kind="primary" onclick={injectPoetry} disabled={!scenes.length || (!!poetry && !['done', 'failed', 'cancelled'].includes(poetry.status))}>Inject Poetry</Button>
            <Slider bind:value={strength} min={0} max={1} step={0.01} label="Strength" width="190px" />
          </div>
        {/if}
        <span class="spacer"></span>
        {#if groups.length || clicked}
          <span class="dim small">{nSelected} selected</span>
          <Button size="sm" onclick={() => selectAll(true)}>Select all</Button>
          <Button size="sm" onclick={() => selectAll(false)}>Clear</Button>
        {/if}
      </div>

      {#if poetry && !['done'].includes(poetry.status)}
        <JobCard job={poetry} ondismiss={() => (poetry = null)} />
      {/if}

      <div class="results" bind:this={resultsEl}>
        {#if clicked}
          <MediaCard paths={[clicked.path]} marks={[clicked.marks]} {spectrogram} bind:selected={selected.__clicked} title="From the map" />
        {/if}
        {#if mode === 'prompt'}
          {#if !groups.length && !clicked}
            <Empty title={app.dataset ? 'Describe what you are looking for, then Search.' : 'Choose a dataset above to begin.'} hint="CLIP sees concrete visual things: objects, places, light, weather, colour, texture, composition. Click any point on the map to see it here." />
          {/if}
          {#each groups as g, i (g.gid)}
            <MediaCard paths={g.paths} marks={g.marks} scores={g.scores} {spectrogram} bind:selected={selected[g.gid]} bind:frame={frames[g.gid]}
                       title="#{i + 1}" />
          {/each}
        {:else}
          {#if !scenes.length}
            <Empty title={app.dataset ? 'Write one scene per line, then Generate Story.' : 'Choose a dataset above to begin.'} hint="Each line finds its closest picture, and the map draws the path between them in order." />
          {/if}
          {#each scenes as s, i (i)}
            <article class="scene">
              <h5 class="scene-t">{i + 1}. {s.text}</h5>
              {#if s.poetryPath}
                <div class="pair">
                  <MediaCard paths={[s.path]} selectable={false} {spectrogram} title="original" />
                  <article class="card poetry-card"><div class="head dim">poetry</div><img src={previewUrl(s.poetryPath, 900)} alt="" /><div class="row"><Button size="sm" kind="ghost" onclick={() => addToCollection([s.poetryPath!])}>+ Collection</Button></div></article>
                </div>
              {:else}
                <MediaCard paths={[s.path]} selectable={false} {spectrogram} />
              {/if}
            </article>
          {/each}
        {/if}
      </div>

      <Disclosure label="Save" note={mode === 'prompt' ? 'selected pictures to output/search' : 'the story to output/stories'} level={1}>
        <div class="row nowrap">
          <div class="grow"><TextInput bind:value={saveFolder} placeholder={mode === 'prompt' ? 'Folder name under output/search (optional)' : 'Folder name under output/stories (optional)'} /></div>
          {#if mode === 'prompt'}
            <Button kind="success" onclick={saveSelected} disabled={!nSelected}>Save selected</Button>
          {:else}
            <Button kind="success" onclick={saveStory} disabled={!scenes.length}>Save story</Button>
          {/if}
        </div>
        <Status text={saveStatus} tone={saveTone} />
      </Disclosure>
    </div>
  {/snippet}
</SplitPane>

<style>
  .left { display: flex; flex-direction: column; gap: var(--sp-2); min-height: 0; height: 100%; }
  .map-wrap { flex: 1; min-height: 0; position: relative; display: flex; }
  .controls { flex: 0 0 auto; display: flex; flex-direction: column; gap: 0; padding: 10px 12px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); }
  .controls .row { gap: 10px; }
  .right { display: flex; flex-direction: column; gap: var(--sp-2); min-height: 0; height: 100%; padding-left: var(--sp-1); }
  .toolbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 8px 12px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); flex: 0 0 auto; }
  .thresh { display: inline-flex; align-items: center; gap: 6px; }
  .poetry { display: inline-flex; align-items: center; gap: 10px; }
  .small { font-size: var(--fs-xs); }
  .results { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; display: flex; flex-direction: column; gap: 12px; padding-right: 4px; }
  .scene { display: flex; flex-direction: column; gap: 6px; }
  .scene-t { font-size: var(--fs-md); color: var(--arc-warn); font-weight: 600; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .poetry-card { display: flex; flex-direction: column; gap: 8px; padding: 10px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); }
  .poetry-card img { width: 100%; border-radius: var(--r-md); }
  .poetry-card .head { font-size: var(--fs-xs); }
</style>
