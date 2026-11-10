<script lang="ts">
  import { onMount } from 'svelte'
  import { app, setMode, type Job } from '../../lib/stores/app.svelte'
  import { post, reason } from '../../lib/api/client'
  import { previewUrl, thumbUrl } from '../../lib/api/media'
  import { toast } from '../../lib/stores/toast.svelte'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { openViewer } from '../../lib/stores/viewer.svelte'
  import { learn, loadAll } from '../../lib/stores/learn.svelte'
  import { setRole } from '../alchemy/alchemy.svelte'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Slider from '../../lib/ui/Slider.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Toggle from '../../lib/ui/Toggle.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Tabs from '../../lib/ui/Tabs.svelte'
  import JobCard from '../../lib/components/JobCard.svelte'
  import Empty from '../../lib/ui/Empty.svelte'

  /**
   * Generate: new material drawn from places in your archive. A place is a set, an IP label (a position
   * measured on your pictures) or words. Several places blend by weight. Drawn from the place, Stable
   * Diffusion reads the blended position (it mixes evenly, pushes, walks and tiles); drawn from the
   * pictures, FLUX.2 klein looks at the places' own photographs (faithful to them). Drafts until kept.
   */
  const visible = $derived(app.mode === 'generate')
  onMount(() => { void loadAll() })
  $effect(() => { if (visible) void loadAll() })

  type Src = { kind: 'set' | 'label'; id: string; name: string; thumbs: string[]; sub: string }
  let weights = $state<Record<string, number>>({})
  const key = (s: Src) => `${s.kind}:${s.id}`
  // A set and the label measured on it are one place: the label is listed, the set only when it has none yet.
  const sources = $derived.by<Src[]>(() => {
    const labelled = new Set(learn.labels.filter((l) => l.origin?.kind === 'set').map((l) => l.origin.set))
    const out: Src[] = learn.labels.filter((l) => !l.error && !l.slug.startsWith('picture-')).map((l) => ({
      kind: 'label', id: l.slug, name: l.name, thumbs: l.nearest.slice(0, 4),
      sub: l.origin?.kind === 'set' ? `set · ${l.pictures} pictures` : l.origin?.kind === 'words' ? `words · ${l.pictures} pictures` : `${l.pictures} pictures`,
    }))
    for (const s of learn.sets) if (!labelled.has(s.slug)) out.push({ kind: 'set', id: s.slug, name: s.name, thumbs: [], sub: `set · ${s.count} pictures` })
    return out
  })
  const chosen = $derived(sources.filter((s) => (weights[key(s)] ?? 0) > 0))

  let words = $state('')
  let engine = $state<'place' | 'pictures'>('place')
  let n = $state(4), seed = $state(0), size = $state(512)
  let tileable = $state(false), push = $state(2.0), spread = $state(0.25), frames = $state(6)
  let axis = $state<string | null>('none'), amount = $state(0)
  const axisOptions = $derived([{ value: 'none', label: 'no axis' }, ...learn.axes.map((a) => ({ value: a.slug, label: `axis: ${a.name}` }))])
  let job = $state<Job | null>(null)
  type Draft = { path: string; seed: number; t?: number | null; kept?: string }
  let results = $state<{ title: string; drafts: Draft[] }[]>([])
  const busy = $derived(!!job && !['done', 'failed', 'cancelled'].includes(job.status))

  function payload() {
    return chosen.map((s) => ({ kind: s.kind, id: s.id, weight: weights[key(s)] }))
  }
  function title(walk = false) {
    const names = chosen.map((s) => `${s.name}${chosen.length > 1 ? ` ${Math.round((weights[key(s)] / chosen.reduce((a, c) => a + weights[key(c)], 0)) * 100)}%` : ''}`)
    const w = words.trim() ? `“${words.trim()}”` : ''
    return walk ? `walk · ${chosen[0]?.name} → ${chosen[1]?.name}${w ? ' · ' + w : ''}` : [...names, w].filter(Boolean).join(' + ') + ` · ${engine === 'place' ? 'from the place' : 'from the pictures'}`
  }
  function follow(r: any, t: string) {
    watchJob(r.job, (j) => { job = j }, (j) => {
      job = j
      if (j.status === 'done' && j.result?.drafts) results = [{ title: t, drafts: j.result.drafts }, ...results]
      else if (j.status === 'failed') toast(j.error || 'Drawing failed.', 'bad')
    })
  }
  async function draw() {
    if (!chosen.length && !words.trim()) { toast('Give a place a weight, or write the material in words.', 'warn'); return }
    try {
      const r = await post('/api/generate/draw', { sources: payload(), words, engine, n, seed, size, tileable: engine === 'place' && tileable, push, spread, axis, amount })
      follow(r, title())
      seed += n
    } catch (e) { toast(reason(e), 'bad') }
  }
  async function walk() {
    if (chosen.length !== 2) { toast('A walk goes between exactly two places: give two a weight.', 'warn'); return }
    try {
      const [a, b] = chosen
      const r = await post('/api/generate/walk', { a: [{ kind: a.kind, id: a.id, weight: 1 }], b: [{ kind: b.kind, id: b.id, weight: 1 }], words, frames, seed, size, tileable })
      follow(r, title(true))
    } catch (e) { toast(reason(e), 'bad') }
  }
  async function keep(d: Draft, collect: boolean, use = false) {
    try {
      if (!d.kept) {
        const r = await post('/api/generate/keep', { draft: d.path, collect: collect || use })
        d.kept = r.path
        toast(`Kept as ${r.handle}${collect || use ? ', on the collection' : ''}`, 'ok', 2400)
      }
      if (use && d.kept) { setRole('1', d.kept); setMode('alchemy') }
    } catch (e) { toast(reason(e), 'bad') }
  }
</script>

<div class="gen">
  <Card title="Places" subtitle="Draw from sets and IP labels measured on your pictures, by weight, and from words. Sets and labels are made in Search (Save to a set) and in Alchemy (Sets & labels)." pad="sm" scroll style="flex: 0 0 340px; max-height: 100%">
    <div class="places">
      {#if !sources.length}<p class="hint">No sets or labels yet. Save search results to a set in Search, and it appears here.</p>{/if}
      {#each sources as s (key(s))}
        {@const w = weights[key(s)] ?? 0}
        <div class="place" class:on={w > 0}>
          <div class="strip">
            {#each s.thumbs as p (p)}<img src={thumbUrl(p)} alt="" loading="lazy" />{/each}
            {#if !s.thumbs.length}<span class="nothumb">set</span>{/if}
          </div>
          <div class="ph"><b>{s.name}</b><span class="dim">{s.sub}</span></div>
          <Slider value={w} min={0} max={1} step={0.05} label="weight" width="100%" onchange={(v: number) => (weights = { ...weights, [key(s)]: v })} />
        </div>
      {/each}
      <TextInput bind:value={words} placeholder="Words, alone or added to the places: molten copper" />
    </div>
  </Card>

  <div class="mid">
    <Card pad="sm" style="flex: 0 0 auto">
      <div class="controls">
        <Tabs bind:value={engine} size="sm" tabs={[{ value: 'place', label: 'From the place' }, { value: 'pictures', label: 'From the pictures' }]} />
        <p class="hint grow">{engine === 'place'
          ? 'Stable Diffusion at the places’ blended position in IP space: mixes evenly, pushes past the archive, walks and tiles.'
          : 'FLUX.2 klein looking at the places’ own photographs: faithful to them, uneven when places are mixed.'}</p>
      </div>
      <div class="controls">
        <NumberInput bind:value={n} min={1} max={16} size="sm" width="64px" suffix="pictures" />
        <NumberInput bind:value={seed} min={0} max={99999} size="sm" width="78px" suffix="seed" />
        <Select bind:value={size} options={[{ value: 512, label: '512' }, { value: 768, label: '768' }, { value: 1024, label: '1024' }]} size="sm" width="80px" />
        {#if engine === 'place'}
          <Toggle bind:checked={tileable} label="tileable" size="sm" />
          <Slider bind:value={push} min={0.5} max={4} step={0.1} label="push" width="150px" />
          <Slider bind:value={spread} min={0} max={1} step={0.05} label="spread" width="150px" />
          {#if learn.axes.length}
            <Select bind:value={axis} options={axisOptions} size="sm" width="140px" />
            {#if axis && axis !== 'none'}<Slider bind:value={amount} min={-2} max={2} step={0.05} label="along" width="150px" />{/if}
          {/if}
        {/if}
        <span class="grow"></span>
        <Button kind="primary" onclick={draw} busy={busy} disabled={busy}>Draw</Button>
        {#if engine === 'place'}
          <NumberInput bind:value={frames} min={2} max={24} size="sm" width="64px" suffix="frames" />
          <Button onclick={walk} disabled={busy || chosen.length !== 2} title="From the first weighted place to the second, the seed held">Walk</Button>
        {/if}
      </div>
    </Card>
    {#if job && job.status !== 'done'}<JobCard {job} ondismiss={() => (job = null)} />{/if}
    <div class="results">
      {#if !results.length}
        <Empty title="Give one or more places a weight, then Draw." hint="Each place is a region of your archive in IP space. Two places with weight draw a blend; Walk draws the way from one to the other." />
      {/if}
      {#each results as r, ri (ri)}
        <section class="batch">
          <h4>{r.title}</h4>
          <div class="grid">
            {#each r.drafts as d, i (d.path)}
              <figure>
                <button class="pic" onclick={() => openViewer(r.drafts.map((x) => x.path), i, r.title)}><img src={previewUrl(d.path, 700)} alt="" loading="lazy" /></button>
                <figcaption>
                  <span class="dim">{d.t !== undefined && d.t !== null ? `t ${d.t}` : `seed ${d.seed}`}</span>
                  <span class="grow"></span>
                  <Button size="sm" kind="ghost" onclick={() => keep(d, false)} disabled={!!d.kept}>{d.kept ? 'Kept' : 'Keep'}</Button>
                  <Button size="sm" onclick={() => keep(d, false, true)} title="Keep it, put it on the collection and make it material [1] in Alchemy">Use in Alchemy</Button>
                </figcaption>
              </figure>
            {/each}
          </div>
        </section>
      {/each}
    </div>
  </div>
</div>

<style>
  .gen { display: flex; gap: 14px; flex: 1; min-height: 0; }
  .places { display: flex; flex-direction: column; gap: 10px; }
  .place { display: flex; flex-direction: column; gap: 4px; padding: 8px; border: 1px solid var(--arc-line); border-radius: var(--r-md); opacity: 0.8; }
  .place.on { opacity: 1; border-color: var(--tab-generate); }
  .strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; }
  .strip img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 3px; display: block; }
  .nothumb { grid-column: 1 / -1; font-size: var(--fs-2xs); color: var(--arc-ink-faint); }
  .ph { display: flex; justify-content: space-between; gap: 6px; font-size: var(--fs-sm); }
  .ph span { font-size: var(--fs-2xs); }
  .mid { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; min-height: 0; }
  .controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .controls + .controls { margin-top: 8px; }
  .grow { flex: 1; }
  .hint { margin: 0; font-size: var(--fs-xs); color: var(--arc-ink-faint); }
  .results { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }
  .batch h4 { margin: 0 0 6px; font-size: var(--fs-sm); font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
  figure { margin: 0; display: flex; flex-direction: column; gap: 4px; }
  .pic { padding: 0; border: 0; background: var(--arc-surface-2); cursor: zoom-in; border-radius: var(--r-md); overflow: hidden; }
  .pic img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
  figcaption { display: flex; align-items: center; gap: 4px; font-size: var(--fs-2xs); }
  @media (max-width: 1000px) { .gen { flex-direction: column; overflow-y: auto; } }
</style>
