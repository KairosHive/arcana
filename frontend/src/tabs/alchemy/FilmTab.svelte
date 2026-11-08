<script lang="ts">
  import { onMount } from 'svelte'
  import { app, onStateChange, type Job } from '../../lib/stores/app.svelte'
  import { post, rpc, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { thumbUrl, filmVideoUrl, basename } from '../../lib/api/media'
  import { film, commit, refreshMeta, addPicture, spacePictures, addLane, removeSelected, newFilm, openFilm, keysOf, clone, EASES, SPACES, SHAPES, type FilmSpec } from './film.svelte'
  import Timeline from './Timeline.svelte'
  import Card from '../../lib/ui/Card.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import ProgressBar from '../../lib/ui/ProgressBar.svelte'

  /** The Video subtab: pictures on the left, the player and timeline, the film's settings. */
  let { active = false }: { active?: boolean } = $props()
  const items = $derived(app.collection.items)

  // ---- the player ----
  let frame = $state<string | null>(null)
  let frameInfo = $state('')
  let out = false, held = false
  let frameTimer: ReturnType<typeof setTimeout> | null = null
  let wall = 0, from = 0

  async function askFrame() {
    if (!active) return
    if (out) { held = true; return }
    out = true
    try {
      const r = await post('/api/film/frame', { spec: film.spec, at: film.t, size: 512 })
      if (r.src) frame = r.src
      frameInfo = r.info || ''
    } catch (e) { frameInfo = reason(e) } finally {
      out = false
      if (film.playing) advance()
      else if (held) { held = false; askFrame() }
    }
  }
  $effect(() => { film.spec; film.t; active; if (frameTimer) clearTimeout(frameTimer); frameTimer = setTimeout(askFrame, 40) })

  function toggle() {
    film.playing = !film.playing
    wall = performance.now(); from = film.t
    if (film.playing) advance()
  }
  function advance() {
    if (!film.playing || !film.meta) return
    const L = film.meta.length || 1
    let t = from + (performance.now() - wall) / 1000
    if (t >= L) {
      if (film.spec.loop === false) { film.playing = false; t = L } else t = t % L
    }
    film.t = t
  }
  const clock = $derived.by(() => {
    const f = (x: number) => { const m = Math.floor(x / 60); const s = (x - 60 * m).toFixed(2); return (m ? m + ':' : '') + (m && x - 60 * m < 10 ? '0' : '') + s }
    return `${f(film.t || 0)} / ${f(film.meta?.length || 0)}`
  })

  // ---- fields ----
  function setField<K extends keyof FilmSpec>(k: K, v: FilmSpec[K]) { const s = clone(film.spec); (s as any)[k] = v; commit(s) }
  const length = $derived(film.meta?.normalised?.length ? Math.round(film.meta.normalised.length * 100) / 100 : film.spec.length ?? null)

  // ---- selected key ----
  const selected = $derived.by(() => {
    const sel = film.sel
    if (!sel) return null
    const s = clone(film.spec)
    const keys = keysOf(s, sel.lane)
    if (!keys || sel.i < 0 || sel.i >= keys.length) return null
    const key = keys[sel.i]
    let label: string
    if (sel.lane === 'pictures') label = `Picture ${sel.i + 1}: ${basename(String(key.ref))}`
    else if (sel.lane === 'looks') { const m = key.settings?.method || 'none'; label = `Look ${sel.i + 1}: ${m === 'none' ? 'morph only' : m}` }
    else label = `${film.meta?.lanes[sel.lane]?.label ?? sel.lane}, key ${sel.i + 1}`
    return { key, label, hasV: 'v' in key }
  })
  function editKey(patch: { t?: number; v?: number; ease?: string }) {
    const sel = film.sel
    if (!sel) return
    const s = clone(film.spec)
    const keys = keysOf(s, sel.lane)
    if (!keys || !keys[sel.i]) return
    const key = keys[sel.i]
    if (patch.t !== undefined) key.t = Math.round(Math.max(0, patch.t) * 1000) / 1000
    if (patch.v !== undefined && 'v' in key) key.v = patch.v
    if (patch.ease) key.ease = patch.ease
    commit(s)
  }
  function key(e: KeyboardEvent) {
    if (!active) return
    const t = e.target as HTMLElement | null
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return
    if ((e.key === 'Delete' || e.key === 'Backspace') && film.sel) { removeSelected(); e.preventDefault() }
    else if (e.key === ' ') { toggle(); e.preventDefault() }
  }

  // ---- kept films and rendering ----
  let films = $state<{ value: string; label: string }[]>([])
  let pick = $state<string | null>(null)
  let lanePick = $state<string | null>(null)
  let draw = $state(768), target = $state(0)
  let job = $state<Job | null>(null)
  let video = $state<string | null>(null)
  let renderStatus = $state('')

  async function listFilms() {
    try { films = (await rpc('alchemy film list')).result.films.map((f: any) => ({ value: f.name, label: `${f.name} · ${f.pictures.length} pictures` })) } catch { films = [] }
  }
  async function open() {
    if (!pick) { film.status = 'Choose a kept film first.'; return }
    try { openFilm((await rpc('alchemy film show', { film: pick })).result.spec, pick) } catch (e) { film.status = reason(e) }
  }
  async function keep() {
    try { const r = await rpc('alchemy film save', { spec: film.spec }); film.status = `Kept as ${r.result.film.name}.`; listFilms() } catch (e) { film.status = reason(e) }
  }
  async function del() {
    if (!pick || !confirm(`Delete the kept film “${pick}”? Its videos stay.`)) return
    try { await rpc('alchemy film delete', { film: pick }); film.status = `Deleted the film ${pick}.`; pick = null; listFilms() } catch (e) { film.status = reason(e) }
  }
  async function render() {
    renderStatus = 'Starting the render…'; video = null
    try {
      const r = await post('/api/film/render', { spec: film.spec, size: draw, target: target || undefined })
      watchJob(r.job, (j) => { job = j; if (j.status === 'running' || j.status === 'queued') renderStatus = j.message || 'Rendering…' }, (j) => {
        if (j.status === 'done' && j.result) {
          const res = j.result
          renderStatus = `Rendered ${res.frames} frames, ${res.seconds.toFixed(1)} s at ${res.size[0]} x ${res.size[1]}, as ${res.id}.`
          video = filmVideoUrl(res.path)
        } else renderStatus = j.error || `The render ended: ${j.status}.`
        listFilms()
      })
    } catch (e) { renderStatus = reason(e) }
  }

  onMount(() => { refreshMeta(); listFilms(); return onStateChange(listFilms) })
</script>

<svelte:window onkeydown={key} />

<div class="filmtab">
  <Card title="Pictures" subtitle="Click a picture to put it on the timeline at the playhead. A picture on the timeline is dragged along it to change when the film passes it." pad="sm" scroll style="flex: 0 0 220px; max-height: 100%">
    <div class="cast">
      {#if !items.length}<p class="hint">The active collection is empty. Pictures added with + Collection on any tab appear here.</p>{/if}
      {#each items as p (p)}
        <button class="pic" title={basename(p)} onclick={() => addPicture(p)}><img src={thumbUrl(p)} alt="" loading="lazy" /></button>
      {/each}
    </div>
  </Card>

  <div class="stage">
    <div class="player">
      {#if frame}<img class="frame" src={frame} alt="" />{/if}
      {#if film.meta?.empty}<p class="empty hint">{film.meta.empty}</p>{/if}
    </div>
    <div class="row nowrap">
      <button class="play" title="Play a draft (space)" onclick={toggle}>{film.playing ? '❚❚' : '▶'}</button>
      <span class="clock mono">{clock}</span>
      <span class="spacer"></span>
      <span class="info faint">{frameInfo}</span>
    </div>
    <Timeline />
    <Status text={film.meta?.status || film.status} tone="ink" size="xs" />
  </div>

  <div class="settings">
    <Disclosure label="Film" note="length, shape, model" open>
      <Field label="Name"><TextInput value={film.spec.name} size="sm" oncommit={(v) => setField('name', (v.trim() || film.spec.name))} /></Field>
      <div class="row">
        <Field label="Length (s)"><NumberInput value={length ?? 0} min={1} step={0.5} size="sm" width="70px" onchange={(v) => { if (v) setField('length', v) }} /></Field>
        <Field label="FPS"><NumberInput value={film.spec.fps} min={6} max={60} step={1} size="sm" width="60px" onchange={(v) => setField('fps', Math.round(v))} /></Field>
        <Field label="Size"><NumberInput value={film.spec.size} min={256} max={2048} step={16} size="sm" width="72px" onchange={(v) => setField('size', Math.round(v))} /></Field>
      </div>
      <Field label="Shape"><Select value={film.spec.aspect} options={[{ value: 'auto', label: "the first picture's" }, ...['3:2', '2:3', '4:3', '16:9', '9:16', '1:1'].map((a) => ({ value: a, label: a }))]} size="sm" onchange={(v) => setField('aspect', String(v))} /></Field>
      <Field label="Morph through"><Select value={film.spec.space} options={Object.entries(SPACES).map(([k, v]) => ({ value: k, label: `${k}: ${v}` }))} size="sm" onchange={(v) => setField('space', String(v))} /></Field>
      <div class="row">
        <Checkbox checked={film.spec.loop} label="loop back to the first picture" onchange={(v) => setField('loop', v)} />
        <Checkbox checked={film.spec.even} label="even motion between pictures" onchange={(v) => setField('even', v)} />
      </div>
    </Disclosure>

    <Disclosure label="Timeline" note="spacing, lanes">
      <span class="caps">Space the pictures</span>
      <div class="row">{#each SHAPES as s}<Button size="sm" title={s.doc} onclick={() => spacePictures(s.key)}>{s.label}</Button>{/each}</div>
      <span class="caps">Add a lane</span>
      <div class="row nowrap">
        <div class="grow"><Select bind:value={lanePick} options={film.meta?.laneOptions ?? []} placeholder="a setting to key in time…" size="sm" /></div>
        <Button size="sm" disabled={!lanePick} onclick={() => { if (lanePick) addLane(lanePick); lanePick = null }}>Add</Button>
      </div>
      <p class="hint">Looks come from the Image subtab: set a step there and press Add look to film. Between two looks every number moves from one to the other.</p>
    </Disclosure>

    <Disclosure label="Selected key" note="time, value, ease" open>
      {#if selected}
        <p class="sel-title">{selected.label}</p>
        <div class="row">
          <Field label="Time (s)"><NumberInput value={Math.round(selected.key.t * 1000) / 1000} min={0} step={0.05} size="sm" width="80px" onchange={(v) => editKey({ t: v })} /></Field>
          {#if selected.hasV}<Field label="Value"><NumberInput value={selected.key.v ?? 0} step="any" size="sm" width="90px" onchange={(v) => editKey({ v })} /></Field>{/if}
          <Field label="Ease to the next key"><Select value={selected.key.ease ?? 'linear'} options={EASES.map((e) => ({ value: e, label: e }))} size="sm" onchange={(v) => editKey({ ease: String(v) })} /></Field>
        </div>
        <div><Button size="sm" kind="ghost" onclick={removeSelected}>Remove key</Button></div>
      {:else}
        <p class="hint">Click a key on the timeline to edit it here.</p>
      {/if}
    </Disclosure>

    <Disclosure label="Keep and render" note="save, open, video" open>
      <div class="row nowrap">
        <div class="grow"><Select bind:value={pick} options={films} placeholder="Kept films…" size="sm" /></div>
        <Button size="sm" onclick={open} disabled={!pick}>Open</Button>
        <Button size="sm" kind="ghost" onclick={newFilm}>New</Button>
        <Button size="sm" kind="danger" onclick={del} disabled={!pick}>Delete</Button>
      </div>
      <div class="row"><Button size="sm" onclick={keep}>Keep this film</Button><Status text={film.status} size="xs" /></div>
      <div class="row">
        <Field label="Draw at"><Select bind:value={draw} options={[512, 640, 768, 1024, 1536].map((n) => ({ value: n, label: `${n} px` }))} size="sm" /></Field>
        <Field label="Video"><Select bind:value={target} options={[{ value: 0, label: 'as drawn' }, ...[1920, 2560, 3840].map((n) => ({ value: n, label: `${n} px (Real-ESRGAN)` }))]} size="sm" /></Field>
        <Button size="sm" kind="primary" onclick={render} disabled={!film.meta?.ok || (!!job && ['running', 'queued'].includes(job.status))}>Render</Button>
      </div>
      {#if job && ['running', 'queued'].includes(job.status)}<ProgressBar fraction={job.fraction} />{/if}
      <Status text={renderStatus} size="xs" tone={job?.status === 'failed' ? 'bad' : 'dim'} />
      {#if video}<video class="video" src={video} controls loop><track kind="captions" /></video>{/if}
    </Disclosure>
  </div>
</div>

<style>
  .filmtab { display: flex; gap: 14px; flex: 1; min-width: 0; min-height: 0; flex-wrap: wrap; align-content: flex-start; overflow-y: auto; }
  .filmtab > :global(.card:first-child) { flex: 0 0 220px; max-height: 100%; }
  .cast { display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px; flex: 1 1 auto; min-height: 160px; max-height: 62vh; overflow-y: auto; padding: 8px 2px; align-content: start; }
  .cast .hint { grid-column: 1 / -1; font-size: var(--fs-xs); }
  .pic { padding: 0; border: 1px solid var(--arc-line); border-radius: 5px; overflow: hidden; background: var(--arc-surface-2); cursor: copy; aspect-ratio: 1; }
  .pic img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .stage { flex: 1 1 460px; min-width: min(460px, 100%); display: flex; flex-direction: column; gap: 8px; padding: 12px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); }
  .player { position: relative; background: #0e0d0c; min-height: 220px; border-radius: var(--r-md); display: flex; align-items: center; justify-content: center; }
  .frame { max-width: 100%; max-height: 50vh; display: block; }
  .empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center; padding: 20px; }
  .play { width: 36px; height: 30px; border-radius: var(--r-md); border: 1px solid var(--arc-line); background: var(--arc-surface-2); color: var(--arc-ink); }
  .clock { font-variant-numeric: tabular-nums; font-size: var(--fs-sm); }
  .info { font-size: 11.5px; }
  .settings { flex: 0 0 330px; max-width: 100%; display: flex; flex-direction: column; gap: var(--sp-2); }
  .sel-title { font-size: var(--fs-sm); }
  .video { width: 100%; margin-top: 8px; border-radius: var(--r-md); }
</style>
