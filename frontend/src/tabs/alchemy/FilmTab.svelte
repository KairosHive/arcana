<script lang="ts">
  import { onMount } from 'svelte'
  import { app, onStateChange, type Job } from '../../lib/stores/app.svelte'
  import { post, rpc, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { thumbUrl, filmVideoUrl, basename } from '../../lib/api/media'
  import { film, commit, refreshMeta, addPicture, spacePictures, addLane, addLook, replaceLook, removeLook, lookSummary, removeSelected, newFilm, openFilm, keysOf, clone, sorted, laneValue, shapeLane, placeSeries, isFilmRef, isClipRef, playsRef, FILM_REF, LANE_SHAPES, SPACES, type FilmSpec, type LookSummary } from './film.svelte'
  import { controls, loadStep, alchemyUi, catalog, method as methodOf, step } from './alchemy.svelte'
  import Timeline from './Timeline.svelte'
  import Card from '../../lib/ui/Card.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Toggle from '../../lib/ui/Toggle.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import ProgressBar from '../../lib/ui/ProgressBar.svelte'

  /** The Video subtab: pictures on the left, the player and timeline, the film's settings. */
  let { active = false }: { active?: boolean } = $props()
  const items = $derived(app.collection.items)
  const PATH_TYPE = 'text/arcana-path'
  const EASES = [['linear', 'straight'], ['smooth', 'smooth'], ['in', 'slow start'], ['out', 'slow end'], ['hold', 'hold']]

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

  // ---- the look at the playhead ----
  let look = $state<LookSummary>({ index: null, count: 0 })
  let lookTimer: ReturnType<typeof setTimeout> | null = null
  $effect(() => { film.spec; film.t; if (!active) return; if (lookTimer) clearTimeout(lookTimer); lookTimer = setTimeout(async () => { try { look = await lookSummary() } catch { /* keep the last */ } }, 120) })
  let lookStatus = $state('')
  async function editLook() {
    if (look.index === null || !look.settings) { lookStatus = 'There is no look here to edit.'; return }
    try {
      await loadStep(look.settings)
      alchemyUi.sub = 'image'
      lookStatus = `Look ${look.index + 1} is open in the Image subtab.`
    } catch (e) { lookStatus = reason(e) }
  }

  // ---- fields ----
  function setField<K extends keyof FilmSpec>(k: K, v: FilmSpec[K]) { const s = clone(film.spec); (s as any)[k] = v; commit(s) }
  const length = $derived(film.meta?.normalised?.length ? Math.round(film.meta.normalised.length * 100) / 100 : film.spec.length ?? null)

  // ---- lanes ----
  let lanePick = $state<string | null>(null)
  const laneDoc = $derived.by(() => { const m = lanePick ? film.meta?.lanes[lanePick] : null; return m ? `${m.doc ?? ''}. From ${m.lo} to ${m.hi}.` : '' })

  // ---- a lane's whole curve ----
  const selLane = $derived(film.sel && film.sel.lane !== 'pictures' && film.sel.lane !== 'looks' && film.spec.lanes[film.sel.lane] ? film.sel.lane : null)
  let shapeLo = $state(0), shapeHi = $state(1), shapeCount = $state(3), shapeOver = $state<'film' | 'after'>('film')
  let shapeStatus = $state('')
  let shapeFor = ''
  $effect(() => {
    const m = selLane ? film.meta?.lanes[selLane] : null
    if (!selLane || !m || shapeFor === selLane) return
    shapeFor = selLane
    const now = laneValue(sorted(film.spec.lanes[selLane] ?? []), film.t)
    shapeLo = Math.round(now * 1000) / 1000
    // A lane of 0 to 1 goes to its far end; a wide slider moves a visible step, 15% of its range.
    const far = Math.abs(m.hi - now) >= Math.abs(now - m.lo) ? m.hi : m.lo
    const span = m.hi - m.lo
    const to = span > 2 ? now + Math.sign(far - now) * 0.15 * span : far
    shapeHi = Math.round(Math.min(Math.max(to, m.lo), m.hi) * 1000) / 1000
  })
  async function shape(kind: string) {
    if (!selLane) return
    const L = film.meta?.length || 1
    const start = shapeOver === 'after' ? film.t : 0
    shapeStatus = await shapeLane(selLane, kind, shapeLo, shapeHi, start, L, shapeCount)
    film.sel = null
  }

  // ---- derivatives of one picture ----
  const VARY = [
    { value: 'seed', label: 'new seeds: similar textures' },
    { value: 'material', label: 'each picture of the collection as material: different textures' },
    { value: 'recipe', label: 'each chosen recipe: different looks' },
    { value: 'knob', label: 'one knob stepped: the look growing' },
  ]
  let vary = $state('seed'), deriveCount = $state(4), deriveKnob = $state<string | null>(null)
  let deriveStart = $state(0), deriveEnd = $state(1), knobFor = ''
  $effect(() => {
    const k = numberKnobs.find((x) => x.key === deriveKnob)
    if (!k || knobFor === k.key) return
    knobFor = k.key
    deriveStart = k.lo ?? 0; deriveEnd = typeof k.default === 'number' && k.default > (k.lo ?? 0) ? k.default : (k.hi ?? 1)
  })
  let deriveEvery = $state(3), deriveBack = $state(true), deriveSteady = $state(true), deriveRecipes = $state<string[]>([])
  let byChange = $state(true), timingStatus = $state('')
  let deriveJob = $state<Job | null>(null), deriveStatus = $state('')
  const derivePic = $derived.by(() => {
    const sel = film.sel
    if (!sel || sel.lane !== 'pictures') return null
    const k = sorted(film.spec.pictures)[sel.i]
    return k && !playsRef(k.ref) ? k : null
  })
  const stepMethod = $derived(methodOf(step.method))
  const numberKnobs = $derived((stepMethod?.knobs ?? []).filter((k) => ['float', 'int', 'angle'].includes(k.kind) && k.lo !== undefined))
  const otherPictures = $derived(items.filter((p) => p !== derivePic?.ref).slice(0, 12))
  const recipeNames = $derived((catalog.data?.recipes ?? []).map((r) => r.name))
  const deriveBusy = $derived(!!deriveJob && ['running', 'queued'].includes(deriveJob.status))
  async function derive() {
    const pic = derivePic
    if (!pic?.ref) { deriveStatus = 'Click a picture on the picture lane first.'; return }
    const body: Record<string, any> = { controls: controls(), picture: pic.ref, vary, count: deriveCount, size: 1024 }
    if (vary === 'material') body.each = otherPictures
    if (vary === 'recipe') body.each = deriveRecipes
    if (vary === 'knob') { body.knob = deriveKnob; body.start = deriveStart; body.end = deriveEnd }
    deriveStatus = 'Starting…'
    try {
      const r = await post('/api/film/derive', body)
      const at = pic.t, ref = String(pic.ref)
      watchJob(r.job, (j) => { deriveJob = j; if (['running', 'queued'].includes(j.status)) deriveStatus = j.message || 'Drawing…' }, async (j) => {
        deriveJob = j
        if (j.status !== 'done' || !j.result) { deriveStatus = j.error || `The derivatives ended: ${j.status}.`; return }
        const ids: string[] = j.result.ids
        const refs = deriveBack ? [...ids, ref] : ids
        let why = await placeSeries(refs, at + deriveEvery, deriveEvery)
        if (!why && deriveSteady) { deriveStatus = 'Measuring the passages for a steady pace…'; why = await spacePictures('even', 'change') }
        deriveStatus = why || `${ids.length} derivatives after the picture, ${deriveEvery} s apart: ${j.result.labels.join(', ')}. They are kept in output and the Archive.`
      })
    } catch (e) { deriveStatus = reason(e) }
  }

  // ---- selected key ----
  const selected = $derived.by(() => {
    const sel = film.sel
    if (!sel) return null
    const s = clone(film.spec)
    const keys = keysOf(s, sel.lane)
    if (!keys || sel.i < 0 || sel.i >= keys.length) return null
    const key = keys[sel.i]
    let label: string, doc: string
    const m = film.meta?.lanes[sel.lane]
    if (sel.lane === 'pictures' && isFilmRef(key.ref)) { label = `Film ${String(key.ref).slice(FILM_REF.length)}`; doc = 'The film morphs into this film\'s frame at In, plays it to Out at its speed, and morphs on from there.' }
    else if (sel.lane === 'pictures' && isClipRef(key.ref)) { label = `Clip ${film.meta?.pictures[sel.i]?.name ?? basename(String(key.ref))}`; doc = 'The film morphs into this clip\'s frame at In, plays it to Out at its speed with its sound, and morphs on from there. In and Out are seconds of the clip\'s file.' }
    else if (sel.lane === 'pictures') { label = `Picture ${sel.i + 1}: ${basename(String(key.ref))}`; doc = 'When the film passes this picture.' }
    else if (sel.lane === 'looks') { const mm = key.settings?.method || 'none'; label = `Look ${sel.i + 1}: ${mm === 'none' ? 'morph only' : mm}`; doc = 'From here the film moves toward the next look.' }
    else { label = `${m?.label ?? sel.lane}, key ${sel.i + 1}`; doc = m ? `${m.doc ?? ''}. From ${m.lo} to ${m.hi}.` : '' }
    return { key, label, doc, hasV: 'v' in key, isFilm: sel.lane === 'pictures' && playsRef(key.ref), isClip: sel.lane === 'pictures' && isClipRef(key.ref) }
  })
  function editKey(patch: { t?: number; v?: number; ease?: string; in?: number; out?: number; speed?: number; mute?: boolean; gain?: number }) {
    const sel = film.sel
    if (!sel) return
    const s = clone(film.spec)
    const keys = keysOf(s, sel.lane)
    if (!keys || !keys[sel.i]) return
    const key = keys[sel.i]
    if (patch.t !== undefined) key.t = Math.round(Math.max(0, patch.t) * 1000) / 1000
    if (patch.v !== undefined && 'v' in key) key.v = patch.v
    if (patch.ease) key.ease = patch.ease
    for (const k of ['in', 'out', 'speed'] as const) if (patch[k] !== undefined) key[k] = Math.round(patch[k]! * 1000) / 1000
    if (patch.mute !== undefined) key.mute = patch.mute
    if (patch.gain !== undefined) key.gain = patch.gain
    commit(s)
  }
  function key(e: KeyboardEvent) {
    if (!active) return
    const t = e.target as HTMLElement | null
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return
    if ((e.key === 'Delete' || e.key === 'Backspace') && film.sel) { removeSelected(); e.preventDefault() }
    else if (e.key === ' ') { toggle(); e.preventDefault() }
  }

  // ---- kept films, templates and rendering ----
  let films = $state<{ value: string; label: string }[]>([])
  let pick = $state<string | null>(null)
  let templates = $state<{ value: string; label: string }[]>([])
  let templatePick = $state<string | null>(null)
  let templateName = $state('')
  let templateStatus = $state('')
  let draw = $state(768), target = $state(0)
  let job = $state<Job | null>(null)
  let video = $state<string | null>(null)
  let renderStatus = $state('')

  async function listFilms() {
    try { films = (await rpc('alchemy film list')).result.films.map((f: any) => ({ value: f.name, label: `${f.name} · ${f.pictures.length} pictures` })) } catch { films = [] }
    try { templates = (await rpc('alchemy template list')).result.templates.map((t: any) => ({ value: t.name, label: `${t.name} · ${t.lanes.length} lane${t.lanes.length === 1 ? '' : 's'}, ${t.looks} look${t.looks === 1 ? '' : 's'}` })) } catch { templates = [] }
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
  async function saveTemplate() {
    const name = templateName.trim() || film.spec.name
    if (!name) { templateStatus = 'Name the template first.'; return }
    try {
      const r = await rpc('alchemy template save', { name, spec: film.spec })
      const t = r.result.template
      templateStatus = `Kept the template ${t.name}, ${t.lanes.length} lane${t.lanes.length === 1 ? '' : 's'} and ${t.looks} look${t.looks === 1 ? '' : 's'} keyed over ${t.length.toFixed(1)} s.`
      templatePick = t.name
      listFilms()
    } catch (e) { templateStatus = reason(e) }
  }
  async function applyTemplate() {
    if (!templatePick) { templateStatus = 'Choose a kept template first.'; return }
    try {
      const r = await rpc('alchemy template apply', { template: templatePick, spec: film.spec })
      commit(r.result.spec, null)
      templateStatus = `Applied ${templatePick}. Keep this film to save it with the template's keys.`
    } catch (e) { templateStatus = reason(e) }
  }
  async function deleteTemplate() {
    if (!templatePick) { templateStatus = 'Choose a kept template first.'; return }
    try { await rpc('alchemy template delete', { template: templatePick }); templateStatus = `Deleted the template ${templatePick}.`; templatePick = null; listFilms() } catch (e) { templateStatus = reason(e) }
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

  function dragPicture(e: DragEvent, p: string) {
    e.dataTransfer?.setData(PATH_TYPE, p)
    if (e.dataTransfer) e.dataTransfer.effectAllowed = 'copy'
  }

  onMount(() => { refreshMeta(); listFilms(); return onStateChange(listFilms) })
</script>

<svelte:window onkeydown={key} />

<div class="filmtab">
  <Card title="Pictures" subtitle="Drag a picture or a clip onto the timeline, or click it to put it at the playhead. A clip plays there with its sound. Drag it along the timeline to change when the film passes it." pad="sm" scroll style="flex: 0 0 220px; max-height: 100%">
    <div class="cast">
      {#if !items.length}<p class="hint">The active collection is empty. Pictures added with + Collection on any tab appear here.</p>{/if}
      {#each items as p (p)}
        <button class="pic" title={basename(p)} draggable="true" ondragstart={(e) => dragPicture(e, p)} onclick={() => addPicture(p)}><img src={thumbUrl(p)} alt="" loading="lazy" draggable="false" /></button>
      {/each}
    </div>
    {#if films.length}
      <p class="hint shelf">Kept films. One placed on the timeline plays there, and this film morphs into it and out of it.</p>
      <div class="films">
        {#each films.filter((f) => f.value !== film.spec.name) as f (f.value)}
          <button class="filmkey" title="Put this film at the playhead, or drag it onto the timeline" draggable="true" ondragstart={(e) => dragPicture(e, FILM_REF + f.value)} onclick={() => addPicture(FILM_REF + f.value)}>▶ {f.value}</button>
        {/each}
      </div>
    {/if}
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
    <Disclosure label="Look" note="method, materials, mask" open>
      {#if look.index === null}
        <p class="txt">No look yet: the film is the morph alone, with the finish lanes on top.</p>
        <p class="hint">To mix materials or use a mask, set a step in the Image subtab (method, materials, mask), then press Add Image's step here.</p>
      {:else}
        <p class="txt"><b>Look {look.index + 1} of {look.count}</b>, from {look.t?.toFixed(2)} s</p>
        <p class="txt">Method: {look.method}{look.model ? `, ${look.model}` : ''}</p>
        <p class="txt">{look.mask ? `Mask: ${look.mask}` : 'Mask: none, so the step covers the whole frame'}</p>
        {#if look.materials?.length}<p class="txt row">Materials: {#each look.materials as m}<img class="mat" src={thumbUrl(m)} alt="" title={basename(m)} />{/each}</p>{/if}
        {#if look.finish?.length}<p class="txt">Finish: {look.finish.join(', ')}</p>{/if}
        <p class="hint">Lanes move these numbers over time. To change the method, materials or mask, press Edit in Image, change them there, and come back to Use Image's step.</p>
      {/if}
      <div class="row">
        <Button size="sm" title="Open this look in the Image subtab, to change its method, materials or mask" onclick={editLook} disabled={look.index === null}>Edit in Image</Button>
        <Button size="sm" title="Replace this look with the step set in the Image subtab now" onclick={async () => (lookStatus = await replaceLook(controls()))} disabled={look.index === null}>Use Image's step</Button>
      </div>
      <div class="row">
        <Button size="sm" title="A new look at the playhead, from the Image subtab's step" onclick={async () => (lookStatus = await addLook(controls()))}>Add Image's step here</Button>
        <Button size="sm" kind="ghost" onclick={async () => (lookStatus = await removeLook())} disabled={look.index === null}>Remove look</Button>
      </div>
      <Status text={lookStatus} size="xs" />
    </Disclosure>

    <Disclosure label="Lanes" note="what moves over time" open>
      <p class="hint">A lane moves one number of the look over the film. Pick what to animate, add it, then set its keys on the timeline.</p>
      <div class="row nowrap">
        <div class="grow"><Select bind:value={lanePick} options={film.meta?.laneOptions ?? []} placeholder="what to animate…" size="sm" /></div>
        <Button size="sm" disabled={!lanePick} onclick={() => { if (lanePick) addLane(lanePick); lanePick = null }}>Add lane</Button>
      </div>
      <Status text={laneDoc} size="xs" />
      {#if selLane}
        <p class="txt">Shape {film.meta?.lanes[selLane]?.label ?? selLane}</p>
        <div class="row">
          <Field label="From"><NumberInput bind:value={shapeLo} step="any" size="sm" width="70px" /></Field>
          <Field label="To"><NumberInput bind:value={shapeHi} step="any" size="sm" width="70px" /></Field>
          <Field label="Times"><NumberInput bind:value={shapeCount} min={1} max={32} step={1} size="sm" width="56px" /></Field>
        </div>
        <Field label="Over"><Select bind:value={shapeOver} options={[{ value: 'film', label: 'the whole film' }, { value: 'after', label: 'the playhead to the end' }]} size="sm" /></Field>
        <div class="eases">{#each LANE_SHAPES as sh}<button class="ease" title={sh.doc} onclick={() => shape(sh.key)}>{sh.label}</button>{/each}</div>
        <Status text={shapeStatus} size="xs" />
      {:else if Object.keys(film.spec.lanes).length}
        <p class="hint">Click a lane's curve to select it, then shape it whole here: a rise, a swell, a wave.</p>
      {/if}
      {#if look.index !== null && !look.mask}<p class="hint">The look's mask: set one in the Image subtab, then Use Image's step, and its layers become lanes.</p>{/if}
    </Disclosure>

    <Disclosure label="Key" note="time, value, curve" open>
      {#if selected}
        <p class="txt">{selected.label}</p>
        <p class="hint">{selected.doc}</p>
        <div class="row">
          <Field label="Time (s)"><NumberInput value={Math.round(selected.key.t * 1000) / 1000} min={0} step={0.05} size="sm" width="80px" onchange={(v) => editKey({ t: v })} /></Field>
          {#if selected.hasV}<Field label="Value"><NumberInput value={selected.key.v ?? 0} step="any" size="sm" width="90px" onchange={(v) => editKey({ v })} /></Field>{/if}
        </div>
        {#if selected.isFilm}
          <div class="row">
            <Field label="In (s)"><NumberInput value={selected.key.in ?? 0} min={0} step={0.1} size="sm" width="70px" onchange={(v) => editKey({ in: v })} /></Field>
            <Field label="Out (s)"><NumberInput value={selected.key.out ?? 0} min={0} step={0.1} size="sm" width="70px" onchange={(v) => editKey({ out: v })} /></Field>
            <Field label="Speed"><NumberInput value={selected.key.speed ?? 1} min={0.1} max={8} step={0.1} size="sm" width="60px" onchange={(v) => editKey({ speed: v })} /></Field>
          </div>
        {/if}
        {#if selected.isClip}
          <div class="row">
            <Toggle checked={!selected.key.mute} label="Sound" size="sm" onchange={(v) => editKey({ mute: !v })} />
            {#if !selected.key.mute}<Field label="Gain (dB)"><NumberInput value={selected.key.gain ?? 0} min={-60} max={24} step={1} size="sm" width="70px" onchange={(v) => editKey({ gain: v })} /></Field>{/if}
          </div>
        {/if}
        <Field label="Curve to the next key">
          <div class="eases">
            {#each EASES as [v, lab]}<button class="ease" class:on={(selected.key.ease ?? 'linear') === v} onclick={() => editKey({ ease: v })}>{lab}</button>{/each}
          </div>
        </Field>
        <div><Button size="sm" kind="ghost" onclick={removeSelected}>Remove key</Button></div>
      {:else}
        <p class="txt">No key selected.</p>
        <p class="hint">Click a key on the timeline to edit it here. To add one, double-click a lane, or press ◆ on a lane to key it at the playhead.</p>
      {/if}
    </Disclosure>

    <Disclosure label="Derivatives" note="one picture, many versions">
      <p class="hint">Versions of one picture, drawn with the Image subtab's step and laid after it on the picture lane, so the film morphs between them. Keep the result as a film, and it plays inside another film from the shelf on the left.</p>
      {#if derivePic}
        <p class="txt row"><img class="mat" src={thumbUrl(String(derivePic.ref))} alt="" /> {basename(String(derivePic.ref))}, at {derivePic.t.toFixed(2)} s</p>
      {:else}
        <p class="hint">Click a picture on the picture lane to derive from it.</p>
      {/if}
      <p class="hint">The step: {stepMethod?.label ?? step.method}{step.roles['1'] ? ', with material [1]' : ''}.</p>
      <Field label="What changes"><Select bind:value={vary} options={VARY} size="sm" /></Field>
      {#if vary === 'seed' || vary === 'knob'}
        <Field label="How many"><NumberInput bind:value={deriveCount} min={2} max={12} step={1} size="sm" width="60px" /></Field>
      {/if}
      {#if vary === 'knob'}
        <Field label="Knob"><Select bind:value={deriveKnob} options={numberKnobs.map((k) => ({ value: k.key, label: `${k.label} (${k.lo} to ${k.hi})` }))} placeholder="a knob of the step…" size="sm" /></Field>
        <div class="row">
          <Field label="From"><NumberInput bind:value={deriveStart} step="any" size="sm" width="70px" /></Field>
          <Field label="To"><NumberInput bind:value={deriveEnd} step="any" size="sm" width="70px" /></Field>
        </div>
      {:else if vary === 'material'}
        <p class="hint">{otherPictures.length} pictures of the collection, each laid in as material [1] in turn.</p>
      {:else if vary === 'recipe'}
        <div class="eases">{#each recipeNames as r}<button class="ease" class:on={deriveRecipes.includes(r)} onclick={() => (deriveRecipes = deriveRecipes.includes(r) ? deriveRecipes.filter((x) => x !== r) : [...deriveRecipes, r])}>{r}</button>{/each}</div>
      {/if}
      <div class="row">
        <Field label="Seconds apart"><NumberInput bind:value={deriveEvery} min={0.5} max={30} step={0.5} size="sm" width="60px" /></Field>
        <Checkbox bind:checked={deriveBack} label="end on the picture again" />
      </div>
      <Checkbox bind:checked={deriveSteady} label="then a steady pace through the whole film" />
      <Button size="sm" kind="primary" onclick={derive} disabled={!derivePic || deriveBusy}>Make derivatives</Button>
      {#if deriveBusy}<ProgressBar fraction={deriveJob?.fraction ?? 0} />{/if}
      <Status text={deriveStatus} size="xs" tone={deriveJob?.status === 'failed' ? 'bad' : 'dim'} />
    </Disclosure>

    <Disclosure label="Pictures timing" note="a steady morph, then speed" open>
      <p class="hint">Re-time every picture between the first and the last. Steady gives each passage as long as the picture changes along it, so the morph runs at one rate however far apart the pictures are. The other shapes bend that rate. Held films keep their length.</p>
      <Checkbox bind:checked={byChange} label="by how much the picture changes (off: every passage as long)" />
      <div class="row">{#each [['even', 'steady', 'one rate all through'], ['faster', 'speeding up', 'the rate rising toward the end'], ['slower', 'slowing down', 'the rate falling toward the end'], ['smooth', 'slow at both ends', 'slow to start and to end']] as [k, lab, doc]}<Button size="sm" title={doc} onclick={async () => { timingStatus = 'Measuring the passages…'; timingStatus = (await spacePictures(k, byChange ? 'change' : 'time')) || (byChange ? `Re-timed: ${lab}, by how much the picture changes.` : `Re-timed: ${lab}, every passage as long.`) }}>{lab}</Button>{/each}</div>
      <Status text={timingStatus} size="xs" />
    </Disclosure>

    <Disclosure label="Film" note="length, shape, model">
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
      {#if film.spec.pictures.some((p) => isClipRef(p.ref))}
        <div class="row" title="How a look's method meets a playing clip. Carried runs it on frames of the clip a few times a second and moves each result along the clip's movement to the frames between, which keeps a method that draws from noise steady. Automatic carries the generative methods and runs the others on every frame.">
          <Field label="Looks on clips"><Select value={film.spec.temporal ?? 'auto'} options={[{ value: 'auto', label: 'automatic' }, { value: 'frame', label: 'on every frame' }, { value: 'keys', label: 'on key frames, carried' }]} size="sm" onchange={(v) => setField('temporal', String(v))} /></Field>
          {#if (film.spec.temporal ?? 'auto') !== 'frame'}<Field label="Key every (s)"><NumberInput value={film.spec.key_every ?? 0.5} min={0.1} max={5} step={0.1} size="sm" width="64px" onchange={(v) => setField('key_every', v)} /></Field>{/if}
          <Checkbox checked={film.spec.sound ?? true} label="sound" onchange={(v) => setField('sound', v)} />
        </div>
        {#if film.spec.looks.some((l) => String(l.settings?.method ?? '').startsWith('generative.'))}
          <p class="hint">Diffusion methods stay off clips for now. While a clip plays, a look with one shows the clip with the look's finish alone.</p>
        {/if}
      {/if}
    </Disclosure>

    <Disclosure label="Templates" note="save, apply">
      <p class="hint">A template keeps this film's settings, looks and lanes without its pictures. Applied to another film, its keys stretch to that film's length.</p>
      <div class="row nowrap"><div class="grow"><TextInput bind:value={templateName} placeholder="template name" size="sm" onenter={saveTemplate} /></div><Button size="sm" onclick={saveTemplate}>Save as template</Button></div>
      <div class="row nowrap">
        <div class="grow"><Select bind:value={templatePick} options={templates} placeholder="Kept templates…" size="sm" /></div>
        <Button size="sm" onclick={applyTemplate} disabled={!templatePick}>Apply</Button>
        <Button size="sm" kind="ghost" onclick={deleteTemplate} disabled={!templatePick}>Delete</Button>
      </div>
      <Status text={templateStatus} size="xs" />
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
  .txt { font-size: var(--fs-sm); }
  .mat { width: 28px; height: 28px; object-fit: cover; border-radius: 3px; }
  .eases { display: flex; flex-wrap: wrap; gap: 4px; }
  .ease { padding: 3px 8px; border: 1px solid var(--arc-line); border-radius: 5px; background: transparent; font-size: 11.5px; color: var(--arc-ink-dim); }
  .ease.on { border-color: var(--arc-accent); color: var(--arc-ink); background: color-mix(in srgb, var(--arc-accent) 18%, transparent); }
  .video { width: 100%; margin-top: 8px; border-radius: var(--r-md); }
  .shelf { font-size: var(--fs-xs); margin-top: 8px; }
  .films { display: flex; flex-direction: column; gap: 4px; padding: 4px 2px; }
  .filmkey { text-align: left; padding: 4px 8px; border: 1px solid var(--arc-line); border-radius: 5px; background: var(--arc-surface-2); color: var(--arc-ink); font-size: 11.5px; cursor: copy; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .filmkey:hover { border-color: var(--tab-alchemy); }
</style>
