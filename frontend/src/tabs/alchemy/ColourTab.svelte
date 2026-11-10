<script lang="ts">
  import { post, reason } from '../../lib/api/client'
  import { app } from '../../lib/stores/app.svelte'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { filmVideoUrl, videoUrl, basename, sectionLabel, isVideoPath } from '../../lib/api/media'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Slider from '../../lib/ui/Slider.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Thumb from '../../lib/ui/Thumb.svelte'
  import ModalityTag from '../../lib/components/ModalityTag.svelte'

  /**
   * The Colour subtab: a clip recoloured toward a picture, a clip or several in
   * order, with one table for the whole clip or tables followed over time. The
   * result plays beside the original with the measured frame-to-frame change,
   * and Steady removes flicker from it.
   */
  const METHODS = [
    { value: 'lab', label: 'Lab statistics' }, { value: 'mkl', label: 'MKL' },
    { value: 'idt', label: 'IDT' }, { value: 'modflows', label: 'ModFlows' },
  ]
  let target = $state('')
  let refs = $state<string[]>([])
  let method = $state('lab'), strength = $state(1), keys = $state(1), windowFrames = $state(9)
  let status = $state(''), tone = $state<'dim' | 'ok' | 'warn' | 'bad'>('dim')
  let busy = $state(false)
  let made = $state<{ path: string; from: string; before: number; after: number; cubes: string[]; steady?: boolean } | null>(null)

  const items = $derived(app.collection.items)
  const clips = $derived(items.filter(isVideoPath))
  // The target falls back to the first clip, and goes when the collection drops it.
  const chosen = $derived(target && items.includes(target) ? target : (clips[0] ?? ''))
  const references = $derived(refs.filter((r) => items.includes(r) && r !== chosen))
  const oneClip = $derived(references.length === 1 && isVideoPath(references[0]))

  function pick(p: string, e: MouseEvent) {
    if (e.shiftKey || !isVideoPath(p)) { toggleRef(p); return }
    target = p
    refs = refs.filter((r) => r !== p)
  }
  function toggleRef(p: string) {
    if (p === chosen) return
    refs = refs.includes(p) ? refs.filter((r) => r !== p) : [...refs, p].slice(0, 8)
  }

  function run(op: string, args: Record<string, any>, done: (r: any) => void) {
    busy = true; tone = 'dim'
    post('/api/rpc', { op, args: { ...args, background: true } }).then((r) => {
      watchJob(r.result.job, (j) => { if (j.message) status = j.message }, (j) => {
        busy = false
        if (j.status === 'done' && j.result) done(j.result)
        else { status = j.error || `Ended: ${j.status}.`; tone = 'bad' }
      })
    }).catch((e) => { busy = false; status = reason(e); tone = 'bad' })
  }

  function recolour() {
    if (!chosen) { status = 'Add a clip to the collection first. It is what gets recoloured.'; tone = 'warn'; return }
    if (!references.length) { status = 'Choose what the colours move toward: click a picture, or shift-click a clip.'; tone = 'warn'; return }
    status = 'Fitting the colour…'
    const from = chosen
    run('colour video', { target: from, reference: references, method, strength, keys: oneClip ? keys : 1 }, (r) => {
      made = { path: r.path, from, before: r.flicker_before, after: r.flicker_after, cubes: r.cubes ?? [] }
      status = `Recoloured ${r.frames} frames, ${r.seconds.toFixed(1)} s, with ${r.keys === 1 ? 'one table' : `${r.keys} tables followed over time`}.`; tone = 'ok'
    })
  }

  function steady() {
    if (!made) return
    status = 'Steadying…'
    const was = made
    run('colour deflicker', { target: was.path, source: was.from, window: windowFrames }, (r) => {
      made = { ...was, path: r.path, before: r.before, after: r.after, steady: true }
      status = `Steadied ${r.frames} frames.`; tone = 'ok'
    })
  }
</script>

<div class="col">
  <Card title="Clip and references" subtitle="Click a clip to recolour it. Click pictures, or shift-click clips, to choose what its colours move toward, in order." pad="sm" scroll style="flex: 0 0 250px; max-height: 100%">
    {#snippet actions()}<ModalityTag feature="alchemy.colour" />{/snippet}
    {#if !items.length}<p class="hint">Search for clips and pictures and add them to the collection. They appear here.</p>
    {:else if !clips.length}<p class="hint">The collection holds no clip yet. Add one from a video dataset.</p>{/if}
    <div class="srcs">
      {#each items as p (p)}
        {@const n = references.indexOf(p)}
        <button class="src" class:target={p === chosen} class:ref={n >= 0} onclick={(e) => pick(p, e)} title={p === chosen ? 'The clip to recolour' : n >= 0 ? `Reference ${n + 1}` : isVideoPath(p) ? 'Click to recolour this clip, shift-click to use it as a reference' : 'Click to use as a reference'}>
          <span class="badge mono">{p === chosen ? 'clip' : n >= 0 ? n + 1 : ''}</span>
          <Thumb path={p} size="64px" />
          <span class="cap faint ellipsis">{basename(p)}{#if sectionLabel(p)}<br />{sectionLabel(p)}{/if}</span>
        </button>
      {/each}
    </div>
  </Card>

  <Card title="Transfer" subtitle="The colour is fitted once on frames pooled across the clip and applied to every frame as one table, which keeps it steady." pad="sm" scroll style="flex: 0 1 330px; min-width: 260px; max-height: 100%">
    <div class="stack">
      <Field label="Method"><Select bind:value={method} options={METHODS} size="sm" /></Field>
      <Slider bind:value={strength} min={0} max={1} step={0.01} label="Strength" />
      {#if oneClip}
        <Field label="Follow the reference clip at" inline><NumberInput bind:value={keys} min={1} max={24} size="sm" width="60px" /> <span class="hint">moments (1 fits it as a whole)</span></Field>
      {:else if references.length > 1}
        <p class="hint">The clip passes through the {references.length} references in order, from its start to its end.</p>
      {/if}
      <div class="row"><Button kind="primary" onclick={recolour} {busy} disabled={busy || !chosen || !references.length}>Recolour the clip</Button></div>
      {#if made}
        <div class="row">
          <Button onclick={steady} disabled={busy}>Steady the result</Button>
          <Field label="over" inline><NumberInput bind:value={windowFrames} min={3} max={61} step={2} size="sm" width="56px" /> <span class="hint">frames</span></Field>
        </div>
        <p class="hint">Steady scales each frame to the slow course of the clip's colour and keeps the changes the original clip had.</p>
      {/if}
      {#if status}<Status text={status} {tone} />{/if}
    </div>
  </Card>

  <Card title="Result" pad="sm" scroll style="flex: 1 1 460px; min-width: 320px; max-height: 100%">
    {#if made}
      <div class="pair">
        <figure><figcaption class="caps">original</figcaption>
          {#key made.from}<!-- svelte-ignore a11y_media_has_caption -->
            <video src={videoUrl(made.from)} controls loop playsinline muted></video>{/key}</figure>
        <figure><figcaption class="caps">{made.steady ? 'recoloured and steadied' : 'recoloured'}</figcaption>
          {#key made.path}<!-- svelte-ignore a11y_media_has_caption -->
            <video src={filmVideoUrl(made.path)} controls loop playsinline></video>{/key}</figure>
      </div>
      <div class="meter" class:over={made.after > made.before * 1.5 + 0.2}>
        <b>Frame-to-frame colour change</b> {made.before.toFixed(2)} before, {made.after.toFixed(2)} after, in levels of 255.
        {#if made.after > made.before * 1.5 + 0.2} The result changes more between frames than the original. Steady it, or lower the strength.{/if}
      </div>
      {#each made.cubes as c (c)}<p class="hint mono ellipsis" title={c}>Table for an editor: {c}</p>{/each}
    {:else if chosen}
      <!-- svelte-ignore a11y_media_has_caption -->
      {#key chosen}<video src={videoUrl(chosen)} controls loop playsinline muted></video>{/key}
      <p class="hint">The clip to recolour. The result plays beside it, with its sound and the measured change between frames.</p>
    {:else}
      <p class="hint">The recoloured clip plays here beside its original.</p>
    {/if}
  </Card>
</div>

<style>
  .col { display: flex; flex-wrap: wrap; gap: 12px; flex: 1; min-height: 0; min-width: 0; align-items: stretch; }
  .srcs { display: flex; flex-direction: column; gap: 6px; }
  .src { display: flex; align-items: center; gap: 8px; min-width: 0; padding: 4px; text-align: left; background: none; border: 1px solid transparent; border-radius: var(--r-md); color: inherit; cursor: pointer; }
  .src:hover { border-color: var(--arc-line); }
  .src.target { border-color: var(--arc-accent); background: color-mix(in srgb, var(--arc-accent) 10%, transparent); }
  .src.ref { border-color: var(--arc-video); }
  .badge { flex: 0 0 28px; font-size: var(--fs-2xs); font-weight: 700; color: var(--arc-accent); }
  .cap { font-size: var(--fs-2xs); min-width: 0; }
  .stack { display: flex; flex-direction: column; gap: 12px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
  figure { margin: 0; display: flex; flex-direction: column; gap: 4px; min-width: 0; }
  video { width: 100%; max-height: 52vh; background: #000; border-radius: var(--r-md); }
  .meter { margin-top: 10px; font-size: var(--fs-xs); padding: 8px 10px; border-radius: var(--r-md); border: 1px solid var(--arc-line); color: var(--arc-ink-dim); }
  .meter.over { border-color: var(--arc-warn); color: var(--arc-warn); }
  .meter b { margin-right: 6px; }
</style>
