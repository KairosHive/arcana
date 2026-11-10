<script lang="ts">
  import { onMount } from 'svelte'
  import { get, post, reason } from '../../lib/api/client'
  import { app } from '../../lib/stores/app.svelte'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { filmVideoUrl, basename, sectionLabel } from '../../lib/api/media'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import TextArea from '../../lib/ui/TextArea.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Thumb from '../../lib/ui/Thumb.svelte'
  import ModalityTag from '../../lib/components/ModalityTag.svelte'

  /**
   * The Pattern subtab: procedural editing. The collection's first items are
   * the sources A, B, C…, a preset or a hand-written rule places their frames,
   * and the render comes back with a flash meter reading.
   */
  const LETTERS = 'ABCDEFGH'
  let presets = $state<Record<string, any>>({})
  let preset = $state('')
  let rule = $state('{\n  "voices": [{"pulse": {"every": 1}, "cut": ["A", "B"]}]\n}')
  let seconds = $state(6), fps = $state(30), width = $state(640), height = $state(360), seed = $state(7)
  let status = $state(''), tone = $state<'dim' | 'ok' | 'warn' | 'bad'>('dim')
  let video = $state(''), busy = $state(false)
  let flash = $state<{ n: number; over: boolean; at: number | null } | null>(null)

  const sources = $derived(app.collection.items.slice(0, LETTERS.length))

  onMount(async () => {
    try { presets = (await post('/api/rpc', { op: 'pattern presets', args: {} })).result.presets } catch { /* the box still works */ }
  })

  function usePreset(name: string) {
    preset = name
    if (presets[name]) rule = JSON.stringify(presets[name], null, 2)
  }

  async function render() {
    let part: any
    try { part = JSON.parse(rule) } catch { status = 'The rule is JSON. Check its brackets and quotes.'; tone = 'bad'; return }
    if (sources.length < 1) { status = 'Add pictures or clips to the collection first. They become the sources A, B, C.'; tone = 'warn'; return }
    const spec = { fps, seconds, size: [width, height], seed, ...part,
      sources: { ...Object.fromEntries(sources.map((p, i) => [LETTERS[i], p])), K: '#000000', W: '#ffffff', ...(part.sources ?? {}) } }
    busy = true; status = 'Rendering…'; tone = 'dim'; flash = null
    try {
      const r = await post('/api/rpc', { op: 'pattern make', args: { spec, name: preset || 'pattern', background: true } })
      const id = r.result.job
      watchJob(id, (j) => { if (j.message) status = j.message }, (j) => {
        busy = false
        if (j.status === 'done' && j.result) {
          const res = j.result
          video = filmVideoUrl(res.path)
          flash = { n: res.flashes_per_second, over: res.over_flash_limit, at: res.flash_at }
          status = `Rendered ${res.frames} frames, ${res.seconds.toFixed(1)} s.`; tone = 'ok'
        } else { status = j.error || `The render ended: ${j.status}.`; tone = 'bad' }
      })
    } catch (e) { busy = false; status = reason(e); tone = 'bad' }
  }
</script>

<div class="pat">
  <Card title="Sources" subtitle="The first items of the collection, in order. K is black and W is white." pad="sm" scroll style="flex: 0 0 230px; max-height: 100%">
    {#snippet actions()}<ModalityTag modalities={['image', 'video']} />{/snippet}
    {#if !sources.length}<p class="hint">Search for pictures or clips and add them to the collection. They appear here as A, B, C.</p>{/if}
    <div class="srcs">
      {#each sources as p, i (p)}
        <div class="src"><span class="letter mono">{LETTERS[i]}</span><Thumb path={p} size="72px" />
          <span class="cap faint ellipsis">{basename(p)}{#if sectionLabel(p)}<br />{sectionLabel(p)}{/if}</span></div>
      {/each}
    </div>
  </Card>

  <Card title="Rule" subtitle="A rule places the frames. Voices fire on pulses and cut, flash or hide a source. Space interlaces two sources inside one frame." pad="sm" scroll style="flex: 1 1 380px; min-width: 300px; max-height: 100%">
    <div class="row">
      <Field label="Start from" inline><Select value={preset || null} placeholder="a preset…" options={Object.keys(presets).map((k) => ({ value: k, label: k }))} onchange={(v) => usePreset(String(v))} size="sm" /></Field>
    </div>
    <TextArea bind:value={rule} rows={12} />
    <p class="hint">Pulses: <span class="mono">{'{"every": 3}'}</span>, <span class="mono">{'{"every": 6.1}'}</span> (drifts), <span class="mono">{'{"on": 2, "off": 5}'}</span>, <span class="mono">{'{"euclid": [5, 8], "step": 4}'}</span>, <span class="mono">{'{"hits": "x..x.x"}'}</span>, <span class="mono">{'{"chance": 0.1}'}</span>. Actions: <span class="mono">"cut": ["A", "B"]</span>, <span class="mono">"flash": "B", "frames": 1</span>, <span class="mono">"pair": "B", "amount": 0.2</span>. Space: <span class="mono">{'{"kind": "lines", "size": 2, "shift": 1, "with": "B"}'}</span> (also columns, checker). Time: <span class="mono">{'{"B": "stepping"}'}</span> or freeze.</p>
    <div class="row">
      <Field label="Seconds" inline><NumberInput bind:value={seconds} min={1} max={600} size="sm" width="64px" /></Field>
      <Field label="fps" inline><NumberInput bind:value={fps} min={6} max={60} size="sm" width="56px" /></Field>
      <Field label="Width" inline><NumberInput bind:value={width} min={64} max={1920} step={2} size="sm" width="70px" /></Field>
      <Field label="Height" inline><NumberInput bind:value={height} min={64} max={1080} step={2} size="sm" width="70px" /></Field>
      <Field label="Seed" inline><NumberInput bind:value={seed} min={0} max={99999} size="sm" width="70px" /></Field>
    </div>
    <div class="row"><Button kind="primary" onclick={render} {busy} disabled={busy}>Render</Button>{#if status}<Status text={status} {tone} />{/if}</div>
  </Card>

  <Card title="Result" pad="sm" style="flex: 1 1 420px; min-width: 320px; max-height: 100%">
    {#if video}
      {#key video}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video src={video} controls loop playsinline></video>
      {/key}
    {:else}
      <p class="hint">The rendered pattern plays here. Every frame is a keyframe, so pausing lands on single frames.</p>
    {/if}
    {#if flash}
      <div class="meter" class:over={flash.over}>
        <b>Flash meter</b>
        {#if flash.over}
          up to {flash.n} flashes in one second, around {flash.at} s. More than three a second can trigger seizures in people with photosensitive epilepsy. Say so wherever this is shown.
        {:else}
          at most {flash.n} flashes in any one second.
        {/if}
      </div>
    {/if}
  </Card>
</div>

<style>
  .pat { display: flex; flex-wrap: wrap; gap: 12px; flex: 1; min-height: 0; min-width: 0; align-items: stretch; }
  .srcs { display: flex; flex-direction: column; gap: 8px; }
  .src { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .letter { flex: 0 0 18px; font-weight: 700; color: var(--arc-accent); }
  .cap { font-size: var(--fs-2xs); min-width: 0; }
  video { width: 100%; max-height: 60vh; background: #000; border-radius: var(--r-md); }
  .meter { font-size: var(--fs-xs); padding: 8px 10px; border-radius: var(--r-md); border: 1px solid var(--arc-line); color: var(--arc-ink-dim); }
  .meter.over { border-color: var(--arc-warn); color: var(--arc-warn); }
  .meter b { margin-right: 6px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
</style>
