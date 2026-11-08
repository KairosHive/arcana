<script lang="ts">
  import { app } from '../../lib/stores/app.svelte'
  import { post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import type { ModelRow } from './types'
  import Card from '../../lib/ui/Card.svelte'
  import Stepper from '../../lib/ui/Stepper.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Chip from '../../lib/ui/Chip.svelte'
  import Spinner from '../../lib/ui/Spinner.svelte'

  /**
   * Add a dataset, in five steps that each check the one before. The folder
   * scan answers steps two and three on the person's behalf wherever it can:
   * media type, name, a time estimate per encoder, the number of groups.
   */
  let { models, onstarted, busy = false }: { models: ModelRow[]; onstarted: (id: string) => void; busy?: boolean } = $props()

  type Scan = {
    folder: string; exists: boolean; message?: string; tone?: 'ok' | 'warn' | 'bad'; counts?: { image: number; audio: number }
    detected?: 'image' | 'audio'; modality?: 'image' | 'audio'; n?: number; suggestedName?: string; decodeMs?: number | null
    autoK?: number | null; folderWords?: string[]; models?: ModelChoice[]
  }
  type ModelChoice = { id: string; label: string; dim: number; download_mb: number; downloaded: boolean; estimate: string; summary: string; note: string; blurb: string; default: boolean }

  const STEPS = [
    { title: 'Point at a folder', hint: 'Everything inside, including subfolders' },
    { title: 'Check what it found', hint: 'Media type and name' },
    { title: 'Choose the quality', hint: 'Which encoder reads the files' },
    { title: 'Name the groups', hint: 'How many, and from which words' },
    { title: 'Extras, then go', hint: 'Palette, style, thumbnails' },
  ]

  let step = $state(0)
  let folder = $state('')
  let scan = $state<Scan | null>(null)
  let scanning = $state(false)
  let modality = $state<'image' | 'audio'>('image')
  let name = $state('')
  let lastSuggested = ''
  let model = $state('')
  let k = $state(0)
  let labels = $state<'builtin' | 'folders' | 'none'>('builtin')
  let dims = $state<2 | 3>(2)
  let features = $state<string[]>([])
  let status = $state(''), tone = $state<'dim' | 'ok' | 'warn' | 'bad'>('dim')
  let starting = $state(false)

  const found = $derived(!!scan?.exists && (scan?.n ?? 0) > 0)
  const done = $derived([found ? 0 : -1, found && name.trim() ? 1 : -1, found && model ? 2 : -1, found ? 3 : -1].filter((i) => i >= 0))
  const reachable = $derived(found ? [0, 1, 2, 3, 4] : [0])
  const choices = $derived<ModelChoice[]>(scan?.models?.length ? scan.models : models.filter((m) => m.modality === modality).map((m) => ({ ...m, estimate: '', summary: `${m.dim}-d · ${m.downloaded ? 'ready' : `${m.download_mb.toLocaleString()} MB download`}`, note: m.blurb, default: false })))
  const chosenModel = $derived(choices.find((m) => m.id === model))

  async function doScan(explicit = false) {
    const f = folder.trim()
    if (!f) { scan = null; return }
    scanning = true
    try {
      const s = await post<Scan>('/api/datasets/scan', { folder: f, modality: explicit ? modality : undefined })
      scan = s
      if (s.exists) {
        if (!explicit && s.detected) modality = s.detected
        if (s.suggestedName && (!name.trim() || name === lastSuggested)) { name = s.suggestedName; lastSuggested = s.suggestedName }
        const def = s.models?.find((m) => m.default) ?? s.models?.[0]
        if (!s.models?.some((m) => m.id === model) && def) model = def.id
        if ((s.n ?? 0) > 0 && step === 0) step = 1
      }
    } catch (e) {
      scan = { folder: f, exists: false, message: reason(e), tone: 'bad' }
    } finally {
      scanning = false
    }
  }

  async function browse() {
    try {
      const r = await post('/api/datasets/browse', { current: folder })
      if (r.folder) { folder = r.folder; await doScan() }
      else if (r.message) toast(r.message, r.tone === 'bad' ? 'bad' : 'warn')
    } catch (e) {
      toast(reason(e), 'bad')
    }
  }

  function toggleFeature(f: string, on: boolean) {
    features = on ? [...new Set([...features, f])] : features.filter((x) => x !== f)
  }

  const labelNote = $derived.by(() => {
    if (labels === 'none') return 'Groups still get found, they just stay numbered. Rework can name them later without re-encoding.'
    const bits: string[] = []
    const n = scan?.n ?? 0
    if (!k) bits.push(scan?.autoK ? `About ${scan.autoK} groups for ${n.toLocaleString()} files.` : 'The number of groups scales with the collection.')
    else bits.push(`Exactly ${k} groups.`)
    if (labels === 'folders') {
      const words = scan?.folderWords ?? []
      bits.push(words.length ? `Names chosen from your ${words.length} subfolders: ${words.slice(0, 6).join(', ')}${words.length > 6 ? '…' : ''}`
                             : "No subfolders with media in them, so Arcana's own word list will be used instead.")
    } else bits.push("Names chosen from Arcana's 100-word list.")
    return bits.join(' ')
  })

  async function start() {
    status = ''
    starting = true
    try {
      const r = await post('/api/datasets/start', { folder, name, modality, model, features, k, labels, dims })
      status = 'Started.'; tone = 'ok'
      if (r.job) onstarted(r.job)
    } catch (e) {
      status = reason(e); tone = 'bad'
    } finally {
      starting = false
    }
  }
</script>

<Card title="Add a dataset" subtitle="Arcana reads your files where they are. Nothing is copied, moved or uploaded. It builds a searchable index beside them and leaves the originals alone.">
  <div class="wizard">
    <aside class="rail">
      <Stepper steps={STEPS} bind:current={step} {done} {reachable} />
      {#if found && scan}
        <div class="summary">
          <Chip tone={modality === 'audio' ? 'audio' : 'accent'}>{(scan.n ?? 0).toLocaleString()} {modality === 'audio' ? 'sounds' : 'images'}</Chip>
          {#if name}<span class="ellipsis dim">{name}</span>{/if}
        </div>
      {/if}
    </aside>

    <div class="body">
      {#if step === 0}
        <h4 class="st">Point at a folder</h4>
        <p class="hint">Everything inside it, including subfolders, gets indexed.</p>
        <div class="row nowrap">
          <div class="grow"><TextInput bind:value={folder} placeholder="paste a path, or use Browse" oncommit={() => doScan()} /></div>
          <Button onclick={browse}>Browse…</Button>
          <Button kind="primary" onclick={() => doScan()} busy={scanning} disabled={!folder.trim()}>Scan</Button>
        </div>
        {#if scanning}<Spinner label="Looking through the folder…" />{:else if scan?.message}<Status text={scan.message} tone={scan.tone || 'dim'} />{/if}
        {#if found}<div class="next"><Button kind="primary" onclick={() => (step = 1)}>Next: check what it found</Button></div>{/if}

      {:else if step === 1}
        <h4 class="st">Check what it found</h4>
        <p class="hint">The media type is detected from the folder. Change it if the guess is wrong. The name is what you will pick from the Dataset menu later.</p>
        <div class="row">
          <Field label="Media type">
            <div class="seg">
              <button class:on={modality === 'image'} onclick={() => { modality = 'image'; doScan(true) }}>Images{#if scan?.counts} · {scan.counts.image.toLocaleString()}{/if}</button>
              <button class:on={modality === 'audio'} onclick={() => { modality = 'audio'; doScan(true) }}>Audio{#if scan?.counts} · {scan.counts.audio.toLocaleString()}{/if}</button>
            </div>
          </Field>
          <Field label="Name" grow><TextInput bind:value={name} placeholder="taken from the folder" /></Field>
        </div>
        {#if scan?.message}<Status text={scan.message} tone={scan.tone || 'dim'} />{/if}
        <div class="next"><Button kind="ghost" onclick={() => (step = 0)}>Back</Button><Button kind="primary" disabled={!found || !name.trim()} onclick={() => (step = 2)}>Next: choose the quality</Button></div>

      {:else if step === 2}
        <h4 class="st">Choose the quality</h4>
        <p class="hint">A stronger encoder understands your prompts better but takes longer to index. You can change your mind later by re-indexing.</p>
        <div class="choices">
          {#each choices as m (m.id)}
            <button class="choice" class:on={model === m.id} onclick={() => (model = m.id)}>
              <span class="c-head"><span class="c-label">{m.label}</span>{#if m.default}<Chip tone="accent" small>suggested</Chip>{/if}</span>
              <span class="c-sum dim">{m.summary}</span>
              <span class="c-note faint">{m.blurb}</span>
            </button>
          {/each}
        </div>
        {#if chosenModel}<Status text={chosenModel.note} tone="dim" />{/if}
        <p class="hw faint">{app.hardware.line}{#if app.hardware.note} {app.hardware.note}{/if}</p>
        <div class="next"><Button kind="ghost" onclick={() => (step = 1)}>Back</Button><Button kind="primary" disabled={!model} onclick={() => (step = 3)}>Next: name the groups</Button></div>

      {:else if step === 3}
        <h4 class="st">How to name the groups</h4>
        <p class="hint">Arcana clusters the collection and gives each group a name. Leave this alone and it picks both for you.</p>
        <div class="row">
          <Field label="How many groups">
            <Select bind:value={k} options={[{ value: 0, label: 'Automatic' }, { value: 6, label: '6 · broad strokes' }, { value: 12, label: '12' }, { value: 18, label: '18' }, { value: 24, label: '24 · fine grained' }]} width="190px" />
          </Field>
          <Field label="Names come from">
            <Select bind:value={labels} options={[{ value: 'builtin', label: "Arcana's word list" }, { value: 'folders', label: "This folder's subfolder names" }, { value: 'none', label: "Don't name them" }]} width="260px" />
          </Field>
          <Field label="Map">
            <Select bind:value={dims} options={[{ value: 2, label: '2-D' }, { value: 3, label: '3-D' }]} width="100px" />
          </Field>
        </div>
        <Status text={labelNote} tone="dim" />
        <div class="next"><Button kind="ghost" onclick={() => (step = 2)}>Back</Button><Button kind="primary" onclick={() => (step = 4)}>Next: extras</Button></div>

      {:else}
        <h4 class="st">Extras, then go</h4>
        <p class="hint">Palette and style power the moodboard's similarity search. They cost time now, but you can add them later from Rework without re-reading a single file, so leave them unticked if you are not sure.</p>
        <div class="row">
          <Checkbox checked={features.includes('palette')} label="Colour palette" onchange={(v) => toggleFeature('palette', v)} />
          <Checkbox checked={features.includes('style')} label="Style / texture" onchange={(v) => toggleFeature('style', v)} />
          <Checkbox checked={features.includes('thumbnails')} label="Thumbnails" onchange={(v) => toggleFeature('thumbnails', v)} />
        </div>
        <div class="recap">
          <span><b>{name || '—'}</b> · {modality} · {(scan?.n ?? 0).toLocaleString()} files</span>
          <span class="dim">{chosenModel?.label ?? model}{#if chosenModel?.estimate} · {chosenModel.estimate}{/if}</span>
          <span class="dim">{k ? `${k} groups` : 'automatic groups'} · {labels === 'folders' ? 'subfolder names' : labels === 'none' ? 'numbered' : "Arcana's words"} · {dims}-D map</span>
        </div>
        <div class="next">
          <Button kind="ghost" onclick={() => (step = 3)}>Back</Button>
          <Button kind="primary" size="lg" onclick={start} busy={starting} disabled={busy || !found || !name.trim()}>Start indexing</Button>
          <Status text={busy && !status ? 'Something is already running. Wait for it, or cancel it.' : status} tone={busy && !status ? 'warn' : tone} />
        </div>
      {/if}
    </div>
  </div>
</Card>

<style>
  .wizard { display: grid; grid-template-columns: 240px 1fr; gap: var(--sp-5); }
  .rail { display: flex; flex-direction: column; gap: var(--sp-3); border-right: 1px solid var(--arc-line); padding-right: var(--sp-4); }
  .summary { display: flex; align-items: center; gap: 8px; padding: 6px 10px; font-size: var(--fs-xs); }
  .body { display: flex; flex-direction: column; gap: var(--sp-3); min-width: 0; }
  .st { font-size: var(--fs-lg); font-weight: 700; }
  .next { display: flex; align-items: center; gap: var(--sp-2); margin-top: var(--sp-1); flex-wrap: wrap; }
  .seg { display: inline-flex; border: 1px solid var(--arc-line); border-radius: var(--r-md); overflow: hidden; }
  .seg button { height: var(--ctl-h); padding: 0 14px; background: var(--arc-surface-2); color: var(--arc-ink-dim); border: none; font-size: var(--fs-sm); font-weight: 600; }
  .seg button + button { border-left: 1px solid var(--arc-line); }
  .seg button.on { background: color-mix(in srgb, var(--arc-accent) 25%, var(--arc-surface-2)); color: var(--arc-ink); }
  .choices { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: var(--sp-2); }
  .choice { display: flex; flex-direction: column; gap: 4px; text-align: left; padding: 10px 12px; border-radius: var(--r-md); border: 1px solid var(--arc-line); background: var(--arc-surface-2); color: var(--arc-ink); }
  .choice:hover { border-color: var(--arc-line-strong); }
  .choice.on { border-color: var(--arc-accent); box-shadow: inset 0 0 0 1px var(--arc-accent); }
  .c-head { display: flex; align-items: center; gap: 8px; }
  .c-label { font-weight: 700; font-size: var(--fs-md); }
  .c-sum { font-size: var(--fs-xs); }
  .c-note { font-size: var(--fs-xs); line-height: 1.4; }
  .hw { font-size: var(--fs-xs); line-height: 1.5; max-width: 80ch; }
  .recap { display: flex; flex-direction: column; gap: 3px; padding: 10px 12px; border-radius: var(--r-md); background: var(--arc-surface-2); border: 1px solid var(--arc-line); font-size: var(--fs-sm); }
  @media (max-width: 820px) { .wizard { grid-template-columns: 1fr; } .rail { border-right: none; padding-right: 0; border-bottom: 1px solid var(--arc-line); padding-bottom: var(--sp-3); } }
</style>
