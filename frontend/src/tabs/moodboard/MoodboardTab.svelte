<script lang="ts">
  import { app, current } from '../../lib/stores/app.svelte'
  import { get, post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { previewUrl, paletteUrl, basename } from '../../lib/api/media'
  import { roles, forceReference } from './moodboard.svelte'
  import Rail from './Rail.svelte'
  import MediaCard from '../../lib/components/MediaCard.svelte'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import Toggle from '../../lib/ui/Toggle.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Empty from '../../lib/ui/Empty.svelte'
  import SplitPane from '../../lib/ui/SplitPane.svelte'
  import Chip from '../../lib/ui/Chip.svelte'

  /**
   * Three columns: the collection rail, the bench (the similarity search and
   * the two role pictures), and the stage with the results.
   */
  type Group = { gid: string; keys: number[]; paths: string[]; marks: any[]; scores: number[] }

  let features = $state({ palette: false, style: false })
  let usePalette = $state(true), useStyle = $state(false)
  let paletteMethod = $state<'emd' | 'histogram' | 'moments'>('emd')
  let styleMethod = $state<'gram' | 'edge' | 'lbp'>('gram')
  let nColors = $state(16)
  let prompt = $state('')
  let k = $state(50)
  let showPalettes = $state(true)
  let imgSize = $state<'small' | 'medium' | 'large' | 'full'>('medium')
  let columns = $state(2)
  let groupTwins = $state(true), thresh = $state(0.08)
  let searching = $state(false)
  let groups = $state<Group[]>([])
  let message = $state('')
  let frames = $state<Record<string, number>>({})
  let selected = $state<Record<string, boolean>>({})
  let saveFolder = $state('')
  let saveStatus = $state(''), saveTone = $state<'ok' | 'warn' | 'bad' | 'dim'>('dim')
  let db = $state('')
  let checkedFor = ''

  $effect(() => {
    const ds = current.dataset
    if (!ds || ds.value === checkedFor) return
    checkedFor = ds.value
    get('/api/features', { name: ds.name }).then((f) => {
      features = f
      usePalette = f.palette
      useStyle = false
    }).catch(() => {})
  })

  const WIDTH = { small: 300, medium: 600, large: 900, full: 1400 }
  const isImage = $derived(current.modality === 'image')

  async function search(ref = roles.reference) {
    if (!ref) { toast('Choose a reference picture [R] in the collection first.', 'warn'); return }
    if (!app.dataset) { toast('Choose a dataset above to begin.', 'warn'); return }
    searching = true; message = ''
    try {
      const r = await post('/api/similar', { dataset: app.dataset, reference: ref, palette: usePalette, style: useStyle,
        paletteMethod, styleMethod, nColors, prompt, k, group: groupTwins, thresh })
      groups = r.groups; message = r.message || ''; db = r.db || ''
      frames = Object.fromEntries(r.groups.map((g: Group) => [g.gid, 0]))
      selected = Object.fromEntries(r.groups.map((g: Group) => [g.gid, false]))
    } catch (e) {
      groups = []; message = reason(e)
    } finally { searching = false }
  }

  function step(path: string) { forceReference(path); search(path) }
  function selectAll(on: boolean) { const s: Record<string, boolean> = {}; for (const g of groups) s[g.gid] = on; selected = s }
  const nSelected = $derived(Object.values(selected).filter(Boolean).length)

  async function saveSelected() {
    const paths = groups.filter((g) => selected[g.gid]).map((g) => g.paths[frames[g.gid] ?? 0])
    if (!saveFolder.trim()) { saveStatus = 'Please enter a folder name'; saveTone = 'warn'; return }
    if (!paths.length) { saveStatus = 'No images selected'; saveTone = 'warn'; return }
    try {
      const r = await post('/api/rpc', { op: 'item export', args: { items: paths, folder: saveFolder.trim(), kind: 'search' } })
      saveStatus = `Saved ${r.result.copied} images to ${saveFolder.trim()}`; saveTone = 'ok'
    } catch (e) { saveStatus = reason(e); saveTone = 'warn' }
  }
</script>

<SplitPane storeKey="moodboard-rail" initial={380} min={200} max={640}>
  {#snippet a()}
    <Rail />
  {/snippet}
  {#snippet b()}
    <SplitPane storeKey="moodboard-bench" initial={400} min={300} max={720}>
      {#snippet a()}
        <div class="bench">
          <Card title="Similarity search" subtitle="Uses the [R] picture in the collection as the reference.">
            {#if !isImage}<Status text="Similarity search is only available for image datasets." tone="warn" />{/if}
            <div class="feats">
              <div class="feat" class:off={!features.palette}>
                <div class="row"><Checkbox bind:checked={usePalette} label="Colour palette" disabled={!features.palette} />{#if !features.palette}<Chip tone="warn" small>no features</Chip>{/if}</div>
                <Field label="Method"><Select bind:value={paletteMethod} options={[{ value: 'emd', label: 'EMD' }, { value: 'histogram', label: 'Histogram' }, { value: 'moments', label: 'Moments' }]} size="sm" /></Field>
                <Field label="Colours"><Select bind:value={nColors} options={[4, 8, 12, 16, 24, 32].map((n) => ({ value: n, label: String(n) }))} size="sm" /></Field>
              </div>
              <div class="feat" class:off={!features.style}>
                <div class="row"><Checkbox bind:checked={useStyle} label="Style / texture" disabled={!features.style} />{#if !features.style}<Chip tone="warn" small>no features</Chip>{/if}</div>
                <Field label="Method"><Select bind:value={styleMethod} options={[{ value: 'gram', label: 'Gram' }, { value: 'edge', label: 'Edge' }, { value: 'lbp', label: 'LBP' }]} size="sm" /></Field>
              </div>
            </div>
            <div class="row nowrap">
              <div class="grow"><TextInput bind:value={prompt} placeholder="Filter by prompt (optional)…" onenter={() => search()} /></div>
              <NumberInput bind:value={k} min={1} max={500} suffix="results" />
            </div>
            <div class="row">
              <Button kind="primary" onclick={() => search()} busy={searching} disabled={!roles.reference || !isImage || (!usePalette && !useStyle)}>Find similar</Button>
              {#if !roles.reference}<span class="hint">Mark a picture [R] in the collection first.</span>{/if}
            </div>
            <Disclosure label="Refine" level={2} note="twins, layout">
              <div class="row">
                <Toggle bind:checked={groupTwins} label="Group twins" size="sm" />
                <span class="caps">distance ≤</span><NumberInput bind:value={thresh} min={0} max={0.5} step={0.01} size="sm" width="60px" />
              </div>
              <div class="row">
                <Checkbox bind:checked={showPalettes} label="Show palettes" />
                <Field label="Image size" inline><Select bind:value={imgSize} options={[{ value: 'small', label: 'Small' }, { value: 'medium', label: 'Medium' }, { value: 'large', label: 'Large' }, { value: 'full', label: 'Full' }]} size="sm" /></Field>
                <Field label="Columns" inline><Select bind:value={columns} options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))} size="sm" /></Field>
              </div>
            </Disclosure>
          </Card>

          {#if roles.reference || roles.target}
            <Card title="Roles" level={1} accent="var(--arc-accent)">
              <div class="roles">
                <div class="role">
                  <span class="caps ref">Reference</span>
                  {#if roles.reference}
                    <img class="pic ref" src={previewUrl(roles.reference, 300)} alt="" />
                    <img class="sw" src={paletteUrl(roles.reference, nColors, 200, 16)} alt="" />
                    <span class="fn faint ellipsis">{basename(roles.reference)}</span>
                  {:else}<div class="none">Not selected</div>{/if}
                </div>
                <div class="role">
                  <span class="caps tgt">Target</span>
                  {#if roles.target}
                    <img class="pic tgt" src={previewUrl(roles.target, 300)} alt="" />
                    <img class="sw" src={paletteUrl(roles.target, nColors, 200, 16)} alt="" />
                    <span class="fn faint ellipsis">{basename(roles.target)}</span>
                  {:else}<div class="none">Not selected</div>{/if}
                </div>
              </div>
              <p class="hint">The target receives the reference's colours through the colour transfer op, or on the Alchemy tab. Both are saved with the board.</p>
            </Card>
          {/if}
        </div>
      {/snippet}
      {#snippet b()}
        <div class="stage">
          <div class="bar">
            <span class="caps">Results</span>
            {#if groups.length}<span class="dim small">{groups.length} · {nSelected} selected</span>{/if}
            <Button size="sm" onclick={() => selectAll(true)} disabled={!groups.length}>Select all</Button>
            <Button size="sm" onclick={() => selectAll(false)} disabled={!groups.length}>Clear</Button>
            <div class="grow"><TextInput bind:value={saveFolder} placeholder="Folder to save selected results…" size="sm" /></div>
            <Button size="sm" kind="success" onclick={saveSelected} disabled={!nSelected}>Save selected</Button>
          </div>
          <Status text={saveStatus} tone={saveTone} size="xs" />
          <div class="results">
            {#if message}<Status text={message} tone="warn" />{/if}
            {#if !groups.length && !message}
              <Empty title="Pictures like your reference appear here." hint="Pick [R] on a picture in the collection, tick palette or style, then Find similar. Step here on a result makes it the new reference." />
            {/if}
            <div class="grid" style="grid-template-columns: repeat({columns}, minmax(0, 1fr))">
              {#each groups as g, i (g.gid)}
                <MediaCard paths={g.paths} marks={g.marks} scores={g.scores} bind:selected={selected[g.gid]} bind:frame={frames[g.gid]}
                           title="#{i + 1}" width={WIDTH[imgSize]} sizes="(max-width: 900px) 90vw, 50vw" palette={showPalettes ? nColors : 0} {db}
                           onstep={g.paths.length === 1 ? step : undefined} />
              {/each}
            </div>
          </div>
        </div>
      {/snippet}
    </SplitPane>
  {/snippet}
</SplitPane>

<style>
  .bench { display: flex; flex-direction: column; gap: var(--sp-3); height: 100%; min-height: 0; overflow-y: auto; padding: 0 var(--sp-1); }
  .feats { display: flex; gap: 10px; flex-wrap: wrap; }
  .feat { flex: 1 1 0; min-width: 150px; display: flex; flex-direction: column; gap: 8px; padding: 10px; background: var(--arc-surface-2); border: 1px solid var(--arc-line); border-radius: var(--r-md); }
  .feat.off { opacity: 0.6; }
  .roles { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .role { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .ref { color: var(--arc-accent); } .tgt { color: var(--arc-target); }
  .pic { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: var(--r-md); border: 2px solid var(--arc-line); }
  .pic.ref { border-color: var(--arc-accent); } .pic.tgt { border-color: var(--arc-target); }
  .sw { width: 100%; height: 14px; border-radius: var(--r-sm); }
  .fn { font-size: var(--fs-2xs); }
  .none { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border: 1px dashed var(--arc-line); border-radius: var(--r-md); color: var(--arc-ink-faint); font-style: italic; font-size: var(--fs-sm); }
  .stage { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; padding-left: var(--sp-1); }
  .bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 8px 12px; background: var(--arc-surface); border-bottom: 2px solid var(--arc-accent); border-radius: var(--r-lg) var(--r-lg) 0 0; flex: 0 0 auto; }
  .small { font-size: var(--fs-xs); }
  .results { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; padding-right: 4px; }
  .grid { display: grid; gap: 12px; }
</style>
