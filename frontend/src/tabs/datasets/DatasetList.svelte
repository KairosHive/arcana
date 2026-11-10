<script lang="ts">
  import { caps } from '../../lib/stores/caps.svelte'
  import { post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { refreshDatasets } from '../../lib/stores/app.svelte'
  import type { DatasetRow } from './types'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Chip from '../../lib/ui/Chip.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Spinner from '../../lib/ui/Spinner.svelte'

  let { rows, loading = false, busy = false, onstarted, onchanged }: {
    rows: DatasetRow[]; loading?: boolean; busy?: boolean; onstarted: (id: string) => void; onchanged: () => void
  } = $props()

  const TONE: Record<string, 'ok' | 'warn' | 'bad' | 'accent'> = { ready: 'ok', building: 'accent', unfinished: 'warn', 'files missing': 'warn', unreadable: 'bad' }
  let status = $state(''), tone = $state<'dim' | 'ok' | 'warn' | 'bad'>('dim')

  // Rework: a panel under the row, fitted to what the dataset already has.
  let rework = $state<DatasetRow | null>(null)
  let rk = $state(0), rvocab = $state<'builtin' | 'folders' | 'none'>('builtin'), rdims = $state(2), rfeat = $state<string[]>([])
  // Delete: ask first, and say exactly what goes.
  let pending = $state<DatasetRow | null>(null)
  let preview = $state<{ removed: string[]; bytes: number } | null>(null)

  function openRework(r: DatasetRow) {
    rework = r; rdims = r.dims[0] ?? 2; rfeat = []
  }

  async function action(r: DatasetRow, kind: 'extend' | 'finish' | 'marks', extra: Record<string, unknown> = {}, message = '') {
    status = ''
    try {
      const res = await post('/api/datasets/action', { action: kind, name: r.name, modality: r.modality, ...extra })
      status = message; tone = 'ok'
      if (res.job) onstarted(res.job)
      rework = null
    } catch (e) {
      status = reason(e); tone = 'bad'
    }
  }

  async function armDelete(r: DatasetRow) {
    pending = r; preview = null
    try {
      preview = await post('/api/datasets/delete', { name: r.name, modality: r.modality, dryRun: true })
    } catch (e) {
      status = `Could not inspect that dataset: ${reason(e)}`; tone = 'bad'
    }
  }

  async function doDelete() {
    if (!pending) return
    const r = pending
    try {
      const res = await post('/api/datasets/delete', { name: r.name, modality: r.modality })
      if (res.failed?.length) { status = `Removed ${res.removed.length} files; ${res.failed.length} could not be deleted (in use?).`; tone = 'warn' }
      else { status = `Removed “${r.name}” and freed ${(res.bytes / 1048576).toFixed(1)} MB.`; tone = 'ok' }
      pending = null
      onchanged()
      await refreshDatasets()
    } catch (e) {
      status = reason(e); tone = 'bad'
    }
  }

  function toggle(f: string, on: boolean) { rfeat = on ? [...new Set([...rfeat, f])] : rfeat.filter((x) => x !== f) }
</script>

<Card title="Your datasets" subtitle="Green means the dataset is finished and every file is still where the index expects it. Only those appear in the Dataset menu. The tags show what each one can do.">
  {#if loading}
    <Spinner label="Reading the datasets…" />
  {:else if !rows.length}
    <div class="empty">
      <p class="t">No datasets yet.</p>
      <p class="hint">Use the steps above to point Arcana at a folder of images or audio. Once it has indexed one, it appears here and in the Dataset menu at the top of every tab.</p>
    </div>
  {:else}
    <div class="list">
      {#each rows as r (r.name + r.modality)}
        <div class="drow" style="--tone: var(--arc-{TONE[r.status] === 'accent' ? 'accent' : TONE[r.status]})">
          <span class="stripe"></span>
          <span class="name">{r.name}</span>
          <Chip tone={r.modality === 'image' ? 'accent' : r.modality} small>{caps.label(r.modality)}</Chip>
          <span class="count dim mono">{r.total ? `${r.total.toLocaleString()} items` : '—'}</span>
          <span class="root faint ellipsis" title={r.root}>{r.root}</span>
          <span class="caps-row" title="Named clusters / colour-palette features / style features. Grey means the dataset was indexed without it. Rework adds it without re-encoding anything. 'palette · dated' means those features were built before a fix to the colour conversion: search still works, but much more coarsely. Rework rebuilds them properly.">
            <Chip tone={r.labels ? 'ok' : 'faint'} small>labels</Chip>
            <Chip tone={r.palette ? (r.paletteStale ? 'warn' : 'ok') : 'faint'} small>palette{r.paletteStale ? ' · dated' : ''}</Chip>
            <Chip tone={r.style ? 'ok' : 'faint'} small>style</Chip>
          </span>
          <Chip tone={TONE[r.status]} small>{r.status === 'building' ? 'building…' : r.status}</Chip>
          <span class="acts">
            <Button size="sm" disabled={!r.canRework || busy} title="Re-cluster {r.name}, rename its groups, lay it out in 3-D, or add palette and style, using the index it already has. Nothing is re-encoded." onclick={() => openRework(r)}>Rework</Button>
            <Button size="sm" disabled={!r.canExtend || busy} title="Look in {r.root || 'the dataset folder'} for files that were not there when {r.name} was indexed, and add them. Only the new files are encoded." onclick={() => action(r, 'extend', {}, `Checking ${r.name}'s folder for new files…`)}>Add new</Button>
            <Button size="sm" disabled={!r.canMarks || busy} title="Re-read ON1 stars, colour tags and capture dates. Seconds, and nothing is re-encoded." onclick={() => action(r, 'marks', {}, `Re-reading ON1 marks for ${r.name}…`)}>Refresh marks</Button>
            <Button size="sm" kind="danger" title="Delete the index for {r.name}. Your files stay where they are." onclick={() => armDelete(r)}>Remove</Button>
          </span>
        </div>
        {#if !r.hasMap && !r.building}
          <div class="sub">
            <span class="warn">Indexed, but the map was never built, probably an interrupted run. The {r.total.toLocaleString()} encoded items are safe; only the layout is missing.</span>
            <Button size="sm" kind="primary" disabled={busy} title="Lay out and save the dataset using the index that already exists. Nothing is re-encoded." onclick={() => action(r, 'finish', {}, `Laying out ${r.name} from the index it already has…`)}>Finish this</Button>
          </div>
        {/if}
        {#if rework && rework.name === r.name && rework.modality === r.modality}
          <div class="panel">
            <p class="pt">Reworking <b>{r.name}</b> from the index it already has. Nothing is re-encoded.</p>
            <div class="row">
              <Field label="How many groups"><Select bind:value={rk} options={[{ value: 0, label: 'Automatic' }, ...Array.from({ length: 19 }, (_, i) => 4 + 2 * i).map((n) => ({ value: n, label: String(n) }))]} width="150px" /></Field>
              <Field label="Names come from"><Select bind:value={rvocab} options={[{ value: 'builtin', label: "Arcana's word list" }, { value: 'folders', label: 'Your subfolder names' }, { value: 'none', label: 'Leave groups numbered' }]} width="210px" /></Field>
              <Field label="Map"><Select bind:value={rdims} options={[{ value: 2, label: '2-D' }, { value: 3, label: '3-D' }]} width="100px" /></Field>
            </div>
            <div class="row">
              <span class="caps">Also build</span>
              <Checkbox checked={rfeat.includes('palette')} label={'Colour palette' + (r.palette ? (r.paletteStale ? ' (rebuild, currently dated)' : ' (rebuild)') : '')} onchange={(v) => toggle('palette', v)} />
              <Checkbox checked={rfeat.includes('style')} label={'Style / texture' + (r.style ? ' (rebuild)' : '')} onchange={(v) => toggle('style', v)} />
              <Checkbox checked={rfeat.includes('thumbnails')} label="Thumbnails" onchange={(v) => toggle('thumbnails', v)} />
            </div>
            <div class="row">
              <Button size="sm" kind="primary" disabled={busy} onclick={() => action(r, 'finish', { rework: true, k: rk, labels: rvocab, dims: rdims, features: rfeat }, `Reworking ${r.name}…`)}>Rework</Button>
              <Button size="sm" onclick={() => (rework = null)}>Cancel</Button>
            </div>
          </div>
        {/if}
        {#if pending && pending.name === r.name && pending.modality === r.modality}
          <div class="confirm">
            <p class="pt">Remove “{r.name}”?</p>
            {#if preview}
              <p class="hint">Deletes {preview.removed.length} index and feature files, freeing {(preview.bytes / 1048576).toFixed(1)} MB. Your {r.modality} files are not touched. Arcana only ever read them where they are. Re-indexing the folder rebuilds this.</p>
            {:else}<Spinner label="Looking at what would go…" />{/if}
            <div class="row">
              <Button size="sm" kind="danger" onclick={doDelete}>Delete</Button>
              <Button size="sm" onclick={() => (pending = null)}>Cancel</Button>
            </div>
          </div>
        {/if}
      {/each}
    </div>
  {/if}
  <Status text={status} {tone} />
</Card>

<style>
  .empty { padding: 10px 0; display: flex; flex-direction: column; gap: 4px; }
  .empty .t { font-weight: 600; }
  .list { display: flex; flex-direction: column; }
  .drow { display: flex; align-items: center; gap: 12px; padding: 10px 0; flex-wrap: wrap; border-top: 1px solid var(--arc-line); }
  .drow:first-child { border-top: none; }
  .stripe { width: 3px; align-self: stretch; min-height: 22px; border-radius: 2px; background: var(--tone); }
  .name { font-weight: 600; min-width: 140px; }
  .count { font-size: var(--fs-xs); min-width: 80px; }
  .root { font-size: var(--fs-xs); flex: 1; min-width: 160px; }
  .caps-row { display: flex; gap: 5px; }
  .acts { display: flex; gap: 6px; flex-wrap: wrap; }
  .sub { display: flex; align-items: center; gap: 12px; padding: 0 0 10px 17px; flex-wrap: wrap; font-size: var(--fs-xs); line-height: 1.5; }
  .sub .warn { max-width: 62ch; }
  .panel, .confirm { display: flex; flex-direction: column; gap: 10px; padding: 12px 14px; margin: 4px 0 8px; border-radius: var(--r-md); background: var(--arc-surface-2); border: 1px solid var(--arc-line); }
  .confirm { border-color: color-mix(in srgb, var(--arc-bad) 35%, transparent); background: color-mix(in srgb, var(--arc-bad) 6%, var(--arc-surface)); }
  .pt { font-size: var(--fs-md); }
</style>
