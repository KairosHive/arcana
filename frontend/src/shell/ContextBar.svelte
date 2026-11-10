<script lang="ts">
  import { app, setDataset, useCollection, newCollection, foldHeader } from '../lib/stores/app.svelte'
  import { reason } from '../lib/api/client'
  import { toast } from '../lib/stores/toast.svelte'
  import Select from '../lib/ui/Select.svelte'
  import TextInput from '../lib/ui/TextInput.svelte'
  import Chip from '../lib/ui/Chip.svelte'

  let newName = $state('')
  const options = $derived(app.datasets.map((d) => ({ value: d.value, label: d.label })))
  // Collections (moodboards) first; sets, gathered by searches for Alchemy's labels and LoRAs, after them.
  const boardOptions = $derived([
    ...app.boards.filter((b) => b.kind !== 'set').map((b) => ({ value: b.slug, label: `${b.name}  ·  ${b.count} picture${b.count === 1 ? '' : 's'}`, group: 'Collections' })),
    ...app.boards.filter((b) => b.kind === 'set').map((b) => ({ value: b.slug, label: `${b.name}  ·  ${b.count} picture${b.count === 1 ? '' : 's'}`, group: 'Sets' })),
  ])
  const chosen = $derived(app.datasets.find((d) => d.value === app.dataset))

  async function create() {
    const name = newName.trim()
    if (!name) return
    try {
      const c = await newCollection(name)
      toast(`“${c.name}” is the active collection.`, 'ok')
      newName = ''
    } catch (e) {
      toast(reason(e), 'bad')
    }
  }
</script>

<div class="ctx">
  <div class="group">
    <span class="caps">Dataset</span>
    <Select value={app.dataset || null} {options} placeholder={options.length ? 'Choose a dataset…' : 'No datasets yet'}
            width="260px" disabled={!options.length} onchange={(v) => setDataset(String(v))} />
    {#if chosen}<Chip tone={chosen.modality === 'audio' ? 'audio' : 'accent'} small>{chosen.modality}</Chip>{/if}
  </div>
  <div class="divider"></div>
  <div class="group">
    <span class="caps">Collection</span>
    <Select value={app.collection.slug} options={boardOptions} placeholder="Choose a collection…" width="260px"
            onchange={(v) => useCollection(String(v))} />
    {#if app.collection.slug}<span class="count dim">{app.collection.items.length} picture{app.collection.items.length === 1 ? '' : 's'}</span>{/if}
    <div class="new"><TextInput bind:value={newName} placeholder="+ new collection" size="sm" onenter={create} /></div>
  </div>
</div>

<style>
  .ctx { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 18px; padding: 8px 14px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); flex: 0 0 auto; }
  .group { display: flex; align-items: center; gap: 10px; }
  .divider { width: 1px; align-self: stretch; background: var(--arc-line); }
  .count { font-size: var(--fs-xs); white-space: nowrap; }
  .new { width: 160px; }
</style>
