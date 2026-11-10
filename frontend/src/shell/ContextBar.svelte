<script lang="ts">
  import { app, setDataset, useCollection, newCollection, foldHeader } from '../lib/stores/app.svelte'
  import { reason } from '../lib/api/client'
  import { toast } from '../lib/stores/toast.svelte'
  import Select from '../lib/ui/Select.svelte'
  import TextInput from '../lib/ui/TextInput.svelte'
  import Chip from '../lib/ui/Chip.svelte'
  import { caps } from '../lib/stores/caps.svelte'

  let newName = $state('')
  // Grouped by modality, photos first, so the menu reads as the kinds of archive on this machine.
  const options = $derived([...app.datasets].sort((a, b) => caps.order(a.modality) - caps.order(b.modality) || a.label.localeCompare(b.label))
    .map((d) => ({ value: d.value, label: `${d.name} · ${caps.label(d.modality)} (${d.dim}D)` })))
  const boardOptions = $derived(app.boards.map((b) => ({ value: b.slug, label: `${b.name}  ·  ${b.count} picture${b.count === 1 ? '' : 's'}` })))
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
    {#if chosen}<Chip tone={chosen.modality === 'image' ? 'accent' : chosen.modality} small>{caps.label(chosen.modality)}</Chip>{/if}
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
