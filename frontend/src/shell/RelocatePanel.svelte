<script lang="ts">
  import { app, current, refreshDatasets } from '../lib/stores/app.svelte'
  import { get, post, reason } from '../lib/api/client'
  import Button from '../lib/ui/Button.svelte'
  import TextInput from '../lib/ui/TextInput.svelte'
  import Status from '../lib/ui/Status.svelte'

  type Health = { ok: boolean; total: number; checked: number; present: number; missing: number; root: string; error: string }
  let health = $state<Health | null>(null)
  let root = $state('')
  let status = $state(''), tone = $state<'dim' | 'ok' | 'warn' | 'bad'>('dim')
  let busy = $state(false)
  let checkedFor = ''

  $effect(() => {
    const ds = current.dataset
    if (!ds || ds.value === checkedFor) return
    checkedFor = ds.value
    health = null; status = ''
    get<Health>('/api/datasets/health', { name: ds.name, modality: ds.modality }).then((h) => {
      health = h
      root = h.root || ''
    }).catch(() => { health = null })
  })

  const shown = $derived(!!health && (!!health.error || health.missing > 0) && app.mode !== 'datasets' && app.mode !== 'agent')

  async function run(apply: boolean) {
    const ds = current.dataset
    if (!ds) { status = 'Select a dataset first.'; tone = 'bad'; return }
    busy = true
    try {
      const r = await post('/api/datasets/relocate', { name: ds.name, modality: ds.modality, root, apply })
      if (r.found === 0) {
        status = `Found none of the ${r.total.toLocaleString()} files under ${root}. Point at the folder that directly contains them (or their subfolders).`; tone = 'bad'
      } else if (!apply) {
        status = `Would match ${r.found.toLocaleString()} of ${r.total.toLocaleString()} files here` + (r.missing ? `, ${r.missing} would still be missing.` : '.') + ' Press Relocate to apply.'; tone = 'ok'
      } else {
        status = `✓ Relocated ${r.found.toLocaleString()} of ${r.total.toLocaleString()} files.` + (r.missing ? ` ${r.missing} still missing.` : '') + ' Backups were written next to the originals as .bak.'; tone = 'ok'
        checkedFor = ''
        await refreshDatasets()
      }
    } catch (e) {
      status = reason(e); tone = 'bad'
    } finally { busy = false }
  }
</script>

{#if shown && health}
  <div class="panel">
    <p class="msg">
      {#if health.error}⚠ {health.error}{:else}
        <b>⚠ This dataset's files are missing: {health.missing} of {health.checked} sampled{#if health.total} (dataset has {health.total.toLocaleString()} items){/if}.</b>
        Indexed from <code>{health.root || 'an unknown folder'}</code>. If that folder moved or lives on a drive that is not connected, point Arcana at its new location below. Files are matched by content, so nothing is re-indexed.
      {/if}
    </p>
    <div class="row nowrap">
      <div class="grow"><TextInput bind:value={root} placeholder="Paste the folder where these files live now…" /></div>
      <Button onclick={() => run(false)} busy={busy}>Check</Button>
      <Button kind="success" onclick={() => run(true)} disabled={busy}>Relocate</Button>
    </div>
    <Status text={status} {tone} />
  </div>
{/if}

<style>
  .panel { flex: 0 0 auto; padding: 12px 14px; background: #2a2318; border: 1px solid #5a4a22; border-radius: var(--r-md); display: flex; flex-direction: column; gap: 8px; }
  .msg { font-size: var(--fs-md); color: var(--arc-warn); line-height: 1.5; }
  .msg b { display: block; margin-bottom: 2px; }
  code { color: var(--arc-ink); }
</style>
