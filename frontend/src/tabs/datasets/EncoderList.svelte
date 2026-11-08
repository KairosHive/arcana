<script lang="ts">
  import { post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import type { ModelRow } from './types'
  import Button from '../../lib/ui/Button.svelte'
  import Chip from '../../lib/ui/Chip.svelte'

  let { models, busy = false, onstarted }: { models: ModelRow[]; busy?: boolean; onstarted: (id: string) => void } = $props()

  async function fetchModel(m: ModelRow) {
    try {
      const r = await post('/api/datasets/action', { action: 'download', model: m.id })
      if (r.job) { onstarted(r.job); toast(`Preparing ${m.label}…`, 'info') }
      else toast(`${m.label} is ready.`, 'ok')
    } catch (e) {
      toast(reason(e), 'bad')
    }
  }
</script>

<div class="list">
  {#each models as m (m.id)}
    <div class="mrow">
      <span class="stripe" class:on={m.complete}></span>
      <div class="txt">
        <div class="label">{m.label}</div>
        <div class="blurb dim">{m.blurb}</div>
      </div>
      <div class="pills">
        {#if m.downloaded}<Chip tone="ok" small>downloaded</Chip>
        {:else if m.partial}<Chip tone="bad" small>incomplete</Chip>
        {:else}<Chip tone="faint" small>{m.download_mb.toLocaleString()} MB</Chip>{/if}
        <Chip tone={m.labelsReady ? 'ok' : 'warn'} small>{m.labelsReady ? 'labels ready' : 'labels pending'}</Chip>
      </div>
      <Button size="sm" disabled={m.complete || busy} onclick={() => fetchModel(m)}>{m.complete ? 'Ready' : m.downloaded ? 'Prepare labels' : 'Download'}</Button>
    </div>
  {/each}
</div>

<style>
  .list { display: flex; flex-direction: column; }
  .mrow { display: flex; align-items: center; gap: 14px; padding: 10px 0; flex-wrap: wrap; border-top: 1px solid var(--arc-line); }
  .mrow:first-child { border-top: none; }
  .stripe { width: 3px; align-self: stretch; min-height: 26px; border-radius: 2px; background: var(--arc-ink-faint); }
  .stripe.on { background: var(--arc-ok); }
  .txt { flex: 1; min-width: 260px; }
  .label { font-weight: 600; }
  .blurb { font-size: var(--fs-xs); line-height: 1.45; margin-top: 2px; }
  .pills { display: flex; gap: 6px; flex-wrap: wrap; }
</style>
