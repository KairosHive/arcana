<script lang="ts">
  import type { Filter } from './map'
  import Select from '../../lib/ui/Select.svelte'
  import Button from '../../lib/ui/Button.svelte'

  import type { MarksInfo } from './types'

  /** Stars, colour tags and a date range: the filters a culled archive gives us. */
  let { info, filter = $bindable(), count = '', panelOpen = $bindable(false) }: {
    info: MarksInfo; filter: Filter; count?: string; panelOpen?: boolean
  } = $props()

  function toggleColour(c: string) {
    filter = { ...filter, colours: filter.colours.includes(c) ? filter.colours.filter((x) => x !== c) : [...filter.colours, c] }
  }
  function clear() { filter = { stars: 0, colours: [], dateFrom: '', dateTo: '' } }
  const active = $derived(!!(filter.stars || filter.colours.length || filter.dateFrom || filter.dateTo))
</script>

<div class="bar">
  {#if info.hasStars && info.stars}
    <Select value={filter.stars} options={info.stars} size="sm" width="170px" onchange={(v) => (filter = { ...filter, stars: Number(v) })} />
  {/if}
  {#if info.hasColours && info.colours}
    <div class="tags">
      {#each info.colours as c (c.value)}
        <button class="tag" class:on={filter.colours.includes(c.value)} onclick={() => toggleColour(c.value)} title={c.label}>
          <span class="dot" style:background={c.colour}></span>{c.label}
        </button>
      {/each}
    </div>
  {/if}
  {#if info.dates}
    <label class="date"><span class="caps">from</span><input type="date" min={info.dates.min} max={info.dates.max} value={filter.dateFrom} onchange={(e) => (filter = { ...filter, dateFrom: (e.target as HTMLInputElement).value })} /></label>
    <label class="date"><span class="caps">to</span><input type="date" min={info.dates.min} max={info.dates.max} value={filter.dateTo} onchange={(e) => (filter = { ...filter, dateTo: (e.target as HTMLInputElement).value })} /></label>
  {/if}
  {#if active}<Button size="sm" kind="ghost" onclick={clear}>Clear</Button>{/if}
  <span class="count dim">{count}</span>
  {#if info.hasStars || info.hasColours}
    <Button size="sm" onclick={() => (panelOpen = !panelOpen)}>{panelOpen ? 'Hide' : "What's marked"}</Button>
  {/if}
</div>

<style>
  .bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding-top: 10px; border-top: 1px solid var(--arc-line); }
  .tags { display: flex; gap: 5px; flex-wrap: wrap; }
  .tag { display: inline-flex; align-items: center; gap: 5px; height: var(--ctl-h-sm); padding: 0 9px; border-radius: var(--r-pill); border: 1px solid var(--arc-line); background: transparent; color: var(--arc-ink-dim); font-size: var(--fs-xs); }
  .tag.on { border-color: var(--arc-accent); color: var(--arc-ink); background: color-mix(in srgb, var(--arc-accent) 15%, transparent); }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .date { display: inline-flex; align-items: center; gap: 6px; }
  .date input { height: var(--ctl-h-sm); background: var(--arc-surface-2); color: var(--arc-ink); border: 1px solid var(--arc-line); border-radius: var(--r-md); padding: 0 6px; font-size: var(--fs-xs); color-scheme: dark; }
  .count { font-size: var(--fs-xs); margin-left: auto; }
</style>
