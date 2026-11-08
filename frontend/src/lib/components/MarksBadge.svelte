<script lang="ts">
  /** What was already decided about a picture: stars, colour tag, capture date. */
  import type { Marks } from '../../tabs/search/types'
  let { marks, swatch = {} }: { marks: Marks; swatch?: Record<string, string> } = $props()
  const SWATCH: Record<string, string> = { Red: '#e05252', Yellow: '#d9b234', Green: '#43b581', Blue: '#4a90d9', Purple: '#a982d9' }
  const stars = $derived(marks && marks.stars !== null && marks.stars !== undefined ? '★'.repeat(Math.max(0, Math.min(5, marks.stars))) + '☆'.repeat(5 - Math.max(0, Math.min(5, marks.stars))) : '')
</script>

{#if marks && (stars || marks.colour || marks.captured)}
  <div class="badge">
    {#if stars}<span class="stars" class:lit={marks.stars! > 0}>{stars}</span>{/if}
    {#if marks.colour}<span class="tag"><span class="dot" style:background={swatch[marks.colour] || SWATCH[marks.colour] || 'var(--arc-muted)'}></span>{marks.colour}</span>{/if}
    {#if marks.captured}<span class="when faint">{marks.captured}</span>{/if}
  </div>
{/if}

<style>
  .badge { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: var(--fs-xs); }
  .stars { letter-spacing: 1px; font-size: var(--fs-sm); color: var(--arc-ink-faint); }
  .stars.lit { color: var(--arc-warn); }
  .tag { display: inline-flex; align-items: center; gap: 5px; color: var(--arc-ink-dim); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
</style>
