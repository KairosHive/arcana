<script lang="ts">
  import type { Snippet } from 'svelte'
  let { title, hint = '', compact = false, children }: { title: string; hint?: string; compact?: boolean; children?: Snippet } = $props()
</script>
<div class="empty" class:compact>
  <p class="t">{title}</p>
  {#if hint}<p class="hint">{hint}</p>{/if}
  {@render children?.()}
</div>
<style>
  .empty { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: var(--sp-8) var(--sp-4); text-align: center; color: var(--arc-ink-dim); height: 100%; min-height: 120px; }
  .compact { padding: var(--sp-4); min-height: 0; }
  .t { font-size: var(--fs-lg); color: var(--arc-ink); }
  .compact .t { font-size: var(--fs-md); }
  .hint { max-width: 48ch; text-align: center; }
</style>
