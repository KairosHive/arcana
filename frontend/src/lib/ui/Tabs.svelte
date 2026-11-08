<script lang="ts" generics="T extends string">
  let { value = $bindable(), tabs, size = 'md', onchange }: {
    value?: T; tabs: { value: T; label: string; badge?: string; disabled?: boolean }[]
    size?: 'sm' | 'md'; onchange?: (v: T) => void
  } = $props()
</script>

<div class="tabs {size}" role="tablist">
  {#each tabs as t (t.value)}
    <button role="tab" class="tab" class:on={t.value === value} aria-selected={t.value === value}
            disabled={t.disabled} onclick={() => { value = t.value; onchange?.(t.value) }}>
      {t.label}{#if t.badge}<span class="badge">{t.badge}</span>{/if}
    </button>
  {/each}
</div>

<style>
  .tabs { display: inline-flex; gap: 2px; padding: 3px; background: var(--arc-surface-2); border: 1px solid var(--arc-line); border-radius: var(--r-md); }
  .tab { height: var(--ctl-h-sm); padding: 0 12px; border-radius: var(--r-sm); border: none; background: transparent; color: var(--arc-ink-dim); font-size: var(--fs-sm); font-weight: 600; display: inline-flex; align-items: center; gap: 6px; transition: background var(--ease), color var(--ease); }
  .sm .tab { height: 22px; font-size: var(--fs-xs); padding: 0 9px; }
  .tab:hover:not(:disabled) { color: var(--arc-ink); }
  .tab.on { background: var(--arc-surface); color: var(--arc-ink); box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
  .tab:disabled { opacity: 0.4; cursor: not-allowed; }
  .badge { font-size: var(--fs-2xs); color: var(--arc-accent); }
</style>
