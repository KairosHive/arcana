<script lang="ts">
  import type { Snippet } from 'svelte'
  let { checked = $bindable(false), label = '', disabled = false, onchange, children }: {
    checked?: boolean; label?: string; disabled?: boolean; onchange?: (v: boolean) => void; children?: Snippet
  } = $props()
</script>

<label class="cb" class:disabled>
  <input type="checkbox" bind:checked {disabled} onchange={() => onchange?.(checked)} />
  <span class="box" aria-hidden="true">{#if checked}✓{/if}</span>
  {#if label}<span class="lab">{label}</span>{/if}
  {@render children?.()}
</label>

<style>
  .cb { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; user-select: none; font-size: var(--fs-sm); position: relative; }
  .cb input { position: absolute; opacity: 0; width: 0; height: 0; }
  .box { width: 15px; height: 15px; border-radius: var(--r-sm); border: 1px solid var(--arc-line-strong); background: var(--arc-surface-2); display: inline-flex; align-items: center; justify-content: center; font-size: 11px; color: var(--arc-accent-ink); flex: 0 0 auto; transition: background var(--ease); }
  input:checked + .box { background: var(--arc-accent); border-color: var(--arc-accent); }
  input:focus-visible + .box { box-shadow: var(--focus); }
  .disabled { opacity: 0.5; cursor: not-allowed; }
</style>
