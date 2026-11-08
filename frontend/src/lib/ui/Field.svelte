<script lang="ts">
  import type { Snippet } from 'svelte'
  /** A labelled control: caption above, control, optional hint under it. Lays
      controls out on one rhythm so a form reads as one thing. */
  let { label = '', hint = '', inline = false, width = '', grow = false, children }: {
    label?: string; hint?: string; inline?: boolean; width?: string; grow?: boolean; children?: Snippet
  } = $props()
</script>

<label class="field" class:inline class:grow style:width={width || null}>
  {#if label}<span class="caps lab">{label}</span>{/if}
  <span class="ctl">{@render children?.()}</span>
  {#if hint}<span class="hint">{hint}</span>{/if}
</label>

<style>
  .field { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
  .field.inline { flex-direction: row; align-items: center; gap: var(--sp-2); }
  .field.grow { flex: 1 1 auto; }
  .lab { line-height: 1; white-space: nowrap; }
  .ctl { display: flex; align-items: center; gap: var(--sp-2); min-width: 0; }
  .ctl > :global(*) { min-width: 0; }
  .grow .ctl > :global(*) { flex: 1 1 auto; }
  .hint { font-size: var(--fs-xs); }
</style>
