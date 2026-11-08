<script lang="ts">
  import type { Snippet } from 'svelte'
  /** A fold for the narrower-scope settings under a card's main controls:
      "Refine", "Advanced". Closed by default so the primary action reads first. */
  let { label, open = $bindable(false), note = '', level = 1, children }: {
    label: string; open?: boolean; note?: string; level?: 1 | 2; children?: Snippet
  } = $props()
</script>

<details class="disc l{level}" bind:open>
  <summary>
    <span class="mark" aria-hidden="true"></span>
    <span class="lab">{label}</span>
    {#if note}<span class="note">{note}</span>{/if}
  </summary>
  <div class="body">{@render children?.()}</div>
</details>

<style>
  .disc.l1 { background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); }
  .disc.l2 { border-top: 1px dashed var(--arc-line); }
  summary { list-style: none; cursor: pointer; display: flex; align-items: center; gap: 9px; user-select: none; }
  .l1 summary { padding: 10px 14px; }
  .l2 summary { padding: 8px 0 2px; }
  summary::-webkit-details-marker { display: none; }
  .mark { width: 7px; height: 7px; border-right: 1.5px solid var(--arc-ink-faint); border-bottom: 1.5px solid var(--arc-ink-faint); transform: rotate(-45deg); transition: transform var(--ease); flex: 0 0 auto; margin-left: 2px; }
  .disc[open] > summary .mark { transform: rotate(45deg); }
  .lab { font-size: var(--fs-sm); font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: var(--arc-ink); }
  .l2 .lab { font-size: var(--fs-xs); color: var(--arc-ink-dim); font-weight: 600; }
  .note { margin-left: auto; font-size: var(--fs-xs); color: var(--arc-ink-dim); text-align: right; }
  .l1[open] > summary { border-bottom: 1px solid var(--arc-line); }
  .body { display: flex; flex-direction: column; gap: var(--sp-3); }
  .l1 .body { padding: 12px 14px 14px; }
  .l2 .body { padding-top: var(--sp-2); }
</style>
