<script lang="ts">
  import type { Snippet } from 'svelte'
  let { tone = 'neutral', small = false, title = '', color = '', onclick, children }: {
    tone?: 'neutral' | 'accent' | 'ok' | 'warn' | 'bad' | 'target' | 'audio' | 'video' | 'faint'; small?: boolean; title?: string
    /** an explicit hue, for colour tags */
    color?: string; onclick?: () => void; children?: Snippet
  } = $props()
</script>
{#if onclick}
  <button class="chip {tone}" class:small {title} style:--c={color || null} {onclick}>{@render children?.()}</button>
{:else}
  <span class="chip {tone}" class:small {title} style:--c={color || null}>{@render children?.()}</span>
{/if}
<style>
  .chip { display: inline-flex; align-items: center; gap: 4px; height: 20px; padding: 0 8px; border-radius: var(--r-pill); font-size: var(--fs-xs); font-weight: 600; border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); color: var(--c); background: color-mix(in srgb, var(--c) 10%, transparent); white-space: nowrap; line-height: 1; }
  .small { height: 16px; padding: 0 6px; font-size: var(--fs-2xs); }
  button.chip { cursor: pointer; } button.chip:hover { background: color-mix(in srgb, var(--c) 22%, transparent); }
  .neutral { --c: var(--arc-ink-dim); } .faint { --c: var(--arc-ink-faint); } .accent { --c: var(--arc-accent); } .ok { --c: var(--arc-ok); } .warn { --c: var(--arc-warn); } .bad { --c: var(--arc-bad); } .target { --c: var(--arc-target); } .audio { --c: var(--arc-audio); } .video { --c: var(--arc-video); }
</style>
