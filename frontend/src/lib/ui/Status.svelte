<script lang="ts">
  /** One line of feedback under a control. Empty text draws nothing. */
  let { text = '', tone = 'dim', size = 'sm' }: { text?: string; tone?: 'dim' | 'ok' | 'warn' | 'bad' | 'accent' | 'ink'; size?: 'xs' | 'sm' | 'md' } = $props()
</script>
{#if text}<p class="st {tone} {size}">{text}</p>{/if}
<style>
  .st { line-height: 1.45; margin: 0; white-space: pre-wrap; }
  .xs { font-size: var(--fs-xs); } .sm { font-size: var(--fs-sm); } .md { font-size: var(--fs-md); }
  .dim { color: var(--arc-ink-dim); } .ok { color: var(--arc-ok); } .warn { color: var(--arc-warn); } .bad { color: var(--arc-bad); } .accent { color: var(--arc-accent); } .ink { color: var(--arc-ink); }
</style>
