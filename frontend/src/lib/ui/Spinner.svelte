<script lang="ts">
  let { size = 14, label = '' }: { size?: number; label?: string } = $props()
</script>
<span class="sp"><span class="ring" style:width="{size}px" style:height="{size}px"></span>{#if label}<span class="lab">{label}</span>{/if}</span>
<style>
  .sp { display: inline-flex; align-items: center; gap: 8px; color: var(--arc-accent); font-size: var(--fs-sm); }
  .ring { border-radius: 50%; border: 2px solid currentColor; border-right-color: transparent; animation: arc-spin 0.8s linear infinite; flex: 0 0 auto; }
  .lab { color: var(--arc-ink-dim); }
</style>
