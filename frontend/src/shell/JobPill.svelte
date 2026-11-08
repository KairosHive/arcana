<script lang="ts">
  import { app, setMode } from '../lib/stores/app.svelte'
  const pct = $derived(app.job && app.job.fraction !== null ? Math.round(app.job.fraction * 100) : null)
</script>

{#if app.job && (app.job.status === 'running' || app.job.status === 'queued')}
  <button class="pill" title={app.job.message || app.job.label} onclick={() => setMode(app.job?.kind === 'film' || app.job?.kind === 'effect' ? 'alchemy' : app.job?.kind === 'poetry' ? 'story' : 'datasets')}>
    <span class="dot"></span>
    <span class="lab ellipsis">{app.job.label}</span>
    {#if pct !== null}<span class="pct mono">{pct}%</span>{/if}
  </button>
{/if}

<style>
  .pill { display: inline-flex; align-items: center; gap: 7px; height: 24px; padding: 0 10px; border-radius: var(--r-pill); border: 1px solid color-mix(in srgb, var(--arc-accent) 40%, transparent); background: color-mix(in srgb, var(--arc-accent) 10%, transparent); color: var(--arc-ink); font-size: var(--fs-xs); max-width: 280px; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--arc-accent); animation: pulse 1.2s ease-in-out infinite; }
  .pct { color: var(--arc-accent); }
  @keyframes pulse { 50% { opacity: 0.3; } }
</style>
