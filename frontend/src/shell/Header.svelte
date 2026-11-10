<script lang="ts">
  import { app, MODES, setMode, foldHeader } from '../lib/stores/app.svelte'
  import JobPill from './JobPill.svelte'
  import ModalityTag from '../lib/components/ModalityTag.svelte'
  import { caps } from '../lib/stores/caps.svelte'
</script>

<header class="top">
  <div class="brand">
    <img src="/assets/arcana-mark.png" alt="" />
    <span class="name">Arcana</span>
    {#if app.version}<span class="ver mono">{app.version}</span>{/if}
  </div>

  <nav class="tabs" aria-label="Modes">
    {#each MODES as m (m.value)}
      {@const t = caps.tab(m.value)}
      <button class="tab" class:on={app.mode === m.value} class:off={t.off} style="--tab: {m.hue}" title={t.why} onclick={() => setMode(m.value)}>
        <span>{m.label}</span><ModalityTag modalities={t.modalities} dots />
      </button>
    {/each}
  </nav>

  <div class="right">
    <JobPill />
    <button class="fold" title="Hide the header band" onclick={() => foldHeader(true)}>▴ hide header</button>
  </div>
</header>

<style>
  .top { display: flex; align-items: center; gap: var(--sp-6); flex: 0 0 auto; min-height: var(--header-h); }
  .brand { display: flex; align-items: center; gap: 10px; flex: 0 0 auto; }
  .brand img { height: 34px; width: auto; }
  .name { font-size: var(--fs-2xl); font-weight: 600; letter-spacing: 0.5px; }
  .ver { font-size: var(--fs-2xs); color: var(--arc-ink-faint); align-self: flex-end; margin-bottom: 4px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .tab {
    display: inline-flex; align-items: center; gap: 7px;
    height: 32px; padding: 0 14px; border-radius: var(--r-md); font-weight: 600; font-size: var(--fs-md); letter-spacing: 0.3px;
    border: 1px solid color-mix(in srgb, var(--tab) 40%, transparent); color: var(--tab);
    background: color-mix(in srgb, var(--tab) 6%, transparent); transition: background var(--ease), border-color var(--ease);
  }
  .tab:hover { background: color-mix(in srgb, var(--tab) 16%, transparent); border-color: var(--tab); }
  .tab.on { background: color-mix(in srgb, var(--tab) 26%, transparent); border-color: var(--tab); color: var(--arc-ink); box-shadow: inset 0 -2px 0 var(--tab); }
  .tab.off { opacity: 0.45; }
  .right { margin-left: auto; display: flex; align-items: center; gap: var(--sp-3); }
  .fold { height: 20px; padding: 0 9px; font-size: var(--fs-xs); letter-spacing: 0.4px; color: var(--arc-ink-dim); background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-pill); }
  .fold:hover { color: var(--arc-accent); border-color: var(--arc-accent); }
</style>
