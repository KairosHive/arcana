<script lang="ts">
  import { onMount } from 'svelte'
  import { app, bootstrap, foldHeader } from './lib/stores/app.svelte'
  import Header from './shell/Header.svelte'
  import ContextBar from './shell/ContextBar.svelte'
  import RelocatePanel from './shell/RelocatePanel.svelte'
  import ActivityStrip from './shell/ActivityStrip.svelte'
  import Viewer from './shell/Viewer.svelte'
  import Toast from './lib/ui/Toast.svelte'
  import Spinner from './lib/ui/Spinner.svelte'
  import DatasetsTab from './tabs/datasets/DatasetsTab.svelte'
  import SearchTab from './tabs/search/SearchTab.svelte'
  import MoodboardTab from './tabs/moodboard/MoodboardTab.svelte'
  import AlchemyTab from './tabs/alchemy/AlchemyTab.svelte'
  import GenerateTab from './tabs/generate/GenerateTab.svelte'
  import AgentTab from './tabs/agent/AgentTab.svelte'
  import ArchiveTab from './tabs/archive/ArchiveTab.svelte'

  // A tab is mounted the first time it is opened and kept after that, hidden,
  // so its results, its map and its terminals survive a switch.
  let visited = $state<Record<string, boolean>>({})
  $effect(() => { visited[app.mode] = true })
  $effect(() => { app.mode; requestAnimationFrame(() => window.dispatchEvent(new Event('resize'))) })

  onMount(() => { bootstrap() })
</script>

<div class="shell" class:folded={app.headerFolded}>
  {#if !app.headerFolded}
    <Header />
    <ContextBar />
  {:else}
    <button class="unfold" title="Show the header band" onclick={() => foldHeader(false)}>▾ show header</button>
  {/if}
  <RelocatePanel />
  {#if app.ready}<ActivityStrip />{/if}

  <main class="main">
    {#if app.error}
      <div class="boot-error"><p class="t">Arcana could not start its page.</p><p class="hint">{app.error}</p></div>
    {:else if !app.ready}
      <div class="boot"><Spinner size={18} label="Opening Arcana…" /></div>
    {:else}
      {#if visited.datasets}<section class="tab" hidden={app.mode !== 'datasets'}><DatasetsTab /></section>{/if}
      {#if visited.prompt}<section class="tab" hidden={app.mode !== 'prompt'}><SearchTab mode="prompt" /></section>{/if}
      {#if visited.story}<section class="tab" hidden={app.mode !== 'story'}><SearchTab mode="story" /></section>{/if}
      {#if visited.moodboard}<section class="tab" hidden={app.mode !== 'moodboard'}><MoodboardTab /></section>{/if}
      {#if visited.alchemy}<section class="tab" hidden={app.mode !== 'alchemy'}><AlchemyTab /></section>{/if}
      {#if visited.generate}<section class="tab" hidden={app.mode !== 'generate'}><GenerateTab /></section>{/if}
      {#if visited.archive}<section class="tab" hidden={app.mode !== 'archive'}><ArchiveTab /></section>{/if}
      {#if visited.agent}<section class="tab" hidden={app.mode !== 'agent'}><AgentTab /></section>{/if}
    {/if}
  </main>
</div>

<Viewer />
<Toast />

<style>
  .shell { height: 100vh; display: flex; flex-direction: column; gap: var(--sp-2); padding: var(--sp-3) var(--sp-4) var(--sp-2); box-sizing: border-box; overflow: hidden; }
  .main { flex: 1; min-height: 0; display: flex; }
  .tab { flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
  .tab[hidden] { display: none; }
  .unfold { position: fixed; top: 4px; right: 10px; z-index: 60; height: 20px; padding: 0 9px; font-size: var(--fs-xs); letter-spacing: 0.4px; color: var(--arc-ink-dim); background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-pill); }
  .unfold:hover { color: var(--arc-accent); border-color: var(--arc-accent); }
  .boot, .boot-error { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
  .boot-error .t { font-size: var(--fs-lg); }
</style>
