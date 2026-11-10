<script lang="ts">
  import { onMount } from 'svelte'
  import { app } from '../../lib/stores/app.svelte'
  import { catalog, loadCatalog, controls, alchemyUi, alchemyLeft } from './alchemy.svelte'
  import { film, addLook } from './film.svelte'
  import Cast from './Cast.svelte'
  import Learn from './Learn.svelte'
  import Stage from './Stage.svelte'
  import Settings from './Settings.svelte'
  import FilmTab from './FilmTab.svelte'
  import Tabs from '../../lib/ui/Tabs.svelte'
  import Spinner from '../../lib/ui/Spinner.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import SplitPane from '../../lib/ui/SplitPane.svelte'

  /** Alchemy: Image (the step) and Video (the film). */
  let stage: Stage | undefined = $state()
  const visible = $derived(app.mode === 'alchemy')

  onMount(() => { loadCatalog() })

  async function look() {
    const msg = await addLook(controls())
    return msg
  }
</script>

<div class="alc">
  <div class="subbar">
    <Tabs bind:value={alchemyUi.sub} tabs={[{ value: 'image', label: 'Image' }, { value: 'video', label: 'Video', badge: film.spec.pictures.length ? String(film.spec.pictures.length) : '' }]} />
  </div>
  {#if !catalog.data}
    <div class="center">{#if catalog.error}<Status text={catalog.error} tone="bad" />{:else}<Spinner size={18} label="Reading the methods…" />{/if}</div>
  {:else}
    <div class="view" hidden={alchemyUi.sub !== 'image'}>
      <SplitPane storeKey="alchemy-cast" initial={250} min={180} max={420}>
        {#snippet a()}
          <div class="leftcol">
            <Tabs bind:value={alchemyLeft.pane} size="sm" tabs={[{ value: 'pictures', label: 'Pictures' }, { value: 'learn', label: 'Sets & labels' }]} />
            <div class="pane" hidden={alchemyLeft.pane !== 'pictures'}><Cast /></div>
            <div class="pane" hidden={alchemyLeft.pane !== 'learn'}><Learn onchange={() => stage?.changed()} /></div>
          </div>
        {/snippet}
        {#snippet b()}
          <div class="mid">
            <div class="stage-col"><Stage bind:this={stage} onlook={async () => { await look() }} /></div>
            <div class="settings-col"><Settings onchange={() => stage?.changed()} /></div>
          </div>
        {/snippet}
      </SplitPane>
    </div>
    <div class="view" hidden={alchemyUi.sub !== 'video'}>
      <FilmTab active={alchemyUi.sub === 'video' && visible} />
    </div>
  {/if}
</div>

<style>
  .alc { display: flex; flex-direction: column; gap: var(--sp-2); flex: 1; min-height: 0; min-width: 0; }
  .subbar { display: flex; align-items: center; gap: 12px; flex: 0 0 auto; }
  .center { flex: 1; display: flex; align-items: center; justify-content: center; }
  .view { flex: 1; min-height: 0; display: flex; }
  .view[hidden] { display: none; }
  .leftcol { display: flex; flex-direction: column; gap: 6px; flex: 1; min-height: 0; min-width: 0; height: 100%; }
  .pane { flex: 1; min-height: 0; display: flex; }
  .pane[hidden] { display: none; }
  .mid { display: flex; gap: 14px; flex: 1; min-width: 0; min-height: 0; padding-left: var(--sp-1); }
  .stage-col { flex: 1 1 auto; min-width: 0; min-height: 0; display: flex; }
  .settings-col { flex: 0 0 370px; min-height: 0; display: flex; }
  @media (max-width: 1100px) { .mid { flex-wrap: wrap; overflow-y: auto; } .settings-col { flex: 1 1 100%; } }
</style>
