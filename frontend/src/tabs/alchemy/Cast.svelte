<script lang="ts">
  import { app } from '../../lib/stores/app.svelte'
  import { openViewer } from '../../lib/stores/viewer.svelte'
  import { thumbUrl } from '../../lib/api/media'
  import { step, setRole, pruneRoles, ROLE_KEYS, ROLE_COLOUR, type RoleKey } from './alchemy.svelte'
  import Card from '../../lib/ui/Card.svelte'

  /** The collection as the cast: T is the picture, 1 to 3 the materials mixed into it. */
  const items = $derived(app.collection.items)
  $effect(() => { pruneRoles(items) })
  function held(path: string): RoleKey[] { return ROLE_KEYS.filter((r) => step.roles[r] === path) }
</script>

<Card title="Pictures" subtitle="T is the picture. 1, 2 and 3 are the materials mixed into it. They come from the active collection, chosen in the header." pad="sm" scroll style="flex: 1 1 auto; min-height: 0">
  <div class="slots">
    {#each ROLE_KEYS as r}
      {@const p = step.roles[r]}
      <div class="slot">
        <span class="lab" style:color={ROLE_COLOUR[r]}>[{r}]</span>
        {#if p}<img src={thumbUrl(p)} alt="" />{:else}<div class="ph">{r === 'T' ? 'picture' : `material ${r}`}</div>{/if}
      </div>
    {/each}
  </div>
  <div class="cast">
    {#if !items.length}
      <p class="hint full">The active collection is empty. Pictures added with + Collection on any tab appear here.</p>
    {/if}
    {#each items as path, i (path)}
      {@const h = held(path)}
      <div class="tile">
        <button class="pic" style:border-color={h.length ? ROLE_COLOUR[h[0]] : 'var(--arc-line)'} style:border-width={h.length ? '2px' : '1px'} onclick={() => openViewer(items, i, app.collection.name)}>
          <img src={thumbUrl(path)} alt="" loading="lazy" />
        </button>
        <div class="chips">
          {#each ROLE_KEYS as r}
            <button class="chip" class:on={step.roles[r] === path} style:--c={ROLE_COLOUR[r]} title={r === 'T' ? 'the picture' : `material ${r}`} onclick={() => setRole(r, path)}>{r}</button>
          {/each}
        </div>
      </div>
    {/each}
  </div>
</Card>

<style>
  .slots { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .slot { display: flex; flex-direction: column; gap: 2px; }
  .lab { font-size: var(--fs-2xs); font-weight: 700; }
  .slot img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 5px; }
  .ph { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 10.5px; color: var(--arc-ink-faint); border: 1px dashed var(--arc-line); border-radius: 5px; text-align: center; }
  .cast { display: grid; grid-template-columns: repeat(auto-fill, minmax(76px, 1fr)); gap: 6px; flex: 1 1 0; min-height: 0; overflow-y: auto; align-content: start; }
  .full { grid-column: 1 / -1; font-size: var(--fs-xs); }
  .tile { display: flex; flex-direction: column; gap: 3px; }
  .pic { padding: 0; border: 1px solid; border-radius: 5px; overflow: hidden; background: var(--arc-surface-2); cursor: zoom-in; display: block; width: 100%; aspect-ratio: 1; }
  .pic img { width: 100%; height: 100%; object-fit: cover; }
  .chips { display: flex; gap: 2px; }
  .chip { flex: 1; height: 18px; padding: 0; font-size: 10px; font-weight: 700; border-radius: 3px; border: 1px solid var(--arc-line); background: var(--arc-role-off); color: var(--arc-ink-dim); }
  .chip.on { background: var(--c); border-color: var(--c); color: #fff; }
</style>
