<script lang="ts">
  import TreeNode from './TreeNode.svelte'
  import { archive, ICON, prefix, dropTarget, op, type Node } from './archive.svelte'

  /**
   * One node of the archive tree with what is inside it. A row drags; a kind,
   * a folder or a collection takes the drop, which is `archive move`.
   */
  let { node, depth = 0, open = false }: { node: Node; depth?: number; open?: boolean } = $props()
  const TYPE = 'text/arcana-archive'
  // svelte-ignore state_referenced_locally
  let isOpen = $state(open)
  let over = $state(false)
  const kids = $derived(node.folders?.length + node.items?.length > 0)
  const showCount = $derived(node.count > 0 && ['collection', 'film', 'dir'].includes(prefix(node.id)))
  // Pictures made group by the folder they sit in on disk.
  const groups = $derived.by(() => {
    if (node.id !== 'kind:pictures') return null
    const g: Record<string, Node[]> = {}
    for (const it of node.items) (g[it.group || ''] ??= []).push(it)
    return Object.entries(g).sort(([a], [b]) => a.localeCompare(b))
  })
  let openGroups = $state<Record<string, boolean>>({})

  function dragstart(e: DragEvent) {
    if (prefix(node.id) === 'kind') { e.preventDefault(); return }
    e.dataTransfer?.setData(TYPE, node.id)
    if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move'
    e.stopPropagation()
  }
  function dragover(e: DragEvent) {
    if (!dropTarget(node) || !e.dataTransfer?.types.includes(TYPE)) return
    e.preventDefault(); e.stopPropagation()
    e.dataTransfer.dropEffect = 'move'
    over = true
  }
  async function drop(e: DragEvent) {
    over = false
    if (!dropTarget(node) || !e.dataTransfer?.types.includes(TYPE)) return
    e.preventDefault(); e.stopPropagation()
    const id = e.dataTransfer.getData(TYPE)
    if (!id || id === node.id) return
    archive.status = await op('archive move', { id, into: node.id })
    archive.sel = id
  }
  function select(e: MouseEvent) {
    e.stopPropagation()
    archive.sel = node.id
    if (kids && (e.detail === 2 || prefix(node.id) === 'kind')) isOpen = !isOpen
  }
</script>

<div class="node">
  <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
  <div class="row" class:sel={archive.sel === node.id} class:over class:target={dropTarget(node)} style:padding-left="{8 + 14 * depth}px"
       draggable={prefix(node.id) !== 'kind'} data-id={node.id}
       ondragstart={dragstart} ondragover={dragover} ondragleave={() => (over = false)} ondrop={drop} onclick={select}>
    {#if kids}<button class="twist" onclick={(e) => { e.stopPropagation(); isOpen = !isOpen }} aria-label={isOpen ? 'Collapse' : 'Expand'}>{isOpen ? '▾' : '▸'}</button>{:else}<span class="twist"></span>{/if}
    <span class="icon">{ICON[node.kind] ?? '▪'}</span>
    <span class="name ellipsis">{node.shown}</span>
    {#if showCount}<span class="count mono">{node.count}</span>{/if}
  </div>
  {#if isOpen && kids}
    {#each [...node.folders].sort((a, b) => a.shown.toLowerCase().localeCompare(b.shown.toLowerCase())) as f (f.id)}
      <TreeNode node={f} depth={depth + 1} />
    {/each}
    {#if groups}
      {#each groups as [g, items] (g)}
        {#if g}
          <button class="group" style:padding-left="{8 + 14 * (depth + 1)}px" onclick={() => (openGroups[g] = !openGroups[g])}>{openGroups[g] ? '▾' : '▸'} {g} <span class="count mono">{items.length}</span></button>
          {#if openGroups[g]}{#each items as it (it.id)}<TreeNode node={it} depth={depth + 2} />{/each}{/if}
        {:else}
          {#each items as it (it.id)}<TreeNode node={it} depth={depth + 1} />{/each}
        {/if}
      {/each}
    {:else}
      {#each node.items as it (it.id)}
        <TreeNode node={it} depth={depth + 1} />
      {/each}
    {/if}
  {/if}
</div>

<style>
  .row { display: flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 5px; cursor: pointer; border: 1px solid transparent; font-size: 12.5px; user-select: none; }
  .row:hover { background: var(--arc-surface-2); }
  .row.sel { background: color-mix(in srgb, var(--tab-archive) 22%, transparent); border-color: var(--tab-archive); }
  .row.over { background: color-mix(in srgb, var(--arc-accent) 25%, transparent); border-color: var(--arc-accent); }
  .twist { width: 14px; flex: 0 0 auto; border: none; background: transparent; color: var(--arc-ink-faint); font-size: 10px; padding: 0; text-align: center; }
  .icon { color: var(--arc-ink-dim); width: 12px; text-align: center; font-size: 11px; }
  .name { flex: 1; min-width: 0; }
  .count { color: var(--arc-ink-faint); font-size: var(--fs-xs); }
  .group { display: flex; align-items: center; gap: 6px; width: 100%; text-align: left; background: transparent; border: none; color: var(--arc-ink-dim); font-size: 11.5px; padding: 3px 8px; }
</style>
