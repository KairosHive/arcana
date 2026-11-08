<script lang="ts">
  import { onMount } from 'svelte'
  import { app, onStateChange, useCollection, setMode } from '../../lib/stores/app.svelte'
  import { rpc, reason } from '../../lib/api/client'
  import { previewUrl, thumbUrl, filmVideoUrl, basename } from '../../lib/api/media'
  import { openViewer } from '../../lib/stores/viewer.svelte'
  import { archive, loadArchive, flat, prefix, op, rename, show, type Node, type Shown } from './archive.svelte'
  import { alchemyUi } from '../alchemy/alchemy.svelte'
  import { openFilm } from '../alchemy/film.svelte'
  import TreeNode from './TreeNode.svelte'
  import Card from '../../lib/ui/Card.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Spinner from '../../lib/ui/Spinner.svelte'
  import Empty from '../../lib/ui/Empty.svelte'

  /**
   * Everything the person has kept, as one tree: collections, films, videos,
   * stories, pictures made, search saves, imports and exports. The tree is on
   * the left, the selection plays in the middle, and the right side renames,
   * moves, merges, makes folders, deletes into the trash and restores.
   */
  const visible = $derived(app.mode === 'archive')
  const sel = $derived(archive.sel)
  const nodes = $derived(flat())
  const node = $derived(sel ? nodes[sel] : undefined)

  let name = $state('')
  let moveTo = $state<string | null>(null)
  let mergeTo = $state<string | null>(null)
  let folderName = $state('')
  let emptyArmed = $state(0)
  $effect(() => { name = node?.shown ?? '' })

  const moveTargets = $derived.by(() => {
    if (!node || prefix(node.id) === 'kind') return []
    const out = [{ value: `kind:${node.kind}`, label: `top of ${archive.titles[node.kind] ?? node.kind}` }]
    for (const o of Object.values(nodes)) {
      if (o.id === node.id || o.kind !== node.kind) continue
      if (prefix(o.id) === 'folder') out.push({ value: o.id, label: `folder · ${o.shown}` })
      else if (prefix(o.id) === 'collection' && prefix(node.id) === 'collection') out.push({ value: o.id, label: `collection · ${o.shown}` })
    }
    return out
  })
  const mergeTargets = $derived.by(() => {
    if (!node || !['collection', 'folder'].includes(prefix(node.id))) return []
    return Object.values(nodes).filter((o) => o.id !== node.id && o.kind === node.kind && prefix(o.id) === prefix(node.id)).map((o) => ({ value: o.id, label: o.shown }))
  })

  // ---- the selection, shown or played ----
  let shown = $state<Shown | null>(null)
  let showErr = $state('')
  let frame = $state(0)
  let playing = $state(false)
  let timer: ReturnType<typeof setInterval> | null = null
  $effect(() => {
    const id = sel
    frame = 0; playing = false
    if (!id || ['kind', 'folder'].includes(prefix(id))) { shown = null; return }
    show(id).then((s) => { if (archive.sel === id) { shown = s; showErr = '' } }).catch((e) => { shown = null; showErr = reason(e) })
  })
  $effect(() => {
    if (timer) clearInterval(timer)
    timer = playing ? setInterval(() => { if (shown?.pictures.length) frame = (frame + 1) % shown.pictures.length }, 2500) : null
    return () => { if (timer) clearInterval(timer) }
  })
  const pics = $derived(shown?.pictures ?? [])
  const current = $derived(pics.length ? pics[((frame % pics.length) + pics.length) % pics.length] : '')

  async function doRename() { if (sel && name.trim()) archive.status = await rename(sel, name.trim()) }
  async function doMove() { if (sel) archive.status = await op('archive move', { id: sel, into: moveTo || `kind:${node?.kind}` }) }
  async function doMerge() {
    if (!sel) return
    if (!mergeTo) { archive.status = 'Choose what to merge it into.'; return }
    archive.status = await op('archive merge', { id: sel, into: mergeTo })
    archive.sel = mergeTo
  }
  async function doFolder() {
    if (!node) { archive.status = 'Select a kind or a folder to make the new folder in.'; return }
    const args: Record<string, unknown> = { name: folderName, kind: node.kind }
    if (prefix(node.id) === 'folder') args.into = node.id
    try {
      const r = await rpc('archive folder add', args)
      await loadArchive()
      archive.sel = r.result.id; archive.status = r.text.trim(); folderName = ''
    } catch (e) { archive.status = reason(e) }
  }
  async function doDelete() {
    if (!sel) return
    archive.status = await op('archive delete', { id: sel })
    archive.sel = null
  }
  async function restore(id: string) { archive.status = await op('archive restore', { id }); archive.sel = id }
  async function empty() {
    const now = Date.now()
    if (!emptyArmed || now - emptyArmed > 6000) { emptyArmed = now; archive.status = 'Click Empty the trash again within a few seconds to delete it all for good.'; return }
    emptyArmed = 0
    archive.status = await op('archive empty', { yes: true })
  }
  async function useAsCollection() { if (sel) { await useCollection(sel.split(':')[1]); archive.status = `${sel.split(':')[1]} is the active collection.` } }
  async function openInVideo() {
    if (!sel) return
    try {
      const r = await rpc('alchemy film show', { film: sel.split(':')[1] })
      openFilm(r.result.spec, sel.split(':')[1])
      alchemyUi.sub = 'video'
      setMode('alchemy')
    } catch (e) { archive.status = reason(e) }
  }

  onMount(() => { loadArchive(); return onStateChange(() => { if (visible) loadArchive() }) })
  $effect(() => { if (visible) loadArchive() })
</script>

<div class="arch">
  <Card title="Archive" subtitle="Everything kept, by kind. Drag a row onto a folder, or a collection onto another collection, to move it." pad="sm" scroll style="flex: 0 0 330px; max-height: 100%">
    {#if archive.loading && !archive.kinds.length}<Spinner label="Reading the archive…" />{/if}
    <div class="tree">
      {#each archive.kinds as k (k.id)}
        <TreeNode node={k} open={['collections', 'films', 'videos'].includes(k.kind)} />
      {/each}
    </div>
  </Card>

  <Card pad="sm" scroll style="flex: 1 1 460px; min-width: min(460px, 100%); max-height: 100%">
    {#if !sel || ['kind', 'folder'].includes(prefix(sel))}
      <Empty title="Select a collection, a film, a video, a story or a picture to see it here." />
    {:else if showErr}
      <Status text={showErr} tone="bad" />
    {:else if !shown}
      <Spinner label="Opening…" />
    {:else}
      <div class="head"><span class="title">{shown.name}</span>{#if shown.kind !== 'videos'}<span class="dim small">{pics.length} picture{pics.length === 1 ? '' : 's'}</span>{/if}</div>
      {#if shown.kind === 'videos'}
        <video class="video" src={filmVideoUrl(shown.path)} controls autoplay loop><track kind="captions" /></video>
      {:else if shown.kind === 'pictures' || shown.kind === 'imports'}
        <button class="big" onclick={() => openViewer([shown!.path], 0, shown!.name)}><img src={previewUrl(shown.path, 2048)} alt="" /></button>
      {:else}
        <div class="row">
          {#if shown.kind === 'collections'}<Button size="sm" onclick={useAsCollection}>Use as the active collection</Button>{/if}
          {#if shown.kind === 'films'}<Button size="sm" onclick={openInVideo}>Open in Alchemy's Video</Button>{/if}
        </div>
        {#if !pics.length}
          <p class="hint">Nothing in it yet.</p>
        {:else}
          <div class="stage">
            <button class="big" onclick={() => openViewer(pics, frame % pics.length, shown!.name)}><img class="slide" src={previewUrl(current, 1600)} alt="" /></button>
            <div class="nav">
              <button class="nb" onclick={() => (frame = frame - 1)}>‹</button>
              <button class="nb" title="Play a slideshow" onclick={() => (playing = !playing)}>{playing ? '❚❚' : '▶'}</button>
              <button class="nb" onclick={() => (frame = frame + 1)}>›</button>
              <span class="dim small">{((frame % pics.length) + pics.length) % pics.length + 1} / {pics.length}  ·  {basename(current)}</span>
            </div>
          </div>
          <div class="strip">
            {#each pics.slice(0, 300) as p, i (p + i)}
              <button class="st" class:on={i === ((frame % pics.length) + pics.length) % pics.length} onclick={() => (frame = i)}><img src={thumbUrl(p)} alt="" loading="lazy" /></button>
            {/each}
          </div>
        {/if}
      {/if}
    {/if}
  </Card>

  <Card pad="sm" style="flex: 0 0 320px; max-width: 100%">
    <div class="title">{node ? node.shown : 'Nothing selected'}</div>
    <div class="where faint">{node ? (node.path || (prefix(node.id) === 'folder' ? 'a folder of the archive' : '')) : 'Click something in the tree.'}</div>
    <Field label="Name"><div class="row nowrap grow"><div class="grow"><TextInput bind:value={name} size="sm" onenter={doRename} disabled={!node || prefix(node.id) === 'kind'} /></div><Button size="sm" onclick={doRename} disabled={!node || prefix(node.id) === 'kind'}>Rename</Button></div></Field>
    <Field label="Move into"><div class="row nowrap grow"><div class="grow"><Select bind:value={moveTo} options={moveTargets} placeholder="a folder…" size="sm" disabled={!moveTargets.length} /></div><Button size="sm" onclick={doMove} disabled={!moveTargets.length}>Move</Button></div></Field>
    {#if mergeTargets.length}
      <Field label="Merge into"><div class="row nowrap grow"><div class="grow"><Select bind:value={mergeTo} options={mergeTargets} placeholder="another of the same kind…" size="sm" /></div><Button size="sm" onclick={doMerge}>Merge</Button></div></Field>
    {/if}
    <Field label="New folder here"><div class="row nowrap grow"><div class="grow"><TextInput bind:value={folderName} placeholder="folder name…" size="sm" onenter={doFolder} /></div><Button size="sm" onclick={doFolder} disabled={!node}>Make</Button></div></Field>
    <div class="row">
      <Button size="sm" kind="danger" onclick={doDelete} disabled={!node || prefix(node.id) === 'kind'}>Delete</Button>
      <span class="hint small">Deleting moves it to the trash, where Restore brings it back.</span>
    </div>
    <Status text={archive.status} tone="ink" size="xs" />
    <div class="trash">
      <span class="caps">Trash</span>
      {#if !archive.trash.length}<p class="hint small">Empty.</p>{/if}
      {#each archive.trash as t (t.id)}
        <div class="trow"><span class="ellipsis dim">{t.name}</span><Button size="sm" kind="ghost" onclick={() => restore(t.id)}>Restore</Button></div>
      {/each}
      {#if archive.trash.length}<div><Button size="sm" kind="ghost" onclick={empty}>{emptyArmed && Date.now() - emptyArmed < 6000 ? 'Click again to empty' : 'Empty the trash'}</Button></div>{/if}
    </div>
  </Card>
</div>

<style>
  .arch { display: flex; gap: 14px; flex: 1; min-width: 0; min-height: 0; flex-wrap: wrap; align-content: flex-start; overflow-y: auto; }
  .tree { display: flex; flex-direction: column; gap: 1px; }
  .head { display: flex; align-items: baseline; gap: 10px; }
  .title { font-weight: 700; font-size: var(--fs-md); }
  .small { font-size: var(--fs-xs); }
  .where { font-size: var(--fs-xs); word-break: break-all; }
  .video { width: 100%; max-height: 70vh; border-radius: var(--r-md); background: #0e0d0c; }
  .big { padding: 0; border: none; background: transparent; cursor: zoom-in; display: block; margin: 0 auto; max-width: 100%; }
  .big img { max-width: 100%; max-height: 60vh; display: block; margin: 0 auto; border-radius: var(--r-md); }
  .stage { background: #0e0d0c; border-radius: var(--r-md); padding: 8px; display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .nav { display: flex; align-items: center; gap: 8px; }
  .nb { width: 34px; height: 28px; border-radius: var(--r-md); border: 1px solid var(--arc-line); background: var(--arc-surface-2); color: var(--arc-ink); font-size: 14px; padding: 0; }
  .strip { display: flex; gap: 4px; overflow-x: auto; padding: 4px 0; }
  .st { flex: 0 0 auto; width: 64px; height: 64px; padding: 0; border: 2px solid transparent; border-radius: var(--r-sm); background: var(--arc-surface-2); overflow: hidden; }
  .st.on { border-color: var(--tab-archive); }
  .st img { width: 100%; height: 100%; object-fit: cover; }
  .trash { border-top: 1px solid var(--arc-line); padding-top: 8px; display: flex; flex-direction: column; gap: 4px; }
  .trow { display: flex; align-items: center; justify-content: space-between; gap: 6px; font-size: var(--fs-sm); }
</style>
