<script lang="ts">
  import { app, removeFromCollection, clearCollection, useCollection, touch, setPixels } from '../../lib/stores/app.svelte'
  import { upload, rpc, post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { openViewer } from '../../lib/stores/viewer.svelte'
  import { roles, setReference, setTarget, pruneRoles } from './moodboard.svelte'
  import Thumb from '../../lib/ui/Thumb.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'
  import Toggle from '../../lib/ui/Toggle.svelte'
  import Status from '../../lib/ui/Status.svelte'
  import Empty from '../../lib/ui/Empty.svelte'
  import BoardPage from './BoardPage.svelte'

  /**
   * The collection: the pictures every tool draws from, with the two roles
   * [R] and [T] on them, and the board controls under them.
   */
  const items = $derived(app.collection.items)
  $effect(() => { pruneRoles(items) })

  let dragOver = $state(false)
  let boardPick = $state<string | null>(null)
  let boardName = $state('')
  let exportFolder = $state('')
  let status = $state(''), tone = $state<'dim' | 'ok' | 'warn' | 'bad'>('dim')
  let saveStatus = $state('')
  let pageOpen = $state(false)
  let pageSlug = $state('')

  const boardOptions = $derived(app.boards.map((b) => ({ value: b.slug, label: `${b.name}  ·  ${b.count} image${b.count === 1 ? '' : 's'}${b.missing ? `, ${b.missing} missing` : ''}${b.author === 'agent' ? ', by an agent' : ''}` })))
  const picked = $derived(app.boards.find((b) => b.slug === boardPick))
  let pagedSlugs = $state<Record<string, boolean>>({})
  $effect(() => {
    // Which boards carry a designed page, so the View page button shows for them.
    app.signature
    for (const b of app.boards) {
      rpc('board show', { board: b.slug }).then((r) => { pagedSlugs[b.slug] = !!r.result.page }).catch(() => {})
    }
  })

  async function drop(files: FileList | null) {
    dragOver = false
    if (!files?.length) return
    for (const f of Array.from(files)) {
      try { await upload('/api/upload', f) } catch (e) { toast(reason(e), 'bad') }
    }
    await touch()
  }

  async function useBoard() {
    if (!boardPick) return
    try {
      const r = await rpc('board show', { board: boardPick })
      await useCollection(boardPick)
      const b = r.result
      roles.reference = b.reference || null; roles.target = b.transfer || null
      boardName = b.name
      const n = (b.unsectioned?.length ?? 0) + (b.sections ?? []).reduce((a: number, s: any) => a + (s.items?.length ?? 0), 0)
      const gone = b.metrics?.missing ?? 0
      status = `Working in “${b.name}”, ${n} image${n === 1 ? '' : 's'}${b.sections?.length ? ` in ${b.sections.length} section${b.sections.length === 1 ? '' : 's'}` : ''}${b.author === 'agent' ? ', made by an agent' : ''}` + (gone ? `. ${gone} could not be found; relocate the dataset if the files moved.` : '.')
      tone = 'ok'
    } catch (e) { status = `That board could not be used: ${reason(e)}`; tone = 'bad' }
  }

  async function saveCopy() {
    const name = boardName.trim()
    if (!name) { saveStatus = 'Give the board a name first.'; return }
    if (!items.length) { saveStatus = 'Nothing in the collection to save.'; return }
    const args: Record<string, unknown> = { name, items, reference: roles.reference || '', transfer: roles.target || '', dataset: app.dataset.split('::')[0] || '' }
    if (picked && picked.name === name) args.id = picked.slug
    try {
      const r = await rpc('board save', args)
      saveStatus = `Saved “${name}”, ${items.length} image${items.length === 1 ? '' : 's'}.`
      boardPick = r.result.board.slug
      await touch()
    } catch (e) { saveStatus = `Could not save: ${reason(e)}` }
  }

  async function exportCopies() {
    if (!items.length) return
    const folder = exportFolder.trim()
    if (!folder) { saveStatus = 'Please enter a folder name'; return }
    try {
      const r = await rpc('item export', { items, folder, kind: 'collections' })
      saveStatus = `Saved ${r.result.copied} images to ${folder}`
    } catch (e) { saveStatus = reason(e) }
  }

  async function deleteBoard() {
    if (!picked) return
    if (!confirm(`Delete the board “${picked.name}”? This can be undone with the undo op.`)) return
    try {
      await rpc('board delete', { board: picked.slug })
      status = `Deleted “${picked.name}”.`; tone = 'ok'
      boardPick = null
      await touch()
    } catch (e) { status = `Could not delete it: ${reason(e)}`; tone = 'bad' }
  }

  async function togglePixels(on: boolean) {
    try {
      await setPixels(on)
      status = on ? 'Agents may now see thumbnails of your pictures.' : 'Agents work blind again: no picture leaves this machine.'; tone = 'dim'
    } catch (e) { status = `Could not change it: ${reason(e)}`; tone = 'bad' }
  }

  function kindBadge(path: string) {
    const d = app.collection.details[path]
    return d?.kind === 'derived' || d?.kind === 'generated' ? d.kind.toUpperCase() : ''
  }
</script>

<div class="rail" class:over={dragOver} role="region" aria-label="Collection"
     ondragover={(e) => { e.preventDefault(); dragOver = true }} ondragleave={() => (dragOver = false)}
     ondrop={(e) => { e.preventDefault(); drop(e.dataTransfer?.files ?? null) }}>
  <div class="head">
    <span class="caps title">Collection{#if app.collection.name} · {app.collection.name}{/if}</span>
    <span class="count dim">{items.length}</span>
    <label class="add" title="Add a picture from outside the datasets">
      + add images
      <input type="file" accept="image/*" multiple onchange={(e) => drop((e.target as HTMLInputElement).files)} />
    </label>
  </div>

  <div class="gallery">
    {#if !items.length}
      <div class="empty-wrap"><Empty compact title="Nothing here yet." hint="Search for images, then use “+ Collection” to collect the ones you want to work from. Or drop pictures here." /></div>
    {:else}
      {#each items as path, i (path)}
        {@const d = app.collection.details[path]}
        <Thumb {path} size="100%" title={d?.note || path} role={roles.reference === path && roles.target === path ? 'R' : roles.reference === path ? 'R' : roles.target === path ? 'T' : ''}
               badge={kindBadge(path)} onopen={() => openViewer(items, i, app.collection.name)} onremove={() => removeFromCollection([path])}>
          <div class="roles">
            <button class="r" class:on={roles.reference === path} title="Reference: the palette source" onclick={() => setReference(path)}>R</button>
            <button class="t" class:on={roles.target === path} title="Target: the picture that receives the palette" onclick={() => setTarget(path)}>T</button>
            {#if d?.role}<span class="role-chip">{d.role}</span>{/if}
          </div>
        </Thumb>
      {/each}
    {/if}
  </div>

  <div class="foot">
    <div class="row nowrap">
      <div class="grow"><Select bind:value={boardPick} options={boardOptions} placeholder="Saved boards…" size="sm" /></div>
      <Button size="sm" onclick={useBoard} disabled={!boardPick} title="Make this board the collection">Use</Button>
      {#if boardPick && pagedSlugs[boardPick]}<Button size="sm" kind="primary" onclick={() => { pageSlug = boardPick!; pageOpen = true }}>View page</Button>{/if}
    </div>
    <Status text={status} {tone} size="xs" />
    <Disclosure label="save, export, more" level={2}>
      <div class="row nowrap"><div class="grow"><TextInput bind:value={boardName} placeholder="Name this board…" size="sm" /></div><Button size="sm" onclick={saveCopy}>Save copy</Button></div>
      <div class="row nowrap"><div class="grow"><TextInput bind:value={exportFolder} placeholder="Export folder name…" size="sm" /></div><Button size="sm" onclick={exportCopies} title="Copies the image files out to a folder under output/collections">Export copies</Button></div>
      <Toggle checked={app.settings.pixels} label="Agents may see thumbnails" size="sm" onchange={togglePixels} />
      <div class="row">
        <Button size="sm" kind="danger" onclick={clearCollection} disabled={!items.length}>Clear collection</Button>
        <Button size="sm" kind="danger" onclick={deleteBoard} disabled={!picked}>Delete board</Button>
      </div>
      <Status text={saveStatus} tone="ok" size="xs" />
    </Disclosure>
  </div>
</div>

<BoardPage bind:open={pageOpen} slug={pageSlug} />

<style>
  .rail { display: flex; flex-direction: column; height: 100%; min-height: 0; padding: 10px; background: var(--arc-bg); border: 1px solid var(--arc-line); border-radius: var(--r-lg); gap: 8px; transition: border-color var(--ease); }
  .rail.over { border-color: var(--arc-accent); }
  .head { display: flex; align-items: center; gap: 8px; flex: 0 0 auto; }
  .title { font-weight: 700; letter-spacing: 1px; color: var(--arc-ink-dim); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .count { font-size: var(--fs-xs); }
  .add { padding: 3px 9px; border: 1px dashed var(--arc-line); border-radius: var(--r-md); background: var(--arc-surface); cursor: pointer; font-size: var(--fs-xs); color: var(--arc-ink-dim); flex: 0 0 auto; }
  .add:hover { border-color: var(--arc-ink-dim); color: var(--arc-ink); }
  .add input { display: none; }
  .gallery { flex: 1 1 0; min-height: 0; overflow-y: auto; overflow-x: hidden; display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--thumb), 1fr)); gap: 10px; padding: 6px 2px; align-content: start; }
  .empty-wrap { grid-column: 1 / -1; }
  .roles { display: flex; gap: 3px; align-items: center; }
  .roles button { flex: 0 0 auto; width: 22px; height: 18px; border-radius: var(--r-sm); border: 1px solid var(--arc-line); background: var(--arc-role-off); color: var(--arc-ink-dim); font-size: var(--fs-2xs); font-weight: 800; padding: 0; }
  .roles .r.on { background: var(--arc-accent); border-color: var(--arc-accent); color: var(--arc-bg); }
  .roles .t.on { background: var(--arc-target); border-color: var(--arc-target); color: var(--arc-bg); }
  .role-chip { font-size: var(--fs-2xs); color: var(--arc-ink-faint); margin-left: auto; }
  .foot { flex: 0 0 auto; display: flex; flex-direction: column; gap: 6px; padding-top: 8px; border-top: 1px solid var(--arc-line); }
  .foot .row { gap: 6px; }
</style>
