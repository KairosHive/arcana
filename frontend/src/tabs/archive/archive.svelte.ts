// The archive: everything kept, as one tree by kind, and the trash. Names and
// folders live in the server's catalogue; this holds the tree the page shows
// and the selection.

import { get, post, rpc, reason } from '../../lib/api/client'
import { toast } from '../../lib/stores/toast.svelte'

export type Node = { id: string; kind: string; shown: string; count: number; path: string; group: string; folders: Node[]; items: Node[] }
export type TrashEntry = { id: string; name: string; kind: string; at: number }
export type Shown = { id: string; kind: string; name: string; path: string; contents: string[]; pictures: string[] }

export const archive = $state({
  kinds: [] as Node[],
  titles: {} as Record<string, string>,
  trash: [] as TrashEntry[],
  sel: null as string | null,
  loading: false,
  status: '',
})

export const ICON: Record<string, string> = { collections: '▦', films: '▶', videos: '▶', stories: '≡', pictures: '▣', searches: '⌕', imports: '↓', exports: '↑' }

export async function loadArchive() {
  archive.loading = true
  try {
    const r = await get('/api/archive')
    archive.kinds = r.kinds; archive.titles = r.titles; archive.trash = r.trash
  } catch (e) {
    archive.status = reason(e)
  } finally {
    archive.loading = false
  }
}

/** Every node by id. */
export function flat(): Record<string, Node> {
  const out: Record<string, Node> = {}
  const walk = (n: Node) => { out[n.id] = n; n.folders?.forEach(walk); n.items?.forEach(walk) }
  archive.kinds.forEach(walk)
  return out
}

export function prefix(id: string) { return id.split(':', 1)[0] }

/** A row takes drops when it is a kind, a folder, or a collection. */
export function dropTarget(n: Node) { return ['kind', 'folder', 'collection'].includes(prefix(n.id)) }

export async function op(name: string, args: Record<string, unknown>): Promise<string> {
  try {
    const r = await rpc(name, args)
    await loadArchive()
    return r.text.trim()
  } catch (e) {
    const msg = reason(e)
    toast(msg, 'bad')
    return msg
  }
}

export async function rename(id: string, name: string): Promise<string> {
  try {
    const r = await post('/api/archive/rename', { id, name })
    await loadArchive()
    archive.sel = r.id
    return r.message.trim()
  } catch (e) {
    const msg = reason(e)
    toast(msg, 'bad')
    return msg
  }
}

export async function show(id: string): Promise<Shown> {
  return get<Shown>('/api/archive/show', { id })
}
