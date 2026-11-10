// The shell's shared state: which tab, which dataset, the boards, the active
// collection, settings, the running job, and the two-second state poll that
// keeps all of it in step with the terminal and agents.

import { get, post, reason, ApiError } from '../api/client'
import { load, save } from './persist'
import { toast } from './toast.svelte'
import { EMPTY, type Capabilities } from './capabilities'

export type Mode = 'datasets' | 'prompt' | 'story' | 'moodboard' | 'alchemy' | 'generate' | 'archive' | 'agent'
export const MODES: { value: Mode; label: string; hue: string }[] = [
  { value: 'datasets', label: 'Datasets', hue: 'var(--tab-datasets)' },
  { value: 'prompt', label: 'Prompt Search', hue: 'var(--tab-search)' },
  { value: 'story', label: 'Generate Story', hue: 'var(--tab-story)' },
  { value: 'moodboard', label: 'Moodboard', hue: 'var(--tab-moodboard)' },
  { value: 'alchemy', label: 'Alchemy', hue: 'var(--tab-alchemy)' },
  { value: 'generate', label: 'Generate', hue: 'var(--tab-generate)' },
  { value: 'archive', label: 'Archive', hue: 'var(--tab-archive)' },
  { value: 'agent', label: 'Agent', hue: 'var(--tab-agent)' },
]

export type DatasetOption = { label: string; value: string; name: string; modality: 'image' | 'audio' | 'video'; dim: number }
export type BoardRow = { slug: string; name: string; author: string; count: number; missing: number; sections: number; updated: number; dataset: string; kind?: 'board' | 'set' }
export type Detail = { role?: string | null; kind?: string; section?: string | null; note?: string | null; derived_from?: string | null; handle?: string | null; missing?: boolean }
export type Collection = { slug: string | null; name: string; items: string[]; details: Record<string, Detail> }
export type Settings = { pixels: boolean; collection: string | null }
export type Job = { id: string; kind: string; label: string; status: string; error: string; fraction: number | null; message: string; detail: string; done: number; total: number; log: string[]; result?: any; result_text?: string | null }
export type Hardware = { line: string; note: string; gpu: boolean }

const VALID: Mode[] = MODES.map((m) => m.value)
const savedMode = load<Mode>('mode', 'datasets')

export const app = $state({
  ready: false,
  error: '',
  version: '',
  mode: (VALID.includes(savedMode) ? savedMode : 'datasets') as Mode,
  datasets: [] as DatasetOption[],
  dataset: load<string>('dataset', '') as string,
  boards: [] as BoardRow[],
  collection: { slug: null, name: '', items: [], details: {} } as Collection,
  settings: { pixels: false, collection: null } as Settings,
  hardware: { line: '', note: '', gpu: false } as Hardware,
  signature: '',
  job: null as Job | null,
  headerFolded: load<boolean>('folded', false),
  capabilities: EMPTY as Capabilities,
})

export const current = {
  get dataset(): DatasetOption | null {
    return app.datasets.find((d) => d.value === app.dataset) ?? null
  },
  get modality(): 'image' | 'audio' | 'video' {
    return this.dataset?.modality ?? 'image'
  },
  get mode() { return MODES.find((m) => m.value === app.mode)! },
}

export function setMode(m: Mode) {
  app.mode = m
  save('mode', m)
}

export function setDataset(v: string) {
  app.dataset = v
  save('dataset', v)
}

export function foldHeader(folded: boolean) {
  app.headerFolded = folded
  save('folded', folded)
  window.dispatchEvent(new Event('resize'))
}

function applyDatasets(list: DatasetOption[]) {
  app.datasets = list
  if (app.dataset && !list.some((d) => d.value === app.dataset)) {
    // The dataset this browser remembered is gone (deleted, moved): fall back
    // rather than show a menu pointing at nothing.
    setDataset(list[0]?.value ?? '')
  }
}

export async function bootstrap() {
  try {
    const b = await get('/api/bootstrap')
    app.version = b.version
    applyDatasets(b.datasets)
    app.boards = b.boards
    app.collection = b.collection
    app.settings = b.settings
    app.hardware = b.hardware
    app.signature = b.signature
    app.job = b.job
    if (b.capabilities) app.capabilities = b.capabilities
    app.ready = true
    startPolling()
  } catch (e) {
    app.error = reason(e)
  }
}

/** Re-read the dataset menu after a job built or removed one. */
export async function refreshDatasets() {
  try {
    const b = await get('/api/bootstrap')
    applyDatasets(b.datasets)
  } catch (e) {
    toast(reason(e), 'bad')
  }
}

let timer: ReturnType<typeof setInterval> | null = null
let polling = false
const listeners = new Set<(sig: string) => void>()

/** Called whenever the state signature moves: boards, collection or settings changed. */
export function onStateChange(fn: (sig: string) => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export async function pollState() {
  if (polling) return
  polling = true
  try {
    const s = await get('/api/state', { since: app.signature })
    app.job = s.job
    if (s.signature !== app.signature) {
      app.signature = s.signature
      if (s.boards) app.boards = s.boards
      if (s.collection) app.collection = s.collection
      if (s.settings) app.settings = s.settings
      for (const fn of listeners) fn(s.signature)
    }
  } catch (e) {
    // A 403 means the server restarted and minted a new token: this page's
    // token is stale, and a reload is the only way to get the new one.
    if (e instanceof ApiError && e.status === 403) location.reload()
    /* otherwise the server may be restarting; the next tick tries again */
  } finally {
    polling = false
  }
}

function startPolling() {
  if (timer) return
  timer = setInterval(pollState, 2000)
  document.addEventListener('visibilitychange', () => { if (!document.hidden) pollState() })
}

/** After a write this page made: fetch the new state at once instead of waiting a tick. */
export async function touch() {
  await pollState()
}

// ─────────────────────────── the collection, from anywhere ───────────────────────────

export async function addToCollection(items: string[]) {
  try {
    const c = await post('/api/collection/add', { items })
    app.collection = { slug: c.slug, name: c.name, items: c.items, details: c.details }
    toast(items.length === 1 ? 'Image added to the collection' : `${items.length} added to the collection`, 'ok', 2200)
    void pollState()
  } catch (e) {
    toast(reason(e), 'bad')
  }
}

export async function removeFromCollection(items: string[]) {
  try {
    app.collection = await post('/api/collection/remove', { items })
    void pollState()
  } catch (e) {
    toast(reason(e), 'bad')
  }
}

export async function clearCollection() {
  try {
    app.collection = await post('/api/collection/clear')
    void pollState()
  } catch (e) {
    toast(reason(e), 'bad')
  }
}

export async function useCollection(slug: string) {
  try {
    app.collection = await post('/api/collection/use', { slug })
    void pollState()
  } catch (e) {
    toast(reason(e), 'bad')
  }
}

export async function newCollection(name: string) {
  const c = await post('/api/collection/new', { name })
  app.collection = c
  void pollState()
  return c
}

export async function setPixels(on: boolean) {
  await post('/api/settings', { pixels: on })
  app.settings = { ...app.settings, pixels: on }
  void pollState()
}
