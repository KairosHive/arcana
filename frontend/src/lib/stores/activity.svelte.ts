// What an agent (or the terminal) just did, shown as it happens.
//
// The server keeps a short list of the ops that ran and pushes a nudge down
// an event stream the moment the list or the state changes. This store
// listens, fetches the new entries, and lets a tab show one: a search run by
// an agent opens in Prompt Search with its results, a board change opens the
// Moodboard, something rendered opens the Archive.

import { get, session } from '../api/client'
import { app, pollState, setDataset, setMode } from './app.svelte'
import { load, save } from './persist'
import { alchemyUi, showOp } from '../../tabs/alchemy/alchemy.svelte'

export type Activity = { seq: number; t: number; actor: string; op: string; args: Record<string, any>; summary: string; result: Record<string, any> }

export const activity = $state({
  items: [] as Activity[],
  latest: 0,
  /** follow the agent: show each thing it does as it does it */
  follow: load<boolean>('follow-agent', false),
  /** a search for the Prompt Search tab to run: bumped `n` means "do this now" */
  show: { n: 0, text: '', across: false } as { n: number; text: string; across: boolean },
  live: false,
})

export function setFollow(on: boolean) { activity.follow = on; save('follow-agent', on) }

/** The words of an op as a person would type it, short. */
export function line(a: Activity): string {
  const bits: string[] = [a.op]
  for (const [k, v] of Object.entries(a.args)) {
    if (v === null || v === '' || k === 'background' || k === 'spec') continue
    const s = Array.isArray(v) ? v.join(', ') : String(v)
    bits.push(['dataset', 'handles', 'board', 'target', 'picture'].includes(k) ? s : `${k}: ${s}`)
  }
  const out = bits.join('  ')
  return out.length > 140 ? out.slice(0, 137) + '…' : out
}

function datasetOption(id: string) {
  const [name, kind] = id.includes('@') ? id.split('@') : [id, 'image']
  return app.datasets.find((d) => d.name === name && d.modality === kind)
}

/** Open what an op did in the tab that shows it. Returns false when there is nothing to open. */
export function show(a: Activity): boolean {
  if (a.op === 'search text' || a.op === 'search facets' || a.op === 'search across') {
    const ds = datasetOption(String(a.args.dataset ?? ''))
    if (ds) setDataset(ds.value)
    const text = (a.args.phrase ?? a.args.text ?? [])[0] ?? ''
    setMode('prompt')
    activity.show = { n: activity.show.n + 1, text, across: a.op === 'search across' }
    return true
  }
  if (a.op.startsWith('board ') || a.op === 'colour harmonize') { setMode('moodboard'); return true }
  if (a.op.startsWith('dataset ')) { setMode('datasets'); return true }
  if (a.op.startsWith('film ') || a.op.startsWith('alchemy film')) { setMode('alchemy'); alchemyUi.sub = 'video'; return true }
  if (a.op.startsWith('alchemy recipe')) { setMode('alchemy'); return true }
  if (a.op === 'alchemy preview' || (a.op === 'alchemy make' && !a.result?.path)) {
    setMode('alchemy'); void showOp(a.args); return true
  }
  if (a.result?.path || ['pattern make', 'sequence make', 'colour video', 'colour transfer', 'style transfer', 'generate', 'alchemy make'].includes(a.op)) {
    setMode('archive'); return true
  }
  return false
}

async function refresh() {
  try {
    const r = await get<{ items: Activity[]; latest: number }>('/api/activity', { after: activity.latest })
    if (!r.items.length) { activity.latest = r.latest; return }
    activity.items = [...activity.items, ...r.items].slice(-60)
    activity.latest = r.latest
    const mine = r.items.filter((i) => i.actor !== 'person')
    if (activity.follow && mine.length) show(mine[mine.length - 1])
  } catch { /* the next nudge tries again */ }
}

let source: EventSource | null = null

/** Listen for the server's nudges. Safe to call twice. */
export async function listen() {
  if (source) return
  try {
    const first = await get<{ items: Activity[]; latest: number }>('/api/activity', { after: 0 })
    activity.items = first.items.slice(-60)
    activity.latest = first.latest
    const { ui } = await session()
    source = new EventSource(`/api/events?ui=${encodeURIComponent(ui)}`)
    source.onopen = () => { activity.live = true }
    source.onmessage = () => { void refresh(); void pollState() }
    source.onerror = () => { activity.live = false }      // the browser reconnects by itself; the poll covers the gap
  } catch { /* no stream: the two-second poll still runs */ }
}
