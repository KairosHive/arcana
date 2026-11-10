// Sets, IP labels and LoRAs: what the Search tab gathers and Alchemy learns from.
//
// A set is a board of kind "set" (pictures gathered by searches). An IP label is
// a named position in IP-Adapter space measured on a set, on words or on one
// picture. A LoRA is trained on a set. Search saves into sets and searches with
// labels; Alchemy makes labels and LoRAs from sets and draws with both.

import { get, post, reason } from '../api/client'
import { watchJob } from './jobs.svelte'
import { toast } from './toast.svelte'
import type { Job } from './app.svelte'

export type SetRow = { slug: string; name: string; count: number; missing: number; updated: number; dataset: string }
export type SetQuery = { text: string; dataset: string; k: number; added: number; when: number }
export type LabelRow = { slug: string; name: string; origin: Record<string, any>; members: number; pictures: number; created: number; nearest: string[]; tightness: number | null; error?: string }
export type Checkpoint = { name: string; step: number | null; sheet: string | null; trigger: string | null; when: number }
export type AxisRow = { slug: string; name: string; pos: string | null; neg: string | null; apart: number | null; created: number }
export type LoraRun = { name: string; checkpoints: Checkpoint[]; when: number }

export const learn = $state({
  sets: [] as SetRow[],
  labels: [] as LabelRow[],
  axes: [] as AxisRow[],
  loras: [] as LoraRun[],
  gpu: false,
  /** The running label or LoRA job, shown where it was started. */
  labelBusy: false,
  loraJob: null as Job | null,
  /** Bumped when labels or LoRAs change, so Alchemy re-reads the knob options. */
  rev: 0,
  /** A label Alchemy asked Search to search with; Search takes it and clears it. */
  searchLabel: '',
})

export async function loadSets() {
  try { learn.sets = (await get('/api/sets')).sets } catch (e) { toast(reason(e), 'bad') }
}

export async function loadLabels() {
  try { learn.labels = (await get('/api/labels')).labels } catch (e) { toast(reason(e), 'bad') }
}

export async function loadLoras() {
  try { const r = await get('/api/loras'); learn.loras = r.runs; learn.gpu = r.gpu } catch (e) { toast(reason(e), 'bad') }
}

export async function loadAxes() {
  try { learn.axes = (await get('/api/axes')).axes } catch (e) { toast(reason(e), 'bad') }
}

export async function loadAll() {
  await Promise.all([loadSets(), loadLabels(), loadLoras(), loadAxes()])
}

/** Fit a concept axis from two ends: set:<id>, label:<id> or words:<words> (matched in `dataset`). */
export async function makeAxis(name: string, pos: string, neg: string, dataset = ''): Promise<AxisRow | null> {
  learn.labelBusy = true
  try {
    const r = await post('/api/axes/make', { name, pos, neg, dataset })
    learn.axes = r.axes; learn.rev++
    toast(`Axis "${name}" fitted, ${r.axis.apart} spreads apart`, 'ok', 2600)
    return r.axis
  } catch (e) { toast(reason(e), 'bad'); return null } finally { learn.labelBusy = false }
}

export async function deleteAxis(slug: string) {
  try { learn.axes = (await post('/api/axes/delete', { slug })).axes; learn.rev++ } catch (e) { toast(reason(e), 'bad') }
}

/** Pictures into a set, made when new. Returns the message. */
export async function saveToSet(name: string, items: string[], query: { text?: string; dataset?: string; k?: number } = {}, slug?: string): Promise<string> {
  const r = await post('/api/sets/save', { name, slug, items, query: query.text, dataset: query.dataset, k: query.k })
  learn.sets = r.sets
  return r.message
}

export async function setShow(slug: string): Promise<{ items: string[]; missing: string[]; queries: SetQuery[]; name: string }> {
  return await get(`/api/sets/${encodeURIComponent(slug)}`)
}

export async function removeFromSet(slug: string, items: string[]) {
  const r = await post('/api/sets/remove', { set: slug, items })
  learn.sets = r.sets
  return r
}

export async function exportSet(slug: string, folder = ''): Promise<string> {
  const r = await post('/api/sets/export', { set: slug, folder })
  return `${r.copied} files copied to ${r.folder}.`
}

export async function setKind(slug: string, kind: 'board' | 'set') {
  learn.sets = (await post('/api/sets/kind', { slug, kind })).sets
}

/** Measure a label: a second from a ViT-H index, longer when pictures are encoded. */
export async function makeLabel(args: { name: string; set?: string; words?: string; dataset?: string; n?: number; picture?: string; crops?: number }): Promise<LabelRow | null> {
  learn.labelBusy = true
  try {
    const r = await post('/api/labels/make', args)
    await loadLabels(); learn.rev++
    toast(`IP label "${args.name}" made`, 'ok', 2400)
    return r.label ?? null
  } catch (e) { toast(reason(e), 'bad'); return null } finally { learn.labelBusy = false }
}

export async function deleteLabel(slug: string) {
  try { learn.labels = (await post('/api/labels/delete', { slug })).labels; learn.rev++ } catch (e) { toast(reason(e), 'bad') }
}

export async function trainLora(args: { name: string; set: string; steps?: number; every?: number }) {
  try {
    const r = await post('/api/loras/train', args)
    watchJob(r.job, (j) => {
      const before = learn.loraJob?.message || ''
      learn.loraJob = j
      // A new checkpoint is written with a "keeping a checkpoint" line: list it at once.
      if (j.message !== before && /checkpoint/.test(before)) { void loadLoras(); learn.rev++ }
    }, async (j) => {
      learn.loraJob = j
      await loadLoras(); learn.rev++
      if (j.status === 'done') toast(`LoRA "${args.name}" trained`, 'ok', 3000)
      else if (j.status === 'failed') toast(j.error || 'Training failed.', 'bad')
    }, 1500)
  } catch (e) { toast(reason(e), 'bad') }
}
