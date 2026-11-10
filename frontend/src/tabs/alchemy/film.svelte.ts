// The film: a spec the page edits, kept in this browser between visits, and
// what the server says about it (length, lanes on offer, labels). Every edit
// goes through `commit`, which stores the spec and asks for fresh meta.

import { post, reason } from '../../lib/api/client'
import { load, save } from '../../lib/stores/persist'
import { toast } from '../../lib/stores/toast.svelte'

export type Key = { t: number; ease?: string; v?: number; ref?: string; settings?: Record<string, any>; in?: number; out?: number; speed?: number; mute?: boolean; gain?: number }
export type FilmSpec = {
  name: string; fps: number; size: number; aspect: string; space: string; loop: boolean; even: boolean; length?: number
  temporal?: string; key_every?: number; sound?: boolean; audio?: Record<string, any>[]
  pictures: Key[]; looks: Key[]; lanes: Record<string, Key[]>
}
export type LaneMeta = { label: string; lo: number; hi: number; step: number; group: string; doc?: string }
export type LookSummary = { index: number | null; count: number; t?: number; method?: string; model?: string | null; mask?: string; materials?: string[]; finish?: string[]; settings?: Record<string, any> }
export type FilmMeta = {
  length: number; lanes: Record<string, LaneMeta>; laneOptions: { label: string; value: string; group: string }[]
  pictures: { t: number; path: string; name: string; film?: string; clip?: boolean; dur?: number; in?: number; out?: number; speed?: number }[]; looks: { t: number; label: string }[]
  ok: boolean; status: string; empty: string; normalised: any
}
export type Selection = { lane: string; i: number } | null

export const EMPTY_FILM: FilmSpec = { name: 'untitled film', fps: 30, size: 640, aspect: 'auto', space: 'DINOv2', loop: true, even: true, pictures: [], looks: [], lanes: {} }
export const EASES = ['linear', 'smooth', 'in', 'out', 'hold']
export const SPACES: Record<string, string> = { DINOv2: 'one new surface: the decoder redraws, so materials fuse', 'FLUX.2 VAE': "the photographs' own grain: a mix reads as layers" }
export const SHAPES = [
  { key: 'even', label: 'even', doc: 'every passage as long' },
  { key: 'faster', label: 'speeding up', doc: 'each passage shorter than the one before' },
  { key: 'slower', label: 'slowing down', doc: 'each passage longer than the one before' },
  { key: 'smooth', label: 'slow at both ends', doc: 'slow to start and to end' },
]

export const film = $state({
  spec: normalise(load<FilmSpec>('film.spec', EMPTY_FILM)),
  meta: null as FilmMeta | null,
  t: 0,
  sel: null as Selection,
  playing: false,
  status: '',
  loadedAt: 0,
})

function normalise(s: Partial<FilmSpec>): FilmSpec {
  return { ...EMPTY_FILM, ...s, pictures: s.pictures ?? [], looks: s.looks ?? [], lanes: s.lanes ?? {} }
}

export function sorted(keys: Key[]): Key[] { return [...(keys || [])].sort((a, b) => a.t - b.t) }

let metaTimer: ReturnType<typeof setTimeout> | null = null
let metaSeq = 0

export async function refreshMeta() {
  const seq = ++metaSeq
  try {
    const m = await post<FilmMeta>('/api/film/meta', { spec: film.spec })
    if (seq === metaSeq) film.meta = m
  } catch (e) {
    film.status = reason(e)
  }
}

/** Store a new spec and ask for its meta. */
export function commit(spec: FilmSpec, sel?: Selection) {
  film.spec = normalise(spec)
  if (sel !== undefined) film.sel = sel
  save('film.spec', film.spec)
  if (metaTimer) clearTimeout(metaTimer)
  metaTimer = setTimeout(refreshMeta, 80)
}

export function keysOf(spec: FilmSpec, lane: string): Key[] | undefined {
  if (lane === 'pictures' || lane === 'looks') { spec[lane] = sorted(spec[lane]); return spec[lane] }
  const k = spec.lanes[lane]
  if (k) k.sort((a, b) => a.t - b.t)
  return k
}

export function clone<T>(x: T): T { return JSON.parse(JSON.stringify(x)) }

export function ease(name: string | undefined, s: number): number {
  s = Math.min(Math.max(s, 0), 1)
  switch (name) {
    case 'smooth': return s * s * (3 - 2 * s)
    case 'in': return s * s
    case 'out': return 1 - (1 - s) * (1 - s)
    case 'hold': return 0
    default: return s
  }
}

export function laneValue(keys: Key[], t: number): number {
  if (!keys.length) return 0
  if (t <= keys[0].t) return keys[0].v ?? 0
  for (let i = 0; i + 1 < keys.length; i++) {
    const a = keys[i], b = keys[i + 1]
    if (t < b.t) return (a.v ?? 0) + ((b.v ?? 0) - (a.v ?? 0)) * ease(a.ease, (t - a.t) / Math.max(b.t - a.t, 1e-9))
  }
  return keys[keys.length - 1].v ?? 0
}

// ─────────────────────────── edits that need the engine ───────────────────────────

export async function addPicture(ref: string, t: number = film.t) {
  try {
    const r = await post('/api/film/add', { spec: film.spec, ref, t })
    commit(r.spec); film.t = r.t
  } catch (e) { toast(reason(e), 'bad') }
}

export const FILM_REF = 'film:'
export const isFilmRef = (ref?: string) => String(ref ?? '').startsWith(FILM_REF)
/** A video on the picture lane: a video dataset's handle, a video file or a section of one. */
export const isClipRef = (ref?: string) => {
  const r = String(ref ?? '')
  return !isFilmRef(r) && (/^[^\\/]+@video:\d+$/.test(r) || /\.(mp4|mov|mkv|webm|avi|m4v)(#t=[\d.,]+)?$/i.test(r))
}
/** A key that plays for a while, a kept film or a clip. */
export const playsRef = (ref?: string) => isFilmRef(ref) || isClipRef(ref)

export const LANE_SHAPES = [
  { key: 'rise', label: 'rise', doc: 'from the first value to the second' },
  { key: 'fall', label: 'fall', doc: 'from the second value back to the first' },
  { key: 'swell', label: 'swell', doc: 'up to the second value at the middle, and back' },
  { key: 'dip', label: 'dip', doc: 'down to the first value at the middle, and back' },
  { key: 'wave', label: 'wave', doc: 'back and forth, a number of times' },
  { key: 'steps', label: 'steps', doc: 'in a number of steps, each held, then a cut' },
]

/** A whole curve on a lane between two times, in place of its keys there. */
export async function shapeLane(path: string, shape: string, lo: number, hi: number, start: number, end: number, count: number): Promise<string> {
  try {
    commit((await post('/api/film/shape', { spec: film.spec, path, shape, lo, hi, start, end, count })).spec, null)
    return ''
  } catch (e) { return reason(e) }
}

/** Pictures laid one after another from a time, keys already after it moved later. */
export async function placeSeries(refs: string[], t: number, every: number): Promise<string> {
  try {
    const r = await post('/api/film/series', { spec: film.spec, refs, t, every })
    commit(r.spec, null)
    return ''
  } catch (e) { return reason(e) }
}

export async function spacePictures(shape: string, by: 'time' | 'change' = 'change'): Promise<string> {
  try { commit((await post('/api/film/space', { spec: film.spec, shape, by })).spec); return '' } catch (e) { toast(reason(e), 'bad'); return reason(e) }
}

export async function addLane(path: string) {
  try {
    const r = await post('/api/film/lane', { spec: film.spec, path, t: film.t })
    commit(r.spec, { lane: path, i: 0 })
  } catch (e) { toast(reason(e), 'bad') }
}

export async function addLook(controls: Record<string, any>): Promise<string> {
  try {
    const r = await post('/api/film/look', { spec: film.spec, t: film.t, controls })
    commit(r.spec)
    toast(r.message, 'ok')
    return r.message
  } catch (e) { toast(reason(e), 'bad'); return reason(e) }
}

export async function replaceLook(controls: Record<string, any>): Promise<string> {
  try {
    const r = await post('/api/film/look-replace', { spec: film.spec, t: film.t, controls })
    commit(r.spec)
    return r.message
  } catch (e) { return reason(e) }
}

export async function removeLook(): Promise<string> {
  try {
    const r = await post('/api/film/look-remove', { spec: film.spec, t: film.t })
    commit(r.spec)
    return r.message
  } catch (e) { return reason(e) }
}

export async function lookSummary(): Promise<LookSummary> {
  return post('/api/film/look-summary', { spec: film.spec, t: film.t })
}

/** A picture dropped on the timeline at a time of its own. */
export function dropPicture(ref: string, t: number) {
  const s = clone(film.spec)
  const pics = keysOf(s, 'pictures')!
  while (pics.some((p) => Math.abs(p.t - t) < 0.01)) t = Math.round((t + 0.05) * 100) / 100
  const key: Key = { t, ref, ease: 'linear' }
  pics.push(key)
  const sortedKeys = keysOf(s, 'pictures')!
  if (s.length && t >= s.length) delete s.length
  film.t = t
  commit(s, { lane: 'pictures', i: sortedKeys.indexOf(key) })
}

export function removeSelected() {
  const sel = film.sel
  if (!sel) return
  const s = clone(film.spec)
  const keys = keysOf(s, sel.lane)
  if (!keys || !keys[sel.i]) return
  keys.splice(sel.i, 1)
  if (sel.lane !== 'pictures' && sel.lane !== 'looks' && !keys.length) delete s.lanes[sel.lane]
  commit(s, null)
}

export function newFilm() {
  commit(clone(EMPTY_FILM), null)
  film.t = 0; film.loadedAt = Date.now(); film.status = 'A new film.'
}

export function openFilm(spec: FilmSpec, name: string) {
  commit(spec, null)
  film.t = 0; film.loadedAt = Date.now(); film.status = `Opened ${name}.`
}
