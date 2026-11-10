// The Alchemy step's controls, as one piece of state the three columns share,
// and the catalogue that says what the controls are.

import { get, post, reason } from '../../lib/api/client'
import { load, save } from '../../lib/stores/persist'

export type Knob = { key: string; kind: 'float' | 'int' | 'choice' | 'bool' | 'angle' | 'text'; default: any; lo?: number; hi?: number; step?: number; label: string; doc: string; options?: any[]; when: [string, string, any][]; section: 'main' | 'more' | 'looks'; px?: boolean }
export type Method = { id: string; family: string; label: string; doc: string; cost: 'live' | 'seconds'; materials: [number, number]; picture: boolean; needs: string[]; knobs: Knob[] }
export type Preset = { name: string; group: string; doc: string; coverage: number | null }
export type Refine = { key: string; label: string; lo: number; hi: number; step: number; default: number; doc: string }
export type FinishSpec = { key: string; group: string; label: string; lo: number; hi: number; step: number; doc: string; default: number }
export type Recipe = { name: string; group: string; doc: string; settings: Record<string, any>; own: boolean }
export type Catalog = {
  families: { id: string; label: string; doc: string }[]; methods: Method[]; maskGroups: string[]; maskPresets: Preset[]
  refine: Refine[]; finish: FinishSpec[]; recipes: Recipe[]; roles: { role: string; colour: string; label: string }[]; words: string; custom: string
}

export const WORDS = '__words', CUSTOM = '__custom'
export const ROLE_KEYS = ['T', '1', '2', '3'] as const
export type RoleKey = (typeof ROLE_KEYS)[number]
export const ROLE_COLOUR: Record<RoleKey, string> = { T: 'var(--arc-target)', '1': 'var(--arc-accent)', '2': '#b08d57', '3': '#7c8f8a' }

export type Layer = { choice: string; words: string; coverage: number; combine: 'and' | 'or' | 'but not' }

export const catalog = $state<{ data: Catalog | null; error: string }>({ data: null, error: '' })

/** Which subtab is open, shared so the Archive can open a film in Video and the film can open a look in Image. */
export const alchemyUi = $state({ sub: 'image' as 'image' | 'video' | 'pattern' })

export async function loadCatalog() {
  if (catalog.data) return catalog.data
  try {
    const c = await get<Catalog>('/api/alchemy/catalog')
    // Settle the knobs before the page sees the catalogue, so every control
    // has a value the moment it is drawn.
    const m = c.methods.find((x) => x.id === step.method) ?? c.methods[0]
    step.method = m.id
    step.family = m.family
    step.knobs = settleKnobs(m, step.knobs)
    catalog.data = c
  } catch (e) {
    catalog.error = reason(e)
  }
  return catalog.data
}

export function method(id: string): Method | undefined {
  return catalog.data?.methods.find((m) => m.id === id)
}

export const step = $state({
  roles: load<Record<RoleKey, string | null>>('alchemy.roles', { T: null, '1': null, '2': null, '3': null }),
  w: { '1': 1.0, '2': 1.0, '3': 1.0 } as Record<'1' | '2' | '3', number>,
  lead: 'picture' as 'picture' | 'material',
  zoom: 1.0,
  crop: 0,
  family: 'tokens',
  method: 'tokens.mix',
  knobs: {} as Record<string, any>,
  layers: [emptyLayer(), emptyLayer(), emptyLayer()] as Layer[],
  invert: false,
  refine: { feather: 4, grow: 0, follow: 0, ragged: 0, amount: 1 } as Record<string, number>,
  finish: {} as Record<string, number>,
  scope: 'whole' as 'whole' | 'mask',
  // the stage
  view: 'compare' as 'compare' | 'mask',
  size: 1024,
  live: true,
  keepSize: 0,
})

export function emptyLayer(): Layer { return { choice: 'none', words: '', coverage: 0.5, combine: 'and' } }

export function setRole(r: RoleKey, path: string) {
  step.roles[r] = step.roles[r] === path ? null : path
  save('alchemy.roles', step.roles)
}

export function pruneRoles(items: string[]) {
  const set = new Set(items)
  let changed = false
  for (const r of ROLE_KEYS) if (step.roles[r] && !set.has(step.roles[r]!)) { step.roles[r] = null; changed = true }
  if (changed) save('alchemy.roles', step.roles)
}

/** A knob applies when its `when` clauses hold against the current values. */
export function applies(k: Knob, values: Record<string, any>): boolean {
  for (const [key, op, val] of k.when) {
    const v = values[key]
    if (op === '==' && String(v) !== String(val)) return false
    if (op === '!=' && String(v) === String(val)) return false
    if (op === 'in' && !(val as any[]).includes(v)) return false
  }
  return true
}

/** The method's knobs: defaults, overridden by what is given, each in range. */
export function settleKnobs(m: Method, given: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {}
  for (const k of m.knobs) out[k.key] = k.default
  for (const [key, v0] of Object.entries(given || {})) {
    const k = m.knobs.find((x) => x.key === key)
    if (!k) continue
    let v = v0
    if (k.kind === 'float' || k.kind === 'angle') { v = Number(v); if (k.lo !== undefined && k.hi !== undefined) v = Math.min(Math.max(v, k.lo), k.hi) }
    else if (k.kind === 'int') { v = Math.round(Number(v)); if (k.lo !== undefined && k.hi !== undefined) v = Math.min(Math.max(v, k.lo), k.hi) }
    else if (k.kind === 'choice') { if (!(k.options ?? []).map(String).includes(String(v))) continue; v = (k.options ?? []).find((o) => String(o) === String(v)) }
    else if (k.kind === 'bool') v = ['1', 'true', 'yes', 'on'].includes(String(v).toLowerCase())
    out[key] = v
  }
  return out
}

export function chooseMethod(id: string, seed: Record<string, any> = {}) {
  const m = method(id)
  if (!m) return
  step.method = id
  step.family = m.family
  step.knobs = settleKnobs(m, seed)
}

/** The flat control state the server's settings() reads. */
export function controls(): Record<string, any> {
  const v: Record<string, any> = {
    roles: { ...step.roles }, w1: step.w['1'], w2: step.w['2'], w3: step.w['3'],
    lead: step.lead, zoom: step.zoom, crop: step.crop, method: step.method,
    knobs: { ...step.knobs }, invert: step.invert, finish: { ...step.finish }, scope: step.scope,
  }
  step.layers.forEach((l, i) => {
    v[`layer${i + 1}`] = l.choice; v[`words${i + 1}`] = l.words; v[`cov${i + 1}`] = l.coverage
    if (i > 0) v[`comb${i + 1}`] = l.combine
  })
  for (const [k, val] of Object.entries(step.refine)) v[k] = val
  return v
}

/** A recipe's control values, applied: it sets only what it holds. */
export async function applyRecipe(name: string): Promise<string> {
  const r = await get(`/api/alchemy/recipe/${encodeURIComponent(name)}`)
  if (r.method) chooseMethod(r.method, r.knobs || {})
  if (r.lead) step.lead = r.lead
  if (r.zoom !== undefined) step.zoom = r.zoom
  if (r.finish) { step.finish = { ...step.finish, ...r.finish }; step.scope = r.scope || 'whole' }
  if (r.mask) {
    step.layers = r.mask.layers.map((l: any) => ({ choice: l.choice, words: l.words || '', coverage: l.coverage ?? coverageFor(l.choice), combine: l.combine || 'and' }))
    step.invert = !!r.mask.invert
    step.refine = { ...step.refine, ...r.mask.refine }
  }
  return r.doc || ''
}

/** A whole step (a film's look) into the controls: what it does not set goes back to its default. */
export async function loadStep(settings: Record<string, any>) {
  const r = await post('/api/alchemy/load', { settings })
  if (r.method) chooseMethod(r.method, r.knobs || {})
  step.lead = r.lead ?? 'picture'
  step.zoom = r.zoom ?? 1.0
  step.crop = r.crop ?? 0
  step.finish = r.finish ? Object.fromEntries(Object.entries(r.finish).filter(([k, v]) => v !== catalog.data?.finish.find((f) => f.key === k)?.default)) as Record<string, number> : {}
  step.scope = r.scope || 'whole'
  if (r.mask) {
    step.layers = r.mask.layers.map((l: any) => ({ choice: l.choice, words: l.words || '', coverage: l.coverage ?? coverageFor(l.choice), combine: l.combine || 'and' }))
    step.invert = !!r.mask.invert
    step.refine = { ...step.refine, ...r.mask.refine }
  }
  const mats: string[] = r.materials || []
  ;(['1', '2', '3'] as const).forEach((k, i) => { step.roles[k] = mats[i] ?? null })
  const w: number[] = r.weights || []
  ;(['1', '2', '3'] as const).forEach((k, i) => { step.w[k] = w[i] ?? 1.0 })
  save('alchemy.roles', step.roles)
}

export function coverageFor(choice: string): number {
  const p = catalog.data?.maskPresets.find((x) => x.name === choice)
  return p?.coverage ?? 0.5
}

export function resetFinish() {
  step.finish = {}
  step.scope = 'whole'
}

export const materials = () => (['1', '2', '3'] as const).filter((r) => step.roles[r])
