// What each feature of the page works with: photos, sounds, videos.
//
// The table comes from the server (arcana/capabilities.py) at bootstrap, and
// the routes refuse by the same table, so a control the page disables and a
// request the server turns down are always the same cases, in the same words.

import { app, current } from './app.svelte'

import type { Modality, Feature } from './capabilities'
export type { Modality, Feature, Capabilities } from './capabilities'

const AUDIO_EXTS = new Set(['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'])
const VIDEO_EXTS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'])

/** The kind of one file, from its extension. Everything that is not a sound or a video is drawn as a picture. */
export function kindOf(path: string): Modality {
  const cut = path.lastIndexOf('#t=')
  const file = cut >= 0 ? path.slice(0, cut) : path
  const i = file.lastIndexOf('.')
  const ext = i >= 0 ? file.slice(i).toLowerCase() : ''
  return AUDIO_EXTS.has(ext) ? 'audio' : VIDEO_EXTS.has(ext) ? 'video' : 'image'
}

function words(mods: Modality[]): string {
  const c = app.capabilities
  const names = c.order.filter((m) => mods.includes(m)).map((m) => c.plural[m])
  if (!names.length) return 'nothing'
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
}

export const caps = {
  feature(id: string): Feature | undefined { return app.capabilities.features.find((f) => f.id === id) },
  /** The modalities a feature works with, in display order. */
  of(id: string): Modality[] { return this.feature(id)?.modalities ?? [] },
  label(m: Modality): string { return app.capabilities.labels[m] ?? m },
  /** Where a modality sits in the fixed display order. */
  order(m: Modality): number { return app.capabilities.order.indexOf(m) },
  /** Whether a feature works with a modality (the chosen dataset's, unless one is given). */
  can(id: string, modality: Modality = current.modality): boolean {
    const f = this.feature(id)
    return !f || f.modalities.includes(modality)
  },
  /** Whether a feature can take this one file. */
  canPath(id: string, path: string): boolean { return this.can(id, kindOf(path)) },
  /** The sentence the server would refuse with, or '' when the feature works. */
  why(id: string, modality: Modality = current.modality): string {
    const f = this.feature(id)
    if (!f || f.modalities.includes(modality)) return ''
    const given = f.scope === 'dataset' ? `This dataset holds ${app.capabilities.plural[modality]}` : `This is a ${modality === 'image' ? 'picture' : modality === 'audio' ? 'sound' : 'clip'}`
    return `${f.label} works with ${words(f.modalities)}. ${given}.`
  },
  /** A tab's modalities, and whether the chosen dataset leaves it with nothing to do. */
  tab(mode: string): { modalities: Modality[]; off: boolean; why: string } {
    const c = app.capabilities
    const mods = c.tabs[mode] ?? []
    const ds = current.dataset
    const off = !!ds && c.datasetTabs.includes(mode) && mods.length > 0 && !mods.includes(ds.modality)
    return { modalities: mods, off, why: off ? `This tab works with ${words(mods)}. This dataset holds ${c.plural[ds!.modality]}.` : '' }
  },
}
