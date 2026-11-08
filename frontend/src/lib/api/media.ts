// URL builders for the media routes. Paths are percent-encoded once; the
// server decodes once.

const AUDIO_EXTS = new Set(['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'])

export function isAudioPath(p: string): boolean {
  const i = p.lastIndexOf('.')
  return i >= 0 && AUDIO_EXTS.has(p.slice(i).toLowerCase())
}

const q = (p: string) => encodeURIComponent(p)

export function thumbUrl(p: string): string { return `/thumb?p=${q(p)}` }
export function previewUrl(p: string, w = 900, quality?: number): string {
  return `/preview?p=${q(p)}&w=${w}` + (quality ? `&q=${quality}` : '')
}
export function previewSrcset(p: string): string {
  return [600, 900, 1400].map((w) => `${previewUrl(p, w)} ${w}w`).join(', ')
}
export function paletteUrl(p: string, n = 16, w = 300, h = 20, db = ''): string {
  return `/palette?p=${q(p)}&n=${n}&w=${w}&h=${h}` + (db ? `&db=${q(db)}` : '')
}
export function audioUrl(p: string): string { return `/audio?p=${q(p)}` }
export function waveUrl(p: string): string { return `/awave?p=${q(p)}` }
export function specUrl(p: string): string { return `/aspec?p=${q(p)}` }
export function audioImageUrl(p: string, spectrogram: boolean): string { return spectrogram ? specUrl(p) : waveUrl(p) }
export function filmVideoUrl(p: string): string { return `/film-video?p=${q(p)}` }
export function downloadUrl(p: string): string { return `/download?p=${q(p)}` }
export function boardPageUrl(slug: string, v: number | string, print = false): string {
  return `/board-page/${encodeURIComponent(slug)}?v=${v}` + (print ? '&print=1' : '')
}

export function basename(p: string): string {
  return p.split(/[\\/]/).pop() || p
}
