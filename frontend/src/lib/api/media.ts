// URL builders for the media routes. Paths are percent-encoded once; the
// server decodes once.

const AUDIO_EXTS = new Set(['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'])

const VIDEO_EXTS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'])

/** A section of a video is its file plus a time range: walk.mp4#t=12.000,22.500. */
export function splitRef(p: string): { file: string; t0: number | null; t1: number | null } {
  const at = p.lastIndexOf('#t=')
  if (at < 0) return { file: p, t0: null, t1: null }
  const [a, b] = p.slice(at + 3).split(',')
  return { file: p.slice(0, at), t0: Number(a), t1: b === undefined ? null : Number(b) }
}

function ext(p: string): string {
  const f = splitRef(p).file
  const i = f.lastIndexOf('.')
  return i >= 0 ? f.slice(i).toLowerCase() : ''
}

export function isAudioPath(p: string): boolean { return AUDIO_EXTS.has(ext(p)) }
export function isVideoPath(p: string): boolean { return VIDEO_EXTS.has(ext(p)) }

/** How long a section lasts, as a person reads it: 12.0 to 22.5 s. */
export function sectionLabel(p: string): string {
  const { t0, t1 } = splitRef(p)
  if (t0 === null || t1 === null) return ''
  return `${t0.toFixed(1)} to ${t1.toFixed(1)} s · ${(t1 - t0).toFixed(1)} s long`
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
/** The video file, with the section as a media fragment, so the player starts and stops on it. */
export function videoUrl(p: string): string {
  const { file, t0, t1 } = splitRef(p)
  return `/video?p=${q(file)}` + (t0 === null ? '' : `#t=${t0}` + (t1 === null ? '' : `,${t1}`))
}
export function waveUrl(p: string): string { return `/awave?p=${q(p)}` }
export function specUrl(p: string): string { return `/aspec?p=${q(p)}` }
export function audioImageUrl(p: string, spectrogram: boolean): string { return spectrogram ? specUrl(p) : waveUrl(p) }
export function filmVideoUrl(p: string): string { return `/film-video?p=${q(p)}` }
export function downloadUrl(p: string): string { return `/download?p=${q(p)}` }
export function boardPageUrl(slug: string, v: number | string, print = false): string {
  return `/board-page/${encodeURIComponent(slug)}?v=${v}` + (print ? '&print=1' : '')
}

export function basename(p: string): string {
  const f = splitRef(p).file
  return f.split(/[\\/]/).pop() || f
}
