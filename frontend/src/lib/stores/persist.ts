// Per-browser conveniences: the open tab, the chosen dataset, column widths.
// Nothing here is state the app depends on; every read falls back.

const PREFIX = 'arcana.v2.'

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    /* a lost preference is not worth an error */
  }
}

export function remove(key: string): void {
  try { localStorage.removeItem(PREFIX + key) } catch { /* ignore */ }
}
