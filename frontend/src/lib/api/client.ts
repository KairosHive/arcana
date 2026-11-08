// The one way the page talks to Arcana's server.
//
// Every /api call carries the page's token, which the server wrote into
// index.html (window.__ARCANA__). Under `vite` the page comes from another
// port, so the token is fetched once from /api/session, which the server only
// answers while ARCANA_UI_DEV is set.

export type Boot = { ui: string; agent: string; version: string }

declare global {
  interface Window { __ARCANA__?: Boot }
}

let boot: Boot | null = window.__ARCANA__ ?? null
let booting: Promise<Boot> | null = null

export async function session(): Promise<Boot> {
  if (boot) return boot
  if (!booting) {
    booting = fetch('/api/session').then(async (r) => {
      if (!r.ok) throw new ApiError('The page has no token. Start the server with ARCANA_UI_DEV=1 for development.', r.status)
      boot = (await r.json()) as Boot
      return boot
    })
  }
  return booting
}

export class ApiError extends Error {
  status: number
  help: string
  op: string
  constructor(message: string, status = 0, help = '', op = '') {
    super(message)
    this.status = status
    this.help = help
    this.op = op
  }
}

async function headers(extra: Record<string, string> = {}): Promise<Record<string, string>> {
  const b = await session()
  return { 'X-Arcana-UI': b.ui, ...extra }
}

async function parse<T>(r: Response): Promise<T> {
  const text = await r.text()
  let data: any = null
  try { data = text ? JSON.parse(text) : null } catch { data = { error: text } }
  if (!r.ok) throw new ApiError(data?.error || `${r.status} ${r.statusText}`, r.status, data?.help || '', data?.op || '')
  return data as T
}

export async function get<T = any>(path: string, params?: Record<string, string | number | boolean | undefined | null>): Promise<T> {
  let url = path
  if (params) {
    const q = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') q.set(k, String(v))
    const s = q.toString()
    if (s) url += (url.includes('?') ? '&' : '?') + s
  }
  const r = await fetch(url, { headers: await headers() })
  return parse<T>(r)
}

export async function post<T = any>(path: string, body: unknown = {}): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: await headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body ?? {}),
  })
  return parse<T>(r)
}

export async function upload<T = any>(path: string, file: File, fields: Record<string, string> = {}): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  for (const [k, v] of Object.entries(fields)) form.append(k, v)
  const r = await fetch(path, { method: 'POST', headers: await headers(), body: form })
  return parse<T>(r)
}

/** One op, as the person. Returns the result model as JSON plus text and attachments. */
export async function rpc<T = any>(op: string, args: Record<string, unknown> = {}): Promise<{ op: string; result: T; text: string; attachments: Attachment[] }> {
  return post('/api/rpc', { op, args })
}

export type Attachment = { caption: string; mime: string; data: string }

export function attachmentUrl(a: Attachment): string {
  return `data:${a.mime};base64,${a.data}`
}

/** The agent terminal token, for X-Arcana-Term and the SSE query string. */
export async function agentToken(): Promise<string> {
  return (await session()).agent
}

/** A message a person can read out of any thrown thing. */
export function reason(e: unknown): string {
  if (e instanceof ApiError) return e.message
  if (e instanceof Error) return e.message
  return String(e)
}
