export type Tone = 'ok' | 'warn' | 'bad' | 'info'
export type ToastItem = { id: number; text: string; tone: Tone }

let next = 1
export const toasts = $state<{ list: ToastItem[] }>({ list: [] })

export function toast(text: string, tone: Tone = 'info', ms = 4500) {
  const id = next++
  toasts.list.push({ id, text, tone })
  if (ms > 0) setTimeout(() => dismiss(id), ms)
  return id
}
export function dismiss(id: number) {
  const i = toasts.list.findIndex((t) => t.id === id)
  if (i >= 0) toasts.list.splice(i, 1)
}
