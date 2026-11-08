// The full-page picture viewer, opened from any tile. It steps through a list
// the opener gives it (the collection by default) with the arrow keys.

export const viewer = $state({
  open: false,
  items: [] as string[],
  at: 0,
  where: '',
})

export function openViewer(items: string[], at: number, where = '') {
  if (!items.length) return
  viewer.items = items
  viewer.at = Math.max(0, Math.min(items.length - 1, at))
  viewer.where = where
  viewer.open = true
}

export function closeViewer() { viewer.open = false }

export function stepViewer(d: number) {
  const n = viewer.items.length
  if (!n) return
  viewer.at = (viewer.at + d + n) % n
}
