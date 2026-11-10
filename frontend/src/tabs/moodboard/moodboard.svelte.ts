// The moodboard's reference [R], kept for the session: the picture the
// similarity search starts from. It reaches a board only when the board is
// saved. Colour transfer onto a picture lives in Alchemy, with its own roles.
// A board's old `transfer` field stays on disk and is no longer shown here.

import { load, save } from '../../lib/stores/persist'

export const roles = $state({
  reference: load<string | null>('moodboard.reference', null),
})

export function setReference(path: string | null) {
  roles.reference = roles.reference === path ? null : path
  save('moodboard.reference', roles.reference)
}

export function forceReference(path: string) {
  roles.reference = path
  save('moodboard.reference', path)
}

/** Drop roles whose pictures left the collection. */
export function pruneRoles(items: string[]) {
  const set = new Set(items)
  if (roles.reference && !set.has(roles.reference)) { roles.reference = null; save('moodboard.reference', null) }
}
