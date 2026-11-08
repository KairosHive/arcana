// The moodboard's two picture roles, kept for the session: [R] the palette
// source and [T] the picture that receives it. They reach a board only when
// the board is saved. Shared by the rail, the similarity card and Alchemy.

import { load, save } from '../../lib/stores/persist'

export const roles = $state({
  reference: load<string | null>('moodboard.reference', null),
  target: load<string | null>('moodboard.target', null),
})

export function setReference(path: string | null) {
  roles.reference = roles.reference === path ? null : path
  save('moodboard.reference', roles.reference)
}

export function setTarget(path: string | null) {
  roles.target = roles.target === path ? null : path
  save('moodboard.target', roles.target)
}

export function forceReference(path: string) {
  roles.reference = path
  save('moodboard.reference', path)
}

/** Drop roles whose pictures left the collection. */
export function pruneRoles(items: string[]) {
  const set = new Set(items)
  if (roles.reference && !set.has(roles.reference)) { roles.reference = null; save('moodboard.reference', null) }
  if (roles.target && !set.has(roles.target)) { roles.target = null; save('moodboard.target', null) }
}
