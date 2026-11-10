// The shape of the capability table served at /api/bootstrap, and what the
// page assumes before it arrives. Kept apart from caps.svelte.ts so the app
// store can hold the table without importing the helpers that read the store.

export type Modality = 'image' | 'audio' | 'video'
export type Feature = { id: string; tabs: string[]; label: string; scope: 'dataset' | 'item' | 'none'; modalities: Modality[] }
export type Capabilities = {
  features: Feature[]; tabs: Record<string, Modality[]>; datasetTabs: string[]
  labels: Record<string, string>; plural: Record<string, string>; order: Modality[]
}

export const EMPTY: Capabilities = {
  features: [], tabs: {}, datasetTabs: [],
  labels: { image: 'photo', audio: 'audio', video: 'video' },
  plural: { image: 'photos', audio: 'sounds', video: 'videos' },
  order: ['image', 'audio', 'video'],
}
