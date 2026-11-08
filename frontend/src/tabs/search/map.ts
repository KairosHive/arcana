// The latent map's data, fetched once per dataset and shared by both search
// tabs. A 100k-point map is a few hundred kilobytes of JSON; drawing it is
// the browser's job (LatentMap.svelte), so nothing here touches Plotly.

import { get, post } from '../../lib/api/client'

export type MapData = {
  dataset: string; n: number; modality: 'image' | 'audio'
  x: number[]; y: number[]; z?: number[]
  labels: string[]; cluster: number[]; paths: string[]; hasMarks: boolean
  stars?: (number | null)[] | null; colour?: string[] | null; captured?: (string | null)[] | null
}

const cache = new Map<string, Promise<MapData>>()

export function loadMap(dataset: string): Promise<MapData> {
  let p = cache.get(dataset)
  if (!p) {
    p = get<MapData>('/api/map', { dataset })
    p.catch(() => cache.delete(dataset))
    cache.set(dataset, p)
  }
  return p
}

export function forgetMap(dataset?: string) {
  if (dataset) cache.delete(dataset)
  else cache.clear()
}

export type Filter = { stars: number; colours: string[]; dateFrom: string; dateTo: string }
export const EMPTY_FILTER: Filter = { stars: 0, colours: [], dateFrom: '', dateTo: '' }
export function filterActive(f: Filter) { return !!(f.stars || f.colours.length || f.dateFrom || f.dateTo) }

export async function filteredRows(dataset: string, f: Filter): Promise<{ kept: number[] | null; count: number; total: number }> {
  return post('/api/map/filter', { dataset, stars: f.stars, colours: f.colours, dateFrom: f.dateFrom || null, dateTo: f.dateTo || null })
}

// Plotly's Dark24, which the old map used, so cluster hues stay familiar.
export const CLUSTER_COLOURS = [
  '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100', '#750D86',
  '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D', '#6C7C32', '#778AAE',
  '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038',
]
