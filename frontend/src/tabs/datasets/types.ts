export type DatasetRow = {
  name: string; modality: 'image' | 'audio'; total: number; root: string; missing: number; error: string
  hasMap: boolean; dims: number[]; labels: boolean; palette: boolean; paletteStale: boolean; style: boolean
  status: 'ready' | 'building' | 'unfinished' | 'files missing' | 'unreadable'; building: boolean
  canRework: boolean; canExtend: boolean; canMarks: boolean
}

export type ModelRow = {
  id: string; label: string; modality: string; dim: number; download_mb: number; quality: string; blurb: string
  downloaded: boolean; partial: boolean; labelsReady: boolean; complete: boolean; whyNot: string
}
