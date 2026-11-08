export type MarksInfo = {
  has: boolean; total?: number
  stars?: { label: string; value: number }[]; colours?: { label: string; value: string; colour: string }[]
  dates?: { min: string; max: string } | null; hasStars?: boolean; hasColours?: boolean
  chips?: { value: string; caption: string }[]
  matrix?: { rows: number[]; cols: string[]; cells: number[][]; swatch: Record<string, string> }
  timeline?: { unit: string; bins: string[]; series: { tag: string; colour: string; counts: number[] }[] } | null
}

export type Marks = { stars: number | null; colour: string; captured: string } | null | undefined
