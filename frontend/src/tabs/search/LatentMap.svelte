<script lang="ts">
  import { onMount } from 'svelte'
  import type { MapData } from './map'
  import { CLUSTER_COLOURS } from './map'
  import { thumbUrl, audioImageUrl } from '../../lib/api/media'
  import Spinner from '../../lib/ui/Spinner.svelte'
  import Empty from '../../lib/ui/Empty.svelte'

  /**
   * The dataset as a map: one WebGL trace per cluster, a ghost trace for the
   * points a marks filter leaves out, crosses on the hits, a gold path for a
   * story. Plotly is loaded on first use and the figure is redrawn with
   * Plotly.react, which keeps zoom and camera between redraws.
   */
  let {
    data = null, loading = false, message = 'Choose a dataset above to begin.',
    kept = null, hits = [], story = [], spectrogram = false, onclick, visible = true,
  }: {
    data?: MapData | null; loading?: boolean; message?: string
    /** row indices the filter keeps, or null for all */
    kept?: number[] | null
    /** row indices to cross */
    hits?: number[]
    /** story scenes in order: row index and text */
    story?: { row: number | null; text: string }[]
    spectrogram?: boolean; onclick?: (path: string) => void; visible?: boolean
  } = $props()

  let host: HTMLDivElement | undefined = $state()
  let plotly: typeof import('plotly.js-dist-min') | null = null
  let drawn = false
  let hover = $state<{ src: string; alt: string } | null>(null)
  let hoverTimer: ReturnType<typeof setTimeout> | null = null

  const BG = getComputedStyle(document.documentElement).getPropertyValue('--arc-bg').trim() || '#141311'
  const LINE = getComputedStyle(document.documentElement).getPropertyValue('--arc-line').trim() || '#38332d'
  const ACCENT = getComputedStyle(document.documentElement).getPropertyValue('--arc-accent').trim() || '#c8743f'
  const MUTED = getComputedStyle(document.documentElement).getPropertyValue('--arc-muted').trim() || '#8a8278'

  async function lib() {
    if (!plotly) plotly = (await import('plotly.js-dist-min')) as any
    return plotly!
  }

  function pick<T>(arr: T[], rows: Iterable<number>): T[] { const out: T[] = []; for (const r of rows) out.push(arr[r]); return out }

  function traces(d: MapData): any[] {
    const is3d = !!d.z
    const keepSet = kept ? new Set(kept) : null
    const byCluster: number[][] = d.labels.map(() => [])
    const ghost: number[] = []
    for (let i = 0; i < d.n; i++) {
      if (keepSet && !keepSet.has(i)) ghost.push(i)
      else byCluster[d.cluster[i]].push(i)
    }
    const out: any[] = []
    const base = is3d
      ? { type: 'scatter3d', mode: 'markers' }
      : { type: 'scattergl', mode: 'markers' }
    if (ghost.length) {
      out.push({ ...base, name: 'filtered out', hoverinfo: 'skip', showlegend: false,
        x: pick(d.x, ghost), y: pick(d.y, ghost), ...(is3d ? { z: pick(d.z!, ghost) } : {}),
        marker: { size: is3d ? 3 : 6, color: MUTED, opacity: 0.12 } })
    }
    d.labels.forEach((label, ci) => {
      const rows = byCluster[ci]
      if (!rows.length) return
      out.push({ ...base, name: label || `cluster ${ci}`, legendgroup: label,
        x: pick(d.x, rows), y: pick(d.y, rows), ...(is3d ? { z: pick(d.z!, rows) } : {}),
        customdata: rows, hovertemplate: `${label}<extra></extra>`,
        marker: { size: is3d ? 4 : 8, color: CLUSTER_COLOURS[ci % CLUSTER_COLOURS.length], opacity: 0.6 } })
    })
    if (hits.length) {
      out.push({ ...base, name: 'Search Results', hoverinfo: 'skip', showlegend: true,
        x: pick(d.x, hits), y: pick(d.y, hits), ...(is3d ? { z: pick(d.z!, hits) } : {}),
        marker: is3d ? { size: 10, symbol: 'cross', color: ACCENT, opacity: 1 }
                     : { symbol: 'x', size: 20, color: ACCENT, line: { width: 2, color: '#ffffff' }, opacity: 1 } })
    }
    const scenes = story.filter((s) => s.row !== null && s.row !== undefined) as { row: number; text: string }[]
    if (scenes.length) {
      const rows = scenes.map((s) => s.row)
      out.push({ type: is3d ? 'scatter3d' : 'scattergl', mode: 'lines+markers', name: 'Story Path',
        x: pick(d.x, rows), y: pick(d.y, rows), ...(is3d ? { z: pick(d.z!, rows) } : {}),
        text: scenes.map((s) => s.text), hovertemplate: '%{text}<extra></extra>',
        line: { color: 'gold', width: 4 },
        marker: is3d ? { size: 8, symbol: 'cross', color: 'gold' } : { size: 14, color: 'gold', line: { width: 1, color: '#ffffff' } } })
    }
    return out
  }

  function layout(d: MapData): any {
    const is3d = !!d.z
    const bare = { showgrid: false, zeroline: false, showline: false, showticklabels: false, title: { text: '' }, ticks: '' }
    const scenes = story.filter((s) => s.row !== null) as { row: number; text: string }[]
    const l: any = {
      plot_bgcolor: BG, paper_bgcolor: BG, font: { color: '#ede8df', size: 11 },
      margin: { l: 8, r: 8, t: 8, b: 8 }, uirevision: d.dataset, autosize: true,
      legend: { bgcolor: 'rgba(20,19,17,0.55)', bordercolor: LINE, borderwidth: 1, font: { size: 10 }, itemsizing: 'constant' },
      hoverlabel: { bgcolor: '#1c1a17', bordercolor: LINE, font: { color: '#ede8df', size: 11 } },
      dragmode: is3d ? 'orbit' : 'pan',
    }
    if (is3d) {
      const ax = { backgroundcolor: BG, color: LINE, ...bare }
      l.scene = { xaxis: ax, yaxis: ax, zaxis: ax }
    } else {
      l.xaxis = bare; l.yaxis = bare
      if (scenes.length) l.annotations = [{ x: d.x[scenes[0].row], y: d.y[scenes[0].row], text: 'Beginning', showarrow: false,
        font: { size: 14, color: 'gold' }, yshift: 28, bgcolor: 'rgba(0,0,0,0.5)', borderpad: 6 }]
    }
    return l
  }

  async function draw() {
    if (!host || !data) return
    const P = await lib()
    if (!host || !data) return
    const config = { responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toImage'] }
    await P.react(host, traces(data), layout(data), config as any)
    if (!drawn) {
      drawn = true
      ;(host as any).on('plotly_hover', (ev: any) => {
        const pt = ev.points?.[0]
        const row = pt?.customdata
        if (row === undefined || !data) return
        const path = data.paths[row]
        if (hoverTimer) clearTimeout(hoverTimer)
        hover = { src: data.modality === 'audio' ? audioImageUrl(path, spectrogram) : thumbUrl(path), alt: path }
      })
      ;(host as any).on('plotly_unhover', () => {
        hoverTimer = setTimeout(() => { hover = null }, 120)
      })
      ;(host as any).on('plotly_click', (ev: any) => {
        const row = ev.points?.[0]?.customdata
        if (row !== undefined && data) onclick?.(data.paths[row])
      })
    }
  }

  // Redraw when the data, the filter, the hits or the story change.
  $effect(() => {
    data; kept; hits; story
    if (data && host && visible) draw()
  })

  // The host has no size while hidden; redraw when shown.
  $effect(() => {
    if (visible && host && data) requestAnimationFrame(() => { if (plotly && host) plotly.Plots.resize(host) })
  })

  onMount(() => {
    const ro = new ResizeObserver(() => { if (plotly && host && drawn && host.clientWidth > 0) plotly.Plots.resize(host) })
    if (host) ro.observe(host)
    return () => { ro.disconnect(); if (plotly && host) plotly.purge(host) }
  })
</script>

<div class="wrap">
  {#if data}
    <div class="plot" bind:this={host}></div>
    {#if hover}<img class="hover" src={hover.src} alt="" />{/if}
    {#if loading}<div class="veil"><Spinner label="Loading the map…" /></div>{/if}
  {:else if loading}
    <div class="blank"><Spinner size={18} label="Loading the map…" /></div>
  {:else}
    <div class="blank"><Empty title={message} /></div>
  {/if}
</div>

<style>
  .wrap { position: relative; flex: 1; min-height: 160px; border-radius: var(--r-lg); overflow: hidden; background: var(--arc-bg); border: 1px solid var(--arc-line); }
  .plot { position: absolute; inset: 0; }
  .blank { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
  .veil { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(20,19,17,0.5); }
  .hover { position: absolute; top: 6px; left: 6px; z-index: 20; max-width: 170px; max-height: 130px; border-radius: var(--r-md); border: 1px solid #ffffff55; box-shadow: 0 6px 20px #000c; background: #000; object-fit: contain; pointer-events: none; }
</style>
