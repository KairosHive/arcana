<script lang="ts">
  import type { MarksInfo } from './types'
  import Button from '../../lib/ui/Button.svelte'

  /**
   * What's marked: the collection as stars against colour tags, and when it
   * was shot. Every cell is a filter: clicking one searches only those frames.
   */
  let { info, onpick, onclose }: { info: MarksInfo; onpick: (stars: number | 'all', colour: string) => void; onclose: () => void } = $props()
  const m = $derived(info.matrix!)
  const cellMax = $derived(Math.max(1, ...m.cells.flat()))
  const total = $derived(info.total || 1)
  const rowTotals = $derived(m.cells.map((r) => r.reduce((a, b) => a + b, 0)))
  const colTotals = $derived(m.cols.map((_, j) => m.cells.reduce((a, r) => a + r[j], 0)))
  const UNRATED = -1, NO_COLOUR = '(none)'
  function rowLabel(s: number) { return s === UNRATED ? 'unrated' : s === 0 ? 'zero' : '★'.repeat(s) }
  function side(n: number, max: number) { return 10 + 30 * Math.sqrt(n / max) }
  function alpha(n: number, max: number) { return 0.3 + 0.55 * Math.sqrt(n / max) }
  const tl = $derived(info.timeline)
  const tlMax = $derived(tl ? Math.max(1, ...tl.bins.map((_, i) => tl.series.reduce((a, s) => a + s.counts[i], 0))) : 1)
  function binLabel(b: string) {
    const d = new Date(b + 'T00:00:00')
    if (!tl) return b
    if (tl.unit === 'M') return d.toLocaleDateString(undefined, { month: 'short', year: 'numeric' })
    if (tl.unit === 'W') return 'week of ' + d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
    return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
  }
</script>

<div class="panel">
  <div class="head">
    <div class="chips">
      {#each info.chips ?? [] as c}
        <div class="chip"><div class="v">{c.value}</div><div class="c">{c.caption}</div></div>
      {/each}
    </div>
    <Button size="sm" kind="ghost" onclick={onclose}>Close</Button>
  </div>
  <p class="hint">Every cell is a filter. Click one to search only those frames.</p>

  <div class="matrix" style="grid-template-columns: 76px repeat({m.cols.length + 1}, minmax(58px, 1fr))">
    <div></div>
    {#each m.cols as c}<div class="caps colh">{c === NO_COLOUR ? 'No tag' : c}</div>{/each}
    <div class="caps colh">all</div>
    {#each m.rows as s, i}
      <div class="rowh dim">{rowLabel(s)}</div>
      {#each m.cols as c, j}
        {@const n = m.cells[i][j]}
        <button class="cell" class:empty={!n} disabled={!n} title={n ? `${n.toLocaleString()} frames, click to search only these` : 'none'} onclick={() => onpick(s, c)}>
          <span class="sq" style:width="{side(n, cellMax)}px" style:height="{side(n, cellMax)}px" style:opacity={alpha(n, cellMax)} style:background={m.swatch[c] || 'var(--arc-muted)'}></span>
          <span class="n" class:faint={!n}>{n ? n.toLocaleString() : '·'}</span>
        </button>
      {/each}
      {@const rt = rowTotals[i]}
      <button class="cell" class:empty={!rt} disabled={!rt} onclick={() => onpick(s, '')} title="{rt.toLocaleString()} frames">
        <span class="sq" style:width="{side(rt, total)}px" style:height="{side(rt, total)}px" style:opacity={alpha(rt, total)} style:background="var(--arc-muted)"></span>
        <span class="n">{rt.toLocaleString()}</span>
      </button>
    {/each}
    <div></div>
    {#each m.cols as c, j}
      {@const ct = colTotals[j]}
      <button class="cell" class:empty={!ct} disabled={!ct} onclick={() => onpick('all', c)} title="{ct.toLocaleString()} frames">
        <span class="sq" style:width="{side(ct, total)}px" style:height="{side(ct, total)}px" style:opacity={alpha(ct, total)} style:background={m.swatch[c] || 'var(--arc-muted)'}></span>
        <span class="n">{ct.toLocaleString()}</span>
      </button>
    {/each}
    <div class="grand dim">{total.toLocaleString()}</div>
  </div>

  {#if tl && tl.bins.length}
    <div class="caps tlh">When it was shot</div>
    <div class="timeline" role="img" aria-label="Frames per period, stacked by colour tag">
      {#each tl.bins as b, i}
        {@const sum = tl.series.reduce((a, s) => a + s.counts[i], 0)}
        <div class="bin" title="{binLabel(b)}: {sum.toLocaleString()} frames">
          {#each tl.series as s}
            {#if s.counts[i]}<span class="seg" style:height="{(s.counts[i] / tlMax) * 100}%" style:background={s.colour} title="{s.tag}: {s.counts[i].toLocaleString()}"></span>{/if}
          {/each}
        </div>
      {/each}
    </div>
    <div class="tl-axis faint"><span>{binLabel(tl.bins[0])}</span><span>{binLabel(tl.bins[tl.bins.length - 1])}</span></div>
  {/if}
</div>

<style>
  .panel { position: absolute; inset: 0; z-index: 6; overflow-y: auto; padding: var(--sp-4); background: var(--arc-surface); border-radius: var(--r-lg); display: flex; flex-direction: column; gap: 10px; }
  .head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { background: var(--arc-surface-2); border: 1px solid var(--arc-line); border-radius: var(--r-md); padding: 8px 12px; }
  .v { font-size: 15px; font-weight: 600; } .c { font-size: 10.5px; color: var(--arc-ink-faint); text-transform: uppercase; letter-spacing: 0.6px; }
  .matrix { display: grid; gap: 3px; }
  .colh { text-align: center; align-self: end; }
  .rowh { font-size: var(--fs-sm); text-align: right; padding-right: 10px; align-self: center; white-space: nowrap; }
  .cell { position: relative; display: flex; align-items: center; justify-content: center; min-height: 46px; background: var(--arc-surface-2); border: 1px solid var(--arc-line); border-radius: 3px; cursor: pointer; }
  .cell.empty { background: transparent; opacity: 0.4; cursor: default; }
  .sq { position: absolute; border-radius: 2px; }
  .n { position: relative; z-index: 1; font-weight: 600; font-size: var(--fs-sm); text-shadow: 0 1px 3px rgba(0,0,0,0.75); }
  .grand { display: flex; align-items: center; justify-content: center; font-size: var(--fs-sm); font-weight: 600; }
  .tlh { margin-top: 8px; }
  .timeline { display: flex; align-items: flex-end; gap: 1px; height: 120px; padding: 4px 0; border-bottom: 1px solid var(--arc-line); }
  .bin { flex: 1 1 0; display: flex; flex-direction: column-reverse; height: 100%; min-width: 2px; }
  .seg { display: block; width: 100%; }
  .tl-axis { display: flex; justify-content: space-between; font-size: var(--fs-2xs); }
</style>
