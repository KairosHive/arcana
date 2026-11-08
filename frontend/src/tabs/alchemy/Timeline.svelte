<script lang="ts">
  import { film, commit, keysOf, clone, laneValue, sorted, type Key } from './film.svelte'
  import { thumbUrl } from '../../lib/api/media'

  /**
   * The film as a timeline: a ruler with the playhead, the picture lane, the
   * look lane, and one lane per keyed number drawn as its curve.
   *
   *   drag the ruler            scrub
   *   drag a picture or a look  move it in time
   *   drag a lane's key         move it in time and value
   *   double-click a lane       add a key on its curve
   *   click a key               select it; Delete removes it
   */
  const LABEL_W = 150, RULER = 22, PICS = 58, LOOKS = 26, LANE = 46, PAD = 8
  let host: HTMLDivElement | undefined = $state()
  let W = $state(640)

  const spec = $derived(film.spec)
  const meta = $derived(film.meta)
  const L = $derived(Math.max(meta?.length || 1, 0.5))
  const x0 = $derived(LABEL_W + PAD)
  const x1 = $derived(W - PAD)
  const lanes = $derived(Object.keys(spec.lanes))
  const H = $derived(laneTop(lanes.length) + 4)
  function laneTop(i: number) { return RULER + PICS + LOOKS + i * LANE }
  const X = (t: number) => x0 + (t / L) * (x1 - x0)
  const T = (x: number) => Math.min(Math.max(((x - x0) / (x1 - x0)) * L, 0), L)
  const tickStep = $derived([0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => (x1 - x0) / (L / s) >= 46) || 120)
  const ticks = $derived.by(() => { const out: number[] = []; for (let s = 0; s <= L + 1e-6; s += tickStep) out.push(s); return out })
  const pics = $derived(sorted(spec.pictures))
  const looks = $derived(sorted(spec.looks))
  function tickLabel(s: number) { return s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : `${+s.toFixed(2)}s` }
  function laneY(p: string, v: number, top: number) {
    const m = meta?.lanes[p] ?? { lo: 0, hi: 1 }
    const lo = m.lo, hi = m.hi === m.lo ? m.lo + 1 : m.hi
    return top + 6 + (1 - (v - lo) / (hi - lo)) * (LANE - 12)
  }
  function curve(p: string, top: number) {
    const keys = sorted(spec.lanes[p])
    const pts: string[] = []
    for (let k = 0; k <= 160; k++) { const tt = (L * k) / 160; pts.push(`${X(tt).toFixed(1)},${laneY(p, laneValue(keys, tt), top).toFixed(1)}`) }
    return pts.join(' ')
  }
  function isSel(lane: string, i: number) { return film.sel?.lane === lane && film.sel?.i === i }
  function range(p: string) { const m = meta?.lanes[p] ?? { lo: 0, hi: 1 }; return `${+m.lo.toFixed(2)}–${+m.hi.toFixed(2)}` }

  // ---- interaction ----
  type Drag = { kind: 'ruler' } | { kind: 'key'; lane: string; key: Key; spec: typeof film.spec; moved: boolean; startX: number; startY: number }
  let drag: Drag | null = null
  let draft = $state<typeof film.spec | null>(null)
  const shown = $derived(draft ?? spec)

  function down(e: PointerEvent) {
    const target = e.target as SVGElement
    const kind = target.getAttribute?.('data-kind')
    if (!kind || e.button !== 0 || !host) return
    host.focus()
    const r = host.getBoundingClientRect()
    const x = e.clientX - r.left
    if (kind === 'ruler' || kind === 'band') {
      drag = { kind: 'ruler' }; film.t = T(x)
    } else if (kind === 'remove') {
      const s = clone(spec); delete s.lanes[target.getAttribute('data-lane')!]; commit(s, null); return
    } else if (kind === 'key') {
      const lane = target.getAttribute('data-lane')!, i = +target.getAttribute('data-i')!
      const s = clone(spec), keys = keysOf(s, lane)!
      film.sel = { lane, i }
      drag = { kind: 'key', lane, key: keys[i], spec: s, moved: false, startX: e.clientX, startY: e.clientY }
    }
    host.setPointerCapture(e.pointerId)
    e.preventDefault()
  }
  function move(e: PointerEvent) {
    if (!drag || !host) return
    const r = host.getBoundingClientRect()
    if (drag.kind === 'ruler') { film.t = T(e.clientX - r.left); return }
    const d = drag
    if (!d.moved && Math.abs(e.clientX - d.startX) + Math.abs(e.clientY - d.startY) < 3) return
    d.moved = true
    d.key.t = Math.round(T(e.clientX - r.left) * 100) / 100
    if (d.lane !== 'pictures' && d.lane !== 'looks') {
      const li = lanes.indexOf(d.lane), top = laneTop(li), m = meta?.lanes[d.lane] ?? { lo: 0, hi: 1, step: 0.01 }
      const f = 1 - (e.clientY - r.top - top - 6) / (LANE - 12)
      const v = m.lo + Math.min(Math.max(f, 0), 1) * (m.hi - m.lo)
      const stp = m.step || 0.01
      d.key.v = Math.round(v / stp) * stp
    }
    draft = clone(d.spec)
  }
  function up() {
    const d = drag
    drag = null; draft = null
    if (!d || d.kind !== 'key' || !d.moved) return
    const keys = keysOf(d.spec, d.lane)!
    const i = keys.indexOf(d.key)
    if (d.lane === 'pictures' || d.lane === 'looks') film.t = d.key.t
    commit(d.spec, { lane: d.lane, i })
  }
  function dbl(e: MouseEvent) {
    const target = e.target as SVGElement
    const lane = target.getAttribute?.('data-lane')
    if (!lane || lane === 'pictures' || lane === 'looks' || target.getAttribute('data-kind') !== 'band' || !host) return
    const r = host.getBoundingClientRect()
    const t = Math.round(T(e.clientX - r.left) * 100) / 100
    const s = clone(spec)
    const keys = keysOf(s, lane)!
    const key: Key = { t, v: laneValue(keys, t), ease: 'linear' }
    keys.push(key)
    const sortedKeys = keysOf(s, lane)!
    commit(s, { lane, i: sortedKeys.indexOf(key) })
  }

  $effect(() => {
    if (!host) return
    const ro = new ResizeObserver(() => { W = Math.max(host!.clientWidth, 320) })
    ro.observe(host)
    W = Math.max(host.clientWidth, 320)
    return () => ro.disconnect()
  })
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="timeline" bind:this={host} tabindex="0" role="application" aria-label="Film timeline"
     onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up} ondblclick={dbl}>
  <svg width={W} height={H} class="svg">
    <rect class="ruler" data-kind="ruler" x="0" y="0" width={W} height={RULER} />
    {#each ticks as s}
      <line class="tick" x1={X(s)} y1={RULER - 7} x2={X(s)} y2={RULER} />
      <text class="tick-text" x={X(s) + 3} y="12">{tickLabel(s)}</text>
    {/each}

    <rect class="band" data-kind="band" data-lane="pictures" x="0" y={RULER} width={W} height={PICS} />
    <text class="row-label" x="10" y={RULER + 16}>pictures</text>
    <rect class="band odd" data-kind="band" data-lane="looks" x="0" y={RULER + PICS} width={W} height={LOOKS} />
    <text class="row-label" x="10" y={RULER + PICS + LOOKS / 2 + 4}>looks</text>
    {#each lanes as p, i (p)}
      <rect class="band" class:odd={i % 2 === 0} data-kind="band" data-lane={p} x="0" y={laneTop(i)} width={W} height={LANE} />
      <text class="row-label" x="10" y={laneTop(i) + 16}>{meta?.lanes[p]?.label ?? p}</text>
      <text class="row-remove" data-kind="remove" data-lane={p} x={LABEL_W - 16} y={laneTop(i) + 16}>×</text>
    {/each}

    {#each sorted(shown.pictures) as p, i}
      {@const xx = X(p.t)}
      <line class="key-line" x1={xx} y1={RULER} x2={xx} y2={RULER + PICS} />
      <image href={meta?.pictures[i]?.path ? thumbUrl(meta.pictures[i].path) : ''} x={xx - 22} y={RULER + 5} width="44" height="44" preserveAspectRatio="xMidYMid slice" class="thumb" data-kind="key" data-lane="pictures" data-i={i} />
      <rect class="thumb-frame" class:sel={isSel('pictures', i)} x={xx - 22} y={RULER + 5} width="44" height="44" rx="4" pointer-events="none" />
    {/each}

    {#each sorted(shown.looks) as k, i}
      {@const xx = X(k.t)}
      {@const y = RULER + PICS + LOOKS / 2}
      <path class="look" class:sel={isSel('looks', i)} data-kind="key" data-lane="looks" data-i={i} d="M{xx},{y - 7} l7,7 l-7,7 l-7,-7z" />
      <text class="look-text" x={xx + 10} y={y + 4}>{meta?.looks[i]?.label ?? ''}</text>
    {/each}

    {#each lanes as p, li (p)}
      {@const top = laneTop(li)}
      <polyline class="curve" points={curve(p, top)} />
      {#each sorted(shown.lanes[p] ?? []) as key, i}
        <circle class="dot" class:sel={isSel(p, i)} data-kind="key" data-lane={p} data-i={i} cx={X(key.t)} cy={laneY(p, key.v ?? 0, top)} r={isSel(p, i) ? 6 : 5} />
      {/each}
      <text class="lane-range" x={LABEL_W - 4} y={top + LANE - 6} text-anchor="end">{range(p)}</text>
    {/each}

    <line class="playhead" x1={X(film.t)} y1="0" x2={X(film.t)} y2={H} />
    <path class="playhead-cap" data-kind="ruler" d="M{X(film.t) - 6},0 h12 l-6,8z" />
  </svg>
</div>

<style>
  .timeline { user-select: none; touch-action: none; border: 1px solid var(--arc-line); border-radius: var(--r-md); overflow: hidden; background: var(--arc-surface); }
  .timeline:focus-visible { border-color: var(--arc-accent); box-shadow: none; }
  .svg { display: block; }
  .ruler { fill: var(--arc-surface-2); cursor: ew-resize; }
  .tick { stroke: var(--arc-line-strong); }
  .tick-text { fill: var(--arc-ink-faint); font-size: 10px; }
  .band { fill: transparent; cursor: ew-resize; }
  .band.odd { fill: rgba(255,255,255,0.025); }
  .row-label { fill: var(--arc-ink-dim); font-size: 11px; pointer-events: none; }
  .row-remove { fill: var(--arc-ink-faint); font-size: 14px; cursor: pointer; }
  .row-remove:hover { fill: var(--arc-bad); }
  .key-line { stroke: var(--arc-line-strong); stroke-dasharray: 2 2; }
  .thumb { cursor: grab; }
  .thumb-frame { fill: none; stroke: var(--arc-line-strong); stroke-width: 1; }
  .thumb-frame.sel { stroke: var(--tab-alchemy); stroke-width: 2; }
  .look { fill: var(--arc-target); cursor: grab; }
  .look.sel { fill: var(--tab-alchemy); }
  .look-text { fill: var(--arc-ink-dim); font-size: 10px; pointer-events: none; }
  .curve { fill: none; stroke: var(--arc-accent); stroke-width: 1.6; pointer-events: none; }
  .dot { fill: var(--arc-surface); stroke: var(--arc-accent); stroke-width: 2; cursor: grab; }
  .dot.sel { fill: var(--tab-alchemy); stroke: var(--tab-alchemy); }
  .lane-range { fill: var(--arc-ink-faint); font-size: 9.5px; pointer-events: none; }
  .playhead { stroke: var(--arc-ink); stroke-width: 1; pointer-events: none; }
  .playhead-cap { fill: var(--arc-ink); cursor: ew-resize; }
</style>
