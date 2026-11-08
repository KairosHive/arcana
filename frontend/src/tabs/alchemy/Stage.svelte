<script lang="ts">
  import { post, reason } from '../../lib/api/client'
  import { toast } from '../../lib/stores/toast.svelte'
  import { app, touch } from '../../lib/stores/app.svelte'
  import { load, save } from '../../lib/stores/persist'
  import { step, controls, method, materials } from './alchemy.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Status from '../../lib/ui/Status.svelte'

  /**
   * Before and after, cut by a line you drag, and the bar above that says a
   * draw is coming. One preview request is out at a time: a change that
   * arrives while it runs is held and sent when it lands. A slow method draws
   * on Draw; a live one on every change, unless live is off.
   */
  let { onlook }: { onlook?: () => Promise<void> } = $props()

  let after = $state<string | null>(null)
  let before = $state<string | null>(null)
  let beforeKey = ''
  let info = $state(''), status = $state(''), keepStatus = $state('')
  let cut = $state(50)
  let out = false, held = false, heldDraw = false
  let timer: ReturnType<typeof setTimeout> | null = null
  let keeping = $state(false)

  // Expected timings per step key, so the bar can pace itself.
  const TIMINGS = 'alchemy.timings'
  let timings = load<Record<string, number>>(TIMINGS, {})
  let drawing = $state(false)
  let expect = $state(2500)
  let startedAt = 0

  const m = $derived(method(step.method))
  const hasPictures = $derived(!!step.roles.T || materials().length > 0)
  const stageKey = $derived(`${step.method}|${step.knobs.model ?? ''}|${step.view}|${step.size}`)

  async function send(draw: boolean) {
    if (out) { held = true; heldDraw = heldDraw || draw; return }
    if (!hasPictures) { status = ''; return }
    if (!draw && (!step.live || (m && m.cost !== 'live' && step.view === 'compare'))) { status = 'Draw shows this step.'; return }
    out = true; drawing = true; status = ''
    expect = timings[stageKey] ?? 2500
    startedAt = performance.now()
    try {
      const r = await post('/api/alchemy/preview', { controls: controls(), view: step.view, size: step.size, beforeKey })
      after = r.after
      if (r.before) { before = r.before; beforeKey = r.beforeKey }
      info = r.info
      timings[stageKey] = Math.round(performance.now() - startedAt)
      save(TIMINGS, timings)
    } catch (e) {
      status = reason(e)
    } finally {
      out = false; drawing = false
      if (held) { const d = heldDraw; held = false; heldDraw = false; send(d) }
    }
  }

  /** Any control change: debounced, so a slider drag is one request. */
  export function changed() {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => send(false), 160)
  }
  export function draw() { send(true) }

  // Redraw when the stage's own inputs change.
  $effect(() => { step.view; step.size; step.roles.T; step.roles['1']; step.roles['2']; step.roles['3']; changed() })

  async function keep() {
    keeping = true; keepStatus = ''
    try {
      const r = await post('/api/alchemy/keep', { controls: controls(), size: step.keepSize })
      keepStatus = r.message
      if (r.collection) app.collection = r.collection
      await touch()
    } catch (e) { keepStatus = reason(e) } finally { keeping = false }
  }

  function pointer(e: PointerEvent) {
    const el = e.currentTarget as HTMLElement
    const r = el.getBoundingClientRect()
    const move = (ev: PointerEvent) => { cut = Math.max(0, Math.min(100, ((ev.clientX - r.left) / r.width) * 100)) }
    move(e)
    el.setPointerCapture(e.pointerId)
    const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up) }
    el.addEventListener('pointermove', move); el.addEventListener('pointerup', up)
  }
</script>

<div class="stage">
  <div class="bar">
    <div class="seg">
      <button class:on={step.view === 'compare'} onclick={() => (step.view = 'compare')}>before and after</button>
      <button class:on={step.view === 'mask'} onclick={() => (step.view = 'mask')}>mask</button>
    </div>
    <Select bind:value={step.size} options={[512, 640, 768, 1024].map((n) => ({ value: n, label: `preview ${n}` }))} size="sm" width="128px" />
    <Checkbox bind:checked={step.live} label="live" />
    <Button size="sm" onclick={draw} disabled={!hasPictures}>Draw</Button>
    <span class="spacer"></span>
    <Select bind:value={step.keepSize} options={[{ value: 0, label: 'keep at full size' }, { value: 2048, label: 'keep at 2048' }, { value: 4096, label: 'keep at 4096' }]} size="sm" width="150px" />
    <Button size="sm" title="This step, as a look on the Video subtab's timeline at its playhead" onclick={() => onlook?.()}>Add look to film</Button>
    <Button size="sm" kind="primary" onclick={keep} busy={keeping} disabled={!hasPictures}>Keep</Button>
  </div>

  <div class="canvas">
    <div class="loadbar" class:on={drawing}><div class="fill" style:animation-duration="{Math.max(300, expect)}ms"></div></div>
    {#if !hasPictures}
      <p class="empty hint">Give a picture [T] and a material [1] from the collection on the left.</p>
    {:else if after}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="compare" style="--cut: {cut}%" onpointerdown={pointer}>
        <img class="before" src={before ?? after} alt="" />
        <div class="after"><img src={after} alt="" /></div>
        <div class="handle"></div>
        <span class="tag l">before</span>
        <span class="tag r">{step.view === 'mask' ? 'mask' : 'after'}</span>
      </div>
    {:else}
      <p class="empty hint">{status || 'Drawing the first preview…'}</p>
    {/if}
  </div>
  <div class="under">
    <span class="info faint">{info}</span>
    <Status text={after ? status : ''} tone="ink" size="xs" />
    <Status text={keepStatus} tone="ok" size="xs" />
  </div>
</div>

<style>
  .stage { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; padding: 12px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); }
  .bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; flex: 0 0 auto; }
  .seg { display: inline-flex; border: 1px solid var(--arc-line); border-radius: var(--r-md); overflow: hidden; }
  .seg button { height: var(--ctl-h-sm); padding: 0 10px; background: var(--arc-surface-2); color: var(--arc-ink-dim); border: none; font-size: var(--fs-xs); font-weight: 600; }
  .seg button + button { border-left: 1px solid var(--arc-line); }
  .seg button.on { background: color-mix(in srgb, var(--arc-accent) 25%, var(--arc-surface-2)); color: var(--arc-ink); }
  .canvas { position: relative; flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; overflow: auto; padding-top: 8px; }
  .loadbar { position: absolute; top: 0; left: 0; right: 0; height: 4px; background: color-mix(in srgb, var(--arc-accent) 16%, transparent); opacity: 0; transition: opacity 0.18s; z-index: 3; overflow: hidden; }
  .loadbar.on { opacity: 1; }
  .fill { height: 100%; transform-origin: left; transform: scaleX(0); background: linear-gradient(90deg, #a85a2c, var(--arc-accent) 70%, #f0b07a); }
  .loadbar.on .fill { animation: fill cubic-bezier(0.3, 0.55, 0.45, 1) forwards; }
  @keyframes fill { 0% { transform: scaleX(0); } 80% { transform: scaleX(0.92); } 100% { transform: scaleX(0.985); } }
  .compare { position: relative; display: inline-block; max-width: 100%; line-height: 0; user-select: none; touch-action: none; cursor: ew-resize; border-radius: var(--r-md); overflow: hidden; }
  .before { display: block; max-width: 100%; max-height: 74vh; }
  .after { position: absolute; inset: 0; clip-path: inset(0 0 0 var(--cut)); }
  .after img { width: 100%; height: 100%; object-fit: fill; display: block; }
  .handle { position: absolute; top: 0; bottom: 0; left: var(--cut); width: 2px; background: var(--arc-ink); box-shadow: 0 0 6px rgba(0,0,0,0.6); transform: translateX(-1px); }
  .handle::after { content: ''; position: absolute; top: 50%; left: 50%; width: 22px; height: 22px; border-radius: 50%; border: 2px solid var(--arc-ink); background: rgba(20,19,17,0.7); transform: translate(-50%, -50%); }
  .tag { position: absolute; top: 8px; font-size: 10.5px; line-height: 1; padding: 4px 7px; border-radius: var(--r-sm); background: rgba(20,19,17,0.6); color: var(--arc-ink); }
  .tag.l { left: 8px; } .tag.r { right: 8px; }
  .empty { text-align: center; padding: 40px 10px; font-size: 12.5px; }
  .under { display: flex; flex-direction: column; align-items: center; gap: 2px; text-align: center; flex: 0 0 auto; }
  .info { font-size: 11.5px; }
</style>
