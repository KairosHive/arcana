<script lang="ts">
  import type { Snippet } from 'svelte'
  import { load, save } from '../stores/persist'
  /** Two panes side by side with a drag handle between them. The first pane's
      width is the thing dragged and is remembered under `storeKey`. */
  let { storeKey = '', initial = 380, min = 220, max = 900, a, b }: {
    storeKey?: string; initial?: number; min?: number; max?: number; a: Snippet; b: Snippet
  } = $props()
  // svelte-ignore state_referenced_locally
  let width = $state(storeKey ? load<number>('split.' + storeKey, initial) : initial)
  let dragging = $state(false)
  function down(e: PointerEvent) {
    dragging = true
    const startX = e.clientX, startW = width
    const move = (ev: PointerEvent) => { width = Math.max(min, Math.min(max, startW + ev.clientX - startX)) }
    const up = () => {
      dragging = false
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up)
      if (storeKey) save('split.' + storeKey, width)
      window.dispatchEvent(new Event('resize'))
    }
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up)
  }
  function reset() { width = initial; if (storeKey) save('split.' + storeKey, width); window.dispatchEvent(new Event('resize')) }
</script>

<div class="split" class:dragging>
  <div class="pane a" style:flex="0 0 {width}px">{@render a()}</div>
  <div class="handle" role="separator" aria-orientation="vertical" tabindex="-1" title="Drag to resize · double-click to reset" onpointerdown={down} ondblclick={reset}></div>
  <div class="pane b">{@render b()}</div>
</div>

<style>
  .split { display: flex; min-height: 0; height: 100%; min-width: 0; flex: 1; }
  .pane { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
  .pane.b { flex: 1 1 auto; }
  .handle { flex: 0 0 11px; cursor: col-resize; position: relative; }
  .handle::after { content: ''; position: absolute; left: 5px; top: 0; bottom: 0; width: 1px; background: var(--arc-line); transition: background var(--ease); }
  .handle:hover::after, .dragging .handle::after { background: var(--arc-accent); }
  .dragging { cursor: col-resize; user-select: none; }
</style>
