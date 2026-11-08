<script lang="ts">
  let { value = $bindable(0), min = 0, max = 1, step = 0.01, label = '', disabled = false, format, onchange, oncommit, width = '', title = '' }: {
    value?: number; min?: number; max?: number; step?: number; label?: string; disabled?: boolean
    format?: (v: number) => string; onchange?: (v: number) => void; oncommit?: (v: number) => void; width?: string; title?: string
  } = $props()
  const shown = $derived(format ? format(value) : (Number.isInteger(step) ? String(Math.round(value)) : value.toFixed(step < 0.1 ? 2 : 1)))
  const pct = $derived(max > min ? ((value - min) / (max - min)) * 100 : 0)
</script>

<label class="sl" class:disabled style:width={width || null} {title}>
  {#if label}<span class="caps lab">{label}</span>{/if}
  <span class="track" style="--pct: {Math.max(0, Math.min(100, pct))}%">
    <input type="range" bind:value {min} {max} {step} {disabled} oninput={() => onchange?.(value)} onchange={() => oncommit?.(value)} />
  </span>
  <span class="val mono">{shown}</span>
</label>

<style>
  .sl { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .lab { flex: 0 0 auto; }
  .track { flex: 1 1 auto; min-width: 70px; display: flex; align-items: center; height: 18px; }
  input[type=range] {
    appearance: none; width: 100%; height: 4px; border-radius: var(--r-pill); margin: 0;
    background: linear-gradient(to right, var(--arc-accent) var(--pct), var(--arc-line) var(--pct));
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb { appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--arc-ink); border: 2px solid var(--arc-accent); }
  input[type=range]::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%; background: var(--arc-ink); border: 2px solid var(--arc-accent); }
  input[type=range]:focus-visible { box-shadow: var(--focus); outline: none; }
  .val { font-size: var(--fs-xs); color: var(--arc-ink-dim); min-width: 3.5ch; text-align: right; flex: 0 0 auto; }
  .disabled { opacity: 0.5; }
</style>
