<script lang="ts">
  let {
    value = $bindable(0), min, max, step = 1, width = '64px', disabled = false, suffix = '', size = 'md', onchange,
  }: {
    value?: number; min?: number; max?: number; step?: number | 'any'; width?: string; disabled?: boolean
    suffix?: string; size?: 'sm' | 'md'; onchange?: (v: number) => void
  } = $props()
  function clamp(v: number) {
    if (Number.isNaN(v)) return value
    if (min !== undefined) v = Math.max(min, v)
    if (max !== undefined) v = Math.min(max, v)
    return v
  }
  function commit(e: Event) {
    const v = clamp(parseFloat((e.target as HTMLInputElement).value))
    value = v
    onchange?.(v)
  }
</script>

<span class="num {size}" class:disabled>
  <input type="number" style:width {value} {min} {max} {step} {disabled} onchange={commit} />
  {#if suffix}<span class="suffix">{suffix}</span>{/if}
</span>

<style>
  .num { display: inline-flex; align-items: center; gap: 6px; }
  input {
    height: var(--ctl-h); padding: 0 6px; text-align: center;
    background: var(--arc-surface-2); color: var(--arc-ink);
    border: 1px solid var(--arc-line); border-radius: var(--r-md); font-size: var(--fs-md);
  }
  .sm input { height: var(--ctl-h-sm); font-size: var(--fs-sm); }
  input:focus { border-color: var(--arc-accent); box-shadow: var(--focus); outline: none; }
  input::-webkit-inner-spin-button { opacity: 0.4; }
  .suffix { font-size: var(--fs-sm); color: var(--arc-ink-dim); white-space: nowrap; }
  .disabled { opacity: 0.5; }
</style>
