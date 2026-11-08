<script lang="ts">
  let { checked = $bindable(false), label = '', disabled = false, size = 'md', onchange }: {
    checked?: boolean; label?: string; disabled?: boolean; size?: 'sm' | 'md'; onchange?: (v: boolean) => void
  } = $props()
</script>

<label class="tg {size}" class:disabled>
  <input type="checkbox" bind:checked {disabled} onchange={() => onchange?.(checked)} />
  <span class="track" aria-hidden="true"><span class="knob"></span></span>
  {#if label}<span class="lab">{label}</span>{/if}
</label>

<style>
  .tg { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; position: relative; }
  .tg input { position: absolute; opacity: 0; width: 0; height: 0; }
  .track { width: 32px; height: 18px; border-radius: var(--r-pill); background: var(--arc-surface-3); border: 1px solid var(--arc-line); position: relative; transition: background var(--ease); flex: 0 0 auto; }
  .sm .track { width: 26px; height: 15px; }
  .knob { position: absolute; top: 2px; left: 2px; width: 12px; height: 12px; border-radius: 50%; background: var(--arc-ink-dim); transition: transform var(--ease), background var(--ease); }
  .sm .knob { width: 9px; height: 9px; }
  input:checked + .track { background: color-mix(in srgb, var(--arc-accent) 35%, var(--arc-surface-2)); border-color: var(--arc-accent); }
  input:checked + .track .knob { transform: translateX(14px); background: var(--arc-accent); }
  .sm input:checked + .track .knob { transform: translateX(11px); }
  input:focus-visible + .track { box-shadow: var(--focus); }
  .lab { font-size: var(--fs-sm); color: var(--arc-ink); white-space: nowrap; }
  .disabled { opacity: 0.5; cursor: not-allowed; }
</style>
