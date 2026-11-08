<script lang="ts" generics="T extends string | number">
  type Option<T> = { value: T; label: string; disabled?: boolean }
  let {
    value = $bindable(), options = [], placeholder = '', disabled = false, size = 'md', width = '', onchange,
  }: {
    value?: T | null; options?: Option<T>[]; placeholder?: string; disabled?: boolean
    size?: 'sm' | 'md'; width?: string; onchange?: (v: T) => void
  } = $props()
  const numeric = $derived(options.length > 0 && typeof options[0].value === 'number')
  function pick(e: Event) {
    const raw = (e.target as HTMLSelectElement).value
    if (raw === '') { value = null; return }
    const v = (numeric ? Number(raw) : raw) as T
    value = v
    onchange?.(v)
  }
</script>

<span class="sel {size}" class:disabled style:width={width || null}>
  <select value={value === null || value === undefined ? '' : String(value)} {disabled} onchange={pick}>
    {#if placeholder}<option value="" disabled={value !== null && value !== undefined}>{placeholder}</option>{/if}
    {#each options as o (o.value)}
      <option value={String(o.value)} disabled={o.disabled}>{o.label}</option>
    {/each}
  </select>
  <span class="chev" aria-hidden="true">▾</span>
</span>

<style>
  .sel { position: relative; display: inline-flex; min-width: 0; }
  select {
    appearance: none; width: 100%; height: var(--ctl-h); padding: 0 26px 0 var(--ctl-pad-x);
    background: var(--arc-surface-2); color: var(--arc-ink);
    border: 1px solid var(--arc-line); border-radius: var(--r-md); font-size: var(--fs-md);
    text-overflow: ellipsis; cursor: pointer;
  }
  .sm select { height: var(--ctl-h-sm); font-size: var(--fs-sm); padding-left: 8px; }
  select:hover:not(:disabled) { border-color: var(--arc-line-strong); }
  select:focus { border-color: var(--arc-accent); box-shadow: var(--focus); outline: none; }
  select option { background: var(--arc-surface-2); color: var(--arc-ink); }
  .chev { position: absolute; right: 9px; top: 50%; transform: translateY(-50%); pointer-events: none; color: var(--arc-ink-dim); font-size: 11px; }
  .disabled { opacity: 0.5; }
</style>
