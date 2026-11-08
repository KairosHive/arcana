<script lang="ts">
  let {
    value = $bindable(''),
    placeholder = '',
    disabled = false,
    size = 'md',
    mono = false,
    onenter,
    oncommit,
    ...rest
  }: {
    value?: string; placeholder?: string; disabled?: boolean; size?: 'sm' | 'md'; mono?: boolean
    onenter?: (v: string) => void
    /** fires on Enter and on blur, for fields that act once a value is settled */
    oncommit?: (v: string) => void
    [key: string]: unknown
  } = $props()
  function key(e: KeyboardEvent) {
    if (e.key === 'Enter') { onenter?.(value); oncommit?.(value) }
  }
</script>

<input class="in {size}" class:mono type="text" bind:value {placeholder} {disabled}
       onkeydown={key} onblur={() => oncommit?.(value)} {...rest} />

<style>
  .in {
    height: var(--ctl-h); padding: 0 var(--ctl-pad-x); width: 100%;
    background: var(--arc-surface-2); color: var(--arc-ink);
    border: 1px solid var(--arc-line); border-radius: var(--r-md);
    font-size: var(--fs-md); caret-color: var(--arc-accent);
    transition: border-color var(--ease);
  }
  .sm { height: var(--ctl-h-sm); font-size: var(--fs-sm); padding: 0 8px; }
  .in::placeholder { color: var(--arc-ink-faint); }
  .in:hover:not(:disabled) { border-color: var(--arc-line-strong); }
  .in:focus { border-color: var(--arc-accent); box-shadow: var(--focus); outline: none; }
  .in:disabled { opacity: 0.5; }
  .in:-webkit-autofill { -webkit-text-fill-color: var(--arc-ink); -webkit-box-shadow: 0 0 0 1000px var(--arc-surface-2) inset; }
</style>
