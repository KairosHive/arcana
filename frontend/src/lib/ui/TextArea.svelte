<script lang="ts">
  let { value = $bindable(''), placeholder = '', rows = 3, disabled = false, mono = false, ...rest }: {
    value?: string; placeholder?: string; rows?: number; disabled?: boolean; mono?: boolean; [key: string]: unknown
  } = $props()
</script>

<textarea class="ta" class:mono bind:value {placeholder} {rows} {disabled} {...rest}></textarea>

<style>
  .ta {
    width: 100%; padding: 8px var(--ctl-pad-x); resize: vertical;
    background: var(--arc-surface-2); color: var(--arc-ink);
    border: 1px solid var(--arc-line); border-radius: var(--r-md);
    font-size: var(--fs-md); line-height: var(--lh-normal); caret-color: var(--arc-accent);
  }
  .ta::placeholder { color: var(--arc-ink-faint); }
  .ta:focus { border-color: var(--arc-accent); box-shadow: var(--focus); outline: none; }
  .ta:disabled { opacity: 0.5; }
</style>
