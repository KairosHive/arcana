<script lang="ts">
  import type { Snippet } from 'svelte'
  /** kind: primary = the one action in a card; success = a save with somewhere to go;
      danger = destructive; ghost = quiet; secondary = everything else. */
  let {
    kind = 'secondary',
    size = 'md',
    disabled = false,
    busy = false,
    title = '',
    type = 'button',
    onclick,
    children,
    ...rest
  }: {
    kind?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost'
    size?: 'sm' | 'md' | 'lg'
    disabled?: boolean
    busy?: boolean
    title?: string
    type?: 'button' | 'submit'
    onclick?: (e: MouseEvent) => void
    children?: Snippet
    [key: string]: unknown
  } = $props()
</script>

<button class="btn {kind} {size}" class:busy {type} {title} disabled={disabled || busy} {onclick} {...rest}>
  {#if busy}<span class="spin" aria-hidden="true"></span>{/if}
  {@render children?.()}
</button>

<style>
  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    height: var(--ctl-h); padding: 0 14px;
    border-radius: var(--r-md); border: 1px solid transparent;
    font-size: var(--fs-sm); font-weight: 600; line-height: var(--lh-tight);
    white-space: nowrap; flex: 0 0 auto;
    transition: background var(--ease), border-color var(--ease), color var(--ease), opacity var(--ease);
  }
  .sm { height: var(--ctl-h-sm); padding: 0 10px; font-size: var(--fs-xs); }
  .lg { height: var(--ctl-h-lg); padding: 0 20px; font-size: var(--fs-md); }
  .primary { background: var(--arc-accent); color: var(--arc-accent-ink); }
  .primary:hover:not(:disabled) { background: color-mix(in srgb, var(--arc-accent) 88%, white); }
  .success { background: var(--arc-ok); color: var(--arc-ok-ink); }
  .success:hover:not(:disabled) { background: color-mix(in srgb, var(--arc-ok) 88%, white); }
  .secondary { background: transparent; color: var(--arc-ink); border-color: var(--arc-line); }
  .secondary:hover:not(:disabled) { border-color: var(--arc-accent); color: var(--arc-accent); }
  .danger { background: transparent; color: var(--arc-bad); border-color: color-mix(in srgb, var(--arc-bad) 35%, transparent); }
  .danger:hover:not(:disabled) { background: color-mix(in srgb, var(--arc-bad) 14%, transparent); border-color: var(--arc-bad); }
  .ghost { background: transparent; color: var(--arc-ink-dim); }
  .ghost:hover:not(:disabled) { color: var(--arc-ink); background: var(--arc-surface-2); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .spin {
    width: 11px; height: 11px; border-radius: 50%;
    border: 2px solid currentColor; border-right-color: transparent;
    animation: arc-spin 0.8s linear infinite;
  }
</style>
