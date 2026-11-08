<script lang="ts">
  import type { Snippet } from 'svelte'
  let { open = $bindable(false), title = '', width = '520px', full = false, onclose, children, actions }: {
    open?: boolean; title?: string; width?: string; full?: boolean; onclose?: () => void; children?: Snippet; actions?: Snippet
  } = $props()
  function close() { open = false; onclose?.() }
  function key(e: KeyboardEvent) { if (e.key === 'Escape') close() }
</script>

<svelte:window onkeydown={open ? key : undefined} />

{#if open}
  <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
  <div class="scrim" onclick={(e) => { if (e.target === e.currentTarget) close() }}>
    <div class="dlg" class:full style:width={full ? null : width} role="dialog" aria-modal="true" aria-label={title}>
      {#if title || actions}
        <header class="head">
          <h3 class="title ellipsis">{title}</h3>
          <div class="acts">{@render actions?.()}<button class="x" title="Close (Esc)" onclick={close}>×</button></div>
        </header>
      {/if}
      <div class="body">{@render children?.()}</div>
    </div>
  </div>
{/if}

<style>
  .scrim { position: fixed; inset: 0; z-index: 900; background: rgba(14, 13, 12, 0.78); display: flex; align-items: center; justify-content: center; padding: var(--sp-6); animation: arc-fade-in 0.12s ease; }
  .dlg { background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); box-shadow: var(--shadow-2); max-width: 100%; max-height: 100%; display: flex; flex-direction: column; min-height: 0; }
  .dlg.full { width: 100%; height: 100%; }
  .head { display: flex; align-items: center; gap: var(--sp-3); padding: 10px 14px; border-bottom: 1px solid var(--arc-line); }
  .title { flex: 1; font-size: var(--fs-sm); font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
  .acts { display: flex; align-items: center; gap: var(--sp-2); }
  .x { border: none; background: transparent; color: var(--arc-ink-dim); font-size: 20px; line-height: 1; padding: 0 4px; }
  .x:hover { color: var(--arc-ink); }
  .body { padding: var(--sp-4); overflow: auto; flex: 1; min-height: 0; display: flex; flex-direction: column; gap: var(--sp-3); }
  .full .body { padding: 0; }
</style>
