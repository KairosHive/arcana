<script lang="ts">
  import { toasts, dismiss } from '../stores/toast.svelte'
</script>

<div class="toasts" aria-live="polite">
  {#each toasts.list as t (t.id)}
    <div class="toast {t.tone}" role="status">
      <span class="msg">{t.text}</span>
      <button class="x" onclick={() => dismiss(t.id)} aria-label="Dismiss">×</button>
    </div>
  {/each}
</div>

<style>
  .toasts { position: fixed; right: 16px; bottom: 16px; z-index: 1200; display: flex; flex-direction: column; gap: 8px; max-width: 420px; }
  .toast { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: var(--r-md); background: var(--arc-surface); border: 1px solid var(--arc-line); border-left: 3px solid var(--c); box-shadow: var(--shadow-1); font-size: var(--fs-sm); animation: arc-fade-in 0.15s ease; }
  .msg { flex: 1; white-space: pre-wrap; }
  .x { border: none; background: transparent; color: var(--arc-ink-dim); font-size: 16px; line-height: 1; padding: 0; }
  .ok { --c: var(--arc-ok); } .warn { --c: var(--arc-warn); } .bad { --c: var(--arc-bad); } .info { --c: var(--arc-accent); }
</style>
