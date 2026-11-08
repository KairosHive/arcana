<script lang="ts">
  import type { Snippet } from 'svelte'
  /** A bordered surface. `level` sets how recessed it sits: 1 is a card on the
      page, 2 a well inside a card. `accent` draws a coloured left edge. */
  let { title = '', subtitle = '', level = 1, accent = '', pad = 'md', scroll = false, children, actions, ...rest }: {
    title?: string; subtitle?: string; level?: 1 | 2; accent?: string; pad?: 'sm' | 'md' | 'lg' | 'none'
    scroll?: boolean; children?: Snippet; actions?: Snippet; [key: string]: unknown
  } = $props()
</script>

<section class="card l{level} pad-{pad}" class:scroll style:border-left-color={accent || null} class:accented={!!accent} {...rest}>
  {#if title || actions}
    <header class="head">
      <div class="titles">
        {#if title}<h3 class="title">{title}</h3>{/if}
        {#if subtitle}<p class="sub">{subtitle}</p>{/if}
      </div>
      {#if actions}<div class="acts">{@render actions()}</div>{/if}
    </header>
  {/if}
  {@render children?.()}
</section>

<style>
  .card { border: 1px solid var(--arc-line); border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; gap: var(--sp-3); }
  .scroll { overflow-y: auto; min-height: 0; }
  .l1 { background: var(--arc-surface); }
  .l2 { background: var(--arc-surface-2); border-radius: var(--r-md); gap: var(--sp-2); }
  .pad-md { padding: var(--sp-4); }
  .pad-sm { padding: var(--sp-3); }
  .pad-lg { padding: var(--sp-5) var(--sp-6); }
  .pad-none { padding: 0; }
  .accented { border-left-width: 3px; }
  .head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--sp-3); }
  .title { font-size: var(--fs-md); font-weight: 700; letter-spacing: 0.3px; }
  .sub { font-size: var(--fs-sm); color: var(--arc-ink-dim); margin-top: 2px; max-width: 70ch; line-height: 1.5; }
  .acts { display: flex; align-items: center; gap: var(--sp-2); flex: 0 0 auto; }
</style>
