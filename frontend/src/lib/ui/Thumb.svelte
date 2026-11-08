<script lang="ts">
  import type { Snippet } from 'svelte'
  import { thumbUrl } from '../api/media'
  /** One picture tile. Every gallery and rail uses this so badges, selection
      and the hover look are the same everywhere. */
  let {
    path, size = 'var(--thumb)', selected = false, selectable = false, role = '', badge = '', badgeTone = 'warn',
    caption = '', title = '', dim = false, onselect, onopen, onremove, children,
  }: {
    path: string; size?: string; selected?: boolean; selectable?: boolean
    /** R or T: the two moodboard roles, each with its own hue */
    role?: '' | 'R' | 'T'; badge?: string; badgeTone?: 'warn' | 'accent' | 'ok' | 'dim'
    caption?: string; title?: string; dim?: boolean
    onselect?: (v: boolean) => void; onopen?: () => void; onremove?: () => void; children?: Snippet
  } = $props()
</script>

<div class="thumb" class:selected class:dim style="--sz: {size}" title={title || path}>
  <button class="pic" class:R={role === 'R'} class:T={role === 'T'} onclick={onopen} tabindex={onopen ? 0 : -1} aria-label={caption || path}>
    <img src={thumbUrl(path)} alt="" loading="lazy" decoding="async" />
  </button>
  {#if selectable}
    <label class="sel" title="Select"><input type="checkbox" checked={selected} onchange={(e) => onselect?.((e.target as HTMLInputElement).checked)} /><span></span></label>
  {/if}
  {#if badge}<span class="badge {badgeTone}">{badge}</span>{/if}
  {#if onremove}<button class="rm" title="Remove" onclick={onremove}>×</button>{/if}
  {#if caption}<span class="cap ellipsis">{caption}</span>{/if}
  {@render children?.()}
</div>

<style>
  .thumb { position: relative; width: var(--sz); min-width: 0; display: flex; flex-direction: column; gap: 3px; }
  .pic { width: 100%; aspect-ratio: 1; padding: 0; border: 2px solid var(--arc-line); border-radius: var(--r-md); overflow: hidden; background: var(--arc-surface-2); cursor: zoom-in; transition: border-color var(--ease); }
  .pic img { width: 100%; height: 100%; object-fit: cover; }
  .thumb:hover .pic { border-color: var(--arc-line-strong); }
  .selected .pic { border-color: var(--arc-accent); }
  .pic.R { border-color: var(--arc-accent); border-width: 3px; }
  .pic.T { border-color: var(--arc-target); border-width: 3px; }
  .dim .pic { opacity: 0.45; }
  .sel { position: absolute; top: 6px; left: 6px; }
  .sel input { position: absolute; opacity: 0; width: 0; height: 0; }
  .sel span { display: block; width: 16px; height: 16px; border-radius: var(--r-sm); border: 1px solid var(--arc-ink); background: rgba(20,19,17,0.6); cursor: pointer; }
  .sel input:checked + span { background: var(--arc-accent); border-color: var(--arc-accent); }
  .badge { position: absolute; left: 6px; top: 6px; padding: 1px 5px; border-radius: var(--r-sm); font-size: var(--fs-2xs); font-weight: 700; letter-spacing: 0.4px; background: rgba(20,19,17,0.75); }
  .badge.warn { color: var(--arc-warn); } .badge.accent { color: var(--arc-accent); } .badge.ok { color: var(--arc-ok); } .badge.dim { color: var(--arc-ink-dim); }
  .rm { position: absolute; top: -6px; right: -6px; width: 20px; height: 20px; border-radius: 50%; border: 1px solid var(--arc-line); background: var(--arc-surface); color: var(--arc-ink-dim); font-size: 14px; line-height: 1; padding: 0; opacity: 0; transition: opacity var(--ease); }
  .thumb:hover .rm { opacity: 1; } .rm:hover { color: var(--arc-bad); border-color: var(--arc-bad); }
  .cap { font-size: var(--fs-2xs); color: var(--arc-ink-faint); }
</style>
