<script lang="ts">
  import type { Snippet } from 'svelte'
  import { previewUrl, previewSrcset, audioUrl, audioImageUrl, isAudioPath, paletteUrl } from '../api/media'
  import { addToCollection } from '../stores/app.svelte'
  import { openViewer } from '../stores/viewer.svelte'
  import MarksBadge from './MarksBadge.svelte'
  import type { Marks } from '../../tabs/search/types'
  import Button from '../ui/Button.svelte'
  import Checkbox from '../ui/Checkbox.svelte'

  /**
   * One result: a picture (or a sound with its waveform) and the row of
   * actions under it. A group with several paths becomes a carousel, so the
   * search tabs and the moodboard share one card.
   */
  let {
    paths, marks = [], scores = [], spectrogram = false, selectable = true, selected = $bindable(false),
    frame = $bindable(0), title = '', width = 900, sizes = '(max-width: 900px) 90vw, 42vw', palette = 0, db = '',
    header, actions, onstep, onframe,
  }: {
    paths: string[]; marks?: Marks[]; scores?: number[]; spectrogram?: boolean; selectable?: boolean; selected?: boolean
    frame?: number; title?: string; width?: number; sizes?: string; palette?: number; db?: string
    header?: Snippet; actions?: Snippet; onstep?: (path: string) => void; onframe?: (i: number) => void
  } = $props()

  const n = $derived(paths.length)
  const path = $derived(paths[Math.min(frame, n - 1)] ?? '')
  const audio = $derived(isAudioPath(path))
  function step(d: number) { frame = (frame + d + n) % n; onframe?.(frame) }
</script>

<article class="card">
  {#if title || header}
    <div class="head dim">{#if title}{title}{/if}{@render header?.()}{#if n > 1}<span class="twins">· twins: {n}</span>{/if}</div>
  {/if}
  <div class="media" class:has-nav={n > 1}>
    {#if audio}
      <img class="wave" src={audioImageUrl(path, spectrogram)} alt="" />
      <audio src={audioUrl(path)} controls preload="none"></audio>
    {:else}
      <button class="pic" onclick={() => openViewer(paths, frame, title)} title="Open">
        <img src={previewUrl(path, width)} srcset={previewSrcset(path)} {sizes} alt="" loading="lazy" decoding="async" />
      </button>
    {/if}
    {#if n > 1}
      <button class="nav l" onclick={() => step(-1)} aria-label="Previous twin">◀</button>
      <button class="nav r" onclick={() => step(1)} aria-label="Next twin">▶</button>
      <span class="counter mono">{frame + 1}/{n}</span>
    {/if}
  </div>
  {#if palette && !audio}
    <img class="swatch" src={paletteUrl(path, palette, width, 20, db)} alt="" loading="lazy" />
  {/if}
  <div class="row acts">
    {#if selectable}<Checkbox bind:checked={selected} label="Select" />{/if}
    <Button size="sm" kind="ghost" onclick={() => addToCollection([path])}>+ Collection</Button>
    {#if onstep}<Button size="sm" kind="primary" title="Make this the reference and search again" onclick={() => onstep(path)}>Step here</Button>{/if}
    {@render actions?.()}
    {#if scores[frame] !== undefined}<span class="score mono faint">{scores[frame].toFixed(2)}</span>{/if}
  </div>
  <MarksBadge marks={marks[frame]} />
</article>

<style>
  .card { display: flex; flex-direction: column; gap: 8px; padding: 10px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); min-width: 0; }
  .head { font-size: var(--fs-xs); display: flex; gap: 6px; align-items: baseline; }
  .media { position: relative; overflow: hidden; border-radius: var(--r-md); background: var(--arc-surface-2); display: flex; flex-direction: column; gap: 6px; }
  .pic { padding: 0; border: none; background: transparent; cursor: zoom-in; display: block; width: 100%; }
  .pic img, .wave { width: 100%; display: block; border-radius: var(--r-md); }
  audio { width: 100%; }
  .nav { position: absolute; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border-radius: 50%; border: none; background: rgba(0,0,0,0.55); color: #fff; font-size: 13px; opacity: 0; transition: opacity var(--ease); }
  .media:hover .nav { opacity: 1; }
  .nav.l { left: 8px; } .nav.r { right: 8px; }
  .counter { position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); font-size: var(--fs-xs); padding: 2px 8px; border-radius: var(--r-pill); background: rgba(20,19,17,0.7); color: var(--arc-ink); }
  .swatch { width: 100%; height: 20px; border-radius: var(--r-sm); object-fit: fill; }
  .acts { gap: 10px; }
  .score { margin-left: auto; font-size: var(--fs-xs); }
  .twins { color: var(--arc-ink-faint); }
</style>
