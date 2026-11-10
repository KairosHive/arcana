<script lang="ts">
  import type { Snippet } from 'svelte'
  import { previewUrl, previewSrcset, audioUrl, audioImageUrl, isAudioPath, isVideoPath, videoUrl, sectionLabel, basename, paletteUrl } from '../api/media'
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
    paths, marks = [], scores = [], notes = [], strips = [], spectrogram = false, selectable = true, selected = $bindable(false),
    frame = $bindable(0), title = '', width = 900, sizes = '(max-width: 900px) 90vw, 42vw', palette = 0, db = '',
    header, actions, onstep, onframe,
  }: {
    paths: string[]; marks?: Marks[]; scores?: number[]; notes?: string[]; strips?: (Strip | null | undefined)[]; spectrogram?: boolean; selectable?: boolean; selected?: boolean
    frame?: number; title?: string; width?: number; sizes?: string; palette?: number; db?: string
    header?: Snippet; actions?: Snippet; onstep?: (path: string) => void; onframe?: (i: number) => void
  } = $props()

  const n = $derived(paths.length)
  const path = $derived(paths[Math.min(frame, n - 1)] ?? '')
  const audio = $derived(isAudioPath(path))
  const video = $derived(isVideoPath(path))
  /** Where in a whole video a search matches: its sections, each with a score from 0 to 1. */
  type Strip = { seconds: number; parts: [number, number, number][]; best: [number, number] }
  const strip = $derived(strips[frame] ?? null)
  let player: HTMLVideoElement | undefined = $state()
  function seek(t: number) { if (player) { player.currentTime = t; void player.play() } }
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
    {:else if video}
      <!-- A section of a video: its middle frame until played, then that range of the file. -->
      {#key path}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video bind:this={player} src={videoUrl(path)} poster={previewUrl(path, width)} controls preload="none" playsinline></video>
      {/key}
      <div class="clip faint"><span class="ellipsis">{basename(path)}</span><span class="mono">{sectionLabel(path)}</span></div>
      {#if strip}
        <div class="strip" title="Where this video matches the search. Click a part to play from there.">
          {#each strip.parts as [t0, t1, s] (t0)}
            <button class="part" class:best={t0 === strip.best[0]} style="flex: {Math.max(t1 - t0, 0.01)}; --s: {s}" onclick={() => seek(t0)}
                    aria-label="{t0.toFixed(1)} to {t1.toFixed(1)} seconds" title="{t0.toFixed(1)} to {t1.toFixed(1)} s"></button>
          {/each}
        </div>
      {/if}
      {#if notes[frame]}<div class="clip dim" title="How this section moves, read from optical flow"><span>{notes[frame]}</span></div>{/if}
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
  {#if palette && !audio && !video}
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
  video { width: 100%; display: block; border-radius: var(--r-md); background: #000; max-height: 70vh; }
  .strip { display: flex; gap: 1px; height: 14px; padding: 0 4px; }
  .part { min-width: 2px; padding: 0; border: none; border-radius: 2px; cursor: pointer; background: color-mix(in srgb, var(--arc-video) calc(var(--s) * 100%), var(--arc-surface-2)); }
  .part.best { outline: 1px solid var(--arc-ink); }
  .part:hover { filter: brightness(1.4); }
  .clip { display: flex; justify-content: space-between; gap: 10px; font-size: var(--fs-xs); padding: 0 4px 4px; }
  .nav { position: absolute; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border-radius: 50%; border: none; background: rgba(0,0,0,0.55); color: #fff; font-size: 13px; opacity: 0; transition: opacity var(--ease); }
  .media:hover .nav { opacity: 1; }
  .nav.l { left: 8px; } .nav.r { right: 8px; }
  .counter { position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); font-size: var(--fs-xs); padding: 2px 8px; border-radius: var(--r-pill); background: rgba(20,19,17,0.7); color: var(--arc-ink); }
  .swatch { width: 100%; height: 20px; border-radius: var(--r-sm); object-fit: fill; }
  .acts { gap: 10px; }
  .score { margin-left: auto; font-size: var(--fs-xs); }
  .twins { color: var(--arc-ink-faint); }
</style>
