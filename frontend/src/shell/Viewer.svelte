<script lang="ts">
  import { viewer, closeViewer, stepViewer } from '../lib/stores/viewer.svelte'
  import { app } from '../lib/stores/app.svelte'
  import { previewUrl, basename, isAudioPath, isVideoPath, videoUrl, sectionLabel, audioUrl, waveUrl } from '../lib/api/media'

  const path = $derived(viewer.items[viewer.at] ?? '')
  const detail = $derived(app.collection.details[path])

  function key(e: KeyboardEvent) {
    if (!viewer.open) return
    const t = e.target as HTMLElement | null
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return
    if (e.key === 'ArrowLeft') { stepViewer(-1); e.preventDefault() }
    else if (e.key === 'ArrowRight') { stepViewer(1); e.preventDefault() }
    else if (e.key === 'Escape') { closeViewer(); e.preventDefault() }
  }
</script>

<svelte:window onkeydown={key} />

{#if viewer.open && path}
  <div class="viewer" role="dialog" aria-modal="true">
    <div class="bar">
      <span class="where">{viewer.at + 1} of {viewer.items.length}{#if viewer.where}  ·  {viewer.where}{/if}</span>
      <span class="spacer"></span>
      <button class="round small" title="Close (Esc)" onclick={closeViewer}>×</button>
    </div>
    <div class="stage">
      <button class="round" title="Previous (←)" onclick={() => stepViewer(-1)}>‹</button>
      {#if isAudioPath(path)}
        <div class="audio"><img src={waveUrl(path)} alt="" /><audio src={audioUrl(path)} controls></audio></div>
      {:else if isVideoPath(path)}
        <!-- svelte-ignore a11y_media_has_caption -->
        {#key path}<video class="pic" src={videoUrl(path)} poster={previewUrl(path, 1400)} controls autoplay></video>{/key}
      {:else}
        <img class="pic" src={previewUrl(path, 2048, 88)} alt="" />
      {/if}
      <button class="round" title="Next (→)" onclick={() => stepViewer(1)}>›</button>
    </div>
    <div class="cap">
      {#if detail}
        <div class="head">
          {#if detail.kind === 'derived' || detail.kind === 'generated'}<span class="kind">{detail.kind.toUpperCase()}</span>{/if}
          {#if detail.role}<span class="dim">{detail.role}</span>{/if}
          {#if detail.section}<span class="dim">{detail.section}</span>{/if}
        </div>
        {#if detail.note}<div class="note">{detail.note}</div>{/if}
      {/if}
      <div class="file faint">{basename(path)}{#if sectionLabel(path)}  ·  {sectionLabel(path)}{/if}</div>
    </div>
  </div>
{/if}

<style>
  .viewer { position: fixed; inset: 0; z-index: 1000; background: rgba(14, 13, 12, 0.94); display: flex; flex-direction: column; animation: arc-fade-in 0.12s ease; }
  .bar { display: flex; align-items: center; padding: 14px 20px; font-size: var(--fs-md); }
  .stage { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; gap: 24px; padding: 0 20px; }
  .pic { max-width: calc(100vw - 180px); max-height: calc(100vh - 190px); object-fit: contain; border-radius: var(--r-md); }
  .audio { display: flex; flex-direction: column; gap: 12px; width: min(900px, calc(100vw - 180px)); }
  .audio img { width: 100%; border-radius: var(--r-md); } .audio audio { width: 100%; }
  .round { width: 52px; height: 52px; border-radius: 50%; border: none; background: rgba(20, 19, 17, 0.55); color: var(--arc-ink); font-size: 28px; flex: 0 0 auto; }
  .round.small { width: 40px; height: 40px; font-size: 24px; }
  .round:hover { background: rgba(200, 116, 63, 0.35); }
  .cap { text-align: center; margin: 10px auto 20px; max-width: 80ch; font-size: var(--fs-md); line-height: 1.5; }
  .head { display: flex; justify-content: center; gap: 10px; margin-bottom: 4px; }
  .kind { color: var(--arc-warn); font-weight: 700; letter-spacing: 0.6px; font-size: var(--fs-xs); }
  .file { font-size: var(--fs-xs); margin-top: 4px; }
</style>
