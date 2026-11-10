<script lang="ts">
  import { caps, type Modality } from '../stores/caps.svelte'
  import { current } from '../stores/app.svelte'
  /**
   * The tag that says what a feature works with: photo, audio, video.
   * Give a feature id from the capability table, or the modalities themselves.
   * A tag for a modality other than the chosen dataset's is drawn faint, so the
   * one that applies now stands out.
   */
  let { feature = '', modalities, dots = false }: { feature?: string; modalities?: Modality[]; dots?: boolean } = $props()
  const mods = $derived(modalities ?? (feature ? caps.of(feature) : []))
  const title = $derived(mods.length ? `Works with ${mods.map((m) => caps.label(m)).join(', ')}` : '')
</script>

{#if mods.length}
  <span class="tags" class:dots {title}>
    {#each mods as m (m)}
      {#if dots}
        <i class="dot {m}" class:other={!!current.dataset && current.modality !== m}></i>
      {:else}
        <span class="tag {m}" class:other={!!current.dataset && current.modality !== m}>{caps.label(m)}</span>
      {/if}
    {/each}
  </span>
{/if}

<style>
  .tags { display: inline-flex; align-items: center; gap: 4px; flex: 0 0 auto; vertical-align: middle; }
  .tag { display: inline-flex; align-items: center; height: 15px; padding: 0 6px; border-radius: var(--r-pill); font-size: var(--fs-2xs); font-weight: 600; letter-spacing: 0.3px; line-height: 1; white-space: nowrap; color: var(--c); border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); background: color-mix(in srgb, var(--c) 10%, transparent); }
  .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--c); display: block; }
  .image { --c: var(--arc-accent); } .audio { --c: var(--arc-audio); } .video { --c: var(--arc-video); }
  .other { opacity: 0.4; }
  .dots { gap: 3px; }
</style>
