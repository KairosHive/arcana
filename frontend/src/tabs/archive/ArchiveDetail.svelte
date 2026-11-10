<script lang="ts">
  import { previewUrl, thumbUrl, boardPageUrl, downloadUrl, basename } from '../../lib/api/media'
  import { openViewer } from '../../lib/stores/viewer.svelte'
  import type { Shown } from './archive.svelte'

  /**
   * What a kept thing is beyond its pictures. A story reads as a story, each
   * sentence beside the picture found for it. A set, or a folder copied from
   * search, leads with the searches that filled it. A label says what it was
   * measured on, a LoRA shows its checkpoints' sheets side by side, and a board
   * page opens as the page the agent made.
   */
  let { shown }: { shown: Shown } = $props()
  const d = $derived(shown.detail ?? {})

  type Query = { text: string; dataset?: string; k?: number; added?: number; when?: number }
  const queries = $derived((d.queries ?? []) as Query[])
  const when = (t?: number) => (t ? new Date(t * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '')

  function originText(o: Record<string, any>): string {
    if (!o) return ''
    if (o.kind === 'set') return `the ${o.from_index ?? ''} pictures of the set “${o.set}”${o.crops ? `, ${o.crops} crops each` : ''}`
    if (o.kind === 'words') return `the ${o.n} best matches for “${o.text}” in ${o.dataset}`
    if (o.kind === 'picture') return `one picture, ${o.picture}`
    return ''
  }
  const hasPage = $derived((d.files ?? []).some((f: string) => f.endsWith('.page.html')))
  const scenePics = $derived((d.scenes ?? []).map((s: any) => s.poetry || s.original).filter(Boolean))
</script>

{#if shown.kind === 'stories'}
  <div class="story">
    {#each d.scenes ?? [] as s, i (i)}
      <section class="scene">
        <p class="sentence"><span class="n">{s.n}</span>{s.text || '(no sentence)'}</p>
        <div class="pair" class:two={s.original && s.poetry}>
          {#if s.original}<button class="pic" onclick={() => openViewer(scenePics, scenePics.indexOf(s.poetry || s.original), shown.name)}><img src={previewUrl(s.original, 900)} alt={s.text} loading="lazy" />{#if s.poetry}<span class="tag">found</span>{/if}</button>{/if}
          {#if s.poetry}<button class="pic" onclick={() => openViewer(scenePics, scenePics.indexOf(s.poetry), shown.name)}><img src={previewUrl(s.poetry, 900)} alt={s.text} loading="lazy" /><span class="tag">poetry</span></button>{/if}
        </div>
      </section>
    {/each}
  </div>
{:else}
  {#if queries.length}
    <div class="queries">
      <span class="caps">{shown.kind === 'searches' ? 'Copied from the search' : 'Gathered by'}{queries.length > 1 ? 'es' : ''}</span>
      {#each queries as q}
        <div class="q">
          <span class="words">“{q.text || '(no words)'}”</span>
          <span class="meta dim">{[q.dataset, q.k ? `${q.k} results asked` : '', q.added !== undefined ? `${q.added} added` : '', when(q.when)].filter(Boolean).join('  ·  ')}</span>
        </div>
      {/each}
      {#if d.guessed}<span class="faint small">Read from the folder's name: this copy was made before searches were recorded with it{d.date ? `, on ${d.date}` : ''}.</span>{/if}
    </div>
  {/if}
  {#if d.author === 'agent'}<p class="agent small">Made by an agent{d.brief ? `, from the brief “${d.brief}”` : ''}.</p>{/if}
  {#if d.concept}<p class="concept">{d.concept}</p>{/if}

  {#if shown.kind === 'labels'}
    <div class="facts">
      <span>Measured on {originText(d.origin)}.</span>
      {#if d.tightness !== null && d.tightness !== undefined}<span class="dim">Alike {Number(d.tightness).toFixed(2)}: the mean cosine of a member to the centre. Near 1 is one look; lower draws as an average.</span>{/if}
    </div>
  {/if}

  {#if shown.kind === 'loras'}
    <div class="facts">
      {#if d.trigger}<span>Trigger <b>“{d.trigger}”</b></span>{/if}
      {#if d.prompt}<span class="dim">"Picture in a learned look" prompts with “{d.prompt}”.</span>{/if}
    </div>
    <div class="cps">
      {#each d.checkpoints ?? [] as c}
        <figure>
          {#if c.sheet}<button class="pic" onclick={() => openViewer((d.checkpoints ?? []).map((x: any) => x.sheet).filter(Boolean), (d.checkpoints ?? []).filter((x: any) => x.sheet).indexOf(c), shown.name)}><img src={previewUrl(c.sheet, 1400)} alt="" loading="lazy" /></button>{/if}
          <figcaption class="dim small">step {c.step ?? basename(c.name)}  ·  {c.name}</figcaption>
        </figure>
      {/each}
    </div>
  {/if}

  {#if shown.kind === 'pages'}
    <div class="facts">
      <div class="row">
        {#if hasPage}<a class="link" href={boardPageUrl(d.slug, Date.now())} target="_blank" rel="noopener">Open the page</a>{/if}
        {#each (d.files ?? []).filter((f: string) => !f.endsWith('.page.html')) as f}<a class="link" href={downloadUrl(f)}>{basename(f)}</a>{/each}
      </div>
    </div>
    {#if hasPage}<iframe class="page" title={shown.name} src={boardPageUrl(d.slug, Date.now())}></iframe>{/if}
  {/if}
{/if}

<style>
  .story { display: flex; flex-direction: column; gap: 28px; padding: 6px 2px 20px; }
  .scene { display: flex; flex-direction: column; gap: 10px; }
  .sentence { margin: 0; font-family: var(--font-serif, Georgia, serif); font-size: 1.25rem; line-height: 1.45; color: var(--arc-ink); }
  .sentence .n { display: inline-block; min-width: 1.6em; color: var(--arc-ink-faint); font-size: 0.85em; }
  .pair { display: grid; grid-template-columns: 1fr; gap: 8px; }
  .pair.two { grid-template-columns: 1fr 1fr; }
  .pic { position: relative; padding: 0; border: 0; background: none; cursor: zoom-in; border-radius: var(--r-md); overflow: hidden; }
  .pic img { width: 100%; display: block; }
  .tag { position: absolute; left: 8px; top: 8px; font-size: var(--fs-2xs); background: rgba(0, 0, 0, 0.55); color: #fff; padding: 2px 6px; border-radius: 3px; }
  .queries { display: flex; flex-direction: column; gap: 6px; padding: 10px 12px; margin-bottom: 10px; border: 1px solid var(--arc-line); border-left: 3px solid var(--arc-accent); border-radius: var(--r-md); background: var(--arc-surface-2); }
  .q { display: flex; flex-direction: column; gap: 2px; }
  .words { font-size: 1.15rem; color: var(--arc-ink); }
  .meta, .small { font-size: var(--fs-xs); }
  .agent { color: var(--arc-warn); margin: 0 0 6px; }
  .concept { margin: 0 0 10px; font-style: italic; color: var(--arc-ink-dim); }
  .facts { display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; font-size: var(--fs-sm); }
  .cps { display: flex; flex-direction: column; gap: 14px; }
  figure { margin: 0; display: flex; flex-direction: column; gap: 4px; }
  .link { color: var(--arc-accent); font-size: var(--fs-sm); }
  .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .page { width: 100%; height: 70vh; border: 1px solid var(--arc-line); border-radius: var(--r-md); background: #fff; }
</style>
