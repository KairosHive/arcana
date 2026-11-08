<script lang="ts">
  import { onMount } from 'svelte'
  import { app, refreshDatasets, onStateChange, type Job } from '../../lib/stores/app.svelte'
  import { get, post, reason } from '../../lib/api/client'
  import { watchJob } from '../../lib/stores/jobs.svelte'
  import { toast } from '../../lib/stores/toast.svelte'
  import Wizard from './Wizard.svelte'
  import DatasetList from './DatasetList.svelte'
  import EncoderList from './EncoderList.svelte'
  import JobCard from '../../lib/components/JobCard.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'

  import type { DatasetRow, ModelRow } from './types'

  let rows = $state<DatasetRow[]>([])
  let models = $state<ModelRow[]>([])
  let agentCommand = $state('')
  let loading = $state(true)
  let job = $state<Job | null>(null)
  let watcher: { stop: () => void } | null = null

  async function reload() {
    try {
      const d = await get('/api/datasets')
      rows = d.rows; models = d.models; agentCommand = d.agentCommand
    } catch (e) {
      toast(reason(e), 'bad')
    } finally {
      loading = false
    }
  }

  /** Follow a job this tab started, or one already running when the tab opened. */
  function follow(id: string) {
    watcher?.stop()
    watcher = watchJob(id, (j) => { job = j }, async () => {
      await post('/api/datasets/invalidate').catch(() => {})
      await reload()
      await refreshDatasets()
    })
  }

  onMount(() => {
    reload()
    if (app.job && !['done', 'failed', 'cancelled'].includes(app.job.status)) follow(app.job.id)
    return onStateChange(() => { /* boards changed: nothing here */ })
  })

  // A job started elsewhere (a terminal, an agent) shows here too.
  $effect(() => {
    const j = app.job
    if (j && (!job || job.id !== j.id) && !['done', 'failed', 'cancelled'].includes(j.status)) follow(j.id)
  })

  function started(id: string) { follow(id) }
</script>

<div class="tab-scroll">
  <div class="col wide">
    {#if job}
      <JobCard {job} ondismiss={() => { job = null }} />
    {/if}

    <Wizard {models} onstarted={started} busy={!!job && !['done', 'failed', 'cancelled'].includes(job.status)} />

    <DatasetList {rows} {loading} onstarted={started} onchanged={reload} busy={!!job && !['done', 'failed', 'cancelled'].includes(job.status)} />

    <Disclosure label="Encoders and labels" note="{models.filter((m) => m.downloaded).length} of {models.length} downloaded">
      <p class="hint">Downloaded once each. A dataset can only be searched with the encoder that built it, which is why changing encoder means re-indexing.</p>
      <EncoderList {models} onstarted={started} busy={!!job && !['done', 'failed', 'cancelled'].includes(job.status)} />
    </Disclosure>

    <Disclosure label="Working with an agent">
      <p class="hint">An agent such as Claude Code can search these datasets, build moodboards and recolour pictures with the same ops you have here. It changes only boards it made, and sees thumbnails only if you allow it in the Moodboard tab. To connect Claude Code, run:</p>
      <pre class="cmd">{agentCommand}</pre>
    </Disclosure>
  </div>
</div>

<style>
  .tab-scroll { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; padding-right: 4px; }
  .wide { max-width: 1180px; gap: var(--sp-4); padding-bottom: var(--sp-6); }
  .cmd { background: var(--arc-surface-2); border: 1px solid var(--arc-line); border-radius: var(--r-md); padding: 10px 12px; user-select: all; white-space: pre-wrap; word-break: break-all; margin: 0; }
</style>
