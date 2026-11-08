<script lang="ts">
  import type { Job } from '../stores/app.svelte'
  import { jobTone, jobTitle, jobMessage, cancelJob } from '../stores/jobs.svelte'
  import ProgressBar from '../ui/ProgressBar.svelte'
  import Button from '../ui/Button.svelte'

  /** One running (or just finished) job: title, message, detail, a bar, Cancel. */
  let { job, ondismiss }: { job: Job; ondismiss?: () => void } = $props()
  const finished = $derived(['done', 'failed', 'cancelled'].includes(job.status))
  const fraction = $derived(job.status === 'done' ? 1 : job.fraction)
  const pct = $derived(fraction === null ? null : Math.round(fraction * 100))
</script>

<div class="job tone-{jobTone(job)}">
  <div class="head">
    <span class="title">{jobTitle(job)}</span>
    <span class="msg dim ellipsis">{jobMessage(job)}</span>
    {#if job.detail}<span class="detail faint ellipsis">{job.detail}</span>{/if}
    {#if pct !== null}<span class="pct mono">{pct}%</span>{/if}
  </div>
  <ProgressBar fraction={finished ? 1 : fraction} tone={jobTone(job)} />
  <div class="foot">
    {#if !finished}
      <Button size="sm" onclick={() => cancelJob(job.id)}>Cancel</Button>
    {:else if ondismiss}
      <Button size="sm" kind="ghost" onclick={ondismiss}>Dismiss</Button>
    {/if}
    {#if job.log?.length && !finished}<span class="log faint ellipsis">{job.log[job.log.length - 1]}</span>{/if}
  </div>
</div>

<style>
  .job { display: flex; flex-direction: column; gap: 10px; padding: 12px 14px; border-radius: var(--r-lg); background: var(--arc-surface); border: 1px solid var(--arc-line); border-left: 3px solid var(--c); }
  .tone-accent { --c: var(--arc-accent); } .tone-ok { --c: var(--arc-ok); } .tone-warn { --c: var(--arc-warn); } .tone-bad { --c: var(--arc-bad); }
  .head { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
  .title { font-weight: 700; font-size: var(--fs-md); flex: 0 0 auto; }
  .msg { flex: 1 1 auto; font-size: var(--fs-sm); }
  .detail { font-size: var(--fs-xs); max-width: 40%; }
  .pct { font-size: var(--fs-xs); color: var(--c); }
  .foot { display: flex; align-items: center; gap: 12px; min-height: 24px; }
  .log { font-size: var(--fs-xs); }
</style>
