<script lang="ts">
  import { onMount } from 'svelte'
  import { activity, line, listen, setFollow, show, type Activity } from '../lib/stores/activity.svelte'
  import Toggle from '../lib/ui/Toggle.svelte'
  import Button from '../lib/ui/Button.svelte'

  /**
   * One line under the header: the last thing an agent or the terminal did,
   * with Show to open it in its tab, and Follow to do that every time. It
   * appears once something other than the page itself has run an op.
   */
  let open = $state(false)
  const theirs = $derived(activity.items.filter((i) => i.actor !== 'person'))
  const last = $derived<Activity | null>(theirs.length ? theirs[theirs.length - 1] : null)
  let follow = $state(activity.follow)
  $effect(() => { setFollow(follow) })

  onMount(() => { void listen() })

  function ago(t: number): string {
    const s = Math.max(0, Math.round(Date.now() / 1000 - t))
    return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`
  }
</script>

{#if last}
  <div class="strip">
    <span class="who" class:live={activity.live} title={activity.live ? 'Changes arrive as they happen' : 'Checking every two seconds'}>{last.actor}</span>
    <button class="what mono ellipsis" title={last.summary} onclick={() => (open = !open)}>{line(last)}</button>
    <span class="dim small">{ago(last.t)}</span>
    <Button size="sm" onclick={() => show(last)}>Show</Button>
    <Toggle bind:checked={follow} label="Follow" size="sm" />
  </div>
  {#if open}
    <div class="list">
      {#each [...theirs].reverse().slice(0, 12) as a (a.seq)}
        <div class="row">
          <span class="who small">{a.actor}</span>
          <span class="mono ellipsis grow" title={a.summary}>{line(a)}</span>
          <span class="dim small">{ago(a.t)}</span>
          <Button size="sm" kind="ghost" onclick={() => { show(a); open = false }}>Show</Button>
        </div>
      {/each}
    </div>
  {/if}
{/if}

<style>
  .strip { display: flex; align-items: center; gap: 10px; padding: 4px 14px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); flex: 0 0 auto; min-width: 0; }
  .who { font-size: var(--fs-2xs); font-weight: 700; letter-spacing: 0.4px; text-transform: uppercase; color: var(--tab-agent); padding: 1px 7px; border: 1px solid color-mix(in srgb, var(--tab-agent) 45%, transparent); border-radius: var(--r-pill); flex: 0 0 auto; }
  .who.live::before { content: '● '; }
  .what { flex: 1 1 auto; min-width: 0; text-align: left; border: none; background: transparent; color: var(--arc-ink); font-size: var(--fs-xs); padding: 0; cursor: pointer; }
  .list { display: flex; flex-direction: column; gap: 2px; padding: 6px 14px; background: var(--arc-surface); border: 1px solid var(--arc-line); border-radius: var(--r-lg); max-height: 240px; overflow-y: auto; flex: 0 0 auto; }
  .row { display: flex; align-items: center; gap: 10px; font-size: var(--fs-xs); min-width: 0; }
  .grow { flex: 1 1 auto; min-width: 0; }
</style>
