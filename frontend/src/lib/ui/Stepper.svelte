<script lang="ts">
  /** The numbered rail of a guided flow. Done steps are ticked, the current one
      filled, the rest outlined; clicking a reachable step moves there. */
  let { steps, current = $bindable(0), done = [], reachable = [], onselect }: {
    steps: { title: string; hint?: string }[]; current?: number; done?: number[]; reachable?: number[]
    onselect?: (i: number) => void
  } = $props()
</script>

<ol class="stepper">
  {#each steps as s, i}
    {@const can = reachable.includes(i) || done.includes(i) || i === current}
    <li class:on={i === current} class:done={done.includes(i)} class:can>
      <button disabled={!can} onclick={() => { current = i; onselect?.(i) }}>
        <span class="n">{done.includes(i) && i !== current ? '✓' : i + 1}</span>
        <span class="txt"><span class="t">{s.title}</span>{#if s.hint}<span class="h">{s.hint}</span>{/if}</span>
      </button>
    </li>
  {/each}
</ol>

<style>
  .stepper { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
  li button { display: flex; align-items: flex-start; gap: 10px; width: 100%; text-align: left; background: transparent; border: none; padding: 8px 10px; border-radius: var(--r-md); color: var(--arc-ink-dim); }
  li.can button:hover { background: var(--arc-surface-2); }
  li button:disabled { cursor: default; opacity: 0.55; }
  .n { flex: 0 0 auto; width: 22px; height: 22px; border-radius: 50%; border: 1px solid var(--arc-line-strong); display: inline-flex; align-items: center; justify-content: center; font-size: var(--fs-xs); font-weight: 700; }
  li.on .n { background: var(--arc-accent); border-color: var(--arc-accent); color: var(--arc-accent-ink); }
  li.done .n { border-color: var(--arc-ok); color: var(--arc-ok); }
  li.on .t { color: var(--arc-ink); }
  .txt { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
  .t { font-size: var(--fs-md); font-weight: 600; }
  .h { font-size: var(--fs-xs); color: var(--arc-ink-faint); }
</style>
