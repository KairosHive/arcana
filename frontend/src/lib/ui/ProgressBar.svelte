<script lang="ts">
  /** fraction null draws an indeterminate sweep. */
  let { fraction = null, tone = 'accent', height = '6px' }: {
    fraction?: number | null; tone?: 'accent' | 'ok' | 'warn' | 'bad'; height?: string
  } = $props()
</script>

<div class="bar tone-{tone}" style:height role="progressbar" aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}>
  {#if fraction === null}
    <div class="sweep"></div>
  {:else}
    <div class="fill" style:width="{Math.max(0, Math.min(1, fraction)) * 100}%"></div>
  {/if}
</div>

<style>
  .bar { width: 100%; border-radius: var(--r-pill); background: var(--arc-line); overflow: hidden; position: relative; }
  .fill { height: 100%; background: var(--c); transition: width 0.3s; border-radius: inherit; }
  .sweep { position: absolute; top: 0; bottom: 0; width: 35%; border-radius: inherit; background: var(--c); animation: sweep 1.3s ease-in-out infinite; }
  .tone-accent { --c: var(--arc-accent); } .tone-ok { --c: var(--arc-ok); } .tone-warn { --c: var(--arc-warn); } .tone-bad { --c: var(--arc-bad); }
  @keyframes sweep { 0% { left: -35%; } 100% { left: 100%; } }
</style>
