<script lang="ts">
  import type { Knob } from './alchemy.svelte'
  import Slider from '../../lib/ui/Slider.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'

  /** One knob of a method, drawn by its kind. The lead knob gets its doc under it. */
  let { knob, value = $bindable(), lead = false, onchange }: { knob: Knob; value: any; lead?: boolean; onchange?: () => void } = $props()
  const step = $derived(knob.step ?? (knob.kind === 'int' ? 1 : 0.01))
</script>

<div class="knob" class:lead title={knob.doc}>
  {#if knob.kind === 'float' || knob.kind === 'angle' || knob.kind === 'int'}
    <Slider bind:value label={knob.label} min={knob.lo ?? 0} max={knob.hi ?? 1} {step} onchange={onchange} />
  {:else if knob.kind === 'choice'}
    <label class="row nowrap"><span class="caps">{knob.label}</span><Select bind:value options={(knob.options ?? []).map((o) => ({ value: o, label: String(o) }))} size="sm" onchange={onchange} /></label>
  {:else if knob.kind === 'bool'}
    <Checkbox bind:checked={value} label={knob.label} onchange={onchange} />
  {:else}
    <label class="col"><span class="caps">{knob.label}</span><TextInput bind:value placeholder={knob.doc} size="sm" oncommit={onchange} /></label>
  {/if}
  {#if lead}<p class="doc hint">{knob.doc}</p>{/if}
</div>

<style>
  .knob { display: flex; flex-direction: column; gap: 4px; }
  .lead { background: var(--arc-surface-2); border-left: 3px solid var(--arc-accent); border-radius: var(--r-md); padding: 8px 10px; }
  .doc { font-size: var(--fs-xs); }
  .col { gap: 4px; }
</style>
