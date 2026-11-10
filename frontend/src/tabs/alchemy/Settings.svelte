<script lang="ts">
  import { post, reason } from '../../lib/api/client'
  import { app } from '../../lib/stores/app.svelte'
  import { toast } from '../../lib/stores/toast.svelte'
  import { catalog, step, method, chooseMethod, applyRecipe, applies, resetFinish, coverageFor, materials, controls, WORDS, CUSTOM, type Knob as KnobT } from './alchemy.svelte'
  import Knob from './Knob.svelte'
  import Disclosure from '../../lib/ui/Disclosure.svelte'
  import Select from '../../lib/ui/Select.svelte'
  import Button from '../../lib/ui/Button.svelte'
  import Slider from '../../lib/ui/Slider.svelte'
  import NumberInput from '../../lib/ui/NumberInput.svelte'
  import TextInput from '../../lib/ui/TextInput.svelte'
  import Checkbox from '../../lib/ui/Checkbox.svelte'
  import Field from '../../lib/ui/Field.svelte'
  import Status from '../../lib/ui/Status.svelte'

  /**
   * The settings column, scoped from broad to narrow: a recipe that sets
   * everything at once; the pictures (who leads, shares, zoom); the method
   * and its knobs; the mask the step shows through; the finish on top.
   */
  let { onchange }: { onchange: () => void } = $props()
  const cat = $derived(catalog.data!)
  const m = $derived(method(step.method))

  let recipe = $state<string | null>(null)
  let recipeDoc = $state('')
  let recipeName = $state('')
  let recipeStatus = $state('')
  // Grouped by what a recipe does, in the order the groups are declared; your own recipes last.
  const recipeOptions = $derived(cat.recipes.map((r) => ({ value: r.name, label: r.name, group: r.own ? 'yours' : r.group })))
  $effect(() => { recipeDoc = cat.recipes.find((r) => r.name === recipe)?.doc ?? '' })

  async function apply() {
    if (!recipe) return
    try { recipeDoc = await applyRecipe(recipe); onchange() } catch (e) { toast(reason(e), 'bad') }
  }
  async function saveRecipe() {
    const name = recipeName.trim()
    if (!name) { recipeStatus = 'Give the recipe a name.'; return }
    try {
      const r = await post('/api/alchemy/recipe', { controls: controls(), name })
      cat.recipes = r.recipes
      recipeStatus = r.message
    } catch (e) { recipeStatus = reason(e) }
  }

  // A method that needs a graphics card this build cannot use is shown, and marked, but cannot be chosen.
  // Unlisted methods (superseded ones) stay out of the menu, except the one a film or saved step opened.
  const methodOptions = $derived(cat.methods.filter((x) => x.family === step.family && (x.listed !== false || x.id === step.method)).map((x) => {
    const noGpu = x.needs.includes('gpu') && !app.hardware.gpu
    return { value: x.id, label: x.label + (noGpu ? '  (needs a GPU)' : x.cost !== 'live' ? '  (seconds)' : ''), disabled: noGpu }
  }))
  function pickFamily(f: string) {
    step.family = f
    const first = cat.methods.find((x) => x.family === f && x.listed !== false)
    if (m?.family !== f && first) chooseMethod(first.id)
    onchange()
  }
  const methodDoc = $derived.by(() => {
    if (!m) return ''
    const [lo, hi] = m.materials
    let d = `${m.doc}. Takes ${lo === hi ? lo : `${lo} to ${hi}`} material${hi !== 1 ? 's' : ''}.`
    if (m.cost !== 'live') d += ' Takes several seconds, so it draws on Draw.'
    return d
  })
  const sections = $derived.by(() => {
    const parts: Record<string, KnobT[]> = { main: [], more: [], looks: [] }
    for (const k of m?.knobs ?? []) if (step.knobs[k.key] !== undefined && applies(k, { ...step.knobs, _materials: materials().length })) parts[k.section].push(k)
    return parts
  })

  const presetOptions = $derived.by(() => {
    const opts: { value: string; label: string; disabled?: boolean }[] = [{ value: 'none', label: 'none' }, { value: WORDS, label: 'words…' }, { value: CUSTOM, label: 'custom source…' }]
    for (const g of cat.maskGroups) {
      opts.push({ value: `__group ${g}`, label: `── ${g} ──`, disabled: true })
      for (const p of cat.maskPresets.filter((p) => p.group === g)) opts.push({ value: p.name, label: p.name })
    }
    return opts
  })
  function layerChanged(i: number) {
    const l = step.layers[i]
    if (l.choice !== WORDS && l.choice !== CUSTOM && l.choice !== 'none') l.coverage = coverageFor(l.choice)
    onchange()
  }
  const mats = $derived(materials())
  const finishGroups = $derived([...new Set(cat.finish.map((f) => f.group))])
  const finishCount = $derived(Object.keys(step.finish).filter((k) => step.finish[k] !== cat.finish.find((f) => f.key === k)?.default).length)
  const maskCount = $derived(step.layers.filter((l) => l.choice !== 'none').length)
</script>

<div class="settings">
  <Disclosure label="Recipe" note="sets only what it holds" open>
    <div class="row nowrap">
      <div class="grow"><Select bind:value={recipe} options={recipeOptions} placeholder="Choose a recipe…" size="sm" /></div>
      <Button size="sm" kind="primary" onclick={apply} disabled={!recipe}>Apply</Button>
    </div>
    {#if recipeDoc}<p class="hint">{recipeDoc}</p>{/if}
    <Disclosure label="keep these settings as a recipe" level={2}>
      <div class="row nowrap"><div class="grow"><TextInput bind:value={recipeName} placeholder="Recipe name…" size="sm" onenter={saveRecipe} /></div><Button size="sm" onclick={saveRecipe}>Save recipe</Button></div>
      <Status text={recipeStatus} size="xs" />
    </Disclosure>
  </Disclosure>

  <Disclosure label="Pictures" note="lead, shares, zoom">
    <Field label="Who leads">
      <div class="seg">
        <button class:on={step.lead === 'picture'} onclick={() => { step.lead = 'picture'; onchange() }}>the picture</button>
        <button class:on={step.lead === 'material'} onclick={() => { step.lead = 'material'; onchange() }}>the material</button>
      </div>
    </Field>
    <p class="hint">With the material leading, the picture is hidden in material [1].</p>
    {#if mats.length > 1}
      {#each mats as r}
        <Slider bind:value={step.w[r]} min={0} max={1} step={0.05} label="share of [{r}]" onchange={onchange} />
      {/each}
    {/if}
    <Slider bind:value={step.zoom} min={0.5} max={4} step={0.25} label="Material zoom" onchange={onchange} />
    <Field label="Crop and seed" inline><NumberInput bind:value={step.crop} min={0} max={99999} step={1} width="90px" size="sm" onchange={onchange} /></Field>
  </Disclosure>

  <Disclosure label="Method" open note={m?.label}>
    <div class="seg wrap">
      {#each cat.families as f}
        <button class:on={step.family === f.id} onclick={() => pickFamily(f.id)}>{f.label}</button>
      {/each}
    </div>
    <p class="hint">{cat.families.find((f) => f.id === step.family)?.doc}</p>
    <Select value={step.method} options={methodOptions} size="sm" onchange={(v) => { chooseMethod(String(v)); onchange() }} />
    <p class="hint">{methodDoc}</p>
    {#each sections.main as k, i (k.key)}
      <Knob knob={k} bind:value={step.knobs[k.key]} lead={i === 0} onchange={onchange} />
    {/each}
    {#if sections.more.length}
      <Disclosure label="more" level={2}>
        {#each sections.more as k (k.key)}<Knob knob={k} bind:value={step.knobs[k.key]} onchange={onchange} />{/each}
      </Disclosure>
    {/if}
    {#if sections.looks.length}
      <Disclosure label="looks" level={2}>
        {#each sections.looks as k (k.key)}<Knob knob={k} bind:value={step.knobs[k.key]} onchange={onchange} />{/each}
      </Disclosure>
    {/if}
  </Disclosure>

  <Disclosure label="Mask" note={maskCount ? `${maskCount} layer${maskCount === 1 ? '' : 's'}` : 'where the step shows'}>
    {#each step.layers as l, i}
      <div class="layer">
        <div class="row nowrap">
          {#if i > 0}<Select bind:value={l.combine} options={[{ value: 'and', label: 'and' }, { value: 'or', label: 'or' }, { value: 'but not', label: 'but not' }]} size="sm" width="92px" onchange={onchange} />{/if}
          <div class="grow"><Select bind:value={l.choice} options={presetOptions} size="sm" onchange={() => layerChanged(i)} /></div>
        </div>
        {#if l.choice === WORDS}
          <TextInput bind:value={l.words} placeholder="what to find, such as woman, sky, rust" size="sm" oncommit={onchange} />
        {:else if l.choice === CUSTOM}
          <TextInput bind:value={l.words} placeholder="a source line, such as colour:#3a6ea5 or spot:0.5,0.4,0.2" size="sm" oncommit={onchange} />
        {/if}
        {#if l.choice !== 'none' && l.choice !== WORDS}
          <Slider bind:value={l.coverage} min={0.05} max={1} step={0.05} label="coverage" onchange={onchange} />
        {/if}
      </div>
    {/each}
    <Checkbox bind:checked={step.invert} label="invert" onchange={onchange} />
    <Disclosure label="refine the edge" level={2}>
      {#each cat.refine as r (r.key)}
        <Slider bind:value={step.refine[r.key]} min={r.lo} max={r.hi} step={r.step} label={r.label} title={r.doc} onchange={onchange} />
      {/each}
    </Disclosure>
  </Disclosure>

  <Disclosure label="Finish" note={finishCount ? `${finishCount} changed` : 'light and colour'}>
    <div class="seg">
      <button class:on={step.scope === 'whole'} onclick={() => { step.scope = 'whole'; onchange() }}>whole picture</button>
      <button class:on={step.scope === 'mask'} onclick={() => { step.scope = 'mask'; onchange() }}>inside the mask</button>
    </div>
    {#each finishGroups as g, gi}
      <Disclosure label={g} level={2} open={gi === 0}>
        {#each cat.finish.filter((f) => f.group === g) as f (f.key)}
          <Slider value={step.finish[f.key] ?? f.default} min={f.lo} max={f.hi} step={f.step} label={f.label} title={f.doc}
                  onchange={(v) => { step.finish[f.key] = v; onchange() }} />
        {/each}
      </Disclosure>
    {/each}
    <div><Button size="sm" kind="ghost" onclick={() => { resetFinish(); onchange() }}>Reset finish</Button></div>
  </Disclosure>
</div>

<style>
  .settings { display: flex; flex-direction: column; gap: var(--sp-2); height: 100%; min-height: 0; overflow-y: auto; padding-right: 2px; }
  .seg { display: inline-flex; border: 1px solid var(--arc-line); border-radius: var(--r-md); overflow: hidden; }
  .seg.wrap { flex-wrap: wrap; }
  .seg button { height: var(--ctl-h-sm); padding: 0 10px; background: var(--arc-surface-2); color: var(--arc-ink-dim); border: none; font-size: var(--fs-xs); font-weight: 600; }
  .seg button + button { border-left: 1px solid var(--arc-line); }
  .seg button.on { background: color-mix(in srgb, var(--tab-alchemy) 28%, var(--arc-surface-2)); color: var(--arc-ink); }
  .layer { display: flex; flex-direction: column; gap: 6px; padding-bottom: 6px; border-bottom: 1px dashed var(--arc-line); }
  .layer .row { gap: 6px; }
</style>
