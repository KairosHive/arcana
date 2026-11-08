<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { app } from '../../lib/stores/app.svelte'
  import { agentToken } from '../../lib/api/client'
  import Button from '../../lib/ui/Button.svelte'
  import Spinner from '../../lib/ui/Spinner.svelte'
  import '@xterm/xterm/css/xterm.css'

  /**
   * An agent's own command line, in the Arcana workspace, with Arcana's tools.
   * Output streams over Server-Sent Events; keystrokes go back as small POSTs.
   * The server keeps the agents and their scrollback, so a reload comes back
   * to them; agents stop once no page has looked at them for three minutes.
   */
  type Offered = { name: string; command: string }
  type Info = { id: string; name: string; running: boolean; exit_code: number | null; started: number; cwd: string }
  type Term = { term: any; fit: any; es: EventSource | null; replaying: boolean; pending: string; timer: ReturnType<typeof setTimeout> | null; size: string }

  let offered = $state<Offered[]>([])
  let instances = $state<Info[]>([])
  let workspace = $state('')
  let current = $state<string | null>(null)
  let loading = $state(true)
  let host: HTMLDivElement | undefined = $state()
  const terms = new Map<string, Term>()
  let token = ''
  let xtermMod: typeof import('@xterm/xterm') | null = null
  let fitMod: typeof import('@xterm/addon-fit') | null = null
  let checkIn: ReturnType<typeof setInterval> | null = null
  const visible = $derived(app.mode === 'agent')

  async function api(path: string, body?: unknown) {
    const r = await fetch(path, { method: body === undefined ? 'GET' : 'POST', headers: { 'X-Arcana-Term': token, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
    return r.json()
  }

  async function refresh() {
    try {
      const r = await api('/agent/roster')
      offered = r.offered; workspace = r.workspace
      const live = new Set((r.instances as Info[]).map((i) => i.id))
      // Keep terminals the server dropped, marked exited, so their last screen stays readable.
      const kept = instances.filter((i) => terms.has(i.id) && !live.has(i.id)).map((i) => ({ ...i, running: false }))
      instances = [...r.instances, ...kept]
      if (!current || !instances.some((i) => i.id === current)) {
        const first = instances.find((i) => i.running)
        if (first) show(first.id)
      }
    } catch { /* the server may be busy; next check-in */ } finally { loading = false }
  }

  async function libs() {
    if (!xtermMod) xtermMod = await import('@xterm/xterm')
    if (!fitMod) fitMod = await import('@xterm/addon-fit')
  }

  async function terminalFor(id: string): Promise<Term> {
    let t = terms.get(id)
    if (t) return t
    await libs()
    const term = new xtermMod!.Terminal({
      fontFamily: '"Cascadia Mono", "Cascadia Code", Consolas, "Courier New", monospace', fontSize: 13, scrollback: 10000, cursorBlink: true, allowProposedApi: true,
      theme: { background: '#141311', foreground: '#ede8df', cursor: '#c8743f', selectionBackground: '#c8743f55', black: '#1c1a17', red: '#b5423a', green: '#7a9a5e', yellow: '#d6a74d', blue: '#7c8f8a', magenta: '#a37a8a', cyan: '#8c9a5b', white: '#ede8df' },
    })
    const fit = new fitMod!.FitAddon()
    term.loadAddon(fit)
    t = { term, fit, es: null, replaying: false, pending: '', timer: null, size: '' }
    terms.set(id, t)
    term.onData((data: string) => {
      if (t!.replaying) return
      t!.pending += data
      if (!t!.timer) t!.timer = setTimeout(() => { const d = t!.pending; t!.pending = ''; t!.timer = null; api(`/agent/${id}/input`, { data: d }) }, 8)
    })
    const es = new EventSource(`/agent/${id}/stream?token=${encodeURIComponent(token)}`)
    t.es = es
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.replay) { t!.replaying = true; term.write(msg.d, () => { t!.replaying = false }) }
      else if (msg.d !== undefined) term.write(msg.d)
      if (msg.exit !== undefined) {
        term.write(`\r\n\x1b[2m[the agent exited with code ${msg.exit}]\x1b[0m\r\n`)
        es.close(); t!.es = null
        instances = instances.map((i) => (i.id === id ? { ...i, running: false, exit_code: msg.exit } : i))
      }
    }
    return t
  }

  async function show(id: string) {
    current = id
    const t = await terminalFor(id)
    if (!host) return
    host.innerHTML = ''
    t.term.open(host)
    requestAnimationFrame(() => fit(id))
    t.term.focus()
  }

  function fit(id = current) {
    if (!id) return
    const t = terms.get(id)
    if (!t || !host || host.clientWidth === 0) return
    t.fit.fit()
    const dims = t.fit.proposeDimensions()
    if (!dims) return
    const size = `${dims.cols}x${dims.rows}`
    if (size !== t.size) { t.size = size; api(`/agent/${id}/resize`, { cols: dims.cols, rows: dims.rows }) }
  }

  async function start(name: string) {
    await libs()
    const dims = { cols: 120, rows: 32 }
    const info = await api('/agent/start', { name, ...dims })
    if (info.error) { alert(info.error); return }
    await refresh()
    show(info.id)
  }

  async function close(i: Info) {
    if (i.running) {
      if (!confirm(`Stop ${i.name}? Its session ends, and the workspace stays.`)) return
      await api(`/agent/${i.id}/stop`, {})
    } else {
      await api(`/agent/${i.id}/forget`, {})
    }
    const t = terms.get(i.id)
    t?.es?.close(); t?.term.dispose(); terms.delete(i.id)
    if (current === i.id) { current = null; if (host) host.innerHTML = '' }
    instances = instances.filter((x) => x.id !== i.id)
    await refresh()
  }

  onMount(() => {
    const ro = new ResizeObserver(() => fit())
    if (host) ro.observe(host)
    ;(async () => {
      token = await agentToken()
      await refresh()
      checkIn = setInterval(() => { if (visible) refresh(); else api('/agent/roster').catch(() => {}) }, 20000)
    })()
    return () => ro.disconnect()
  })
  $effect(() => { if (visible) requestAnimationFrame(() => fit()) })
  onDestroy(() => { if (checkIn) clearInterval(checkIn); for (const t of terms.values()) { t.es?.close(); t.term.dispose() } })
</script>

<div class="agent">
  <div class="top">
    <div class="row"><span class="caps title">Agent</span><span class="where dim">Works in {workspace}</span></div>
    <div class="bar">
      {#each instances as i (i.id)}
        <span class="tab" class:on={current === i.id}>
          <button class="name" onclick={() => show(i.id)}>{i.name}{i.running ? '' : ' (exited)'}</button>
          <button class="x" title={i.running ? 'Stop this agent' : 'Close'} onclick={() => close(i)}>×</button>
        </span>
      {/each}
      <button class="add" title="Start another agent" onclick={() => (current = null)}>+</button>
    </div>
  </div>

  <div class="launcher" hidden={current !== null}>
    {#if loading}<Spinner label="Looking for agents…" />{:else}
      <div class="box">
        <p class="t">Start an agent</p>
        {#if offered.length}
          <p class="hint">It runs in the Arcana workspace with Arcana's tools and skills, and acts as an agent: it changes only boards it made, and sees thumbnails only if you allow it.</p>
          <div class="grid">{#each offered as o}<button class="launch" title={o.command} onclick={() => start(o.name)}>{o.name}</button>{/each}</div>
        {:else}
          <p class="hint">No agent command line was found on this machine. Install Claude Code, or list one in agents.json in Arcana's data folder.</p>
        {/if}
      </div>
    {/if}
  </div>
  <div class="host" bind:this={host} hidden={current === null}></div>
</div>

<style>
  .agent { display: flex; flex-direction: column; gap: 8px; flex: 1; min-height: 0; padding: 10px 12px 12px; }
  .top { display: flex; flex-direction: column; gap: 8px; flex: 0 0 auto; }
  .title { font-weight: 700; letter-spacing: 1px; color: var(--arc-ink); }
  .where { font-size: var(--fs-xs); word-break: break-all; }
  .bar { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .tab { display: inline-flex; align-items: stretch; border: 1px solid var(--arc-line); border-radius: var(--r-md); overflow: hidden; background: var(--arc-surface); }
  .tab.on { border-color: var(--arc-accent); }
  .name { background: transparent; border: none; color: var(--arc-ink); padding: 4px 10px; font-size: var(--fs-sm); }
  .tab.on .name { color: var(--arc-accent); }
  .x { background: transparent; border: none; border-left: 1px solid var(--arc-line); color: var(--arc-ink-dim); padding: 0 8px; }
  .x:hover { color: var(--arc-bad); }
  .add { width: 28px; height: 28px; border-radius: var(--r-md); border: 1px dashed var(--arc-line); background: transparent; color: var(--arc-ink-dim); }
  .launcher { flex: 1; display: flex; align-items: flex-start; }
  .launcher[hidden], .host[hidden] { display: none; }
  .box { max-width: 560px; padding: 16px 18px; border: 1px solid var(--arc-line); border-radius: var(--r-lg); background: var(--arc-surface); display: flex; flex-direction: column; gap: 10px; }
  .t { font-weight: 700; }
  .grid { display: flex; gap: 8px; flex-wrap: wrap; }
  .launch { padding: 8px 14px; border-radius: var(--r-md); border: none; background: var(--arc-accent); color: var(--arc-accent-ink); font-weight: 600; }
  .host { flex: 1; min-height: 0; border-radius: 8px; padding: 6px; background: var(--arc-bg); border: 1px solid var(--arc-line); overflow: hidden; }
  .host :global(.xterm) { height: 100%; }
</style>
