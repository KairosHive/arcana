<script lang="ts">
  import { rpc, reason } from '../../lib/api/client'
  import { boardPageUrl, downloadUrl } from '../../lib/api/media'
  import Modal from '../../lib/ui/Modal.svelte'
  import Button from '../../lib/ui/Button.svelte'

  /**
   * A board's own designed page, in a frame the server sandboxes (its
   * scripts run with no origin of their own). Save as PDF runs `board pdf`
   * on the server; Print view opens the page itself with the print dialog.
   */
  let { open = $bindable(false), slug }: { open?: boolean; slug: string } = $props()
  const v = $derived(open ? Date.now() : 0)
  let status = $state('')
  let saving = $state(false)

  async function pdf() {
    saving = true; status = ''
    try {
      const r = await rpc('board pdf', { board: slug })
      status = `${r.result.pages} page${r.result.pages === 1 ? '' : 's'}, saved to ${r.result.path}`
      window.open(downloadUrl(r.result.path), '_blank')
    } catch (e) {
      status = `${reason(e)} The Print view link prints from your browser instead.`
    } finally { saving = false }
  }
</script>

<Modal bind:open title="PAGE · {slug}" full onclose={() => (status = '')}>
  {#snippet actions()}
    <span class="st dim">{status}</span>
    <Button size="sm" kind="primary" onclick={pdf} busy={saving}>Save as PDF</Button>
    <a class="print" href={boardPageUrl(slug, v, true)} target="_blank" rel="noopener">Print view</a>
  {/snippet}
  {#if open}<iframe title="Board page" src={boardPageUrl(slug, v)}></iframe>{/if}
</Modal>

<style>
  iframe { flex: 1; width: 100%; border: none; background: #fff; min-height: 0; }
  .st { font-size: var(--fs-xs); max-width: 40ch; }
  .print { font-size: var(--fs-sm); color: var(--arc-ink-dim); }
</style>
