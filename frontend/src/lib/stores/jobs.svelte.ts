// Following one job to its end: poll its snapshot every 700 ms, hand each one
// to the caller, and stop when it finishes. Used by the dataset manager, the
// poetry pass and the film render.

import { get, post } from '../api/client'
import type { Job } from './app.svelte'

export type JobWatcher = { stop: () => void }

export function watchJob(id: string, onUpdate: (job: Job) => void, onDone?: (job: Job) => void, every = 700): JobWatcher {
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | null = null
  async function tick() {
    if (stopped) return
    try {
      const job = await get<Job>(`/api/jobs/${id}`)
      onUpdate(job)
      if (['done', 'failed', 'cancelled'].includes(job.status)) {
        stopped = true
        onDone?.(job)
        return
      }
    } catch {
      /* the next tick tries again */
    }
    timer = setTimeout(tick, every)
  }
  tick()
  return { stop: () => { stopped = true; if (timer) clearTimeout(timer) } }
}

export async function cancelJob(id: string) {
  await post(`/api/jobs/${id}/cancel`)
}

export function jobTone(job: Job | null): 'accent' | 'ok' | 'warn' | 'bad' {
  if (!job) return 'accent'
  if (job.status === 'failed') return 'bad'
  if (job.status === 'cancelled') return 'warn'
  if (job.status === 'done') return 'ok'
  return 'accent'
}

export function jobTitle(job: Job): string {
  if (job.status === 'failed') return 'Failed'
  if (job.status === 'cancelled') return 'Cancelled'
  if (job.status === 'done') return 'Finished'
  return job.label || 'Working'
}

export function jobMessage(job: Job): string {
  if (job.status === 'failed') return job.error || 'Something went wrong.'
  if (job.status === 'done') return job.message || 'Done.'
  return job.message || ''
}
