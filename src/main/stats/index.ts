/**
 * Observability: live inference stats, hardware meters, request log, historical totals.
 *
 * Everything is local. There is no telemetry of any kind, by decision — these numbers exist so
 * the user can see what their machine is doing and verify the auto-fit engine's predictions,
 * not so anything is reported anywhere.
 */

import { all, get, run } from '../storage/db'
import { llama } from '../runtime/llama'
import { detectHardware } from '../hardware/gpu'
import type { HardwareSnapshot } from '@shared/types'

export interface LiveStats {
  loaded: string | null
  contextLength: number
  contextUsed: number
  kvFillPercent: number
  tokensPerSecond: number
  ttftMs: number | null
  lastCompletionTokens: number
  hardware: HardwareSnapshot | null
}

export interface RequestLogEntry {
  id: number
  ts: number
  endpoint: string
  modelId: string | null
  tokensIn: number
  tokensOut: number
  ms: number
  client: string
  ip: string
  status: number
}

export interface HistoricalStats {
  totalTokensOut: number
  totalRequests: number
  activeHours: number
  byModel: { modelId: string; tokensOut: number; requests: number; avgTokensPerSecond: number }[]
  byDay: { date: string; tokensOut: number }[]
}

/** Tokens currently occupying the context window, tracked by the caller. */
let contextUsed = 0

export function setContextUsed(tokens: number): void {
  contextUsed = tokens
}

export async function liveStats(hardware?: HardwareSnapshot | null): Promise<LiveStats> {
  const loaded = llama.loaded
  const t = llama.timings
  const ctx = loaded?.plan.contextLength ?? 0

  return {
    loaded: loaded?.model.filename ?? null,
    contextLength: ctx,
    contextUsed,
    kvFillPercent: ctx > 0 ? Math.min(100, (contextUsed / ctx) * 100) : 0,
    tokensPerSecond: t?.tokensPerSecond ?? 0,
    ttftMs: t?.ttftMs ?? null,
    lastCompletionTokens: t?.completionTokens ?? 0,
    hardware: hardware ?? (await detectHardware().catch(() => null))
  }
}

export function requestLog(limit = 100): RequestLogEntry[] {
  return all<{
    id: number
    ts: number
    endpoint: string
    model_id: string | null
    tokens_in: number
    tokens_out: number
    ms: number
    client: string
    ip: string
    status: number
  }>('SELECT * FROM requests ORDER BY ts DESC LIMIT ?', limit).map((r) => ({
    id: r.id,
    ts: r.ts,
    endpoint: r.endpoint,
    modelId: r.model_id,
    tokensIn: r.tokens_in ?? 0,
    tokensOut: r.tokens_out ?? 0,
    ms: r.ms ?? 0,
    client: r.client ?? 'local',
    ip: r.ip ?? '',
    status: r.status ?? 200
  }))
}

/** Record a completed generation against today's totals. */
export function recordGeneration(modelId: string, tokensOut: number, seconds: number): void {
  const date = new Date().toISOString().slice(0, 10)
  run(
    `INSERT INTO stats_daily (date, model_id, tokens_in, tokens_out, active_seconds)
     VALUES (?, ?, 0, ?, ?)
     ON CONFLICT(date, model_id) DO UPDATE SET
       tokens_out = tokens_out + excluded.tokens_out,
       active_seconds = active_seconds + excluded.active_seconds`,
    date,
    modelId,
    Math.round(tokensOut),
    Math.round(seconds)
  )
}

export function historicalStats(): HistoricalStats {
  const totals = get<{ tokens: number; seconds: number }>(
    'SELECT COALESCE(SUM(tokens_out),0) AS tokens, COALESCE(SUM(active_seconds),0) AS seconds FROM stats_daily'
  )
  const requests = get<{ n: number }>('SELECT COUNT(*) AS n FROM requests')

  const byModel = all<{ model_id: string; tokens: number; seconds: number }>(
    'SELECT model_id, SUM(tokens_out) AS tokens, SUM(active_seconds) AS seconds FROM stats_daily GROUP BY model_id ORDER BY tokens DESC'
  ).map((r) => ({
    modelId: r.model_id,
    tokensOut: r.tokens,
    requests:
      get<{ n: number }>('SELECT COUNT(*) AS n FROM requests WHERE model_id = ?', r.model_id)?.n ?? 0,
    avgTokensPerSecond: r.seconds > 0 ? r.tokens / r.seconds : 0
  }))

  const byDay = all<{ date: string; tokens: number }>(
    'SELECT date, SUM(tokens_out) AS tokens FROM stats_daily GROUP BY date ORDER BY date DESC LIMIT 30'
  ).map((r) => ({ date: r.date, tokensOut: r.tokens }))

  return {
    totalTokensOut: totals?.tokens ?? 0,
    totalRequests: requests?.n ?? 0,
    activeHours: (totals?.seconds ?? 0) / 3600,
    byModel,
    byDay
  }
}

export function clearStats(): void {
  run('DELETE FROM stats_daily')
  run('DELETE FROM requests')
}
