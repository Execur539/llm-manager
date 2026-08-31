/**
 * Tool registry primitives.
 *
 * Every tool declares a JSON Schema (compiled to GBNF so calls are structurally valid) and
 * a tier that drives the permission gate. Results are truncated before they enter the model's
 * context, with the full output written to disk and re-readable on demand — the single
 * biggest lever on how long an agent session survives.
 */

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import type { Backend, ToolDefinition, ToolResult, ToolTier } from '@shared/types'
import type { ContentPart } from '../../runtime/llama'
import { TOOL_OUTPUT_DIR } from '../../storage/paths'
import { log } from '../../log'

export interface ToolContext {
  cwd: string
  sessionId: string
  /** honoured by long-running tools */
  signal: AbortSignal
  timeoutMs: number
  /** Which llama.cpp build is in use; the embedding server needs it to start. */
  backend: Backend
  /** Whether the loaded model can actually be shown an image. */
  vision: boolean
  settings: {
    hfToken: string | null
  }
}

/**
 * What a tool hands back.
 *
 * Text for almost everything. A tool that produces something the model must *look* at rather
 * than read — a screenshot, an image on disk, the clipboard — returns media parts alongside a
 * text summary, and the loop feeds those to the model as image content.
 *
 * This is why `screenshot` used to be a dead end: it saved a PNG and returned a path, and there
 * was no way for anything in the tool set to turn that path into something the model could see.
 * Its own description told the model to "use read_image", a tool that did not exist.
 */
export type ToolOutput = string | { text: string; media: ContentPart[] }

export interface Tool extends ToolDefinition {
  run(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolOutput>
}

/** Split a tool's return value into the text that goes in the transcript and the parts that do not. */
export function splitOutput(out: ToolOutput): { text: string; media: ContentPart[] } {
  return typeof out === 'string' ? { text: out, media: [] } : { text: out.text, media: out.media }
}

/** Head/tail truncation budget for tool output entering context. */
const MAX_CHARS = 24000
const HEAD_CHARS = 16000
const TAIL_CHARS = 6000

export interface Truncated {
  content: string
  truncated: boolean
  fullOutputPath?: string
}

/**
 * Truncate a large result, keeping the head and tail (the two parts that carry meaning)
 * and writing the complete text to disk so the agent can page back into it.
 */
export function truncateForContext(raw: string, label: string): Truncated {
  if (raw.length <= MAX_CHARS) return { content: raw, truncated: false }

  // The overflow file is best-effort: a full disk or a permission error must not fail the tool
  // call. But we only advertise the path once the bytes are actually on disk — promising a file
  // that was never written sends the agent off to read_file a path that does not exist, and it
  // burns a turn discovering that for itself.
  const id = crypto.randomBytes(6).toString('hex')
  const file = path.join(TOOL_OUTPUT_DIR, `${label.replace(/[^a-z0-9]/gi, '_')}-${id}.txt`)
  let saved = false
  try {
    fs.mkdirSync(TOOL_OUTPUT_DIR, { recursive: true })
    fs.writeFileSync(file, raw, 'utf8')
    saved = true
  } catch (err) {
    log('warn', 'tools', `could not persist full output for ${label}`, err)
  }

  const head = raw.slice(0, HEAD_CHARS)
  const tail = raw.slice(-TAIL_CHARS)
  const omitted = raw.length - HEAD_CHARS - TAIL_CHARS

  const note = saved
    ? `full output saved to ${file}. Use read_file with offset/limit on that path to page through it.`
    : 'the full output could not be saved to disk, so the omitted middle is unavailable.'

  return {
    truncated: true,
    fullOutputPath: saved ? file : undefined,
    content:
      `${head}\n\n` +
      `... [${omitted.toLocaleString()} characters omitted; ${note}] ...\n\n` +
      tail
  }
}

export class ToolRegistry {
  private tools = new Map<string, Tool>()

  register(tool: Tool): void {
    this.tools.set(tool.name, tool)
  }

  registerAll(tools: Tool[]): void {
    for (const t of tools) this.register(t)
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name)
  }

  list(): Tool[] {
    return [...this.tools.values()]
  }

  /** Definitions in the shape the model sees (no implementation). */
  definitions(): ToolDefinition[] {
    return this.list().map(({ name, description, tier, parameters, sensitive }) => ({
      name,
      description,
      tier,
      parameters,
      sensitive
    }))
  }

  byTier(tier: ToolTier): Tool[] {
    return this.list().filter((t) => t.tier === tier)
  }
}

/** Small helper so tool definitions stay readable. */
export function schema(
  properties: Record<string, unknown>,
  required: string[] = []
): Record<string, unknown> {
  return { type: 'object', properties, required }
}

export const str = (description: string) => ({ type: 'string', description })
export const int = (description: string) => ({ type: 'integer', description })
export const bool = (description: string) => ({ type: 'boolean', description })

export function makeResult(
  callId: string,
  raw: string,
  label: string,
  startedAt: number,
  ok = true,
  error?: string
): ToolResult {
  const t = truncateForContext(raw, label)
  return {
    callId,
    ok,
    content: t.content,
    fullOutputPath: t.fullOutputPath,
    truncated: t.truncated,
    error,
    durationMs: Date.now() - startedAt
  }
}
