/**
 * Tools that buy turns back, and one that spends them deliberately.
 *
 * A turn is the expensive unit here: every tool call is a full round trip through the model, it
 * counts against `maxToolCallsPerTurn`, and each result is appended to a context window that is
 * already the binding constraint. Reading six files one at a time is six turns and six approval
 * checks to learn what one call could have said.
 */

import fsp from 'node:fs/promises'
import path from 'node:path'
import type { Tool, ToolContext } from './base'
import { schema, str, int } from './base'

function resolve(cwd: string, p: string): string {
  return path.isAbsolute(p) ? path.normalize(p) : path.resolve(cwd, p)
}

// ---------------------------------------------------------------- editing

interface Edit {
  old_string?: unknown
  new_string?: unknown
  replace_all?: unknown
}

const multiEdit: Tool = {
  name: 'multi_edit',
  description:
    'Make several exact-string edits to one file in a single call. Every edit is checked first ' +
    'and the file is written once — either all of them apply or none do, so a half-finished ' +
    'refactor cannot be left behind. Edits apply in order, so a later one sees the result of an ' +
    'earlier one. Prefer this over repeated edit_file: it is one approval and one turn.',
  tier: 'write',
  parameters: schema(
    {
      path: str('File to edit'),
      edits: {
        type: 'array',
        description: 'Edits applied in order; each needs old_string and new_string',
        items: {
          type: 'object',
          properties: {
            old_string: { type: 'string', description: 'Exact text to find, including indentation' },
            new_string: { type: 'string', description: 'Replacement text' },
            replace_all: { type: 'boolean', description: 'Replace every occurrence instead of requiring exactly one' }
          },
          required: ['old_string', 'new_string']
        }
      }
    },
    ['path', 'edits']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.path))
    const edits = (args.edits as Edit[] | undefined) ?? []
    if (!edits.length) throw new Error('No edits given.')

    let content = await fsp.readFile(file, 'utf8')
    const applied: string[] = []

    /*
     * Every edit is applied to an in-memory copy and the whole thing is written once.
     *
     * Failing on the third of five edits after two have already hit the disk leaves the file in
     * a state neither the model nor the user asked for, and the model's next read shows
     * something that matches neither its plan nor its memory of the file.
     */
    edits.forEach((edit, i) => {
      const oldStr = String(edit.old_string ?? '')
      const newStr = String(edit.new_string ?? '')
      if (!oldStr) throw new Error(`Edit ${i + 1}: old_string is empty.`)

      const count = content.split(oldStr).length - 1
      if (count === 0) {
        throw new Error(
          `Edit ${i + 1}: old_string not found. Nothing was written. ` +
            (i > 0 ? 'Note that earlier edits in this call may have changed the text you were matching against.' : '')
        )
      }
      if (count > 1 && !edit.replace_all) {
        throw new Error(
          `Edit ${i + 1}: old_string appears ${count} times. Nothing was written. Add surrounding context, or set replace_all.`
        )
      }
      content = edit.replace_all ? content.split(oldStr).join(newStr) : content.replace(oldStr, newStr)
      applied.push(`${i + 1}. ${edit.replace_all ? `${count} occurrence(s)` : '1 occurrence'}`)
    })

    await fsp.writeFile(file, content, 'utf8')
    return `Applied ${edits.length} edit(s) to ${file}:\n${applied.join('\n')}`
  }
}

// ---------------------------------------------------------------- reading

/** Total across all files in one call, so a wide read cannot swallow the context window. */
const MAX_TOTAL_CHARS = 120_000

const readManyFiles: Tool = {
  name: 'read_many_files',
  description:
    'Read several files in one call. Use this when orienting in an unfamiliar codebase rather ' +
    'than one read_file per file — it is one turn instead of many, and the whole set arrives ' +
    'together so you can compare them. Each file is capped, and the total is capped.',
  tier: 'read',
  parameters: schema(
    {
      paths: { type: 'array', items: { type: 'string' }, description: 'Files to read' },
      max_lines_each: int('Line cap per file (default 400)')
    },
    ['paths']
  ),
  async run(args, ctx) {
    const paths = (args.paths as string[] | undefined) ?? []
    if (!paths.length) throw new Error('No paths given.')
    if (paths.length > 40) throw new Error(`${paths.length} files is too many for one call; ask for at most 40.`)

    const perFile = Math.max(1, Number(args.max_lines_each ?? 400))
    const sections: string[] = []
    let total = 0
    let stoppedAt = -1

    for (const [i, p] of paths.entries()) {
      if (total >= MAX_TOTAL_CHARS) {
        stoppedAt = i
        break
      }
      const file = resolve(ctx.cwd, p)
      try {
        const stat = await fsp.stat(file)
        if (stat.isDirectory()) {
          sections.push(`===== ${file} =====\n(directory — use list_dir)`)
          continue
        }
        const lines = (await fsp.readFile(file, 'utf8')).split(/\r?\n/)
        const shown = lines.slice(0, perFile)
        const body = shown.map((l, n) => `${String(n + 1).padStart(5)}\t${l}`).join('\n')
        const more = lines.length > perFile ? `\n[${lines.length - perFile} more lines; read_file this path to continue]` : ''
        const section = `===== ${file} =====\n${body}${more}`
        total += section.length
        sections.push(section)
      } catch (err) {
        // One unreadable file must not lose the other thirty-nine.
        sections.push(`===== ${file} =====\n(could not read: ${err instanceof Error ? err.message : String(err)})`)
      }
    }

    if (stoppedAt >= 0) {
      sections.push(`[stopped after ${stoppedAt} of ${paths.length} files — the combined output hit its limit]`)
    }
    return sections.join('\n\n')
  }
}

// ---------------------------------------------------------------- waiting

/**
 * How long a single wait may last, and how the ceiling grows.
 *
 * A model has no way to know how long a build, a download or a server start actually takes, and
 * left to itself it guesses badly in the expensive direction — asking to sleep a minute when the
 * thing it is waiting for finished in two seconds. So the first wait in a run is capped short
 * and the ceiling roughly doubles each consecutive time, which is the behaviour you would want
 * from someone checking on a pot: glance, glance, then go and do something else.
 *
 * The escalation resets as soon as the agent does anything other than wait, because that means
 * it has learned something and is no longer in the same holding pattern.
 */
const WAIT_CEILINGS = [5, 10, 20, 40, 60]
const MAX_WAIT_SECONDS = 60

/** Consecutive waits per session, so the ceiling can climb. Reset by any other tool. */
const consecutiveWaits = new Map<string, number>()

/** Called by the loop when a tool other than `wait` runs. */
export function resetWaitEscalation(sessionId: string): void {
  consecutiveWaits.delete(sessionId)
}

const wait: Tool = {
  name: 'wait',
  description:
    'Pause before checking something again — a build finishing, a server coming up, a file ' +
    'appearing. Bias towards short waits: you cannot know how long something takes, and a wait ' +
    'that is too short costs one cheap check while one that is too long wastes real time. Start ' +
    'at a few seconds and ask for longer only after a short wait proved insufficient. The first ' +
    'wait in a sequence is capped at 5 seconds and the cap roughly doubles each consecutive ' +
    'wait, up to 60. Doing anything else resets it.',
  tier: 'read',
  parameters: schema(
    {
      seconds: int('How long to wait, 1 to 60. Start small.'),
      reason: str('What you are waiting for — shown to the user')
    },
    ['seconds']
  ),
  async run(args, ctx) {
    const asked = Math.max(1, Math.min(MAX_WAIT_SECONDS, Math.round(Number(args.seconds ?? 5) || 5)))

    const previous = consecutiveWaits.get(ctx.sessionId) ?? 0
    const ceiling = WAIT_CEILINGS[Math.min(previous, WAIT_CEILINGS.length - 1)]
    const actual = Math.min(asked, ceiling)
    consecutiveWaits.set(ctx.sessionId, previous + 1)

    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, actual * 1000)
      // Stop must not have to sit through a minute of nothing.
      const onAbort = (): void => {
        clearTimeout(timer)
        resolve()
      }
      if (ctx.signal.aborted) onAbort()
      else ctx.signal.addEventListener('abort', onAbort, { once: true })
    })

    if (ctx.signal.aborted) return `Wait interrupted after less than ${actual}s.`

    const nextCeiling = WAIT_CEILINGS[Math.min(previous + 1, WAIT_CEILINGS.length - 1)]
    const capped =
      actual < asked
        ? ` (you asked for ${asked}s; the cap at this point in the sequence is ${ceiling}s, so check now and wait longer next time if it is still not ready)`
        : ''
    return (
      `Waited ${actual}s${args.reason ? ` for: ${String(args.reason)}` : ''}${capped}. ` +
      `Check whether it is ready. If not, the next wait may be up to ${nextCeiling}s.`
    )
  }
}

export const workflowTools: Tool[] = [multiEdit, readManyFiles, wait]
