/**
 * Data tools: SQLite queries and structured-format parsing.
 *
 * These exist so the agent can work with data without shelling out to a script for every
 * small transformation, which is both slow and noisy in the transcript.
 */

import { DatabaseSync } from 'node:sqlite'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { Tool } from './base'
import { schema, str, int } from './base'

function resolve(cwd: string, p: string): string {
  return path.isAbsolute(p) ? path.normalize(p) : path.resolve(cwd, p)
}

/**
 * A rough "this writes" test, used only to give a better error than SQLite's own.
 *
 * Not a security boundary and not treated as one — the read tool opens the file read-only, so
 * anything this misses (a leading comment, a CTE wrapping a DELETE) still fails at the engine.
 */
const MODIFIES_SQL = /^\s*(insert|update|delete|drop|create|alter|replace|truncate|vacuum|attach|pragma\s+\w+\s*=)/i

/**
 * Reading and writing are separate tools because tier is what the permission gate reads.
 *
 * There used to be one `sqlite_query` at read tier with a `write: true` flag. Read tier means
 * "runs freely, no prompt", so setting that flag dropped tables in the user's database with no
 * approval asked — and, being read tier, the tool stayed available in plan mode and in Ultra's
 * read-only planning samples, both of which promise nothing will be modified. A tool that can be
 * either is a tool whose tier is a lie about half its calls.
 */
const sqliteQuery: Tool = {
  name: 'sqlite_query',
  description:
    'Run a read-only SQL query against a SQLite database file and return the rows. Use ' +
    'sqlite_execute for statements that modify data.',
  tier: 'read',
  parameters: schema(
    {
      database: str('Path to the .db / .sqlite file'),
      sql: str('SELECT (or other read-only) SQL'),
      limit: int('Maximum rows to return (default 200)')
    },
    ['database', 'sql']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.database))
    const sql = String(args.sql)

    // The database is opened read-only, so this is a clearer error rather than the gate itself.
    if (MODIFIES_SQL.test(sql)) {
      throw new Error('This statement modifies the database. Use sqlite_execute instead.')
    }

    const db = new DatabaseSync(file, { readOnly: true })
    try {
      const rows = db.prepare(sql).all() as Record<string, unknown>[]
      const limit = Number(args.limit ?? 200)
      const shown = rows.slice(0, limit)
      if (!shown.length) return '(no rows)'
      const header = Object.keys(shown[0]).join(' | ')
      const body = shown.map((r) => Object.values(r).map((v) => String(v ?? '')).join(' | ')).join('\n')
      const more = rows.length > limit ? `\n\n[${rows.length - limit} more rows]` : ''
      return `${header}\n${'-'.repeat(header.length)}\n${body}${more}`
    } finally {
      db.close()
    }
  }
}

const sqliteExecute: Tool = {
  name: 'sqlite_execute',
  description:
    'Run a statement that modifies a SQLite database — INSERT, UPDATE, DELETE, CREATE, ALTER, ' +
    'DROP. Changes are not reversible, so say what you intend to change before calling it.',
  tier: 'write',
  parameters: schema(
    { database: str('Path to the .db / .sqlite file'), sql: str('The statement to execute') },
    ['database', 'sql']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.database))
    const db = new DatabaseSync(file)
    try {
      db.prepare(String(args.sql)).run()
      return 'Statement executed.'
    } finally {
      db.close()
    }
  }
}

const sqliteSchema: Tool = {
  name: 'sqlite_schema',
  description: 'List tables, columns and indexes in a SQLite database.',
  tier: 'read',
  parameters: schema({ database: str('Path to the database file') }, ['database']),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.database))
    const db = new DatabaseSync(file, { readOnly: true })
    try {
      const objects = db
        .prepare("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name")
        .all() as { type: string; name: string; sql: string }[]
      return objects.map((o) => `-- ${o.type}: ${o.name}\n${o.sql};`).join('\n\n')
    } finally {
      db.close()
    }
  }
}

/** Minimal CSV parser handling quoted fields and embedded separators. */
function parseCsv(text: string, delimiter: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let inQuotes = false

  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else inQuotes = false
      } else field += ch
    } else if (ch === '"') inQuotes = true
    else if (ch === delimiter) {
      row.push(field)
      field = ''
    } else if (ch === '\n') {
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else if (ch !== '\r') field += ch
  }
  if (field || row.length) {
    row.push(field)
    rows.push(row)
  }
  return rows
}

const parseData: Tool = {
  name: 'parse_data',
  description:
    'Parse a JSON, CSV or TSV file and return a readable summary: shape, columns, and a sample ' +
    'of rows. Much cheaper than reading a large data file line by line.',
  tier: 'read',
  parameters: schema(
    {
      path: str('File to parse'),
      format: str("'json', 'csv' or 'tsv'; inferred from the extension when omitted"),
      sample: int('How many rows to show (default 10)')
    },
    ['path']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.path))
    // Reading is unbounded otherwise, and this tool exists precisely because the file might be
    // large. `read_file` refuses at the same point, so the two agree on what "too big" means.
    const { size } = await fsp.stat(file)
    if (size > 200 * 1024 * 1024) {
      throw new Error(`${file} is ${(size / 1e6).toFixed(0)} MB — too large to parse in memory.`)
    }
    const text = await fsp.readFile(file, 'utf8')
    const ext = path.extname(file).toLowerCase()
    const format = String(args.format ?? (ext === '.json' ? 'json' : ext === '.tsv' ? 'tsv' : 'csv'))
    const sample = Number(args.sample ?? 10)

    if (format === 'json') {
      const data = JSON.parse(text)
      if (Array.isArray(data)) {
        const keys = data.length && typeof data[0] === 'object' && data[0] ? Object.keys(data[0]) : []
        return [
          `JSON array, ${data.length} items`,
          keys.length ? `Keys: ${keys.join(', ')}` : '',
          `First ${Math.min(sample, data.length)}:`,
          JSON.stringify(data.slice(0, sample), null, 2)
        ]
          .filter(Boolean)
          .join('\n')
      }
      const keys = typeof data === 'object' && data ? Object.keys(data) : []
      return `JSON object with keys: ${keys.join(', ')}\n\n${JSON.stringify(data, null, 2).slice(0, 4000)}`
    }

    const rows = parseCsv(text, format === 'tsv' ? '\t' : ',')
    if (!rows.length) return '(empty file)'
    const [header, ...body] = rows
    return [
      `${format.toUpperCase()}: ${body.length} data rows, ${header.length} columns`,
      `Columns: ${header.join(', ')}`,
      '',
      header.join(' | '),
      '-'.repeat(Math.min(120, header.join(' | ').length)),
      ...body.slice(0, sample).map((r) => r.join(' | '))
    ].join('\n')
  }
}

export const dataTools: Tool[] = [sqliteQuery, sqliteExecute, sqliteSchema, parseData]
