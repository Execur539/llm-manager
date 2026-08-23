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
import { schema, str, int, bool } from './base'

function resolve(cwd: string, p: string): string {
  return path.isAbsolute(p) ? path.normalize(p) : path.resolve(cwd, p)
}

const sqliteQuery: Tool = {
  name: 'sqlite_query',
  description:
    'Run a SQL query against a SQLite database file and return the rows. Read-only by default; ' +
    'set write=true for statements that modify data.',
  tier: 'read',
  parameters: schema(
    {
      database: str('Path to the .db / .sqlite file'),
      sql: str('SQL to execute'),
      write: bool('Required for INSERT/UPDATE/DELETE/CREATE and similar'),
      limit: int('Maximum rows to return (default 200)')
    },
    ['database', 'sql']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.database))
    const sql = String(args.sql)
    const isWrite = /^\s*(insert|update|delete|drop|create|alter|replace|truncate|pragma\s+\w+\s*=)/i.test(sql)

    if (isWrite && !args.write) {
      throw new Error('This statement modifies the database. Set write=true to proceed.')
    }

    const db = new DatabaseSync(file, { readOnly: !args.write })
    try {
      if (isWrite) {
        db.prepare(sql).run()
        return 'Statement executed.'
      }
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

export const dataTools: Tool[] = [sqliteQuery, sqliteSchema, parseData]
