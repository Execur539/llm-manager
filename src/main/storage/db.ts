/**
 * SQLite persistence.
 *
 * Uses Node's built-in `node:sqlite` (Electron 38 ships Node 22), so there is no native module
 * to rebuild and nothing to go wrong at install time on a user's machine.
 *
 * Schema covers chats, agent sessions, the model index, downloads, RAG documents, the API
 * request log, and daily stats. Migrations are forward-only and idempotent.
 */

import { DatabaseSync, type StatementSync } from 'node:sqlite'
import fs from 'node:fs'
import path from 'node:path'
import { APPDATA_DIR, DB_FILE } from './paths'

let db: DatabaseSync | null = null

const MIGRATIONS: { id: number; sql: string }[] = [
  {
    id: 1,
    sql: `
    CREATE TABLE IF NOT EXISTS chats (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      model_id TEXT,
      preset_id TEXT,
      system_prompt TEXT,
      kind TEXT NOT NULL DEFAULT 'chat',   -- 'chat' | 'agent'
      cwd TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
      parent_id TEXT,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      tool_calls TEXT,
      tool_result TEXT,
      tokens INTEGER,
      timings TEXT,
      created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);

    CREATE TABLE IF NOT EXISTS attachments (
      id TEXT PRIMARY KEY,
      message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
      kind TEXT NOT NULL,
      path TEXT NOT NULL,
      meta TEXT,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS presets (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      temperature REAL,
      top_p REAL,
      top_k INTEGER,
      min_p REAL,
      repeat_penalty REAL,
      system_prompt TEXT,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS model_meta (
      model_id TEXT PRIMARY KEY,
      repo TEXT,
      favourite INTEGER NOT NULL DEFAULT 0,
      tags TEXT,
      last_used_at INTEGER
    );

    CREATE TABLE IF NOT EXISTS model_configs (
      model_id TEXT PRIMARY KEY,
      plan TEXT NOT NULL,
      predicted_vram TEXT,
      actual_vram TEXT,
      updated_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS downloads (
      id TEXT PRIMARY KEY,
      repo TEXT,
      filename TEXT NOT NULL,
      url TEXT NOT NULL,
      dest TEXT NOT NULL,
      bytes_total INTEGER NOT NULL DEFAULT 0,
      bytes_done INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL,
      error TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS collections (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
      chat_id TEXT,
      filename TEXT NOT NULL,
      path TEXT NOT NULL,
      mime TEXT,
      bytes INTEGER,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      ord INTEGER NOT NULL,
      text TEXT NOT NULL,
      tokens INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, ord);

    CREATE TABLE IF NOT EXISTS embeddings (
      chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
      dim INTEGER NOT NULL,
      vector BLOB NOT NULL,
      embed_model TEXT
    );

    CREATE TABLE IF NOT EXISTS requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER NOT NULL,
      endpoint TEXT NOT NULL,
      model_id TEXT,
      tokens_in INTEGER,
      tokens_out INTEGER,
      ms INTEGER,
      client TEXT,
      ip TEXT,
      status INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);

    CREATE TABLE IF NOT EXISTS stats_daily (
      date TEXT NOT NULL,
      model_id TEXT NOT NULL,
      tokens_in INTEGER NOT NULL DEFAULT 0,
      tokens_out INTEGER NOT NULL DEFAULT 0,
      active_seconds INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (date, model_id)
    );

    CREATE TABLE IF NOT EXISTS permission_rules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      tool TEXT NOT NULL,
      exact TEXT,
      scope TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent_memory (
      id TEXT PRIMARY KEY,
      text TEXT NOT NULL,
      source TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tasks (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      text TEXT NOT NULL,
      done INTEGER NOT NULL DEFAULT 0,
      ord INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS mcp_servers (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      transport TEXT NOT NULL,       -- 'stdio' | 'http'
      command TEXT,
      args TEXT,
      url TEXT,
      env TEXT,
      enabled INTEGER NOT NULL DEFAULT 1,
      created_at INTEGER NOT NULL
    );
    `
  },
  {
    id: 2,
    sql: `
    CREATE TABLE IF NOT EXISTS scheduled_tasks (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      prompt TEXT NOT NULL,
      cron TEXT NOT NULL,
      cwd TEXT,
      enabled INTEGER NOT NULL DEFAULT 1,
      last_run INTEGER,
      next_run INTEGER,
      created_at INTEGER NOT NULL
    );
    `
  }
]

export function getDb(): DatabaseSync {
  if (db) return db
  fs.mkdirSync(APPDATA_DIR, { recursive: true })
  db = new DatabaseSync(DB_FILE)
  db.exec('PRAGMA journal_mode = WAL;')
  db.exec('PRAGMA foreign_keys = ON;')
  db.exec('CREATE TABLE IF NOT EXISTS _migrations (id INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL);')

  const applied = new Set(
    (db.prepare('SELECT id FROM _migrations').all() as { id: number }[]).map((r) => r.id)
  )
  for (const m of MIGRATIONS) {
    if (applied.has(m.id)) continue
    db.exec(m.sql)
    db.prepare('INSERT INTO _migrations (id, applied_at) VALUES (?, ?)').run(m.id, Date.now())
  }
  return db
}

export function closeDb(): void {
  db?.close()
  db = null
}

/** Convenience wrappers so call sites stay readable. */
export function run(sql: string, ...params: unknown[]): void {
  getDb().prepare(sql).run(...(params as never[]))
}

export function all<T>(sql: string, ...params: unknown[]): T[] {
  return getDb().prepare(sql).all(...(params as never[])) as T[]
}

export function get<T>(sql: string, ...params: unknown[]): T | undefined {
  return getDb().prepare(sql).get(...(params as never[])) as T | undefined
}

export function prepare(sql: string): StatementSync {
  return getDb().prepare(sql)
}

/** Run a set of writes atomically. */
export function transaction<T>(fn: () => T): T {
  const d = getDb()
  d.exec('BEGIN')
  try {
    const result = fn()
    d.exec('COMMIT')
    return result
  } catch (err) {
    d.exec('ROLLBACK')
    throw err
  }
}

/** Export a chat to markdown or JSON. */
export function exportChat(chatId: string, format: 'md' | 'json'): string {
  const chat = get<{ id: string; title: string; created_at: number; model_id: string | null }>(
    'SELECT * FROM chats WHERE id = ?',
    chatId
  )
  if (!chat) throw new Error(`No chat ${chatId}`)

  const messages = all<{
    role: string
    content: string
    created_at: number
    tool_calls: string | null
    tool_result: string | null
  }>('SELECT role, content, created_at, tool_calls, tool_result FROM messages WHERE chat_id = ? ORDER BY created_at', chatId)

  if (format === 'json') {
    return JSON.stringify({ chat, messages }, null, 2)
  }

  const lines = [`# ${chat.title}`, '', `_${new Date(chat.created_at).toISOString()}_`, '']
  for (const m of messages) {
    if (m.role === 'tool' && m.tool_calls) {
      const call = JSON.parse(m.tool_calls)[0] as { name: string; args: Record<string, unknown> }
      lines.push(`### tool: ${call.name}`, '', '```json', JSON.stringify(call.args, null, 2), '```', '', '```', m.content, '```', '')
    } else {
      lines.push(`### ${m.role}`, '', m.content, '')
    }
  }
  return lines.join('\n')
}

export function writeExport(chatId: string, format: 'md' | 'json', dir: string): string {
  const content = exportChat(chatId, format)
  const chat = get<{ title: string }>('SELECT title FROM chats WHERE id = ?', chatId)
  const safe = (chat?.title ?? chatId).replace(/[^a-z0-9._-]+/gi, '-').slice(0, 60)
  const file = path.join(dir, `${safe}.${format}`)
  fs.writeFileSync(file, content, 'utf8')
  return file
}
