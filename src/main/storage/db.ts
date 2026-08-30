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
import { exportFilename, uniquePath } from './filenames'

let db: DatabaseSync | null = null

export const MIGRATIONS: { id: number; sql: string }[] = [
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
  },
  {
    id: 3,
    sql: `
    -- Reasoning models return their chain of thought separately from the answer. Stored so it
    -- survives a reload rather than existing only for the life of the stream.
    ALTER TABLE messages ADD COLUMN reasoning TEXT;
    `
  },
  {
    id: 4,
    sql: `
    -- tasks.chat_id was declared without a foreign key, so deleting a chat left its task list
    -- behind forever — every sibling table cascades. SQLite cannot add a constraint in place,
    -- so the table is rebuilt. Rows whose chat is already gone are dropped on the way across,
    -- which is the cleanup the missing cascade should have been doing all along.
    CREATE TABLE tasks_new (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
      text TEXT NOT NULL,
      done INTEGER NOT NULL DEFAULT 0,
      ord INTEGER NOT NULL DEFAULT 0
    );

    INSERT INTO tasks_new (id, chat_id, text, done, ord)
      SELECT t.id, t.chat_id, t.text, t.done, t.ord
      FROM tasks t
      WHERE EXISTS (SELECT 1 FROM chats c WHERE c.id = t.chat_id);

    DROP TABLE tasks;
    ALTER TABLE tasks_new RENAME TO tasks;
    CREATE INDEX IF NOT EXISTS idx_tasks_chat ON tasks(chat_id, ord);
    `
  },
  {
    id: 5,
    sql: `
    -- Ultra chooses a plan before an agent turn acts on it. It used to reach the transcript by
    -- being appended to the prompt, which put it inside the user's own message; stored properly,
    -- it belongs to the turn without pretending to be something the user typed.
    ALTER TABLE messages ADD COLUMN plan TEXT;
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
  /*
   * Each migration is all-or-nothing, including the row that records it.
   *
   * SQLite makes DDL transactional, and these need it: migration 4 rebuilds a table across four
   * statements, so a failure between the DROP and the RENAME would have left the database with
   * neither `tasks` nor a record of having tried — and the retry on next launch would then fail
   * on `tasks_new` already existing, permanently. Rolling back leaves the schema exactly as it
   * was, so the next launch tries the same migration from the same starting point.
   */
  for (const m of MIGRATIONS) {
    if (applied.has(m.id)) continue
    db.exec('BEGIN')
    try {
      db.exec(m.sql)
      db.prepare('INSERT INTO _migrations (id, applied_at) VALUES (?, ?)').run(m.id, Date.now())
      db.exec('COMMIT')
    } catch (err) {
      db.exec('ROLLBACK')
      throw err
    }
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
    reasoning: string | null
    tool_calls: string | null
    tool_result: string | null
  }>(
    // rowid breaks ties on created_at, which is a millisecond stamp — an assistant turn and the
    // tool result it produced routinely share one, and without the tiebreak the export could
    // print the answer before the question it answered.
    'SELECT role, content, created_at, reasoning, tool_calls, tool_result FROM messages WHERE chat_id = ? ORDER BY created_at, rowid',
    chatId
  )

  if (format === 'json') {
    return JSON.stringify({ chat, messages }, null, 2)
  }

  const lines = [`# ${chat.title}`, '', `_${new Date(chat.created_at).toISOString()}_`, '']
  for (const m of messages) {
    if (m.role === 'tool' && m.tool_calls) {
      const call = JSON.parse(m.tool_calls)[0] as { name: string; args: Record<string, unknown> }
      lines.push(`### tool: ${call.name}`, '', '```json', JSON.stringify(call.args, null, 2), '```', '', '```', m.content, '```', '')
    } else {
      lines.push(`### ${m.role}`, '')
      // Reasoning is stored alongside the answer; an export that drops it loses the part of a
      // thinking model's transcript the user most often wanted to keep.
      if (m.reasoning) lines.push('<details><summary>Reasoning</summary>', '', m.reasoning, '', '</details>', '')
      lines.push(m.content, '')
    }
  }
  return lines.join('\n')
}

export function writeExport(chatId: string, format: 'md' | 'json', dirOrFile: string): string {
  const content = exportChat(chatId, format)

  // A save dialog hands back a full path; a directory picker hands back a folder.
  let file: string
  if (path.extname(dirOrFile).toLowerCase() === `.${format}`) {
    file = dirOrFile
  } else {
    const chat = get<{ title: string }>('SELECT title FROM chats WHERE id = ?', chatId)
    const base = exportFilename(chat?.title ?? '', `chat-${chatId.slice(0, 8)}`)
    file = uniquePath(dirOrFile, base, format, (p) => fs.existsSync(p))
  }

  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, content, 'utf8')
  return file
}
