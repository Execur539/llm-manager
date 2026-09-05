/**
 * Checks the database contracts that fail silently.
 *
 * Both of these bit for real. Attachments hang off messages with ON DELETE CASCADE, and messages
 * are stored more than once as a matter of course — the agent loop writes the user's turn under
 * an id the bridge minted before it. Written with INSERT OR REPLACE, SQLite resolved that by
 * deleting the row and inserting a new one, and the cascade quietly took the message's files with
 * it. Nothing failed; the attachments were simply gone by the time anyone looked.
 *
 * The other half is the ordering rule that the same foreign key implies: an attachment cannot be
 * written before the message it names. That one at least throws, but the throw was being caught
 * and logged, so the symptom was still just an absence.
 *
 * The migrations and the statement under test are read out of the source rather than restated,
 * so this cannot drift into passing against a schema the app no longer has.
 *
 *   node scripts/storage-check.mjs
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { DatabaseSync } from 'node:sqlite'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

let pass = 0
let fail = 0
function check(name, ok, detail = '') {
  if (ok) {
    pass++
    console.log(`  ok   ${name}`)
  } else {
    fail++
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

/** The real migration list, in order, pulled straight out of the module that defines it. */
function migrations() {
  const src = fs.readFileSync(path.join(ROOT, 'src/main/storage/db.ts'), 'utf8')
  const start = src.indexOf('export const MIGRATIONS')
  if (start === -1) throw new Error('MIGRATIONS not found in db.ts')

  // Each entry is `{ id: N, sql: `...` }`; the SQL is the only part that matters here.
  const out = []
  const re = /\{\s*id:\s*(\d+),\s*sql:\s*`([\s\S]*?)`\s*\}/g
  let m
  while ((m = re.exec(src.slice(start))) !== null) out.push({ id: Number(m[1]), sql: m[2] })
  return out.sort((a, b) => a.id - b.id)
}

/** The statement `appendMessage` actually runs, so this tests the app's SQL and not a copy. */
function appendMessageSql() {
  const src = fs.readFileSync(path.join(ROOT, 'src/main/chat/repo.ts'), 'utf8')
  const m = /run\(\s*`(INSERT INTO messages[\s\S]*?)`/.exec(src)
  if (m) return { sql: m[1], replace: false }
  const legacy = /run\(\s*'(INSERT OR REPLACE INTO messages[^']*)'/.exec(src)
  if (legacy) return { sql: legacy[1], replace: true }
  throw new Error('could not find the message-insert statement in repo.ts')
}

const db = new DatabaseSync(':memory:')
db.exec('PRAGMA foreign_keys = ON;')

console.log('\nschema')
const steps = migrations()
check('the migration list is readable and ordered', steps.length > 0 && steps[0].id === 1,
  `${steps.length} migrations, first id ${steps[0]?.id}`)
for (const step of steps) db.exec(step.sql)
const tables = db
  .prepare("SELECT name FROM sqlite_master WHERE type='table'")
  .all()
  .map((r) => r.name)
check('messages and attachments both exist after migrating',
  tables.includes('messages') && tables.includes('attachments'), tables.join(', '))

console.log('\nattachments survive their message being re-stored')

const insert = appendMessageSql()
check('the message insert is an upsert rather than INSERT OR REPLACE', !insert.replace,
  'REPLACE deletes the row and the ON DELETE CASCADE takes the attachments with it')

db.prepare("INSERT INTO chats (id, title, kind, created_at, updated_at) VALUES ('c1','t','agent',0,0)").run()

/** Run the app's own statement with the nine parameters it takes. */
function storeMessage(id, content) {
  db.prepare(insert.sql).run(id, 'c1', 'user', content, null, null, null, null, 1)
}

storeMessage('m1', 'first write')
db.prepare("INSERT INTO attachments (id, message_id, kind, path, meta, created_at) VALUES ('a1','m1','video','/clip.mp4',NULL,0)").run()

const before = db.prepare("SELECT COUNT(*) n FROM attachments WHERE message_id='m1'").get().n
storeMessage('m1', 'second write, same id')
const after = db.prepare("SELECT COUNT(*) n FROM attachments WHERE message_id='m1'").get().n

check('the attachment is there to begin with', before === 1, `${before}`)
check('re-storing the message keeps its attachments', after === 1,
  `${before} before, ${after} after — the cascade fired`)
check('re-storing the message does update it',
  db.prepare("SELECT content FROM messages WHERE id='m1'").get().content === 'second write, same id')

console.log('\nan attachment cannot precede its message')
let threw = false
try {
  db.prepare("INSERT INTO attachments (id, message_id, kind, path, meta, created_at) VALUES ('a2','nope','video','/x.mp4',NULL,0)").run()
} catch {
  threw = true
}
/*
 * Not a complaint about SQLite — a reminder to callers. Both send paths mint a message id before
 * the turn is stored, and the agent one recorded its files against that id first and lost them
 * to a caught exception.
 */
check('a row naming a message that does not exist is rejected', threw,
  'foreign keys are off, so a stray attachment row would be accepted and never render')

console.log('\ndeleting a message still takes its attachments')
db.prepare("DELETE FROM messages WHERE id='m1'").run()
check('the cascade still works where it is wanted',
  db.prepare("SELECT COUNT(*) n FROM attachments WHERE message_id='m1'").get().n === 0)

db.close()
console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
