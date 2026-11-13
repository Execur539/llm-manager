/**
 * Chat and agent-session persistence.
 *
 * Chats and agent sessions share one table distinguished by `kind`, because they are the same
 * thing with different tool access — and it means session resume, export and search are
 * written once.
 */

import { cleanTitle } from './title'
import crypto from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { all, get, run, transaction } from '../storage/db'
import {
  LockedError,
  chatBox,
  chatOfFile,
  forgetOwner,
  isLocked,
  plainBox,
  primeRecord,
  protectedIds,
  readOpen,
  statusOf,
  type Box
} from '../protect/keys'
import type { AgentMessage, AgentSessionState, MessageAttachment } from '@shared/types'

export interface ChatSummary {
  id: string
  title: string
  kind: 'chat' | 'agent'
  modelId: string | null
  cwd: string | null
  createdAt: number
  updatedAt: number
  messageCount: number
  /** Stored encrypted; see protect/. */
  protected?: boolean
  /** Protected by a password and not unlocked: the title is a stand-in and nothing else can be read. */
  locked?: boolean
  /** Whether a password is set. */
  password?: boolean
}

interface ChatRow {
  id: string
  title: string
  model_id: string | null
  preset_id: string | null
  system_prompt: string | null
  kind: string
  cwd: string | null
  created_at: number
  updated_at: number
  context_used: number | null
  summary: string | null
  summary_upto: string | null
  title_by_user: number
  protection: string | null
}

const lockedTitle = (kind: string): string => (kind === 'agent' ? 'Locked session' : 'Locked chat')

/** The chat's box, from a row that already says whether it is protected. */
function boxOfRow(r: ChatRow): Box {
  primeRecord({ kind: 'chat', id: r.id }, r.protection)
  return chatBox(r.id)
}

/** The box of the conversation a message belongs to. */
function boxOfMessage(messageId: string): Box {
  const row = get<{ chat_id: string }>('SELECT chat_id FROM messages WHERE id = ?', messageId)
  return row ? chatBox(row.chat_id) : plainBox()
}

/** The box of the conversation an attachment belongs to. */
function boxOfAttachment(id: string): Box {
  const row = get<{ chat_id: string }>(
    'SELECT m.chat_id FROM attachments a JOIN messages m ON m.id = a.message_id WHERE a.id = ?',
    id
  )
  return row ? chatBox(row.chat_id) : plainBox()
}

/**
 * How an attachment's path is stored. The app's kept copies have names that say nothing and stay
 * readable, so tidying files never needs a key; a path to the user's own file is encrypted with the
 * rest when the conversation is protected.
 */
const storedPath = (box: Box, file: string): string => (chatOfFile(file) ? file : box.seal(file))

interface MessageRow {
  id: string
  chat_id: string
  parent_id: string | null
  role: string
  content: string
  reasoning: string | null
  plan: string | null
  tool_calls: string | null
  tool_result: string | null
  follow_up: string | null
  created_at: number
}

export function createChat(opts: {
  title?: string
  kind?: 'chat' | 'agent'
  modelId?: string | null
  cwd?: string | null
  systemPrompt?: string | null
}): ChatSummary {
  const id = crypto.randomBytes(8).toString('hex')
  const now = Date.now()
  run(
    'INSERT INTO chats (id, title, model_id, preset_id, system_prompt, kind, cwd, created_at, updated_at) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?)',
    id,
    opts.title ?? 'New chat',
    opts.modelId ?? null,
    opts.systemPrompt ?? null,
    opts.kind ?? 'chat',
    opts.cwd ?? null,
    now,
    now
  )
  return {
    id,
    title: opts.title ?? 'New chat',
    kind: opts.kind ?? 'chat',
    modelId: opts.modelId ?? null,
    cwd: opts.cwd ?? null,
    createdAt: now,
    updatedAt: now,
    messageCount: 0
  }
}

export function listChats(kind?: 'chat' | 'agent'): ChatSummary[] {
  const rows = kind
    ? all<ChatRow & { n: number }>(
        'SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS n FROM chats c WHERE kind = ? ORDER BY updated_at DESC',
        kind
      )
    : all<ChatRow & { n: number }>(
        'SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS n FROM chats c ORDER BY updated_at DESC'
      )

  return rows.map((r) => {
    const box = boxOfRow(r)
    const status = statusOf({ kind: 'chat', id: r.id })
    let title = lockedTitle(r.kind)
    if (!status.locked) {
      try {
        title = box.open(r.title)
      } catch {
        title = '(unreadable title)'
      }
    }
    return {
      id: r.id,
      title,
      kind: r.kind as 'chat' | 'agent',
      modelId: r.model_id,
      cwd: r.cwd,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
      // A locked conversation shows only that it exists and when it was last used.
      messageCount: status.locked ? 0 : r.n,
      ...(status.protected ? { protected: true, locked: status.locked, password: status.password } : {})
    }
  })
}

/** The conversation's own fields, readable, for an export. Throws while it is locked. */
export function chatHeader(id: string): {
  id: string
  title: string
  kind: string
  model_id: string | null
  cwd: string | null
  created_at: number
  updated_at: number
  summary: string | null
} | null {
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', id)
  if (!row) return null
  const box = boxOfRow(row)
  if (box.locked) throw new LockedError('chat')
  return {
    id: row.id,
    title: box.open(row.title),
    kind: row.kind,
    model_id: row.model_id,
    cwd: row.cwd,
    created_at: row.created_at,
    updated_at: row.updated_at,
    summary: box.openOrNull(row.summary)
  }
}

/** The title as the user sees it: the stand-in while locked. */
export function chatTitle(id: string): string | null {
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', id)
  if (!row) return null
  const box = boxOfRow(row)
  return box.locked ? lockedTitle(row.kind) : box.open(row.title)
}

/** A name the user gave; the model never renames over it. */
export function renameChat(id: string, title: string): void {
  run('UPDATE chats SET title = ?, title_by_user = 1, updated_at = ? WHERE id = ?', chatBox(id).seal(title), Date.now(), id)
}

/**
 * A name the model gave, tidied into a sidebar title.
 *
 * Returns the title now shown, or null when nothing changed: the user named the conversation
 * themselves, there is no such conversation, or the model's text held no usable title.
 */
export function setModelTitle(id: string, text: string): string | null {
  const title = cleanTitle(text)
  if (!title) return null
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', id)
  if (!row || row.title_by_user) return null
  const box = boxOfRow(row)
  if (box.open(row.title) === title) return null
  // Not a change of activity, so the conversation keeps its place in the list.
  run('UPDATE chats SET title = ? WHERE id = ?', box.seal(title), id)
  return title
}

/** What deciding whether to ask the model for a title needs. */
export function titleState(id: string): { title: string; byUser: boolean; userTurns: number } | null {
  const row = get<ChatRow & { turns: number }>(
    "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id AND m.role = 'user') AS turns FROM chats c WHERE id = ?",
    id
  )
  return row ? { title: boxOfRow(row).open(row.title), byUser: !!row.title_by_user, userTurns: row.turns } : null
}

export function deleteChat(id: string): void {
  run('DELETE FROM chats WHERE id = ?', id)
  forgetOwner({ kind: 'chat', id })
}

export function searchChats(query: string): { chatId: string; title: string; snippet: string }[] {
  // `%` and `_` are LIKE wildcards, so a literal search for "50%" or "read_file" matched far
  // more than the user asked for — and then the snippet could not find the term it matched on.
  const escaped = query.replace(/[\\%_]/g, (c) => `\\${c}`)
  const found = all<{ chat_id: string; title: string; content: string; created_at: number }>(
    `SELECT m.chat_id, c.title, m.content, m.created_at
     FROM messages m JOIN chats c ON c.id = m.chat_id
     WHERE c.protection IS NULL AND m.content LIKE ? ESCAPE '\\' ORDER BY m.created_at DESC LIMIT 50`,
    `%${escaped}%`
  )
  /*
   * Protected conversations are never matched in the database, where their words are encrypted.
   * The ones that are unlocked are searched here instead, in memory; locked ones are not searched.
   */
  const needle = query.toLowerCase()
  for (const id of protectedIds('chat')) {
    if (isLocked({ kind: 'chat', id })) continue
    const box = chatBox(id)
    const title = chatTitle(id) ?? ''
    for (const r of all<{ content: string; created_at: number }>('SELECT content, created_at FROM messages WHERE chat_id = ?', id)) {
      const content = box.open(r.content)
      if (content.toLowerCase().includes(needle)) found.push({ chat_id: id, title, content, created_at: r.created_at })
    }
  }
  return found
    .sort((a, b) => b.created_at - a.created_at)
    .slice(0, 50)
    .map((r) => {
    const idx = r.content.toLowerCase().indexOf(query.toLowerCase())
    const start = Math.max(0, idx - 60)
    return {
      chatId: r.chat_id,
      title: r.title,
      snippet: `${start > 0 ? '…' : ''}${r.content.slice(start, start + 160)}…`
    }
  })
}

export function appendMessage(chatId: string, message: AgentMessage): void {
  const box = chatBox(chatId)
  transaction(() => {
    /*
     * An upsert, not INSERT OR REPLACE.
     *
     * `REPLACE` resolves a conflict by deleting the existing row and inserting a new one, and
     * attachments reference messages with ON DELETE CASCADE — so re-storing a message silently
     * took its files with it. Storing the same message twice is routine here: the agent loop
     * writes the user's turn under an id the bridge minted before it, and every turn that gets
     * revised comes back through this function.
     *
     * Verified against node:sqlite with foreign keys on — REPLACE dropped the child rows, the
     * upsert kept them.
     */
    run(
      `INSERT INTO messages (id, chat_id, parent_id, role, content, reasoning, plan, tool_calls, tool_result, follow_up, created_at)
       VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         chat_id = excluded.chat_id,
         role = excluded.role,
         content = excluded.content,
         reasoning = excluded.reasoning,
         plan = excluded.plan,
         tool_calls = excluded.tool_calls,
         tool_result = excluded.tool_result,
         follow_up = excluded.follow_up,
         created_at = excluded.created_at`,
      message.id,
      chatId,
      message.role,
      box.seal(message.content),
      box.sealOrNull(message.reasoning),
      box.sealOrNull(message.plan),
      message.toolCalls ? box.seal(JSON.stringify(message.toolCalls)) : null,
      message.toolResult ? box.seal(JSON.stringify(message.toolResult)) : null,
      box.sealOrNull(message.followUp),
      message.createdAt
    )
    run('UPDATE chats SET updated_at = ? WHERE id = ?', Date.now(), chatId)
  })
}

/**
 * A conversation's messages in the order they happened.
 *
 * Ordered by rowid as well as timestamp. `created_at` is milliseconds, and an agent turn writes
 * an assistant message and its tool result in the same one routinely — leaving the tie to
 * SQLite meant a transcript that could come back with a tool result above the call that made it.
 */
export function loadMessages(chatId: string): AgentMessage[] {
  const box = chatBox(chatId)
  if (box.locked) throw new LockedError('chat')
  const attachments = attachmentsByMessage(chatId)
  return all<MessageRow>('SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at, rowid', chatId).map((r) => ({
    id: r.id,
    role: r.role as AgentMessage['role'],
    content: box.open(r.content),
    attachments: attachments.get(r.id),
    reasoning: box.openOrNull(r.reasoning) ?? undefined,
    plan: box.openOrNull(r.plan) ?? undefined,
    toolCalls: r.tool_calls ? JSON.parse(box.open(r.tool_calls)) : undefined,
    toolResult: r.tool_result ? JSON.parse(box.open(r.tool_result)) : undefined,
    followUp: box.openOrNull(r.follow_up) ?? undefined,
    createdAt: r.created_at
  }))
}

/**
 * Drop everything from a message onward — used when rewinding to a checkpoint.
 *
 * Cut at the same (created_at, rowid) position the transcript is ordered by. Comparing
 * timestamps alone made the boundary ambiguous whenever messages shared a millisecond: rewinding
 * to a tool result also deleted the assistant turn that preceded it, and rewinding to that turn
 * left the result behind.
 */
export function truncateFrom(chatId: string, messageId: string): number {
  const target = get<MessageRow & { rowid: number }>('SELECT rowid, * FROM messages WHERE id = ?', messageId)
  if (!target) return 0
  const where = 'chat_id = ? AND (created_at > ? OR (created_at = ? AND rowid >= ?))'
  const args = [chatId, target.created_at, target.created_at, target.rowid] as const
  const doomed = all<{ id: string }>(`SELECT id FROM messages WHERE ${where}`, ...args)
  run(`DELETE FROM messages WHERE ${where}`, ...args)
  return doomed.length
}

/** Restore a full agent session for resume after a close or crash. */
export function loadSession(chatId: string): AgentSessionState | null {
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', chatId)
  if (!row) return null
  const box = boxOfRow(row)
  if (box.locked) throw new LockedError('chat')
  return {
    id: row.id,
    title: box.open(row.title),
    /*
     * The home directory, not the process's.
     *
     * `process.cwd()` is wherever the launcher happened to start the app — for a portable build
     * that is the extraction cache under LOCALAPPDATA, which is deleted on upgrade. A session
     * saved without a folder would have pointed the agent's writes at disposable storage. Home is
     * what the agent is constructed with elsewhere, so this matches.
     */
    cwd: row.cwd ?? os.homedir(),
    planMode: false,
    messages: loadMessages(chatId),
    taskList: all<{ id: string; text: string; done: number }>(
      'SELECT id, text, done FROM tasks WHERE chat_id = ? ORDER BY ord',
      chatId
    ).map((t) => ({ id: t.id, text: box.open(t.text), done: !!t.done })),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    contextUsed: row.context_used ?? undefined,
    summary: box.openOrNull(row.summary) ?? undefined,
    summaryUpto: row.summary_upto ?? undefined
  }
}

/** Whether a conversation is locked: protected by a password and not unlocked. */
export function chatLocked(chatId: string): boolean {
  return isLocked({ kind: 'chat', id: chatId })
}

/**
 * Remember how much context this conversation was last measured to occupy.
 *
 * Written per turn rather than per update: the live figure changes several times a second while
 * a response streams, and persisting that would be a database write per frame to record a number
 * that is superseded immediately. What matters across a restart is where the conversation ended
 * up, not every step it took getting there.
 */
export function setSummary(chatId: string, summary: string, uptoMessageId: string): void {
  run('UPDATE chats SET summary = ?, summary_upto = ? WHERE id = ?', chatBox(chatId).seal(summary), uptoMessageId, chatId)
}

/**
 * Forget a session's summary, so the model sees the whole transcript again.
 *
 * Needed when history is rewound past the point the summary covers: a summary of messages that
 * no longer exist describes work the session has been told never happened.
 */
export function clearSummary(chatId: string): void {
  run('UPDATE chats SET summary = NULL, summary_upto = NULL WHERE id = ?', chatId)
}

/** A conversation's stored summary and the last message it covers, without loading the messages. */
export function getSummary(chatId: string): { summary: string; uptoMessageId: string } | null {
  const row = get<{ summary: string | null; summary_upto: string | null }>(
    'SELECT summary, summary_upto FROM chats WHERE id = ?',
    chatId
  )
  return row?.summary && row.summary_upto ? { summary: chatBox(chatId).open(row.summary), uptoMessageId: row.summary_upto } : null
}

/** How much context a conversation was last measured — or, straight after a compaction, estimated — to fill. */
export function getContextUsed(chatId: string): number | null {
  return get<{ context_used: number | null }>('SELECT context_used FROM chats WHERE id = ?', chatId)?.context_used ?? null
}

export function setContextUsed(chatId: string, tokens: number): void {
  run('UPDATE chats SET context_used = ? WHERE id = ?', Math.max(0, Math.round(tokens)), chatId)
}

/** The most recent unfinished agent session, offered for resume on launch. */
export function mostRecentAgentSession(): ChatSummary | null {
  const rows = listChats('agent')
  return rows.length ? rows[0] : null
}

/**
 * The system prompt and tools an agent conversation first sent, and what it has told the model
 * about the session since, as JSON. See agent/prompt-state.
 */
export function getPromptState(chatId: string): string | null {
  return chatBox(chatId).openOrNull(
    get<{ prompt_state: string | null }>('SELECT prompt_state FROM chats WHERE id = ?', chatId)?.prompt_state ?? null
  )
}

/** Null forgets it, so the next turn fixes the prompt and tools of the moment instead. */
export function setPromptState(chatId: string, state: string | null): void {
  run('UPDATE chats SET prompt_state = ? WHERE id = ?', chatBox(chatId).sealOrNull(state), chatId)
}

export function setChatCwd(chatId: string, cwd: string): void {
  run('UPDATE chats SET cwd = ?, updated_at = ? WHERE id = ?', cwd, Date.now(), chatId)
}

/**
 * Give a chat a title from its first user message, so the sidebar is readable without the
 * user having to name anything.
 */
/** Record which model a conversation last ran on. */
export function setChatModel(chatId: string, modelId: string | null): void {
  run('UPDATE chats SET model_id = ? WHERE id = ?', modelId, chatId)
}

export function autoTitle(chatId: string): void {
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', chatId)
  if (!row) return
  const box = boxOfRow(row)
  const current = box.open(row.title)
  // Both placeholders: agent sessions start as 'New session', and were never given a title.
  if (current !== 'New chat' && current !== 'New session') return
  const first = get<MessageRow>(
    "SELECT * FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY created_at, rowid LIMIT 1",
    chatId
  )
  if (!first) return
  // A stand-in until the model names it, so it is not marked as a name the user chose.
  const title = box.open(first.content).replace(/\s+/g, ' ').trim().slice(0, 60)
  if (title) run('UPDATE chats SET title = ?, updated_at = ? WHERE id = ?', box.seal(title), Date.now(), chatId)
}

// ---------------------------------------------------------------- attachments

export interface AttachmentInput {
  path: string
  kind: 'image' | 'audio' | 'video' | 'doc'
}

export function recordAttachment(messageId: string, att: AttachmentInput, meta?: AttachmentMeta): string {
  const id = crypto.randomBytes(6).toString('hex')
  const box = boxOfMessage(messageId)
  // A protected conversation's kept copy has a name that says nothing; the name it had goes in the record.
  const full = box.on ? { ...meta, name: meta?.name ?? path.basename(meta?.original ?? att.path) } : meta
  run(
    'INSERT INTO attachments (id, message_id, kind, path, meta, created_at) VALUES (?, ?, ?, ?, ?, ?)',
    id,
    messageId,
    att.kind,
    storedPath(box, att.path),
    full && Object.keys(full).length ? box.seal(JSON.stringify(full)) : null,
    Date.now()
  )
  return id
}

/**
 * Fill in what preparing an attachment produced, after the row already exists.
 *
 * The row is written before the file is processed so the transcript can show it straight away —
 * a two-minute video can take a minute to sample, and a message that does not appear until that
 * finishes looks like the app has hung. What the sampler made of it is known only afterwards.
 */
export function setAttachmentMeta(id: string, meta: AttachmentMeta): void {
  const row = get<{ meta: string | null }>('SELECT meta FROM attachments WHERE id = ?', id)
  const box = boxOfAttachment(id)
  // Merged, so what preparing the file found does not erase where its kept copy came from.
  const defined = Object.fromEntries(Object.entries(meta ?? {}).filter(([, v]) => v !== undefined))
  const merged = { ...parseMeta(box.openOrNull(row?.meta ?? null)), ...defined }
  run('UPDATE attachments SET meta = ? WHERE id = ?', Object.keys(merged).length ? box.seal(JSON.stringify(merged)) : null, id)
}

export function attachmentsFor(messageId: string): { id: string; kind: string; path: string }[] {
  const box = boxOfMessage(messageId)
  return all<{ id: string; kind: string; path: string }>(
    'SELECT id, kind, path FROM attachments WHERE message_id = ?',
    messageId
  ).map((r) => ({ ...r, path: box.open(r.path) }))
}

/** What was stored alongside an attachment when the message was sent. */
export interface AttachmentMeta {
  /** The re-encoded clip actually sent to the model, when one was built. */
  optimised?: string
  stills?: string[]
  note?: string
  /** The user's own file, when the attachment now points at the app's kept copy of it. */
  original?: string
  /** The name to show, when the kept copy's own name says nothing (a protected conversation's). */
  name?: string
}

function parseMeta(raw: string | null): AttachmentMeta {
  try {
    return (JSON.parse(raw ?? '{}') as AttachmentMeta) ?? {}
  } catch {
    return {}
  }
}

/**
 * Resolve an attachment to a file on disk, by id.
 *
 * The one place that turns an id into a path, so the media route can serve attachments without
 * ever accepting a path from a client. `which` picks between the file the user attached and the
 * clip that was sent in its place.
 */
export function attachmentFile(id: string, which: 'source' | 'optimised' = 'source'): string | null {
  const row = get<{ path: string; meta: string | null }>('SELECT path, meta FROM attachments WHERE id = ?', id)
  if (!row) return null
  const box = boxOfAttachment(id)
  if (box.locked) return null
  if (which === 'source') return box.open(row.path)
  return parseMeta(box.openOrNull(row.meta)).optimised ?? null
}

/**
 * Every attachment in a conversation, grouped by the message it belongs to.
 *
 * One query rather than one per message: a long transcript with attachments scattered through it
 * would otherwise pay a round trip per turn every time it was opened.
 *
 * Paths are deliberately dropped here. The renderer addresses files by id through a route that
 * looks them up again, so a remote session cannot name a file of its own choosing, and the
 * machine's directory layout never reaches a browser.
 */
export function attachmentsByMessage(chatId: string): Map<string, MessageAttachment[]> {
  const rows = all<{ id: string; message_id: string; kind: string; path: string; meta: string | null }>(
    `SELECT a.id, a.message_id, a.kind, a.path, a.meta
       FROM attachments a
       JOIN messages m ON m.id = a.message_id
      WHERE m.chat_id = ?
      ORDER BY a.created_at, a.rowid`,
    chatId
  )

  const box = chatBox(chatId)
  const byMessage = new Map<string, MessageAttachment[]>()
  for (const r of rows) {
    const list = byMessage.get(r.message_id) ?? []
    list.push(toMessageAttachment(r, box))
    byMessage.set(r.message_id, list)
  }
  return byMessage
}

/** The attachments on one message, for announcing a turn that has just been sent. */
export function attachmentsForMessage(messageId: string): MessageAttachment[] {
  const box = boxOfMessage(messageId)
  return all<{ id: string; kind: string; path: string; meta: string | null }>(
    'SELECT id, kind, path, meta FROM attachments WHERE message_id = ? ORDER BY created_at, rowid',
    messageId
  ).map((r) => toMessageAttachment(r, box))
}

/** Every attachment in a conversation, with its path and extras, for keeping its files. */
export function attachmentRowsForChat(chatId: string): { id: string; kind: string; path: string; meta: AttachmentMeta }[] {
  return all<{ id: string; kind: string; path: string; meta: string | null }>(
    `SELECT a.id, a.kind, a.path, a.meta
       FROM attachments a
       JOIN messages m ON m.id = a.message_id
      WHERE m.chat_id = ?`,
    chatId
  ).map((r) => {
    const box = chatBox(chatId)
    return { id: r.id, kind: r.kind, path: box.open(r.path), meta: parseMeta(box.openOrNull(r.meta)) }
  })
}

/** Point an attachment at the app's kept copy of its file, with its extras. */
export function setAttachmentStored(id: string, file: string, meta: AttachmentMeta): void {
  const box = boxOfAttachment(id)
  run('UPDATE attachments SET path = ?, meta = ? WHERE id = ?', storedPath(box, file), box.seal(JSON.stringify(meta)), id)
}

/** Every file any attachment refers to, clips and stills included: what the kept copies must not lose. */
export function allAttachmentPaths(): string[] {
  // A locked conversation's records cannot be read; tidying leaves its folder alone (lockedChatIds).
  const boxes = new Map<string, Box>()
  return all<{ path: string; meta: string | null; chat_id: string }>(
    'SELECT a.path, a.meta, m.chat_id FROM attachments a JOIN messages m ON m.id = a.message_id'
  ).flatMap((r) => {
    const box = boxes.get(r.chat_id) ?? chatBox(r.chat_id)
    boxes.set(r.chat_id, box)
    if (box.locked) return []
    const meta = parseMeta(box.openOrNull(r.meta))
    return [box.open(r.path), ...(meta.optimised ? [meta.optimised] : []), ...(meta.stills ?? [])]
  })
}

/** Protected conversations that are locked: their files are left alone, since what refers to them cannot be read. */
export function lockedChatIds(): string[] {
  return protectedIds('chat').filter((id) => isLocked({ kind: 'chat', id }))
}

/** One message's attachments, with their paths and extras. */
export function attachmentRowsForMessage(messageId: string): { id: string; kind: string; path: string; meta: AttachmentMeta }[] {
  return all<{ id: string; kind: string; path: string; meta: string | null }>(
    'SELECT id, kind, path, meta FROM attachments WHERE message_id = ? ORDER BY created_at, rowid',
    messageId
  ).map((r) => {
    const box = boxOfMessage(messageId)
    return { id: r.id, kind: r.kind, path: box.open(r.path), meta: parseMeta(box.openOrNull(r.meta)) }
  })
}

/** Record what a message sent the model, as stored-content JSON — see sent-content. */
export function saveSentContent(chatId: string, messageId: string, content: string): void {
  run(
    `INSERT INTO message_content (message_id, chat_id, content, created_at) VALUES (?, ?, ?, ?)
     ON CONFLICT(message_id) DO UPDATE SET chat_id = excluded.chat_id, content = excluded.content, created_at = excluded.created_at`,
    messageId,
    chatId,
    chatBox(chatId).seal(content),
    Date.now()
  )
}

/** What a message sent the model, as stored-content JSON, or null when nothing beyond its text was recorded. */
export function loadSentContent(messageId: string): string | null {
  const row = get<{ content: string; chat_id: string }>('SELECT content, chat_id FROM message_content WHERE message_id = ?', messageId)
  return row ? chatBox(row.chat_id).open(row.content) : null
}

/** Every file a sent-content record refers to: what the kept copies must not lose. */
export function sentContentFiles(): string[] {
  const boxes = new Map<string, Box>()
  return all<{ content: string; chat_id: string }>('SELECT content, chat_id FROM message_content').flatMap((r) => {
    try {
      const box = boxes.get(r.chat_id) ?? chatBox(r.chat_id)
      boxes.set(r.chat_id, box)
      if (box.locked) return []
      const parts = JSON.parse(box.open(r.content)) as unknown
      return Array.isArray(parts)
        ? parts.flatMap((p) => (p && typeof p.file === 'string' ? [p.file as string] : []))
        : []
    } catch {
      return []
    }
  })
}

/** Drop records whose message no longer exists, left behind by deleted or rewound messages. */
export function pruneSentContent(olderThan: number): void {
  run('DELETE FROM message_content WHERE created_at < ? AND message_id NOT IN (SELECT id FROM messages)', olderThan)
}

function toMessageAttachment(r: { id: string; kind: string; path: string; meta: string | null }, box: Box): MessageAttachment {
  // A malformed row costs the attachment its extras, not its place in the transcript.
  const meta = parseMeta(box.openOrNull(r.meta))
  const file = box.open(r.path)
  return {
    id: r.id,
    kind: (r.kind as MessageAttachment['kind']) ?? 'doc',
    name: meta.name ?? path.basename(file),
    bytes: fileBytes(file),
    // Checked rather than trusted: the clip lives in a scratch directory, and offering a toggle
    // that opens a dead player is worse than not offering one.
    optimised: !!meta.optimised && fs.existsSync(meta.optimised),
    note: meta.note
  }
}

/** Size on disk, or undefined if the file has since been moved or deleted. */
function fileBytes(file: string): number | undefined {
  try {
    return fs.statSync(file).size
  } catch {
    return undefined
  }
}

/** Classify a dropped file so the UI and the model handler agree on what it is. */
/** Extensions without the dot, so they can be handed straight to a dialog filter. */
export const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']
export const AUDIO_EXT = ['mp3', 'wav', 'flac', 'ogg', 'm4a']
export const VIDEO_EXT = ['mp4', 'mkv', 'mov', 'webm', 'avi']

/**
 * Text-ish formats worth offering in a picker.
 *
 * Not exhaustive and not a gate: anything unrecognised is still classified as a document and the
 * extractor decides whether it is readable by looking at the bytes. This list only shapes what
 * the file dialog suggests.
 */
export const TEXT_EXT = [
  'txt', 'md', 'markdown', 'rst', 'log', 'csv', 'tsv', 'json', 'yaml', 'yml', 'toml', 'ini', 'env',
  'ts', 'tsx', 'js', 'jsx', 'mjs', 'cjs', 'py', 'rb', 'go', 'rs', 'java', 'kt', 'swift', 'c', 'h',
  'cpp', 'hpp', 'cc', 'cs', 'php', 'sh', 'bash', 'zsh', 'ps1', 'bat', 'sql', 'html', 'htm', 'css',
  'scss', 'xml', 'svg', 'gradle', 'dockerfile', 'makefile', 'pdf'
]

export function classifyAttachment(file: string): AttachmentInput['kind'] {
  const ext = path.extname(file).toLowerCase().replace(/^\./, '')
  if (IMAGE_EXT.includes(ext)) return 'image'
  if (AUDIO_EXT.includes(ext)) return 'audio'
  if (VIDEO_EXT.includes(ext)) return 'video'
  return 'doc'
}

/** Read a media file as a data URL for the OpenAI-style multimodal message format. */
export async function toDataUrl(file: string): Promise<string> {
  // A protected conversation's kept copy is decrypted on the way out.
  const buf = await readOpen(file)
  const ext = path.extname(file).toLowerCase().slice(1)
  const mime =
    ext === 'jpg' ? 'image/jpeg' : ext === 'svg' ? 'image/svg+xml' : `image/${ext}`
  return `data:${mime};base64,${buf.toString('base64')}`
}
