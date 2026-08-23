/**
 * Chat and agent-session persistence.
 *
 * Chats and agent sessions share one table distinguished by `kind`, because they are the same
 * thing with different tool access — and it means session resume, export and search are
 * written once.
 */

import crypto from 'node:crypto'
import path from 'node:path'
import fsp from 'node:fs/promises'
import { all, get, run, transaction } from '../storage/db'
import type { AgentMessage, AgentSessionState } from '@shared/types'

export interface ChatSummary {
  id: string
  title: string
  kind: 'chat' | 'agent'
  modelId: string | null
  cwd: string | null
  createdAt: number
  updatedAt: number
  messageCount: number
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
}

interface MessageRow {
  id: string
  chat_id: string
  parent_id: string | null
  role: string
  content: string
  tool_calls: string | null
  tool_result: string | null
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

  return rows.map((r) => ({
    id: r.id,
    title: r.title,
    kind: r.kind as 'chat' | 'agent',
    modelId: r.model_id,
    cwd: r.cwd,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
    messageCount: r.n
  }))
}

export function renameChat(id: string, title: string): void {
  run('UPDATE chats SET title = ?, updated_at = ? WHERE id = ?', title, Date.now(), id)
}

export function deleteChat(id: string): void {
  run('DELETE FROM chats WHERE id = ?', id)
}

export function searchChats(query: string): { chatId: string; title: string; snippet: string }[] {
  return all<{ chat_id: string; title: string; content: string }>(
    `SELECT m.chat_id, c.title, m.content
     FROM messages m JOIN chats c ON c.id = m.chat_id
     WHERE m.content LIKE ? ORDER BY m.created_at DESC LIMIT 50`,
    `%${query}%`
  ).map((r) => {
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
  transaction(() => {
    run(
      'INSERT OR REPLACE INTO messages (id, chat_id, parent_id, role, content, tool_calls, tool_result, created_at) VALUES (?, ?, NULL, ?, ?, ?, ?, ?)',
      message.id,
      chatId,
      message.role,
      message.content,
      message.toolCalls ? JSON.stringify(message.toolCalls) : null,
      message.toolResult ? JSON.stringify(message.toolResult) : null,
      message.createdAt
    )
    run('UPDATE chats SET updated_at = ? WHERE id = ?', Date.now(), chatId)
  })
}

export function loadMessages(chatId: string): AgentMessage[] {
  return all<MessageRow>('SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at', chatId).map((r) => ({
    id: r.id,
    role: r.role as AgentMessage['role'],
    content: r.content,
    toolCalls: r.tool_calls ? JSON.parse(r.tool_calls) : undefined,
    toolResult: r.tool_result ? JSON.parse(r.tool_result) : undefined,
    createdAt: r.created_at
  }))
}

/** Drop everything from a message onward — used when rewinding to a checkpoint. */
export function truncateFrom(chatId: string, messageId: string): number {
  const target = get<MessageRow>('SELECT * FROM messages WHERE id = ?', messageId)
  if (!target) return 0
  const doomed = all<{ id: string }>('SELECT id FROM messages WHERE chat_id = ? AND created_at >= ?', chatId, target.created_at)
  run('DELETE FROM messages WHERE chat_id = ? AND created_at >= ?', chatId, target.created_at)
  return doomed.length
}

/** Restore a full agent session for resume after a close or crash. */
export function loadSession(chatId: string): AgentSessionState | null {
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', chatId)
  if (!row) return null
  return {
    id: row.id,
    title: row.title,
    cwd: row.cwd ?? process.cwd(),
    planMode: false,
    messages: loadMessages(chatId),
    taskList: all<{ id: string; text: string; done: number }>(
      'SELECT id, text, done FROM tasks WHERE chat_id = ? ORDER BY ord',
      chatId
    ).map((t) => ({ id: t.id, text: t.text, done: !!t.done })),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  }
}

/** The most recent unfinished agent session, offered for resume on launch. */
export function mostRecentAgentSession(): ChatSummary | null {
  const rows = listChats('agent')
  return rows.length ? rows[0] : null
}

export function setChatCwd(chatId: string, cwd: string): void {
  run('UPDATE chats SET cwd = ?, updated_at = ? WHERE id = ?', cwd, Date.now(), chatId)
}

/**
 * Give a chat a title from its first user message, so the sidebar is readable without the
 * user having to name anything.
 */
export function autoTitle(chatId: string): void {
  const row = get<ChatRow>('SELECT * FROM chats WHERE id = ?', chatId)
  if (!row || row.title !== 'New chat') return
  const first = get<MessageRow>("SELECT * FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY created_at LIMIT 1", chatId)
  if (!first) return
  const title = first.content.replace(/\s+/g, ' ').trim().slice(0, 60)
  if (title) renameChat(chatId, title)
}

// ---------------------------------------------------------------- attachments

export interface AttachmentInput {
  path: string
  kind: 'image' | 'audio' | 'video' | 'doc'
}

export function recordAttachment(messageId: string, att: AttachmentInput, meta?: unknown): string {
  const id = crypto.randomBytes(6).toString('hex')
  run(
    'INSERT INTO attachments (id, message_id, kind, path, meta, created_at) VALUES (?, ?, ?, ?, ?, ?)',
    id,
    messageId,
    att.kind,
    att.path,
    meta ? JSON.stringify(meta) : null,
    Date.now()
  )
  return id
}

export function attachmentsFor(messageId: string): { id: string; kind: string; path: string }[] {
  return all<{ id: string; kind: string; path: string }>(
    'SELECT id, kind, path FROM attachments WHERE message_id = ?',
    messageId
  )
}

/** Classify a dropped file so the UI and the model handler agree on what it is. */
export function classifyAttachment(file: string): AttachmentInput['kind'] {
  const ext = path.extname(file).toLowerCase()
  if (['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'].includes(ext)) return 'image'
  if (['.mp3', '.wav', '.flac', '.ogg', '.m4a'].includes(ext)) return 'audio'
  if (['.mp4', '.mkv', '.mov', '.webm', '.avi'].includes(ext)) return 'video'
  return 'doc'
}

/** Read a media file as a data URL for the OpenAI-style multimodal message format. */
export async function toDataUrl(file: string): Promise<string> {
  const buf = await fsp.readFile(file)
  const ext = path.extname(file).toLowerCase().slice(1)
  const mime =
    ext === 'jpg' ? 'image/jpeg' : ext === 'svg' ? 'image/svg+xml' : `image/${ext}`
  return `data:${mime};base64,${buf.toString('base64')}`
}
