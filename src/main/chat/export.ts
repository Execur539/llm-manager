/**
 * Exporting a conversation to Markdown or JSON.
 *
 * Read through the repository, so a protected conversation exports as what it says (and only
 * while it is unlocked) rather than as the encrypted text it is stored as. The file written is
 * not encrypted; the window says so before a protected conversation is exported.
 */

import fs from 'node:fs'
import path from 'node:path'
import { exportFilename, uniquePath } from '../storage/filenames'
import { chatHeader, loadMessages } from './repo'

export function exportChat(chatId: string, format: 'md' | 'json'): string {
  const chat = chatHeader(chatId)
  if (!chat) throw new Error(`No chat ${chatId}`)

  // In the shape the export always had: the stored row's names, tool calls and results as JSON text.
  const messages = loadMessages(chatId).map((m) => ({
    role: m.role,
    content: m.content,
    created_at: m.createdAt,
    reasoning: m.reasoning ?? null,
    tool_calls: m.toolCalls ? JSON.stringify(m.toolCalls) : null,
    tool_result: m.toolResult ? JSON.stringify(m.toolResult) : null
  }))

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
    const base = exportFilename(chatHeader(chatId)?.title ?? '', `chat-${chatId.slice(0, 8)}`)
    file = uniquePath(dirOrFile, base, format, (p) => fs.existsSync(p))
  }

  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, content, 'utf8')
  return file
}
