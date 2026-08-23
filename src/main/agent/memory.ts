/**
 * Agent memory.
 *
 * Decided in Round 15: memory is the *only* carried context — there is no per-folder
 * instructions file. The agent writes notes to itself across sessions, and the user can read
 * and edit them directly, so nothing accumulates invisibly.
 */

import crypto from 'node:crypto'
import { all, get, run } from '../storage/db'

export interface MemoryEntry {
  id: string
  text: string
  source: string | null
  createdAt: number
  updatedAt: number
}

interface Row {
  id: string
  text: string
  source: string | null
  created_at: number
  updated_at: number
}

const toEntry = (r: Row): MemoryEntry => ({
  id: r.id,
  text: r.text,
  source: r.source,
  createdAt: r.created_at,
  updatedAt: r.updated_at
})

/** Cap what gets injected into the system prompt, newest first. */
const MAX_INJECTED = 40

export function readMemory(): MemoryEntry[] {
  return all<Row>('SELECT * FROM agent_memory ORDER BY updated_at DESC LIMIT ?', MAX_INJECTED).map(toEntry)
}

export function allMemory(): MemoryEntry[] {
  return all<Row>('SELECT * FROM agent_memory ORDER BY updated_at DESC').map(toEntry)
}

export function addMemory(text: string, source = 'agent'): MemoryEntry {
  const trimmed = text.trim()
  if (!trimmed) throw new Error('Memory text cannot be empty')

  // Avoid piling up near-identical notes: an exact repeat just refreshes the timestamp.
  const existing = get<Row>('SELECT * FROM agent_memory WHERE text = ?', trimmed)
  if (existing) {
    run('UPDATE agent_memory SET updated_at = ? WHERE id = ?', Date.now(), existing.id)
    return toEntry({ ...existing, updated_at: Date.now() })
  }

  const entry: MemoryEntry = {
    id: crypto.randomBytes(6).toString('hex'),
    text: trimmed,
    source,
    createdAt: Date.now(),
    updatedAt: Date.now()
  }
  run(
    'INSERT INTO agent_memory (id, text, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
    entry.id,
    entry.text,
    entry.source,
    entry.createdAt,
    entry.updatedAt
  )
  return entry
}

export function updateMemory(id: string, text: string): void {
  run('UPDATE agent_memory SET text = ?, updated_at = ? WHERE id = ?', text.trim(), Date.now(), id)
}

export function deleteMemory(id: string): void {
  run('DELETE FROM agent_memory WHERE id = ?', id)
}

export function searchMemory(query: string): MemoryEntry[] {
  return all<Row>('SELECT * FROM agent_memory WHERE text LIKE ? ORDER BY updated_at DESC', `%${query}%`).map(toEntry)
}
