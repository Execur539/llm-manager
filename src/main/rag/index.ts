/**
 * RAG: document extraction, chunking, embedding and retrieval.
 *
 * Two shapes, both decided in Round 10: quick per-chat file drops for one-offs, and named
 * reusable collections attachable to any chat.
 *
 * Embeddings come from a second llama-server running the small bundled embedding model, kept
 * resident alongside the chat model. That is the one exception to one-model-at-a-time — it is
 * ~100 MB and reloading it per query would make RAG unusable.
 *
 * Vector search is brute-force cosine over Float32 blobs. For a local library measured in
 * thousands of chunks that is microseconds, and it avoids a native vector extension.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { spawn, ChildProcess } from 'node:child_process'
import net from 'node:net'
import { all, get, run, transaction } from '../storage/db'
import { childEnv, embeddingModelPath, llamaServerPath } from '../runtime/binaries'
import type { Backend } from '@shared/types'

export interface DocumentRecord {
  id: string
  collectionId: string | null
  chatId: string | null
  filename: string
  path: string
  mime: string | null
  bytes: number
  chunks: number
}

export interface RetrievedChunk {
  documentId: string
  filename: string
  ord: number
  text: string
  score: number
}

// ---------------------------------------------------------------- embedding server

/**
 * A dedicated llama-server for the embedding model. Separate from the chat runtime so a
 * chat-model hot-swap does not tear down RAG mid-query.
 */
class EmbeddingService {
  private child: ChildProcess | null = null
  private port: number | null = null
  private starting: Promise<number> | null = null
  private modelPath: string | null = null

  async ensure(backend: Backend, overridePath?: string): Promise<number> {
    const wanted = overridePath ?? embeddingModelPath()
    if (this.port && this.child && this.modelPath === wanted) return this.port
    if (this.starting) return this.starting

    this.starting = (async () => {
      await this.stop()
      if (!fs.existsSync(wanted)) {
        throw new Error(
          `No embedding model at ${wanted}. Run \`npm run fetch-vendor\` or set a different embedding model in Settings.`
        )
      }

      const port = await freePort()
      const exe = llamaServerPath(backend)
      const child = spawn(
        exe,
        [
          '--model', wanted,
          '--host', '127.0.0.1',
          '--port', String(port),
          '--embeddings',
          // Embedding models are tiny; keep them off the GPU so they never compete with
          // the chat model for VRAM.
          '--n-gpu-layers', '0',
          '--ctx-size', '8192'
        ],
        { windowsHide: true, env: childEnv() }
      )
      this.child = child
      child.on('exit', () => {
        this.child = null
        this.port = null
      })

      const deadline = Date.now() + 120000
      while (Date.now() < deadline) {
        if (child.exitCode !== null) throw new Error(`Embedding server exited (${child.exitCode})`)
        try {
          const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1500) })
          if (res.ok) break
        } catch {
          /* not up yet */
        }
        await new Promise((r) => setTimeout(r, 400))
      }

      this.port = port
      this.modelPath = wanted
      return port
    })()

    try {
      return await this.starting
    } finally {
      this.starting = null
    }
  }

  async embed(texts: string[], backend: Backend, overridePath?: string): Promise<number[][]> {
    const port = await this.ensure(backend, overridePath)
    const out: number[][] = []
    // Batch so a large document doesn't produce one enormous request.
    const BATCH = 32
    for (let i = 0; i < texts.length; i += BATCH) {
      const slice = texts.slice(i, i + BATCH)
      const res = await fetch(`http://127.0.0.1:${port}/v1/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: slice })
      })
      if (!res.ok) throw new Error(`Embedding request failed: HTTP ${res.status}`)
      const json = (await res.json()) as { data: { embedding: number[] }[] }
      out.push(...json.data.map((d) => d.embedding))
    }
    return out
  }

  async stop(): Promise<void> {
    const child = this.child
    this.child = null
    this.port = null
    if (!child) return
    await new Promise<void>((resolve) => {
      child.once('exit', () => resolve())
      child.kill()
      setTimeout(resolve, 3000)
    })
  }
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer()
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      if (typeof addr === 'object' && addr) {
        const p = addr.port
        srv.close(() => resolve(p))
      } else srv.close(() => reject(new Error('no port')))
    })
    srv.on('error', reject)
  })
}

export const embeddings = new EmbeddingService()

// ---------------------------------------------------------------- extraction

/**
 * Extract text from a document.
 * PDF handling is deliberately simple: pull text objects out of the raw stream. It copes with
 * ordinary text PDFs and gives up honestly on scanned ones rather than returning noise.
 */
export async function extractText(file: string): Promise<string> {
  const ext = path.extname(file).toLowerCase()

  if (ext === '.pdf') return extractPdf(file)

  if (ext === '.docx' || ext === '.xlsx' || ext === '.pptx') {
    throw new Error(
      `${ext} extraction is not implemented. Convert to text or PDF first, or ask the agent to run a converter.`
    )
  }

  const buf = await fsp.readFile(file)
  // Reject binaries rather than embedding garbage.
  const sample = buf.subarray(0, 4096)
  const nullBytes = sample.filter((b) => b === 0).length
  if (nullBytes > sample.length * 0.01) {
    throw new Error(`${path.basename(file)} looks like a binary file, not text.`)
  }
  return buf.toString('utf8')
}

async function extractPdf(file: string): Promise<string> {
  const buf = await fsp.readFile(file)
  const raw = buf.toString('latin1')

  const pieces: string[] = []
  // Text-showing operators: (string) Tj and [(a) -250 (b)] TJ
  const tjRe = /\((?:\\.|[^\\()])*\)\s*Tj/g
  const tjArrayRe = /\[((?:\s*\((?:\\.|[^\\()])*\)\s*-?[\d.]*)+)\]\s*TJ/g

  let m: RegExpExecArray | null
  while ((m = tjArrayRe.exec(raw)) !== null) {
    const inner = m[1].match(/\((?:\\.|[^\\()])*\)/g) ?? []
    pieces.push(inner.map(unescapePdfString).join(''))
  }
  while ((m = tjRe.exec(raw)) !== null) {
    const s = m[0].match(/\((?:\\.|[^\\()])*\)/)?.[0]
    if (s) pieces.push(unescapePdfString(s))
  }

  const text = pieces.join(' ').replace(/\s+/g, ' ').trim()
  if (text.length < 40) {
    throw new Error(
      `Could not extract text from ${path.basename(file)}. It may be a scanned PDF, which needs OCR.`
    )
  }
  return text
}

function unescapePdfString(s: string): string {
  return s
    .slice(1, -1)
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\r')
    .replace(/\\t/g, '\t')
    .replace(/\\([()\\])/g, '$1')
    .replace(/\\(\d{1,3})/g, (_, oct) => String.fromCharCode(parseInt(oct, 8)))
}

// ---------------------------------------------------------------- chunking

const CHUNK_CHARS = 1600
const OVERLAP_CHARS = 200

/**
 * Split on paragraph boundaries where possible, with overlap so a fact spanning a boundary
 * is still retrievable from at least one chunk.
 */
export function chunkText(text: string): string[] {
  const normalised = text.replace(/\r\n/g, '\n').trim()
  if (normalised.length <= CHUNK_CHARS) return normalised ? [normalised] : []

  const paragraphs = normalised.split(/\n\s*\n/)
  const chunks: string[] = []
  let current = ''

  const push = (): void => {
    const trimmed = current.trim()
    if (trimmed) chunks.push(trimmed)
    current = ''
  }

  for (const para of paragraphs) {
    if (para.length > CHUNK_CHARS) {
      push()
      // A single huge paragraph gets a sliding window.
      for (let i = 0; i < para.length; i += CHUNK_CHARS - OVERLAP_CHARS) {
        chunks.push(para.slice(i, i + CHUNK_CHARS).trim())
      }
      continue
    }
    if (current.length + para.length + 2 > CHUNK_CHARS) {
      const tail = current.slice(-OVERLAP_CHARS)
      push()
      current = `${tail}\n\n${para}`
    } else {
      current = current ? `${current}\n\n${para}` : para
    }
  }
  push()
  return chunks.filter(Boolean)
}

// ---------------------------------------------------------------- storage

function toBlob(vector: number[]): Buffer {
  const f32 = new Float32Array(vector)
  return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength)
}

function fromBlob(blob: Buffer | Uint8Array, dim: number): Float32Array {
  const buf = Buffer.isBuffer(blob) ? blob : Buffer.from(blob)
  return new Float32Array(buf.buffer, buf.byteOffset, dim)
}

export function createCollection(name: string): { id: string; name: string } {
  const id = crypto.randomBytes(6).toString('hex')
  run('INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)', id, name, Date.now())
  return { id, name }
}

export function listCollections(): { id: string; name: string; documents: number }[] {
  return all<{ id: string; name: string; documents: number }>(
    `SELECT c.id, c.name, COUNT(d.id) AS documents
     FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
     GROUP BY c.id ORDER BY c.name`
  )
}

export function deleteCollection(id: string): void {
  run('DELETE FROM collections WHERE id = ?', id)
}

/** Ingest one file: extract, chunk, embed, store. */
export async function ingestDocument(
  file: string,
  backend: Backend,
  target: { collectionId?: string; chatId?: string },
  embeddingModelOverride?: string,
  onProgress?: (done: number, total: number) => void
): Promise<DocumentRecord> {
  const st = await fsp.stat(file)
  const text = await extractText(file)
  const chunks = chunkText(text)
  if (!chunks.length) throw new Error(`No usable text found in ${path.basename(file)}`)

  const vectors = await embeddings.embed(chunks, backend, embeddingModelOverride)
  if (vectors.length !== chunks.length) {
    throw new Error(`Embedding count mismatch: ${vectors.length} vectors for ${chunks.length} chunks`)
  }
  onProgress?.(chunks.length, chunks.length)

  const docId = crypto.randomBytes(8).toString('hex')
  const dim = vectors[0]?.length ?? 0

  transaction(() => {
    run(
      'INSERT INTO documents (id, collection_id, chat_id, filename, path, mime, bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      docId,
      target.collectionId ?? null,
      target.chatId ?? null,
      path.basename(file),
      file,
      null,
      st.size,
      Date.now()
    )
    chunks.forEach((chunkText, i) => {
      const chunkId = `${docId}-${i}`
      run('INSERT INTO chunks (id, document_id, ord, text, tokens) VALUES (?, ?, ?, ?, ?)', chunkId, docId, i, chunkText, Math.ceil(chunkText.length / 4))
      run('INSERT INTO embeddings (chunk_id, dim, vector, embed_model) VALUES (?, ?, ?, ?)', chunkId, dim, toBlob(vectors[i]), 'bundled')
    })
  })

  return {
    id: docId,
    collectionId: target.collectionId ?? null,
    chatId: target.chatId ?? null,
    filename: path.basename(file),
    path: file,
    mime: null,
    bytes: st.size,
    chunks: chunks.length
  }
}

export function listDocuments(filter: { collectionId?: string; chatId?: string }): DocumentRecord[] {
  const rows = filter.collectionId
    ? all<{ id: string; collection_id: string | null; chat_id: string | null; filename: string; path: string; mime: string | null; bytes: number }>(
        'SELECT * FROM documents WHERE collection_id = ? ORDER BY created_at DESC',
        filter.collectionId
      )
    : filter.chatId
      ? all<{ id: string; collection_id: string | null; chat_id: string | null; filename: string; path: string; mime: string | null; bytes: number }>(
          'SELECT * FROM documents WHERE chat_id = ? ORDER BY created_at DESC',
          filter.chatId
        )
      : all<{ id: string; collection_id: string | null; chat_id: string | null; filename: string; path: string; mime: string | null; bytes: number }>(
          'SELECT * FROM documents ORDER BY created_at DESC'
        )

  return rows.map((r) => ({
    id: r.id,
    collectionId: r.collection_id,
    chatId: r.chat_id,
    filename: r.filename,
    path: r.path,
    mime: r.mime,
    bytes: r.bytes,
    chunks: get<{ n: number }>('SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?', r.id)?.n ?? 0
  }))
}

export function deleteDocument(id: string): void {
  run('DELETE FROM documents WHERE id = ?', id)
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0
  let na = 0
  let nb = 0
  const len = Math.min(a.length, b.length)
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

/** Retrieve the top-k chunks for a query, scoped to a collection and/or chat. */
export async function retrieve(
  query: string,
  backend: Backend,
  scope: { collectionId?: string; chatId?: string },
  k = 6,
  embeddingModelOverride?: string
): Promise<RetrievedChunk[]> {
  const [queryVector] = await embeddings.embed([query], backend, embeddingModelOverride)
  if (!queryVector) return []

  const where: string[] = []
  const params: unknown[] = []
  if (scope.collectionId) {
    where.push('d.collection_id = ?')
    params.push(scope.collectionId)
  }
  if (scope.chatId) {
    where.push('d.chat_id = ?')
    params.push(scope.chatId)
  }
  const clause = where.length ? `WHERE ${where.join(' OR ')}` : ''

  const rows = all<{
    document_id: string
    filename: string
    ord: number
    text: string
    dim: number
    vector: Uint8Array
  }>(
    `SELECT c.document_id, d.filename, c.ord, c.text, e.dim, e.vector
     FROM chunks c
     JOIN embeddings e ON e.chunk_id = c.id
     JOIN documents d ON d.id = c.document_id
     ${clause}`,
    ...params
  )

  const q = new Float32Array(queryVector)
  return rows
    .map((r) => ({
      documentId: r.document_id,
      filename: r.filename,
      ord: r.ord,
      text: r.text,
      score: cosine(q, fromBlob(r.vector, r.dim))
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
}

/** Format retrieved context for injection, with sources the model can cite. */
export function formatContext(chunks: RetrievedChunk[]): string {
  if (!chunks.length) return ''
  return [
    'Relevant excerpts from the attached documents. Cite the source filename when you use one.',
    '',
    ...chunks.map((c, i) => `[${i + 1}] ${c.filename} (chunk ${c.ord}, similarity ${c.score.toFixed(3)})\n${c.text}`)
  ].join('\n\n')
}
