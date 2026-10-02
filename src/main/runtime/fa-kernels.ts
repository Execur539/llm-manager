/**
 * Which key/value cache pairings a CUDA build of llama.cpp can attend over directly.
 *
 * ggml-cuda compiles its flash-attention vector kernels for a fixed list of K/V type pairs
 * (`GGML_CUDA_FA_QUANTS`; matched pairs only, unless the build asked for more). A cache outside
 * that list still works, but every attention call first converts the whole cache to f16 — on
 * every generated token — which costs the speed and the VRAM the smaller cache was meant to save.
 *
 * The list is compiled into ggml-cuda.dll as its `FA_QUANTS` build feature, so it is read straight
 * from the file: no process to start, nothing to ask of the GPU.
 */

import fs from 'node:fs'

/** What a CUDA build compiles when nobody changed the list: each type paired with itself. */
export const MATCHED_KV_PAIRS: readonly string[] = ['f16-f16', 'q8_0-q8_0', 'q4_0-q4_0']

const KEY = Buffer.from('FA_QUANTS', 'latin1')
/** Longest value worth reading; every pair ggml can compile fits in far less. */
const MAX_VALUE = 1024
/** NUL padding a compiler may put between the key string and its value. */
const MAX_PADDING = 16
const CHUNK = 8 * 1024 * 1024

/** The pairs named in a feature value such as `q4_0-q4_0,q8_0-q8_0`; null when it is not one. */
export function parseFaQuants(value: string): string[] | null {
  const pairs = value
    .split(',')
    .map((p) => p.trim())
    .filter(Boolean)
  if (!pairs.length || !pairs.every((p) => /^[a-z0-9_]+-[a-z0-9_]+$/.test(p))) return null
  return pairs
}

/**
 * The pairs from the `FA_QUANTS` feature in a buffer, or null.
 *
 * The key has to end in a NUL: the same letters also appear inside llama.cpp's warning text
 * ("Add ... to GGML_CUDA_FA_QUANTS to compile it"), followed by a space.
 */
export function findFaQuants(buf: Buffer): string[] | null {
  for (let at = buf.indexOf(KEY); at !== -1; at = buf.indexOf(KEY, at + 1)) {
    let p = at + KEY.length
    if (buf[p] !== 0) continue
    const paddingEnd = Math.min(buf.length, p + MAX_PADDING)
    while (p < paddingEnd && buf[p] === 0) p++
    const end = buf.indexOf(0, p)
    if (end === -1 || end - p > MAX_VALUE) continue
    const pairs = parseFaQuants(buf.toString('latin1', p, end))
    if (pairs) return pairs
  }
  return null
}

/**
 * The pairs from a file, read a chunk at a time so a 140 MB DLL is never held in memory whole.
 *
 * Each chunk keeps the tail of the one before it, so a feature that straddles a boundary is still
 * seen whole. Null when the file is missing or carries no such feature.
 */
export async function readFaQuants(file: string): Promise<string[] | null> {
  let handle: fs.promises.FileHandle
  try {
    handle = await fs.promises.open(file, 'r')
  } catch {
    return null
  }
  try {
    const overlap = KEY.length + MAX_PADDING + MAX_VALUE + 1
    const buf = Buffer.alloc(CHUNK + overlap)
    let position = 0
    let carry = 0
    for (;;) {
      const { bytesRead } = await handle.read(buf, carry, CHUNK, position)
      if (bytesRead === 0) return null
      position += bytesRead
      const filled = carry + bytesRead
      const found = findFaQuants(buf.subarray(0, filled))
      if (found) return found
      carry = Math.min(overlap, filled)
      buf.copy(buf, 0, filled - carry, filled)
    }
  } finally {
    await handle.close()
  }
}

const known = new Map<string, { size: number; mtimeMs: number; pairs: Promise<string[] | null> }>()

/** The pairs a ggml-cuda.dll was built with, read once for each version of the file. */
export async function cudaKvPairs(dll: string): Promise<string[] | null> {
  let stat: fs.Stats
  try {
    stat = await fs.promises.stat(dll)
  } catch {
    return null
  }
  const hit = known.get(dll)
  if (hit && hit.size === stat.size && hit.mtimeMs === stat.mtimeMs) return hit.pairs
  const pairs = readFaQuants(dll)
  known.set(dll, { size: stat.size, mtimeMs: stat.mtimeMs, pairs })
  return pairs
}
