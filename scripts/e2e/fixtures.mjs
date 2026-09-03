/**
 * Test fixtures: an isolated app environment and synthetic models.
 *
 * Every run gets its own userData directory, so tests never touch the real settings, chat
 * history or model library — and never inherit state from a previous run.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

// fileURLToPath, not manual pathname munging: the project path contains a space, which arrives
// percent-encoded in import.meta.url and produced a "LLM%20Manager" path that does not exist.
export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

/** Build a valid GGUF file with the given architecture facts. */
export function buildGguf(opts = {}) {
  const {
    architecture = 'qwen35',
    name = 'Test Model 27B',
    blockCount = 64,
    embeddingLength = 5120,
    headCount = 24,
    headCountKv = 4,
    keyLength = 256,
    contextLength = 262144,
    vocabSize = 32000,
    fullAttentionInterval = 4,
    ssm = true,
    tensorCount = 4,
    chatTemplateMentionsTools = true
  } = opts

  const chunks = []
  const u32 = (n) => {
    const b = Buffer.alloc(4)
    b.writeUInt32LE(n)
    return b
  }
  const u64 = (n) => {
    const b = Buffer.alloc(8)
    b.writeBigUInt64LE(BigInt(n))
    return b
  }
  const f32 = (n) => {
    const b = Buffer.alloc(4)
    b.writeFloatLE(n)
    return b
  }
  const str = (s) => {
    const body = Buffer.from(s, 'utf8')
    return Buffer.concat([u64(body.length), body])
  }

  const kv = []
  const addStr = (k, v) => kv.push(Buffer.concat([str(k), u32(8), str(v)]))
  const addU32 = (k, v) => kv.push(Buffer.concat([str(k), u32(4), u32(v)]))
  const addF32 = (k, v) => kv.push(Buffer.concat([str(k), u32(6), f32(v)]))

  addStr('general.architecture', architecture)
  addStr('general.name', name)
  addStr('general.license', 'apache-2.0')
  addU32(`${architecture}.block_count`, blockCount)
  addU32(`${architecture}.embedding_length`, embeddingLength)
  addU32(`${architecture}.attention.head_count`, headCount)
  addU32(`${architecture}.attention.head_count_kv`, headCountKv)
  addU32(`${architecture}.attention.key_length`, keyLength)
  addU32(`${architecture}.context_length`, contextLength)
  addF32(`${architecture}.attention.layer_norm_rms_epsilon`, 1e-6)

  if (ssm) {
    addU32(`${architecture}.full_attention_interval`, fullAttentionInterval)
    addU32(`${architecture}.ssm.conv_kernel`, 4)
    addU32(`${architecture}.ssm.state_size`, 128)
    addU32(`${architecture}.ssm.inner_size`, 6144)
    addU32(`${architecture}.ssm.group_count`, 16)
  }

  const REASONING_TEMPLATES = {
    // Levels enumerated by a validation tuple, with 'high' aliased onto 'xhigh'.
    effort:
      "{%- set r = reasoning_effort|default('xhigh') %}" +
      "{%- if r == 'high' %}{%- set r = 'xhigh' %}{%- endif %}" +
      "{%- if r not in ('xhigh', 'medium', 'low') %}{{- raise_exception('bad effort') }}{%- endif %}" +
      '{%- if enable_thinking is undefined or enable_thinking is true %}{{- "<think>" }}{%- endif %}',
    // A plain on/off switch and nothing more.
    toggle: '{%- if enable_thinking is defined and enable_thinking is false %}{{- "<think></think>" }}{%- endif %}',
    none: ''
  }

  const reasoningTemplate = REASONING_TEMPLATES[opts.reasoning ?? 'none'] ?? ''

  addStr(
    'tokenizer.chat_template',
    (chatTemplateMentionsTools
      ? '{%- if tools %}{{- "You may call tools." }}{%- endif %}{% for m in messages %}{{ m.content }}{% endfor %}'
      : '{% for m in messages %}{{ m.content }}{% endfor %}') + reasoningTemplate
  )
  addStr('tokenizer.ggml.model', 'gpt2')

  // A large token array, elided by the parser — exercises the skip path.
  const tokenEntries = []
  for (let i = 0; i < vocabSize; i++) tokenEntries.push(str(`t${i}`))
  kv.push(Buffer.concat([str('tokenizer.ggml.tokens'), u32(9), u32(8), u64(vocabSize), ...tokenEntries]))

  chunks.push(u32(0x46554747), u32(3), u64(tensorCount), u64(kv.length), ...kv)

  // Tensor directory: a couple of block tensors plus embeddings and output.
  const tensors = [
    ['blk.0.attn_q.weight', [embeddingLength, embeddingLength], 12],
    ['blk.1.attn_q.weight', [embeddingLength, embeddingLength], 12],
    ['token_embd.weight', [embeddingLength, vocabSize], 12],
    ['output.weight', [embeddingLength, vocabSize], 14]
  ].slice(0, tensorCount)

  let offset = 0
  for (const [tname, dims, type] of tensors) {
    chunks.push(str(tname), u32(dims.length), ...dims.map((d) => u64(d)), u32(type), u64(offset))
    offset += 4096
  }

  return Buffer.concat(chunks)
}

/** Create an isolated environment: userData, models dir, and the models it should contain. */
export async function createEnv(label = 'e2e') {
  const base = await fsp.mkdtemp(path.join(os.tmpdir(), `llmm-${label}-`))
  const userData = path.join(base, 'userData')
  const appData = path.join(base, 'appData')
  // Must match what defaultModelsDir() computes from exeDir(), or the app opens a modal
  // relocation dialog on launch and blocks the main process before any window appears.
  const modelsDir = path.join(base, 'LLMManagerModels')

  await fsp.mkdir(userData, { recursive: true })
  await fsp.mkdir(path.join(appData, 'LLMManager'), { recursive: true })
  await fsp.mkdir(modelsDir, { recursive: true })

  // Point the breadcrumb at our models dir so no relocation dialog appears on launch.
  await fsp.writeFile(
    path.join(appData, 'LLMManager', 'models-path.json'),
    JSON.stringify({ modelsDir, exeDir: base, updatedAt: Date.now() }, null, 2)
  )

  /*
   * Its own API port, rather than the shipped default of 1234.
   *
   * Every sandbox shared that default, so the scenarios that start a server could not run beside
   * each other — and, more awkwardly, could not run at all while the developer had their own copy
   * of the app open, because the real app was already holding the port. That produced three
   * failures describing a server that would not start, none of which were about the code.
   *
   * Port 0 asks the OS for a free one, which is what everything else here already does.
   */
  const apiPort = await freePort()
  await fsp.writeFile(
    path.join(appData, 'LLMManager', 'settings.json'),
    JSON.stringify({ server: { port: apiPort } }, null, 2)
  )

  return { base, userData, appData, modelsDir, apiPort }
}

/** Ask the OS for a port nothing is using, and let it go again. */
async function freePort() {
  const net = await import('node:net')
  return new Promise((resolve, reject) => {
    const srv = net.createServer()
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address()
      srv.close(() => resolve(port))
    })
    srv.on('error', reject)
  })
}

export async function addModel(env, filename, opts = {}) {
  const dir = path.join(env.modelsDir, opts.repoDir ?? 'test__models')
  await fsp.mkdir(dir, { recursive: true })
  const file = path.join(dir, filename)
  await fsp.writeFile(file, buildGguf(opts))

  if (opts.withMmproj) {
    // A projector is recognised by name; its contents only need to parse as GGUF.
    await fsp.writeFile(
      path.join(dir, `mmproj-${filename}`),
      buildGguf({ architecture: 'clip', name: 'projector', tensorCount: 1, vocabSize: 16, ssm: false })
    )
  }
  return file
}

export async function cleanupEnv(env) {
  try {
    await fsp.rm(env.base, { recursive: true, force: true, maxRetries: 3 })
  } catch {
    /* Windows sometimes holds a handle briefly; a leftover temp dir is harmless */
  }
}

/** Start the HuggingFace stand-in and return its base URL. */
export async function startMockHf() {
  const { spawn } = await import('node:child_process')
  const child = spawn(process.execPath, [path.join(ROOT, 'scripts', 'e2e', 'mock-hf.mjs'), '--port', '0'], {
    stdio: ['ignore', 'pipe', 'inherit']
  })
  const port = await new Promise((resolve, reject) => {
    let buf = ''
    child.stdout.on('data', (d) => {
      buf += d
      const first = buf.trim().split(/\r?\n/)[0]
      if (/^\d+$/.test(first)) resolve(Number(first))
    })
    child.on('error', reject)
    setTimeout(() => reject(new Error('mock-hf did not start')), 10000)
  })
  return {
    base: `http://127.0.0.1:${port}`,
    async control(params) {
      const qs = new URLSearchParams(params).toString()
      await fetch(`http://127.0.0.1:${port}/__control?${qs}`)
    },
    stop() {
      child.kill()
    }
  }
}

/** Write a plain-text document for RAG ingestion. */
export async function addDocument(env, name, content) {
  const dir = path.join(env.base, 'docs')
  await fsp.mkdir(dir, { recursive: true })
  const file = path.join(dir, name)
  await fsp.writeFile(file, content, 'utf8')
  return file
}

/** Place a stand-in embedding model where the app expects to find one. */
export async function addEmbeddingModel(env) {
  const dir = path.join(ROOT, 'vendor', 'models')
  await fsp.mkdir(dir, { recursive: true })
  const file = path.join(dir, 'embedding.gguf')
  if (!fs.existsSync(file)) {
    await fsp.writeFile(file, buildGguf({ architecture: 'nomic-bert', name: 'embed', ssm: false, vocabSize: 512 }))
  }
  return file
}

export function envVars(env, extra = {}) {
  return {
    ...process.env,
    // Electron's appData path ignores APPDATA, so the app needs an explicit override.
    LLMM_APPDATA_DIR: path.join(env.appData, 'LLMManager'),
    LLMM_MOCK_LLAMA: '1',
    LLMM_MOCK_SCRIPT: path.join(ROOT, 'scripts', 'mock-llama.mjs'),
    LLMM_MOCK_READY_MS: '400',
    LLMM_VENDOR_DIR: path.join(ROOT, 'vendor'),
    // Makes exeDir() resolve to the sandbox, so models live beside it exactly as they would
    // beside a real portable exe.
    LLMM_PORTABLE_DIR: env.base,
    LLMM_E2E: '1',
    ...extra
  }
}
