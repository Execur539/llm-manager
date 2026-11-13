/**
 * Saving and loading a conversation's state against a real llama-server, on the CPU.
 *
 * Covers what only a real server can show: that a save lands beside the last good one and replaces
 * it only once whole, that what was saved loads back with the same number of tokens, that a
 * conversation which has grown is saved again over the old state, and that a state file cut short
 * is refused rather than loaded. Uses a small hybrid model (recurrent layers, the case where a
 * state cannot be stepped back) if one is in the library, and skips otherwise.
 *
 *   node scripts/state-save-check.mjs [path\to\small-model.gguf]
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import net from 'node:net'
import { spawn } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const model = process.argv[2] ?? 'D:\\Apps\\LLM-Manager\\LLMManagerModels\\unsloth__Qwen3.5-2B-GGUF\\Qwen3.5-2B-Q4_K_M.gguf'
const server = path.join(ROOT, 'vendor', 'llama.cpp', 'cpu', 'llama-server.exe')
if (!fs.existsSync(model) || !fs.existsSync(server)) {
  console.log('Skipped: needs a small GGUF model and vendor/llama.cpp/cpu.')
  process.exit(0)
}

let pass = 0
let fail = 0
const check = (name, cond, extra = '') => {
  if (cond) {
    pass++
    console.log(`  PASS  ${name}`)
  } else {
    fail++
    console.log(`  FAIL  ${name} ${extra}`)
  }
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-state-real-'))
const states = path.join(work, 'states')
fs.mkdirSync(states)
await build({
  entryPoints: { sc: path.join(ROOT, 'src/main/runtime/state-cache.ts') },
  outdir: work,
  bundle: true,
  platform: 'node',
  format: 'esm',
  logLevel: 'error',
  alias: { '@shared': path.join(ROOT, 'src/shared') }
})
const sc = await import(pathToFileURL(path.join(work, 'sc.js')).href)

const port = await new Promise((resolve) => {
  const s = net.createServer().listen(0, '127.0.0.1', () => {
    const p = s.address().port
    s.close(() => resolve(p))
  })
})
const base = `http://127.0.0.1:${port}`
const proc = spawn(
  server,
  ['-m', model, '-ngl', '0', '-c', '4096', '-t', '4', '--parallel', '1', '--host', '127.0.0.1', '--port', String(port), '--jinja', '--slot-save-path', states, '--ctx-checkpoints', '64'],
  { windowsHide: true, stdio: 'ignore' }
)

try {
  const deadline = Date.now() + 120_000
  let up = false
  while (Date.now() < deadline && !up) {
    up = await fetch(`${base}/health`).then((r) => r.ok).catch(() => false)
    if (!up) await new Promise((r) => setTimeout(r, 500))
  }
  check('the server starts', up)
  if (!up) throw new Error('server did not start')

  const key = 'chat:real-test'
  const fingerprint = 'test-fingerprint'
  const conversation = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Name three colours of the rainbow, and say a sentence about each of them.' },
    { role: 'assistant', content: 'Red is the colour of ripe tomatoes. Green is the colour of spring grass. Blue is the colour of a clear sky.' }
  ]
  const stem = sc.fileStem(key)
  const bin = path.join(states, `${stem}.bin`)

  console.log('\nA first save')
  const first = await sc.saveState({ base, dir: states, key, fingerprint, req: { messages: conversation } })
  check('succeeds', first.ok, first.ok ? '' : first.why)
  check('the state is in place', fs.existsSync(bin))
  check('nothing half-written is left beside it', !fs.existsSync(path.join(states, `${stem}.next.bin`)))
  check('its record names the file\'s exact size', first.ok && first.meta.fileBytes === fs.statSync(bin).size)

  console.log('\nLoading it back')
  const restored = await sc.restoreState({ base, dir: states, key, fingerprint, req: { messages: conversation } })
  check('loads with every token that was saved', restored.ok, restored.ok ? '' : restored.why)

  console.log('\nThe conversation grows and is saved again')
  const longer = [
    ...conversation,
    { role: 'user', content: 'Now do the same for three more colours.' },
    { role: 'assistant', content: 'Yellow is the colour of sunflowers. Orange is the colour of autumn leaves. Violet is the colour of lavender.' }
  ]
  const before = fs.statSync(bin).size
  const second = await sc.saveState({ base, dir: states, key, fingerprint, req: { messages: longer } })
  check('succeeds', second.ok, second.ok ? '' : second.why)
  check('replacing the earlier state with a larger one', fs.statSync(bin).size > before && second.ok && second.meta.tokens > first.meta.tokens)
  const again = await sc.restoreState({ base, dir: states, key, fingerprint, req: { messages: longer } })
  check('which loads back too', again.ok, again.ok ? '' : again.why)

  console.log('\nA state file cut short')
  const full = fs.readFileSync(bin)
  fs.writeFileSync(bin, full.subarray(0, Math.floor(full.length / 2)))
  check('is not taken for the saved one', sc.readMeta(states, key) === null)
  const refused = await sc.restoreState({ base, dir: states, key, fingerprint, req: { messages: longer } })
  check('and is never loaded', !refused.ok)
} catch (err) {
  fail++
  console.log(`  FAIL  ${err.message}`)
} finally {
  proc.kill()
  await new Promise((r) => setTimeout(r, 800))
  fs.rmSync(work, { recursive: true, force: true })
}

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
