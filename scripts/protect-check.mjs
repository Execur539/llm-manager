/**
 * Checks the encryption protected conversations are stored with, in plain Node.
 *
 *   node scripts/protect-check.mjs
 *
 * The cipher (text, blobs, files of several chunks, damage detection, passwords and recovery
 * keys) and encrypted saved model states, against a stand-in for llama-server's slot endpoints
 * that records exactly which files it was asked to write and read.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import crypto from 'node:crypto'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
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
const throws = async (fn) => {
  try {
    await fn()
    return false
  } catch {
    return true
  }
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-protect-'))
await build({
  entryPoints: {
    cipher: path.join(ROOT, 'src/main/protect/cipher.ts'),
    sc: path.join(ROOT, 'src/main/runtime/state-cache.ts')
  },
  outdir: work,
  bundle: true,
  platform: 'node',
  format: 'esm',
  logLevel: 'error',
  alias: { '@shared': path.join(ROOT, 'src/shared') }
})
const c = await import(pathToFileURL(path.join(work, 'cipher.js')).href)
const sc = await import(pathToFileURL(path.join(work, 'sc.js')).href)

try {
  console.log('text and blobs')
  const key = c.newKey()
  const other = c.newKey()
  const text = 'The launch code is 0000 — and émoji 🔒 too'
  const sealed = c.sealText(key, text)
  check('text comes back as it went in', c.openText(key, sealed) === text)
  check('and is not readable sealed', !sealed.includes('launch') && c.isSealedText(sealed))
  check('the same text seals differently each time', c.sealText(key, text) !== sealed)
  check('plain text passes through openText untouched', c.openText(key, 'hello') === 'hello')
  check('the wrong key does not open it', await throws(() => c.openText(other, sealed)))
  const tampered = sealed.slice(0, -4) + (sealed.endsWith('A') ? 'BBBB' : 'AAAA')
  check('an altered value does not open', await throws(() => c.openText(key, tampered)))
  const vec = new Float32Array([0.25, -1.5, 3, 42])
  const blob = c.sealBlob(key, new Uint8Array(vec.buffer))
  const back = c.openBlob(key, blob)
  check('an embedding blob round-trips', Buffer.from(back).equals(Buffer.from(vec.buffer)) && c.isSealedBlob(blob))

  console.log('files')
  for (const size of [0, 1, 1024 * 1024, 1024 * 1024 + 7, 3 * 1024 * 1024]) {
    const plain = crypto.randomBytes(size)
    const src = path.join(work, `p${size}`)
    const enc = path.join(work, `e${size}`)
    const out = path.join(work, `o${size}`)
    fs.writeFileSync(src, plain)
    await c.sealFile(key, src, enc)
    await c.openFile(key, enc, out)
    const viaBuffer = c.openBuffer(key, fs.readFileSync(enc))
    const bufferSealed = c.openBuffer(key, c.sealBuffer(key, plain))
    check(
      `${size} bytes: streamed, in memory and mixed all round-trip`,
      fs.readFileSync(out).equals(plain) && viaBuffer.equals(plain) && bufferSealed.equals(plain)
    )
    check(`${size} bytes: recognised as encrypted, the original not`, (await c.isSealedFile(enc)) && (size === 0 || !(await c.isSealedFile(src))))
  }
  const big = path.join(work, 'e3145728')
  const full = fs.readFileSync(big)
  fs.writeFileSync(path.join(work, 'cut'), full.subarray(0, full.length - (1024 * 1024 + 16)))
  check('a file cut short at a chunk boundary does not open', await throws(() => c.openBuffer(key, fs.readFileSync(path.join(work, 'cut')))))
  const swapped = Buffer.from(full)
  const a = 16, n = 1024 * 1024 + 16
  Buffer.from(full.subarray(a + n, a + 2 * n)).copy(swapped, a)
  Buffer.from(full.subarray(a, a + n)).copy(swapped, a + n)
  check('reordered chunks do not open', await throws(() => c.openBuffer(key, swapped)))
  const rekeyed = path.join(work, 'rekeyed')
  await c.resealFile(key, other, big, rekeyed)
  check(
    're-keying gives the same bytes under the other key only',
    c.openBuffer(other, fs.readFileSync(rekeyed)).equals(fs.readFileSync(path.join(work, 'p3145728'))) &&
      (await throws(() => c.openBuffer(key, fs.readFileSync(rekeyed))))
  )

  console.log('passwords and recovery keys')
  const t0 = Date.now()
  const wrap = await c.wrapKey('correct horse battery', key, c.PASSWORD_COST)
  const ms = Date.now() - t0
  check(`wrapping under a password takes a noticeable moment (${ms} ms)`, ms > 50 && ms < 5000)
  check('the password unwraps the key', (await c.unwrapKey('correct horse battery', wrap))?.equals(key) === true)
  check('a wrong password unwraps nothing', (await c.unwrapKey('correct horse batterY', wrap)) === null)
  const rk = c.newRecoveryKey()
  check('a recovery key is five groups of five', /^[0-9A-HJKMNP-TV-Z]{5}(-[0-9A-HJKMNP-TV-Z]{5}){4}$/.test(rk), rk)
  const rwrap = await c.wrapKey(rk, key, c.RECOVERY_COST)
  const typed = rk.toLowerCase().replace(/-/g, ' ').replace(/0/g, 'o').replace(/1/g, 'l')
  check('a recovery key typed loosely still reads as itself', c.normaliseRecoveryKey(typed) === rk, `${typed} -> ${c.normaliseRecoveryKey(typed)}`)
  check('and unwraps the key', (await c.unwrapKey(c.normaliseRecoveryKey(typed), rwrap))?.equals(key) === true)
  check('a password is not taken for a recovery key', c.normaliseRecoveryKey('correct horse battery') === null)

  console.log('saved model states')
  const states = path.join(work, 'states')
  fs.mkdirSync(states)
  const STATE = crypto.randomBytes(2 * 1024 * 1024 + 5).toString('base64').slice(0, 2 * 1024 * 1024 + 5) + ' the words of the conversation '
  const seen = { saved: [], restored: [], restoredBytes: null, plainLeftAfterSave: null }
  const server = http.createServer((req, res) => {
    let body = ''
    req.on('data', (d) => (body += d))
    req.on('end', () => {
      const json = body ? JSON.parse(body) : {}
      const reply = (o) => {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify(o))
      }
      if (req.url === '/apply-template') {
        const last = json.messages.at(-1).content
        return reply({ prompt: `<s>${json.messages.slice(0, -1).map((m) => m.content).join('|')}<u>${last}` })
      }
      if (req.url === '/completion') return reply({ content: '' })
      if (req.url.startsWith('/slots/0?action=save')) {
        seen.saved.push(json.filename)
        fs.writeFileSync(path.join(states, json.filename), STATE)
        return reply({ n_saved: 123, n_written: STATE.length })
      }
      if (req.url.startsWith('/slots/0?action=restore')) {
        seen.restored.push(json.filename)
        seen.restoredBytes = fs.readFileSync(path.join(states, json.filename), 'utf8')
        return reply({ n_restored: 123 })
      }
      res.writeHead(404)
      res.end('{}')
    })
  })
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  const base = `http://127.0.0.1:${server.address().port}`
  const req = { messages: [{ role: 'user', content: 'hello' }, { role: 'assistant', content: 'hi' }] }
  const common = { base, dir: states, key: 'chat:abc', fingerprint: 'fp', req }

  const saved = await sc.saveState({ ...common, seal: key })
  const stem = sc.fileStem('chat:abc')
  const onDisk = fs.readdirSync(states)
  check('a protected conversation\'s state saves', saved.ok, JSON.stringify(saved))
  check('and the file on disk is encrypted', await c.isSealedFile(path.join(states, `${stem}.bin`)))
  check('with no plain copy left beside it', !onDisk.some((f) => f.endsWith('.next.bin') || f.endsWith('.next.lk')), onDisk.join(','))
  check('nothing of the conversation readable in it', !fs.readFileSync(path.join(states, `${stem}.bin`)).includes('the words of the conversation'))

  const restored = await sc.restoreState({ ...common, seal: key })
  check('it restores with the key', restored.ok, JSON.stringify(restored))
  check('from a file opened just for the load', seen.restored.at(-1) === `${stem}.open.bin` && seen.restoredBytes === STATE)
  check('which is gone again once loaded', !fs.existsSync(path.join(states, `${stem}.open.bin`)))

  const noKey = await sc.restoreState({ ...common })
  check('an encrypted state is never handed over without its key', !noKey.ok && !seen.restored.slice(1).includes(`${stem}.bin`), JSON.stringify(noKey))
  check('and is removed rather than tried again', !fs.existsSync(path.join(states, `${stem}.bin`)))

  const plain = await sc.saveState({ ...common })
  check('an unprotected conversation still saves plain', plain.ok && !(await c.isSealedFile(path.join(states, `${stem}.bin`))))
  const wrongWay = await sc.restoreState({ ...common, seal: key })
  check('a plain state is not loaded for a protected conversation', !wrongWay.ok && !fs.existsSync(path.join(states, `${stem}.bin`)), JSON.stringify(wrongWay))
  server.close()
} finally {
  fs.rmSync(work, { recursive: true, force: true })
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail ? 1 : 0)
