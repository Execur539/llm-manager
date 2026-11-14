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
  const seen = { saved: [], restored: [], restoredBytes: null, plainLeftAfterSave: null, completions: [] }
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
        // Like a real template: each message wrapped in its role, so a conversation's text begins with any shorter one's.
        // Pictures become the pinned marker, as llama-server renders them.
        const text = (content) =>
          typeof content === 'string' ? content : content.map((part) => (part.type === 'text' ? part.text : sc.MEDIA_MARKER)).join('')
        return reply({ prompt: '<s>' + json.messages.map((m) => `<${m.role}>${text(m.content)}</${m.role}>`).join('') })
      }
      if (req.url === '/completion') {
        seen.completions.push(json.prompt)
        return reply({ content: '' })
      }
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

  const current = await sc.currentState(common)
  check('a state saved of the conversation as it is counts as up to date', current?.tokens === 123, JSON.stringify(current))
  const grown = { ...common, req: { messages: [...req.messages, { role: 'user', content: 'and then?' }, { role: 'assistant', content: 'more' }] } }
  check('once more has been said it does not', (await sc.currentState(grown)) === null)
  check('nor under another load of the model', (await sc.currentState({ ...common, fingerprint: 'other' })) === null)

  const restored = await sc.restoreState({ ...common, seal: key })
  check('it restores with the key', restored.ok, JSON.stringify(restored))
  check('from a file opened just for the load', seen.restored.at(-1) === `${stem}.open.bin` && seen.restoredBytes === STATE)
  check('which is gone again once loaded', !fs.existsSync(path.join(states, `${stem}.open.bin`)))

  const noKey = await sc.restoreState({ ...common })
  check('an encrypted state is never handed over without its key', !noKey.ok && !seen.restored.slice(1).includes(`${stem}.bin`), JSON.stringify(noKey))
  check('and is removed rather than tried again', !fs.existsSync(path.join(states, `${stem}.bin`)))

  console.log('a conversation with a picture in it')
  const png = (seed) => `data:image/png;base64,${Buffer.from(`picture-${seed}`).toString('base64')}`
  const withPicture = (seed) => ({
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'what is this?' }, { type: 'image_url', image_url: { url: png(seed) } }] },
      { role: 'assistant', content: 'a cat' }
    ]
  })
  const pictured = { base, dir: states, key: 'chat:pictured', fingerprint: 'fp' }
  seen.completions.length = 0
  const savedPicture = await sc.saveState({ ...pictured, req: withPicture('one') })
  check('it saves', savedPicture.ok, JSON.stringify(savedPicture))
  const sync = seen.completions.at(-1)
  check(
    'with the picture sent along when the server is brought to the save point, not its marker as words',
    !!sync && typeof sync === 'object' && sync.multimodal_data?.[0] === Buffer.from('picture-one').toString('base64') && sync.prompt_string.includes(sc.MEDIA_MARKER),
    JSON.stringify(sync).slice(0, 120)
  )
  check('the same conversation counts as saved', (await sc.currentState({ ...pictured, req: withPicture('one') }))?.tokens === 123)
  check('the same words with a different picture do not', (await sc.currentState({ ...pictured, req: withPicture('two') })) === null)
  const pictureBack = await sc.restoreState({ ...pictured, req: withPicture('one') })
  check('and it loads back', pictureBack.ok, JSON.stringify(pictureBack))
  const otherPicture = await sc.restoreState({ ...pictured, req: withPicture('two') })
  check('but not for a conversation whose picture is different', !otherPicture.ok, JSON.stringify(otherPicture))
  const linked = await sc.saveState({
    ...pictured,
    key: 'chat:linked',
    req: { messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'https://example.com/a.png' } }] }, { role: 'assistant', content: 'ok' }] }
  })
  check('a picture given as a web address is refused, with the reason', !linked.ok && /web address/.test(linked.why), JSON.stringify(linked))

  console.log('borrowing another conversation\'s save')
  const lenderReq = { messages: [{ role: 'system', content: 'You are helpful.' }, { role: 'user', content: 'hello' }, { role: 'assistant', content: 'hi' }] }
  const lent = await sc.saveState({ base, dir: states, key: 'chat:lender', fingerprint: 'fp', req: lenderReq })
  const fork = { messages: [...lenderReq.messages, { role: 'user', content: 'and a fork of it' }, { role: 'assistant', content: 'yes' }] }
  const forkPrefix = await sc.renderPrefix(base, fork)
  const found = sc.longestReusable(states, 'fp', forkPrefix, fork.messages, (s) => s.key !== 'chat:fork')
  check('a fork finds the save of the conversation it came from', lent.ok && found?.key === 'chat:lender', JSON.stringify(found?.key))
  check('but not one saved under another load', sc.longestReusable(states, 'other', forkPrefix, fork.messages, () => true) === null)
  const unrelated = await sc.renderPrefix(base, { messages: [{ role: 'user', content: 'something else entirely' }, { role: 'assistant', content: 'ok' }] })
  check('nor one it does not begin with', sc.longestReusable(states, 'fp', unrelated, [], () => true) === null)
  const borrowed = await sc.restoreState({ base, dir: states, key: 'chat:lender', fingerprint: 'fp', req: fork, borrowed: true })
  check('the borrowed save loads for the fork', borrowed.ok, JSON.stringify(borrowed))
  const mismatch = await sc.restoreState({ base, dir: states, key: 'chat:lender', fingerprint: 'fp', req: { messages: [{ role: 'user', content: 'different' }] }, borrowed: true })
  check('and a borrowed save that does not fit is left alone, not removed', !mismatch.ok && !!sc.readMeta(states, 'chat:lender'), JSON.stringify(mismatch))

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
