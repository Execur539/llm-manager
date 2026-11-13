/**
 * Chat's scratchpad, run for real: the store, the loopback server ffmpeg reads it through, and the
 * tools a chat's model uses, against a real video made with the bundled ffmpeg.
 *
 * What it has to guarantee: what goes in comes back byte for byte, wherever it is held; nothing
 * is ever written to disk in memory mode; idle files go on time and say why they went; a folder
 * chosen in Settings only ever has the app's own subfolder touched; and the model can look at a
 * video held in memory, including one whose index sits at the end of the file.
 *
 *   node scripts/scratchpad-check.mjs
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'
import { Readable } from 'node:stream'
import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'
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

const work = await fsp.mkdtemp(path.join(os.tmpdir(), 'llmm-scratch-test-'))
process.env.LLMM_APPDATA_DIR = path.join(work, 'appdata')
process.env.LLMM_VENDOR_DIR = path.join(ROOT, 'vendor')

const out = path.join(work, 'built')
await build({
  entryPoints: {
    store: path.join(ROOT, 'src/main/scratchpad/store.ts'),
    serve: path.join(ROOT, 'src/main/scratchpad/serve.ts'),
    tools: path.join(ROOT, 'src/main/scratchpad/tools.ts')
  },
  outdir: out,
  bundle: true,
  platform: 'node',
  format: 'esm',
  logLevel: 'error',
  alias: { '@shared': path.join(ROOT, 'src/shared'), electron: path.join(ROOT, 'scripts/electron-stub.mjs') }
})
const url = (f) => `file://${path.join(out, f).replace(/\\/g, '/')}`
const { Scratchpad, ScratchpadError, safeName, OWNED_FOLDER, MAX_TEXT_BYTES } = await import(url('store.js'))
const { ScratchServer, parseRange } = await import(url('serve.js'))
const { runChatTool, CHAT_TOOLS } = await import(url('tools.js'))

const streamOf = (buf) => Readable.from([buf])
let clock = 1_000_000
const now = () => clock

// ---------------------------------------------------------------- names

console.log('\nNames are names, never paths')
check('a path is cut to its last part', safeName('C:\\Users\\x\\..\\evil.txt') === 'evil.txt')
check('so is one written with slashes', safeName('../../etc/passwd') === 'passwd')
check('characters Windows refuses are replaced', safeName('a<b>c:d?.txt') === 'a_b_c_d_.txt')
check('a reserved device name is made safe', safeName('con.txt') === '_con.txt')
check('an empty name still gets one', safeName('') === 'file')
check('leading dots go, so nothing hides', safeName('...hidden') === 'hidden')

// ---------------------------------------------------------------- memory

console.log('\nIn memory')
{
  const pad = new Scratchpad({ folder: () => null, memoryLimit: () => 1024 ** 3, now })
  const big = crypto.randomBytes(20 * 1024 * 1024 + 123)
  const item = await pad.add('chat-a', 'clip.mp4', streamOf(big), 'video', big.length)
  check('a file goes in under its name', item.name === 'clip.mp4' && item.bytes === big.length)
  check('it is found again whatever the case', pad.get('chat-a', 'CLIP.MP4')?.id === item.id)
  check('another conversation cannot see it', pad.get('chat-b', 'clip.mp4') === undefined)
  check('nothing about it is on disk', pad.pathOf(item.id) === null)

  const read = async (start, end) => {
    const chunks = []
    for await (const c of pad.readRange(item.id, start, end)) chunks.push(c)
    return Buffer.concat(chunks)
  }
  check('it comes back byte for byte', Buffer.compare(await read(0, big.length - 1), big) === 0)
  const mid = await read(5 * 1024 * 1024, 17 * 1024 * 1024)
  check('a range across blocks is exact', Buffer.compare(mid, big.subarray(5 * 1024 * 1024, 17 * 1024 * 1024 + 1)) === 0)
  check('a range past the end stops at the end', (await read(big.length - 10, big.length + 500)).length === 10)

  const twin = await pad.add('chat-a', 'clip.mp4', streamOf(Buffer.from('x')), 'video', 1)
  check('a second file of the same name gets its own', twin.name === 'clip (2).mp4')

  const tight = new Scratchpad({ folder: () => null, memoryLimit: () => 1000, now })
  let refused = null
  try {
    await tight.add('c', 'huge.mp4', streamOf(Buffer.alloc(5000)), 'video', 5000)
  } catch (err) {
    refused = err
  }
  check('a file too large for memory is refused with a reason', refused instanceof ScratchpadError && /folder/i.test(refused.message))
  check('and nothing of it is kept', tight.list('c').length === 0)
  let midway = null
  try {
    await tight.add('c', 'unknown.bin', Readable.from([Buffer.alloc(600), Buffer.alloc(600)]), 'file')
  } catch (err) {
    midway = err
  }
  check('a stream of unknown size is stopped once it passes the limit', midway instanceof ScratchpadError && tight.list('c').length === 0)
}

// ---------------------------------------------------------------- text, deletion, sweeping

console.log('\nNotes, deletion and the idle sweep')
{
  clock = 1_000_000
  const pad = new Scratchpad({ folder: () => null, memoryLimit: () => 1024 ** 3, now })
  await pad.writeText('c', 'notes.md', 'first line\n')
  await pad.writeText('c', 'notes.md', 'second line\n', true)
  const notes = pad.get('c', 'notes.md')
  check('notes are written and added to', (await pad.readText(notes)) === 'first line\nsecond line\n')
  check('writing again replaces', (await pad.readText(await pad.writeText('c', 'notes.md', 'fresh'))) === 'fresh')
  let tooLong = null
  try {
    await pad.writeText('c', 'big.md', 'x'.repeat(MAX_TEXT_BYTES + 1))
  } catch (err) {
    tooLong = err
  }
  check('a note too large is refused', tooLong instanceof ScratchpadError)

  await pad.add('c', 'video.mp4', streamOf(Buffer.alloc(10)), 'video', 10)
  clock += 20 * 60_000
  pad.touch(pad.get('c', 'video.mp4'))
  clock += 15 * 60_000
  const swept = await pad.sweep(30 * 60_000)
  check('a file unused past the limit is swept', swept.map((i) => i.name).join() === 'notes.md', swept.map((i) => i.name).join())
  check('one used since is kept', !!pad.get('c', 'video.mp4'))
  check('the swept one says why it went', /30 minutes without being used/.test(pad.whyGone('c', 'notes.md') ?? ''))

  await pad.remove('c', 'video.mp4', 'you deleted it when you had finished with it')
  check('a deleted file is gone', !pad.get('c', 'video.mp4'))
  check('and says why', /you deleted it/.test(pad.whyGone('c', 'video.mp4') ?? ''))
  await pad.add('c', 'video.mp4', streamOf(Buffer.alloc(3)), 'video', 3)
  check('the same name given again is a fresh file, with no note', !pad.whyGone('c', 'video.mp4'))
  await pad.clearChat('c')
  check('a conversation cleared holds nothing', pad.list('c').length === 0 && pad.totals().files === 0)
}

// ---------------------------------------------------------------- a folder chosen in Settings

console.log('\nIn a folder chosen in Settings')
{
  const chosen = path.join(work, 'My stuff')
  await fsp.mkdir(chosen, { recursive: true })
  await fsp.writeFile(path.join(chosen, 'my-own-file.txt'), 'the user\'s, not ours')
  const pad = new Scratchpad({ folder: () => chosen, memoryLimit: () => 1, now })
  const item = await pad.add('chat-1', 'clip.mp4', streamOf(Buffer.from('VIDEO BYTES')), 'video', 11)
  const where = pad.pathOf(item.id)
  check('files go in the app\'s own subfolder', where === path.join(chosen, OWNED_FOLDER, 'chat-1', 'clip.mp4'), where)
  check('the memory limit does not apply to a folder', item.bytes === 11)
  check('they are on disk there', fs.readFileSync(where, 'utf8') === 'VIDEO BYTES')
  await pad.remove('chat-1', 'clip.mp4', 'done')
  check('deleting one deletes the file', !fs.existsSync(where))
  await pad.add('chat-1', 'again.mp4', streamOf(Buffer.from('x')), 'video', 1)
  await pad.clearAll()
  check('closing empties the app\'s subfolder', !fs.existsSync(path.join(chosen, OWNED_FOLDER)))
  check('and leaves everything else in the folder alone', fs.readFileSync(path.join(chosen, 'my-own-file.txt'), 'utf8') === 'the user\'s, not ours')
  const weird = await new Scratchpad({ folder: () => chosen, memoryLimit: () => 1, now }).add('a/../..\\b', 'x.txt', streamOf(Buffer.from('y')), 'file', 1)
  const weirdPath = path.dirname(path.join(chosen, OWNED_FOLDER, 'x'))
  check('a conversation id cannot climb out of the folder', weird && fs.readdirSync(weirdPath).every((n) => /^[0-9a-f]{24}$|^chat-1$/.test(n)), fs.readdirSync(weirdPath).join(','))
}

// ---------------------------------------------------------------- the loopback server

console.log('\nThe loopback server')
check('a range is read', JSON.stringify(parseRange('bytes=10-19', 100)) === '{"start":10,"end":19}')
check('an open range runs to the end', JSON.stringify(parseRange('bytes=90-', 100)) === '{"start":90,"end":99}')
check('a suffix range is the last bytes', JSON.stringify(parseRange('bytes=-5', 100)) === '{"start":95,"end":99}')
check('a range past the end is refused', parseRange('bytes=200-300', 100) === null)
check('no range is the whole file', parseRange(undefined, 100) === 'all')
const memPad = new Scratchpad({ folder: () => null, memoryLimit: () => 1024 ** 3, now })
const server = new ScratchServer(memPad)
{
  const data = crypto.randomBytes(100_000)
  const item = await memPad.add('s', 'blob.bin', streamOf(data), 'file', data.length)
  const u = await server.urlFor(item.id)
  check('it listens on loopback only', u.startsWith('http://127.0.0.1:'))
  const whole = await fetch(u)
  check('the whole file is served', whole.status === 200 && Buffer.compare(Buffer.from(await whole.arrayBuffer()), data) === 0)
  check('and says it can be read in parts', whole.headers.get('accept-ranges') === 'bytes')
  const part = await fetch(u, { headers: { Range: 'bytes=1000-1999' } })
  check('a range comes back as a part', part.status === 206 && part.headers.get('content-range') === `bytes 1000-1999/${data.length}`)
  check('holding exactly those bytes', Buffer.compare(Buffer.from(await part.arrayBuffer()), data.subarray(1000, 2000)) === 0)
  const guess = u.replace(/\/[0-9a-f]{48}\//, `/${'0'.repeat(48)}/`)
  check('a URL without the token gets nothing', (await fetch(guess)).status === 404)
  check('an unsatisfiable range is refused', (await fetch(u, { headers: { Range: 'bytes=999999-' } })).status === 416)
  check('only reading is allowed', (await fetch(u, { method: 'POST' })).status === 405)
  await memPad.remove('s', 'blob.bin', '')
  check('a removed file is not served', (await fetch(u)).status === 404)
}

// ---------------------------------------------------------------- the tools, on a real video

console.log('\nThe model watches a video it holds in memory')
const ffmpeg = path.join(ROOT, 'vendor', 'ffmpeg', 'ffmpeg.exe')
if (!fs.existsSync(ffmpeg)) {
  console.log('  (skipped: vendor/ffmpeg is not present)')
} else {
  const video = path.join(work, 'test clip.mp4')
  // Written without faststart, so the index sits at the end and reading it needs a seek.
  await new Promise((resolve, reject) =>
    execFile(
      ffmpeg,
      ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=duration=6:size=320x240:rate=10', '-pix_fmt', 'yuv420p', '-y', video],
      { windowsHide: true },
      (err) => (err ? reject(err) : resolve())
    )
  )
  const pad = new Scratchpad({ folder: () => null, memoryLimit: () => 1024 ** 3, now: Date.now })
  // As in the app: the server reads from the same scratchpad the tools look in.
  const padServer = new ScratchServer(pad)
  await pad.addFile('watch', video, 'video')
  const ctx = { chatId: 'watch', pad, urlFor: (id) => padServer.urlFor(id), vision: true, signal: new AbortController().signal, idleMinutes: 30 }

  const overview = await runChatTool('watch_video', { name: 'test clip.mp4' }, ctx)
  check('an overview returns frames', typeof overview === 'object' && overview.media.filter((p) => p.type === 'image_url').length === 8, typeof overview === 'string' ? overview : '')
  check('each frame is a JPEG', typeof overview === 'object' && overview.media.filter((p) => p.type === 'image_url').every((p) => p.image_url.url.startsWith('data:image/jpeg;base64,/9j/')))
  check('the text says what it is and how to look closer', typeof overview === 'object' && /6:00|0:06/.test(overview.text) && /watch_video again with name "test clip.mp4"/.test(overview.text))
  const close = await runChatTool('watch_video', { name: 'test clip.mp4', start: 2, end: 3, frames: 4, detail: 'high' }, ctx)
  const times = typeof close === 'object' ? (/Times: (.*)\./.exec(close.text)?.[1] ?? '') : ''
  check('a closer look stays in the stretch asked for', times.split(', ').length === 4 && times.split(', ').every((t) => /^0:0[23]/.test(t)), times)
  check('nothing was written to the tool output folder', !fs.existsSync(path.join(process.env.LLMM_APPDATA_DIR, 'tool-output')) || fs.readdirSync(path.join(process.env.LLMM_APPDATA_DIR, 'tool-output')).length === 0)

  const blind = await runChatTool('watch_video', { name: 'test clip.mp4' }, { ...ctx, vision: false })
  check('a model with no vision is told why it cannot see', typeof blind === 'string' && /no vision projector/.test(blind))

  const listing = await runChatTool('scratchpad_list', {}, ctx)
  check('the list names the video and the idle rule', /test clip\.mp4: video/.test(listing) && /30 minutes/.test(listing), listing)
  check('notes can be written', /Wrote notes\.md/.test(await runChatTool('scratchpad_write', { name: 'notes.md', content: 'red square at 0:02' }, ctx)))
  check('and read back', (await runChatTool('scratchpad_read', { name: 'notes.md' }, ctx)) === 'red square at 0:02')
  check('a video is not read as text', /watch_video instead/.test(await runChatTool('scratchpad_read', { name: 'test clip.mp4' }, ctx)))
  check('a name not there lists what is', /It holds: test clip\.mp4, notes\.md/.test(await runChatTool('watch_video', { name: 'other.mp4' }, ctx)))
  check('deleting is the model\'s signal it is done', /Deleted test clip\.mp4/.test(await runChatTool('scratchpad_delete', { name: 'test clip.mp4' }, ctx)))
  check('after which the video is gone, and the model is told why', /no longer in your scratchpad: you deleted it/.test(await runChatTool('watch_video', { name: 'test clip.mp4' }, ctx)))
  check('a tool that does not exist says what does', /scratchpad_list/.test(await runChatTool('read_file', { path: 'C:\\Windows\\win.ini' }, ctx)))
  check('every tool is read-tier, so chat never stops to ask permission', CHAT_TOOLS.every((t) => t.tier === 'read'))
  check('no tool takes a path', CHAT_TOOLS.every((t) => !('path' in (t.parameters.properties ?? {}))))

  console.log('\nAnd one kept in a folder')
  const folderPad = new Scratchpad({ folder: () => path.join(work, 'folder-mode'), memoryLimit: () => 1, now: Date.now })
  await folderPad.addFile('f', video, 'video')
  const fromDisk = await runChatTool('watch_video', { name: 'test clip.mp4', frames: 3 }, { ...ctx, chatId: 'f', pad: folderPad })
  check('is watched from its file', typeof fromDisk === 'object' && fromDisk.media.filter((p) => p.type === 'image_url').length === 3)
  await folderPad.clearAll()
  padServer.stop()
}

server.stop()
await fsp.rm(work, { recursive: true, force: true }).catch(() => undefined)
console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
