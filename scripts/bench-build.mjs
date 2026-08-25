/**
 * Time each stage of the portable build.
 *
 * The target is a sub-80s rebuild, and the only way to get there deliberately is to know which
 * stage is actually spending the time. Measures against the real tree; compresses to throwaway
 * files that are deleted as it goes.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const RELEASE = path.join(ROOT, 'release')
const UNPACKED = path.join(RELEASE, 'win-unpacked')
const VENDOR = path.join(UNPACKED, 'resources', 'vendor')
const threads = os.cpus().length

const sevenZip = [
  path.join(ROOT, 'node_modules', '7zip-bin', 'win', 'x64', '7za.exe'),
  'C:\\Program Files\\7-Zip\\7z.exe'
].find((c) => fs.existsSync(c))

function run(file, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { stdio: ['ignore', 'ignore', 'pipe'], shell: opts.shell ?? false, cwd: ROOT })
    let err = ''
    child.stderr?.on('data', (d) => (err += d))
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(err.slice(-600) || `exit ${code}`))))
  })
}

async function dirSize(dir) {
  let total = 0
  for (const e of await fsp.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name)
    total += e.isDirectory() ? await dirSize(full) : (await fsp.stat(full)).size
  }
  return total
}

async function hashTree(dir) {
  const hash = crypto.createHash('sha256')
  const files = []
  const walk = async (cur) => {
    for (const e of (await fsp.readdir(cur, { withFileTypes: true })).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const full = path.join(cur, e.name)
      if (e.isDirectory()) await walk(full)
      else files.push(full)
    }
  }
  await walk(dir)
  for (const f of files) {
    hash.update(path.relative(dir, f))
    hash.update(await fsp.readFile(f))
  }
  return hash.digest('hex').slice(0, 12)
}

async function time(label, fn) {
  const started = Date.now()
  let note = ''
  try {
    note = (await fn()) ?? ''
  } catch (err) {
    note = `FAILED: ${String(err.message).split('\n')[0].slice(0, 60)}`
  }
  const seconds = (Date.now() - started) / 1000
  console.log(`${label.padEnd(34)} ${`${seconds.toFixed(1)}s`.padStart(8)}   ${note}`)
  return seconds
}

const gb = (n) => `${(n / 1024 ** 3).toFixed(2)} GB`

console.log(`\nThreads: ${threads}`)
console.log(`Payload: ${gb(await dirSize(UNPACKED))}   vendor ${gb(await dirSize(VENDOR))}\n`)
console.log('stage                                  time   note')
console.log('---------------------------------------------------------------------')

await time('electron-vite build', () => run('npm.cmd', ['run', 'build']))
await time('electron-builder --dir', () => run('npx.cmd', ['electron-builder', '--win', '--dir']))
await time('fingerprint whole payload', async () => await hashTree(UNPACKED))
await time('fingerprint app only (no vendor)', async () => {
  // What it would cost if the vendor hash were cached.
  const hash = crypto.createHash('sha256')
  const walk = async (cur) => {
    for (const e of (await fsp.readdir(cur, { withFileTypes: true })).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const full = path.join(cur, e.name)
      if (full === VENDOR) continue
      if (e.isDirectory()) await walk(full)
      else hash.update(await fsp.readFile(full))
    }
  }
  await walk(UNPACKED)
  return hash.digest('hex').slice(0, 12)
})

const out = (n) => path.join(RELEASE, `bench-${n}.7z`)

await time('7z -mx=9 whole payload', async () => {
  await fsp.rm(out('all'), { force: true })
  await run(sevenZip, ['a', '-t7z', '-m0=lzma2', '-mx=9', `-mmt=${threads}`, '-bso0', '-bsp0', out('all'), path.join(UNPACKED, '*')])
  const s = (await fsp.stat(out('all'))).size
  await fsp.rm(out('all'), { force: true })
  return gb(s)
})

await time('7z -mx=9 vendor only (one-off)', async () => {
  await fsp.rm(out('vendor'), { force: true })
  await run(sevenZip, ['a', '-t7z', '-m0=lzma2', '-mx=9', `-mmt=${threads}`, '-bso0', '-bsp0', out('vendor'), path.join(VENDOR, '*')])
  const s = (await fsp.stat(out('vendor'))).size
  await fsp.rm(out('vendor'), { force: true })
  return gb(s)
})

await time('7z -mx=9 app only (per rebuild)', async () => {
  await fsp.rm(out('app'), { force: true })
  // Everything except the vendor tree — what a rebuild would actually have to compress.
  await run(sevenZip, [
    'a', '-t7z', '-m0=lzma2', '-mx=9', `-mmt=${threads}`, '-bso0', '-bsp0',
    '-xr!vendor', out('app'), path.join(UNPACKED, '*')
  ])
  const s = (await fsp.stat(out('app'))).size
  await fsp.rm(out('app'), { force: true })
  return gb(s)
})

await time('7z -mx=1 app only', async () => {
  await fsp.rm(out('appfast'), { force: true })
  await run(sevenZip, [
    'a', '-t7z', '-m0=lzma2', '-mx=1', `-mmt=${threads}`, '-bso0', '-bsp0',
    '-xr!vendor', out('appfast'), path.join(UNPACKED, '*')
  ])
  const s = (await fsp.stat(out('appfast'))).size
  await fsp.rm(out('appfast'), { force: true })
  return gb(s)
})

console.log()
