/**
 * Measure the compression tradeoff for the portable payload.
 *
 * "Is there a faster way" deserves numbers rather than an opinion, and the answer turned out to
 * be worth measuring: NSIS's own LZMA is single-threaded, and dropping its dictionary from 64 MB
 * to 8 MB only took it from ~900s to 793s while making the output *larger*. The constraint was
 * never the dictionary — it was using one core of twenty-four.
 *
 * This benchmarks the 7-Zip levels the build actually offers, so `--fast` can be chosen on
 * evidence rather than feel.
 *
 *   node scripts/bench-compression.mjs [level ...]
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const RELEASE = path.join(ROOT, 'release')
const UNPACKED = path.join(RELEASE, 'win-unpacked')

const LEVELS = {
  store: { args: ['-mx=0'], label: 'stored' },
  fastest: { args: ['-m0=lzma2', '-mx=1'], label: 'LZMA2 -mx=1  (--fast)' },
  normal: { args: ['-m0=lzma2', '-mx=5'], label: 'LZMA2 -mx=5' },
  max: { args: ['-m0=lzma2', '-mx=9'], label: 'LZMA2 -mx=9  (default)' }
}

function find7za() {
  return [
    path.join(ROOT, 'node_modules', '7zip-bin', 'win', 'x64', '7za.exe'),
    'C:\\Program Files\\7-Zip\\7z.exe'
  ].find((c) => fs.existsSync(c))
}

function run(file, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { stdio: ['ignore', 'ignore', 'pipe'] })
    let err = ''
    child.stderr?.on('data', (d) => (err += d))
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(err.slice(-800) || `exit ${code}`))))
  })
}

async function dirSize(dir) {
  let total = 0
  for (const entry of await fsp.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    total += entry.isDirectory() ? await dirSize(full) : (await fsp.stat(full)).size
  }
  return total
}

const sevenZip = find7za()
if (!sevenZip || !fs.existsSync(UNPACKED)) {
  console.error(!sevenZip ? '7za.exe not found' : `No payload at ${UNPACKED}`)
  process.exit(1)
}

const threads = os.cpus().length
const payload = await dirSize(UNPACKED)
const wanted = process.argv.slice(2).filter((m) => m in LEVELS)
const levels = wanted.length ? wanted : Object.keys(LEVELS)

console.log(`\nPayload: ${(payload / 1024 ** 3).toFixed(2)} GB`)
console.log(`Threads: ${threads}\n`)
console.log('level     setting                     time      size      ratio')
console.log('----------------------------------------------------------------')

for (const level of levels) {
  const { args, label } = LEVELS[level]
  const out = path.join(RELEASE, `bench-${level}.7z`)
  await fsp.rm(out, { force: true })

  const started = Date.now()
  try {
    await run(sevenZip, ['a', '-t7z', ...args, `-mmt=${threads}`, '-bso0', '-bsp0', out, path.join(UNPACKED, '*')])
  } catch (err) {
    console.log(`${level.padEnd(9)} ${label.padEnd(27)} FAILED — ${String(err.message).split('\n')[0].slice(0, 40)}`)
    continue
  }

  const seconds = (Date.now() - started) / 1000
  const size = (await fsp.stat(out)).size
  console.log(
    `${level.padEnd(9)} ${label.padEnd(27)} ${`${seconds.toFixed(0)}s`.padStart(6)}  ` +
      `${`${(size / 1024 ** 3).toFixed(2)} GB`.padStart(8)}  ${((size / payload) * 100).toFixed(0)}%`
  )
  await fsp.rm(out, { force: true })
}

console.log('\nFor reference, NSIS built-in LZMA on the same payload:')
console.log('          lzma solid, 64 MB dict       ~900s   0.86 GB  45%   (single-threaded)')
console.log('          lzma solid,  8 MB dict        793s   0.87 GB  46%   (single-threaded)\n')
