#!/usr/bin/env node
/**
 * Builds the extract-once portable exe.
 *
 * Why this exists rather than electron-builder's `portable` target: that target re-extracts the
 * whole payload to TEMP on every launch and deletes it on exit. Measured with this app's 3 GB
 * payload, that is ~24 seconds of unpacking *every single time* the app is opened.
 * `portable.unpackDirName` only stabilises the directory name; it does not make the extraction
 * skippable.
 *
 * This wraps the same `win-unpacked` output in an NSIS launcher that unpacks once to a versioned
 * directory under LOCALAPPDATA and writes a completion marker, so later launches start straight
 * away. It reuses the NSIS that electron-builder already downloads, so there is nothing extra to
 * install.
 *
 *   npm run pack:dir && node scripts/make-portable.mjs
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const RELEASE = path.join(ROOT, 'release')
const UNPACKED = path.join(RELEASE, 'win-unpacked')
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'))

const VERSION = pkg.version
const PRODUCT = pkg.build?.productName ?? 'LLM Manager'
const OUTFILE = path.join(RELEASE, `${PRODUCT.replace(/\s+/g, '-')}-${VERSION}-portable.exe`)

/**
 * The payload ships as two archives.
 *
 * 82% of it is vendored binaries — llama.cpp, Chromium, ffmpeg — that change only when
 * `fetch-vendor` runs. The part that changes on a normal build is a 10 MB asar. Compressing the
 * two together meant every rebuild spent minutes on 1.56 GB of bytes that were already
 * identical to last time.
 */
const APP_ARCHIVE = path.join(RELEASE, 'app.7z')
const VENDOR_ARCHIVE = path.join(RELEASE, 'vendor.7z')
const VENDOR_STAMP = `${VENDOR_ARCHIVE}.id`

/**
 * Compression level for the app portion.
 *
 * Measured on this payload: at `-mx=9` the app portion takes **102s** and produces 0.08 GB; at
 * `-mx=1` it takes **0.9s** and produces 0.11 GB. A hundred seconds to save thirty megabytes is
 * a bad trade on a build you run repeatedly, so fast is the default and `--max` is there for a
 * release where the 3% matters more than the wait.
 */
const APP_LEVEL = process.argv.includes('--store')
  ? { args: ['-mx=0'], label: 'stored' }
  : process.argv.includes('--max')
    ? { args: ['-m0=lzma2', '-mx=9'], label: 'LZMA2 -mx=9' }
    : { args: ['-m0=lzma2', '-mx=1'], label: 'LZMA2 -mx=1' }

/** The vendor tree is compressed once and cached, so it always gets the best level. */
const VENDOR_LEVEL = ['-m0=lzma2', '-mx=9']

/**
 * A cheap key for "has the vendor tree changed".
 *
 * Content-hashing 1.56 GB to answer a question that only changes when `fetch-vendor` rewrites
 * the files would be most of the time this split exists to save. Path, size and mtime are
 * sufficient for a directory nothing else writes to.
 */
async function vendorKey(dir) {
  const hash = crypto.createHash('sha256')
  const walk = async (cur) => {
    for (const e of (await fsp.readdir(cur, { withFileTypes: true })).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const full = path.join(cur, e.name)
      if (e.isDirectory()) await walk(full)
      else {
        const st = await fsp.stat(full)
        hash.update(`${path.relative(dir, full)}:${st.size}:${st.mtimeMs}`)
      }
    }
  }
  await walk(dir)
  return hash.digest('hex').slice(0, 12)
}

/** Report how long each stage took, because the whole point here is the total. */
async function stage(label, fn) {
  const started = Date.now()
  const note = (await fn()) ?? ''
  const seconds = (Date.now() - started) / 1000
  console.log(`  ${label.padEnd(30)} ${`${seconds.toFixed(1)}s`.padStart(7)}  ${note}`)
  return seconds
}

/**
 * The bundled 7-Zip.
 *
 * Already present as an electron-builder dependency, so this adds nothing to install. It is used
 * instead of NSIS's own compressor because that one is single-threaded: measured on this payload
 * it took ~900s on one core of twenty-four, where 7-Zip's LZMA2 took 131s across all of them and
 * produced an archive 8 MB *smaller*. There is no tradeoff here — it is faster and better.
 */
function find7za() {
  const candidates = [
    path.join(ROOT, 'node_modules', '7zip-bin', 'win', 'x64', '7za.exe'),
    path.join(ROOT, 'node_modules', '7zip-bin', 'win', 'ia32', '7za.exe'),
    'C:\\Program Files\\7-Zip\\7z.exe'
  ]
  return candidates.find((c) => fs.existsSync(c)) ?? null
}

/** electron-builder caches NSIS; fall back to a system install or NSIS_HOME. */
function findMakensis() {
  const cacheRoot = path.join(process.env.LOCALAPPDATA ?? '', 'electron-builder', 'Cache', 'nsis')
  const candidates = [process.env.NSIS_HOME && path.join(process.env.NSIS_HOME, 'makensis.exe')]

  if (fs.existsSync(cacheRoot)) {
    for (const dir of fs.readdirSync(cacheRoot)) {
      candidates.push(path.join(cacheRoot, dir, 'Bin', 'makensis.exe'))
      candidates.push(path.join(cacheRoot, dir, 'makensis.exe'))
    }
  }
  candidates.push('C:\\Program Files (x86)\\NSIS\\makensis.exe', 'C:\\Program Files\\NSIS\\makensis.exe')

  return candidates.filter(Boolean).find((c) => fs.existsSync(c)) ?? null
}

function run(file, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { stdio: ['ignore', 'pipe', 'pipe'] })
    let out = ''
    child.stdout?.on('data', (d) => {
      out += d
      const line = d.toString().trim().split('\n').pop()
      if (/^(Output|Processing|Install|Total)/i.test(line)) process.stdout.write(`  ${line}\n`)
    })
    child.stderr?.on('data', (d) => {
      out += d
    })
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve(out) : reject(new Error(out.slice(-2000) || `makensis exited ${code}`))))
  })
}

/**
 * A content fingerprint of the payload.
 *
 * The completion marker used to be named after the app version, so rebuilding the same version —
 * every iteration during development, and any hotfix that does not bump the number — left a
 * marker the launcher accepted, and it ran the *previous* payload. Silently. Naming the marker
 * after the bytes it describes means a changed payload cannot match an old marker, and an
 * unchanged one still skips the unpack.
 */
async function fingerprint(dir) {
  const hash = crypto.createHash('sha256')
  const files = []

  const walk = async (current) => {
    const entries = await fsp.readdir(current, { withFileTypes: true })
    // Sorted, so the hash does not depend on directory-order quirks between machines.
    for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const full = path.join(current, entry.name)
      if (entry.isDirectory()) await walk(full)
      else files.push(full)
    }
  }
  await walk(dir)

  for (const file of files) {
    hash.update(path.relative(dir, file).replace(/\\/g, '/'))
    hash.update(await fsp.readFile(file))
  }
  return hash.digest('hex').slice(0, 12)
}

async function dirSize(dir) {
  let total = 0
  for (const entry of await fsp.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    total += entry.isDirectory() ? await dirSize(full) : (await fsp.stat(full)).size
  }
  return total
}

async function main() {
  if (!fs.existsSync(UNPACKED)) {
    console.error(`No build at ${UNPACKED}.\nRun \`npm run pack:dir\` first.`)
    process.exit(1)
  }

  const sevenZip = find7za()
  if (!sevenZip) {
    console.error(
      'Could not find 7za.exe.\n' +
        'It normally ships with electron-builder — run `npm install`, or install 7-Zip from\n' +
        'https://www.7-zip.org and place it on PATH.'
    )
    process.exit(1)
  }

  const makensis = findMakensis()
  if (!makensis) {
    console.error(
      'Could not find makensis.exe.\n' +
        'It is normally downloaded by electron-builder — run `npm run pack:installer` once, or\n' +
        'install NSIS from https://nsis.sourceforge.io and set NSIS_HOME.'
    )
    process.exit(1)
  }

  const vendor = path.join(UNPACKED, 'resources', 'vendor')
  if (!fs.existsSync(vendor)) {
    console.warn('\n! WARNING: resources/vendor is missing — the exe will ship without a runtime.')
    console.warn('  Run `npm run fetch-vendor` before packaging.\n')
  }

  const payloadBytes = await dirSize(UNPACKED)
  process.stdout.write('\nFingerprinting payload… ')
  const buildId = await fingerprint(UNPACKED)
  console.log(buildId)

  /*
   * A rebuild of an unchanged payload has nothing to do.
   *
   * The fingerprint already identifies the bytes exactly, so an exe built from the same ones is
   * the same exe. Worth checking because `npm run pack:portable` is easy to run twice, and the
   * second run would otherwise spend a quarter of an hour producing an identical file.
   */
  const stamp = `${OUTFILE}.buildid`
  const previous = await fsp.readFile(stamp, 'utf8').catch(() => '')
  if (previous.trim() === `${buildId} ${APP_LEVEL.label}` && fs.existsSync(OUTFILE)) {
    const existing = (await fsp.stat(OUTFILE)).size
    console.log(`\nPayload unchanged (${buildId}) — keeping the existing exe.`)
    console.log(`  ${OUTFILE}`)
    console.log(`  ${(existing / 1024 ** 3).toFixed(2)} GB`)
    console.log('\nDelete it, or change the payload, to force a rebuild.')
    return
  }

  console.log(`\nPayload:  ${(payloadBytes / 1024 ** 3).toFixed(2)} GB  (${UNPACKED})`)
  console.log(`NSIS:     ${makensis}`)
  console.log(`7-Zip:    ${sevenZip}`)
  console.log(`Output:   ${OUTFILE}\n`)

  await fsp.rm(OUTFILE, { force: true })

  const threads = os.cpus().length
  const total = Date.now()
  console.log(`Stages (${threads} threads):`)

  // ---- 1. the vendor tree, compressed once and cached
  const key = await vendorKey(vendor)
  const cached = (await fsp.readFile(VENDOR_STAMP, 'utf8').catch(() => '')).trim()

  if (cached === key && fs.existsSync(VENDOR_ARCHIVE)) {
    const bytes = (await fsp.stat(VENDOR_ARCHIVE)).size
    console.log(`  ${'vendor.7z (cached)'.padEnd(30)} ${'0.0s'.padStart(7)}  ${(bytes / 1024 ** 3).toFixed(2)} GB`)
  } else {
    await stage('vendor.7z (one-off)', async () => {
      await fsp.rm(VENDOR_ARCHIVE, { force: true })
      await run(sevenZip, ['a', '-t7z', ...VENDOR_LEVEL, `-mmt=${threads}`, '-bso0', '-bsp0', VENDOR_ARCHIVE, path.join(vendor, '*')])
      await fsp.writeFile(VENDOR_STAMP, key)
      return `${((await fsp.stat(VENDOR_ARCHIVE)).size / 1024 ** 3).toFixed(2)} GB`
    })
  }

  // ---- 2. everything else, which is what actually changed
  await stage(`app.7z (${APP_LEVEL.label})`, async () => {
    await fsp.rm(APP_ARCHIVE, { force: true })
    await run(sevenZip, [
      'a',
      '-t7z',
      ...APP_LEVEL.args,
      `-mmt=${threads}`,
      '-bso0',
      '-bsp0',
      // The vendor tree is carried by its own archive.
      '-xr!vendor',
      APP_ARCHIVE,
      path.join(UNPACKED, '*')
    ])
    return `${((await fsp.stat(APP_ARCHIVE)).size / 1024 ** 2).toFixed(0)} MB`
  })

  // ---- 3. the launcher, which only has to carry them
  await stage('launcher', () =>
    run(makensis, [
      '/V2',
      `/DVERSION=${VERSION}`,
      `/DAPP_ARCHIVE=${APP_ARCHIVE}`,
      `/DVENDOR_ARCHIVE=${VENDOR_ARCHIVE}`,
      `/DEXTRACTOR=${sevenZip}`,
      `/DAPPEXE=${PRODUCT}.exe`,
      `/DBUILDID=${buildId}`,
      `/DOUTFILE=${OUTFILE}`,
      path.join(ROOT, 'build', 'portable.nsi')
    ])
  )

  // app.7z is rebuilt every time; vendor.7z is the cache and stays.
  await fsp.rm(APP_ARCHIVE, { force: true })
  console.log(`  ${'total'.padEnd(30)} ${`${((Date.now() - total) / 1000).toFixed(1)}s`.padStart(7)}`)

  // Recorded only after a successful build, so an interrupted one is never mistaken for done.
  await fsp.writeFile(stamp, `${buildId} ${APP_LEVEL.label}`)
  const size = (await fsp.stat(OUTFILE)).size
  console.log(`\nDone: ${OUTFILE}`)
  console.log(`  ${(size / 1024 ** 3).toFixed(2)} GB  (${((size / payloadBytes) * 100).toFixed(0)}% of payload)`)
  console.log(`\n  build id   ${buildId}`)
  console.log(`  first run  unpacks to %LOCALAPPDATA%\\LLMManager\\runtime-${VERSION}`)
  console.log('  later runs find the completion marker and start immediately')
  console.log('  older runtime-* directories are removed after a successful upgrade')
  console.log('\nUnsigned, so SmartScreen warns on first run.')
}

main().catch((err) => {
  console.error(err.message ?? err)
  process.exit(1)
})
