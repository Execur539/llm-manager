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
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const RELEASE = path.join(ROOT, 'release')
const UNPACKED = path.join(RELEASE, 'win-unpacked')
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'))

const VERSION = pkg.version
const PRODUCT = pkg.build?.productName ?? 'LLM Manager'
const OUTFILE = path.join(RELEASE, `${PRODUCT.replace(/\s+/g, '-')}-${VERSION}-portable.exe`)

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
  console.log(`\nPayload:  ${(payloadBytes / 1024 ** 3).toFixed(2)} GB  (${UNPACKED})`)
  console.log(`NSIS:     ${makensis}`)
  console.log(`Output:   ${OUTFILE}\n`)
  console.log('Compressing (LZMA solid — this takes a while)…')

  await fsp.rm(OUTFILE, { force: true })

  await run(makensis, [
    '/V2',
    `/DVERSION=${VERSION}`,
    `/DPAYLOAD=${UNPACKED}`,
    `/DAPPEXE=${PRODUCT}.exe`,
    `/DOUTFILE=${OUTFILE}`,
    path.join(ROOT, 'build', 'portable.nsi')
  ])

  const size = (await fsp.stat(OUTFILE)).size
  console.log(`\nDone: ${OUTFILE}`)
  console.log(`  ${(size / 1024 ** 3).toFixed(2)} GB  (${((size / payloadBytes) * 100).toFixed(0)}% of payload)`)
  console.log(`\n  first run  unpacks to %LOCALAPPDATA%\\LLMManager\\runtime-${VERSION}`)
  console.log('  later runs find the completion marker and start immediately')
  console.log('  older runtime-* directories are removed after a successful upgrade')
  console.log('\nUnsigned, so SmartScreen warns on first run.')
}

main().catch((err) => {
  console.error(err.message ?? err)
  process.exit(1)
})
