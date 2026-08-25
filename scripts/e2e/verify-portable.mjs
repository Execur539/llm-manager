/**
 * End-to-end check of the portable exe itself.
 *
 * The packaged-app test covers what the app can see once it is running. This covers the launcher
 * around it: that the payload unpacks, the completion marker matches *this* build, the app
 * actually starts, a second launch skips the unpack, and — the regression that cost 18 GB of
 * relocated models once — that the app treats the folder holding the exe as its own, not the
 * extraction cache under LOCALAPPDATA.
 *
 * Run after `node scripts/make-portable.mjs`.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { spawn, execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'))
const VERSION = pkg.version
const EXE = path.join(ROOT, 'release', `LLM-Manager-${VERSION}-portable.exe`)
const RUNTIME = path.join(process.env.LOCALAPPDATA ?? '', 'LLMManager', `runtime-${VERSION}`)

/** A sample across every vendored component, plus the Python stdlib that a filter once ate. */
const EXPECTED_VENDOR = [
  'llama.cpp/cuda/llama-server.exe',
  'llama.cpp/vulkan/llama-server.exe',
  'llama.cpp/cpu/llama-server.exe',
  'chromium/chrome.exe',
  'ffmpeg/ffmpeg.exe',
  'ffmpeg/ffprobe.exe',
  'cloudflared/cloudflared.exe',
  'python/python.exe',
  'python/python312.zip',
  'rg/rg.exe',
  'models/embedding.gguf'
]

let failures = 0
const check = (label, ok, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function appRunning() {
  try {
    const out = execFileSync('tasklist', ['/FI', 'IMAGENAME eq LLM Manager.exe', '/NH'], { encoding: 'utf8' })
    return /LLM Manager\.exe/i.test(out)
  } catch {
    return false
  }
}

function killApp() {
  try {
    execFileSync('taskkill', ['/F', '/IM', 'LLM Manager.exe', '/T'], { stdio: 'ignore' })
  } catch {
    /* not running */
  }
}

/** Wait for a predicate, or give up. */
async function waitFor(label, predicate, timeoutMs) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    if (await predicate()) return Date.now() - started
    await sleep(500)
  }
  return null
}

if (!fs.existsSync(EXE)) {
  console.error(`No portable exe at ${EXE}`)
  process.exit(1)
}

// A throwaway folder standing in for wherever a user drops the exe. The app must treat this as
// its home — the models library belongs beside the exe, not in the extraction cache.
const drop = await fsp.mkdtemp(path.join(os.tmpdir(), 'llmm-drop-'))
const dropExe = path.join(drop, path.basename(EXE))
const appData = path.join(drop, 'appdata')
await fsp.mkdir(appData, { recursive: true })

console.log(`\nCopying the exe to a throwaway folder…\n  ${drop}`)
await fsp.copyFile(EXE, dropExe)

// Point app-data at the sandbox so this cannot touch real settings, and leave a breadcrumb so
// the relocation dialog — a modal that would block the run — never appears.
const portableModels = path.join(drop, 'LLMManagerModels')
await fsp.mkdir(portableModels, { recursive: true })
await fsp.writeFile(
  path.join(appData, 'models-path.json'),
  JSON.stringify({ modelsDir: portableModels, exeDir: drop, updatedAt: Date.now() }, null, 2)
)
const env = { ...process.env, LLMM_APPDATA_DIR: appData }

killApp()
await fsp.rm(RUNTIME, { recursive: true, force: true }).catch(() => undefined)

try {
  // ---- first launch: unpacks
  console.log('\nFirst launch (expect an unpack)…')
  const t0 = Date.now()
  spawn(dropExe, [], { env, detached: true, stdio: 'ignore' }).unref()

  const markerFound = await waitFor(
    'marker',
    async () => {
      const entries = await fsp.readdir(RUNTIME).catch(() => [])
      return entries.some((f) => f.startsWith('.unpacked-'))
    },
    600000
  )
  check('the payload unpacks and a marker is written', markerFound !== null,
    markerFound === null ? 'timed out after 10 minutes' : `${(markerFound / 1000).toFixed(0)}s`)

  const markers = (await fsp.readdir(RUNTIME).catch(() => [])).filter((f) => f.startsWith('.unpacked-'))
  check('exactly one marker exists', markers.length === 1, markers.join(', '))

  // The marker must name a build fingerprint, not just the version — that was the bug that let a
  // rebuild of the same version silently run the previous payload.
  const markerName = markers[0] ?? ''
  check('the marker is keyed to a build fingerprint, not the version',
    markerName !== `.unpacked-${VERSION}` && /^\.unpacked-[0-9a-f]{12}$/.test(markerName), markerName)

  const started = await waitFor('app', async () => appRunning(), 120000)
  check('the app starts after unpacking', started !== null,
    started === null ? 'no process appeared' : `${(started / 1000).toFixed(0)}s from launch`)
  console.log(`  first launch total: ${((Date.now() - t0) / 1000).toFixed(0)}s`)

  // Give it a moment to settle and write anything it means to write.
  await sleep(6000)

  // ---- the regression that relocated 18 GB
  const strayModels = path.join(RUNTIME, 'LLMManagerModels')
  check('no model library is created inside the extraction cache', !fs.existsSync(strayModels), strayModels)

  // The vendor tree arrives in its own archive, extracted to a path the launcher chooses. If
  // that path were wrong the app would still start and then report every bundled binary as
  // missing — which is exactly how the vendorRoot() bug presented.
  const missing = EXPECTED_VENDOR.filter((rel) => !fs.existsSync(path.join(RUNTIME, 'resources', 'vendor', rel)))
  check('the vendor tree lands where the app looks for it', missing.length === 0, missing.join(', '))

  // Staging is where the archive and extractor land during unpacking. Leaving it behind wastes
  // most of a gigabyte, and RMDir fails silently when the extractor is still briefly locked —
  // which is exactly what happened the first time this design was built.
  const staging = (await fsp.readdir(path.dirname(RUNTIME)).catch(() => [])).filter((d) => d.startsWith('staging-'))
  check('no staging directory is left behind', staging.length === 0, staging.join(', '))

  const breadcrumb = path.join(appData, 'models-path.json')
  const recorded = JSON.parse(await fsp.readFile(breadcrumb, 'utf8').catch(() => '{}'))
  check('the app records the drop folder as its exe directory',
    path.resolve(recorded.exeDir ?? '').toLowerCase() === path.resolve(drop).toLowerCase(),
    `recorded ${recorded.exeDir}`)
  check('the models directory stays beside the exe',
    path.resolve(recorded.modelsDir ?? '').toLowerCase().startsWith(path.resolve(drop).toLowerCase()),
    `recorded ${recorded.modelsDir}`)

  killApp()
  await sleep(3000)

  // ---- second launch: must skip the unpack
  console.log('\nSecond launch (expect no unpack)…')
  const t1 = Date.now()
  spawn(dropExe, [], { env, detached: true, stdio: 'ignore' }).unref()
  const restarted = await waitFor('app', async () => appRunning(), 120000)
  const elapsed = restarted === null ? null : Date.now() - t1
  check('the app starts again', restarted !== null)
  check('the second launch skips the unpack', elapsed !== null && elapsed < 30000,
    elapsed === null ? 'never started' : `${(elapsed / 1000).toFixed(0)}s`)

  killApp()
} finally {
  killApp()
  await sleep(1500)
  await fsp.rm(drop, { recursive: true, force: true }).catch(() => undefined)
}

console.log(failures === 0 ? '\nPortable exe verified.\n' : `\n${failures} problem(s).\n`)
process.exit(failures === 0 ? 0 : 1)
