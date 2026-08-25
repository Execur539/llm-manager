/**
 * Smoke-test the packaged build before wrapping it in an installer.
 *
 * The bug this exists to catch: `vendorRoot()` once pointed at a directory that only exists in a
 * dev tree, so a packaged app reported every bundled binary as missing and failed to spawn
 * anything — while the files were sitting right there in `resources/vendor`. Nothing in the
 * source tree can detect that; it only shows up once the app is actually packaged.
 *
 * Runs against a throwaway app-data directory so it cannot touch real settings or models, and
 * loads nothing: it asks the app what it can see, then quits.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { _electron as electron } from 'playwright-core'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const EXE = path.join(ROOT, 'release', 'win-unpacked', 'LLM Manager.exe')

if (!fs.existsSync(EXE)) {
  console.error(`No packaged build at ${EXE}\nRun \`npm run pack:dir\` first.`)
  process.exit(1)
}

const base = await fsp.mkdtemp(path.join(os.tmpdir(), 'llmm-packaged-'))
const appData = path.join(base, 'appData')
const modelsDir = path.join(base, 'LLMManagerModels')
await fsp.mkdir(path.join(appData, 'LLMManager'), { recursive: true })
await fsp.mkdir(modelsDir, { recursive: true })
// Breadcrumb, so the app does not open its relocation dialog and block on a modal.
await fsp.writeFile(
  path.join(appData, 'LLMManager', 'models-path.json'),
  JSON.stringify({ modelsDir, exeDir: base, updatedAt: Date.now() }, null, 2)
)

let failures = 0
const check = (label, ok, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

const app = await electron.launch({
  executablePath: EXE,
  args: [],
  env: {
    ...process.env,
    LLMM_APPDATA_DIR: path.join(appData, 'LLMManager'),
    LLMM_MODELS_DIR: modelsDir,
    LLMM_PORTABLE_DIR: base
  },
  timeout: 90000
})

try {
  const page = await app.firstWindow({ timeout: 60000 })
  const consoleErrors = []
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()))
  page.on('pageerror', (e) => consoleErrors.push(`uncaught: ${e.message}`))
  await page.waitForSelector('.nav-item', { timeout: 40000 })
  check('the packaged window opens and renders', true)

  // ---- the thing that was broken before
  const paths = await page.evaluate(() => window.api.invoke('app:paths'))
  console.log(`\n  resources: ${paths?.resources ?? '?'}`)
  console.log(`  models:    ${paths?.modelsDir ?? '?'}`)

  for (const backend of ['cuda', 'vulkan', 'cpu']) {
    const d = await page.evaluate((b) => window.api.invoke('runtime:vendor-diagnostics', b), backend)
    const missing = d?.missing ?? []
    check(
      `${backend}: all bundled binaries resolve`,
      d?.rootExists === true && missing.length === 0,
      missing.length ? `missing ${missing.join(', ')}` : `${(d?.present ?? []).length} present at ${d?.root ?? '?'}`
    )
  }

  // ---- the sidecars the agent tools depend on
  const info = await page.evaluate(() => window.api.invoke('runtime:vendor-info'))
  for (const [name, present] of Object.entries(info ?? {})) {
    check(`sidecar present: ${name}`, !!present, String(present))
  }

  // ---- hardware detection has to work from the packaged process too
  const hw = await page.evaluate(() => window.api.invoke('hardware:get'))
  check('hardware is detected', Array.isArray(hw?.gpus), `backend=${hw?.backend} gpus=${hw?.gpus?.length}`)

  // ---- the database opens and writes in the sandboxed app-data directory
  const chat = await page.evaluate(() => window.api.invoke('chat:create', { kind: 'chat' }))
  check('the database initialises and accepts a write', !!chat?.id, chat?.id ?? 'no id')

  // ---- version, so the artifact can be labelled honestly
  const version = await page.evaluate(() => window.api.invoke('app:version'))
  console.log(`\n  version: ${JSON.stringify(version)}`)

  const meaningful = consoleErrors.filter((e) => !/DevTools|Autofill|Electron Security Warning/i.test(e))
  check('no console errors on startup', meaningful.length === 0, meaningful.slice(0, 2).join(' | '))
} finally {
  await app.close().catch(() => undefined)
  await fsp.rm(base, { recursive: true, force: true }).catch(() => undefined)
}

console.log(failures === 0 ? '\nPackaged build looks good.\n' : `\n${failures} problem(s).\n`)
process.exit(failures === 0 ? 0 : 1)
