/**
 * Runs the portable launcher and the entry stub for real, in a sandbox.
 *
 * The launcher decides what every shortcut, every pin and every update ends up starting, and until
 * now it had only ever been tested by releasing it. This builds a test copy of build/portable.nsi and
 * build/stub.nsi with %LOCALAPPDATA% pointed at a scratch folder and the registry key renamed, packs a
 * fake app that only records how it was started, and walks the life of an install:
 *
 *   first run → second run → an update to a new version → a pin made to the old version →
 *   a download waiting to install → a launcher that has been moved.
 *
 * Nothing pops up: the setup banner is removed from the test copy and message boxes become log
 * lines. The real install, the real registry key and the real app are never touched.
 *
 *   node scripts/launcher-check.mjs
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { execFile, spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const TEST_KEY = 'Software\\LLM Manager Launcher Test'

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

function findMakensis() {
  const cache = path.join(process.env.LOCALAPPDATA ?? '', 'electron-builder', 'Cache', 'nsis')
  const found = [process.env.NSIS_HOME && path.join(process.env.NSIS_HOME, 'makensis.exe')]
  if (fs.existsSync(cache)) {
    for (const dir of fs.readdirSync(cache)) {
      found.push(path.join(cache, dir, 'Bin', 'makensis.exe'), path.join(cache, dir, 'makensis.exe'))
    }
  }
  found.push('C:\\Program Files (x86)\\NSIS\\makensis.exe', 'C:\\Program Files\\NSIS\\makensis.exe')
  return found.find((f) => f && fs.existsSync(f))
}

const run = (file, args, opts = {}) =>
  new Promise((resolve, reject) =>
    execFile(file, args, { windowsHide: true, maxBuffer: 1 << 24, ...opts }, (err, stdout, stderr) =>
      err ? reject(new Error(`${path.basename(file)} ${args.join(' ')}\n${stdout}${stderr}`)) : resolve(stdout)
    )
  )

/** Start an exe and wait for it to exit, as a double-click would start it. */
const launch = (file, env = {}, args = []) =>
  new Promise((resolve) => {
    const child = spawn(file, args, { env: { ...process.env, ...env }, stdio: 'ignore', cwd: path.dirname(file) })
    child.on('exit', (code) => resolve(code))
    child.on('error', () => resolve(-1))
  })

const until = async (cond, ms = 15000) => {
  const end = Date.now() + ms
  while (Date.now() < end) {
    if (await cond()) return true
    await new Promise((r) => setTimeout(r, 100))
  }
  return false
}

const regRead = async () => {
  try {
    const out = await run('reg.exe', ['query', `HKCU\\${TEST_KEY}`, '/v', 'Launcher'])
    return /Launcher\s+REG_SZ\s+(.+)/.exec(out)?.[1].trim() ?? null
  } catch {
    return null
  }
}

const makensis = findMakensis()
const sevenZip = path.join(ROOT, 'node_modules', '7zip-bin', 'win', 'x64', '7za.exe')
if (process.platform !== 'win32' || !makensis || !fs.existsSync(sevenZip)) {
  console.log('Skipped: needs Windows, makensis (from electron-builder) and 7zip-bin.')
  process.exit(0)
}

const work = await fsp.mkdtemp(path.join(os.tmpdir(), 'llmm-launcher-'))
const local = path.join(work, 'Local')
const cacheRoot = path.join(local, 'LLMManager')
const build = path.join(work, 'build')
await fsp.mkdir(local, { recursive: true })
await fsp.mkdir(build, { recursive: true })

/** The test copy of a script: sandboxed paths and key, and nothing that opens a window. */
async function testCopy(name) {
  let text = (await fsp.readFile(path.join(ROOT, 'build', name), 'utf8')).replace(/^\uFEFF/, '')
  text = text.replaceAll('$LOCALAPPDATA', local)
  text = text.replaceAll('"Software\\LLM Manager"', `"${TEST_KEY}"`)
  text = text.replace(/^\s*Banner::\w+.*$/gm, '')
  text = text.replace(/MessageBox MB_ICON\w+ /g, 'DetailPrint ')
  const out = path.join(build, name)
  await fsp.writeFile(out, text, 'utf8')
  return out
}

/** A stand-in for the app: records where it ran from and what it was told, then exits. */
async function fakeApp(out) {
  const script = path.join(build, 'fake-app.nsi')
  await fsp.writeFile(
    script,
    `Unicode true
RequestExecutionLevel user
SilentInstall silent
OutFile "${out}"
Section
  ReadEnvStr $1 "LLMM_PORTABLE_EXE"
  ReadEnvStr $2 "LLMM_PORTABLE_DIR"
  ReadEnvStr $3 "LLMM_UPDATE_TARGET"
  FileOpen $0 "${work}\\last-run.txt" w
  FileWriteUTF16LE $0 "$EXEDIR|$1|$2|$3"
  FileClose $0
SectionEnd
`,
    'utf8'
  )
  await run(makensis, ['/V2', script])
}

const lastRun = async () => {
  try {
    const raw = await fsp.readFile(path.join(work, 'last-run.txt'))
    const text = raw.toString('utf16le').replace(/^\uFEFF/, '').replace(/\0/g, '')
    const [dir, exe, portableDir, updateTarget] = text.split('|')
    return { dir, exe, portableDir, updateTarget }
  } catch {
    return null
  }
}
const clearRun = () => fsp.rm(path.join(work, 'last-run.txt'), { force: true })

async function buildLauncher(version) {
  const payload = path.join(build, `payload-${version}`)
  await fsp.mkdir(path.join(payload, 'resources'), { recursive: true })
  await fakeApp(path.join(payload, 'LLM Manager.exe'))
  await fsp.writeFile(path.join(payload, 'resources', 'app.asar'), `app ${version}`)
  const appArchive = path.join(build, `app-${version}.7z`)
  const vendorArchive = path.join(build, `vendor-${version}.7z`)
  await fsp.rm(appArchive, { force: true })
  await fsp.rm(vendorArchive, { force: true })
  await run(sevenZip, ['a', '-t7z', '-bso0', '-bsp0', appArchive, path.join(payload, '*')])
  const vendorDir = path.join(build, `vendor-${version}`)
  await fsp.mkdir(vendorDir, { recursive: true })
  await fsp.writeFile(path.join(vendorDir, 'llama.txt'), version)
  await run(sevenZip, ['a', '-t7z', '-bso0', '-bsp0', vendorArchive, path.join(vendorDir, '*')])

  const stubScript = await testCopy('stub.nsi')
  const stub = path.join(build, `stub-${version}.exe`)
  await run(makensis, ['/V2', `/DVERSION=${version}`, `/DVERSION4=${version}.0`, `/DOUTFILE=${stub}`, stubScript])
  const launcherScript = await testCopy('portable.nsi')
  const out = path.join(build, `launcher-${version}.exe`)
  await run(makensis, [
    '/V2',
    `/DVERSION=${version}`,
    '/DNEED_MB=1',
    `/DVERSION4=${version}.0`,
    `/DAPP_ARCHIVE=${appArchive}`,
    `/DVENDOR_ARCHIVE=${vendorArchive}`,
    `/DEXTRACTOR=${sevenZip}`,
    '/DAPPEXE=LLM Manager.exe',
    `/DBUILDID=b${version.replace(/\./g, '')}`,
    `/DOUTFILE=${out}`,
    `/DSTUB=${stub}`,
    launcherScript
  ])
  return { launcher: out, stub }
}

try {
  await run('reg.exe', ['delete', `HKCU\\${TEST_KEY}`, '/f']).catch(() => undefined)
  console.log('\nBuilding two test versions…')
  const v1 = await buildLauncher('1.0.0')
  const v2 = await buildLauncher('1.1.0')

  const apps = path.join(work, 'Apps')
  await fsp.mkdir(apps, { recursive: true })
  const portable = path.join(apps, 'LLM-Manager-portable.exe')
  const entry = path.join(cacheRoot, 'LLM Manager.exe')
  const runtime = (v) => path.join(cacheRoot, `runtime-${v}`)

  console.log('\nFirst run')
  await fsp.copyFile(v1.launcher, portable)
  await clearRun()
  await launch(portable)
  check('the app is started', await until(async () => !!(await lastRun())))
  let r = await lastRun()
  check('from the unpacked copy of its version', r?.dir.toLowerCase() === runtime('1.0.0').toLowerCase(), r?.dir)
  check('and told which portable exe started it', r?.exe.toLowerCase() === portable.toLowerCase(), r?.exe)
  check('the vendor tree lands under resources', fs.existsSync(path.join(runtime('1.0.0'), 'resources', 'vendor', 'llama.txt')))
  check('the entry stub is installed where it never moves', fs.existsSync(entry))
  check('the launcher records itself for the stub', (await regRead())?.toLowerCase() === portable.toLowerCase(), String(await regRead()))

  console.log('\nSecond run: no unpacking')
  await clearRun()
  const markerTime = fs.statSync(runtime('1.0.0')).mtimeMs
  await launch(portable)
  check('the app is started again', await until(async () => !!(await lastRun())))
  check('from the same unpacked copy, untouched', fs.statSync(runtime('1.0.0')).mtimeMs === markerTime)

  console.log('\nThe stub starts the app')
  await clearRun()
  await launch(entry)
  check('a pin to the stub starts the app', await until(async () => !!(await lastRun())))
  r = await lastRun()
  check('through the portable exe, so updates still find it', r?.exe.toLowerCase() === portable.toLowerCase(), r?.exe)

  console.log('\nAn update to a new version')
  await fsp.copyFile(v2.launcher, portable)
  await clearRun()
  await launch(portable)
  check('the new version is started', await until(async () => !!(await lastRun())))
  r = await lastRun()
  check('from its own unpacked copy', r?.dir.toLowerCase() === runtime('1.1.0').toLowerCase(), r?.dir)
  check('the old version is cleaned up', !fs.existsSync(path.join(runtime('1.0.0'), 'resources')))
  check('and a forwarder is left where its app was', fs.existsSync(path.join(runtime('1.0.0'), 'LLM Manager.exe')) && fs.existsSync(path.join(runtime('1.0.0'), '.forwarder')))
  check(
    'the forwarder is the stub',
    fs.existsSync(path.join(runtime('1.0.0'), 'LLM Manager.exe')) &&
      Buffer.compare(fs.readFileSync(path.join(runtime('1.0.0'), 'LLM Manager.exe')), fs.readFileSync(entry)) === 0
  )

  console.log('\nA pin made to the old version')
  await clearRun()
  await launch(path.join(runtime('1.0.0'), 'LLM Manager.exe'))
  check('starts something', await until(async () => !!(await lastRun())))
  r = await lastRun()
  check('and that is the newest version', r?.dir.toLowerCase() === runtime('1.1.0').toLowerCase(), r?.dir)

  console.log('\nThe next launch leaves forwarders alone')
  await clearRun()
  await launch(portable)
  await until(async () => !!(await lastRun()))
  check('the forwarder is still there', fs.existsSync(path.join(runtime('1.0.0'), '.forwarder')))
  check('no staging folders are left behind', !fs.readdirSync(cacheRoot).some((n) => n.startsWith('staging-')), fs.readdirSync(cacheRoot).join(','))

  console.log('\nA download waiting to install')
  const staged = path.join(apps, '.update-1.1.0-123.exe')
  await fsp.copyFile(v2.launcher, staged)
  await clearRun()
  await launch(staged, { LLMM_UPDATE_TARGET: portable })
  check('starts the app', await until(async () => !!(await lastRun())))
  r = await lastRun()
  check('which is told the exe it will replace', r?.updateTarget.toLowerCase() === portable.toLowerCase(), r?.updateTarget)
  check('and runs as the download until then', r?.exe.toLowerCase() === staged.toLowerCase(), r?.exe)
  check('shortcuts keep leading to the real exe, not the download', (await regRead())?.toLowerCase() === portable.toLowerCase(), String(await regRead()))

  console.log('\nA launcher that has been moved')
  const movedDir = path.join(work, 'Moved here')
  await fsp.mkdir(movedDir, { recursive: true })
  const moved = path.join(movedDir, 'LLM-Manager-portable.exe')
  await fsp.rename(portable, moved)
  await clearRun()
  await launch(moved)
  await until(async () => !!(await lastRun()))
  check('records its new place', (await regRead())?.toLowerCase() === moved.toLowerCase(), String(await regRead()))
  await clearRun()
  await launch(entry)
  check('so the stub follows it', await until(async () => (await lastRun())?.exe.toLowerCase() === moved.toLowerCase()))
} catch (err) {
  fail++
  console.log(`  FAIL  ${err.message.split('\n').slice(0, 6).join('\n')}`)
} finally {
  await run('reg.exe', ['delete', `HKCU\\${TEST_KEY}`, '/f']).catch(() => undefined)
  await new Promise((r) => setTimeout(r, 500))
  await fsp.rm(work, { recursive: true, force: true }).catch(() => undefined)
}

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
