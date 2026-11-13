/**
 * The update installer's decisions, run against the shipping module.
 *
 * Installing hands over to the new version instead of swapping files behind the app's back (see
 * src/main/update/install.ts). This checks every decision that hand-off rests on: which file is
 * replaced, where the download waits, what a recorded install means to whichever version starts
 * next, which shortcuts get a forwarder, and that the final rename retries rather than giving up
 * at the first sharing violation. The launcher and the stub themselves are run for real by
 * scripts/launcher-check.mjs.
 *
 *   node scripts/update-install-check.mjs
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import esbuild from 'esbuild'

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

// Transpile the one module and import it, so what is tested is what ships.
const built = await esbuild.build({
  entryPoints: [path.join(ROOT, 'src', 'main', 'update', 'install.ts')],
  format: 'esm',
  platform: 'node',
  write: false,
  bundle: false
})
const tmpMod = path.join(os.tmpdir(), `llmm-install-${process.pid}.mjs`)
await fsp.writeFile(tmpMod, built.outputFiles[0].text, 'utf8')
const m = await import(`file://${tmpMod.replace(/\\/g, '/')}`)

const cache = 'C:\\Users\\u\\AppData\\Local\\LLMManager'
const cacheExe = `${cache}\\runtime-1.0.11\\LLM Manager.exe`
const folder = 'D:\\Apps\\LLM-Manager'
const portable = `${folder}\\LLM-Manager-portable.exe`
const on = (...files) => (f) => files.some((x) => x.toLowerCase() === f.toLowerCase())

// ---------------------------------------------------------------- which file is replaced

console.log('\nAn update replaces the portable exe the user launches')
{
  const pick = (input) => m.chooseUpdateTarget({ appExe: cacheExe, cacheRoot: cache, ...input })
  const remembered = pick({ rememberedExe: portable, exists: () => true })
  check('a run started from a taskbar pin updates the exe an earlier run came from', remembered.target === portable)
  check('a remembered exe that has gone is not used', 'error' in pick({ rememberedExe: 'D:\\gone\\LLM-Manager-portable.exe', exists: () => false }))
  check('the exe the launcher names is the one replaced', pick({ portableExe: portable, portableDir: folder, exists: on(portable) }).target === portable)
  check('never the unpacked copy the app runs from', pick({ portableExe: cacheExe, portableDir: folder, exists: on(portable, cacheExe) }).target === portable)
  check(
    'never a download still waiting to install: the real exe beside it is',
    pick({ portableExe: `${folder}\\.update-2.0.0-1.exe`, portableDir: folder, exists: () => true }).target === portable
  )
  check('a launcher that only passed its folder still has its exe found', pick({ portableDir: folder, exists: on(portable) }).target === portable)
  const blind = pick({ exists: on(cacheExe) })
  check('a copy started from its unpacked files is told why it cannot update', 'error' in blind && /unpacked/.test(blind.error))
  check(
    'an install elsewhere is not overwritten with the portable exe',
    'error' in m.chooseUpdateTarget({ appExe: 'C:\\Program Files\\LLM Manager\\LLM Manager.exe', cacheRoot: cache, exists: () => true })
  )
}

console.log('\nAn update an older version put in the wrong place is found')
{
  const found = m.misplacedLauncher({ portableExe: cacheExe, portableDir: folder, cacheRoot: cache, exists: on(portable, cacheExe) })
  check('a launcher in the cache, handed the real folder, is copied to it', found?.from === cacheExe && found?.to === portable)
  check('an ordinary launch has nothing to repair', m.misplacedLauncher({ portableExe: portable, portableDir: folder, cacheRoot: cache, exists: on(portable) }) === null)
  check('without the real folder there is nothing to repair', m.misplacedLauncher({ portableExe: cacheExe, cacheRoot: cache, exists: on(cacheExe) }) === null)
}

// ---------------------------------------------------------------- the download

console.log('\nThe download waits beside the exe, under a name of its own')
{
  const staged = m.stagedPath(portable, '1.8.0', 1234)
  check('in the same folder, so the final step is a rename on one drive', path.dirname(staged) === folder)
  check('hidden and versioned', path.basename(staged) === '.update-1.8.0-1234.exe', staged)
  check('a version cannot put anything but itself in the name', !/[\\/:]/.test(path.basename(m.stagedPath(portable, '1.8.0/../../x', 1))))
  check('is recognised as waiting to replace that exe', m.isStagedFor(staged, portable))
  check('the exe itself is not a download', !m.isStagedFor(portable, portable))
  check('a download in another folder is not trusted to replace this exe', !m.isStagedFor(`E:\\elsewhere\\.update-1.8.0-1.exe`, portable))
  check('nor is any other file in the folder', !m.isStagedFor(`${folder}\\notes.exe`, portable))
}

console.log('\nThe download is checked before it is used')
{
  const hex = 'a'.repeat(64)
  check("GitHub's digest is read", m.parseDigest(`sha256:${hex}`) === hex)
  check('in any case', m.parseDigest(`SHA256:${hex.toUpperCase()}`) === hex)
  check('a missing digest is null, not a pass', m.parseDigest(undefined) === null && m.parseDigest(null) === null)
  check('a malformed one is null', m.parseDigest('sha256:abc') === null && m.parseDigest(`md5:${hex}`) === null)
  check('a Windows program starts with MZ', m.looksLikeExe(Buffer.from('MZ\x90\x00')))
  check('an HTML error page does not', !m.looksLikeExe(Buffer.from('<!DOCTYPE html>')))
  check('nor does nothing', !m.looksLikeExe(Buffer.alloc(0)))
}

// ---------------------------------------------------------------- what a recorded install means

console.log('\nA recorded install, as seen by whichever version starts next')
{
  const pending = { version: '1.8.0', staged: m.stagedPath(portable, '1.8.0', 1), target: portable, from: '1.7.3', at: 1_000_000 }
  check('the new version sees it has arrived', m.pendingVerdict(pending, '1.8.0', 1_000_000 + 5_000) === 'arrived')
  check('so does a later one', m.pendingVerdict(pending, '1.9.0', 1_000_000 + 5_000) === 'arrived')
  check('the old version waits while the new one may still be unpacking', m.pendingVerdict(pending, '1.7.3', 1_000_000 + 60_000) === 'starting')
  check(
    'and calls it failed once it is well past starting',
    m.pendingVerdict(pending, '1.7.3', 1_000_000 + m.START_GRACE_MS + 1) === 'failed'
  )
  check('a record survives the round trip', JSON.stringify(m.parsePending(JSON.stringify(pending))) === JSON.stringify(pending))
  check('a damaged record is ignored', m.parsePending('{"version":') === null && m.parsePending('{"version":"1"}') === null)
  check('versions compare as numbers, not text', m.compareVersions('1.10.0', '1.9.9') > 0 && m.compareVersions('v1.7.2', '1.7.2') === 0)
}

// ---------------------------------------------------------------- forwarders

console.log('\nShortcuts to an unpacked copy are recognised')
{
  check('a removed version\'s app gets a forwarder', m.isRuntimeExe(`${cache}\\runtime-1.6.2\\LLM Manager.exe`, cache))
  check('in any case', m.isRuntimeExe(`${cache.toLowerCase()}\\RUNTIME-1.6.2\\llm manager.exe`, cache))
  check('not the stub itself', !m.isRuntimeExe(`${cache}\\LLM Manager.exe`, cache))
  check('not another program in the folder', !m.isRuntimeExe(`${cache}\\runtime-1.6.2\\resources\\vendor\\llama-server.exe`, cache))
  check('not a folder of the same name somewhere else', !m.isRuntimeExe(`D:\\runtime-1.6.2\\LLM Manager.exe`, cache))
}

// ---------------------------------------------------------------- the final rename, on real files

console.log('\nThe new version takes the old exe\'s place')
const work = await fsp.mkdtemp(path.join(os.tmpdir(), 'llmm-install-test-'))
{
  const target = path.join(work, 'LLM-Manager-portable.exe')
  const staged = m.stagedPath(target, '1.8.0', Date.now())
  await fsp.writeFile(target, 'OLD VERSION')
  await fsp.writeFile(staged, 'NEW VERSION')
  const ok = await m.replaceWithRetry(staged, target, { rename: (a, b) => fsp.rename(a, b), tries: 3, delayMs: 50 })
  check('the rename reports success', ok)
  check('the exe now holds the new version', fs.readFileSync(target, 'utf8') === 'NEW VERSION')
  check('the download is gone, not left beside it', !fs.existsSync(staged))
  check('exactly one exe remains', fs.readdirSync(work).filter((f) => f.endsWith('.exe')).length === 1)
}
{
  // A file held open the way a scanner holds it: two refusals, then it lets go.
  let refusals = 2
  let calls = 0
  const started = Date.now()
  const ok = await m.replaceWithRetry('a', 'b', {
    tries: 5,
    delayMs: 100,
    rename: async () => {
      calls++
      if (refusals-- > 0) throw Object.assign(new Error('EBUSY'), { code: 'EBUSY' })
    }
  })
  check('a file held for a moment is waited out', ok && calls === 3)
  check('and actually waited for between attempts', Date.now() - started >= 180, `${Date.now() - started}ms`)
}
{
  const target = path.join(work, 'no-such-folder', 'LLM-Manager-portable.exe')
  const staged = path.join(work, '.update-9.9.9-1.exe')
  await fsp.writeFile(staged, 'NEW VERSION')
  const ok = await m.replaceWithRetry(staged, target, { rename: (a, b) => fsp.rename(a, b), tries: 2, delayMs: 20 })
  check('a rename that can never work gives up and says so', ok === false)
  check('without losing the download, so the running version can report it', fs.existsSync(staged))
}

console.log('\nWhat the user is told')
{
  check('a version that never started says this one was kept', /was kept/.test(m.notStartedText('1.8.0')))
  check('one that could not replace the exe names the file', m.notReplacedText('1.8.0', portable).includes(portable))
}

await fsp.rm(work, { recursive: true, force: true }).catch(() => undefined)
await fsp.rm(tmpMod, { force: true }).catch(() => undefined)

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
