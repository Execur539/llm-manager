/**
 * Runs the updater's exe-swap script for real.
 *
 * This code had never once executed. The release feed pointed at a repository that did not
 * exist, so every "Check for updates" ended at HTTP 404 and the swap helper was unreachable —
 * which is how it came to ship with a sleep that does not sleep and no check on whether the
 * thing it exists to do actually happened.
 *
 * The script is generated from the real source rather than mirrored here, because a mirror that
 * drifts from the original would pass while testing nothing. `stdio: 'ignore'` matches how the
 * app spawns it, and that detail is the whole point of one of these cases.
 *
 *   node scripts/update-swap-check.mjs
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'
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

// Transpile the one module and import it, so the script under test is the shipping one.
const built = await esbuild.build({
  entryPoints: [path.join(ROOT, 'src', 'main', 'update', 'swap-script.ts')],
  format: 'esm',
  platform: 'node',
  write: false,
  bundle: false
})
const tmpMod = path.join(os.tmpdir(), `llmm-swap-${process.pid}.mjs`)
await fsp.writeFile(tmpMod, built.outputFiles[0].text, 'utf8')
const { buildSwapScript, FAILURE_TEXT } = await import(`file://${tmpMod.replace(/\\/g, '/')}`)

/*
 * Run a generated script exactly as the app does.
 *
 * The exit code is returned but deliberately not asserted on: the helper's last act is to delete
 * itself, and cmd cannot run an `exit /b` it no longer has a file to read. Nothing observes the
 * code in production either — the app has already quit and the process is detached. What the
 * script did to the files on disk is the only thing that matters, so that is what is checked.
 */
function runScript(text, dir) {
  const file = path.join(dir, 'swap.cmd')
  fs.writeFileSync(file, text, 'utf8')
  return new Promise((resolve) => {
    const child = spawn('cmd.exe', ['/c', file], { stdio: 'ignore', windowsHide: true })
    child.on('exit', (code) => resolve({ code, file }))
  })
}

const work = await fsp.mkdtemp(path.join(os.tmpdir(), 'llmm-swap-test-'))

// ---------------------------------------------------------------- shape

console.log('\nScript shape')
{
  const text = buildSwapScript({ target: 'T.exe', staged: 'S.exe', marker: 'M.txt' })
  check('sleeps with ping, by absolute path', text.includes('%SystemRoot%\\System32\\ping.exe'))
  check(
    'never uses timeout — it exits immediately when stdin is redirected',
    !/\btimeout\b/i.test(text)
  )
  check(
    'never matches the process by image name — every copy shares one filename now',
    !/tasklist/i.test(text)
  )
  check('checks whether the move actually succeeded', text.includes('if not errorlevel 1'))
  check('the failure path removes the staged download', /:failed[\s\S]*del "%STAGED%"/.test(text))
  check('the failure path leaves a marker', /:failed[\s\S]*> "%MARKER%" echo /.test(text))
  check('relaunches by default', text.includes('start "" "%TARGET%"'))
  /*
   * cmd reads the batch file as it goes, so a plain `del "%~f0"` followed by `exit /b` never
   * reaches the exit — and the success path would then fall through into `:failed` and report a
   * failure after a swap that worked. Popping the batch context first is what prevents that.
   */
  check('the success path cannot fall through into the failure path', /:ok[\s\S]*\(goto\) 2>nul & del "%~f0"[\s\S]*:failed/.test(text))
  check('can be built without the relaunch, for this test', !buildSwapScript({ target: 'T', staged: 'S', marker: 'M', relaunch: false }).includes('start ""'))
}

// ---------------------------------------------------------------- the move succeeds

console.log('\nWhen the exe is free, the swap happens')
{
  const dir = await fsp.mkdtemp(path.join(work, 'ok-'))
  const target = path.join(dir, 'LLM-Manager-portable.exe')
  const staged = path.join(dir, '.update-123.exe')
  const marker = path.join(dir, 'update-failed.txt')
  await fsp.writeFile(target, 'OLD VERSION')
  await fsp.writeFile(staged, 'NEW VERSION')

  const { file } = await runScript(buildSwapScript({ target, staged, marker, relaunch: false }), dir)
  check('the target now holds the downloaded build', fs.readFileSync(target, 'utf8') === 'NEW VERSION')
  check('the staged download is gone, not left beside the app', !fs.existsSync(staged))
  check('no failure marker was written', !fs.existsSync(marker))
  check('the helper deletes itself', !fs.existsSync(file))
  check('exactly one exe remains', fs.readdirSync(dir).filter((f) => f.endsWith('.exe')).length === 1)
}

// ---------------------------------------------------------------- the move never succeeds

console.log('\nWhen the exe cannot be replaced, it gives up cleanly')
{
  const dir = await fsp.mkdtemp(path.join(work, 'fail-'))
  // A target inside a directory that does not exist fails every attempt, deterministically.
  const target = path.join(dir, 'no-such-dir', 'LLM-Manager-portable.exe')
  const staged = path.join(dir, '.update-456.exe')
  const marker = path.join(dir, 'update-failed.txt')
  await fsp.writeFile(staged, 'NEW VERSION')

  const started = Date.now()
  const { file } = await runScript(
    buildSwapScript({ target, staged, marker, maxTries: 2, relaunch: false }),
    dir
  )
  const elapsed = Date.now() - started

  check('the staged download is cleaned up', !fs.existsSync(staged))
  check('a marker is left for the next check to report', fs.existsSync(marker))
  check(
    'the marker carries the message the UI shows',
    fs.existsSync(marker) && fs.readFileSync(marker, 'utf8').trim() === FAILURE_TEXT
  )
  check('nothing was left in the folder but the marker', fs.readdirSync(dir).every((f) => f === 'update-failed.txt'))
  check('the helper deletes itself', !fs.existsSync(file))
  /*
   * The regression that matters most.
   *
   * Two attempts means one sleep between them. With the old `timeout`, which refuses to run
   * against redirected stdin, that sleep returned instantly and the retry loop became a busy
   * spin — 120 attempts would have burned a core and finished in milliseconds.
   */
  check('it actually waits between attempts', elapsed >= 700, `only ${elapsed}ms for one retry`)
}

await fsp.rm(work, { recursive: true, force: true }).catch(() => undefined)
await fsp.rm(tmpMod, { force: true }).catch(() => undefined)

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
