/**
 * Regression tests for portable path resolution.
 *
 * The bug this guards against: the portable build runs from an extraction cache under
 * LOCALAPPDATA, so `app.getPath('exe')` pointed at the unpacked copy rather than the file the
 * user actually launched. The models folder was therefore created inside a cache directory that
 * the upgrade path deletes — and 18 GB of models were moved into it before this was caught.
 *
 * Windows paths are written with String.raw so backslashes survive verbatim.
 */

import path from 'node:path'

// Mirrors isInsideRuntimeCache / exeDir from src/main/storage/paths.ts without needing Electron.
const CACHE_RE = /[\\/]LLMManager[\\/]runtime-[^\\/]+$/i

function runtimeCacheDir(exePath) {
  const dir = path.win32.dirname(exePath)
  return CACHE_RE.test(dir) ? dir : null
}

function isInsideRuntimeCache(exePath, target) {
  const cache = runtimeCacheDir(exePath)
  if (!cache) return false
  const rel = path.win32.relative(cache, path.win32.resolve(target))
  return rel === '' || (!rel.startsWith('..') && !path.win32.isAbsolute(rel))
}

function exeDir(exePath, portableEnv, isPackaged, appPath) {
  if (portableEnv) return portableEnv
  if (isPackaged) return path.win32.dirname(exePath)
  return appPath
}

// Mirrors exeLocationUnknown from src/main/storage/paths.ts.
function exeLocationUnknown(exePath, portableEnv) {
  return runtimeCacheDir(exePath) !== null && !portableEnv
}

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

const CACHE_EXE = String.raw`C:\Users\dev\AppData\Local\LLMManager\runtime-0.1.0\LLM Manager.exe`
const CACHE_DIR = String.raw`C:\Users\dev\AppData\Local\LLMManager\runtime-0.1.0`
const REAL_DIR = String.raw`D:\CODE\LLM Manager\release`

console.log('\nPortable launcher passes the real exe location')
check('env var wins over the extraction cache', exeDir(CACHE_EXE, REAL_DIR, true, '') === REAL_DIR)
check(
  'without it, the exe path is the cache — this was the bug',
  exeDir(CACHE_EXE, undefined, true, '').includes('runtime-0.1.0')
)

console.log('\nRuntime cache detection')
check('recognises the cache directory', runtimeCacheDir(CACHE_EXE) !== null)
check('an installed location is not a cache', runtimeCacheDir(String.raw`C:\Program Files\LLM Manager\LLM Manager.exe`) === null)
check('a portable folder is not a cache', runtimeCacheDir(String.raw`D:\Apps\LLM Manager\LLM Manager.exe`) === null)
check('a lookalike folder elsewhere is not a cache', runtimeCacheDir(String.raw`D:\stuff\runtime-0.1.0\LLM Manager.exe`) === null)

console.log('\nModels must never land inside the cache')
check('a models path under the cache is rejected', isInsideRuntimeCache(CACHE_EXE, `${CACHE_DIR}\\LLMManagerModels`))
check('the cache root itself is rejected', isInsideRuntimeCache(CACHE_EXE, CACHE_DIR))
check('nested deeper is still rejected', isInsideRuntimeCache(CACHE_EXE, `${CACHE_DIR}\\a\\b\\c`))
check(
  'a sibling runtime dir is not inside this one',
  !isInsideRuntimeCache(CACHE_EXE, String.raw`C:\Users\dev\AppData\Local\LLMManager\runtime-0.2.0\LLMManagerModels`)
)
check('a path on another drive is fine', !isInsideRuntimeCache(CACHE_EXE, String.raw`D:\CODE\LLM Manager\LLMManagerModels`))
check('traversal back out of the cache is not inside', !isInsideRuntimeCache(CACHE_EXE, `${CACHE_DIR}\\..\\Models`))

console.log('\nWith the fix, models land beside the real exe')
const resolved = path.win32.join(exeDir(CACHE_EXE, REAL_DIR, true, ''), 'LLMManagerModels')
check('models sit beside the portable exe', resolved === String.raw`D:\CODE\LLM Manager\release\LLMManagerModels`)
check('and that location is not in the cache', !isInsideRuntimeCache(CACHE_EXE, resolved))

console.log('\nInstalled builds are unaffected')
const installedExe = String.raw`C:\Program Files\LLM Manager\LLM Manager.exe`
const installedModels = path.win32.join(exeDir(installedExe, undefined, true, ''), 'LLMManagerModels')
check('installer build still puts models beside the exe', installedModels === String.raw`C:\Program Files\LLM Manager\LLMManagerModels`)
check('and is never treated as a cache', !isInsideRuntimeCache(installedExe, installedModels))

/*
 * Running the unpacked copy directly, which is what a taskbar pin does.
 *
 * Windows resolves a pin to the executable it can see, and for a portable build that is the
 * copy inside the extraction cache — so the launcher never runs and LLMM_PORTABLE_DIR is never
 * set. Every earlier case here assumed the variable was present, so nothing covered the state
 * a user reaches simply by pinning the app they are running.
 *
 * The app cannot know where it really lives in that state, and must say so rather than guess.
 * Guessing is what produced an offer to copy a 17 GB library off the drive it was already on.
 */
console.log('\nLaunched from the extraction cache, with no launcher to say where the exe is')
check('the exe location is known to be unknowable', exeLocationUnknown(CACHE_EXE, undefined))
check(
  'a portable launch is not affected — the launcher passes the real folder',
  !exeLocationUnknown(CACHE_EXE, REAL_DIR)
)
check('nor is an installed build, which is never in a cache', !exeLocationUnknown(installedExe, undefined))
check(
  'and without the launcher, the exe dir really is the cache — which is why it must not be trusted',
  exeDir(CACHE_EXE, undefined, true, '') === CACHE_DIR
)

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
