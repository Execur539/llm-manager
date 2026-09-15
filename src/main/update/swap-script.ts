/**
 * The batch file that replaces the exe once the app has let go of it.
 *
 * Split out from the updater so it can be run against real files in a test without dragging in
 * Electron. That is the point of the split: the version this replaced was a template literal in
 * the middle of an async function that only ever ran during a genuine update, so it had never
 * once been executed — and it carried two faults that would have surfaced the first time it was.
 */

import path from 'node:path'

/** Where the helper records that it gave up, relative to the app's userData directory. */
export const FAILURE_MARKER = 'update-failed.txt'

/** The portable exe's name — stable across versions since 1.0, so a download replaces the file shortcuts point at. */
export const PORTABLE_EXE_NAME = 'LLM-Manager-portable.exe'

function isInside(root: string, target: string): boolean {
  const rel = path.relative(root, target)
  return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel)
}

export interface UpdateTargetInput {
  /** LLMM_PORTABLE_EXE: the launcher that started this run, as it reports its own path. */
  portableExe?: string
  /** LLMM_PORTABLE_DIR: the folder the user's portable exe is in. */
  portableDir?: string
  /** The running app's own executable. */
  appExe: string
  /** %LOCALAPPDATA%\LLMManager, where the launcher unpacks the app. */
  cacheRoot: string
  exists: (file: string) => boolean
}

/**
 * Which file an update replaces: the portable exe the user actually launches.
 *
 * Every updater before this one replaced the running app's own executable. In the portable build
 * that is the copy unpacked under LOCALAPPDATA, not the file anyone double-clicks — so an update ran
 * once, from the cache, and the next launch of the user's own exe started the old version again,
 * which unpacked itself over the top and deleted the new one as stale. The real file was never
 * touched. An install that was not started from a portable exe cannot be updated in place at all,
 * since what the release feed offers is a portable exe, so it is told so rather than overwritten.
 */
export function chooseUpdateTarget(input: UpdateTargetInput): { target: string } | { error: string } {
  const { portableExe, portableDir, appExe, cacheRoot, exists } = input
  if (portableExe && !isInside(cacheRoot, portableExe) && exists(portableExe)) return { target: portableExe }
  // Launchers before 1.6.6 passed their folder but not their own path; the name has not changed.
  if (portableDir && !isInside(cacheRoot, portableDir)) {
    const beside = path.join(portableDir, PORTABLE_EXE_NAME)
    if (exists(beside)) return { target: beside }
  }
  return {
    error: isInside(cacheRoot, appExe)
      ? `This copy was started from its unpacked files rather than from ${PORTABLE_EXE_NAME}, so there is no exe to update. Start the app from ${PORTABLE_EXE_NAME} and update again, or download the new version from the releases page.`
      : `This copy was not started from ${PORTABLE_EXE_NAME}, so there is nothing to update in place. Download the new version from the releases page.`
  }
}

/**
 * A launcher an older updater left in the unpack cache, and the user's exe it was meant to replace.
 *
 * Those updaters overwrote the unpacked app with the new launcher and started it from there. The
 * launcher keeps the real folder it was handed down, so the app it starts can finish the update by
 * copying the launcher over the exe the user launches. Null when this run did not come about that way.
 */
export function misplacedLauncher(input: Omit<UpdateTargetInput, 'appExe'>): { from: string; to: string } | null {
  const { portableExe, portableDir, cacheRoot, exists } = input
  if (!portableExe || !portableDir) return null
  if (!isInside(cacheRoot, portableExe) || isInside(cacheRoot, portableDir)) return null
  const to = path.join(portableDir, PORTABLE_EXE_NAME)
  return exists(portableExe) && exists(to) ? { from: portableExe, to } : null
}

/**
 * Kept free of anything cmd would interpret — no percent signs, ampersands, pipes, angle
 * brackets, carets or parentheses — because it is written out by `echo` from the batch file.
 */
export const FAILURE_TEXT =
  'The download finished but the application file could not be replaced, so the update was discarded and this version is unchanged. Another copy of the app may still have been running, or security software may have been holding the file. Try again.'

/**
 * Remove the helper, and stop dead rather than running on into the next label.
 *
 * `del "%~f0"` on its own is not enough. cmd reads the batch file from disk as it executes, so
 * an `exit /b` placed after the delete never runs — there is no longer a file to read it from.
 * That leaves the success path falling straight through into `:failed`, which would write a
 * failure marker immediately after a swap that had in fact worked.
 *
 * `(goto) 2>nul` pops the batch context, which ends execution for certain and releases the
 * handle; the `&` continuation still gets to delete the file.
 */
const SELF_DELETE = '(goto) 2>nul & del "%~f0"'

export interface SwapScriptOptions {
  target: string
  staged: string
  marker: string
  /** Roughly seconds to keep trying. Past this, something other than us is holding the exe. */
  maxTries?: number
  /**
   * Whether to relaunch the app afterwards. Only a test turns this off, so that running the
   * generated script does not start a program.
   */
  relaunch?: boolean
  /**
   * The app's process id, waited on before anything is moved.
   *
   * The file replaced is the portable launcher, which exits as soon as it has started the app, so
   * nothing holds it open: the move would succeed at once and the new version would start while
   * this one was still shutting down. Waited on by process id, never by image name, so another copy
   * of the app running from somewhere else is not mistaken for this one.
   */
  waitPid?: number
}

/**
 * Retrying the move *is* the synchronisation.
 *
 * The move fails with a sharing violation while the exe is locked and succeeds the moment it is
 * not, so there is nothing to poll and nothing to match by name. That last part matters: every
 * copy of the portable build now shares one filename, and the previous implementation waited on
 * `tasklist /fi "IMAGENAME eq ..."`, which cannot tell this copy from another one running from a
 * different folder. Two copies open meant an update that blocked until both were closed.
 *
 * `ping` is the sleep because `timeout` refuses to run at all when stdin is redirected — it
 * prints "Input redirection is not supported" and exits immediately — and redirected stdin is
 * exactly what spawning the helper with `stdio: 'ignore'` produces. The old loop's sleep
 * therefore did nothing and the wait was a busy spin on `tasklist`.
 *
 * Both helpers are named by absolute path. On a machine with Git's `usr/bin` ahead of System32,
 * a bare `timeout` resolves to the GNU build, which rejects `/t` outright.
 */
export function buildSwapScript({
  target,
  staged,
  marker,
  maxTries = 120,
  relaunch = true,
  waitPid
}: SwapScriptOptions): string {
  const start = relaunch ? ['start "" "%TARGET%"'] : []
  // A whole number only, so nothing but a process id can reach the command line.
  const wait =
    Number.isInteger(waitPid) && (waitPid as number) > 0
      ? [
          `"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -NonInteractive -Command "Wait-Process -Id ${waitPid} -Timeout 120 -ErrorAction SilentlyContinue" >nul 2>&1`
        ]
      : []
  return [
    '@echo off',
    'setlocal',
    `set "TARGET=${target}"`,
    `set "STAGED=${staged}"`,
    `set "MARKER=${marker}"`,
    ...wait,
    'set /a tries=0',
    '',
    ':retry',
    'move /y "%STAGED%" "%TARGET%" >nul 2>&1',
    'if not errorlevel 1 goto ok',
    'set /a tries+=1',
    `if %tries% geq ${maxTries} goto failed`,
    '"%SystemRoot%\\System32\\ping.exe" -n 2 127.0.0.1 >nul 2>&1',
    'goto retry',
    '',
    ':ok',
    ...start,
    SELF_DELETE,
    '',
    ':failed',
    // Written before the relaunch so the next check cannot race past it.
    `> "%MARKER%" echo ${FAILURE_TEXT}`,
    // The download is worthless now, and at this size leaving it behind is very noticeable.
    'del "%STAGED%" >nul 2>&1',
    ...start,
    SELF_DELETE
  ].join('\r\n')
}
