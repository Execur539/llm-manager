/**
 * The batch file that replaces the exe once the app has let go of it.
 *
 * Split out from the updater so it can be run against real files in a test without dragging in
 * Electron. That is the point of the split: the version this replaced was a template literal in
 * the middle of an async function that only ever ran during a genuine update, so it had never
 * once been executed — and it carried two faults that would have surfaced the first time it was.
 */

/** Where the helper records that it gave up, relative to the app's userData directory. */
export const FAILURE_MARKER = 'update-failed.txt'

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
  relaunch = true
}: SwapScriptOptions): string {
  const start = relaunch ? ['start "" "%TARGET%"'] : []
  return [
    '@echo off',
    'setlocal',
    `set "TARGET=${target}"`,
    `set "STAGED=${staged}"`,
    `set "MARKER=${marker}"`,
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
