/**
 * Archives, and knowing what is on screen.
 *
 * Both were reachable only by hand-writing PowerShell through `run_command`, which meant the
 * model composed a script under approval pressure every time — and the desktop-automation tools
 * were worse than inconvenient without window awareness: `click_mouse(x, y)` is a guess at
 * coordinates on a screen whose contents the agent could capture but never enumerate, and
 * `type_text` goes to whatever happens to have focus.
 *
 * Everything shells out through `execFile` with an argument array. Nothing here builds a command
 * string from a model-supplied value.
 */

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { Tool, ToolContext } from './base'
import { schema, str, int, bool } from './base'

const exec = promisify(execFile)

async function powershell(script: string, ctx: ToolContext, timeoutMs = 60_000): Promise<string> {
  const { stdout } = await exec(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
    { timeout: timeoutMs, windowsHide: true, maxBuffer: 16 * 1024 * 1024, signal: ctx.signal, cwd: ctx.cwd }
  )
  return stdout
}

function resolve(cwd: string, p: string): string {
  return path.isAbsolute(p) ? path.normalize(p) : path.resolve(cwd, p)
}

/** Single-quoted PowerShell strings take a doubled quote as a literal one; nothing else escapes. */
function psLiteral(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

// ---------------------------------------------------------------- archives

const extractArchive: Tool = {
  name: 'extract_archive',
  description:
    'Extract a .zip, .tar, .tar.gz or .tgz archive. Extracts beside the archive unless a ' +
    'destination is given. Lists what came out.',
  tier: 'write',
  parameters: schema(
    { path: str('Archive to extract'), dest: str('Directory to extract into (default: alongside the archive)') },
    ['path']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.path))
    if (!fs.existsSync(file)) throw new Error(`${file} does not exist.`)

    const lower = file.toLowerCase()
    const dest = args.dest
      ? resolve(ctx.cwd, String(args.dest))
      : path.join(path.dirname(file), path.basename(file).replace(/\.(zip|tar|tar\.gz|tgz|gz)$/i, ''))
    await fsp.mkdir(dest, { recursive: true })

    if (lower.endsWith('.zip')) {
      // Expand-Archive resolves entry paths under the destination, so a crafted archive cannot
      // write outside it.
      await powershell(`Expand-Archive -LiteralPath ${psLiteral(file)} -DestinationPath ${psLiteral(dest)} -Force`, ctx)
    } else if (/\.(tar|tar\.gz|tgz)$/i.test(lower)) {
      // bsdtar ships with Windows 10 1803 and later.
      await exec('tar', ['-xf', file, '-C', dest], {
        windowsHide: true,
        timeout: ctx.timeoutMs,
        signal: ctx.signal,
        maxBuffer: 8 * 1024 * 1024
      })
    } else {
      throw new Error(`Unsupported archive type for ${path.basename(file)}. Handles .zip, .tar, .tar.gz and .tgz.`)
    }

    const entries = await fsp.readdir(dest).catch(() => [] as string[])
    const listed = entries.slice(0, 50).join('\n')
    const more = entries.length > 50 ? `\n[and ${entries.length - 50} more]` : ''
    return `Extracted to ${dest}\n\n${listed}${more}`
  }
}

const createArchive: Tool = {
  name: 'create_archive',
  description: 'Create a .zip from files or directories.',
  tier: 'write',
  parameters: schema(
    {
      paths: { type: 'array', items: { type: 'string' }, description: 'Files and directories to include' },
      dest: str('Path of the .zip to write')
    },
    ['paths', 'dest']
  ),
  async run(args, ctx) {
    const paths = (args.paths as string[] | undefined) ?? []
    if (!paths.length) throw new Error('Nothing to archive.')

    const dest = resolve(ctx.cwd, String(args.dest))
    const sources = paths.map((p) => psLiteral(resolve(ctx.cwd, p))).join(', ')
    await powershell(`Compress-Archive -LiteralPath ${sources} -DestinationPath ${psLiteral(dest)} -Force`, ctx)

    const size = await fsp.stat(dest).then((s) => s.size).catch(() => 0)
    return `Wrote ${dest} (${(size / 1024).toFixed(0)} KB) from ${paths.length} path(s).`
  }
}

// ---------------------------------------------------------------- windows

interface WindowRow {
  Id: number
  Name: string
  Title: string
  Handle: number
}

const listWindows: Tool = {
  name: 'list_windows',
  description:
    'List the open application windows with their titles and process ids. Use this before ' +
    'clicking or typing so you know what is on screen and what has focus, rather than guessing ' +
    'at coordinates from a screenshot alone.',
  tier: 'read',
  parameters: schema({ filter: str('Only windows whose title or process name contains this') }),
  async run(args, ctx) {
    const out = await powershell(
      `Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } | ` +
        `Select-Object Id, @{n='Name';e={$_.ProcessName}}, @{n='Title';e={$_.MainWindowTitle}}, ` +
        `@{n='Handle';e={[int64]$_.MainWindowHandle}} | ConvertTo-Json -Compress`,
      ctx
    )

    let rows: WindowRow[] = []
    try {
      const parsed = JSON.parse(out.trim() || '[]')
      rows = Array.isArray(parsed) ? parsed : [parsed]
    } catch {
      return out.trim() || 'No windows found.'
    }

    const needle = args.filter ? String(args.filter).toLowerCase() : null
    const filtered = needle
      ? rows.filter((r) => `${r.Title} ${r.Name}`.toLowerCase().includes(needle))
      : rows

    if (!filtered.length) return needle ? `No window matches "${args.filter}".` : 'No windows with titles found.'
    return filtered.map((r) => `pid ${String(r.Id).padStart(6)}  ${r.Name.padEnd(20)} ${r.Title}`).join('\n')
  }
}

const focusWindow: Tool = {
  name: 'focus_window',
  description:
    'Bring a window to the front by process id, so that click_mouse and type_text act on the ' +
    'right application. Restores it first if it is minimised. Execute-tier: it changes what the ' +
    'user is looking at and where their keystrokes would go.',
  tier: 'execute',
  parameters: schema({ pid: int('Process id from list_windows') }, ['pid']),
  async run(args, ctx) {
    const pid = Number(args.pid)
    if (!Number.isInteger(pid) || pid <= 0) throw new Error('pid must be a process id from list_windows.')

    const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
}
'@
$p = Get-Process -Id ${pid} -ErrorAction Stop
$h = $p.MainWindowHandle
if ($h -eq 0) { Write-Output 'no-window'; exit }
if ([W]::IsIconic($h)) { [W]::ShowWindow($h, 9) | Out-Null }
[W]::SetForegroundWindow($h) | Out-Null
Write-Output $p.MainWindowTitle`

    const out = (await powershell(script, ctx, 20_000)).trim()
    if (out === 'no-window') return `Process ${pid} has no visible main window.`
    return `Focused: ${out || `pid ${pid}`}`
  }
}

const windowBounds: Tool = {
  name: 'window_bounds',
  description:
    'Screen rectangle of a window, so a click can be aimed relative to it rather than to the ' +
    'whole desktop. Returns left, top, width and height in pixels.',
  tier: 'read',
  parameters: schema({ pid: int('Process id from list_windows') }, ['pid']),
  async run(args, ctx) {
    const pid = Number(args.pid)
    if (!Number.isInteger(pid) || pid <= 0) throw new Error('pid must be a process id from list_windows.')

    const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public struct RECT { public int Left, Top, Right, Bottom; }
public class B {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
}
'@
$p = Get-Process -Id ${pid} -ErrorAction Stop
$r = New-Object RECT
if ([B]::GetWindowRect($p.MainWindowHandle, [ref]$r)) {
  [pscustomobject]@{ left=$r.Left; top=$r.Top; width=($r.Right-$r.Left); height=($r.Bottom-$r.Top) } | ConvertTo-Json -Compress
} else { Write-Output 'failed' }`

    const out = (await powershell(script, ctx, 20_000)).trim()
    if (!out || out === 'failed') return `Could not read the bounds of pid ${pid}.`
    return out
  }
}

export const desktopTools: Tool[] = [extractArchive, createArchive, listWindows, focusWindow, windowBounds]
