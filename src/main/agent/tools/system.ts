/**
 * OS control tools: screenshots, clipboard, notifications, process inspection, input automation.
 *
 * Implemented through Electron APIs and PowerShell rather than native modules, so nothing has
 * to be compiled at install time. Input automation is the highest-blast-radius tool in the app
 * — it is execute-tier and therefore always passes through the approval gate.
 */

import { clipboard, desktopCapturer, Notification, screen } from 'electron'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import type { Tool } from './base'
import { schema, str, int } from './base'
import { TOOL_OUTPUT_DIR } from '../../storage/paths'

const exec = promisify(execFile)

async function powershell(script: string, timeoutMs = 20000): Promise<string> {
  const { stdout } = await exec(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
    { timeout: timeoutMs, windowsHide: true, maxBuffer: 16 * 1024 * 1024 }
  )
  return stdout
}

const takeScreenshot: Tool = {
  name: 'screenshot',
  description:
    'Capture the screen and save it to a PNG file. Returns the path. Use read_image or attach ' +
    'the file to look at it with a vision-capable model.',
  tier: 'read',
  parameters: schema({ display: int('Display index (default 0 = primary)') }),
  async run(args) {
    const displayIndex = Number(args.display ?? 0)
    const displays = screen.getAllDisplays()
    const target = displays[displayIndex] ?? displays[0]
    const { width, height } = target.size

    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width, height }
    })
    const source = sources[displayIndex] ?? sources[0]
    if (!source) throw new Error('No screen source available')

    fs.mkdirSync(TOOL_OUTPUT_DIR, { recursive: true })
    const file = path.join(TOOL_OUTPUT_DIR, `screen-${crypto.randomBytes(4).toString('hex')}.png`)
    fs.writeFileSync(file, source.thumbnail.toPNG())
    return `Screenshot saved to ${file} (${width}x${height}, display "${source.name}")`
  }
}

const readClipboard: Tool = {
  name: 'read_clipboard',
  description: 'Read the current text contents of the clipboard.',
  tier: 'read',
  parameters: schema({}),
  async run() {
    const text = clipboard.readText()
    return text ? text : '(clipboard is empty or does not contain text)'
  }
}

const writeClipboard: Tool = {
  name: 'write_clipboard',
  description: 'Replace the clipboard contents with text.',
  tier: 'write',
  parameters: schema({ text: str('Text to place on the clipboard') }, ['text']),
  async run(args) {
    clipboard.writeText(String(args.text))
    return `Clipboard set (${Buffer.byteLength(String(args.text))} bytes).`
  }
}

const notify: Tool = {
  name: 'notify',
  description: 'Show a desktop notification. Useful for telling the user a long job finished.',
  tier: 'write',
  parameters: schema({ title: str('Notification title'), body: str('Notification body') }, ['title', 'body']),
  async run(args) {
    if (!Notification.isSupported()) return 'Notifications are not supported on this system.'
    new Notification({ title: String(args.title), body: String(args.body) }).show()
    return 'Notification shown.'
  }
}

const listProcesses: Tool = {
  name: 'list_processes',
  description: 'List running processes with CPU and memory usage, highest memory first.',
  tier: 'read',
  parameters: schema({ filter: str('Only show processes whose name matches this text'), top: int('How many to return (default 30)') }),
  async run(args) {
    const top = Number(args.top ?? 30)
    const filter = args.filter ? String(args.filter) : null
    const script = `Get-Process | ${filter ? `Where-Object { $_.ProcessName -like '*${filter.replace(/'/g, "''")}*' } | ` : ''}Sort-Object WorkingSet64 -Descending | Select-Object -First ${top} Id, ProcessName, @{n='MemoryMB';e={[math]::Round($_.WorkingSet64/1MB,1)}}, @{n='CPU';e={[math]::Round($_.CPU,1)}} | ConvertTo-Json -Compress`
    const out = await powershell(script)
    try {
      const parsed = JSON.parse(out.trim() || '[]')
      const list = Array.isArray(parsed) ? parsed : [parsed]
      return list
        .map((p: { Id: number; ProcessName: string; MemoryMB: number; CPU: number }) =>
          `${String(p.Id).padStart(7)}  ${String(p.MemoryMB).padStart(8)} MB  ${String(p.CPU ?? 0).padStart(7)}s  ${p.ProcessName}`
        )
        .join('\n')
    } catch {
      return out.trim() || 'No processes matched.'
    }
  }
}

const installedApps: Tool = {
  name: 'list_installed_apps',
  description: 'List installed applications from the Windows uninstall registry keys.',
  tier: 'read',
  parameters: schema({ filter: str('Only show apps whose name matches this text') }),
  async run(args) {
    const filter = args.filter ? String(args.filter).replace(/'/g, "''") : null
    const script = `
$paths = @(
  'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*',
  'HKLM:\\SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*',
  'HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*'
)
Get-ItemProperty $paths -ErrorAction SilentlyContinue |
  Where-Object { $_.DisplayName ${filter ? `-like '*${filter}*'` : '-ne $null'} } |
  Select-Object DisplayName, DisplayVersion, Publisher |
  Sort-Object DisplayName -Unique | ConvertTo-Json -Compress`
    const out = await powershell(script, 30000)
    try {
      const parsed = JSON.parse(out.trim() || '[]')
      const list = Array.isArray(parsed) ? parsed : [parsed]
      return list.map((a: { DisplayName: string; DisplayVersion?: string }) => `${a.DisplayName}  ${a.DisplayVersion ?? ''}`).join('\n')
    } catch {
      return out.trim() || 'No applications matched.'
    }
  }
}

const readRegistry: Tool = {
  name: 'read_registry',
  description: 'Read a Windows registry key and its values. Read-only; writing the registry is not exposed.',
  tier: 'read',
  parameters: schema({ key: str("Registry path, e.g. 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'") }, ['key']),
  async run(args) {
    const key = String(args.key).replace(/'/g, "''")
    const out = await powershell(`Get-ItemProperty -Path '${key}' | ConvertTo-Json -Compress -Depth 3`)
    return out.trim() || '(no values)'
  }
}

const clickMouse: Tool = {
  name: 'click_mouse',
  description:
    'Move the mouse to screen coordinates and click. Controls the real desktop — the user sees ' +
    'this happen. Take a screenshot first to find the coordinates.',
  tier: 'execute',
  parameters: schema({ x: int('Screen X'), y: int('Screen Y'), button: str("'left' or 'right' (default left)") }, ['x', 'y']),
  async run(args) {
    const x = Number(args.x)
    const y = Number(args.y)
    const right = String(args.button ?? 'left') === 'right'
    // mouse_event flags: LEFTDOWN 0x02 LEFTUP 0x04 RIGHTDOWN 0x08 RIGHTUP 0x10
    const down = right ? 0x08 : 0x02
    const up = right ? 0x10 : 0x04
    const script = `
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class M {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint x, uint y, uint d, IntPtr e);
}
'@
[M]::SetCursorPos(${x}, ${y})
Start-Sleep -Milliseconds 60
[M]::mouse_event(${down},0,0,0,[IntPtr]::Zero)
[M]::mouse_event(${up},0,0,0,[IntPtr]::Zero)
'clicked'`
    await powershell(script)
    return `Clicked ${right ? 'right' : 'left'} at (${x}, ${y}).`
  }
}

const typeText: Tool = {
  name: 'type_text',
  description:
    'Type text into whatever window currently has focus. Controls the real keyboard. Click the ' +
    'target field first.',
  tier: 'execute',
  parameters: schema({ text: str('Text to type') }, ['text']),
  async run(args) {
    const text = String(args.text)
    // SendKeys treats these as control characters, so they must be brace-escaped.
    const escaped = text.replace(/([+^%~(){}[\]])/g, '{$1}').replace(/'/g, "''")
    await powershell(
      `Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${escaped}'); 'typed'`
    )
    return `Typed ${text.length} characters into the focused window.`
  }
}

const pressKey: Tool = {
  name: 'press_key',
  description: "Press a key or combination, e.g. 'ENTER', 'TAB', '^c' (ctrl+c), '%{F4}' (alt+F4).",
  tier: 'execute',
  parameters: schema({ keys: str('SendKeys-format key sequence') }, ['keys']),
  async run(args) {
    const keys = String(args.keys).replace(/'/g, "''")
    await powershell(`Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${keys}'); 'sent'`)
    return `Sent keys: ${args.keys}`
  }
}

export const systemTools: Tool[] = [
  takeScreenshot,
  readClipboard,
  writeClipboard,
  notify,
  listProcesses,
  installedApps,
  readRegistry,
  clickMouse,
  typeText,
  pressKey
]
