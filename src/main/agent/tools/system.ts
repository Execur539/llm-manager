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
import type { ContentPart } from '../../runtime/llama'
import { recordCapture } from './screen-space'

/** Scale a capture down to something a projector can actually resolve, then package it. */
function screenPart(image: Electron.NativeImage): ContentPart {
  const { width, height } = image.getSize()
  const MAX_EDGE = 1568
  const sized =
    Math.max(width, height) > MAX_EDGE
      ? image.resize(width >= height ? { width: MAX_EDGE } : { height: MAX_EDGE })
      : image
  return { type: 'image_url', image_url: { url: `data:image/png;base64,${sized.toPNG().toString('base64')}` } }
}

const exec = promisify(execFile)

/**
 * Run a PowerShell snippet.
 *
 * Takes the turn's abort signal so stop actually stops. Every tool in this file went through
 * here without one, which meant hitting stop left a thirty-second registry sweep running — and,
 * worse, left a `type_text` or `press_key` still driving the real keyboard after the user had
 * asked for the turn to end.
 */
async function powershell(script: string, timeoutMs = 20000, signal?: AbortSignal): Promise<string> {
  const { stdout } = await exec(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
    { timeout: timeoutMs, windowsHide: true, maxBuffer: 16 * 1024 * 1024, signal }
  )
  return stdout
}

/** The displays, primary first, each with the rectangle it occupies in real screen pixels. */
function displayList(): { display: Electron.Display; rect: Electron.Rectangle }[] {
  const primary = screen.getPrimaryDisplay().id
  return screen
    .getAllDisplays()
    .sort((x, y) => Number(y.id === primary) - Number(x.id === primary))
    .map((display) => ({
      display,
      // Electron measures in scaled units; the rest of the desktop tools work in real pixels.
      rect: process.platform === 'win32' ? screen.dipToScreenRect(null, display.bounds) : display.bounds
    }))
}

const takeScreenshot: Tool = {
  name: 'screenshot',
  description:
    'Capture the screen, or part of it. On a model with vision the picture is shown to you ' +
    'directly; the file is also saved and its path returned. A whole display is shrunk to fit, so ' +
    'small text will not be readable: to read it, capture just that area by giving x, y, width and ' +
    'height in screen pixels. To act on what you see, prefer ui_inspect and ui_click; to point at ' +
    'something in the picture with the mouse tools, pass them coords: "image".',
  tier: 'read',
  parameters: schema({
    display: int('Display number (default 0, the primary display)'),
    x: int('Left edge of the area to capture, in screen pixels'),
    y: int('Top edge of the area to capture, in screen pixels'),
    width: int('Width of the area to capture'),
    height: int('Height of the area to capture')
  }),
  async run(args, ctx) {
    const displays = displayList()
    const region = ['x', 'y', 'width', 'height'].every((k) => Number.isFinite(Number(args[k] ?? NaN)))
      ? { x: Number(args.x), y: Number(args.y), width: Number(args.width), height: Number(args.height) }
      : null

    // An area names its own display: whichever one its middle is on.
    const onDisplay = region
      ? displays.findIndex(
          ({ rect }) =>
            region.x + region.width / 2 >= rect.x &&
            region.x + region.width / 2 < rect.x + rect.width &&
            region.y + region.height / 2 >= rect.y &&
            region.y + region.height / 2 < rect.y + rect.height
        )
      : -1
    const index = onDisplay >= 0 ? onDisplay : Math.max(0, Math.min(displays.length - 1, Number(args.display ?? 0) || 0))
    const { display, rect } = displays[index]

    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      // At the display's real size. Asking for its scaled size threw away half the pixels of a
      // display set to 200% before anything had looked at them.
      thumbnailSize: { width: rect.width, height: rect.height }
    })
    const source = sources.find((s) => s.display_id === String(display.id)) ?? sources[index] ?? sources[0]
    if (!source) throw new Error('No screen source available')

    let image = source.thumbnail
    const full = image.getSize()
    // What the capture actually covers, in case the system handed back something smaller.
    const perPixel = rect.width / full.width
    let left = rect.x
    let top = rect.y
    if (region) {
      const cx = Math.max(0, Math.round((region.x - rect.x) / perPixel))
      const cy = Math.max(0, Math.round((region.y - rect.y) / perPixel))
      const cw = Math.min(full.width - cx, Math.round(region.width / perPixel))
      const ch = Math.min(full.height - cy, Math.round(region.height / perPixel))
      if (cw < 8 || ch < 8) {
        return `That area is not on display ${index}, which covers ${rect.width}x${rect.height} pixels from (${rect.x}, ${rect.y}).`
      }
      image = image.crop({ x: cx, y: cy, width: cw, height: ch })
      left = rect.x + cx * perPixel
      top = rect.y + cy * perPixel
    }

    fs.mkdirSync(TOOL_OUTPUT_DIR, { recursive: true })
    const file = path.join(TOOL_OUTPUT_DIR, `screen-${crypto.randomBytes(4).toString('hex')}.png`)
    fs.writeFileSync(file, image.toPNG())

    const captured = image.getSize()
    const part = screenPart(image)
    const shownWidth = Math.min(captured.width, captured.width >= captured.height ? 1568 : Math.round((1568 * captured.width) / captured.height))
    const shown = { width: shownWidth, height: Math.round((shownWidth * captured.height) / captured.width) }
    recordCapture({ left, top, scale: (captured.width * perPixel) / shown.width, width: shown.width, height: shown.height })

    const others = displays.length > 1
      ? ` Displays: ${displays.map(({ rect: r }, i) => `${i} is ${r.width}x${r.height} at (${r.x}, ${r.y})`).join('; ')}.`
      : ''
    const text =
      `Screenshot of ${region ? `the ${Math.round(captured.width * perPixel)}x${Math.round(captured.height * perPixel)} area at (${Math.round(left)}, ${Math.round(top)}) on ` : ''}display ${index}` +
      `${region ? '' : `, which is ${rect.width}x${rect.height} pixels at (${rect.x}, ${rect.y})`}, saved to ${file}.` +
      others
    // Handing back the path alone was the whole problem: nothing could open it. When the model
    // can see, show it; when it cannot, say so rather than pretending the path is useful.
    if (!ctx.vision) return `${text} The loaded model has no vision projector, so it cannot be shown.`
    return {
      text:
        `${text} Shown to you at ${shown.width}x${shown.height}. A position in this picture is not a screen position: ` +
        'give the mouse tools the position as you see it here with coords: "image" and they convert it.' +
        (region ? '' : ' To read small text, capture just that area with x, y, width and height.'),
      media: [part]
    }
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
  async run(args, ctx) {
    const top = Number(args.top ?? 30)
    const filter = args.filter ? String(args.filter) : null
    const script = `Get-Process | ${filter ? `Where-Object { $_.ProcessName -like '*${filter.replace(/'/g, "''")}*' } | ` : ''}Sort-Object WorkingSet64 -Descending | Select-Object -First ${top} Id, ProcessName, @{n='MemoryMB';e={[math]::Round($_.WorkingSet64/1MB,1)}}, @{n='CPU';e={[math]::Round($_.CPU,1)}} | ConvertTo-Json -Compress`
    const out = await powershell(script, 20000, ctx.signal)
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
  async run(args, ctx) {
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
    const out = await powershell(script, 30000, ctx.signal)
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
  async run(args, ctx) {
    const key = String(args.key).replace(/'/g, "''")
    const out = await powershell(`Get-ItemProperty -Path '${key}' | ConvertTo-Json -Compress -Depth 3`, 20000, ctx.signal)
    return out.trim() || '(no values)'
  }
}

const typeText: Tool = {
  name: 'type_text',
  description:
    'Type text into whatever window currently has focus. Controls the real keyboard. Click the ' +
    'target field first.',
  tier: 'execute',
  parameters: schema({ text: str('Text to type') }, ['text']),
  async run(args, ctx) {
    const text = String(args.text)
    // SendKeys treats these as control characters, so they must be brace-escaped.
    const escaped = text.replace(/([+^%~(){}[\]])/g, '{$1}').replace(/'/g, "''")
    await powershell(
      `Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${escaped}'); 'typed'`,
      20000,
      ctx.signal
    )
    return `Typed ${text.length} characters into the focused window.`
  }
}

const pressKey: Tool = {
  name: 'press_key',
  description: "Press a key or combination, e.g. 'ENTER', 'TAB', '^c' (ctrl+c), '%{F4}' (alt+F4).",
  tier: 'execute',
  parameters: schema({ keys: str('SendKeys-format key sequence') }, ['keys']),
  async run(args, ctx) {
    const keys = String(args.keys).replace(/'/g, "''")
    await powershell(
      `Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${keys}'); 'sent'`,
      20000,
      ctx.signal
    )
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
  typeText,
  pressKey
]
