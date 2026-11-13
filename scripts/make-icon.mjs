#!/usr/bin/env node
/**
 * Generate the app's icons from its mark: a speech bubble cut into three layers.
 *
 * The bubble is the app's purpose, talking to a model. The three slices are the model's layers and
 * the memory they take, in the colours the load sheet draws a card's memory in: blue for weights,
 * cyan for the KV cache, violet for working memory. It replaces a four-point sparkle, the symbol
 * every AI product uses, and a stock "layers" glyph that matched nothing else the app shows.
 *
 * Drawn as SVG and rasterised by Electron, the same renderer the app uses, offscreen with nothing
 * shown. Sizes up to 48 px are laid out on whole pixels, so the gaps between the slices stay sharp
 * at 16 px instead of blurring into one gradient.
 *
 *   node scripts/make-icon.mjs [--preview out.png]
 *
 * Writes build/icon.ico (the exe, the portable launcher and its shortcut stub), build/icon.png (256),
 * build/tray.png, src/main/tray-icon.ts (the tray at every display scale) and
 * src/renderer/src/assets/brand-mark.svg (the sidebar). All of them are committed, so a build never
 * depends on having run this.
 */

import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const THIS = fileURLToPath(import.meta.url)
const ROOT = path.resolve(path.dirname(THIS), '..')

// ---------------------------------------------------------------- the mark

/** Each layer, top to bottom: the load sheet's weights, KV cache and working-memory colours, lit from above. */
const LAYERS = [
  ['#5aa0ff', '#2f78f0'],
  ['#6fd6e8', '#36aec6'],
  ['#b1a1ff', '#8a74ef']
]

/**
 * The mark at `size` pixels, as SVG in pixel units.
 *
 * Designed on a 256 grid: a bubble 212 wide and 150 tall with 40-unit corners, three equal slices
 * with 9-unit gaps, and a tail from the lower left. Small sizes round every edge to a whole pixel
 * and keep each gap at least one pixel wide.
 */
export function markSvg(size) {
  const k = size / 256
  const small = size <= 48
  const snap = (v) => (small ? Math.round(v) : v)
  const x0 = snap(22 * k)
  const x1 = snap(234 * k)
  const y0 = snap(36 * k)
  const y1 = snap(186 * k)
  const gap = small ? Math.max(1, Math.round(9 * k)) : 9 * k
  const radius = Math.max(small ? 2 : 0, snap(40 * k))

  // Three equal slices; whatever does not divide evenly goes to the top one.
  const room = y1 - y0 - 2 * gap
  const band = small ? Math.floor(room / 3) : room / 3
  const heights = [room - 2 * band, band, band]
  const tops = [y0, y0 + heights[0] + gap, y0 + heights[0] + heights[1] + 2 * gap]

  // The tail, from the bottom slice's lower left, in the 256 grid then scaled.
  const yb = y1 / k
  const t = (x, y) => `${(x * k).toFixed(2)} ${(y * k).toFixed(2)}`
  const tail =
    `M${t(46, yb - 30)} C${t(50, yb)} ${t(66, yb)} ${t(104, yb)} ` +
    `C${t(88, yb + 20)} ${t(66, yb + 36)} ${t(40, yb + 44)} ` +
    `C${t(32, yb + 46)} ${t(28, yb + 38)} ${t(32, yb + 32)} ` +
    `C${t(42, yb + 18)} ${t(46, yb - 4)} ${t(46, yb - 30)} Z`

  let defs = `<clipPath id="bubble"><rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="${radius}"/><path d="${tail}"/></clipPath>`
  let slices = ''
  LAYERS.forEach(([light, deep], i) => {
    const top = tops[i]
    // The bottom slice runs on down through the tail.
    const bottom = i === 2 ? size : top + heights[i]
    defs += `<linearGradient id="l${i}" x1="0" y1="${top}" x2="0" y2="${i === 2 ? y1 + 44 * k : bottom}" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="${light}"/><stop offset="1" stop-color="${deep}"/></linearGradient>`
    slices += `<rect x="0" y="${top}" width="${size}" height="${bottom - top}" fill="url(#l${i})"/>`
    // A thread of light along each slice's top edge, where it is large enough to see.
    if (!small) slices += `<rect x="0" y="${top}" width="${size}" height="${Math.max(1, 2 * k)}" fill="#fff" opacity="0.22"/>`
  })
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">` +
    `<defs>${defs}</defs><g clip-path="url(#bubble)">${slices}</g></svg>`
  )
}

// ---------------------------------------------------------------- run inside Electron

/** Electron's nativeImage, once running inside it. */
let native = null

if (!process.versions.electron) {
  // Plain Node: the electron package is the path to its binary. Run this same file inside it.
  const result = spawnSync(createRequire(import.meta.url)('electron'), [THIS, ...process.argv.slice(2)], {
    stdio: 'inherit',
    windowsHide: true
  })
  process.exit(result.status ?? 1)
} else {
  // Inside Electron, importing 'electron' is the built-in API. A userland require would find the
  // npm package instead, which only knows where the binary is.
  const { app, BrowserWindow, nativeImage } = await import('electron')
  native = nativeImage
  app.commandLine.appendSwitch('force-device-scale-factor', '1')
  app.disableHardwareAcceleration()
  app.whenReady().then(() =>
    generate(app, BrowserWindow).catch((err) => {
      console.error(err)
      app.exit(1)
    })
  )
}

// ---------------------------------------------------------------- rasterise

const ICON_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
const TRAY_SCALES = [
  [1, 16],
  [1.25, 20],
  [1.5, 24],
  [2, 32]
]

/** Every size drawn once, side by side on a transparent page, and cut apart. */
async function rasterise(BrowserWindow, sizes) {
  const unique = [...new Set(sizes)]
  let x = 0
  const placed = unique.map((s) => {
    const at = { size: s, x, y: 0 }
    x += s + 8
    return at
  })
  const width = x
  const height = Math.max(...unique)
  const html =
    '<html><body style="margin:0;background:transparent">' +
    placed
      .map(
        (p) =>
          `<img style="position:absolute;left:${p.x}px;top:0;width:${p.size}px;height:${p.size}px" src="data:image/svg+xml;base64,${Buffer.from(markSvg(p.size)).toString('base64')}">`
      )
      .join('') +
    '</body></html>'
  const win = new BrowserWindow({ show: false, width, height, transparent: true, frame: false, webPreferences: { offscreen: true } })
  await win.loadURL(`data:text/html;base64,${Buffer.from(html).toString('base64')}`)
  await new Promise((r) => setTimeout(r, 500))
  // The window stays until the app quits: destroying it while its capture was still in use crashed Electron.
  const page = await win.webContents.capturePage()
  /*
   * Cut apart by hand from one read of the whole page. Cropping the captured image and reading the
   * crop back crashed Electron outright, with no error to show for it.
   */
  const pageWidth = page.getSize().width
  const all = page.toBitmap()
  const out = new Map()
  for (const p of placed) {
    const bgra = Buffer.alloc(p.size * p.size * 4)
    for (let row = 0; row < p.size; row++) {
      const from = ((p.y + row) * pageWidth + p.x) * 4
      all.copy(bgra, row * p.size * 4, from, from + p.size * 4)
    }
    const png = native.createFromBitmap(bgra, { width: p.size, height: p.size }).toPNG()
    out.set(p.size, { png, rgba: straightRgba(bgra, p.size) })
  }
  return out
}

/**
 * The capture's pixels as straight RGBA.
 *
 * Chromium hands back BGRA with the colour multiplied by alpha. An icon bitmap wants it divided
 * back out, or every anti-aliased edge darkens into a grey fringe.
 */
function straightRgba(bgra, size) {
  const out = Buffer.alloc(size * size * 4)
  for (let i = 0; i < size * size * 4; i += 4) {
    const a = bgra[i + 3]
    const un = (c) => (a === 0 ? 0 : Math.min(255, Math.round((c * 255) / a)))
    out[i] = un(bgra[i + 2])
    out[i + 1] = un(bgra[i + 1])
    out[i + 2] = un(bgra[i])
    out[i + 3] = a
  }
  return out
}

// ---------------------------------------------------------------- ICO

/**
 * A 32-bit DIB icon entry, for the classic sizes some of the shell still draws from a bitmap.
 * The header claims double the height because the colour rows are followed by a 1bpp mask.
 */
function dib(size, rgba) {
  const header = Buffer.alloc(40)
  header.writeUInt32LE(40, 0)
  header.writeInt32LE(size, 4)
  header.writeInt32LE(size * 2, 8)
  header.writeUInt16LE(1, 12)
  header.writeUInt16LE(32, 14)
  header.writeUInt32LE(0, 16)
  header.writeUInt32LE(size * size * 4, 20)
  const colour = Buffer.alloc(size * size * 4)
  for (let y = 0; y < size; y++) {
    const src = (size - 1 - y) * size * 4
    for (let x = 0; x < size; x++) {
      const s = src + x * 4
      const d = (y * size + x) * 4
      colour[d] = rgba[s + 2]
      colour[d + 1] = rgba[s + 1]
      colour[d + 2] = rgba[s]
      colour[d + 3] = rgba[s + 3]
    }
  }
  const mask = Buffer.alloc(Math.ceil(size / 32) * 4 * size)
  return Buffer.concat([header, colour, mask])
}

function ico(entries) {
  const dir = Buffer.alloc(6)
  dir.writeUInt16LE(1, 2)
  dir.writeUInt16LE(entries.length, 4)
  let offset = 6 + entries.length * 16
  const table = entries.map((e) => {
    const row = Buffer.alloc(16)
    row[0] = e.size >= 256 ? 0 : e.size
    row[1] = e.size >= 256 ? 0 : e.size
    row.writeUInt16LE(1, 4)
    row.writeUInt16LE(32, 6)
    row.writeUInt32LE(e.data.length, 8)
    row.writeUInt32LE(offset, 12)
    offset += e.data.length
    return row
  })
  return Buffer.concat([dir, ...table, ...entries.map((e) => e.data)])
}

// ---------------------------------------------------------------- write

async function generate(app, BrowserWindow) {
  // Nothing above 256: building a 512 px image from its pixels crashed Electron 38 outright.
  const rendered = await rasterise(BrowserWindow, [...ICON_SIZES, ...TRAY_SCALES.map(([, s]) => s)])

  const entries = ICON_SIZES.map((size) => {
    const r = rendered.get(size)
    return { size, data: size <= 48 ? dib(size, r.rgba) : r.png }
  })
  const build = path.join(ROOT, 'build')
  fs.mkdirSync(build, { recursive: true })
  fs.writeFileSync(path.join(build, 'icon.ico'), ico(entries))
  fs.writeFileSync(path.join(build, 'icon.png'), rendered.get(256).png)
  fs.writeFileSync(path.join(build, 'tray.png'), rendered.get(32).png)
  fs.rmSync(path.join(build, 'tray.dataurl.txt'), { force: true })

  // The tray, inlined so it needs nothing from the package layout, at each scale Windows asks for.
  const tray = TRAY_SCALES.map(
    ([scale, size]) => `  [${scale}, 'data:image/png;base64,${rendered.get(size).png.toString('base64')}']`
  ).join(',\n')
  fs.writeFileSync(
    path.join(ROOT, 'src', 'main', 'tray-icon.ts'),
    `/**\n * The tray icon at each display scale. Generated by scripts/make-icon.mjs; do not edit.\n */\nexport const TRAY_ICON: [number, string][] = [\n${tray}\n]\n`
  )

  const assets = path.join(ROOT, 'src', 'renderer', 'src', 'assets')
  fs.mkdirSync(assets, { recursive: true })
  fs.writeFileSync(path.join(assets, 'brand-mark.svg'), markSvg(256))

  const preview = process.argv.indexOf('--preview')
  if (preview !== -1 && process.argv[preview + 1]) await writePreview(BrowserWindow, process.argv[preview + 1])

  console.log(`icon.ico   ${entries.length} sizes (${ICON_SIZES.join(', ')}), ${fs.statSync(path.join(build, 'icon.ico')).size} bytes`)
  console.log('icon.png   256x256')
  console.log(`tray       ${TRAY_SCALES.map(([s, px]) => `${px}px @${s}x`).join(', ')} -> src/main/tray-icon.ts`)
  console.log('sidebar    src/renderer/src/assets/brand-mark.svg')
  app.quit()
}

/** The mark at every size on a dark and a light taskbar, for looking at. */
async function writePreview(BrowserWindow, file) {
  const sizes = [256, 64, 48, 32, 24, 20, 16]
  let html = '<html><body style="margin:0">'
  ;['#202020', '#f3f3f3'].forEach((bg, row) => {
    html += `<div style="position:absolute;left:0;top:${row * 300}px;width:900px;height:300px;background:${bg}"></div>`
    let x = 24
    for (const s of sizes) {
      html += `<img style="position:absolute;left:${x}px;top:${row * 300 + 22 + (256 - s) / 2}px;width:${s}px;height:${s}px" src="data:image/svg+xml;base64,${Buffer.from(markSvg(s)).toString('base64')}">`
      x += s + 36
    }
  })
  html += '</body></html>'
  const win = new BrowserWindow({ show: false, width: 900, height: 600, webPreferences: { offscreen: true } })
  await win.loadURL(`data:text/html;base64,${Buffer.from(html).toString('base64')}`)
  await new Promise((r) => setTimeout(r, 500))
  fs.writeFileSync(file, (await win.webContents.capturePage()).toPNG())
}

