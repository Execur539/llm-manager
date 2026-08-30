/**
 * Generate the application icon set from the app's own brand mark.
 *
 * Everything shipped used a placeholder: the portable launcher had no icon at all (so it wore
 * NSIS's), the packaged exe had none either (so it wore Electron's), and the tray image was a
 * 1x1 transparent PNG — an advertised "minimise to tray" that put the window behind an invisible
 * pixel. All three are branding, and all three are the first thing anyone sees.
 *
 * Written by hand rather than pulled from a raster toolchain: the mark is eight straight edges
 * and a rounded square, which is less code to draw than a dependency is to justify, and it means
 * the icon is regenerated from the same numbers the UI draws from rather than from a binary
 * somebody has to remember to re-export.
 *
 *   node scripts/make-icon.mjs
 *
 * Outputs are committed, so a build never depends on having run this.
 */

import zlib from 'node:zlib'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// ---------------------------------------------------------------- the mark

/**
 * The sparkle from components/Icon.tsx, on its own 24-unit grid.
 *
 * Four points on the axes with a tight waist between them. Kept as the same coordinates the SVG
 * uses so the icon and the sidebar mark cannot drift apart.
 */
const STAR = [
  [12, 5],
  [13.8, 10.2],
  [19, 12],
  [13.8, 13.8],
  [12, 19],
  [10.2, 13.8],
  [5, 12],
  [10.2, 10.2]
]

/** styles.css --bg and --accent. */
const BG = [0x1f, 0x1e, 0x1d]
const ACCENT = [0xd9, 0x77, 0x57]

function pointInPolygon(x, y, poly) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Inside a square with rounded corners, in the same 0..size space as the pixels. */
function inRoundedSquare(x, y, size, radius) {
  const r = radius
  const cx = Math.min(Math.max(x, r), size - r)
  const cy = Math.min(Math.max(y, r), size - r)
  if (x >= 0 && y >= 0 && x <= size && y <= size) {
    const dx = x - cx
    const dy = y - cy
    return dx * dx + dy * dy <= r * r
  }
  return false
}

/**
 * Render one icon as RGBA.
 *
 * Coverage is supersampled rather than analytic — at 16px the waist of the star is barely three
 * pixels across, and an aliased edge there reads as a smudge rather than a shape.
 */
function render(size, { background }) {
  const SS = 4
  const out = Buffer.alloc(size * size * 4)

  /*
   * Sized from the mark's own extent, not its 24-unit box.
   *
   * The star spans 14 of those 24 units, so scaling the box to fill the icon leaves the visible
   * shape covering barely a third of it — inset within an inset, and it reads as lost in the
   * middle of the tile. A bare tray mark takes nearly the whole square; inside a tile it takes a
   * little over half, which is about where an app icon's glyph normally sits.
   */
  const EXTENT = 14
  const CENTRE = 12
  const target = background ? 0.56 : 0.9
  const scale = (size * target) / EXTENT
  const offset = size / 2 - CENTRE * scale
  const star = STAR.map(([x, y]) => [x * scale + offset, y * scale + offset])
  const radius = size * 0.22

  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let bgHits = 0
      let starHits = 0
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const x = px + (sx + 0.5) / SS
          const y = py + (sy + 0.5) / SS
          if (background && inRoundedSquare(x, y, size, radius)) bgHits++
          if (pointInPolygon(x, y, star)) starHits++
        }
      }
      const total = SS * SS
      const bgA = background ? bgHits / total : 0
      const starA = starHits / total

      // Star over tile, tile over nothing.
      const alpha = Math.min(1, bgA + starA)
      const i = (py * size + px) * 4
      if (alpha <= 0) continue
      const mix = alpha > 0 ? starA / alpha : 0
      for (let c = 0; c < 3; c++) {
        out[i + c] = Math.round(ACCENT[c] * mix + BG[c] * (1 - mix))
      }
      out[i + 3] = Math.round(alpha * 255)
    }
  }
  return out
}

// ---------------------------------------------------------------- PNG

const CRC_TABLE = (() => {
  const table = new Int32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c
  }
  return table
})()

function crc32(buf) {
  let c = 0xffffffff
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([len, body, crc])
}

function png(size, rgba) {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(size, 0)
  ihdr.writeUInt32BE(size, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 6 // truecolour with alpha
  const stride = size * 4
  const raw = Buffer.alloc((stride + 1) * size)
  for (let y = 0; y < size; y++) {
    raw[y * (stride + 1)] = 0 // filter: none
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride)
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0))
  ])
}

// ---------------------------------------------------------------- ICO

/**
 * A 32-bit DIB icon entry.
 *
 * Small sizes go in as DIB rather than PNG: Windows has accepted PNG entries since Vista, but
 * parts of the shell still render the classic sizes from the bitmap, and an icon that is correct
 * everywhere is worth forty lines. The header claims double the height because an icon DIB is
 * the colour rows followed by a 1bpp mask, and rows run bottom-up.
 */
function dib(size, rgba) {
  const header = Buffer.alloc(40)
  header.writeUInt32LE(40, 0)
  header.writeInt32LE(size, 4)
  header.writeInt32LE(size * 2, 8)
  header.writeUInt16LE(1, 12) // planes
  header.writeUInt16LE(32, 14) // bpp
  header.writeUInt32LE(0, 16) // BI_RGB
  header.writeUInt32LE(size * size * 4, 20)

  const colour = Buffer.alloc(size * size * 4)
  for (let y = 0; y < size; y++) {
    const src = (size - 1 - y) * size * 4
    for (let x = 0; x < size; x++) {
      const s = src + x * 4
      const d = (y * size + x) * 4
      colour[d] = rgba[s + 2] // B
      colour[d + 1] = rgba[s + 1] // G
      colour[d + 2] = rgba[s] // R
      colour[d + 3] = rgba[s + 3] // A
    }
  }

  // Alpha carries the transparency; the mask is present because the format requires it.
  const maskStride = Math.ceil(size / 8 / 4) * 4
  const mask = Buffer.alloc(maskStride * size)
  return Buffer.concat([header, colour, mask])
}

function ico(entries) {
  const dir = Buffer.alloc(6)
  dir.writeUInt16LE(0, 0)
  dir.writeUInt16LE(1, 2) // type: icon
  dir.writeUInt16LE(entries.length, 4)

  let offset = 6 + entries.length * 16
  const table = []
  for (const e of entries) {
    const row = Buffer.alloc(16)
    row[0] = e.size >= 256 ? 0 : e.size
    row[1] = e.size >= 256 ? 0 : e.size
    row[2] = 0 // palette
    row[3] = 0
    row.writeUInt16LE(1, 4) // planes
    row.writeUInt16LE(32, 6) // bpp
    row.writeUInt32LE(e.data.length, 8)
    row.writeUInt32LE(offset, 12)
    offset += e.data.length
    table.push(row)
  }
  return Buffer.concat([dir, ...table, ...entries.map((e) => e.data)])
}

// ---------------------------------------------------------------- output

const iconEntries = [
  ...[16, 24, 32, 48].map((size) => ({ size, data: dib(size, render(size, { background: true })) })),
  ...[64, 128, 256].map((size) => ({ size, data: png(size, render(size, { background: true })) }))
]

fs.mkdirSync(path.join(ROOT, 'build'), { recursive: true })
const icoPath = path.join(ROOT, 'build', 'icon.ico')
fs.writeFileSync(icoPath, ico(iconEntries))

// The tray image has no tile: a coral mark on transparency reads on a light or a dark taskbar,
// where a dark rounded square would disappear into one of them.
const trayPng = png(32, render(32, { background: false }))
const trayPath = path.join(ROOT, 'build', 'tray.png')
fs.writeFileSync(trayPath, trayPng)

// Inlined into main/index.ts as a data URL, so the tray needs nothing from the package layout.
const trayDataUrl = `data:image/png;base64,${trayPng.toString('base64')}`
fs.writeFileSync(path.join(ROOT, 'build', 'tray.dataurl.txt'), trayDataUrl)

console.log(`icon.ico   ${iconEntries.length} sizes, ${fs.statSync(icoPath).size} bytes`)
console.log(`tray.png   32x32, ${trayPng.length} bytes`)
console.log(`tray data URL written to build/tray.dataurl.txt (${trayDataUrl.length} chars)`)
