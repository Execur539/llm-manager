/**
 * End-to-end harness: launches the real app against a mock inference server, drives it, and
 * audits what it renders.
 *
 * The layout auditor is the part that earns its keep. Functional assertions catch behaviour that
 * is wrong; the auditor catches the long tail nobody writes an assertion for — text clipped by a
 * container, a control pushed off-screen, `undefined` rendered into a label, a button too small
 * to hit, two controls overlapping.
 */

import { _electron as electron } from 'playwright-core'
import { AsyncLocalStorage } from 'node:async_hooks'
import path from 'node:path'
import fsp from 'node:fs/promises'
import { envVars, ROOT } from './fixtures.mjs'

export const SHOTS_DIR = path.join(ROOT, 'scripts', 'e2e', 'screenshots')

// ---------------------------------------------------------------- issue collection

/**
 * Which scenario the currently-running async work belongs to.
 *
 * Scenarios are ordinary async functions that await their way through a browser session, so
 * there is no object to hang "who is running" off. AsyncLocalStorage follows the await chain,
 * which means a `console.log` twelve frames deep inside a helper still knows which scenario it
 * came from — without threading a label through every function that might print.
 */
export const scenarioContext = new AsyncLocalStorage()

/** The real console.log, for printing collected blocks without re-entering the capture. */
export const rawLog = console.log.bind(console)

let sink = null

/*
 * Everything a scenario prints is captured, not just assertions.
 *
 * Scenarios log directly in a dozen places — measured widths, chosen paths, the "=== label ==="
 * banner — and those lines are as much a part of a scenario's block as its checks. Capturing at
 * the console means none of those call sites had to change.
 */
console.log = (...parts) => {
  const key = sink && scenarioContext.getStore()
  if (key) sink(key, parts.map((p) => (typeof p === 'string' ? p : String(p))).join(' '))
  else rawLog(...parts)
}

export class Report {
  constructor() {
    this.issues = []
    this.passes = 0
    /*
     * Output is held per scenario and printed as a block when that scenario finishes.
     *
     * Scenarios run concurrently, and each one logs a line per assertion. Printed as they
     * happened, four interleaved scenarios produce a list of results with no way to tell which
     * belongs to which. Buffering costs nothing and keeps each block readable.
     */
    this.lines = new Map()
    this.buffered = false
  }

  /** Record a defect. `where` is the scenario or view it was found in. */
  issue(where, severity, message, detail) {
    this.issues.push({ where, severity, message, detail })
  }

  /** Collect a line against a scenario without going back through the captured console. */
  pushRaw(where, text) {
    if (!this.lines.has(where)) this.lines.set(where, [])
    this.lines.get(where).push(text)
  }

  line(where, text) {
    const key = scenarioContext.getStore() ?? where
    if (this.buffered) this.pushRaw(key, text)
    else rawLog(text)
  }

  /** Route captured console output into this report's per-scenario buffers. */
  capture(on) {
    this.buffered = on
    sink = on ? (key, text) => this.pushRaw(key, text) : null
  }

  ok(where, label) {
    this.passes++
    this.line(where, `    ok   ${label}`)
  }

  fail(where, label, detail) {
    this.issue(where, 'bug', label, detail)
    this.line(where, `    BUG  ${label}${detail ? ` — ${detail}` : ''}`)
  }

  check(where, label, condition, detail) {
    if (condition) this.ok(where, label)
    else this.fail(where, label, detail)
  }

  /**
   * Print everything a scenario recorded, as one block, once it is done.
   *
   * No header of its own: withApp already prints one, and a scenario that starts several apps
   * gets one per app, which is the grouping that is actually useful.
   */
  flush(where, note) {
    const lines = this.lines.get(where)
    this.lines.delete(where)
    if (!lines?.length) return
    for (const l of lines) rawLog(l)
    if (note) rawLog(note)
  }

  summary() {
    const bySeverity = this.issues.reduce((acc, i) => {
      acc[i.severity] = (acc[i.severity] ?? 0) + 1
      return acc
    }, {})
    return { passes: this.passes, total: this.issues.length, bySeverity, issues: this.issues }
  }
}

// ---------------------------------------------------------------- app lifecycle

export async function launchApp(env, extraEnv = {}) {
  const app = await electron.launch({
    args: [path.join(ROOT, 'out', 'main', 'index.js')],
    env: envVars(env, extraEnv),
    cwd: ROOT,
    timeout: 60000
  })

  const consoleErrors = []
  const page = await app.firstWindow({ timeout: 30000 })
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => consoleErrors.push(`uncaught: ${err.message}`))

  await page.waitForLoadState('domcontentloaded')
  await page.waitForSelector('.nav-item', { timeout: 20000 })

  // Native dialogs are stubbed the moment the app is up, defaulting to "cancelled".
  //
  // An unstubbed picker does not fail a scenario — it opens a real modal window that blocks the
  // main process until a human clicks it, hanging the suite and taking over the desktop. Every
  // scenario therefore starts from a state where dialogs answer instantly; the ones that need a
  // path call stubDialogs() again to override.
  await stubDialogs(app, {})

  return { app, page, consoleErrors }
}

/**
 * Replace Electron's native file dialogs for the rest of the run.
 *
 * Everything behind a file picker — importing a model, exporting a conversation, ingesting a
 * document — was untestable without this, which is exactly why those paths carried the most
 * bugs. The stub runs in the main process, so the handler under test is the real one; only the
 * user's click on a native dialog is simulated.
 *
 * `plan` maps a dialog kind to what the user "chose":
 *   { open: ['C:/a.gguf'], save: 'C:/out.md' }         -> those paths
 *   { open: null }                                     -> the user cancelled
 */
export async function stubDialogs(app, plan) {
  await app.evaluate(async ({ dialog }, p) => {
    dialog.showOpenDialog = async () =>
      p.open === null || p.open === undefined
        ? { canceled: true, filePaths: [] }
        : { canceled: false, filePaths: p.open }
    dialog.showSaveDialog = async () =>
      p.save === null || p.save === undefined
        ? { canceled: true, filePath: undefined }
        : { canceled: false, filePath: p.save }
    // Message boxes would block the run forever waiting for a click that never comes.
    dialog.showMessageBox = async () => ({ response: 0, checkboxChecked: false })
  }, plan)
}

/** Read the toasts currently on screen. */
export async function toastTexts(page) {
  return page.getByTestId('toast').allTextContents()
}

export async function closeApp(app) {
  await app.close().catch(() => undefined)
}

export const VIEWS = [
  'Dashboard',
  'Chat',
  'Agent',
  'Documents',
  'My models',
  'Find a model',
  'API server',
  'Remote access',
  'Settings'
]

/*
 * Navigate, waiting for the view to have actually swapped rather than for a fixed duration.
 *
 * This is called sixty times across the suite, and every one of those was a flat 700ms whether
 * the view took that long or not — three quarters of a minute of the run spent waiting on a
 * number somebody guessed. The nav item marks itself active on the same render that mounts the
 * new view, so that is the thing worth waiting for; the short settle after it is for the fetches
 * a view kicks off on mount.
 */
export async function goTo(page, label) {
  await page.locator('.nav-item', { hasText: new RegExp(`^${label}$`) }).first().click()
  await page
    .waitForFunction(
      (want) => document.querySelector('.nav-item.active')?.textContent?.trim() === want,
      label,
      { timeout: 5000 }
    )
    .catch(() => undefined)
  await page.waitForTimeout(250)
}

export async function shot(page, name) {
  await fsp.mkdir(SHOTS_DIR, { recursive: true })
  const file = path.join(SHOTS_DIR, `${name.replace(/[^a-z0-9-]+/gi, '-').toLowerCase()}.png`)
  await page.screenshot({ path: file, fullPage: false })
  return file
}

// ---------------------------------------------------------------- layout audit

/**
 * Runs in the page. Returns structural and visual defects.
 *
 * Deliberately conservative: every rule has a tolerance, because a rule that cries wolf gets
 * ignored and then the real defects hide among the noise.
 */
const AUDIT_FN = () => {
  const problems = []
  const describeShort = (node) => {
    if (!node) return '?'
    const cls =
      typeof node.className === 'string' && node.className.trim()
        ? `.${node.className.trim().split(/\s+/)[0]}`
        : ''
    return `${node.tagName.toLowerCase()}${cls}`
  }
  const add = (kind, detail, el) => {
    const describe = (node) => {
      if (!node) return '?'
      const id = node.id ? `#${node.id}` : ''
      const cls =
        typeof node.className === 'string' && node.className.trim()
          ? `.${node.className.trim().split(/\s+/).slice(0, 2).join('.')}`
          : ''
      const text = (node.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 40)
      return `${node.tagName.toLowerCase()}${id}${cls}${text ? ` "${text}"` : ''}`
    }
    problems.push({ kind, detail, element: describe(el) })
  }

  const vw = document.documentElement.clientWidth
  const vh = document.documentElement.clientHeight

  // 1. The page itself must never scroll sideways.
  if (document.documentElement.scrollWidth > vw + 1) {
    add('page-h-scroll', `documentElement scrollWidth ${document.documentElement.scrollWidth} > ${vw}`, document.body)
  }

  const all = [...document.querySelectorAll('body *')]

  for (const el of all) {
    const style = getComputedStyle(el)
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue

    const rect = el.getBoundingClientRect()
    if (rect.width === 0 && rect.height === 0) continue

    // Elements deliberately hidden from users are not judged on layout.
    if (el.closest('[aria-hidden="true"], [inert]')) continue

    // Does anything between this element and the document scroll horizontally?
    function scrollableAncestor(node) {
      for (let p = node.parentElement; p && p !== document.body; p = p.parentElement) {
        if (/auto|scroll/.test(getComputedStyle(p).overflowX)) return true
      }
      return false
    }

    const scrollsX = /auto|scroll/.test(style.overflowX)
    const scrollsY = /auto|scroll/.test(style.overflowY)
    const clipsX = /hidden|clip/.test(style.overflowX)
    const clipsY = /hidden|clip/.test(style.overflowY)

// 2. Content wider than its box, with no way to reach it.
    //
    // "No way to reach it" has to consider ancestors. A <code> inside a horizontally scrolling
    // <pre> overflows its own box by design — the content is reachable by scrolling the parent —
    // so checking only the element itself flags every scroll container's contents.
    if (!scrollsX && el.scrollWidth > el.clientWidth + 2 && el.clientWidth > 0) {
      const truncating = style.textOverflow === 'ellipsis' && /nowrap/.test(style.whiteSpace)
      if (!truncating && !scrollableAncestor(el)) {
        add('h-overflow', `scrollWidth ${el.scrollWidth} > clientWidth ${el.clientWidth}`, el)
      }
    }

    // 3. Content taller than its box and clipped outright.
    if (clipsY && !scrollsY && el.scrollHeight > el.clientHeight + 2 && el.clientHeight > 0) {
      add('v-clip', `scrollHeight ${el.scrollHeight} > clientHeight ${el.clientHeight}`, el)
    }

    // 4. Rendered outside the viewport horizontally.
    if (rect.width > 0 && (rect.right > vw + 2 || rect.left < -2)) {
      // Only flag things that carry visible content, not layout wrappers bled off-screen.
      if ((el.textContent ?? '').trim() || el.tagName === 'INPUT' || el.tagName === 'BUTTON') {
        add('offscreen-x', `left ${Math.round(rect.left)} right ${Math.round(rect.right)} vw ${vw}`, el)
      }
    }

    // 5. Interactive controls that cannot realistically be clicked.
    const interactive = ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A'].includes(el.tagName)
    if (interactive && rect.width > 0 && rect.height > 0 && (rect.height < 16 || rect.width < 16)) {
      add('tiny-target', `${Math.round(rect.width)}x${Math.round(rect.height)}`, el)
    }
    if (interactive && (rect.width === 0 || rect.height === 0)) {
      add('zero-size-control', `${Math.round(rect.width)}x${Math.round(rect.height)}`, el)
    }
  }

  // 6. Double scrollbars: a page-level scroll region that also contains its own scrolling pane.
  //    This is what a chat layout looks like when its nested heights exceed the viewport — the
  //    whole page creeps upward and the header clips, while the message list scrolls separately.
  const scrollers = all.filter((el) => {
    const st = getComputedStyle(el)
    return /auto|scroll/.test(st.overflowY) && el.scrollHeight > el.clientHeight + 2 && el.clientHeight > 0
  })
  // A code block or table that scrolls inside a scrolling list is a normal, expected pattern;
  // what is not is a generic container that scrolls only because its own scrolling child
  // overflows it, which forces the reader to operate two scrollbars to see one thing.
  const SELF_SCROLLING = new Set(['PRE', 'CODE', 'TABLE'])
  for (const outer of scrollers) {
    const nested = scrollers.find(
      (inner) =>
        inner !== outer &&
        outer.contains(inner) &&
        !SELF_SCROLLING.has(inner.tagName) &&
        !inner.closest('.md-table-wrap')
    )
    if (nested) {
      add('double-scroll', `${describeShort(outer)} scrolls and contains a scrolling ${describeShort(nested)}`, outer)
    }
  }

  // 7. Content taller than the viewport in a layout that declares a fixed height.
  for (const el of all) {
    const st = getComputedStyle(el)
    if (!/calc|vh/.test(st.height) && !/calc|vh/.test(el.style.height ?? '')) continue
    const r = el.getBoundingClientRect()
    if (r.height > vh + 2) {
      add('taller-than-viewport', `${Math.round(r.height)}px in a ${vh}px viewport`, el)
    }
  }

  // 9. Values that should never reach the screen.
  const leaked = []
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  let node
  while ((node = walker.nextNode())) {
    const text = (node.nodeValue ?? '').trim()
    if (!text) continue
    const parent = node.parentElement
    if (!parent) continue
    const style = getComputedStyle(parent)
    if (style.display === 'none' || style.visibility === 'hidden') continue
    if (/\b(undefined|NaN|\[object Object\]|Invalid Date)\b/.test(text)) {
      leaked.push({ text: text.slice(0, 80), where: parent.tagName.toLowerCase() + (parent.className ? `.${String(parent.className).split(' ')[0]}` : '') })
    }
  }
  for (const l of leaked) add('leaked-value', `${l.where}: "${l.text}"`, null)

  // 7. Overlapping interactive controls — a strong signal of broken layout.
  // A modal deliberately covers the page behind it, so comparing its controls against the
  // inert background is meaningless. When one is open, audit only inside it.
  const modal = document.querySelector('.overlay [role="dialog"], .overlay .modal')
  const scope = modal ?? document
  const controls = [...scope.querySelectorAll('button, input, select, textarea')]
    .map((el) => ({ el, r: el.getBoundingClientRect() }))
    .filter(({ el, r }) => {
      if (r.width === 0 || r.height === 0) return false
      const pos = getComputedStyle(el).position
      if (pos === 'absolute' || pos === 'fixed') return false
      // Anything inside a positioned overlay floats over the layout by design.
      return !el.closest('.overlay') || Boolean(modal)
    })
  for (let i = 0; i < controls.length; i++) {
    for (let j = i + 1; j < controls.length; j++) {
      const a = controls[i].r
      const b = controls[j].r
      const overlapX = Math.min(a.right, b.right) - Math.max(a.left, b.left)
      const overlapY = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top)
      if (overlapX > 3 && overlapY > 3) {
        add('overlap', `overlaps by ${Math.round(overlapX)}x${Math.round(overlapY)}px`, controls[i].el)
      }
    }
  }

  // 8. Text too faint to read against its background.
  const luminance = (rgb) => {
    const [r, g, b] = rgb
    const f = (c) => {
      const s = c / 255
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
    }
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
  }
  // Colours must be composited, not read literally: a badge using rgba(74,222,128,0.08) over a
  // dark panel is a faint tint, but reading the raw value makes it look like solid green on
  // solid green. Without this every translucent surface reports as unreadable and buries the
  // genuine contrast defects.
  const parseRgba = (s) => {
    const m = s.match(/rgba?\(([^)]+)\)/)
    if (!m) return null
    const parts = m[1].split(/[,\s/]+/).filter(Boolean).map((x) => parseFloat(x))
    if (parts.length < 3) return null
    return [parts[0], parts[1], parts[2], parts.length >= 4 ? parts[3] : 1]
  }
  const over = (fg, bg) => {
    const a = fg[3]
    return [
      fg[0] * a + bg[0] * (1 - a),
      fg[1] * a + bg[1] * (1 - a),
      fg[2] * a + bg[2] * (1 - a)
    ]
  }
  const PAGE_BG = [15, 17, 21]
  const bgOf = (el) => {
    // Collect every background from the element upward, then composite back-to-front.
    const layers = []
    let cur = el
    while (cur && cur !== document.documentElement) {
      const c = parseRgba(getComputedStyle(cur).backgroundColor)
      if (c && c[3] > 0) {
        layers.push(c)
        if (c[3] >= 1) break
      }
      cur = cur.parentElement
    }
    let result = PAGE_BG
    for (let i = layers.length - 1; i >= 0; i--) result = over(layers[i], result)
    return result
  }
  // Controls left with the browser default styling. In a dark theme an unstyled input is a
  // white box, and the contrast check misses it because dark-on-white reads as legible.
  const pageLuminance = (() => {
    const c = parseRgba(getComputedStyle(document.body).backgroundColor)
    return c ? luminance(c.slice(0, 3)) : 0
  })()
  if (pageLuminance < 0.3) {
    for (const el of document.querySelectorAll('input, select, textarea')) {
      const type = el.getAttribute('type')
      if (type === 'checkbox' || type === 'radio' || type === 'range') continue
      const c = parseRgba(getComputedStyle(el).backgroundColor)
      if (!c || c[3] === 0) continue
      if (luminance(c.slice(0, 3)) > 0.4) {
        add('unthemed-control', `background ${getComputedStyle(el).backgroundColor} on a dark theme`, el)
      }
    }
  }

  const textEls = all.filter(
    (el) => [...el.childNodes].some((n) => n.nodeType === 3 && (n.nodeValue ?? '').trim().length > 2)
  )
  for (const el of textEls) {
    const style = getComputedStyle(el)
    if (style.display === 'none' || style.visibility === 'hidden') continue
    const fgRaw = parseRgba(style.color)
    if (!fgRaw) continue
    const bg = bgOf(el)
    // Text can be translucent too; composite it over its own background before comparing.
    const fg = fgRaw[3] < 1 ? over(fgRaw, bg) : fgRaw.slice(0, 3)
    const l1 = luminance(fg) + 0.05
    const l2 = luminance(bg) + 0.05
    const contrast = Math.max(l1, l2) / Math.min(l1, l2)
    // 3.0 is lenient: this is looking for unreadable, not for strict WCAG compliance.
    if (contrast < 3.0) {
      add(
        'low-contrast',
        `ratio ${contrast.toFixed(2)} (${style.color} on rgb(${bg.map((c) => Math.round(c)).join(',')}))`,
        el
      )
    }
  }

  return problems
}

export async function auditLayout(page, where, report, { ignore = [] } = {}) {
  const problems = await page.evaluate(AUDIT_FN)
  const filtered = problems.filter((p) => !ignore.includes(p.kind))

  // Collapse repeats: twenty instances of one rule is one defect, not twenty.
  const grouped = new Map()
  for (const p of filtered) {
    const key = `${p.kind}::${p.element}`
    if (!grouped.has(key)) grouped.set(key, { ...p, count: 0 })
    grouped.get(key).count++
  }

  for (const p of grouped.values()) {
    report.issue(where, p.kind === 'low-contrast' ? 'visual' : 'layout', `${p.kind}: ${p.element}`, `${p.detail}${p.count > 1 ? ` (x${p.count})` : ''}`)
  }
  return [...grouped.values()]
}

/** Audit the same view at several widths, which is where layout defects usually surface. */
export async function auditResponsive(page, where, report, widths = [1040, 1280, 1600]) {
  const found = []
  for (const width of widths) {
    await page.setViewportSize({ width, height: 900 })
    await page.waitForTimeout(350)
    const problems = await auditLayout(page, `${where} @${width}px`, report)
    found.push({ width, count: problems.length })
  }
  await page.setViewportSize({ width: 1440, height: 920 })
  await page.waitForTimeout(250)
  return found
}
