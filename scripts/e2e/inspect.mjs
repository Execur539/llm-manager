import { createEnv, addModel, cleanupEnv } from './fixtures.mjs'
import { launchApp, closeApp, goTo } from './harness.mjs'

const env = await createEnv('inspect')
await addModel(env, 'Test-27B-Q4_K_M.gguf', { withMmproj: true })
const { app, page } = await launchApp(env)
await goTo(page, process.argv[2] ?? 'Remote access')

const out = await page.evaluate(() => {
  const res = { unthemedControls: [], strayInSidebar: [], monoProse: [] }
  const lum = (s) => {
    const m = s.match(/rgba?\(([^)]+)\)/)
    if (!m) return null
    const [r, g, b] = m[1].split(',').map(Number)
    const f = (c) => { const x = c / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4 }
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
  }
  const pageLum = lum(getComputedStyle(document.body).backgroundColor) ?? 0

  for (const el of document.querySelectorAll('input, select, textarea')) {
    const cs = getComputedStyle(el)
    const l = lum(cs.backgroundColor)
    if (l !== null && pageLum < 0.2 && l > 0.4 && el.type !== 'checkbox' && el.type !== 'radio') {
      res.unthemedControls.push({ type: el.type || el.tagName, bg: cs.backgroundColor, placeholder: el.placeholder })
    }
  }

  const sidebar = document.querySelector('.sidebar')
  const sb = sidebar.getBoundingClientRect()
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect()
    if (r.width === 0 || r.height === 0) continue
    const inSidebarBox = r.left >= sb.left && r.right <= sb.right && r.top > 700 && r.bottom < sb.bottom - 60
    if (inSidebarBox && !sidebar.contains(el)) {
      res.strayInSidebar.push({ tag: el.tagName, cls: String(el.className), text: (el.textContent||'').trim().slice(0,30), rect: [Math.round(r.left), Math.round(r.top)] })
    }
  }

  // Prose rendered in the monospace face because it inherits from a kv cell.
  for (const el of document.querySelectorAll('.kv dd .faint, .kv dd span')) {
    const cs = getComputedStyle(el)
    const text = (el.textContent || '').trim()
    if (/mono|Cascadia|Consolas/i.test(cs.fontFamily) && /\s/.test(text) && text.length > 12) {
      res.monoProse.push({ text: text.slice(0, 50), font: cs.fontFamily.split(',')[0] })
    }
  }
  return res
})
console.log(JSON.stringify(out, null, 2))
await closeApp(app)
await cleanupEnv(env)
