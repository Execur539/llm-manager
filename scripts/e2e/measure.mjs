import path from 'node:path'
import { createEnv, addModel, cleanupEnv, ROOT } from './fixtures.mjs'
import { launchApp, closeApp, goTo } from './harness.mjs'

const env = await createEnv('measure')
await addModel(env, 'Test-27B-Q4_K_M.gguf', { withMmproj: true })
const { app, page } = await launchApp(env)

await goTo(page, 'My models')
await page.waitForSelector('.model-card')
await page.locator('.model-card').first().click()
await page.waitForSelector('button:has-text("Load with this plan")')
await page.locator('button:has-text("Load with this plan")').click()
await page.waitForSelector('[data-testid="model-loaded"]', { timeout: 30000 })

await goTo(page, 'Chat')
await page.getByTestId('new-conversation').click()
await page.waitForTimeout(300)
await page.getByTestId('chat-input').fill('[[mock:long]] go')
await page.getByTestId('chat-send').click()
await page.waitForTimeout(6000)

const metrics = await page.evaluate(() => {
  const out = {}
  const vh = document.documentElement.clientHeight
  out.viewport = vh
  out.docScrollHeight = document.documentElement.scrollHeight
  out.bodyScrollTop = document.documentElement.scrollTop
  for (const sel of ['.app', '.main', '.split', '.chat', '.messages', '.side-list', '.composer']) {
    const el = document.querySelector(sel)
    if (!el) { out[sel] = 'absent'; continue }
    const cs = getComputedStyle(el)
    const r = el.getBoundingClientRect()
    out[sel] = {
      cssHeight: cs.height,
      rectTop: Math.round(r.top),
      rectHeight: Math.round(r.height),
      clientH: el.clientHeight,
      scrollH: el.scrollHeight,
      overflowY: cs.overflowY,
      scrolls: el.scrollHeight > el.clientHeight + 2
    }
  }
  return out
})

console.log(JSON.stringify(metrics, null, 2))
await closeApp(app)
await cleanupEnv(env)
