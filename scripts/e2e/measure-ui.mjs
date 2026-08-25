/**
 * Ad-hoc measurement pass.
 *
 * Answers questions the eye gets wrong: is a column actually too wide, do two grid rows really
 * share tracks, is text truncating. Run it rather than guessing from a screenshot.
 */

import { createEnv, addModel, cleanupEnv } from './fixtures.mjs'
import { launchApp, closeApp, goTo } from './harness.mjs'

const env = await createEnv('measure')
await addModel(env, 'Test-27B-Q4_K_M.gguf', { withMmproj: true })
const { app, page } = await launchApp(env)

try {
  // ---- Dashboard: do the grid rows share column tracks?
  await goTo(page, 'Dashboard')
  await page.waitForTimeout(800)
  const grid = await page.evaluate(() => {
    const g = document.querySelector('.grid-2')
    if (!g) return null
    const style = getComputedStyle(g)
    const cards = [...g.children].map((c) => {
      const r = c.getBoundingClientRect()
      return {
        title: c.querySelector('.card-title')?.textContent?.trim().slice(0, 22) ?? '?',
        left: Math.round(r.left),
        width: Math.round(r.width),
        top: Math.round(r.top),
        height: Math.round(r.height)
      }
    })
    return { containers: document.querySelectorAll('.grid-2').length, template: style.gridTemplateColumns, cards }
  })
  console.log('\n=== Dashboard grid ===')
  console.log('grid containers:', grid.containers)
  console.log('tracks:', grid.template)
  for (const c of grid.cards) {
    console.log(`  ${String(c.title).padEnd(24)} left=${String(c.left).padStart(5)} w=${String(c.width).padStart(4)} top=${String(c.top).padStart(4)} h=${c.height}`)
  }
  const lefts = [...new Set(grid.cards.map((c) => c.left))].sort((a, b) => a - b)
  console.log('distinct left edges:', lefts.join(', '))

  // ---- Chat rail: are titles truncating, and how much slack is there?
  await goTo(page, 'Chat')
  for (let i = 0; i < 3; i++) {
    await page.getByTestId('new-conversation').click()
    await page.waitForTimeout(250)
  }
  await page.waitForTimeout(500)

  const rail = await page.evaluate(() => {
    const list = document.querySelector('.side-list')
    if (!list) return null
    const titles = [...list.querySelectorAll('.truncate')].map((el) => ({
      text: el.textContent?.slice(0, 40) ?? '',
      client: el.clientWidth,
      scroll: el.scrollWidth,
      truncated: el.scrollWidth > el.clientWidth + 1
    }))
    const r = list.getBoundingClientRect()
    const chat = document.querySelector('.chat')?.getBoundingClientRect()
    const msgs = document.querySelector('.messages')?.getBoundingClientRect()
    const firstMsgChild = document.querySelector('.messages > *')?.getBoundingClientRect()
    return {
      railWidth: Math.round(r.width),
      railRight: Math.round(r.right),
      chatLeft: chat ? Math.round(chat.left) : null,
      chatRight: chat ? Math.round(chat.right) : null,
      columnLeft: firstMsgChild ? Math.round(firstMsgChild.left) : null,
      columnRight: firstMsgChild ? Math.round(firstMsgChild.right) : null,
      messagesWidth: msgs ? Math.round(msgs.width) : null,
      viewport: window.innerWidth,
      titles
    }
  })
  console.log('\n=== Chat rail / column ===')
  console.log(JSON.stringify({ ...rail, titles: undefined }, null, 2))
  for (const t of rail.titles) {
    console.log(`  ${t.truncated ? 'CUT ' : 'fits'} client=${String(t.client).padStart(4)} needs=${String(t.scroll).padStart(4)}  "${t.text}"`)
  }
  const gapAfterRail = rail.columnLeft - rail.railRight
  const gapAfterColumn = rail.chatRight - rail.columnRight
  console.log(`gap rail->text: ${gapAfterRail}px    gap text->pane edge: ${gapAfterColumn}px`)

  // ---- Discover: what does a result card actually contain?
  await goTo(page, 'Find a model')
  await page.waitForTimeout(400)
  const discover = await page.evaluate(() => {
    const card = document.querySelector('.card')
    return card ? { html: card.innerHTML.slice(0, 400) } : null
  })
  console.log('\n=== Discover first card ===')
  console.log(discover?.html ?? '(none — needs a search)')
} finally {
  await closeApp(app)
  await cleanupEnv(env)
}
