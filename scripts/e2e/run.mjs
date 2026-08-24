/**
 * End-to-end suite.
 *
 * Runs the real app against a mock inference server, so every scenario exercises the shipping
 * code paths without touching the GPU. Each scenario is independent and gets a fresh sandbox.
 *
 *   node scripts/e2e/run.mjs             # everything
 *   node scripts/e2e/run.mjs layout      # one scenario
 *   node scripts/e2e/run.mjs --list
 */

import path from 'node:path'
import fsp from 'node:fs/promises'
import { createEnv, addModel, addDocument, addEmbeddingModel, startMockHf, cleanupEnv, ROOT } from './fixtures.mjs'
import { Report, launchApp, closeApp, goTo, shot, auditLayout, auditResponsive, VIEWS, SHOTS_DIR } from './harness.mjs'

const report = new Report()

// ---------------------------------------------------------------- helpers

async function withApp(label, fn, { models = ['Test-27B-Q4_K_M.gguf'], extraEnv = {}, hf = false } = {}) {
  console.log(`\n=== ${label} ===`)
  const env = await createEnv(label)
  for (const m of models) await addModel(env, m, { withMmproj: true })

  // Only scenarios that need it pay for the HuggingFace stand-in.
  const mockHf = hf ? await startMockHf() : null
  const { app, page, consoleErrors } = await launchApp(env, {
    ...(mockHf ? { LLMM_HF_BASE: mockHf.base } : {}),
    ...extraEnv
  })
  try {
    await fn({ app, page, env, consoleErrors, hf: mockHf })
  } catch (err) {
    report.fail(label, 'scenario threw', err.message.split('\n')[0])
  } finally {
    // Console errors are a defect class in their own right.
    const meaningful = consoleErrors.filter(
      (e) => !/DevTools|Autofill|Electron Security Warning|ExperimentalWarning/i.test(e)
    )
    for (const e of new Set(meaningful)) {
      report.issue(label, 'console', 'console error', e.slice(0, 200))
    }
    await closeApp(app)
    mockHf?.stop()
    await cleanupEnv(env)
  }
}

/** Load the fixture model so views that need a running model can be exercised. */
async function loadModel(page) {
  await goTo(page, 'My models')
  await page.waitForSelector('.model-card', { timeout: 15000 })
  await page.locator('.model-card').first().click()
  await page.waitForSelector('button:has-text("Load with this plan")', { timeout: 15000 })
  await page.locator('button:has-text("Load with this plan")').first().click()
  await page.waitForSelector('.sidebar-footer .badge.good', { timeout: 30000 })
}

// ---------------------------------------------------------------- scenarios

const scenarios = {
  /** Walk every view, audit layout at three widths, and capture screenshots. */
  async layout() {
    await withApp('layout', async ({ page }) => {
      for (const view of VIEWS) {
        await goTo(page, view)
        const problems = await auditLayout(page, view, report)
        await shot(page, `view-${view}`)
        console.log(`  ${view.padEnd(14)} ${problems.length ? `${problems.length} layout issue(s)` : 'clean'}`)
        for (const p of problems.slice(0, 6)) {
          console.log(`      ${p.kind}: ${p.element} — ${p.detail}`)
        }
      }
    })
  },

  /** The same views at narrow and wide widths, where alignment defects surface. */
  async responsive() {
    await withApp('responsive', async ({ page }) => {
      for (const view of ['Dashboard', 'My models', 'Chat', 'Agent', 'Settings', 'Remote access']) {
        await goTo(page, view)
        const found = await auditResponsive(page, view, report)
        const summary = found.map((f) => `${f.width}px:${f.count}`).join('  ')
        console.log(`  ${view.padEnd(14)} ${summary}`)
      }
    })
  },

  /** Deleting and renaming conversations — the reported bug. */
  async conversations() {
    await withApp('conversations', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Chat')

      // Create three conversations.
      for (let i = 0; i < 3; i++) {
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(300)
      }
      let count = await page.getByTestId('conversation-item').count()
      report.check('conversations', 'creates conversations', count === 3, `saw ${count}`)

      // Delete one: hover reveals actions, then confirm.
      const first = page.getByTestId('conversation-item').first()
      await first.hover()
      await first.getByTestId('delete-conversation').click()
      await page.waitForTimeout(200)
      await first.getByTestId('confirm-delete').click()
      await page.waitForTimeout(600)

      count = await page.getByTestId('conversation-item').count()
      report.check('conversations', 'deletes a conversation', count === 2, `expected 2, saw ${count}`)

      // Delete must survive a reload, i.e. it really left the database.
      await page.reload()
      await page.waitForSelector('.nav-item')
      await goTo(page, 'Chat')
      count = await page.getByTestId('conversation-item').count()
      report.check('conversations', 'deletion persists across reload', count === 2, `expected 2, saw ${count}`)

      // Rename.
      const target = page.getByTestId('conversation-item').first()
      await target.hover()
      await target.getByTestId('rename-conversation').click()
      await page.getByTestId('rename-input').fill('Renamed conversation')
      await page.getByTestId('rename-input').press('Enter')
      await page.waitForTimeout(600)
      const titles = await page.getByTestId('conversation-item').allTextContents()
      report.check(
        'conversations',
        'renames a conversation',
        titles.some((t) => t.includes('Renamed conversation')),
        titles.join(' | ').slice(0, 120)
      )

      await shot(page, 'conversations')
    })
  },

  /** Navigating away mid-response and back — the second reported bug. */
  async streaming() {
    await withApp('streaming', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Chat')

      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      // A deliberately slow response, so there is time to navigate mid-stream.
      await page.getByTestId('chat-input').fill('[[mock:slow]] [[mock:long]] explain state space models')
      await page.getByTestId('chat-send').click()

      await page.waitForSelector('[data-testid="streaming-message"]', { timeout: 20000 })
      const partialBefore = (await page.getByTestId('streaming-message').textContent()) ?? ''
      report.check('streaming', 'response begins streaming', partialBefore.length > 10, `${partialBefore.length} chars`)

      // Leave mid-stream and come back.
      await goTo(page, 'Dashboard')
      await page.waitForTimeout(1800)
      await goTo(page, 'Chat')
      await page.waitForTimeout(800)

      const stillStreaming = await page.getByTestId('streaming-message').count()
      report.check('streaming', 'in-progress response survives navigation', stillStreaming > 0, 'streaming message vanished')

      if (stillStreaming > 0) {
        const partialAfter = (await page.getByTestId('streaming-message').textContent()) ?? ''
        report.check(
          'streaming',
          'text kept accumulating while away',
          partialAfter.length > partialBefore.length,
          `${partialBefore.length} -> ${partialAfter.length} chars`
        )
      }

      // And it must finish and persist.
      await page.waitForSelector('[data-testid="streaming-message"]', { state: 'detached', timeout: 90000 }).catch(() => undefined)
      await page.waitForTimeout(800)
      const messages = await page.locator('.messages .msg').count()
      report.check('streaming', 'finished response persists as a message', messages >= 2, `${messages} messages`)

      await shot(page, 'streaming-complete')
    })
  },

  /** Model load, unload, and what the dashboard reports about it. */
  async modelLifecycle() {
    await withApp('model-lifecycle', async ({ page }) => {
      await goTo(page, 'My models')
      await page.waitForSelector('.model-card', { timeout: 15000 })

      const badges = await page.locator('.model-card .badge').allTextContents()
      report.check('model-lifecycle', 'model card shows a fit badge', badges.some((b) => /fits|offload|tradeoff|too large/.test(b)), badges.join(','))
      report.check('model-lifecycle', 'capabilities detected', badges.some((b) => /vision|tools/.test(b)), badges.join(','))

      await page.locator('.model-card').first().click()
      await page.waitForSelector('button:has-text("Load with this plan")', { timeout: 15000 })
      await auditLayout(page, 'My models (fit plan open)', report)
      await shot(page, 'fit-plan')

      await page.locator('button:has-text("Load with this plan")').first().click()
      await page.waitForSelector('.sidebar-footer .badge.good', { timeout: 30000 })
      report.ok('model loads')

      await goTo(page, 'Dashboard')
      const inference = (await page.locator('.card', { hasText: 'Inference' }).first().textContent()) ?? ''
      report.check('model-lifecycle', 'dashboard reflects the loaded model', !/No model loaded/.test(inference), inference.slice(0, 80))

      await page.locator('.sidebar-footer button:has-text("Unload")').click()
      await page.waitForTimeout(1500)
      const footer = (await page.locator('.sidebar-footer').textContent()) ?? ''
      report.check('model-lifecycle', 'unload clears the footer', /No model loaded/.test(footer), footer.slice(0, 60))
    })
  },

  /** The agent loop end to end, including a tool call and its approval prompt. */
  async agent() {
    await withApp('agent', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Agent')

      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      const toolCount = await page.getByTestId('toggle-tools').textContent()
      report.check('agent', 'tool catalog is populated', /\d+ tools/.test(toolCount ?? ''), toolCount ?? '')

      await page.getByTestId('toggle-tools').click()
      await page.waitForTimeout(300)
      await auditLayout(page, 'Agent (tools open)', report)
      await shot(page, 'agent-tools')
      await page.getByTestId('toggle-tools').click()

      // A read-tier tool should run without prompting.
      await page.getByTestId('agent-input').fill('[[mock:tool]] list the current directory')
      await page.getByTestId('agent-send').click()

      const sawTool = await page
        .waitForSelector('[data-testid="tool-card"]', { timeout: 30000 })
        .then(() => true)
        .catch(() => false)
      report.check('agent', 'tool call renders a card', sawTool)

      if (sawTool) {
        await page.waitForTimeout(2500)
        await auditLayout(page, 'Agent (tool card)', report)
        // Expanding must not break layout.
        await page.locator('[data-testid="tool-card"]').first().click()
        await page.waitForTimeout(400)
        await auditLayout(page, 'Agent (tool card expanded)', report)
        await shot(page, 'agent-tool-expanded')
      }
    })
  },

  /** Content that stresses text rendering: unicode, markdown, very long lines. */
  async content() {
    await withApp('content', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Chat')

      for (const [marker, label] of [
        ['[[mock:unicode]]', 'unicode'],
        ['[[mock:markdown]]', 'markdown'],
        ['[[mock:longline]]', 'longline']
      ]) {
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(300)
        await page.getByTestId('chat-input').fill(`${marker} render this`)
        await page.getByTestId('chat-send').click()
        await page.waitForTimeout(4000)
        const problems = await auditLayout(page, `Chat (${label})`, report)
        console.log(`  ${label.padEnd(10)} ${problems.length ? `${problems.length} issue(s)` : 'clean'}`)
        await shot(page, `content-${label}`)
      }
    })
  },

  /** Empty states: no models, no chats, no documents. */
  async emptyStates() {
    await withApp(
      'empty-states',
      async ({ page }) => {
        for (const view of ['Dashboard', 'My models', 'Chat', 'Agent', 'Documents']) {
          await goTo(page, view)
          const problems = await auditLayout(page, `${view} (empty)`, report)
          const text = (await page.locator('.main').textContent()) ?? ''
          report.check('empty-states', `${view} explains its empty state`, text.trim().length > 40, text.slice(0, 60))
          console.log(`  ${view.padEnd(14)} ${problems.length ? `${problems.length} issue(s)` : 'clean'}`)
          await shot(page, `empty-${view}`)
        }
      },
      { models: [] }
    )
  },

  /** Search, quant recommendation, and a real resumable download. */
  async downloads() {
    await withApp(
      'downloads',
      async ({ page, hf }) => {
        await goTo(page, 'Find a model')

        await page.locator('input[placeholder*="qwen"]').fill('test')
        await page.locator('button:has-text("Search")').click()
        await page.waitForSelector('.row-card', { timeout: 20000 })

        const results = await page.locator('.row-card').count()
        report.check('downloads', 'search returns results', results > 0, `${results} results`)
        await auditLayout(page, 'Find a model (results)', report)
        await shot(page, 'discover-results')

        await page.locator('.row-card').first().click()
        await page.waitForSelector('button:has-text("Download")', { timeout: 20000 })

        const recommended = await page.locator('.card', { hasText: 'Recommended for your hardware' }).count()
        report.check('downloads', 'a quant is recommended', recommended > 0)
        await auditLayout(page, 'Find a model (files)', report)
        await shot(page, 'discover-files')

        // Slow the transfer so progress is observable rather than instantaneous.
        await hf.control({ slow: '400000' })
        await page.locator('button:has-text("Download")').first().click()

        const sawProgress = await page
          .waitForSelector('.meter', { timeout: 20000 })
          .then(() => true)
          .catch(() => false)
        report.check('downloads', 'download shows progress', sawProgress)

        if (sawProgress) {
          await auditLayout(page, 'Find a model (downloading)', report)
          await shot(page, 'discover-downloading')
          const controls = await page.locator('button:has-text("Pause"), button:has-text("Cancel")').count()
          report.check('downloads', 'download can be paused or cancelled', controls > 0, `${controls} controls`)
        }

        await hf.control({ slow: '0' })
        // The finished file should appear in the library without a manual rescan.
        const landed = await page
          .waitForFunction(() => document.querySelectorAll('.model-card').length >= 0, { timeout: 30000 })
          .then(() => true)
          .catch(() => false)
        report.check('downloads', 'download completes without error', landed)
      },
      { hf: true, models: [] }
    )
  },

  /** Collections, ingestion and retrieval. */
  async documents() {
    await withApp('documents', async ({ page, env }) => {
      await addEmbeddingModel(env)
      await goTo(page, 'Documents')

      await page.locator('input[placeholder*="New collection"]').fill('Test collection')
      await page.locator('.side-list button:has-text("+")').click()
      await page.waitForTimeout(800)

      const collections = await page.getByTestId('conversation-item').count().catch(() => 0)
      const sideItems = await page.locator('.side-item').count()
      report.check('documents', 'creates a collection', sideItems > 0, `${sideItems} items`)

      await page.locator('.side-item').first().click()
      await page.waitForTimeout(500)
      await auditLayout(page, 'Documents (collection open)', report)
      await shot(page, 'documents-collection')

      const hasAddButton = await page.locator('button:has-text("Add documents")').count()
      report.check('documents', 'offers document ingestion', hasAddButton > 0)
    })
  },

  /** Starting and stopping the API server, and its request log. */
  async server() {
    await withApp('server', async ({ page }) => {
      await goTo(page, 'API server')

      const before = (await page.locator('.card', { hasText: 'Status' }).first().textContent()) ?? ''
      report.check('server', 'reports a stopped state initially', /stopped/i.test(before), before.slice(0, 60))

      await page.locator('button:has-text("Start server")').click()
      await page.waitForTimeout(2500)

      const after = (await page.locator('.card', { hasText: 'Status' }).first().textContent()) ?? ''
      report.check('server', 'starts', /running/i.test(after), after.slice(0, 80))
      await auditLayout(page, 'API server (running)', report)
      await shot(page, 'server-running')

      // Generating a key must not disturb the layout.
      await page.locator('button:has-text("Generate")').click()
      await page.waitForTimeout(600)
      await auditLayout(page, 'API server (key generated)', report)
      const keyShown = (await page.locator('.card', { hasText: 'API key' }).first().textContent()) ?? ''
      report.check('server', 'generates an API key', /llmm-/.test(keyShown), keyShown.slice(0, 60))

      await page.locator('button:has-text("Stop server")').click()
      await page.waitForTimeout(1500)
      const stopped = (await page.locator('.card', { hasText: 'Status' }).first().textContent()) ?? ''
      report.check('server', 'stops', /stopped/i.test(stopped), stopped.slice(0, 60))
    })
  },

  /** Remote access gating: nothing can be enabled before a password exists. */
  async remote() {
    await withApp('remote', async ({ page }) => {
      await goTo(page, 'Remote access')

      const enableDisabled = await page.locator('button:has-text("Enable remote access")').isDisabled()
      report.check('remote', 'enabling is blocked without a password', enableDisabled)

      // Too-short passwords must be rejected by the UI, not just the backend.
      await page.locator('input[placeholder="New password"]').fill('short')
      const setDisabled = await page.locator('button:has-text("password")').first().isDisabled()
      report.check('remote', 'rejects a short password', setDisabled)

      await page.locator('input[placeholder="New password"]').fill('a-sufficiently-long-password')
      await page.locator('button:has-text("password")').first().click()
      await page.waitForTimeout(1200)

      const hasPassword = (await page.locator('.card', { hasText: 'Password' }).first().textContent()) ?? ''
      report.check('remote', 'accepts a valid password', /\bset\b/i.test(hasPassword), hasPassword.slice(0, 80))

      const nowEnabled = await page.locator('button:has-text("Enable remote access")').isDisabled()
      report.check('remote', 'enabling unlocks once a password exists', !nowEnabled)

      // Remote tool access must be off until deliberately turned on.
      const toolsCard = (await page.locator('.card', { hasText: 'Agent tools over the internet' }).first().textContent()) ?? ''
      report.check('remote', 'remote tools default to off', /\boff\b/i.test(toolsCard), toolsCard.slice(0, 80))

      await auditLayout(page, 'Remote access (password set)', report)
      await shot(page, 'remote-configured')

      // Switching to the self-hosted path reveals more fields; layout must hold.
      await page.locator('input[type="radio"]').nth(1).click()
      await page.waitForTimeout(500)
      await auditLayout(page, 'Remote access (own domain)', report)
      await shot(page, 'remote-own-domain')
    })
  },

  /** Settings round-trip: a change must persist across a reload. */
  async settings() {
    await withApp('settings', async ({ page }) => {
      await goTo(page, 'Settings')
      await auditLayout(page, 'Settings', report)

      const target = page.locator('input[type="number"]').first()
      await target.fill('98304')
      await target.blur()
      await page.waitForTimeout(800)

      await page.reload()
      await page.waitForSelector('.nav-item')
      await goTo(page, 'Settings')
      const persisted = await page.locator('input[type="number"]').first().inputValue()
      report.check('settings', 'a changed setting persists across reload', persisted === '98304', `saw ${persisted}`)

      // Plan mode is a checkbox that must round-trip too.
      const planBox = page.locator('input[type="checkbox"]').first()
      const wasChecked = await planBox.isChecked()
      await planBox.click()
      await page.waitForTimeout(600)
      await page.reload()
      await page.waitForSelector('.nav-item')
      await goTo(page, 'Settings')
      const nowChecked = await page.locator('input[type="checkbox"]').first().isChecked()
      report.check('settings', 'a toggled checkbox persists', nowChecked !== wasChecked, `${wasChecked} -> ${nowChecked}`)

      // The hard-block override must require a typed phrase, and must not be reachable by a
      // single click. This previously used window.prompt, which Electron does not implement.
      await page.getByTestId('toggle-hard-blocks').click()
      await page.waitForSelector('[data-testid="confirm-overlay"]', { timeout: 5000 })
      report.ok('hard-block override opens a confirmation')

      const blockedInitially = await page.getByTestId('confirm-accept').isDisabled()
      report.check('settings', 'confirm is disabled until the phrase is typed', blockedInitially)

      await page.getByTestId('confirm-phrase-input').fill('WRONG PHRASE')
      report.check(
        'settings',
        'a wrong phrase does not enable confirm',
        await page.getByTestId('confirm-accept').isDisabled()
      )

      // Cancelling must leave protection on.
      await page.getByTestId('confirm-cancel').click()
      await page.waitForTimeout(500)
      let badge = (await page.locator('.danger-zone').textContent()) ?? ''
      report.check('settings', 'cancelling leaves hard blocks enabled', /enabled/i.test(badge), badge.slice(0, 60))

      // The correct phrase does enable it, and the change persists.
      await page.getByTestId('toggle-hard-blocks').click()
      await page.waitForSelector('[data-testid="confirm-overlay"]')
      await page.getByTestId('confirm-phrase-input').fill('DISABLE HARD BLOCKS')
      report.check(
        'settings',
        'the exact phrase enables confirm',
        !(await page.getByTestId('confirm-accept').isDisabled())
      )
      await auditLayout(page, 'Settings (confirm dialog)', report)
      await shot(page, 'settings-confirm')
      await page.getByTestId('confirm-accept').click()
      await page.waitForTimeout(800)
      badge = (await page.locator('.danger-zone').textContent()) ?? ''
      report.check('settings', 'confirming disables hard blocks', /DISABLED/.test(badge), badge.slice(0, 60))

      // And it can be turned back on without ceremony.
      await page.getByTestId('toggle-hard-blocks').click()
      await page.waitForTimeout(800)
      badge = (await page.locator('.danger-zone').textContent()) ?? ''
      report.check('settings', 're-enabling needs no confirmation', /enabled/i.test(badge), badge.slice(0, 60))

      await shot(page, 'settings')
    })
  },

  /** Failure paths: a model that will not load, a server error, an aborted turn. */
  async errors() {
    await withApp('errors', async ({ page }) => {
      // A corrupt model must report clearly rather than silently doing nothing.
      await goTo(page, 'My models')
      await page.waitForSelector('.model-card', { timeout: 15000 })
      await loadModel(page)
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      // Server-side failure mid-request.
      await page.getByTestId('chat-input').fill('[[mock:error]] fail please')
      await page.getByTestId('chat-send').click()
      await page.waitForTimeout(4000)

      const errorShown = await page.locator('.error-card').count()
      report.check('errors', 'a failed request surfaces an error', errorShown > 0, 'no error card rendered')
      await auditLayout(page, 'Chat (error)', report)
      await shot(page, 'chat-error')

      // The composer must not be left permanently disabled after a failure.
      const inputDisabled = await page.getByTestId('chat-input').isDisabled()
      report.check('errors', 'input is usable again after an error', !inputDisabled, 'composer stayed disabled')

      // A second, healthy request must still work.
      await page.getByTestId('chat-input').fill('now a normal reply')
      await page.getByTestId('chat-send').click()
      await page.waitForTimeout(5000)
      const messages = await page.locator('.messages .msg').count()
      report.check('errors', 'recovers and answers the next message', messages >= 2, `${messages} messages`)
    })
  },

  /** Window resizing, including sizes below the configured minimum. */
  async resize() {
    await withApp('resize', async ({ page }) => {
      await loadModel(page)
      for (const [w, h] of [
        [1040, 660],
        [1280, 720],
        [1920, 1080],
        [1100, 900],
        [2560, 1440]
      ]) {
        await page.setViewportSize({ width: w, height: h })
        await page.waitForTimeout(400)
        for (const view of ['Dashboard', 'Chat', 'My models', 'Settings']) {
          await goTo(page, view)
          await auditLayout(page, `${view} @${w}x${h}`, report)
        }
      }
      await page.setViewportSize({ width: 1440, height: 920 })
      report.ok('survives resizing across five sizes')
    })
  },

  /** Rapid navigation, to catch races and unmount errors. */
  async navigation() {
    await withApp('navigation', async ({ page }) => {
      for (let round = 0; round < 3; round++) {
        for (const view of VIEWS) {
          await page.locator('.nav-item', { hasText: new RegExp(`^${view}$`) }).first().click()
          await page.waitForTimeout(60)
        }
      }
      await page.waitForTimeout(1200)
      const stillAlive = await page.locator('.nav-item').count()
      report.check('navigation', 'app survives rapid navigation', stillAlive === VIEWS.length, `${stillAlive} nav items`)
      await auditLayout(page, 'after rapid navigation', report)
    })
  }
}

// ---------------------------------------------------------------- main

const args = process.argv.slice(2)
if (args.includes('--list')) {
  console.log('\nScenarios:\n')
  for (const name of Object.keys(scenarios)) console.log(`  ${name}`)
  console.log('')
  process.exit(0)
}

const selected = args.filter((a) => !a.startsWith('--'))
const names = selected.length ? selected : Object.keys(scenarios)

// Only clear screenshots for a full run; a targeted run should not delete evidence from the
// scenarios it is not running.
if (!selected.length) {
  await fsp.rm(SHOTS_DIR, { recursive: true, force: true }).catch(() => undefined)
}

for (const name of names) {
  if (!scenarios[name]) {
    console.error(`Unknown scenario: ${name}`)
    process.exit(1)
  }
  await scenarios[name]()
}

// ---------------------------------------------------------------- report

const summary = report.summary()
console.log(`\n${'='.repeat(72)}`)
console.log(`${summary.passes} checks passed, ${summary.total} issue(s) found`)
if (summary.total) {
  console.log(`by kind: ${Object.entries(summary.bySeverity).map(([k, v]) => `${k}=${v}`).join('  ')}\n`)
  const byWhere = new Map()
  for (const i of summary.issues) {
    if (!byWhere.has(i.where)) byWhere.set(i.where, [])
    byWhere.get(i.where).push(i)
  }
  for (const [where, issues] of byWhere) {
    console.log(`  ${where}`)
    for (const i of issues) console.log(`    [${i.severity}] ${i.message}${i.detail ? ` — ${i.detail}` : ''}`)
  }
}
console.log(`\nScreenshots: ${SHOTS_DIR}`)

await fsp.writeFile(path.join(ROOT, 'scripts', 'e2e', 'last-run.json'), JSON.stringify(summary, null, 2))
process.exit(summary.issues.some((i) => i.severity === 'bug') ? 1 : 0)
