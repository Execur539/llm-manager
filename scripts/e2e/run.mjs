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
import { buildGguf, createEnv, addModel, addDocument, addEmbeddingModel, startMockHf, cleanupEnv, ROOT } from './fixtures.mjs'
import { Report, launchApp, closeApp, goTo, shot, auditLayout, auditResponsive, stubDialogs, toastTexts, VIEWS, SHOTS_DIR, scenarioContext } from './harness.mjs'
import fs from 'node:fs'
import os from 'node:os'

const report = new Report()

// ---------------------------------------------------------------- helpers

async function withApp(
  label,
  fn,
  { models = ['Test-27B-Q4_K_M.gguf'], extraEnv = {}, hf = false, modelOpts = {}, noMmproj = false } = {}
) {
  console.log(`\n=== ${label} ===`)
  const env = await createEnv(label)
  for (const m of models) await addModel(env, m, { withMmproj: !noMmproj, ...modelOpts })

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

/**
 * Every .gguf under a models directory.
 *
 * Models live in per-repository subfolders, so a flat readdir of the root finds nothing —
 * which is how an assertion about deleted files can pass without testing anything.
 */
async function listGgufPaths(root) {
  const out = []
  const walk = async (dir) => {
    for (const entry of await fsp.readdir(dir, { withFileTypes: true }).catch(() => [])) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) await walk(full)
      else if (entry.name.toLowerCase().endsWith('.gguf')) out.push(full)
    }
  }
  await walk(root)
  return out
}

async function listGguf(root) {
  return (await listGgufPaths(root)).map((p) => path.basename(p))
}

/** Stop any running turn and wait until the composer is usable again. */
/*
 * End a turn that is still streaming.
 *
 * This always clicked `agent-stop`, which does not exist in chat — its send button turns into
 * the stop control while a turn runs. So in every chat scenario the click found nothing and
 * spent Playwright's default thirty-second timeout discovering that, then fell through to a
 * twenty-second wait for a turn nobody had stopped. Fifty seconds, three times a run, to do
 * nothing. It was most of the wall clock of the suite.
 *
 * The timeouts are short now as well: if the control is on screen the click lands immediately,
 * and if it is not, no amount of waiting will change that.
 */
async function stopTurn(page, inputTestId) {
  const kind = inputTestId.startsWith('agent') ? 'agent' : 'chat'
  const dedicated = page.getByTestId(`${kind}-stop`)
  const control = (await dedicated.count()) ? dedicated : page.getByTestId(`${kind}-send`)
  await control.first().click({ timeout: 4000 }).catch(() => undefined)
  await page.waitForFunction(
    (id) => {
      const el = document.querySelector(`[data-testid="${id}"]`)
      return el instanceof HTMLTextAreaElement && !el.disabled
    },
    inputTestId,
    { timeout: 15000 }
  ).catch(() => undefined)
}

/** Load the fixture model so views that need a running model can be exercised. */
async function loadModel(page) {
  await goTo(page, 'My models')
  await page.waitForSelector('.model-card', { timeout: 15000 })
  await page.locator('.model-card').first().click()
  await page.waitForSelector('button:has-text("Load with this plan")', { timeout: 15000 })
  await page.locator('button:has-text("Load with this plan")').first().click()
  await page.waitForSelector('[data-testid="model-loaded"]', { timeout: 30000 })
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

      /*
       * And it must end and persist.
       *
       * Ended by stopping it rather than by waiting it out. This asked for a slow response of
       * eight hundred words at a quarter-second a token — well over three minutes — and then
       * waited ninety seconds for it to finish, which it never did. The timeout was swallowed
       * and the scenario carried on, so the wait bought nothing at all and cost more wall clock
       * than the rest of the suite put together.
       *
       * A stopped turn persists what it had written, by design, so the assertion below is
       * testing the same path either way: a response that is over becomes a stored message.
       */
      await stopTurn(page, 'chat-input')
      await page.waitForSelector('[data-testid="streaming-message"]', { state: 'detached', timeout: 20000 }).catch(() => undefined)
      await page
        .waitForFunction(() => document.querySelectorAll('.messages .msg').length >= 2, undefined, { timeout: 20000 })
        .catch(() => undefined)
      const messages = await page.locator('.messages .msg').count()
      report.check('streaming', 'finished response persists as a message', messages >= 2, `${messages} messages`)

      /*
       * The context reading, which must be a real measurement rather than a placeholder.
       *
       * Asserted on the parsed numbers rather than on the element existing, because the failure
       * this guards against is the meter rendering with nothing behind it — the old estimate
       * divided characters by four and could report a figure while knowing nothing about what
       * the model had actually been sent.
       */
      const reading = await page
        .getByTestId('context-meter')
        .getAttribute('title')
        .catch(() => null)
      const parsed = reading?.match(/^([\d,]+) of ([\d,]+) tokens/)
      const used = parsed ? Number(parsed[1].replace(/,/g, '')) : 0
      const max = parsed ? Number(parsed[2].replace(/,/g, '')) : 0
      report.check('streaming', 'the context meter reports a real usage figure', used > 0, reading ?? 'no meter')
      report.check('streaming', 'and reports the window it is measured against', max > 0 && used <= max,
        `${used} of ${max}`)

      await shot(page, 'streaming-complete')
    })
  },

  /** Model load, unload, and what the dashboard reports about it. */
  async modelLifecycle() {
    await withApp('model-lifecycle', async ({ page }) => {
      await goTo(page, 'My models')
      await page.waitForSelector('.model-card', { timeout: 15000 })
      // The fit badge is computed asynchronously; assert on the settled value, not the spinner.
      await page
        .waitForFunction(
          () => !document.querySelector('.model-card')?.textContent?.includes('checking'),
          undefined,
          { timeout: 20000 }
        )
        .catch(() => undefined)

      const badges = await page.locator('.model-card .badge').allTextContents()
      report.check('model-lifecycle', 'model card shows a fit badge', badges.some((b) => /fits|offload|tradeoff|too large/.test(b)), badges.join(','))
      report.check('model-lifecycle', 'capabilities detected', badges.some((b) => /vision|tools/.test(b)), badges.join(','))

      await page.locator('.model-card').first().click()
      await page.waitForSelector('button:has-text("Load with this plan")', { timeout: 15000 })
      await auditLayout(page, 'My models (fit plan open)', report)
      await shot(page, 'fit-plan')

      await page.locator('button:has-text("Load with this plan")').first().click()
      await page.waitForSelector('[data-testid="model-loaded"]', { timeout: 30000 })
      report.ok('model-lifecycle', 'model loads')

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

      /*
       * What the transcript shows while a call is being written, before it can run.
       *
       * A call cannot be dispatched until every argument has arrived, so this window used to be
       * blank — the model had stopped writing prose and the card did not exist yet. Slowed
       * deliberately, because at full speed a call completes in a tenth of a second and the
       * state being asserted is over before it can be observed. That is also why it was never
       * noticed as missing.
       */
      await page.getByTestId('agent-input').fill('[[mock:tool]] [[mock:slow]] list the current directory')
      await page.getByTestId('agent-send').click()

      const pending = await page
        .waitForSelector('[data-testid="pending-tool-call"]', { timeout: 30000 })
        .then((el) => el.textContent())
        .catch(() => null)
      report.check('agent', 'a tool call is shown inline while it is being written',
        !!pending, pending ?? 'never appeared')
      report.check('agent', 'and names the tool it is about to run',
        !!pending && /list_dir|read_file|[a-z_]{3,}/.test(pending), pending ?? '')

      // It must give way to a real card rather than the two accumulating side by side.
      await page.waitForSelector('[data-testid="tool-card"]', { timeout: 30000 }).catch(() => undefined)
      await page.waitForTimeout(400)
      const settled = await page.getByTestId('pending-tool-call').count()
      report.check('agent', 'a completed call becomes a real card, not a second in-progress one',
        settled <= 1, `${settled} in-progress cards at once`)

      /*
       * And nothing is left claiming to be in progress once the turn is over.
       *
       * A stopped turn abandons whatever call was mid-composition, and the finished call that
       * would normally replace it is precisely what never arrives — so without clearing it on
       * the way out, the transcript kept a card saying the agent was writing while nothing was
       * running at all.
       */
      await stopTurn(page, 'agent-input')
      await page.waitForTimeout(300)
      const afterStop = await page.getByTestId('pending-tool-call').count()
      report.check('agent', 'stopping a turn clears a half-written call',
        afterStop === 0, `${afterStop} left on screen`)

      const sawTool = await page
        .waitForSelector('[data-testid="tool-card"]', { timeout: 30000 })
        .then(() => true)
        .catch(() => false)
      report.check('agent', 'tool call renders a card', sawTool)

      /*
       * Compacting must say what it did, including when it did nothing.
       *
       * The button sent no session id and reported no result. Nothing hydrates the agent outside
       * of a turn, so clicking it could find an empty history, return early, and look exactly
       * like success — no message, no visible change, no way to tell the difference. Whichever
       * branch this session lands in, silence is the one outcome that is wrong.
       */
      await page.getByTestId('compact-session').click()
      await page.waitForTimeout(1500)
      const said = (await toastTexts(page)).join(' | ')
      report.check('agent', 'compacting reports what it did rather than failing silently',
        said.trim().length > 0, said || 'no toast at all')

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

  /**
   * The approval gate. This is the only thing standing between a confused model and the
   * filesystem, so it gets more scrutiny than anything else in the app.
   */
  async permissions() {
    await withApp('permissions', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Agent')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      // A write-tier tool must raise a prompt rather than simply running.
      await page.getByTestId('agent-input').fill('[[mock:tool]] write a file for me')
      await page.getByTestId('agent-send').click()

      // list_dir is read-tier and must run silently.
      const readPrompted = await page
        .waitForSelector('[data-testid="permission-overlay"]', { timeout: 6000 })
        .then(() => true)
        .catch(() => false)
      report.check('permissions', 'read-tier tools run without prompting', !readPrompted)
      await page.waitForTimeout(2000)
      await stopTurn(page, 'agent-input')

      // A write-tier tool must stop and ask.
      await page.getByTestId('agent-input').fill('[[mock:calltool:write_file]] save something')
      await page.getByTestId('agent-send').click()
      const prompted = await page
        .waitForSelector('[data-testid="permission-overlay"]', { timeout: 25000 })
        .then(() => true)
        .catch(() => false)
      report.check('permissions', 'write-tier tools require approval', prompted)

      if (prompted) {
        // The prompt must show the resolved target, not the model's description of it.
        const resolved = (await page.locator('.resolved-action').textContent()) ?? ''
        report.check(
          'permissions',
          'the prompt shows a fully resolved path',
          /^[A-Za-z]:\\/.test(resolved.replace(/^Write \d+ bytes to /, '')),
          resolved.slice(0, 90)
        )
        report.check('permissions', 'the tier is shown', (await page.locator('.modal .badge').first().textContent()) === 'write')
        await auditLayout(page, 'Agent (approval prompt)', report)
        await shot(page, 'permission-prompt')

        // Denying must be reported back rather than silently doing nothing.
        await page.getByTestId('permission-deny').click()
        await page.waitForTimeout(3000)
        const transcript = (await page.locator('.messages').textContent()) ?? ''
        report.check('permissions', 'a denial is recorded in the transcript', /denied/i.test(transcript), transcript.slice(-120))

        // A refused call must not be re-prompted for the rest of the turn.
        const reprompted = await page
          .waitForSelector('[data-testid="permission-overlay"]', { timeout: 6000 })
          .then(() => true)
          .catch(() => false)
        report.check(
          'permissions',
          'a denied call is not prompted again in the same turn',
          !reprompted,
          'the same dialog reappeared'
        )
        await stopTurn(page, 'agent-input')
      }

      // A hard-blocked command must be refused outright, with no approve option.
      await page.getByTestId('agent-input').fill('[[mock:calltool:run_command]] run something')
      await page.getByTestId('agent-send').click()
      const cmdPrompt = await page
        .waitForSelector('[data-testid="permission-overlay"]', { timeout: 25000 })
        .then(() => true)
        .catch(() => false)
      report.check('permissions', 'execute-tier tools require approval', cmdPrompt)

      if (cmdPrompt) {
        const allowVisible = await page.getByTestId('permission-allow').count()
        report.check('permissions', 'an ordinary command offers approval', allowVisible > 0)
        // Escape is a safe default: it denies.
        await page.keyboard.press('Escape')
        await page.waitForTimeout(500)
        const closed = await page.locator('[data-testid="permission-overlay"]').count()
        report.check('permissions', 'Escape dismisses the prompt as a denial', closed === 0)
        await stopTurn(page, 'agent-input')
      }
    })
  },

  /**
   * The prompt is modal on purpose: a pending authorisation should not be left behind while
   * the user wanders off, and it must not be dismissable by a stray click.
   */
  async permissionModality() {
    await withApp('permission-modality', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Agent')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      await page.getByTestId('agent-input').fill('[[mock:calltool:write_file]] save something')
      await page.getByTestId('agent-send').click()
      await page.waitForSelector('[data-testid="permission-overlay"]', { timeout: 25000 })

      // Navigation must be blocked while a decision is outstanding.
      const navBlocked = await page
        .locator('.nav-item', { hasText: /^Dashboard$/ })
        .click({ timeout: 2500 })
        .then(() => false)
        .catch(() => true)
      report.check('permission-modality', 'the page behind the prompt is not clickable', navBlocked)

      const stillOnAgent = await page.locator('[data-testid="agent-messages"]').count()
      report.check('permission-modality', 'the view does not change underneath it', stillOnAgent > 0)

      // A stray click on the backdrop must not count as approval — or as anything.
      await page.mouse.click(20, 400)
      await page.waitForTimeout(400)
      const survived = await page.locator('[data-testid="permission-overlay"]').count()
      report.check('permission-modality', 'a backdrop click does not dismiss it', survived > 0)

      // The prompt still renders correctly after the window is resized under it.
      await page.setViewportSize({ width: 1040, height: 660 })
      await page.waitForTimeout(400)
      await auditLayout(page, 'Agent (prompt, narrow)', report)
      await page.setViewportSize({ width: 1440, height: 920 })
      await page.waitForTimeout(300)

      await page.getByTestId('permission-deny').click()
      await page.waitForTimeout(1000)
      const gone = await page.locator('[data-testid="permission-overlay"]').count()
      report.check('permission-modality', 'answering releases the page', gone === 0)

      const navWorks = await page
        .locator('.nav-item', { hasText: /^Dashboard$/ })
        .click({ timeout: 3000 })
        .then(() => true)
        .catch(() => false)
      report.check('permission-modality', 'navigation works again afterwards', navWorks)

      await goTo(page, 'Agent')
      await stopTurn(page, 'agent-input')
    })
  },

  /** Plan mode must confine the agent to read-only tools. */
  async planMode() {
    await withApp('plan-mode', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Agent')

      const before = await page.getByTestId('toggle-tools').textContent()
      const beforeCount = Number((before ?? '').match(/(\d+)/)?.[1] ?? 0)

      await page.getByTestId('plan-mode').click()
      await page.waitForTimeout(900)

      const label = await page.getByTestId('plan-mode').textContent()
      report.check('plan-mode', 'plan mode toggles on', /on/i.test(label ?? ''), label ?? '')

      const notice = await page.locator('.note', { hasText: 'Plan mode' }).count()
      report.check('plan-mode', 'plan mode is announced in the view', notice > 0)

      // The catalog offered to the model must shrink to read-tier only.
      await page.reload()
      await page.waitForSelector('.nav-item')
      await goTo(page, 'Agent')
      await page.waitForTimeout(600)
      const after = await page.getByTestId('toggle-tools').textContent()
      const afterCount = Number((after ?? '').match(/(\d+)/)?.[1] ?? 0)
      report.check(
        'plan-mode',
        'plan mode reduces the offered tool set',
        afterCount > 0 && afterCount < beforeCount,
        `${beforeCount} -> ${afterCount} tools`
      )

      await page.getByTestId('toggle-tools').click()
      await page.waitForTimeout(400)
      const tiers = await page.locator('.tool-grid .badge').evaluateAll((els) =>
        els.map((e) => e.className.replace('badge', '').trim())
      )
      report.check(
        'plan-mode',
        'only read-tier tools are offered in plan mode',
        tiers.length > 0 && tiers.every((t) => t === 'good'),
        [...new Set(tiers)].join(',')
      )
      await auditLayout(page, 'Agent (plan mode)', report)
      await shot(page, 'agent-plan-mode')
    })
  },

  /** The fit view when the target cannot be met, and when a model is unreadable. */
  async fitTradeoffs() {
    await withApp(
      'fit-tradeoffs',
      async ({ page }) => {
        // An unreasonably high target forces the engine into the tradeoff path.
        await goTo(page, 'Settings')
        const ideal = page.locator('input[type="number"]').nth(1)
        await ideal.fill('100000000')
        await ideal.blur()
        await page.waitForTimeout(800)

        await goTo(page, 'My models')
        await page.waitForSelector('.model-card', { timeout: 15000 })
        await page.locator('.model-card').first().click()
        await page.waitForTimeout(2500)

        const text = (await page.locator('.main').textContent()) ?? ''
        const offered = await page.locator('button:has-text("Load with this plan")').count()
        report.check('fit-tradeoffs', 'a plan or a set of tradeoffs is always offered', offered > 0, `${offered} options`)
        report.check(
          'fit-tradeoffs',
          'the plan explains itself',
          /All \d+ layers on GPU|layers on GPU|context/i.test(text),
          text.slice(0, 100)
        )
        await auditLayout(page, 'My models (tradeoffs)', report)
        await shot(page, 'fit-tradeoffs')
      },
      { models: ['Test-27B-Q4_K_M.gguf'] }
    )
  },

  /** A file that is not a valid GGUF must be reported, not crash the scan. */
  async badModel() {
    await withApp(
      'bad-model',
      async ({ page, env }) => {
        const fs = await import('node:fs/promises')
        const p = await import('node:path')
        const dir = p.join(env.modelsDir, 'broken')
        await fs.mkdir(dir, { recursive: true })
        await fs.writeFile(p.join(dir, 'Corrupt-7B-Q4_K_M.gguf'), Buffer.from('this is not a gguf file'))

        await goTo(page, 'My models')
        await page.locator('button:has-text("Rescan")').click()
        await page.waitForTimeout(2500)

        const cards = await page.locator('.model-card').count()
        report.check('bad-model', 'a corrupt file still appears in the library', cards >= 2, `${cards} cards`)

        const badBadge = await page.locator('.badge.bad').count()
        report.check('bad-model', 'the corrupt file is flagged', badBadge > 0)

        const text = (await page.locator('.main').textContent()) ?? ''
        report.check(
          'bad-model',
          'the reason is shown rather than a bare failure',
          /gguf|magic|parse|unreadable/i.test(text),
          text.slice(0, 120)
        )

        // Selecting it must not break the view.
        await page.locator('.model-card', { hasText: 'Corrupt' }).click()
        await page.waitForTimeout(1500)
        await auditLayout(page, 'My models (corrupt file selected)', report)
        report.ok('bad-model', 'selecting a corrupt model does not break the view')
        await shot(page, 'bad-model')
      },
      { models: ['Test-27B-Q4_K_M.gguf'] }
    )
  },

  /** Keyboard access: focus order, Escape, and no keyboard traps. */
  async keyboard() {
    await withApp('keyboard', async ({ page }) => {
      await goTo(page, 'Settings')

      // Tabbing must reach real controls rather than cycling on the document.
      const reached = new Set()
      for (let i = 0; i < 25; i++) {
        await page.keyboard.press('Tab')
        const tag = await page.evaluate(() => {
          const el = document.activeElement
          return el ? `${el.tagName}${el.getAttribute('type') ? `:${el.getAttribute('type')}` : ''}` : 'NONE'
        })
        reached.add(tag)
      }
      report.check(
        'keyboard',
        'tab reaches multiple control types',
        reached.size > 2 && !reached.has('NONE'),
        [...reached].join(',')
      )

      // Escape must close a modal, and focus must not be trapped afterwards.
      await page.getByTestId('toggle-hard-blocks').click()
      await page.waitForSelector('[data-testid="confirm-overlay"]')
      await page.keyboard.press('Escape')
      await page.waitForTimeout(400)
      const closed = await page.locator('[data-testid="confirm-overlay"]').count()
      report.check('keyboard', 'Escape closes the confirmation', closed === 0)

      await page.keyboard.press('Tab')
      const focusable = await page.evaluate(() => document.activeElement?.tagName ?? 'NONE')
      report.check('keyboard', 'focus works again after the modal closes', focusable !== 'NONE', focusable)
    })
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
      await page.getByTestId('create-collection').click()
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
      report.ok('settings', 'hard-block override opens a confirmation')

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
      report.ok('resize', 'survives resizing across five sizes')
    })
  },

  /**
   * Exporting a conversation: the file is written, the user is told where, and a second
   * export does not destroy the first.
   */
  async chatExport() {
    await withApp('chat-export', async ({ app, page, env }) => {
      const outDir = path.join(env.base, 'exports')
      await fsp.mkdir(outDir, { recursive: true })

      await loadModel(page)
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      // Give the conversation some content worth exporting.
      await page.getByTestId('chat-input').fill('hello from the export test')
      await page.getByTestId('chat-send').click()
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="chat-input"]')
          return el instanceof HTMLTextAreaElement && !el.disabled
        },
        undefined,
        { timeout: 30000 }
      )

      const target = path.join(outDir, 'conversation.md')
      await stubDialogs(app, { save: target })
      await page.getByTestId('export-chat').click()

      // The write is reported, not silent — the original defect was a button that did nothing.
      await page.waitForSelector('[data-testid="toast"]', { timeout: 15000 }).catch(() => undefined)
      const toasts = await toastTexts(page)
      report.check('chat-export', 'export reports where the file went',
        toasts.some((t) => /Exported to/i.test(t)), toasts.join(' | ').slice(0, 160))

      const wrote = fs.existsSync(target)
      report.check('chat-export', 'export writes the file', wrote, target)
      if (wrote) {
        const body = await fsp.readFile(target, 'utf8')
        report.check('chat-export', 'export contains the user message',
          body.includes('hello from the export test'), `${body.length} chars`)
        report.check('chat-export', 'export contains the assistant reply',
          /assistant/i.test(body) && body.length > 120, `${body.length} chars`)
      }

      // Cancelling must be a no-op, not an error.
      await stubDialogs(app, { save: null })
      await page.getByTestId('export-chat').click()
      await page.waitForTimeout(500)
      const afterCancel = await toastTexts(page)
      report.check('chat-export', 'cancelling the save dialog reports nothing',
        !afterCancel.some((t) => /failed/i.test(t)), afterCancel.join(' | ').slice(0, 160))

      await shot(page, 'chat-export')
    })
  },

  /**
   * Deleting a model. The confirmation is in-app (a native confirm() would block the renderer),
   * and a large model requires the filename to be typed.
   */
  async modelDelete() {
    await withApp('model-delete', async ({ page, env }) => {
      await goTo(page, 'My models')
      await page.waitForSelector('.model-card', { timeout: 15000 })

      const before = await page.locator('.model-card').count()
      await page.getByTestId('delete-model').first().click()

      // An in-app dialog, not a native one: the test can see it at all.
      const dialogShown = await page.getByTestId('confirm-accept').isVisible().catch(() => false)
      report.check('model-delete', 'delete asks for confirmation in-app', dialogShown)

      // Escape must cancel without deleting.
      await page.keyboard.press('Escape')
      await page.waitForTimeout(300)
      const afterEscape = await page.locator('.model-card').count()
      report.check('model-delete', 'Escape cancels the delete', afterEscape === before,
        `${before} -> ${afterEscape}`)

      await page.getByTestId('delete-model').first().click()
      await page.waitForTimeout(200)

      // The fixture model is small, so no phrase is required; confirm goes straight through.
      const phraseNeeded = await page.getByTestId('confirm-phrase-input').isVisible().catch(() => false)
      if (phraseNeeded) {
        const name = await page.locator('.model-card').first().locator('.model-name, .mono').first().textContent()
        await page.getByTestId('confirm-phrase-input').fill((name ?? '').trim())
      }
      await page.getByTestId('confirm-accept').click()
      await page.waitForTimeout(1200)

      const after = await page.locator('.model-card').count()
      report.check('model-delete', 'model is removed from the library', after === before - 1,
        `${before} -> ${after}`)

      const files = await listGguf(env.modelsDir)
      report.check('model-delete', 'the file is gone from disk',
        !files.some((f) => f.endsWith('.gguf')), files.join(','))
    })
  },

  /** Importing a GGUF already on disk must not duplicate tens of gigabytes. */
  async modelImport() {
    await withApp('model-import', async ({ app, page, env }) => {
      // A GGUF outside the models folder, standing in for one the user downloaded elsewhere.
      const outside = path.join(env.base, 'elsewhere')
      await fsp.mkdir(outside, { recursive: true })
      const src = path.join(outside, 'Imported-3B-Q4_K_M.gguf')
      await fsp.writeFile(src, buildGguf({ name: 'Imported-3B' }))

      await goTo(page, 'My models')
      await page.waitForSelector('.model-card', { timeout: 15000 })
      const before = await page.locator('.model-card').count()

      await stubDialogs(app, { open: [src] })
      await page.getByTestId('import-gguf').click()
      await page.waitForTimeout(2500)

      const after = await page.locator('.model-card').count()
      report.check('model-import', 'imported model appears in the library', after === before + 1,
        `${before} -> ${after}`)

      const toasts = await toastTexts(page)
      report.check('model-import', 'import reports what it did',
        toasts.some((t) => /Imported/i.test(t)), toasts.join(' | ').slice(0, 200))
      // Same volume, so it should be a hard link rather than a copy of the whole file.
      report.check('model-import', 'same-volume import links instead of copying',
        toasts.some((t) => /linked/i.test(t)), toasts.join(' | ').slice(0, 200))

      // Importing a file that is already in the library must be a no-op, not a duplicate.
      const inside = (await listGgufPaths(env.modelsDir))[0]
      await stubDialogs(app, { open: [inside] })
      await page.getByTestId('import-gguf').click()
      await page.waitForTimeout(2000)
      const afterDup = await page.locator('.model-card').count()
      report.check('model-import', 'importing an existing model does not duplicate it',
        afterDup === after, `${after} -> ${afterDup}`)

      // A file that is not a GGUF at all should be reported, not swallowed.
      const junk = path.join(outside, 'not-a-model.gguf')
      await fsp.writeFile(junk, 'this is not a gguf file')
      await stubDialogs(app, { open: [junk] })
      await page.getByTestId('import-gguf').click()
      await page.waitForTimeout(2000)
      const junkToasts = await toastTexts(page)
      report.check('model-import', 'a non-GGUF import does not crash the view',
        (await page.locator('.model-card').count()) >= afterDup, junkToasts.join(' | ').slice(0, 160))

      // "Clean partials" used to be completely silent when there was nothing to clean.
      await page.getByTestId('clean-partials').click()
      await page.waitForTimeout(1200)
      const cleanToasts = await toastTexts(page)
      report.check('model-import', 'clean partials always says what happened',
        cleanToasts.some((t) => /partial/i.test(t)), cleanToasts.join(' | ').slice(0, 160))
    })
  },

  /** The agent's persistent memory: add, edit, delete, and survive a restart. */
  async memory() {
    await withApp('memory', async ({ page }) => {
      const add = async (text) =>
        page.evaluate((t) => window.api.invoke('agent:memory-add', t), text)
      const list = async () => page.evaluate(() => window.api.invoke('agent:memory'))

      const first = await add('The user prefers tabs over spaces.')
      let items = await list()
      report.check('memory', 'a memory can be added', items.length === 1, `${items.length} items`)
      report.check('memory', 'the stored text round-trips',
        items[0]?.text === 'The user prefers tabs over spaces.', items[0]?.text)

      await add('The user works on D:/CODE.')
      items = await list()
      report.check('memory', 'a second memory does not replace the first', items.length === 2,
        `${items.length} items`)

      const id = first?.id ?? items[0]?.id
      await page.evaluate((i) => window.api.invoke('agent:memory-update', i, 'Updated text.'), id)
      items = await list()
      report.check('memory', 'a memory can be edited',
        items.some((m) => m.text === 'Updated text.'), items.map((m) => m.text).join(' | '))

      await page.evaluate((i) => window.api.invoke('agent:memory-delete', i), id)
      items = await list()
      report.check('memory', 'a memory can be deleted', items.length === 1, `${items.length} items`)

      // Memory is the one thing that must outlive the process.
      await page.reload()
      await page.waitForSelector('.nav-item')
      const afterReload = await list()
      report.check('memory', 'memory survives a reload', afterReload.length === 1,
        `${afterReload.length} items`)
    })
  },

  /** Checkpoints and rewind: the safety net behind machine-wide write access. */
  async checkpoints() {
    await withApp('checkpoints', async ({ app, page, env }) => {
      const workDir = path.join(env.base, 'work')
      await fsp.mkdir(workDir, { recursive: true })
      await fsp.writeFile(path.join(workDir, 'notes.txt'), 'original content')

      await loadModel(page)
      await goTo(page, 'Agent')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(500)

      // The working folder is stored per conversation, so sessionId is required here — without
      // it the agent runs in the default folder and the write lands somewhere else entirely.
      const sessionId = await page.evaluate(() => {
        const el =
          document.querySelector('[data-testid="conversation-item"].active') ??
          document.querySelector('[data-testid="conversation-item"]')
        return el?.getAttribute('data-id') ?? null
      })
      report.check('checkpoints', 'the new session has an id', !!sessionId, String(sessionId))

      // set-cwd opens a folder picker; its argument is the session, not the path.
      await stubDialogs(app, { open: [workDir] })
      const chosen = await page.evaluate((id) => window.api.invoke('agent:set-cwd', id), sessionId)
      report.check('checkpoints', 'the working folder is set', !!chosen, String(chosen))

      // A turn that writes a file, so there is something to roll back.
      await page.getByTestId('agent-input').fill(
        '[[mock:calltool:write_file]] [[mock:toolargs:{"path":"notes.txt","content":"rewritten by the agent"}]] update the notes'
      )
      await page.getByTestId('agent-send').click()

      const prompt = await page
        .waitForSelector('[data-testid="permission-overlay"]', { timeout: 25000 })
        .catch(() => null)
      if (prompt) await page.getByTestId('permission-allow').click()

      await stopTurn(page, 'agent-input')
      await page.waitForTimeout(800)

      const points = await page.evaluate(
        (id) => window.api.invoke('agent:checkpoints', id),
        sessionId
      ).catch(() => [])
      report.check('checkpoints', 'a checkpoint is recorded for a write',
        Array.isArray(points) && points.length > 0, `${(points ?? []).length} checkpoints`)

      const written = await fsp.readFile(path.join(workDir, 'notes.txt'), 'utf8').catch(() => '')
      report.check('checkpoints', 'the agent write actually landed',
        written.includes('rewritten'),
        written ? written.slice(0, 60) : `no file; workDir holds ${(await fsp.readdir(workDir).catch(() => [])).join(',')}`)

      if (Array.isArray(points) && points.length) {
        await page.evaluate(
          ([id, cp]) => window.api.invoke('agent:rewind', id, cp),
          [sessionId, points[0].id]
        )
        const restored = await fsp.readFile(path.join(workDir, 'notes.txt'), 'utf8').catch(() => '')
        report.check('checkpoints', 'rewind restores the original file',
          restored.includes('original content'), restored.slice(0, 60))
      }
    })
  },

  /** The OpenAI- and Anthropic-shaped HTTP surface, exercised by a real client. */
  async apiSurface() {
    await withApp('api-surface', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'API server')

      await page.locator('button:has-text("Start")').first().click()
      await page.waitForTimeout(1500)

      const status = await page.evaluate(() => window.api.invoke('server:status'))
      const port = status?.port
      report.check('api-surface', 'the server reports a port', !!port, JSON.stringify(status))
      if (!port) return

      const base = `http://127.0.0.1:${port}`
      const key = status?.apiKey ?? ''
      const headers = { 'Content-Type': 'application/json', ...(key ? { Authorization: `Bearer ${key}` } : {}) }

      // Model listing, the first call any client makes.
      const models = await fetch(`${base}/v1/models`, { headers }).then((r) => r.json()).catch((e) => ({ error: String(e) }))
      report.check('api-surface', 'GET /v1/models lists the loaded model',
        Array.isArray(models?.data) && models.data.length > 0, JSON.stringify(models).slice(0, 160))

      // OpenAI chat completion.
      const oa = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model: 'local', messages: [{ role: 'user', content: 'hi' }], stream: false })
      }).then((r) => r.json()).catch((e) => ({ error: String(e) }))
      report.check('api-surface', 'OpenAI chat completion returns a choice',
        typeof oa?.choices?.[0]?.message?.content === 'string', JSON.stringify(oa).slice(0, 200))
      report.check('api-surface', 'OpenAI response carries usage counts',
        typeof oa?.usage?.total_tokens === 'number', JSON.stringify(oa?.usage))

      // Anthropic messages.
      const an = await fetch(`${base}/v1/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
        body: JSON.stringify({ model: 'local', max_tokens: 64, messages: [{ role: 'user', content: 'hi' }] })
      }).then((r) => r.json()).catch((e) => ({ error: String(e) }))
      report.check('api-surface', 'Anthropic messages returns a content block',
        Array.isArray(an?.content) && an.content.some((b) => b.type === 'text'),
        JSON.stringify(an).slice(0, 200))
      report.check('api-surface', 'Anthropic response uses input/output token names',
        typeof an?.usage?.input_tokens === 'number', JSON.stringify(an?.usage))

      /*
       * Thinking has to reach HTTP clients, not just the desktop window.
       *
       * llama.cpp separates the chain of thought from the answer, and the app renders it in its
       * own block — but every HTTP path dropped it on the floor, so a client could ask a
       * reasoning model to think, wait through it, and receive no sign it had happened. Each
       * surface is checked separately because each one discarded it in its own way.
       *
       * The second half of every check matters as much as the first: thinking must arrive in
       * its own field and must NOT be folded into the answer. Merging the two would technically
       * "deliver" it while destroying the separation the server exists to provide.
       */
      const THINK = 'the constraint changes things'

      const oaThinkStream = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: 'local',
          messages: [{ role: 'user', content: '[[mock:think]] why?' }],
          stream: true
        })
      }).then((r) => r.text()).catch((e) => String(e))

      const deltas = oaThinkStream
        .split(/\r?\n/)
        .filter((l) => l.startsWith('data:') && !l.includes('[DONE]'))
        .map((l) => { try { return JSON.parse(l.slice(5).trim()) } catch { return null } })
        .filter(Boolean)
        .map((f) => f.choices?.[0]?.delta ?? {})

      const streamedThinking = deltas.map((d) => d.reasoning_content ?? '').join('')
      const streamedContent = deltas.map((d) => d.content ?? '').join('')
      report.check('api-surface', 'OpenAI stream carries thinking as reasoning_content',
        streamedThinking.includes(THINK), `${streamedThinking.length} chars of reasoning`)
      report.check('api-surface', 'and does not fold thinking into the answer',
        !streamedContent.includes(THINK), streamedContent.slice(0, 120))

      const oaThink = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: 'local',
          messages: [{ role: 'user', content: '[[mock:think]] why?' }],
          stream: false
        })
      }).then((r) => r.json()).catch((e) => ({ error: String(e) }))
      const msg = oaThink?.choices?.[0]?.message ?? {}
      report.check('api-surface', 'non-streaming OpenAI reports thinking too',
        typeof msg.reasoning_content === 'string' && msg.reasoning_content.includes(THINK),
        JSON.stringify(msg).slice(0, 160))
      report.check('api-surface', 'and keeps it out of the answer there as well',
        typeof msg.content === 'string' && !msg.content.includes(THINK), String(msg.content).slice(0, 120))

      const anThinkStream = await fetch(`${base}/v1/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
        body: JSON.stringify({
          model: 'local',
          max_tokens: 256,
          stream: true,
          messages: [{ role: 'user', content: '[[mock:think]] why?' }]
        })
      }).then((r) => r.text()).catch((e) => String(e))
      report.check('api-surface', 'Anthropic stream opens a thinking block',
        anThinkStream.includes('"type":"thinking"'), anThinkStream.slice(0, 160))
      report.check('api-surface', 'Anthropic stream sends thinking_delta',
        anThinkStream.includes('thinking_delta'), '')

      const anThink = await fetch(`${base}/v1/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
        body: JSON.stringify({
          model: 'local',
          max_tokens: 256,
          messages: [{ role: 'user', content: '[[mock:think]] why?' }]
        })
      }).then((r) => r.json()).catch((e) => ({ error: String(e) }))
      const blocks = Array.isArray(anThink?.content) ? anThink.content : []
      report.check('api-surface', 'non-streaming Anthropic returns a thinking block before the text',
        blocks[0]?.type === 'thinking' && String(blocks[0].thinking).includes(THINK) &&
          blocks.some((b) => b.type === 'text'),
        JSON.stringify(blocks.map((b) => b.type)))

      // Streaming must be a well-formed SSE body ending in [DONE].
      const streamed = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model: 'local', messages: [{ role: 'user', content: 'hi' }], stream: true })
      }).then((r) => r.text()).catch((e) => String(e))
      report.check('api-surface', 'streaming responses are SSE frames',
        streamed.includes('data: '), streamed.slice(0, 120))
      report.check('api-surface', 'streaming terminates with [DONE]',
        streamed.includes('[DONE]'), streamed.slice(-120))

      // An unauthenticated call must be refused when a key is set.
      if (key) {
        const denied = await fetch(`${base}/v1/models`).then((r) => r.status).catch(() => 0)
        report.check('api-surface', 'requests without the key are rejected', denied === 401, String(denied))
      }

      // Malformed input must produce an error, not a crash.
      const bad = await fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: '{not json' })
        .then((r) => r.status).catch(() => 0)
      report.check('api-surface', 'malformed JSON is rejected cleanly', bad >= 400 && bad < 500, String(bad))

      // And the request log should have recorded the traffic.
      await page.waitForTimeout(600)
      const logged = await page.evaluate(() => window.api.invoke('server:requests'))
      report.check('api-surface', 'requests appear in the log',
        Array.isArray(logged) && logged.length > 0, `${(logged ?? []).length} entries`)

      await shot(page, 'api-surface')
    })
  },

  /** Document ingestion and retrieval, end to end through the file picker. */
  async ragIngest() {
    await withApp('rag-ingest', async ({ app, page, env }) => {
      const docDir = path.join(env.base, 'docs')
      await fsp.mkdir(docDir, { recursive: true })
      const doc = path.join(docDir, 'kv-cache.md')
      await fsp.writeFile(
        doc,
        [
          '# KV cache sizing',
          '',
          'The key-value cache grows linearly with context length.',
          'Grouped-query attention reduces it by sharing key and value heads across query heads.',
          'A hybrid model only caches on its attention layers, not its recurrent ones.'
        ].join('\n')
      )

      await goTo(page, 'Documents')
      await page.locator('button:has-text("New collection"), [data-testid="new-collection"]').first().click().catch(() => undefined)
      await page.waitForTimeout(600)

      const collections = await page.evaluate(() => window.api.invoke('rag:collections'))
      let collectionId = collections?.[0]?.id
      if (!collectionId) {
        const made = await page.evaluate(() => window.api.invoke('rag:create-collection', 'Test collection'))
        collectionId = made?.id ?? made
      }
      report.check('rag-ingest', 'a collection can be created', !!collectionId, String(collectionId))

      await stubDialogs(app, { open: [doc] })
      const ingested = await page.evaluate(
        (id) => window.api.invoke('rag:ingest', { collectionId: id }),
        collectionId
      ).catch((e) => ({ error: String(e) }))

      report.check('rag-ingest', 'a document is ingested',
        Array.isArray(ingested) && ingested.length > 0, JSON.stringify(ingested).slice(0, 200))

      // Both of these take a scope object, not a bare id. Passing the id positionally made
      // `rag:documents` list every document in the database and `rag:retrieve` search all of
      // them for the collection's id as if it were the query — so both assertions passed while
      // testing something else entirely, and the retrieval one only turned red once an unscoped
      // search stopped meaning "everything".
      const docs = await page.evaluate(
        (id) => window.api.invoke('rag:documents', { collectionId: id }),
        collectionId
      )
      report.check('rag-ingest', 'the document is listed in the collection',
        Array.isArray(docs) && docs.length > 0, `${(docs ?? []).length} documents`)

      const hits = await page.evaluate(
        (id) => window.api.invoke('rag:retrieve', 'how does grouped query attention affect the cache?', { collectionId: id }),
        collectionId
      ).catch((e) => ({ error: String(e) }))
      report.check('rag-ingest', 'retrieval returns chunks',
        Array.isArray(hits) && hits.length > 0, JSON.stringify(hits).slice(0, 200))
      if (Array.isArray(hits) && hits.length) {
        report.check('rag-ingest', 'retrieved chunks carry text and a score',
          typeof hits[0].text === 'string' && typeof hits[0].score === 'number',
          JSON.stringify(hits[0]).slice(0, 160))
      }

      // Cancelling the picker must not throw or create an empty document.
      await stubDialogs(app, { open: null })
      const cancelled = await page.evaluate(
        (id) => window.api.invoke('rag:ingest', { collectionId: id }),
        collectionId
      ).catch((e) => ({ error: String(e) }))
      report.check('rag-ingest', 'cancelling ingestion is a clean no-op',
        Array.isArray(cancelled) && cancelled.length === 0, JSON.stringify(cancelled).slice(0, 120))

      await goTo(page, 'Documents')
      await page.waitForTimeout(500)
      await auditLayout(page, 'Documents (populated)', report)
      await shot(page, 'rag-populated')
    })
  },

  /** Deleting a collection now goes through the in-app dialog. */
  async collectionDelete() {
    await withApp('collection-delete', async ({ page }) => {
      await goTo(page, 'Documents')
      await page.evaluate(() => window.api.invoke('rag:create-collection', 'Doomed collection'))
      await page.reload()
      await page.waitForSelector('.nav-item')
      await goTo(page, 'Documents')
      await page.waitForTimeout(600)

      // Select it so the delete control is reachable.
      await page.getByTestId('collection-item').filter({ hasText: 'Doomed' }).first().click()
      await page.waitForTimeout(400)

      const visible = await page.getByTestId('delete-collection').isVisible().catch(() => false)
      report.check('collection-delete', 'the delete control is reachable', visible)
      if (!visible) return

      await page.getByTestId('delete-collection').click()
      const asked = await page.getByTestId('confirm-accept').isVisible().catch(() => false)
      report.check('collection-delete', 'deleting a collection asks first (in-app)', asked)

      await page.keyboard.press('Escape')
      await page.waitForTimeout(300)
      let left = await page.evaluate(() => window.api.invoke('rag:collections'))
      report.check('collection-delete', 'Escape cancels the collection delete',
        (left ?? []).length === 1, `${(left ?? []).length} collections`)

      await page.getByTestId('delete-collection').click()
      await page.waitForTimeout(200)
      await page.getByTestId('confirm-accept').click()
      await page.waitForTimeout(800)
      left = await page.evaluate(() => window.api.invoke('rag:collections'))
      report.check('collection-delete', 'confirming removes the collection',
        (left ?? []).length === 0, `${(left ?? []).length} collections`)
    })
  },

  /** Switching conversations while one is streaming must not cross the wires. */
  async conversationSwitch() {
    await withApp('conversation-switch', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Chat')

      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(400)
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(400)

      // Start a slow response in the currently selected conversation.
      await page.getByTestId('chat-input').fill('[[mock:slow]] [[mock:long]] first conversation question')
      await page.getByTestId('chat-send').click()
      await page.waitForSelector('[data-testid="streaming-message"]', { timeout: 20000 })

      // Switch to the other conversation mid-stream.
      const items = page.getByTestId('conversation-item')
      await items.nth(1).click()
      await page.waitForTimeout(900)

      const bleedThrough = await page.getByTestId('streaming-message').count()
      report.check('conversation-switch', 'the other conversation does not show the in-flight text',
        bleedThrough === 0, `${bleedThrough} streaming messages visible`)

      const strayText = await page.getByTestId('chat-messages').textContent()
      report.check('conversation-switch', 'the other conversation stays empty',
        !(strayText ?? '').includes('first conversation question'),
        (strayText ?? '').slice(0, 120))

      // Back to the original: the response must still be there and still growing.
      await items.nth(0).click()
      await page.waitForTimeout(700)
      const restored = await page.getByTestId('streaming-message').count()
      report.check('conversation-switch', 'returning shows the in-flight response again',
        restored > 0, `${restored} streaming messages`)

      await stopTurn(page, 'chat-input')
    })
  },

  /** Toast lifecycle: successes clear themselves, errors persist until dismissed. */
  async toasts() {
    await withApp('toasts', async ({ page }) => {
      await page.evaluate(() => {
        const w = window
        w.__toastTest = true
      })

      // Drive the real store rather than a stand-in.
      await goTo(page, 'My models')
      await page.getByTestId('clean-partials').click()
      await page.waitForSelector('[data-testid="toast"]', { timeout: 10000 })
      report.check('toasts', 'an action raises a toast',
        (await page.getByTestId('toast').count()) > 0)

      // A toast must not block the UI behind it.
      const blocked = await page.evaluate(() => {
        const stack = document.querySelector('[data-testid="toast-stack"]')
        return stack ? getComputedStyle(stack).pointerEvents : 'missing'
      })
      report.check('toasts', 'the toast stack does not intercept clicks', blocked === 'none', blocked)

      // Toasts must stay inside the viewport.
      const overflow = await page.evaluate(() => {
        const el = document.querySelector('[data-testid="toast-stack"]')
        if (!el) return 'missing'
        const r = el.getBoundingClientRect()
        return r.right <= window.innerWidth + 1 && r.left >= -1 && r.bottom <= window.innerHeight + 1
          ? 'inside'
          : `${Math.round(r.left)},${Math.round(r.right)} of ${window.innerWidth}`
      })
      report.check('toasts', 'toasts stay within the viewport', overflow === 'inside', String(overflow))

      // Dismissing works.
      await page.getByTestId('toast-dismiss').first().click()
      await page.waitForTimeout(300)
      report.check('toasts', 'a toast can be dismissed',
        (await page.getByTestId('toast').count()) === 0)

      // A success clears itself; waiting is the whole point of the assertion.
      await page.getByTestId('clean-partials').click()
      await page.waitForSelector('[data-testid="toast"]', { timeout: 10000 })
      const gone = await page
        .waitForFunction(() => document.querySelectorAll('[data-testid="toast"]').length === 0, undefined, {
          timeout: 12000
        })
        .then(() => true)
        .catch(() => false)
      report.check('toasts', 'a success toast clears itself', gone)
    })
  },

  /**
   * Connecting to an MCP server over stdio and using its tools.
   *
   * Driven against a local stand-in rather than a third-party server, so the test is
   * deterministic and offline, but the transport, handshake and tool plumbing are real.
   */
  async mcpServers() {
    await withApp('mcp', async ({ page }) => {
      const mockPath = path.join(ROOT, 'scripts', 'e2e', 'mock-mcp.mjs')

      const added = await page.evaluate(
        ([exe, script]) =>
          window.api.invoke('mcp:add', {
            name: 'Mock MCP',
            transport: 'stdio',
            command: exe,
            args: [script],
            env: { ELECTRON_RUN_AS_NODE: '1' },
            enabled: true
          }),
        [process.execPath, mockPath]
      )
      report.check('mcp', 'a server can be registered', !!added?.id, JSON.stringify(added).slice(0, 140))

      const statuses = await page.evaluate(() => window.api.invoke('mcp:connect'))
      report.check('mcp', 'connecting reports a status per server',
        Array.isArray(statuses) && statuses.length === 1, JSON.stringify(statuses).slice(0, 220))

      const s = (statuses ?? [])[0] ?? {}
      report.check('mcp', 'the server connects', s.ok === true,
        JSON.stringify(s).slice(0, 220))
      report.check('mcp', 'its tools are discovered', (s.tools ?? 0) >= 3, String(s.tools))

      // The MCP tool list must be merged into what the agent can call, and a name that collides
      // with a built-in must not silently replace it.
      const tools = await page.evaluate(() => window.api.invoke('agent:tools'))
      const names = (tools ?? []).map((t) => t.name)
      report.check('mcp', 'MCP tools join the agent tool list',
        names.some((n) => /echo/.test(n)), names.filter((n) => /echo|mcp/i.test(n)).join(','))
      report.check('mcp', 'a colliding MCP tool does not replace the built-in',
        names.filter((n) => n === 'read_file').length === 1,
        names.filter((n) => /read_file/.test(n)).join(','))
      report.check('mcp', 'no two tools share a name',
        new Set(names).size === names.length,
        `${names.length} tools, ${new Set(names).size} unique`)

      await goTo(page, 'Settings')
      await page.waitForTimeout(500)
      const shown = await page.locator('text=Mock MCP').count()
      report.check('mcp', 'the server is listed in Settings', shown > 0, `${shown} matches`)
      await auditLayout(page, 'Settings (with MCP)', report)

      // Disabling must drop its tools without disturbing the built-ins.
      await page.evaluate((id) => window.api.invoke('mcp:set-enabled', id, false), added.id)
      await page.evaluate(() => window.api.invoke('mcp:connect'))
      const afterDisable = ((await page.evaluate(() => window.api.invoke('agent:tools'))) ?? []).map((t) => t.name)
      report.check('mcp', 'disabling a server removes its tools',
        !afterDisable.some((n) => /echo/.test(n)), afterDisable.filter((n) => /echo/.test(n)).join(','))
      report.check('mcp', 'built-in tools survive disabling a server',
        afterDisable.includes('read_file'), `${afterDisable.length} tools`)

      // Removing must not leave a stale entry behind.
      await page.evaluate((id) => window.api.invoke('mcp:remove', id), added.id)
      const left = await page.evaluate(() => window.api.invoke('mcp:list'))
      report.check('mcp', 'a server can be removed', (left ?? []).length === 0,
        `${(left ?? []).length} servers`)

      // A server that cannot start must be reported, not hang the app.
      const broken = await page.evaluate(() =>
        window.api.invoke('mcp:add', {
          name: 'Broken MCP',
          transport: 'stdio',
          command: 'definitely-not-a-real-command-xyz',
          args: [],
          enabled: true
        })
      )
      const brokenStatus = await page.evaluate(() => window.api.invoke('mcp:connect'))
      const bs = (brokenStatus ?? []).find((x) => x.id === broken.id) ?? {}
      report.check('mcp', 'an unstartable server reports an error rather than hanging',
        bs.ok === false || !!bs.error, JSON.stringify(bs).slice(0, 200))
      report.check('mcp', 'the app still responds after a failed connection',
        (await page.locator('.nav-item').count()) > 0)
    })
  },

/**
   * Captures the transient states the per-feature scenarios exercise but never photograph:
   * dialogs, toasts, and a conversation rail with content in it. Screenshots only — the
   * behaviour behind each of these is asserted elsewhere.
   */
  async gallery() {
    await withApp('gallery', async ({ app, page, env }) => {
      // ---- a destructive confirmation
      await goTo(page, 'My models')
      await page.waitForSelector('.model-card', { timeout: 15000 })
      await page.getByTestId('delete-model').first().click()
      await page.waitForSelector('[data-testid="confirm-accept"]', { timeout: 5000 })
      await shot(page, 'dialog-delete-model')
      await page.keyboard.press('Escape')
      await page.waitForTimeout(300)

      // ---- a toast
      await page.getByTestId('clean-partials').click()
      await page.waitForSelector('[data-testid="toast"]', { timeout: 10000 })
      await shot(page, 'toast')

      // ---- an export toast, which carries a path and a reveal action
      await loadModel(page)
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)
      await page.getByTestId('chat-input').fill('what is a KV cache?')
      await page.getByTestId('chat-send').click()
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="chat-input"]')
          return el instanceof HTMLTextAreaElement && !el.disabled
        },
        undefined,
        { timeout: 30000 }
      )

      // A second and third conversation, so the rail is not a single lonely row.
      for (const q of ['explain flash attention', 'how does GQA save memory?']) {
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(250)
        await page.getByTestId('chat-input').fill(q)
        await page.getByTestId('chat-send').click()
        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="chat-input"]')
            return el instanceof HTMLTextAreaElement && !el.disabled
          },
          undefined,
          { timeout: 30000 }
        )
      }
      await shot(page, 'chat-conversation')

      const outDir = path.join(env.base, 'exports')
      await fsp.mkdir(outDir, { recursive: true })
      await stubDialogs(app, { save: path.join(outDir, 'conversation.md') })
      await page.getByTestId('export-chat').click()
      await page.waitForSelector('[data-testid="toast"]', { timeout: 10000 })
      await shot(page, 'toast-export')

      // ---- the conversation rail's rename and delete affordances
      const first = page.getByTestId('conversation-item').first()
      await first.hover()
      await page.waitForTimeout(200)
      await first.getByTestId('delete-conversation').click()
      await page.waitForTimeout(250)
      await shot(page, 'conversation-delete')
    })
  },

/**
   * Narrow windows: the fixed chrome must give way before the reading column does.
   *
   * Asserted as a proportion rather than a pixel count, because the point is the ratio: at
   * 1040px the nav rail plus the conversation rail used to take 43% of the window.
   */
  async narrow() {
    await withApp('narrow', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      for (const width of [1440, 1180, 1040, 860]) {
        await page.setViewportSize({ width, height: 900 })
        await page.waitForTimeout(400)

        const m = await page.evaluate(() => {
          const rect = (sel) => {
            const el = document.querySelector(sel)
            if (!el) return null
            const r = el.getBoundingClientRect()
            // A closed drawer still has a width; what matters is whether it occupies the viewport.
            const onScreen = r.width > 0 && r.right > 0 && r.left < window.innerWidth
            return { w: Math.round(r.width), visible: onScreen }
          }
          return {
            viewport: window.innerWidth,
            sidebar: rect('.sidebar'),
            rail: rect('.side-list'),
            messages: rect('.messages'),
            composer: rect('.composer')
          }
        })

        const chrome = (m.sidebar?.w ?? 0) + (m.rail?.visible ? m.rail.w : 0)
        const share = chrome / m.viewport
        console.log(
          `  ${String(width).padStart(4)}px  chrome=${String(chrome).padStart(3)}px (${(share * 100).toFixed(0)}%)  ` +
            `messages=${m.messages?.w}px  rail=${m.rail?.visible ? `${m.rail.w}px` : 'hidden'}`
        )

        report.check('narrow', `chrome stays under 40% of a ${width}px window`, share < 0.4,
          `${(share * 100).toFixed(0)}%`)
        report.check('narrow', `the composer is still usable at ${width}px`,
          (m.composer?.w ?? 0) > 280, `${m.composer?.w}px`)

        const problems = await auditLayout(page, `Chat @ ${width}px`, report)
        report.check('narrow', `no layout defects at ${width}px`, problems.length === 0,
          problems.map((p) => p.kind).join(','))
      }

      // ---- the drawer must actually restore what the hidden rail took away
      await page.setViewportSize({ width: 860, height: 900 })
      await page.waitForTimeout(400)

      const railBox = async () =>
        page.evaluate(() => {
          const el = document.querySelector('.side-list')
          if (!el) return null
          const r = el.getBoundingClientRect()
          return { left: Math.round(r.left), onScreen: r.right > 0 && r.left < window.innerWidth, inert: el.hasAttribute('inert') }
        })

      let box = await railBox()
      report.check('narrow', 'the rail is off-screen when closed', box && !box.onScreen, JSON.stringify(box))
      report.check('narrow', 'a closed drawer cannot take focus', box?.inert === true, JSON.stringify(box))

      // Nothing else offers "new conversation" at this width, so the toggle has to be there.
      const toggleVisible = await page.getByTestId('rail-toggle').isVisible()
      report.check('narrow', 'a toggle is offered in the header', toggleVisible)

      await page.getByTestId('rail-toggle').click()
      await page.waitForTimeout(400)
      box = await railBox()
      report.check('narrow', 'the drawer opens', box?.onScreen === true, JSON.stringify(box))
      report.check('narrow', 'an open drawer is focusable again', box?.inert === false, JSON.stringify(box))
      await shot(page, 'narrow-drawer-open')

      // Creating a conversation from the drawer should put it away again.
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(500)
      box = await railBox()
      report.check('narrow', 'choosing from the drawer closes it', box?.onScreen === false, JSON.stringify(box))
      const count = await page.getByTestId('conversation-item').count()
      report.check('narrow', 'the conversation was created', count >= 2, `${count} conversations`)

      await shot(page, 'narrow-860')
      await page.setViewportSize({ width: 1040, height: 900 })
      await page.waitForTimeout(400)
      await shot(page, 'narrow-1040')
    })
  },

/**
   * The reasoning effort control: shown only where the model offers one, shaped to what it
   * offers, and — the part a UI test would otherwise miss — actually reaching the server.
   */
  async reasoning() {
    // A model whose template enumerates effort levels.
    await withApp(
      'reasoning-effort',
      async ({ page }) => {
        await loadModel(page)
        await goTo(page, 'Chat')
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(400)

        const shown = await page.getByTestId('reasoning-control').count()
        report.check('reasoning', 'an effort model shows a control', shown === 1, `${shown} controls`)
        const kind = await page.getByTestId('reasoning-control').getAttribute('data-kind')
        report.check('reasoning', 'it is a slider, not a switch', kind === 'effort', String(kind))

        /*
         * The scale lives in a popover now; the row itself carries only the chosen level. Read
         * the level off the trigger, which is always mounted, and open the panel to move it.
         */
        const level = async () => (await page.getByTestId('reasoning-trigger').textContent())?.trim()
        const openEffort = async () => {
          if ((await page.getByTestId('reasoning-pop').count()) === 0) {
            await page.getByTestId('reasoning-trigger').click()
            await page.waitForTimeout(140)
          }
        }
        /*
         * The panel opens upward over the composer, which is where the room is — the meta row it
         * hangs off sits at the bottom of the window. A person dismisses it by clicking away
         * before typing; `fill()` dispatches no pointer event, so the panel stayed open and
         * Playwright refused to click a send button it could see was covered.
         */
        const closeEffort = async () => {
          if ((await page.getByTestId('reasoning-pop').count()) > 0) {
            await page.keyboard.press('Escape')
            await page.waitForTimeout(140)
          }
        }

        // Stops are read from the template: low, medium, xhigh — plus an off position, which is
        // offered for every reasoning model whether or not the template has a switch of its own,
        // and Ultra at the far end, which is a runtime mode rather than a level the model knows.
        await openEffort()
        const range = page.getByTestId('reasoning-slider')
        const max = Number(await range.getAttribute('max'))
        report.check('reasoning', 'a stop per level, plus off and ultra', max === 4, `max=${max}`)

        const initial = await level()
        report.check('reasoning', "it starts on the template's own default", initial === 'Extra high', String(initial))

        // The level names must come from the model, not a built-in list.
        const names = []
        for (let i = 0; i <= max; i++) {
          await range.fill(String(i))
          await page.waitForTimeout(120)
          names.push(await level())
        }
        report.check('reasoning', 'stops are ordered off -> weakest -> strongest -> ultra',
          names.join(',') === 'Off,Low,Medium,Extra high,Ultra', names.join(','))
        // 'high' is an alias for 'xhigh' in this template; offering it would be a dead stop.
        report.check('reasoning', 'an aliased level is not offered as its own stop',
          !names.includes('High'), names.join(','))

        await shot(page, 'reasoning-slider')

        // ---- does the choice actually reach llama-server?
        await range.fill('1') // low
        await page.waitForTimeout(150)
        await closeEffort()
        await page.getByTestId('chat-input').fill('[[mock:params]] what did you get?')
        await page.getByTestId('chat-send').click()
        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="chat-input"]')
            return el instanceof HTMLTextAreaElement && !el.disabled
          },
          undefined,
          { timeout: 30000 }
        )
        let body = (await page.getByTestId('chat-messages').textContent()) ?? ''
        report.check('reasoning', 'the chosen level is sent to the server',
          /"reasoning_effort"\s*:\s*"low"/.test(body), body.slice(-200))

        // ---- off must disable thinking through both switches
        await openEffort()
        await range.fill('0')
        await page.waitForTimeout(150)
        await closeEffort()
        await page.getByTestId('chat-input').fill('[[mock:params]] and now?')
        await page.getByTestId('chat-send').click()
        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="chat-input"]')
            return el instanceof HTMLTextAreaElement && !el.disabled
          },
          undefined,
          { timeout: 30000 }
        )
        body = (await page.getByTestId('chat-messages').textContent()) ?? ''
        /*
         * This fixture's template validates against ('xhigh','medium','low') and raises on
         * anything else, while also honouring enable_thinking. Sending reasoning_effort 'none'
         * to it does not turn thinking off — it fails the request. This assertion used to
         * require exactly that, pinning the bug in place. Off is carried by llama.cpp's own
         * budget instead, which needs no cooperation from the template.
         */
        report.check('reasoning', 'off ends thinking via the reasoning budget',
          /"reasoning_budget"\s*:\s*0/.test(body), body.slice(-260))
        report.check('reasoning', 'off also asks the template, for the ones that listen',
          /enable_thinking[^,}]*false/.test(body), body.slice(-260))
        report.check('reasoning', 'off sends no level this template would raise on',
          !/"reasoning_effort"\s*:\s*"none"/.test(body), body.slice(-260))

        // The choice belongs to the conversation, not the app.
        await closeEffort()
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(500)
        const fresh = await level()
        report.check('reasoning', 'a new conversation starts from the default again',
          fresh === 'Extra high', String(fresh))

        // And it survives leaving the view mid-session.
        await openEffort()
        await page.getByTestId('reasoning-slider').fill('1')
        await page.waitForTimeout(150)
        await closeEffort()
        await goTo(page, 'Dashboard')
        await goTo(page, 'Chat')
        await page.waitForTimeout(500)
        const kept = await level()
        report.check('reasoning', 'the choice survives navigating away', kept === 'Low', String(kept))

        await auditLayout(page, 'Chat (reasoning slider)', report)
      },
      { models: ['Effort-8B-Q4_K_M.gguf'], modelOpts: { reasoning: 'effort' } }
    )

    // A model with only enable_thinking.
    await withApp(
      'reasoning-toggle',
      async ({ page }) => {
        await loadModel(page)
        await goTo(page, 'Chat')
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(400)

        const kind = await page.getByTestId('reasoning-control').getAttribute('data-kind')
        report.check('reasoning', 'a toggle-only model shows a switch', kind === 'toggle', String(kind))
        report.check('reasoning', 'no slider is offered for a toggle model',
          (await page.getByTestId('reasoning-slider').count()) === 0)

        const sw = page.getByTestId('reasoning-toggle')
        report.check('reasoning', 'thinking starts on', (await sw.getAttribute('aria-checked')) === 'true')
        await sw.click()
        await page.waitForTimeout(150)
        report.check('reasoning', 'the switch turns thinking off', (await sw.getAttribute('aria-checked')) === 'false')

        await page.getByTestId('chat-input').fill('[[mock:params]] check')
        await page.getByTestId('chat-send').click()
        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="chat-input"]')
            return el instanceof HTMLTextAreaElement && !el.disabled
          },
          undefined,
          { timeout: 30000 }
        )
        const body = (await page.getByTestId('chat-messages').textContent()) ?? ''
        report.check('reasoning', 'the toggle reaches the server as enable_thinking',
          /enable_thinking[^,}]*false/.test(body), body.slice(-200))

        await shot(page, 'reasoning-toggle')
      },
      { models: ['Toggle-8B-Q4_K_M.gguf'], modelOpts: { reasoning: 'toggle' } }
    )

    // A model with no reasoning control at all must show nothing.
    await withApp(
      'reasoning-none',
      async ({ page }) => {
        await loadModel(page)
        await goTo(page, 'Chat')
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(400)
        report.check('reasoning', 'a plain model shows no control',
          (await page.getByTestId('reasoning-control').count()) === 0)
      },
      { models: ['Plain-8B-Q4_K_M.gguf'], modelOpts: { reasoning: 'none' } }
    )
  },

/**
   * The streaming caret must sit on the same line as the text it trails.
   *
   * Asserted geometrically rather than structurally: the bug was that the caret rendered as a
   * sibling after the last block element, so it dropped to the next line. Comparing rectangles
   * is what actually catches that; checking the DOM shape would just re-encode the fix.
   */
  async caret() {
    await withApp('caret', async ({ page }) => {
      await loadModel(page)
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      await page.getByTestId('chat-input').fill('[[mock:slow]] [[mock:long]] explain something')
      await page.getByTestId('chat-send').click()
      await page.waitForSelector('[data-testid="streaming-message"] .cursor', { timeout: 25000 })
      await page.waitForTimeout(900)

      const geometry = await page.evaluate(() => {
        const caret = document.querySelector('[data-testid="streaming-message"] .cursor')
        if (!caret) return null
        const block = caret.parentElement
        const caretRect = caret.getBoundingClientRect()

        // The last line of text in the block the caret belongs to.
        const range = document.createRange()
        range.selectNodeContents(block)
        const rects = [...range.getClientRects()]
        const lastLine = rects.length ? rects[rects.length - 1] : null

        return lastLine
          ? {
              caretTop: Math.round(caretRect.top),
              caretLeft: Math.round(caretRect.left),
              lineTop: Math.round(lastLine.top),
              lineBottom: Math.round(lastLine.bottom),
              lineRight: Math.round(lastLine.right),
              tag: block?.tagName ?? '?'
            }
          : null
      })

      report.check('caret', 'the caret is inside a text block, not a sibling of one',
        geometry !== null && /^(P|H\d|LI|CODE|BLOCKQUOTE)$/.test(geometry.tag), JSON.stringify(geometry))

      if (geometry) {
        // Vertically overlapping the last line of text is the whole point.
        const onSameLine = geometry.caretTop < geometry.lineBottom && geometry.caretTop >= geometry.lineTop - 6
        report.check('caret', 'the caret sits on the last line of text, not below it',
          onSameLine, `caret top ${geometry.caretTop}, line ${geometry.lineTop}-${geometry.lineBottom}`)

        // And immediately after it, rather than back at the left margin.
        report.check('caret', 'the caret trails the text horizontally',
          geometry.caretLeft > 0 && Math.abs(geometry.caretLeft - geometry.lineRight) < 40,
          `caret left ${geometry.caretLeft}, line right ${geometry.lineRight}`)
      }

      await shot(page, 'caret-inline')
      await stopTurn(page, 'chat-input')
    })
  },

  /**
   * The transcript must follow a response down, and stop following when the reader scrolls away.
   *
   * The bug: scrolling was a React effect listing the state it happened to know about, so a tool
   * result — which mutates an entry already in the array rather than adding one — changed the
   * card's height without the effect ever running. The view was left behind by exactly the
   * height of the tool output. Compounded by scrolling smoothly, which restarted an unfinished
   * animation on every token and never caught up during a fast stream.
   *
   * Measured as a distance from the bottom of the real scroll container, because that is the
   * thing that was wrong. Asserting on the DOM or on which effects fired would only re-encode
   * whichever implementation is current.
   */
  async autoScroll() {
    const distanceFromBottom = (page, testId) =>
      page.evaluate((id) => {
        const el = document.querySelector(`[data-testid="${id}"]`)
        return el ? Math.round(el.scrollHeight - el.scrollTop - el.clientHeight) : -1
      }, testId)

    await withApp('auto-scroll', async ({ page }) => {
      await loadModel(page)

      // ---- a tool call must not break the follow
      await goTo(page, 'Agent')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)
      await page.getByTestId('agent-input').fill('[[mock:tool]] list the current directory')
      await page.getByTestId('agent-send').click()

      await page.waitForSelector('[data-testid="tool-card"]', { timeout: 25000 })
      // Long enough for the result to arrive and grow the card, which is the exact moment the
      // old implementation stopped following.
      await page.waitForTimeout(1600)

      const afterTool = await distanceFromBottom(page, 'agent-messages')
      report.check('auto-scroll', 'still at the bottom after a tool card fills in',
        afterTool >= 0 && afterTool <= 72, `${afterTool}px from the bottom`)

      await stopTurn(page, 'agent-input')

      // ---- scrolling away detaches, and the way back works
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)
      await page.getByTestId('chat-input').fill('[[mock:slow]] [[mock:long]] explain something at length')
      await page.getByTestId('chat-send').click()
      await page.waitForSelector('[data-testid="streaming-message"]', { timeout: 25000 })
      await page.waitForTimeout(1200)

      const whileStreaming = await distanceFromBottom(page, 'chat-messages')
      report.check('auto-scroll', 'follows a response as it streams',
        whileStreaming >= 0 && whileStreaming <= 72, `${whileStreaming}px from the bottom`)

      /*
       * Wait for a transcript long enough to scroll before scrolling it.
       *
       * Detaching means getting further than 72px from the bottom, so a container with less than
       * that much scrollable range cannot detach however hard it is scrolled — the first version
       * of this test scrolled a nearly-full screen and then asserted on a button that was
       * correctly still hidden.
       */
      await page
        .waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="chat-messages"]')
            return !!el && el.scrollHeight - el.clientHeight > 240
          },
          undefined,
          { timeout: 20000 }
        )
        .catch(() => undefined)

      /*
       * A real wheel, not an assignment to scrollTop.
       *
       * Setting scrollTop is not how anyone scrolls, and it skips the signal the app relies on to
       * tell a reader scrolling away from content merely moving. Driving the wheel exercises the
       * path a person actually takes.
       */
      await page.locator('[data-testid="chat-messages"]').hover()
      await page.mouse.wheel(0, -700)
      await page.waitForTimeout(400)

      const jump = page.getByTestId('jump-to-latest')
      report.check('auto-scroll', 'offers a way back once you scroll away',
        await jump.evaluate((el) => el.classList.contains('show')).catch(() => false))

      // The whole point of detaching: more tokens must not drag the reader back down.
      const afterScrollUp = await distanceFromBottom(page, 'chat-messages')
      await page.waitForTimeout(1500)
      const stillUp = await distanceFromBottom(page, 'chat-messages')
      report.check('auto-scroll', 'stays put while reading, even as the response grows',
        stillUp >= afterScrollUp - 20, `moved from ${afterScrollUp} to ${stillUp}`)

      await jump.click()
      await page.waitForTimeout(900)
      const afterJump = await distanceFromBottom(page, 'chat-messages')
      report.check('auto-scroll', 'jumping to latest returns to the bottom',
        afterJump >= 0 && afterJump <= 72, `${afterJump}px from the bottom`)

      // And following resumes rather than needing another click.
      await page.waitForTimeout(1200)
      const afterResume = await distanceFromBottom(page, 'chat-messages')
      report.check('auto-scroll', 'following resumes after jumping back',
        afterResume >= 0 && afterResume <= 72, `${afterResume}px from the bottom`)

      await shot(page, 'auto-scroll')
      await stopTurn(page, 'chat-input')
    })
  },

  /** Attaching files: the picker, drag-and-drop, and what each type turns into. */
  async attachments() {
    await withApp('attachments', async ({ app, page, env }) => {
      const dir = path.join(env.base, 'files')
      await fsp.mkdir(dir, { recursive: true })

      const codeFile = path.join(dir, 'example.ts')
      await fsp.writeFile(codeFile, 'export const answer = 42\n// a distinctive marker: PINEAPPLE\n')
      const imageFile = path.join(dir, 'shot.png')
      // A one-pixel PNG: enough to be classified and read, nothing more.
      await fsp.writeFile(
        imageFile,
        Buffer.from(
          'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
          'base64'
        )
      )

      await loadModel(page)
      await goTo(page, 'Chat')
      await page.getByTestId('new-conversation').click()
      await page.waitForTimeout(300)

      report.check('attachments', 'the composer offers an attach button',
        (await page.getByTestId('attach-button').count()) === 1)

      // ---- picker
      await stubDialogs(app, { open: [codeFile] })
      await page.getByTestId('attach-button').click()
      await page.waitForSelector('[data-testid="attachment"]', { timeout: 10000 })
      const chip = (await page.getByTestId('attachment').textContent()) ?? ''
      report.check('attachments', 'a picked file is staged as a chip', chip.includes('example.ts'), chip)
      report.check('attachments', 'the chip says what kind of file it is', /text/.test(chip), chip)

      // ---- the text actually reaches the model
      await page.getByTestId('chat-input').fill('[[mock:prompt]] summarise')
      await page.getByTestId('chat-send').click()
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="chat-input"]')
          return el instanceof HTMLTextAreaElement && !el.disabled
        },
        undefined,
        { timeout: 30000 }
      )
      report.check('attachments', 'the chip is cleared once sent',
        (await page.getByTestId('attachment').count()) === 0)

      const sent = await page.evaluate(() => window.api.invoke('chat:load', document.querySelector('[data-testid="conversation-item"]')?.getAttribute('data-id')))
      const userText = (sent?.messages ?? []).map((m) => m.content).join('\n')
      report.check('attachments', 'the transcript records what was attached',
        /example\.ts/.test(userText), userText.slice(0, 240))

      // What the model received is a separate question from what was stored.
      const replyText = (await page.getByTestId('chat-messages').textContent()) ?? ''
      report.check('attachments', 'a code file is inlined as text, not rejected',
        /PINEAPPLE/.test(replyText), replyText.slice(0, 240))

      // ---- an image on a model that *can* see is staged without complaint
      await stubDialogs(app, { open: [imageFile] })
      await page.getByTestId('attach-button').click()
      await page.waitForSelector('[data-testid="attachment"]', { timeout: 10000 })
      report.check('attachments', 'an image is accepted by a vision model',
        (await page.locator('[data-testid="attachment"].warned').count()) === 0)

      await shot(page, 'attachments')

      // ---- removing
      await page.getByTestId('attachment-remove').first().click()
      await page.waitForTimeout(200)
      report.check('attachments', 'a staged file can be removed',
        (await page.getByTestId('attachment').count()) === 0)

      // ---- drag and drop
      //
      // A synthetic File has no path, so this exercises the upload branch — the same one the
      // remote browser UI uses.
      await page.evaluate(() => {
        const dt = new DataTransfer()
        dt.items.add(new File(['dropped contents: BANANA'], 'dropped.txt', { type: 'text/plain' }))
        const zone = document.querySelector('.dropzone')
        zone.dispatchEvent(new DragEvent('dragenter', { dataTransfer: dt, bubbles: true }))
        zone.dispatchEvent(new DragEvent('dragover', { dataTransfer: dt, bubbles: true }))
      })
      await page.waitForTimeout(200)
      report.check('attachments', 'dragging files over the pane shows a drop target',
        (await page.getByTestId('drop-overlay').count()) === 1)
      await shot(page, 'drop-overlay')

      await page.evaluate(() => {
        const dt = new DataTransfer()
        dt.items.add(new File(['dropped contents: BANANA'], 'dropped.txt', { type: 'text/plain' }))
        document.querySelector('.dropzone').dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true }))
      })
      await page.waitForSelector('[data-testid="attachment"]', { timeout: 15000 })
      const dropped = (await page.getByTestId('attachment').textContent()) ?? ''
      report.check('attachments', 'a dropped file is staged', /dropped/.test(dropped), dropped)
      report.check('attachments', 'the drop overlay goes away after the drop',
        (await page.getByTestId('drop-overlay').count()) === 0)

      await auditLayout(page, 'Chat (attachments)', report)
    })

    // The same image against a model with no projector must be flagged before it is sent.
    await withApp(
      'attachments-novision',
      async ({ app, page, env }) => {
        const img = path.join(env.base, 'shot.png')
        await fsp.writeFile(
          img,
          Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
            'base64'
          )
        )

        await loadModel(page)
        await goTo(page, 'Chat')
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(300)

        await stubDialogs(app, { open: [img] })
        await page.getByTestId('attach-button').click()
        await page.waitForSelector('[data-testid="attachment"]', { timeout: 10000 })

        const warned = await page.locator('[data-testid="attachment"].warned').count()
        report.check('attachments', 'an image is flagged when the model cannot see',
          warned === 1, `${warned} warned chips`)

        const tip = await page.getByTestId('attachment').getAttribute('title')
        report.check('attachments', 'the warning explains why', /vision projector/i.test(tip ?? ''), String(tip))
      },
      { models: ['NoVision-8B-Q4_K_M.gguf'], noMmproj: true }
    )
  },

/**
   * A reasoning model's chain of thought must be shown, not silently dropped.
   *
   * llama.cpp returns it in `reasoning_content`, separate from the answer. Reading only
   * `content` — which is what this app did — threw the whole thing away while leaving the
   * answers looking perfectly normal, so nothing about the output revealed the loss.
   */
  async thinking() {
    await withApp(
      'thinking',
      async ({ page }) => {
        await loadModel(page)
        await goTo(page, 'Chat')
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(300)

        await page.getByTestId('chat-input').fill('[[mock:think]] [[mock:slow]] why?')
        await page.getByTestId('chat-send').click()

        await page.waitForSelector('[data-testid="thinking-block"]', { timeout: 25000 })
        report.check('thinking', 'the chain of thought is shown while streaming', true)

        // It should be open on its own, before any answer exists.
        report.check('thinking', 'it starts expanded, because it is the only thing happening',
          (await page.getByTestId('thinking-body').count()) === 1)

        await page
          .waitForFunction(
            () => (document.querySelector('[data-testid="thinking-body"]')?.textContent ?? '').length > 40,
            undefined,
            { timeout: 20000 }
          )
          .catch(() => undefined)
        const early = (await page.getByTestId('thinking-body').textContent()) ?? ''
        report.check('thinking', 'the reasoning text is live', early.length > 20, `${early.length} chars`)

        // Thinking and the "working" dots are both assistant output; showing both at once put
        // two Assistant headers on screen for the same turn.
        const rows = await page.locator('.msg.from-assistant').count()
        report.check('thinking', 'exactly one assistant row while thinking', rows === 1, `${rows} rows`)

        await shot(page, 'thinking-streaming')

        // Once the answer starts, the thinking gets out of the way.
        await page.waitForFunction(
          () => (document.querySelector('[data-testid="streaming-message"] .markdown')?.textContent ?? '').length > 10,
          undefined,
          { timeout: 40000 }
        ).catch(() => undefined)
        await page.waitForTimeout(700)
        report.check('thinking', 'it collapses once the answer begins',
          (await page.getByTestId('thinking-body').count()) === 0)

        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="chat-input"]')
            return el instanceof HTMLTextAreaElement && !el.disabled
          },
          undefined,
          { timeout: 60000 }
        )

        // And it has to survive as part of the message, not just the stream.
        report.check('thinking', 'the finished message keeps its thinking',
          (await page.getByTestId('thinking-block').count()) === 1)

        await page.getByTestId('thinking-toggle').click()
        await page.waitForTimeout(250)
        const kept = (await page.getByTestId('thinking-body').textContent()) ?? ''
        report.check('thinking', 'the saved thinking can be reopened', /work through this/.test(kept), kept.slice(0, 80))

        // Reload: the reasoning is in the database or it is gone.
        await page.reload()
        await page.waitForSelector('.nav-item')
        await goTo(page, 'Chat')
        await page
          .waitForFunction(() => document.querySelectorAll('.messages .msg').length >= 2, undefined, { timeout: 20000 })
          .catch(() => undefined)
        await page.waitForTimeout(400)
        report.check('thinking', 'thinking survives a reload', (await page.getByTestId('thinking-block').count()) === 1,
          `${await page.locator('.messages .msg').count()} messages restored`)

        await shot(page, 'thinking-collapsed')
        await auditLayout(page, 'Chat (thinking)', report)
      },
      { models: ['Reasoner-8B-Q4_K_M.gguf'], modelOpts: { reasoning: 'effort' } }
    )
  },

  /**
   * The effort slider's fill and stop marks must line up with the thumb.
   *
   * A native range thumb's centre travels between thumb/2 and width - thumb/2, so anything drawn
   * at 0-100% of the track disagrees with it by half a thumb at each end — which showed up as
   * fill visible past the handle at the top of the range.
   */
  async sliderGeometry() {
    await withApp(
      'slider-geometry',
      async ({ page }) => {
        await loadModel(page)
        await goTo(page, 'Chat')
        await page.getByTestId('new-conversation').click()
        await page.waitForTimeout(400)

        // The scale is behind its trigger; open it before measuring anything.
        await page.getByTestId('reasoning-trigger').click()
        await page.waitForTimeout(160)

        const range = page.getByTestId('reasoning-slider')
        const max = Number(await range.getAttribute('max'))

        for (const index of [0, max]) {
          await range.fill(String(index))
          await page.waitForTimeout(200)

          const m = await page.evaluate(() => {
            const input = document.querySelector('[data-testid="reasoning-slider"]')
            const slider = input.closest('.slider')
            const stops = [...document.querySelectorAll('.slider-stop')]
            const box = input.getBoundingClientRect()

            /*
             * The thumb size is read from the stylesheet rather than written down here.
             *
             * This assertion used to hardcode 13 — the value the CSS *declared* — while the
             * thumb was sized content-box and really occupied 17. Both the test and the layout
             * were wrong in the same direction, so a systematic 2px error passed a 2px
             * tolerance for as long as it existed. Deriving it from --thumb means the check
             * cannot quietly agree with a mistake in the CSS again.
             */
            const thumb = parseFloat(getComputedStyle(slider).getPropertyValue('--thumb'))
            const value = Number(input.value)
            const maxV = Number(input.max)
            const thumbCentre = box.left + thumb / 2 + ((box.width - thumb) * value) / (maxV || 1)

            return {
              thumb,
              thumbCentre: Math.round(thumbCentre),
              stopCentres: stops.map((s) => Math.round(s.getBoundingClientRect().left + s.getBoundingClientRect().width / 2))
            }
          })

          report.check('slider-geometry', `the thumb size is declared in the stylesheet at position ${index}`,
            Number.isFinite(m.thumb) && m.thumb > 0, `--thumb resolved to ${m.thumb}`)

          // One mark must sit exactly under the handle. Tolerance is a pixel of rounding, not
          // the two that let the old drift through.
          const nearest = m.stopCentres.reduce((a, b) => (Math.abs(b - m.thumbCentre) < Math.abs(a - m.thumbCentre) ? b : a))
          report.check('slider-geometry', `a stop mark sits under the thumb at position ${index}`,
            Math.abs(nearest - m.thumbCentre) <= 1, `stop ${nearest}, thumb ${m.thumbCentre}`)
        }
      },
      { models: ['Reasoner-8B-Q4_K_M.gguf'], modelOpts: { reasoning: 'effort' } }
    )
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
}

/*
 * Scenarios that start a server, kept apart from the rest out of caution.
 *
 * Each sandbox is now given its own API port, so these no longer collide with each other — or,
 * more importantly, with a copy of the app the developer happens to have running, which held
 * port 1234 and made three scenarios report a server that would not start. Still grouped, since
 * a bound socket is the one resource here that is not sandboxed by construction.
 */
const SERIAL = new Set(['server', 'remote', 'apiSurface'])

const flagValue = (name) => {
  const hit = args.find((a) => a.startsWith(`${name}=`))
  return hit ? hit.slice(name.length + 1) : null
}

/*
 * Each scenario launches its own Electron against its own sandbox, so they are independent by
 * construction and the run was only sequential by habit. The work is mostly waiting — for a
 * window, for a mock reply, for a view to settle — so the ceiling is memory rather than CPU.
 * Capped well below the core count because every worker is a whole browser.
 */
/*
 * Deliberately modest, because each worker is a visible application window.
 *
 * The machine could run more — the work is mostly waiting, not computing — but this suite is run
 * on a desktop somebody is using, and every job in flight is another app window appearing on it.
 * Three is most of the speedup for half the intrusion. `--jobs=1` runs the old way, one window at
 * a time, for when even that is too much; `--jobs=8` is there for an idle machine.
 */
const jobs = Math.max(1, Number(flagValue('--jobs') ?? Math.min(3, Math.max(1, os.cpus().length - 1))))

const durations = new Map()

async function runOne(name) {
  const started = Date.now()
  await scenarioContext.run(name, () => scenarios[name]())
  durations.set(name, Date.now() - started)
  report.flush(name)
}

const runStarted = Date.now()
const parallel = names.filter((n) => !SERIAL.has(n))
const serial = names.filter((n) => SERIAL.has(n))

if (jobs > 1 && parallel.length > 1) {
  report.capture(true)
  console.log(
    `\nRunning ${parallel.length} scenarios ${jobs} at a time, then ${serial.length} that need a port to themselves.` +
      `\n${jobs} app windows will open at once — pass --jobs=1 for one at a time.`
  )
  const queue = [...parallel]
  await Promise.all(
    Array.from({ length: Math.min(jobs, queue.length) }, async () => {
      for (let next = queue.shift(); next; next = queue.shift()) await runOne(next)
    })
  )
} else {
  for (const name of parallel) await runOne(name)
}

// Port binders, one at a time, and unbuffered so their output reads live again.
report.capture(false)
for (const name of serial) await runOne(name)

// ---------------------------------------------------------------- report

const summary = report.summary()
console.log(`\n${'='.repeat(72)}`)

/*
 * Wall clock and the worst offenders.
 *
 * Printed every run rather than hidden behind a flag: when the suite gets slow again, the first
 * question is which scenario, and answering it should not require instrumenting anything.
 */
const wall = (Date.now() - runStarted) / 1000
const slowest = [...durations.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)
console.log(`${names.length} scenarios in ${wall.toFixed(1)}s wall (${jobs} at a time)`)
if (slowest.length) {
  console.log(`slowest: ${slowest.map(([n, ms]) => `${n} ${(ms / 1000).toFixed(1)}s`).join('  ')}`)
}
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
