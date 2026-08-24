import { _electron as electron } from 'playwright-core'
import path from 'node:path'
import { createEnv, addModel, envVars, cleanupEnv, ROOT } from './fixtures.mjs'

const env = await createEnv('probe')
await addModel(env, 'Test-27B-Q4_K_M.gguf', { withMmproj: true })
console.log('env:', env.base)

const app = await electron.launch({
  args: [path.join(ROOT, 'out', 'main', 'index.js')],
  env: envVars(env, { LLMM_USER_DATA: env.userData }),
  cwd: ROOT,
  timeout: 60000
})

app.process().stdout?.on('data', (d) => process.stdout.write(`[out] ${d}`))
app.process().stderr?.on('data', (d) => process.stdout.write(`[err] ${d}`))

try {
  const page = await app.firstWindow({ timeout: 25000 })
  await page.waitForLoadState('domcontentloaded')
  await page.waitForTimeout(2500)
  console.log('title    :', await page.title())
  console.log('nav      :', (await page.locator('.nav-item').allTextContents()).join(', '))
  console.log('models   :', await page.locator('.model-card').count())
  await page.screenshot({ path: path.join(ROOT, 'scripts', 'e2e', 'probe.png') })
  console.log('screenshot ok')
} catch (err) {
  console.log('FAILED:', err.message.split('\n')[0])
  const wins = app.windows()
  console.log('windows known to playwright:', wins.length)
}

await app.close().catch(() => {})
await cleanupEnv(env)
