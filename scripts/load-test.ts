/**
 * End-to-end load test, run inside Electron so it exercises the real code paths:
 * library scan -> capability detection -> auto-fit -> llama-server spawn -> generation
 * -> tool call -> unload. Nothing here is mocked.
 */
import { app } from 'electron'
import path from 'node:path'
import { scanLibrary } from '../src/main/models/library'
import { detectHardware, refreshFreeVram } from '../src/main/hardware/gpu'
import { planFit, DEFAULT_CONSTRAINTS, fmtBytes, verifyPrediction } from '../src/main/autofit/engine'
import { llama } from '../src/main/runtime/llama'
import { missingBinaries } from '../src/main/runtime/binaries'

// Explicit so the harness works regardless of where Electron resolves the app path.
const PROJECT = process.env.LLMM_PROJECT ?? app.getAppPath()
const MODELS_DIR = path.join(PROJECT, 'LLMManagerModels')

function hr(t: string): void {
  console.log(`\n${'─'.repeat(64)}\n${t}`)
}

async function main(): Promise<void> {
  hr('1. Hardware')
  let hw = await detectHardware()
  console.log(`   backend=${hw.backend}  missing binaries: ${missingBinaries(hw.backend).join(', ') || 'none'}`)
  for (const g of hw.gpus) {
    console.log(`   ${g.name}: ${fmtBytes(g.freeVram)} free of ${fmtBytes(g.totalVram)}`)
  }

  hr('2. Library scan (real GGUF parse)')
  const models = await scanLibrary(MODELS_DIR)
  for (const m of models) {
    console.log(`   ${m.filename}`)
    console.log(`     ${fmtBytes(m.bytes)}  arch=${m.arch?.architecture}  quant=${m.arch?.quant}`)
    console.log(`     blocks=${m.arch?.blockCount} (attn=${m.arch?.attentionLayers} ssm=${m.arch?.ssmLayers})`)
    console.log(`     caps: vision=${m.caps.vision} audio=${m.caps.audio} nativeVideo=${m.caps.nativeVideo} tools=${m.caps.tools}`)
    console.log(`     mmproj: ${m.caps.mmprojPath ? path.basename(m.caps.mmprojPath) : 'none'}`)
    console.log(`     tags: ${m.tags.join(', ')}`)
  }
  const model = models.find((m) => /Qwen3\.8-27B/i.test(m.filename))
  if (!model?.arch) throw new Error('Qwen3.8-27B not found or unparsable')

  hr('3. Auto-fit')
  hw = await refreshFreeVram(hw)
  const mmprojBytes = model.caps.mmprojPath ? (await import('node:fs')).statSync(model.caps.mmprojPath).size : 0
  const fit = planFit(model.arch, hw, { ...DEFAULT_CONSTRAINTS, companionBytes: mmprojBytes })
  fit.notes.forEach((n) => console.log(`   note: ${n}`))
  const plan = fit.chosen ?? fit.alternatives[0]
  if (!plan) throw new Error('no workable plan')
  console.log(`   -> ${plan.label}: ${plan.contextLength.toLocaleString()} ctx @ ${plan.kvType}`)
  console.log(`      ${plan.gpuLayers}/${plan.totalLayers} layers on GPU, split ${plan.tensorSplit.map((s) => (s * 100).toFixed(1) + '%').join('/')}`)
  console.log(`      predicted VRAM: ${plan.predictedVramPerGpu.map(fmtBytes).join(' + ')}`)

  hr('4. Loading llama-server')
  const before = await refreshFreeVram(hw)
  llama.on('status', (s: { phase: string }) => console.log(`   [${s.phase}]`))
  const t0 = Date.now()
  const loaded = await llama.load(model, plan, hw.backend)
  console.log(`   ready in ${((Date.now() - t0) / 1000).toFixed(1)}s on 127.0.0.1:${loaded.port}`)

  hr('5. Prediction vs reality')
  const after = await refreshFreeVram(before)
  const actual = after.gpus.map((g, i) => Math.max(0, (before.gpus[i]?.freeVram ?? 0) - g.freeVram))
  const nvidiaActual = after.gpus.filter((g) => g.vendor === 'nvidia').map((_, i) => actual[i])
  console.log(`   predicted: ${plan.predictedVramPerGpu.map(fmtBytes).join(' + ')}`)
  console.log(`   actual:    ${nvidiaActual.map(fmtBytes).join(' + ')}`)
  const v = verifyPrediction(plan, nvidiaActual)
  console.log(`   worst ratio: ${v.worstRatio.toFixed(2)}x  ${v.suggestion ?? '(within margin)'}`)

  hr('6. Generation')
  let text = ''
  const g0 = Date.now()
  for await (const ev of llama.streamEvents({
    messages: [{ role: 'user', content: 'In one sentence, what is a state-space model?' }],
    maxTokens: 120
  })) {
    if (ev.type === 'text') {
      text += ev.text
      process.stdout.write(ev.text)
    }
  }
  const t = llama.timings
  console.log(`\n   ${t?.completionTokens ?? 0} tokens, TTFT ${t?.ttftMs}ms, ${t?.tokensPerSecond.toFixed(1)} tok/s, total ${((Date.now() - g0) / 1000).toFixed(1)}s`)

  hr('7. Tool calling (native tools parameter + real tool schema)')
  const toolEvents: string[] = []
  for await (const ev of llama.streamEvents({
    messages: [{ role: 'user', content: 'What files are in C:\Windows? Use your tools.' }],
    tools: [
      {
        name: 'list_dir',
        description: 'List the entries of a directory with type and size.',
        tier: 'read',
        parameters: {
          type: 'object',
          properties: { path: { type: 'string', description: 'Directory to list' } },
          required: ['path']
        }
      }
    ],
    maxTokens: 200
  })) {
    if (ev.type === 'tool_call') {
      toolEvents.push(`${ev.call.name}(${JSON.stringify(ev.call.args)})`)
    }
  }
  console.log(toolEvents.length ? `   tool call: ${toolEvents.join(', ')}` : '   (model answered without calling a tool)')

  hr('8. Unloading')
  await llama.unload()
  console.log(`   loaded after unload: ${llama.loaded ? 'STILL LOADED' : 'null'}`)
  const freed = await refreshFreeVram(after)
  console.log(`   VRAM now: ${freed.gpus.filter((g) => g.vendor === 'nvidia').map((g) => fmtBytes(g.freeVram)).join(' + ')} free`)

  console.log('\nDone.\n')
}

main()
  .then(() => app.exit(0))
  .catch(async (err) => {
    console.error('\nFAILED:', err instanceof Error ? err.message : err)
    await llama.unload().catch(() => undefined)
    app.exit(1)
  })
