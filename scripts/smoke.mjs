/**
 * Smoke tests for the pure logic: auto-fit maths, GBNF compilation, the permission gate,
 * the GGUF parser, and quant recommendation.
 *
 * Run with `npm test`. These do not need Electron, a GPU, or a model file — anything that
 * needs a real runtime is verified by hand and recorded in BUILD_STATUS.md instead.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// Rebuild the test bundles so the suite always runs against current source.
execFileSync(process.execPath, [path.join(ROOT, 'scripts/build-tests.mjs')], { stdio: 'inherit' })

const { planFit, kvCacheBytes, computeBufferBytes, proportionalSplit, DEFAULT_CONSTRAINTS, fmtBytes, verifyPrediction } =
  await import('./built/engine.js')
const { toolCallGrammar, schemaGrammar } = await import('./built/gbnf.js')
const { checkHardBlock, describeCall, PermissionEngine } = await import('./built/permissions.js')
const { readGguf, extractArchInfo, tensorByteSize, isKnownGgmlType } = await import('./built/gguf.js')
const { recommendQuant, findMmprojFor } = await import('./built/hf.js')
 const { isVirtualAdapter } = await import('./built/gpu.js')
const { exportFilename, uniquePath } = await import('./built/filenames.js')
const { detectReasoning, reasoningRequestFields } = await import('./built/reasoning.js')

const GB = 1024 ** 3
let pass = 0
let fail = 0
const failures = []

function check(name, cond, extra = '') {
  if (cond) {
    pass++
    console.log(`  PASS  ${name}`)
  } else {
    fail++
    failures.push(name)
    console.log(`  FAIL  ${name} ${extra}`)
  }
}

function section(title) {
  console.log(`\n${title}`)
}

// ---------------------------------------------------------------- fixtures

// Qwen3.8-27B-shaped: dense 27B, 62 layers, GQA with 8 KV heads, 262K trained context.
const arch = {
  architecture: 'qwen3',
  name: 'Qwen3.8-27B',
  blockCount: 62,
  embeddingLength: 5120,
  headCount: 40,
  headCountKv: 8,
  headDim: 128,
  contextLength: 262144,
  vocabSize: 152064,
  quant: 'Q4_K',
  weightBytes: 16 * GB,
  perLayerBytes: 15 * GB,
  nonLayerBytes: 1 * GB,
  expertCount: 0
}

const hw = (gpus, backend = 'cuda') => ({
  gpus,
  totalRam: 64 * GB,
  freeRam: 40 * GB,
  cpuName: 'test cpu',
  cpuThreads: 16,
  backend,
  takenAt: Date.now()
})

const gpu = (name, totalGb, freeGb, measured = true, index = 0) => ({
  index,
  name,
  vendor: 'nvidia',
  totalVram: totalGb * GB,
  freeVram: freeGb < 0 ? -1 : freeGb * GB,
  utilisation: 5,
  freeIsMeasured: measured
})

// ---------------------------------------------------------------- auto-fit

section('KV cache maths')
{
  const kv128 = kvCacheBytes(arch, 131072, 'q8_0')
  const expected = 2 * 8 * 128 * 62 * 131072 * (34 / 32)
  check('128K q8_0 KV matches the closed form', Math.abs(kv128 - expected) < 1)
  check('q4_0 is ~53% of q8_0', Math.abs(kvCacheBytes(arch, 131072, 'q4_0') / kv128 - 0.529) < 0.01)
  check('KV scales linearly with context', Math.abs(kvCacheBytes(arch, 2048, 'f16') * 2 - kvCacheBytes(arch, 4096, 'f16')) < 1)
  console.log(`  128K at q8_0 = ${fmtBytes(kv128)}`)
}

section('Hybrid attention/SSM models cache only on attention layers')
{
  // Real geometry from ggml-org/Qwen3.8-27B-GGUF: 64 blocks, full_attention_interval 4,
  // so 16 attention layers and 48 state-space layers. Counting all 64 overestimates the KV
  // cache ~3.9x and costs most of the achievable context.
  const hybrid = {
    ...arch,
    architecture: 'qwen35',
    blockCount: 64,
    headCount: 24,
    headCountKv: 4,
    headDim: 256,
    attentionLayers: 16,
    ssmLayers: 48,
    ssmStateBytesPerLayer: 3211264,
    weightBytes: 17.66 * GB,
    perLayerBytes: 16 * GB,
    nonLayerBytes: 1.66 * GB
  }

  const naive = 2 * hybrid.headCountKv * hybrid.headDim * hybrid.blockCount * 131072 * (34 / 32)
  const actual = kvCacheBytes(hybrid, 131072, 'q8_0')
  check('hybrid KV is far below the all-layers estimate', actual < naive / 3, `${fmtBytes(actual)} vs ${fmtBytes(naive)}`)

  // The SSM state is real memory, but constant — doubling context must not double it.
  // Asking for zero context isolates the state, so this holds whatever safety factor is applied.
  const stateOnly = kvCacheBytes(hybrid, 0, 'q8_0')
  const at64k = kvCacheBytes(hybrid, 65536, 'q8_0')
  const at128k = kvCacheBytes(hybrid, 131072, 'q8_0')
  check('SSM state is included', stateOnly > 0 && at64k > stateOnly)
  check('SSM state does not scale with context', Math.abs((at128k - at64k) - (at64k - stateOnly)) < 1)

  // A dense model of the same block count must be unaffected by the hybrid path.
  const dense = { ...hybrid, attentionLayers: 64, ssmLayers: 0, ssmStateBytesPerLayer: 0 }
  check('dense models still count every layer', Math.abs(kvCacheBytes(dense, 131072, 'q8_0') - naive) < 1)

  // Metadata written before hybrid support leaves these unset; fall back to block count.
  const legacy = { ...dense, attentionLayers: undefined, ssmLayers: undefined, ssmStateBytesPerLayer: undefined }
  check('missing hybrid fields fall back to block count', Math.abs(kvCacheBytes(legacy, 131072, 'q8_0') - naive) < 1)

  // On this rig the fix is the difference between the ideal context and a degraded one.
  const rig = hw([gpu('RTX 5080', 16, 13.4), gpu('RTX 4070 Ti', 12, 11.7, true, 1)])
  const result = planFit(hybrid, rig, DEFAULT_CONSTRAINTS)
  check('hybrid 27B reaches the ideal context on a 5080+4070Ti', result.chosen?.contextLength === 131072, `${result.chosen?.contextLength}`)
  check('at the preferred KV quality, not the floor', result.chosen?.kvType === 'q8_0')
  check('explains the hybrid layout', result.notes.some((n) => /hybrid architecture/i.test(n)))
  console.log(`  128K KV: ${fmtBytes(naive)} naive -> ${fmtBytes(actual)} actual`)
}

section('Compute buffer: flash attention is a precondition at long context')
{
  const withFa = computeBufferBytes(arch, 131072, 512, true)
  const withoutFa = computeBufferBytes(arch, 131072, 512, false)
  check('no-FA attention buffer is far larger at 128K', withoutFa > withFa * 10)
  check('FA buffer does not scale with context', computeBufferBytes(arch, 4096, 512, true) === withFa)
  console.log(`  128K compute buffer: FA on ${fmtBytes(withFa)}, FA off ${fmtBytes(withoutFa)}`)
}

section('P1 — sizes against FREE VRAM, not total')
{
  // A 24 GB card with only 6 GB free. An engine sizing off `total` would plan a 16 GB model on it.
  const tight = planFit(arch, hw([gpu('RTX 4090', 24, 6)]), DEFAULT_CONSTRAINTS)
  check('does not claim a full-GPU fit with 6 GB free', !(tight.chosen && !tight.chosen.spillsToHost))

  const roomy = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
  check('the same card with 23 GB free does fit', !!roomy.chosen)
  check('fitted plan keeps every layer on GPU', roomy.chosen?.gpuLayers === arch.blockCount)
  if (roomy.chosen) {
    console.log(`  chose ${roomy.chosen.contextLength.toLocaleString()} ctx at ${roomy.chosen.kvType}`)
  }
}

section('P2 — proportional multi-GPU split, not an even one')
{
  const mixed = planFit(arch, hw([gpu('RTX 4090', 24, 22), gpu('RTX 3070', 8, 7, true, 1)]), DEFAULT_CONSTRAINTS)
  const split = (mixed.chosen ?? mixed.alternatives[0])?.tensorSplit ?? []
  check('split is not 50/50', split.length === 2 && Math.abs(split[0] - 0.5) > 0.1, JSON.stringify(split))
  check('the larger card gets the larger share', split[0] > split[1])
  check('split sums to 1', Math.abs(split.reduce((a, b) => a + b, 0) - 1) < 1e-9)
  console.log(`  split = ${split.map((s) => `${(s * 100).toFixed(1)}%`).join(' / ')}`)

  // Three mismatched cards, to be sure it is not just a two-element special case.
  const three = proportionalSplit([20 * GB, 10 * GB, 5 * GB])
  check('three-way split is proportional', Math.abs(three[0] - 4 / 7) < 0.01 && Math.abs(three[2] - 1 / 7) < 0.01)
}

section('P4 — full-context KV reserved up front')
{
  const roomy = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
  if (roomy.chosen) {
    const reserved = kvCacheBytes(arch, roomy.chosen.contextLength, roomy.chosen.kvType)
    check('plan.kvBytes equals the whole context, not the current fill', Math.abs(roomy.chosen.kvBytes - reserved) < 1)
  } else {
    check('plan.kvBytes equals the whole context', false, '(no plan produced)')
  }
}

/*
 * The KV preference and the KV floor are independent dropdowns whose option lists are ordered
 * opposite ways, so "prefer q4_0, never go below q8_0" is easy to select — and contradictory.
 * The candidate list used to come out empty for it, which meant both passes that can return a
 * chosen plan iterated zero times: auto-fit stopped choosing anything, on any hardware, and said
 * nothing about why.
 */
section('Auto-fit: a contradictory KV preference still plans')
{
  const roomyRig = hw([gpu('RTX 4090', 24, 23)])

  const contradictory = planFit(arch, roomyRig, {
    ...DEFAULT_CONSTRAINTS,
    preferredKvType: 'q4_0',
    minKvType: 'q8_0'
  })
  // The floor is the stronger statement, so the contradictory setting must behave exactly as if
  // the preference had been the floor all along — not as if no KV type were acceptable at all.
  const floorOnly = planFit(arch, roomyRig, {
    ...DEFAULT_CONSTRAINTS,
    preferredKvType: 'q8_0',
    minKvType: 'q8_0'
  })
  check('a preference below the floor is treated as the floor',
    !!contradictory.chosen === !!floorOnly.chosen && contradictory.chosen?.kvType === floorOnly.chosen?.kvType,
    `contradictory=${contradictory.chosen?.kvType ?? 'none'} floorOnly=${floorOnly.chosen?.kvType ?? 'none'}`)
  check('and the same alternatives are offered',
    contradictory.alternatives.length === floorOnly.alternatives.length,
    `${contradictory.alternatives.length} vs ${floorOnly.alternatives.length}`)
  check('and it says why the preference was not honoured',
    contradictory.notes.some((n) => /below the floor/i.test(n)), contradictory.notes.join(' | ').slice(0, 120))

  // Hardware where the floor genuinely fits must still get a plan chosen for it — the bug made
  // that impossible on any hardware, because the candidate list was empty before it was consulted.
  const hugeRig = hw([gpu('RTX 6000 Ada', 48, 47)])
  const huge = planFit(arch, hugeRig, { ...DEFAULT_CONSTRAINTS, preferredKvType: 'q4_0', minKvType: 'q8_0' })
  check('roomy hardware still gets an automatic plan', !!huge.chosen, huge.chosen ? huge.chosen.label : 'none')
  check('at the floor quality', huge.chosen?.kvType === 'q8_0', String(huge.chosen?.kvType))

  // An unrecognised value — a hand-edited settings file — must not silently pick something else.
  const garbage = planFit(arch, roomyRig, {
    ...DEFAULT_CONSTRAINTS,
    preferredKvType: 'q2_0',
    minKvType: 'q4_0'
  })
  const normal = planFit(arch, roomyRig, DEFAULT_CONSTRAINTS)
  check('an unknown preference falls back to the full acceptable range',
    garbage.chosen?.kvType === normal.chosen?.kvType,
    `garbage=${garbage.chosen?.kvType ?? 'none'} normal=${normal.chosen?.kvType ?? 'none'}`)
  check('and the default configuration says nothing about floors',
    !normal.notes.some((n) => /below the floor/i.test(n)))
}

section('Never degrades silently')
{
  const cramped = planFit(arch, hw([gpu('RTX 3060', 12, 11)]), DEFAULT_CONSTRAINTS)
  check('flags that the user must choose', cramped.needsUserChoice === true)
  check('offers at least one real alternative', cramped.alternatives.length >= 1)
  check('every alternative carries a rationale', cramped.alternatives.every((a) => a.rationale.length > 0))
  check('no alternative quantises below the floor', cramped.alternatives.every((a) => ['f16', 'q8_0', 'q4_0'].includes(a.kvType)))
  cramped.alternatives.forEach((a) =>
    console.log(`  option: ${a.label} — ${a.contextLength.toLocaleString()} ctx, ${a.gpuLayers}/${a.totalLayers} layers, speed ${a.speedScore}`)
  )
}

section('Mixed rigs — only addressable devices take part')
{
  // Regression: this was found by running against a real machine that had an RTX 5080, an
  // RTX 4070 Ti, an AMD iGPU, and two virtual display adapters from screen-sharing tools.
  // The phantom devices stole budget share and cut a 27B model from 89K to 20K context.
  const mixed = hw([
    gpu('NVIDIA GeForce RTX 5080', 16, 13.2),
    gpu('NVIDIA GeForce RTX 4070 Ti', 12, 11.7, true, 1),
    { ...gpu('AMD Radeon(TM) Graphics', 2, -1, false, 2), vendor: 'amd' }
  ])

  const result = planFit(arch, mixed, DEFAULT_CONSTRAINTS)
  const plan = result.chosen ?? result.alternatives[0]

  check('CUDA backend excludes the AMD iGPU', plan?.tensorSplit.length === 2, JSON.stringify(plan?.tensorSplit))
  check('says why it excluded it', result.notes.some((n) => /excluded from the split.*AMD/i.test(n)))
  check('split still favours the bigger card', (plan?.tensorSplit[0] ?? 0) > (plan?.tensorSplit[1] ?? 1))
  check('the 27B fits fully on GPU on this rig', plan?.gpuLayers === arch.blockCount)
  check('reaches a useful context', (plan?.contextLength ?? 0) >= 65536, `${plan?.contextLength}`)
  console.log(`  ${plan?.contextLength.toLocaleString()} ctx across ${plan?.tensorSplit.length} GPUs`)

  // Under Vulkan an iGPU is usable, but not while discrete cards are present.
  const vulkanMixed = planFit(arch, { ...mixed, backend: 'vulkan' }, DEFAULT_CONSTRAINTS)
  const vplan = vulkanMixed.chosen ?? vulkanMixed.alternatives[0]
  check('Vulkan also skips the iGPU when discrete cards exist', vplan?.tensorSplit.length === 2)

  const igpuOnly = planFit(
    { ...arch, weightBytes: 1 * GB, perLayerBytes: 0.9 * GB, nonLayerBytes: 0.1 * GB, blockCount: 8 },
    hw([{ ...gpu('AMD Radeon(TM) Graphics', 8, -1, false), vendor: 'amd' }], 'vulkan'),
    DEFAULT_CONSTRAINTS
  )
  const iplan = igpuOnly.chosen ?? igpuOnly.alternatives[0]
  check('an iGPU is used when it is the only device', (iplan?.tensorSplit.length ?? 0) === 1)
}

section('Virtual display adapters are not GPUs')
{
  // Screen-sharing tools install display adapters that WMI reports like GPUs.
  const virtual = [
    'Virtual Desktop Monitor',
    'Parsec Virtual Display Adapter',
    'IDD HDR',
    'Sunshine Gamestream Mirror',
    'Microsoft Basic Display Adapter'
  ]
  for (const name of virtual) {
    check(`rejects phantom adapter: ${name}`, isVirtualAdapter(name))
  }

  const real = ['NVIDIA GeForce RTX 5080', 'AMD Radeon RX 7900 XTX', 'Intel Arc A770', 'AMD Radeon(TM) Graphics']
  for (const name of real) {
    check(`keeps real adapter: ${name}`, !isVirtualAdapter(name))
  }
}

section('Honest about unmeasurable hardware')
{
  const amd = planFit(arch, hw([{ ...gpu('Radeon RX 7900 XTX', 24, -1, false), vendor: 'amd' }], 'vulkan'), DEFAULT_CONSTRAINTS)
  check('notes that free VRAM was estimated', amd.notes.some((n) => /could not be measured/i.test(n)))

  const cpuOnly = planFit(arch, hw([], 'cpu'), DEFAULT_CONSTRAINTS)
  check('falls back to a CPU plan with no GPU', cpuOnly.chosen?.gpuLayers === 0)
}

section('User overrides are respected, not overwritten')
{
  const forced = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), {
    ...DEFAULT_CONSTRAINTS,
    overrides: { contextLength: 8192, kvType: 'f16', batchSize: 256, flashAttention: false }
  })
  const plan = forced.chosen ?? forced.alternatives[0]
  check('honours the KV type override', plan?.kvType === 'f16')
  check('honours the batch size override', plan?.batchSize === 256)
  check('honours the flash-attention override', plan?.flashAttention === false)
  check('does not exceed the context override', (plan?.contextLength ?? 0) <= 8192)
  check('warns that flash attention is off', forced.notes.some((n) => /flash attention is off/i.test(n)))
}

section('Trained-context ceiling')
{
  const shortCtx = planFit({ ...arch, contextLength: 8192 }, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
  const plan = shortCtx.chosen ?? shortCtx.alternatives[0]
  check('never plans beyond the trained context', (plan?.contextLength ?? 0) <= 8192)
  check('says why it capped', shortCtx.notes.some((n) => /trained for/i.test(n)))
}

section('Prediction verification feeds back')
{
  const plan = { predictedVramPerGpu: [10 * GB] }
  check('flags a large under-prediction', !!verifyPrediction(plan, [13 * GB]).suggestion)
  check('flags a large over-prediction', !!verifyPrediction(plan, [7 * GB]).suggestion)
  check('stays quiet when close', verifyPrediction(plan, [10.2 * GB]).suggestion === null)
}

// ---------------------------------------------------------------- GBNF

section('GBNF compiler')
{
  const grammar = toolCallGrammar([
    {
      name: 'read_file',
      parameters: { type: 'object', properties: { path: { type: 'string' }, limit: { type: 'integer' } }, required: ['path'] }
    },
    { name: 'run_command', parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } }
  ])
  check('has a root rule', /^root ::= /m.test(grammar))
  check('pins each tool name as a literal', grammar.includes('\\"read_file\\"') && grammar.includes('\\"run_command\\"'))
  check('emits one branch per tool', (grammar.match(/call-\d+ ::=/g) ?? []).length === 2)
  check('defines JSON primitives', grammar.includes('string      ::=') && grammar.includes('number      ::='))
  // Strip every optional group; a required key must survive, an optional one must not.
  const withoutOptionals = grammar.replace(/\([^()]*\)\?/g, '')
  check('required fields stay mandatory', withoutOptionals.includes('\\"path\\"'))
  check('optional fields are wrapped in an optional group', !withoutOptionals.includes('\\"limit\\"'))

  const enumGrammar = schemaGrammar({ type: 'object', properties: { mode: { type: 'string', enum: ['fast', 'slow'] } } })
  check('compiles enums to alternations', enumGrammar.includes('fast') && enumGrammar.includes('slow'))

  const arrayGrammar = schemaGrammar({ type: 'object', properties: { items: { type: 'array', items: { type: 'string' } } } })
  check('compiles arrays', /arr-\d+ ::= "\["/.test(arrayGrammar))

  /*
   * An object whose properties are all optional has to allow any one of them alone.
   *
   * The old compiler emitted `(m1)? ("," ws m2)?`, which cannot express "just m2": the comma
   * belongs to m2's group, so emitting it alone gives `{, "m2": …}` — not JSON — while the
   * valid `{"m2": …}` is refused. `browser_click` takes selector *or* text and requires
   * neither, so the grammar made its text-only form unreachable.
   */
  const allOptional = schemaGrammar({
    type: 'object',
    properties: { selector: { type: 'string' }, text: { type: 'string' } },
    required: []
  })
  const tail = allOptional.split('\n').find((l) => /^opt-\d+ ::=/.test(l)) ?? ''
  const alternatives = tail.replace(/^opt-\d+ ::= /, '').split(' | ')

  check('an all-optional object compiles to alternatives', alternatives.length === 2, tail)
  check('no alternative starts with a comma',
    alternatives.every((a) => !a.trimStart().startsWith('","')), alternatives.join('  ||  '))
  check('each optional key can be the first thing emitted',
    alternatives.some((a) => a.trimStart().startsWith('"\\"selector\\""')) &&
      alternatives.some((a) => a.trimStart().startsWith('"\\"text\\""')),
    alternatives.join('  ||  '))
  check('and the whole group is still skippable, so {} is legal',
    /obj-\d+ ::= "\{" ws opt-\d+\? "\}"/.test(allOptional), allOptional.split('\n').find((l) => l.startsWith('obj-')) ?? '')

  // A required key ahead of them makes a leading comma correct, so that form is left alone.
  const mixed = schemaGrammar({
    type: 'object',
    properties: { database: { type: 'string' }, limit: { type: 'integer' } },
    required: ['database']
  })
  check('a required key keeps the simpler skippable form',
    !/^opt-\d+ ::=/m.test(mixed) && /\("," ws "\\"limit\\"".*\)\?/.test(mixed),
    mixed.split('\n').find((l) => l.startsWith('obj-')) ?? '')
}

// ---------------------------------------------------------------- permissions

section('Hard-block list')
{
  const blocked = [
    'format C:',
    'diskpart /s script.txt',
    'rm -rf /',
    'Set-MpPreference -DisableRealtimeMonitoring $true',
    'bcdedit /set nointegritychecks on',
    'vssadmin delete shadows /all',
    'shutdown /r /t 0'
  ]
  for (const cmd of blocked) {
    check(`blocks: ${cmd.slice(0, 40)}`, checkHardBlock(cmd).blocked)
  }

  const allowed = [
    'npm install',
    'git status',
    'rm -rf ./node_modules',
    'Remove-Item -Recurse .\\dist',
    'python train.py --format csv'
  ]
  for (const cmd of allowed) {
    check(`allows: ${cmd.slice(0, 40)}`, !checkHardBlock(cmd).blocked)
  }
}

section('Permission tiers')
{
  const asked = []
  const engine = new PermissionEngine({
    hardBlocksDisabled: () => false,
    ask: async (req) => {
      asked.push(req.tool)
      return 'allow-tool'
    }
  })
  const cwd = process.cwd()

  const read = await engine.authorise('read_file', 'read', { path: 'x.txt' }, cwd)
  check('read-tier runs without prompting', read.allowed && asked.length === 0)

  const write = await engine.authorise('write_file', 'write', { path: 'x.txt', content: 'hi' }, cwd)
  check('write-tier prompts', write.allowed && asked.length === 1)

  const again = await engine.authorise('write_file', 'write', { path: 'y.txt', content: 'hi' }, cwd)
  check('allow-tool is remembered for later calls', again.allowed && asked.length === 1)

  const hard = await engine.authorise('run_command', 'execute', { command: 'format C:' }, cwd)
  check('hard blocks refuse without prompting', !hard.allowed && asked.length === 1)
  check('hard block explains itself', /hard-block/i.test(hard.reason ?? ''))

  const denier = new PermissionEngine({ hardBlocksDisabled: () => false, ask: async () => 'deny' })
  const denied = await denier.authorise('delete_file', 'write', { path: 'x' }, cwd)
  check('denial is reported back to the model', !denied.allowed && /denied/i.test(denied.reason ?? ''))

  const override = new PermissionEngine({ hardBlocksDisabled: () => true, ask: async () => 'allow-once' })
  const overridden = await override.authorise('run_command', 'execute', { command: 'format C:' }, cwd)
  check('hard blocks can be overridden when explicitly disabled', overridden.allowed)
}

section('Approval prompts show resolved targets')
{
  const described = describeCall('write_file', { path: 'sub/../out.txt', content: 'x' }, 'C:\\work')
  check('relative paths are resolved before display', described.includes('out.txt') && !described.includes('..'))
  check('byte count is shown for writes', /\d+ bytes/.test(described))

  const cmd = describeCall('run_command', { command: 'npm test' }, 'C:\\work')
  check('commands are shown verbatim', cmd.includes('npm test') && cmd.includes('C:\\work'))
}

// ---------------------------------------------------------------- GGUF

section('GGUF parser')
{
  check('Q4_K tensor sizing matches the block layout', tensorByteSize(12, [256, 10]) === 10 * 144)
  check('F16 tensor sizing', tensorByteSize(1, [100, 100]) === 20000)
  check('unknown types degrade instead of throwing', tensorByteSize(999, [10]) === 10)

  // Build a minimal but valid GGUF file and read it back.
  const file = path.join(os.tmpdir(), `llmm-test-${Date.now()}.gguf`)
  fs.writeFileSync(file, buildTestGguf())
  try {
    const meta = await readGguf(file)
    check('reads the header', meta.version === 3 && meta.tensorCount === 2)
    check('reads string metadata', meta.kv['general.architecture'] === 'llama')
    check('reads numeric metadata', meta.kv['llama.block_count'] === 2)
    check('elides oversized arrays', meta.kv['tokenizer.ggml.tokens']?.elided === true)
    check('reads the tensor directory', meta.tensors.length === 2 && meta.tensors[0].name === 'blk.0.attn_q.weight')

    const info = extractArchInfo(meta)
    check('extracts architecture', info.architecture === 'llama')
    check('derives head_dim from embedding/heads', info.headDim === 64)
    check('separates per-layer from non-layer weights', info.perLayerBytes > 0 && info.nonLayerBytes > 0)
    check('takes vocab size from the elided token array', info.vocabSize === 32000)
  } finally {
    fs.rmSync(file, { force: true })
  }
}

// ---------------------------------------------------------------- recommendation

section('Unknown quant types reconcile against file size')
{
  // llama.cpp keeps adding tensor types (TQ1_0, TQ2_0, MXFP4, Q1_0...). A type this build has no
  // block layout for would otherwise be counted at 1 byte/element — an under-estimate, which is
  // the dangerous direction: the engine would plan more context than fits and the load OOMs.
  check('known types are recognised', isKnownGgmlType(12) && isKnownGgmlType(14))
  check('a future type is flagged unknown', !isKnownGgmlType(99))
  check('unknown types fall back to 1 byte/element', tensorByteSize(99, [1000]) === 1000)

  const file = path.join(os.tmpdir(), `llmm-unknown-${Date.now()}.gguf`)
  // Same fixture, but the weight tensors claim a type this build does not know.
  fs.writeFileSync(file, buildTestGguf(99))
  try {
    const meta = await readGguf(file)
    const naive = extractArchInfo(meta)
    // Pretend the real file is far larger than the naive 1-byte-per-element arithmetic suggests.
    const realSize = meta.dataOffset + 8 * 1024 ** 3
    const reconciled = extractArchInfo(meta, realSize)

    check('flags the unrecognised type', reconciled.unknownTensorTypes.includes(99))
    check('naive total under-counts', naive.weightBytes < 8 * 1024 ** 3)
    check(
      'reconciled total matches the file',
      Math.abs(reconciled.weightBytes - 8 * 1024 ** 3) < 1024,
      `${reconciled.weightBytes}`
    )
    check('per-layer and non-layer both scale', reconciled.perLayerBytes > naive.perLayerBytes && reconciled.nonLayerBytes > naive.nonLayerBytes)
    check('the split ratio is preserved', Math.abs(
      reconciled.perLayerBytes / reconciled.weightBytes - naive.perLayerBytes / naive.weightBytes
    ) < 1e-6)
  } finally {
    fs.rmSync(file, { force: true })
  }

  // A known-type file must be left alone: small deltas are padding, not error.
  const known = path.join(os.tmpdir(), `llmm-known-${Date.now()}.gguf`)
  fs.writeFileSync(known, buildTestGguf(12))
  try {
    const meta = await readGguf(known)
    const plain = extractArchInfo(meta)
    const withSize = extractArchInfo(meta, meta.dataOffset + plain.weightBytes)
    check('no unknown types on a normal file', withSize.unknownTensorTypes.length === 0)
    check('an accurate total is not rewritten', Math.abs(withSize.weightBytes - plain.weightBytes) < 1)
  } finally {
    fs.rmSync(known, { force: true })
  }
}

section('Quant recommendation')
{
  const files = [
    { filename: 'm-Q2_K.gguf', bytes: 3 * GB, quant: 'Q2_K', url: '', isMmproj: false, shard: null },
    { filename: 'm-Q4_K_M.gguf', bytes: 6 * GB, quant: 'Q4_K_M', url: '', isMmproj: false, shard: null },
    { filename: 'm-Q8_0.gguf', bytes: 12 * GB, quant: 'Q8_0', url: '', isMmproj: false, shard: null },
    { filename: 'mmproj-f16.gguf', bytes: 0.6 * GB, quant: null, url: '', isMmproj: true, shard: null }
  ]

  const big = recommendQuant(files, hw([gpu('RTX 4090', 24, 23)]), 65536)
  check('picks the highest quality that fits on a big card', big?.filename === 'm-Q8_0.gguf', big?.filename)
  check('says it fits', big?.fitsFullyOnGpu === true)

  const small = recommendQuant(files, hw([gpu('RTX 3060', 12, 11)]), 65536)
  check('steps down on a smaller card', small?.filename !== 'm-Q8_0.gguf', small?.filename)

  const tiny = recommendQuant(files, hw([gpu('GTX 1050', 4, 3.5)]), 65536)
  check('admits when nothing fits', tiny?.fitsFullyOnGpu === false)
  check('explains the recommendation', !!tiny?.reason && tiny.reason.length > 20)

  check('never recommends an mmproj as the model', ![big, small, tiny].some((r) => r?.filename.includes('mmproj')))
  check('finds the mmproj companion', findMmprojFor(files)?.filename === 'mmproj-f16.gguf')
}

section('Byte formatting')
{
  // A missing kilobyte step printed "438009 B" where "428 KB" was meant.
  check('bytes below a kilobyte', fmtBytes(512) === '512 B', fmtBytes(512))
  check('kilobytes', fmtBytes(438009) === '428 KB', fmtBytes(438009))
  check('megabytes', fmtBytes(17_000_000) === '16 MB', fmtBytes(17_000_000))
  check('gigabytes keep two decimals below ten', fmtBytes(4 * GB) === '4.00 GB', fmtBytes(4 * GB))
  check('gigabytes drop a decimal above ten', fmtBytes(18_973_870_432) === '17.7 GB', fmtBytes(18_973_870_432))
  check('zero is not a raw number with no unit', fmtBytes(0) === '0 B', fmtBytes(0))
  check('non-finite input is handled', fmtBytes(Number.NaN) === '?', fmtBytes(Number.NaN))
}


section('Reasoning detection')
{
  // Shaped after the templates actually shipped by these models. The local-file check in
  // scripts/reasoning-check.mjs covers the real thing; these pin the awkward cases.

  // Qwen3.8: validates against a tuple *after* remapping 'high' onto 'xhigh'.
  const qwen38 = `
    {%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'high' %}
    {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort') }}
    {%- endif %}
  `
  const a = detectReasoning(qwen38)
  check('effort model is detected', a.kind === 'effort', a.kind)
  check('levels come from the validation tuple', a.levels.join(',') === 'low,medium,xhigh', a.levels.join(','))
  // 'high' is an alias for 'xhigh' here; offering it would be a stop that changes nothing.
  check('an aliased level is not offered', !a.levels.includes('high'), a.levels.join(','))
  check('the template default is used', a.defaultLevel === 'xhigh', String(a.defaultLevel))
  check('enable_thinking implies it can be switched off', a.canDisable === true)

  // gpt-oss: no validation tuple, only equality branches, and different level names.
  const gptoss = `
    {%- if reasoning_effort == 'low' %}Reasoning: low
    {%- elif reasoning_effort == 'medium' %}Reasoning: medium
    {%- elif reasoning_effort == 'high' %}Reasoning: high
    {%- endif %}
  `
  const b = detectReasoning(gptoss)
  check('equality branches are used when there is no tuple', b.kind === 'effort', b.kind)
  check('gpt-oss levels are read, not assumed', b.levels.join(',') === 'low,medium,high', b.levels.join(','))
  check('without a template default, the strongest level is assumed', b.defaultLevel === 'high', String(b.defaultLevel))
  check('no enable_thinking means it cannot be switched off', b.canDisable === false)

  // Qwen3.6 / nanbeige: a toggle and nothing more.
  const toggle = `{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}`
  const c = detectReasoning(toggle)
  check('a toggle-only template is detected', c.kind === 'toggle', c.kind)
  check('a toggle exposes no levels', c.levels.length === 0)

  // Hermes-3 and friends.
  check('a plain template reports none', detectReasoning('{{ messages }}').kind === 'none')
  check('an absent template reports none', detectReasoning(undefined).kind === 'none')
  check('an empty template reports none', detectReasoning('').kind === 'none')

  // 'none' is llama.cpp's off switch, not a level.
  const withNone = `{%- if reasoning_effort in ('none', 'low', 'high') %}{%- endif %}`
  const d = detectReasoning(withNone)
  check("'none' is treated as off, not a level", d.levels.join(',') === 'low,high', d.levels.join(','))
  check("'none' implies it can be switched off", d.canDisable === true)

  // A derived variable with an unrelated name: templates normalise reasoning_effort before
  // branching on it, and the name they pick is arbitrary. Matching only names that happen to
  // contain 'reasoning_effort' worked for Qwen3.8 by luck and lost the control entirely for a
  // template that used a short name.
  const aliased = `
    {%- set r = reasoning_effort|default('medium') %}
    {%- if r not in ('low', 'medium', 'high') %}{{- raise_exception('nope') }}{%- endif %}
  `
  const f = detectReasoning(aliased)
  check('a renamed effort variable is still detected', f.kind === 'effort', f.kind)
  check('its levels are read from the tuple', f.levels.join(',') === 'low,medium,high', f.levels.join(','))
  check('its default is read too', f.defaultLevel === 'medium', String(f.defaultLevel))

  // A single level is not a choice worth a slider.
  const one = `{%- if reasoning_effort == 'high' %}{%- endif %}`
  check('one level alone is not an effort control', detectReasoning(one).kind === 'none', detectReasoning(one).kind)

  // Unknown names must still order deterministically rather than throwing them away.
  const exotic = `{%- if reasoning_effort in ('turbo', 'low', 'max') %}{%- endif %}`
  const e = detectReasoning(exotic)
  check('unknown level names are kept', e.levels.includes('turbo'), e.levels.join(','))
  check('known names still order correctly around them', e.levels.indexOf('low') < e.levels.indexOf('max'),
    e.levels.join(','))

  // The off value is only recorded when the template's own accepted set contains one.
  check('an off level is recorded when the template lists one', d.offValue === 'none', String(d.offValue))
  check('no off level is invented for a levels-only template', a.offValue === null, String(a.offValue))
  check('nor for one with only equality branches', b.offValue === null, String(b.offValue))
}

/*
 * Turning thinking off.
 *
 * Most effort templates enumerate levels and provide no way to say "none" — Qwen3.8 validates
 * against ('xhigh','medium','low') and raises on anything else. Off therefore cannot be built out
 * of the effort level alone; it rests on llama.cpp's own reasoning budget, which ends the thought
 * block without asking the template's permission. What must never happen is sending a level name
 * the template will reject, which turns "stop thinking" into a failed request.
 */
section('Reasoning: off')
{
  const levelsOnly = detectReasoning(
    `{%- set r = reasoning_effort|default('xhigh') %}
     {%- if r not in ('xhigh', 'medium', 'low') %}{{- raise_exception('nope') }}{%- endif %}`
  )
  const withOff = detectReasoning(`{%- if reasoning_effort in ('none', 'low', 'high') %}{%- endif %}`)
  const toggleOnly = detectReasoning(
    `{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}`
  )

  check('a levels-only template cannot disable via its own template', levelsOnly.canDisable === false)

  const offLevels = reasoningRequestFields(levelsOnly, 'off')
  check('off still works on a levels-only model', offLevels.reasoningBudget === 0, JSON.stringify(offLevels))
  check(
    'off never sends a level the template would raise on',
    offLevels.reasoningEffort === undefined,
    String(offLevels.reasoningEffort)
  )
  check('off asks the template too, in case it listens', offLevels.chatTemplateKwargs?.enable_thinking === false)

  const offNamed = reasoningRequestFields(withOff, 'off')
  check('off sends the level when the template accepts one', offNamed.reasoningEffort === 'none', String(offNamed.reasoningEffort))
  check('and still sets the budget', offNamed.reasoningBudget === 0)

  const offToggle = reasoningRequestFields(toggleOnly, 'off')
  check('off on a toggle model clears enable_thinking', offToggle.chatTemplateKwargs?.enable_thinking === false)
  check('off on a toggle model sends no level', offToggle.reasoningEffort === undefined, String(offToggle.reasoningEffort))

  // Everything that is not "off" must be untouched by the above.
  const on = reasoningRequestFields(levelsOnly, 'medium')
  check('a chosen level is sent as-is', on.reasoningEffort === 'medium', String(on.reasoningEffort))
  check('a chosen level sets no budget', on.reasoningBudget === undefined, String(on.reasoningBudget))
  check(
    'a level the model never advertised is dropped',
    reasoningRequestFields(levelsOnly, 'ludicrous').reasoningEffort === undefined
  )
  check('null means leave the template alone', Object.keys(reasoningRequestFields(levelsOnly, null)).length === 0)
  check('a non-reasoning model sends nothing at all',
    Object.keys(reasoningRequestFields(detectReasoning('{{ messages }}'), 'off')).length === 0)
}

/*
 * The effort choice is remembered per conversation, and level names belong to the model. Loading
 * a different model therefore leaves a choice it has never heard of, and the two things that read
 * that choice used to disagree about it: the request builder dropped an unknown level while the
 * slider rendered it as position zero ("Off"), and `isUltra` recognised Ultra on a model whose
 * control cannot offer it, running three sampling passes for no visible reason.
 */
section('Reasoning: a choice the model cannot express')
{
  const { sendableChoice } = await import('./built/reasoning.js')

  const levels = detectReasoning(
    `{%- if reasoning_effort in ('xhigh', 'medium', 'low') %}{%- endif %}`
  )
  const otherLevels = detectReasoning(`{%- if reasoning_effort in ('high', 'medium', 'low') %}{%- endif %}`)
  const toggle = detectReasoning(
    `{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}`
  )
  const none = detectReasoning('{{ messages }}')

  check('a level this model has survives', sendableChoice(levels, 'medium') === 'medium')
  check('a level from a different model does not', sendableChoice(otherLevels, 'xhigh') === null,
    String(sendableChoice(otherLevels, 'xhigh')))
  check('and what it becomes is what the request builder does with it — nothing',
    Object.keys(reasoningRequestFields(otherLevels, sendableChoice(otherLevels, 'xhigh'))).length === 0)

  check('off is expressible on any reasoning model', sendableChoice(levels, 'off') === 'off')
  check('off survives on a toggle model too', sendableChoice(toggle, 'off') === 'off')

  check('ultra survives where the control can offer it', sendableChoice(levels, 'ultra') === 'ultra')
  check('ultra does not survive onto a toggle model', sendableChoice(toggle, 'ultra') === null,
    String(sendableChoice(toggle, 'ultra')))
  check('a toggle model keeps its own on', sendableChoice(toggle, 'on') === 'on')

  check('a model with no reasoning at all expresses nothing', sendableChoice(none, 'medium') === null)
  check('null stays null', sendableChoice(levels, null) === null)
}

section('Export filenames')
{
  const F = (t) => exportFilename(t, 'fallback')

  check('ordinary titles pass through', F('State space models') === 'State space models', F('State space models'))
  check('forbidden characters are replaced', F('a/b:c*d?e') === 'a-b-c-d-e', F('a/b:c*d?e'))
  // Windows drops a trailing dot or space, so the file would not match the name we returned.
  check('trailing dot is removed', F('Notes.') === 'Notes', F('Notes.'))
  check('trailing space is removed', F('Notes   ') === 'Notes', F('Notes   '))
  // A title with nothing usable in it must not produce a bare ".md".
  check('punctuation-only title falls back', F('???') === 'fallback', F('???'))
  check('empty title falls back', F('') === 'fallback', F(''))
  check('dots-only title falls back', F('...') === 'fallback', F('...'))
  check('whitespace-only title falls back', F('   ') === 'fallback', F('   '))
  // Reserved device names are rejected by the filesystem no matter the extension.
  check('reserved name is escaped', F('CON') === 'CON-file', F('CON'))
  check('reserved name is escaped case-insensitively', F('nul') === 'nul-file', F('nul'))
  check('a name merely containing a reserved word is left alone', F('console') === 'console', F('console'))
  // Unicode is legal in Windows filenames; mangling it would be wrong.
  check('emoji survive', F('rocket ' + String.fromCodePoint(0x1f680)) === 'rocket ' + String.fromCodePoint(0x1f680), F('rocket'))
  check('CJK survives', F('\u4e2d\u6587') === '\u4e2d\u6587', F('\u4e2d\u6587'))
  // Control characters would be written verbatim into a path otherwise.
  check('control characters are stripped', F('a' + String.fromCharCode(7) + 'b') === 'a-b', F('a' + String.fromCharCode(7) + 'b'))
  check('long titles are truncated', F('x'.repeat(200)).length <= 60, String(F('x'.repeat(200)).length))
  check('truncation cannot leave a trailing dot', !F('y'.repeat(59) + '.' + 'z'.repeat(20)).endsWith('.'), F('y'.repeat(59) + '.' + 'z'.repeat(20)))
  check('backslash is treated as forbidden', F('a' + String.fromCharCode(92) + 'b') === 'a-b', F('a' + String.fromCharCode(92) + 'b'))

  // Re-exporting the same conversation must not destroy the earlier export.
  const taken = new Set([path.join('d', 'chat.md'), path.join('d', 'chat (2).md')])
  check('first free name is used', uniquePath('d', 'fresh', 'md', (x) => taken.has(x)) === path.join('d', 'fresh.md'))
  check('collisions get a suffix', uniquePath('d', 'chat', 'md', (x) => taken.has(x)) === path.join('d', 'chat (3).md'),
    uniquePath('d', 'chat', 'md', (x) => taken.has(x)))
}

// ---------------------------------------------------------------- schema migrations

/*
 * Migrations run destructive DDL against a database full of the user's chat history, and they
 * are forward-only — a mistake here is not recoverable on the next launch. Migration 4 rebuilds
 * `tasks` to add the foreign key it was originally declared without, so it is exercised against
 * a database shaped like the one a real upgrade would meet: live rows, and orphans left behind
 * by the missing cascade.
 */
section('Schema migrations')

// storage/paths resolves APPDATA_DIR at import time. Point it at a throwaway directory before
// db.js loads, so the suite can never read or write the real chat history — the stubbed
// Electron would refuse anyway, but only after the fact.
process.env.LLMM_APPDATA_DIR ??= fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-test-'))

{
  const { DatabaseSync } = await import('node:sqlite')
  const { MIGRATIONS } = await import('./built/db.js')

  // Mirrors getDb: each migration is applied once, in id order, and only if it has not been.
  const applyRange = (db, afterId, throughId) => {
    for (const m of MIGRATIONS) {
      if (m.id > afterId && m.id <= throughId) db.exec(m.sql)
    }
  }

  const db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON;')
  applyRange(db, 0, 3)

  db.exec("INSERT INTO chats (id, title, kind, created_at, updated_at) VALUES ('live', 'Live', 'agent', 1, 1);")
  db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('t1', 'live', 'live task', 0, 0);")
  db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('t2', 'live', 'finished task', 1, 3);")
  // Rows the missing cascade stranded when their chats were deleted.
  db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('t3', 'gone', 'orphan', 0, 0);")

  check('pre-migration schema keeps orphaned tasks', db.prepare('SELECT COUNT(*) n FROM tasks').get().n === 3)

  applyRange(db, 3, 4)

  const rows = db.prepare('SELECT id, chat_id, text, done, ord FROM tasks ORDER BY id').all()
  check('migration drops orphaned tasks', rows.length === 2 && rows.every((r) => r.chat_id === 'live'),
    JSON.stringify(rows.map((r) => r.id)))
  check('migration preserves done and ord', rows.find((r) => r.id === 't2')?.done === 1 && rows.find((r) => r.id === 't2')?.ord === 3)

  db.exec("DELETE FROM chats WHERE id = 'live';")
  check('deleting a chat now cascades to its tasks', db.prepare('SELECT COUNT(*) n FROM tasks').get().n === 0)

  let rejected = false
  try {
    db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('bad', 'nonexistent', 'x', 0, 0);")
  } catch {
    rejected = true
  }
  check('a task cannot be written against a missing chat', rejected)

  // Migration 5: the turn's plan is stored beside its message rather than pasted into it.
  applyRange(db, 4, 5)
  const cols = db.prepare('PRAGMA table_info(messages)').all().map((c) => c.name)
  check('messages can hold a plan', cols.includes('plan'), cols.join(','))
  check('and still hold reasoning', cols.includes('reasoning'), cols.join(','))

  db.close()
}

{
  // Every migration must be safe to re-run: getDb applies only unapplied ids, but a partially
  // applied upgrade should not wedge the app on the next launch either.
  const { DatabaseSync } = await import('node:sqlite')
  const { MIGRATIONS } = await import('./built/db.js')
  check('migration ids are unique and ordered',
    MIGRATIONS.every((m, i) => i === 0 || m.id > MIGRATIONS[i - 1].id))

  const fresh = new DatabaseSync(':memory:')
  fresh.exec('PRAGMA foreign_keys = ON;')
  let ok = true
  try {
    for (const m of MIGRATIONS) fresh.exec(m.sql)
  } catch {
    ok = false
  }
  check('a fresh database applies every migration cleanly', ok)
  fresh.close()
}

{
  /*
   * A migration that fails part-way must leave nothing behind.
   *
   * Migration 4 rebuilds `tasks` across four statements. Run outside a transaction, a failure
   * between the DROP and the RENAME left a database with no `tasks` table and no record of the
   * attempt — and the retry on the next launch then died on `tasks_new` already existing, for
   * good. This mirrors what getDb now does: BEGIN, apply, record, COMMIT — or ROLLBACK.
   */
  const { DatabaseSync } = await import('node:sqlite')

  const db = new DatabaseSync(':memory:')
  db.exec('CREATE TABLE keep (id INTEGER PRIMARY KEY);')
  db.exec("INSERT INTO keep (id) VALUES (1);")
  db.exec('CREATE TABLE _migrations (id INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL);')

  // A migration that does real work and then fails, exactly like a rebuild dying mid-sequence.
  const broken = 'CREATE TABLE added (id INTEGER); DROP TABLE keep; SELECT this_is_not_valid_sql();'
  let threw = false
  db.exec('BEGIN')
  try {
    db.exec(broken)
    db.prepare('INSERT INTO _migrations (id, applied_at) VALUES (?, ?)').run(99, 1)
    db.exec('COMMIT')
  } catch {
    db.exec('ROLLBACK')
    threw = true
  }

  check('a failing migration reports rather than half-applying', threw)

  const tables = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
    .all()
    .map((r) => r.name)
  check('the table it dropped is still there', tables.includes('keep'), tables.join(','))
  check('the table it created is not', !tables.includes('added'), tables.join(','))
  check('and it is not recorded as applied, so the next launch retries it',
    db.prepare('SELECT COUNT(*) n FROM _migrations').get().n === 0)

  db.close()
}

// ---------------------------------------------------------------- downloads

/*
 * Models are tens of gigabytes, so the download path has two properties worth guarding: the
 * bytes must be exactly right, and an interruption must not cost the whole transfer.
 *
 * The correctness half matters more than the speed half. A parallel download that writes one
 * part at the wrong offset produces a corrupt file that passes every length check and only fails
 * much later, when something tries to load it — by which time the download is long forgotten.
 * Run against a local server so ranges, failures and comparisons are all exact.
 */
section('Downloads: parallel parts, byte-exact and resumable')
{
  const http = await import('node:http')
  const crypto = await import('node:crypto')
  const { downloadQueue } = await import('./built/queue.js')

  // Above MIN_PART_BYTES * 2, so the planner actually splits it. A smaller file is checked
  // separately below, because declining to split a small file is deliberate behaviour.
  const TOTAL = 24 * 1024 * 1024
  const BODY = Buffer.alloc(TOTAL)
  // Structured rather than random, so a misplaced part is a mismatch instead of noise.
  for (let i = 0; i < TOTAL; i += 4) BODY.writeUInt32BE(i, i)
  const EXPECTED = crypto.createHash('sha256').update(BODY).digest('hex')

  let dropNext = false
  let rangeRequests = 0

  const server = http.createServer((req, res) => {
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'content-length': String(TOTAL), 'accept-ranges': 'bytes' })
      return res.end()
    }
    const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '')
    if (m) rangeRequests++
    const start = m ? Number(m[1]) : 0
    const end = m && m[2] ? Number(m[2]) : TOTAL - 1
    const slice = BODY.subarray(start, end + 1)
    res.writeHead(m ? 206 : 200, {
      'content-length': String(slice.length),
      'accept-ranges': 'bytes',
      ...(m ? { 'content-range': `bytes ${start}-${end}/${TOTAL}` } : {})
    })
    if (dropNext) {
      dropNext = false
      res.write(slice.subarray(0, Math.floor(slice.length / 3)))
      return res.destroy()
    }
    res.end(slice)
  })

  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  const base = `http://127.0.0.1:${server.address().port}`
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-dl-'))

  const finished = (id) =>
    new Promise((resolve) => {
      const tick = () => {
        const row = downloadQueue.list().find((d) => d.id === id)
        if (row && (row.status === 'done' || row.status === 'failed' || row.status === 'cancelled')) resolve(row)
        else setTimeout(tick, 50)
      }
      tick()
    })

  const sha = (f) => crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex')

  // ---- several connections, clean run
  downloadQueue.setConnections(4)
  const destA = path.join(dir, 'a.gguf')
  const a = downloadQueue.enqueue({ repo: null, filename: 'a.gguf', url: `${base}/a`, dest: destA, bytesTotal: TOTAL })
  const rowA = await finished(a.id)
  check('a parallel download completes', rowA.status === 'done', `${rowA.status} ${rowA.error ?? ''}`)
  check('and the file is byte-identical', fs.existsSync(destA) && sha(destA) === EXPECTED)
  check('and it really did split into ranges', rangeRequests > 1, `${rangeRequests} range requests`)
  check('and no resume sidecar is left behind', !fs.existsSync(`${destA}.parts`) &&
    !fs.existsSync(path.join(dir, '.partial', 'a.gguf.part.parts')))

  // ---- a dropped connection must be retried automatically, not parked as failed
  dropNext = true
  const destB = path.join(dir, 'b.gguf')
  const b = downloadQueue.enqueue({ repo: null, filename: 'b.gguf', url: `${base}/b`, dest: destB, bytesTotal: TOTAL })
  const rowB = await finished(b.id)
  check('a dropped connection recovers without the user intervening', rowB.status === 'done', `${rowB.status} ${rowB.error ?? ''}`)
  check('and still produces the right bytes', fs.existsSync(destB) && sha(destB) === EXPECTED)

  // ---- one connection, the fallback path for servers without ranges
  downloadQueue.setConnections(1)
  const destC = path.join(dir, 'c.gguf')
  const c = downloadQueue.enqueue({ repo: null, filename: 'c.gguf', url: `${base}/c`, dest: destC, bytesTotal: TOTAL })
  const rowC = await finished(c.id)
  check('a single-connection download still works', rowC.status === 'done', `${rowC.status} ${rowC.error ?? ''}`)
  check('and matches too', fs.existsSync(destC) && sha(destC) === EXPECTED)

  /*
   * A file too small to be worth splitting must quietly use one connection. Eight requests,
   * eight TLS handshakes and eight slow-starts to move a couple of megabytes each is slower than
   * simply asking once.
   */
  downloadQueue.setConnections(8)
  rangeRequests = 0
  const smallServer = http.createServer((req, res) => {
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'content-length': '1048576', 'accept-ranges': 'bytes' })
      return res.end()
    }
    if (req.headers.range) rangeRequests++
    res.writeHead(200, { 'content-length': '1048576', 'accept-ranges': 'bytes' })
    res.end(Buffer.alloc(1024 * 1024))
  })
  await new Promise((r) => smallServer.listen(0, '127.0.0.1', r))
  const smallBase = `http://127.0.0.1:${smallServer.address().port}`
  const destD = path.join(dir, 'd.gguf')
  const d = downloadQueue.enqueue({ repo: null, filename: 'd.gguf', url: `${smallBase}/d`, dest: destD, bytesTotal: 1024 * 1024 })
  const rowD = await finished(d.id)
  check('a small file completes', rowD.status === 'done', `${rowD.status} ${rowD.error ?? ''}`)
  check('and is not split across connections', rangeRequests === 0, `${rangeRequests} range requests`)
  smallServer.close()

  /*
   * A checksum turns "arrived complete" into "arrived correct".
   *
   * The length check catches a transfer that stopped early. It cannot catch one that arrived
   * whole and wrong — a flipped bit, a mangled range, a part written at the wrong offset. That
   * file looks entirely normal until llama.cpp refuses it days later, complaining about the file
   * rather than the download.
   */
  downloadQueue.setConnections(4)
  const destE = path.join(dir, 'e.gguf')
  const e = downloadQueue.enqueue({
    repo: null,
    filename: 'e.gguf',
    url: `${base}/e`,
    dest: destE,
    bytesTotal: TOTAL,
    sha256: EXPECTED
  })
  const rowE = await finished(e.id)
  check('a download matching its checksum completes', rowE.status === 'done', `${rowE.status} ${rowE.error ?? ''}`)
  check('and lands in the library', fs.existsSync(destE) && sha(destE) === EXPECTED)

  // The same bytes, declared to be something else: complete, correct length, wrong contents.
  const destF = path.join(dir, 'f.gguf')
  const wrong = 'f'.repeat(64)
  const f = downloadQueue.enqueue({
    repo: null,
    filename: 'f.gguf',
    url: `${base}/f`,
    dest: destF,
    bytesTotal: TOTAL,
    sha256: wrong
  })
  const rowF = await finished(f.id)
  check('a corrupt download is caught rather than accepted', rowF.status === 'failed', rowF.status)
  check('and says what was wrong', /Checksum mismatch/.test(rowF.error ?? ''), rowF.error ?? '')
  check('and never appears in the library', !fs.existsSync(destF))
  check('and the corrupt partial is discarded, not left to resume',
    !fs.existsSync(path.join(dir, '.partial', 'f.gguf.part')))

  // A source with no published checksum must still work; most files are not LFS.
  const destG = path.join(dir, 'g.gguf')
  const g = downloadQueue.enqueue({ repo: null, filename: 'g.gguf', url: `${base}/g`, dest: destG, bytesTotal: TOTAL })
  const rowG = await finished(g.id)
  check('a download without a checksum is not blocked', rowG.status === 'done', `${rowG.status} ${rowG.error ?? ''}`)
  check('and still arrives intact', fs.existsSync(destG) && sha(destG) === EXPECTED)

  server.close()
  fs.rmSync(dir, { recursive: true, force: true })
  downloadQueue.setConnections(4)
}

// ---------------------------------------------------------------- new agent tools

section('multi_edit: all of them, or none')
{
  const { workflowTools } = await import('./built/workflow.js')
  const multiEdit = workflowTools.find((t) => t.name === 'multi_edit')
  const ctx = { cwd: os.tmpdir(), sessionId: 's', signal: new AbortController().signal, timeoutMs: 5000 }

  const write = (body) => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-edit-')), 'f.txt')
    fs.writeFileSync(file, body)
    return file
  }

  const file = write('alpha\nbeta\ngamma\n')
  await multiEdit.run({ path: file, edits: [
    { old_string: 'alpha', new_string: 'ALPHA' },
    { old_string: 'gamma', new_string: 'GAMMA' }
  ] }, ctx)
  check('applies every edit in one write', fs.readFileSync(file, 'utf8') === 'ALPHA\nbeta\nGAMMA\n', fs.readFileSync(file, 'utf8'))

  // The point of the tool: a bad edit late in the list must not leave the earlier ones on disk.
  const atomic = write('alpha\nbeta\ngamma\n')
  let threw = false
  try {
    await multiEdit.run({ path: atomic, edits: [
      { old_string: 'alpha', new_string: 'ALPHA' },
      { old_string: 'nowhere', new_string: 'x' }
    ] }, ctx)
  } catch {
    threw = true
  }
  check('a failing edit reports rather than half-applying', threw)
  check('and the file is untouched', fs.readFileSync(atomic, 'utf8') === 'alpha\nbeta\ngamma\n', fs.readFileSync(atomic, 'utf8'))

  // Ambiguity is refused rather than guessed at, as edit_file does.
  const ambiguous = write('x\nx\n')
  let refused = false
  try {
    await multiEdit.run({ path: ambiguous, edits: [{ old_string: 'x', new_string: 'y' }] }, ctx)
  } catch {
    refused = true
  }
  check('an ambiguous match is refused', refused)
  check('unless replace_all is set',
    (await multiEdit.run({ path: ambiguous, edits: [{ old_string: 'x', new_string: 'y', replace_all: true }] }, ctx)) &&
      fs.readFileSync(ambiguous, 'utf8') === 'y\ny\n')
}

/*
 * A model cannot know how long a build or a server start takes, and guesses badly in the
 * expensive direction. The ceiling starts short and climbs only while it keeps waiting, so a
 * first guess of "sleep a minute" costs five seconds and a check instead.
 */
section('wait: short first, longer only if it keeps waiting')
{
  const { workflowTools, resetWaitEscalation } = await import('./built/workflow.js')
  const wait = workflowTools.find((t) => t.name === 'wait')
  const ctx = { cwd: os.tmpdir(), sessionId: 'escalation', signal: new AbortController().signal, timeoutMs: 5000 }

  resetWaitEscalation('escalation')

  // Asking for the maximum on the first wait is capped hard, and says so.
  const started = Date.now()
  const first = await wait.run({ seconds: 60 }, ctx)
  const elapsed = Date.now() - started
  check('a 60s first wait is capped to 5s', elapsed >= 4500 && elapsed < 8000, `${elapsed}ms`)
  check('and explains that it was capped', /cap at this point/.test(first), first)
  check('and names the next ceiling', /up to 10s/.test(first), first)

  // The ceiling climbs while it is still waiting.
  resetWaitEscalation('ceilings')
  const ceilings = []
  for (let i = 0; i < 4; i++) {
    const out = await wait.run({ seconds: 1 }, { ...ctx, sessionId: 'ceilings' })
    ceilings.push(out.match(/up to (\d+)s/)?.[1])
  }
  check('the ceiling doubles across consecutive waits', ceilings.join(',') === '10,20,40,60', ceilings.join(','))

  // Doing anything else means it is no longer holding, so the ceiling drops back.
  resetWaitEscalation('ceilings')
  const afterReset = await wait.run({ seconds: 1 }, { ...ctx, sessionId: 'ceilings' })
  check('and resets once the agent does something else', /up to 10s/.test(afterReset), afterReset)

  // A short wait under the ceiling is honoured exactly, with no complaint.
  resetWaitEscalation('honest')
  const short = await wait.run({ seconds: 2 }, { ...ctx, sessionId: 'honest' })
  check('a sensible short wait is honoured as asked', /Waited 2s/.test(short) && !/cap at this point/.test(short), short)

  // Stop must not have to sit through it.
  const aborter = new AbortController()
  resetWaitEscalation('abort')
  setTimeout(() => aborter.abort(), 100)
  const abortStart = Date.now()
  await wait.run({ seconds: 5 }, { ...ctx, sessionId: 'abort', signal: aborter.signal })
  check('an aborted wait returns immediately', Date.now() - abortStart < 1500, `${Date.now() - abortStart}ms`)
}

// ---------------------------------------------------------------- settings bounds

/*
 * The settings file is plain JSON a person can edit, and the UI used to write to it on every
 * keystroke through `Number(input.value)` — so an empty box stored 0 and a lone minus sign
 * stored NaN, which JSON writes as null. `maxToolCallsPerTurn` is the one that bites: the agent
 * loop is `while (calls < max)`, and both `0 < 0` and `0 < null` are false, so the agent
 * answered nothing at all, gave no error, and kept doing so after a restart.
 */
section('Settings: values other code does arithmetic on')
{
  const fresh = fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-settings-'))
  process.env.LLMM_APPDATA_DIR = fresh
  const settingsFile = path.join(fresh, 'settings.json')

  const load = async (stored) => {
    fs.writeFileSync(settingsFile, JSON.stringify(stored))
    // Fresh module per case: loadSettings memoises, which is the behaviour we want in the app.
    const mod = await import(`./built/settings.js?case=${encodeURIComponent(JSON.stringify(stored))}`)
    return mod.loadSettings()
  }

  const broken = await load({ agent: { maxToolCallsPerTurn: null } })
  check('a null tool-call ceiling is repaired, not carried',
    broken.agent.maxToolCallsPerTurn === 50, String(broken.agent.maxToolCallsPerTurn))
  check('and the repaired value lets the loop run at all', 0 < broken.agent.maxToolCallsPerTurn)

  const zero = await load({ agent: { maxToolCallsPerTurn: 0 } })
  check('zero is raised to the floor', zero.agent.maxToolCallsPerTurn === 1, String(zero.agent.maxToolCallsPerTurn))

  const negative = await load({ autoFit: { headroomMb: -500 } })
  check('a negative headroom is clamped to zero', negative.autoFit.headroomMb === 0, String(negative.autoFit.headroomMb))

  const huge = await load({ ultra: { samples: 99 } })
  check('an out-of-range sample count is clamped to the engine ceiling',
    huge.ultra.samples === 8, String(huge.ultra.samples))

  const text = await load({ agent: { commandTimeoutMs: 'soon' } })
  check('a non-number falls back to the default', text.agent.commandTimeoutMs === 120000, String(text.agent.commandTimeoutMs))

  const good = await load({ agent: { maxToolCallsPerTurn: 25 }, autoFit: { headroomMb: 1024 } })
  check('a legitimate value is left alone', good.agent.maxToolCallsPerTurn === 25 && good.autoFit.headroomMb === 1024,
    `${good.agent.maxToolCallsPerTurn} / ${good.autoFit.headroomMb}`)

  fs.rmSync(fresh, { recursive: true, force: true })
}

// ---------------------------------------------------------------- transcript order

/*
 * `created_at` is a millisecond stamp, and an agent turn writes several messages inside one.
 * Ordering by it alone leaves the arrangement of a whole turn to whatever SQLite feels like
 * returning, and cutting a transcript at a timestamp cannot express "from this message onward"
 * when several share one. Both are ordered and cut by (created_at, rowid) instead.
 */
section('Transcript order and truncation')
{
  const { createChat, appendMessage, loadMessages, truncateFrom } = await import('./built/repo.js')

  const chat = createChat({ title: 'Ordering', kind: 'agent' })
  // One millisecond, five messages — the shape an agent turn with two tool calls produces.
  const t = 1_700_000_000_000
  const ids = ['m1-user', 'm2-assistant', 'm3-tool', 'm4-assistant', 'm5-tool']
  for (const id of ids) {
    appendMessage(chat.id, { id, role: 'user', content: id, createdAt: t })
  }

  const ordered = loadMessages(chat.id).map((m) => m.id)
  check('messages sharing a millisecond keep the order they were written', ordered.join(',') === ids.join(','), ordered.join(','))

  const removed = truncateFrom(chat.id, 'm3-tool')
  const left = loadMessages(chat.id).map((m) => m.id)
  check('rewinding cuts from exactly that message', left.join(',') === 'm1-user,m2-assistant', left.join(','))
  check('and reports what it removed', removed === 3, String(removed))

  // A later message must still be cut even though its timestamp is larger.
  appendMessage(chat.id, { id: 'm6-later', role: 'user', content: 'later', createdAt: t + 50 })
  truncateFrom(chat.id, 'm2-assistant')
  check('a later message goes too', loadMessages(chat.id).map((m) => m.id).join(',') === 'm1-user')
}

{
  const { createChat, appendMessage, searchChats } = await import('./built/repo.js')

  // LIKE treats % and _ as wildcards, so an unescaped query matched far more than was asked for.
  const chat = createChat({ title: 'Search', kind: 'chat' })
  appendMessage(chat.id, { id: 's1', role: 'user', content: 'battery at 50% today', createdAt: 1 })
  appendMessage(chat.id, { id: 's2', role: 'user', content: 'battery at 5091 today', createdAt: 2 })
  appendMessage(chat.id, { id: 's3', role: 'user', content: 'call read_file next', createdAt: 3 })
  appendMessage(chat.id, { id: 's4', role: 'user', content: 'call readXfile next', createdAt: 4 })

  const percent = searchChats('50%')
  check('a percent sign is a character, not a wildcard', percent.length === 1, JSON.stringify(percent.map((r) => r.snippet)))

  const underscore = searchChats('read_file')
  check('an underscore matches only itself', underscore.length === 1, JSON.stringify(underscore.map((r) => r.snippet)))

  check('ordinary searches still work', searchChats('battery').length === 2)
}

// ---------------------------------------------------------------- ultra

/*
 * Ultra is mostly orchestration, but two pure pieces decide whether it is worth its cost: the
 * spread of temperatures that makes the samples differ, and the synthesis prompt that has to
 * turn several answers into one without averaging a right answer with a wrong one.
 */
section('Ultra')
{
  const { sampleTemperatures, synthesisMessages, looksDegenerate } = await import('./built/ultra.js')

  const three = sampleTemperatures(3)
  check('a sample per attempt', three.length === 3, three.join(','))
  check('temperatures rise across the run', three[0] < three[1] && three[1] < three[2], three.join(','))
  check('the first attempt is the least adventurous', three[0] <= 0.5, String(three[0]))
  check('none of them are incoherent', three.every((t) => t <= 1), three.join(','))
  check('a single sample still gets a usable temperature', sampleTemperatures(1).length === 1)
  // Identical samples would make best-of-N an expensive way to ask the same thing N times.
  check('two samples still differ from each other', new Set(sampleTemperatures(2)).size === 2)
  check('eight samples are all distinct', new Set(sampleTemperatures(8)).size === 8)

  const original = [{ role: 'user', content: 'what is 2+2?' }]
  const samples = [
    { index: 0, answer: 'four', reasoning: 'x'.repeat(9000), continuations: 2, temperature: 0.45, ms: 10 },
    { index: 1, answer: 'five', reasoning: 'y'.repeat(9000), continuations: 1, temperature: 0.95, ms: 12 }
  ]
  const msgs = synthesisMessages(original, samples)

  check('the original turn is preserved', msgs[0].content === 'what is 2+2?')
  check('one instruction is appended', msgs.length === original.length + 1)

  const prompt = msgs.at(-1).content
  check('every candidate is offered', prompt.includes('four') && prompt.includes('five'))
  /*
   * The reasoning must not come along. A forced sample thinks for thousands of tokens, and
   * passing that back for each candidate would exhaust the window before the pass that has to
   * read them could start.
   */
  check('reasoning is left out of the prompt', !prompt.includes('x'.repeat(50)) && !prompt.includes('y'.repeat(50)),
    `${prompt.length} chars`)
  check('it is told to choose rather than blend', /decide which is correct|blend/i.test(prompt))
  check('it is told not to narrate the process', /do not mention/i.test(prompt))

  /*
   * Degeneration detection, against the real thing.
   *
   * Forced to keep thinking about "Idk", the model produced this: sentence after sentence
   * announcing a check, on a question containing no algebra at all. Every line differs, so
   * comparing lines finds nothing — what gives it away is that they all open the same way.
   */
  const vamping = [
    'Let me verify the coset',
    'Let me reconsider the homomorphism',
    'Let me check the kernel',
    'Let me verify the image',
    'Let me reconsider the exact sequence',
    'Let me check the extension',
    'Let me verify the splitting',
    'Let me reconsider the quotient',
    'Let me check the injection'
  ].join('. ')
  check('a forced loop is recognised', looksDegenerate(vamping))

  // Real reasoning must survive the same test, or forcing is disabled by its own safety net.
  const genuine = [
    'The user is asking about the config key rename',
    'There are three call sites in the loader',
    'The second one reads from an env var, so it needs care',
    'I should grep before editing anything',
    'Renaming without checking the tests would break the suite',
    'Let me look at how the default is applied',
    'The migration also references the old name',
    'That means two changes, not one',
    'I will start with the grep'
  ].join('. ')
  check('methodical reasoning is not mistaken for a loop', !looksDegenerate(genuine))

  check('a short passage is never judged', !looksDegenerate('Let me check. Let me verify. Let me see.'))
  check('empty text is not degenerate', !looksDegenerate(''))

  const empty = synthesisMessages(original, [{ index: 0, answer: '   ', reasoning: '', continuations: 0, temperature: 0.6, ms: 1 }])
  check('an empty answer is labelled, not dropped silently',
    empty.at(-1).content.includes('no answer produced'))
}

/*
 * The agent's variant answers a different question: these candidates are courses of action, and
 * the output has to be one coherent sequence rather than a merge. A plan assembled from parts of
 * several is one nobody checked end to end.
 */
section('Ultra: agent plans')
{
  const { planSynthesisMessages, planPreamble } = await import('./built/ultra.js')

  const msgs = planSynthesisMessages('rename the config key', ['1. grep for it, 2. edit', '1. edit blindly'])
  check('the plan prompt is a single turn', msgs.length === 1 && msgs[0].role === 'user')

  const p = msgs[0].content
  check('the task is restated', p.includes('rename the config key'))
  check('every plan is offered', p.includes('grep for it') && p.includes('edit blindly'))
  check('it is told to pick a base rather than interleave', /interleaving|soundest/i.test(p))
  check('it is told to answer with the plan alone', /no preamble/i.test(p))

  const blank = planSynthesisMessages('task', ['', '   '])
  check('an empty plan is labelled', blank[0].content.includes('no plan produced'))

  /*
   * The preamble has to say the plan is provisional. It was written before any of the work was
   * done, so an agent that follows it past a contradicting tool result is following a guess.
   */
  const pre = planPreamble('  1. do the thing  ')
  check('the plan survives into the preamble', pre.includes('1. do the thing'))
  check('the preamble is trimmed', !pre.endsWith(' '))
  check('the run is told the tool result wins', /trust the tool result/i.test(pre))
}

// ---------------------------------------------------------------- markdown

// Lives in its own module: the grammar has enough cases to be worth grouping.
const { runMarkdownTests } = await import('./markdown-tests.mjs')
await runMarkdownTests(check, section)

// ---------------------------------------------------------------- summary

console.log(`\n${'='.repeat(60)}`)
console.log(`${pass} passed, ${fail} failed`)
if (failures.length) {
  console.log('\nFailures:')
  failures.forEach((f) => console.log(`  - ${f}`))
}
console.log('')
process.exit(fail ? 1 : 0)

// ---------------------------------------------------------------- helpers

/** Construct a minimal valid GGUF v3 file in memory. */
function buildTestGguf(weightType = 12) {
  const chunks = []
  const u32 = (n) => {
    const b = Buffer.alloc(4)
    b.writeUInt32LE(n)
    return b
  }
  const u64 = (n) => {
    const b = Buffer.alloc(8)
    b.writeBigUInt64LE(BigInt(n))
    return b
  }
  const str = (s) => {
    const body = Buffer.from(s, 'utf8')
    return Buffer.concat([u64(body.length), body])
  }

  chunks.push(u32(0x46554747)) // "GGUF"
  chunks.push(u32(3)) // version
  chunks.push(u64(2)) // tensor count
  chunks.push(u64(5)) // kv count

  // general.architecture = "llama"  (type 8 = string)
  chunks.push(str('general.architecture'), u32(8), str('llama'))
  // llama.block_count = 2  (type 4 = uint32)
  chunks.push(str('llama.block_count'), u32(4), u32(2))
  // llama.embedding_length = 512
  chunks.push(str('llama.embedding_length'), u32(4), u32(512))
  // llama.attention.head_count = 8  -> head_dim should derive to 64
  chunks.push(str('llama.attention.head_count'), u32(4), u32(8))
  // tokenizer.ggml.tokens: an array of 32000 strings, which must be elided
  chunks.push(str('tokenizer.ggml.tokens'), u32(9), u32(8), u64(32000))
  for (let i = 0; i < 32000; i++) chunks.push(str(`t${i}`))

  // Tensor directory: one block tensor and one output tensor.
  chunks.push(str('blk.0.attn_q.weight'), u32(2), u64(512), u64(512), u32(weightType), u64(0))
  chunks.push(str('output.weight'), u32(2), u64(512), u64(32000), u32(weightType), u64(1024))

  return Buffer.concat(chunks)
}
