/**
 * The bridge: one map of channel -> handler, used by BOTH the desktop IPC and the remote web
 * server. That is what makes "full parity" true by construction rather than by discipline —
 * there is no second, thinner implementation to drift.
 *
 * Remote gating happens in the web server (a deny-list of desktop-only channels), not here,
 * so a handler cannot forget to check.
 */

import { app, dialog, shell, BrowserWindow } from 'electron'
import path from 'node:path'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import type {
  AppSettings,
  Backend,
  FitPlan,
  FitResult,
  HardwareSnapshot,
  ModelRecord,
  AgentQuestion,
  PermissionDecision,
  PermissionRequest,
  AttachmentInfo
} from '@shared/types'
import { defaultModelsDir, exeDir, TOOL_OUTPUT_DIR } from './storage/paths'
import { loadSettings, patchSettings } from './storage/settings'
import { detectHardware, refreshFreeVram } from './hardware/gpu'
import { scanLibrary, libraryDiskUsage } from './models/library'
import { checkRelocation, keepInPlace, performMove } from './models/relocation'
import { DEFAULT_CONSTRAINTS, planFit, verifyPrediction } from './autofit/engine'
import { llama, type ContentPart, type CompletionOptions } from './runtime/llama'
import { missingBinaries, embeddingModelPath, vendorDiagnostics } from './runtime/binaries'
import { Agent } from './agent/loop'
import { killAllJobs } from './agent/tools/exec'
import { closeBrowser } from './agent/tools/browser'
import { listCheckpoints, rewindTo, discardCheckpoints } from './agent/checkpoints'
import { mcpManager } from './agent/mcp'
import { addMemory, allMemory, deleteMemory, updateMemory } from './agent/memory'
import { searchModels, listFiles, recommendQuant, findMmprojFor } from './downloads/hf'
import { downloadQueue } from './downloads/queue'
import { apiServer, requestQueue } from './api/server'
import { tunnel } from './remote/tunnel'
import { remoteWeb } from './remote/web'
import {
  certificates,
  startDdnsUpdates,
  stopDdnsUpdates,
  updateFreedns,
  verifyDomainPointsHere,
  publicIp
} from './remote/selfhost'
import {
  generateApiKey,
  getApiKey,
  getHfToken,
  hasPassword,
  setApiKey,
  setFreednsToken,
  setHfToken,
  setPassword
} from './remote/auth'
import * as rag from './rag'
import * as chats from './chat/repo'
import { buildContent, type VideoContext } from './chat/multimodal'
import { classifyAttachment, recordAttachment, IMAGE_EXT, AUDIO_EXT, VIDEO_EXT, TEXT_EXT } from './chat/repo'
import { historicalStats, liveStats, requestLog, recordGeneration, clearStats, setContextUsed } from './stats'
import { buildDiagnostics, logger } from './log'
import { exportChat, writeExport, all, run, get } from './storage/db'
import { exportFilename, uniquePath } from './storage/filenames'
import { reasoningRequestFields, isUltra, sendableChoice, type ReasoningChoice } from './models/reasoning'
import { ultraSamples, sampleTemperatures, planSynthesisMessages, planPreamble } from './ultra'
import { checkForUpdate, applyUpdate } from './update'

// ---------------------------------------------------------------- shared state

let hardware: HardwareSnapshot | null = null
let library: ModelRecord[] = []
let agent: Agent | null = null

const pendingPermissions = new Map<string, (d: PermissionDecision) => void>()

/**
 * Clarifying questions the agent is blocked on.
 *
 * Separate from `pendingPermissions` because the two settle differently: an unanswered approval
 * is safely denied, while an unanswered question has no safe default — the honest resolution is
 * to tell the model nobody answered so it can proceed on its own judgement.
 */
const pendingQuestions = new Map<string, (answer: string) => void>()

/**
 * Chat turns currently streaming, so they can be stopped.
 *
 * Keyed by conversation rather than held as a single controller: the API server and a remote
 * browser can each have a turn in flight alongside the desktop window, and stopping one must not
 * cut off another.
 */
const inFlightChats = new Map<string, AbortController>()

/**
 * The agent turn in flight, covering the orchestration around the loop as well as the loop.
 *
 * `agent.stop()` aborts whatever request the loop is making, which is enough for an ordinary
 * turn. Under Ultra it is not: each planning sample runs its own loop with its own controller,
 * so stopping one merely ended that sample and the next began — several more minutes of work
 * after the user asked for none. This is checked between samples and passed to the synthesis
 * call, which had no signal at all and could not be interrupted.
 */
let agentTurnAbort: AbortController | null = null

/**
 * Settle every outstanding approval prompt as a denial.
 *
 * The agent blocks on `ask()` until the renderer answers, and nothing else can unblock it. If
 * the user hits stop while a prompt is up — or a remote browser closes its tab mid-prompt, or
 * the window reloads — the promise never settles, the turn hangs for the life of the process,
 * and the session shows as running forever. Denying is the safe resolution: the tool is skipped
 * rather than run without an answer.
 */
function drainPendingPermissions(): void {
  for (const [id, resolve] of pendingPermissions) {
    pendingPermissions.delete(id)
    resolve('deny')
  }
  // A question left hanging would block the turn for the life of the process. There is no safe
  // default answer, so the model is told plainly that none came.
  for (const [id, resolve] of pendingQuestions) {
    pendingQuestions.delete(id)
    resolve('(no answer — the user did not respond; proceed on your own judgement and say what you assumed)')
  }
}

/**
 * Files this process has written and offered to show the user.
 *
 * `shell:reveal` only accepts paths from this set. The same handler map serves remote browser
 * sessions, so an unguarded reveal would let anyone past the remote password pop file-manager
 * windows anywhere on the machine. Bounded, because the only thing that reads it is a "Show"
 * button on a toast that has usually expired already.
 */
const revealable = new Set<string>()

function offerReveal(file: string): string {
  const resolved = path.resolve(file)
  revealable.add(resolved)
  if (revealable.size > 64) {
    const oldest = revealable.values().next().value
    if (oldest) revealable.delete(oldest)
  }
  return resolved
}

/**
 * The session the agent is currently working on.
 *
 * Every agent event carries this, so the renderer can attribute streamed text to the right
 * conversation even when that conversation's view is not mounted.
 */
let activeAgentSessionId = ''

/** Most recent context measurement for the agent turn in flight, persisted when it ends. */
let lastAgentContext = 0

type Emitter = (channel: string, payload: unknown) => void
let emit: Emitter = () => undefined

export function setEmitter(fn: Emitter): void {
  emit = fn
}

export function modelsDir(): string {
  return loadSettings().modelsDir ?? defaultModelsDir()
}

async function getHardware(refresh = false): Promise<HardwareSnapshot> {
  if (!hardware || refresh) hardware = await detectHardware()
  return hardware
}

/**
 * How often the moving parts of the hardware snapshot are re-measured.
 *
 * Everything the dashboard shows about the machine — free VRAM per adapter, GPU utilisation,
 * free RAM — was detected once at startup and then never again. The figures were not slow to
 * update, they were frozen: a card could fill up and empty again without the number moving.
 *
 * Only the volatile fields are refreshed. Re-running full detection would re-enumerate adapters
 * and re-read model metadata to learn what has not changed since launch.
 */
const HARDWARE_REFRESH_MS = 8_000

let hardwareTimer: NodeJS.Timeout | null = null

export function startHardwareRefresh(): void {
  if (hardwareTimer) return
  hardwareTimer = setInterval(() => {
    void (async () => {
      if (!hardware) return
      try {
        hardware = await refreshFreeVram(hardware)
        emit('hardware:update', hardware)
      } catch {
        // A transient failure to read the GPU is not worth reporting; the next tick will do.
      }
    })()
  }, HARDWARE_REFRESH_MS)
  // Must not be the reason the process stays alive at quit.
  hardwareTimer.unref()
}

export function stopHardwareRefresh(): void {
  if (hardwareTimer) clearInterval(hardwareTimer)
  hardwareTimer = null
}

function getAgent(): Agent {
  const s = loadSettings()
  if (!agent) {
    agent = new Agent({
      cwd: app.getPath('home'),
      planMode: s.agent.planMode,
      maxToolCallsPerTurn: s.agent.maxToolCallsPerTurn,
      commandTimeoutMs: s.agent.commandTimeoutMs,
      hardBlocksDisabled: s.agent.hardBlocksDisabled,
      compaction: s.agent.compaction,
      hfToken: getHfToken(),
      remoteToolsEnabled: s.agent.remoteToolsEnabled,
      requestPermission: (req: PermissionRequest) =>
        new Promise<PermissionDecision>((resolve) => {
          pendingPermissions.set(req.id, resolve)
          emit('agent:permission-request', req)
        }),
      askUser: (question: AgentQuestion) =>
        new Promise<string>((resolve) => {
          pendingQuestions.set(question.id, resolve)
          emit('agent:question', question)
        })
    })

    const sid = (): string => activeAgentSessionId
    agent.on('delta', (t: string) => emit('agent:delta', { sessionId: sid(), text: t }))
    agent.on('reasoning', (t: string) => emit('agent:reasoning', { sessionId: sid(), text: t }))
    agent.on('promptProgress', (p: { percent: number; processed: number; total: number; cached: number }) =>
      emit('agent:prompt-progress', { sessionId: sid(), ...p })
    )
    agent.on('contextUsed', (c: { used: number; max: number }) => {
      emit('agent:context', { sessionId: sid(), ...c })
      setContextUsed(c.used)
      // Kept for the end of the turn rather than written on every update — see the persist below.
      lastAgentContext = c.used
    })
    agent.on('message', (m) => emit('agent:message', { sessionId: sid(), message: m }))
    agent.on('toolCall', (c) => emit('agent:tool-call', { sessionId: sid(), call: c }))
    agent.on('toolCallPartial', (p: { index: number; name: string; args: string }) =>
      emit('agent:tool-call-partial', { sessionId: sid(), ...p })
    )
    agent.on('toolResult', (r) => emit('agent:tool-result', { sessionId: sid(), result: r }))
    agent.on('subToolCall', (c) => emit('agent:sub-tool-call', { sessionId: sid(), call: c }))
    agent.on('compacting', (info) => emit('agent:compacting', { sessionId: sid(), ...(info as object) }))
    agent.on('compacted', (info) => emit('agent:compacted', { sessionId: sid(), ...(info as object) }))
    agent.on('done', (reason) => {
      /*
       * Persisted once the turn is over, not while it runs.
       *
       * The live figure changes twice a second during a response; writing each one would be a
       * database round trip per update to record a number superseded immediately. Where the
       * conversation ended up is the only part that has to survive a restart.
       */
      const id = sid()
      if (id && lastAgentContext > 0) chats.setContextUsed(id, lastAgentContext)
      emit('agent:done', { sessionId: id, reason })
    })
    agent.on('error', (e) => emit('agent:error', { sessionId: sid(), message: e }))

    // Remembered approvals are per folder and survive restarts.
    agent.loadPermissionRules(
      all<{ tool: string; exact: string | null; scope: string }>('SELECT tool, exact, scope FROM permission_rules').map(
        (r) => ({ tool: r.tool, exact: r.exact ?? undefined, scope: r.scope })
      )
    )
  }
  return agent
}

/**
 * Bring the agent's options in line with current settings.
 *
 * Options used to be refreshed only at the start of a run, so anything that asked the agent a
 * question in between — the tool catalog, most visibly — answered from stale state. With plan
 * mode on, the UI still listed write and execute tools as available.
 */
function syncAgentOptions(): void {
  const s = loadSettings()
  agent?.updateOptions({
    planMode: s.agent.planMode,
    maxToolCallsPerTurn: s.agent.maxToolCallsPerTurn,
    commandTimeoutMs: s.agent.commandTimeoutMs,
    hardBlocksDisabled: s.agent.hardBlocksDisabled,
    compaction: s.agent.compaction,
    remoteToolsEnabled: s.agent.remoteToolsEnabled,
    backend: hardware?.backend,
    hfToken: getHfToken()
  })
}

function persistPermissionRules(): void {
  if (!agent) return
  run('DELETE FROM permission_rules')
  for (const rule of agent.exportPermissionRules()) {
    run(
      'INSERT INTO permission_rules (tool, exact, scope, created_at) VALUES (?, ?, ?, ?)',
      rule.tool,
      rule.exact ?? null,
      rule.scope,
      Date.now()
    )
  }
}

/** Size of the multimodal projector loaded alongside a model, if it has one. */
function companionSize(model: ModelRecord): number {
  if (!model.caps.mmprojPath) return 0
  try {
    return fs.statSync(model.caps.mmprojPath).size
  } catch {
    return 0
  }
}

/** Load a model by id using a plan, or auto-fit one if no plan is supplied. */
async function loadModelById(modelId: string, plan?: FitPlan): Promise<{ port: number; plan: FitPlan }> {
  const model = library.find((m) => m.id === modelId || m.filename === modelId)
  if (!model) throw new Error(`Model not found: ${modelId}`)
  if (!model.arch) throw new Error(model.error ?? 'Model metadata could not be parsed')

  const hw = await getHardware()
  const fresh = await refreshFreeVram(hw)
  hardware = fresh

  let chosen = plan
  if (!chosen) {
    const s = loadSettings()
    const result = planFit(model.arch, fresh, {
      ...DEFAULT_CONSTRAINTS,
      minKvType: s.autoFit.minKvType,
      preferredKvType: s.autoFit.preferredKvType,
      targetContext: s.autoFit.targetContext,
      idealContext: s.autoFit.idealContext,
      headroomBytes: s.autoFit.headroomMb * 1024 * 1024,
      companionBytes: companionSize(model),
      overrides: {}
    })
    chosen = result.chosen ?? result.alternatives[0]
    if (!chosen) throw new Error('No workable configuration was found for this model on this hardware.')
  }

  const loaded = await llama.load(model, chosen, fresh.backend)
  logger.info('model', `loaded ${model.filename}`, { ctx: chosen.contextLength, layers: chosen.gpuLayers })

  // Verify prediction against reality and feed the delta back into the headroom margin.
  const after = await refreshFreeVram(fresh)
  const actual = after.gpus.map((g, i) => Math.max(0, (fresh.gpus[i]?.freeVram ?? 0) - g.freeVram))
  const verdict = verifyPrediction(chosen, actual)
  if (verdict.suggestion) {
    logger.info('autofit', verdict.suggestion, { predicted: chosen.predictedVramPerGpu, actual })
    emit('autofit:verified', { ...verdict, predicted: chosen.predictedVramPerGpu, actual })
  }

  run(
    'INSERT OR REPLACE INTO model_configs (model_id, plan, predicted_vram, actual_vram, updated_at) VALUES (?, ?, ?, ?, ?)',
    model.id,
    JSON.stringify(chosen),
    JSON.stringify(chosen.predictedVramPerGpu),
    JSON.stringify(actual),
    Date.now()
  )
  run(
    'INSERT OR REPLACE INTO model_meta (model_id, repo, favourite, tags, last_used_at) VALUES (?, ?, COALESCE((SELECT favourite FROM model_meta WHERE model_id = ?), 0), ?, ?)',
    model.id,
    model.repo,
    model.id,
    JSON.stringify(model.tags),
    Date.now()
  )

  emit('model:status', {
    model: model.filename,
    modelId: model.id,
    port: loaded.port,
    plan: chosen,
    startedAt: loaded.startedAt,
    caps: model.caps
  })
  return { port: loaded.port, plan: chosen }
}

// ---------------------------------------------------------------- handlers

/** Staged uploads older than this are removed. */
const STAGED_UPLOAD_TTL_MS = 7 * 24 * 60 * 60 * 1000

/**
 * Remove everything in the tool-output directory that has aged out.
 *
 * Four things write here and only one of them was ever cleaned up:
 *   - staged uploads, because a remote session cannot send a path and posts the bytes instead;
 *   - the full text of any tool result too large to fit in context, kept so the agent can page
 *     back into it with read_file;
 *   - desktop and browser screenshots, several megabytes each on a high-resolution display;
 *   - video frames, sixteen JPEGs per clip.
 *
 * Only the first had a sweep, so the rest grew for the life of the install — a few long agent
 * sessions with screenshots and big greps put hundreds of megabytes here permanently. A week is
 * long enough that everything outlives the conversation that produced it.
 */
async function pruneStagedUploads(): Promise<void> {
  const cutoff = Date.now() - STAGED_UPLOAD_TTL_MS

  const sweep = async (dir: string): Promise<void> => {
    for (const entry of await fsp.readdir(dir, { withFileTypes: true }).catch(() => [])) {
      const target = path.join(dir, entry.name)
      try {
        // `attachments` is the long-lived home for staged uploads; its contents age out, it does
        // not. Everything else here — including each `frames-*` directory — goes as a unit.
        if (entry.isDirectory() && dir === TOOL_OUTPUT_DIR && entry.name === 'attachments') {
          await sweep(target)
          continue
        }
        if ((await fsp.stat(target)).mtimeMs < cutoff) {
          await fsp.rm(target, { recursive: true, force: true })
        }
      } catch {
        // Best effort: anything that cannot be examined is left alone.
      }
    }
  }

  await sweep(TOOL_OUTPUT_DIR)
}

/**
 * Fold attachments into an agent turn.
 *
 * Text-bearing files are inlined into the prompt so the agent can reason about them straight
 * away. Images and audio are handed over as content parts, exactly as the chat path does.
 *
 * They used to be reduced to a list of paths, on the reasoning that the agent has file tools and
 * could open them itself. It cannot: nothing in the tool set decodes an image, so a vision model
 * that was perfectly able to see the picture was instead handed a filename and left to guess —
 * reaching for Python, EXIF and the shape of the file name to answer "what is this?".
 *
 * Paths are still named alongside, because they remain genuinely useful: the agent can copy,
 * move or convert a file it can also see.
 */
/**
 * What a video may spend of this model's context, and whether the server can take one whole.
 *
 * Read at send time rather than cached: the window belongs to the loaded plan, which changes
 * whenever a different model or fit is loaded, and the server's modalities are a property of the
 * running process.
 */
async function videoContext(): Promise<VideoContext> {
  const s = loadSettings()
  const modalities = await llama.modalities()
  return {
    contextLength: llama.loaded?.plan.contextLength ?? 8192,
    share: s.video.contextShare,
    detail: s.video.detail,
    serverTakesVideo: modalities.video
  }
}

async function attachmentTurn(input: string, files: string[]): Promise<{ text: string; media: ContentPart[] }> {
  const caps = llama.loaded?.model.caps
  if (!caps) return { text: input, media: [] }

  const built = await buildContent('', files, caps, await videoContext())
  const sections: string[] = []
  const media: ContentPart[] = []

  for (const part of built.parts) {
    if (part.type === 'text') sections.push(part.text ?? '')
    else media.push(part)
  }

  // Named whether or not they were also sent as media, so the agent can act on the file itself.
  const onDisk = files.filter((f) => classifyAttachment(f) !== 'doc')
  if (onDisk.length) {
    sections.push(`Attached files on disk:\n${onDisk.map((f) => `- ${f}`).join('\n')}`)
  }
  if (built.notes.length) sections.push(built.notes.join('\n'))

  return {
    text: sections.length ? `${sections.join('\n\n')}\n\n${input}` : input,
    media
  }
}

export const handlers: Record<string, (...args: never[]) => unknown> = {
  // ---- settings
  'settings:get': () => ({ ...loadSettings(), hfToken: getHfToken() ? '***set***' : null }),
  'settings:patch': (patch: Partial<AppSettings>) => {
    const next = patchSettings(patch)
    // Apply straight away so the agent never operates under superseded settings.
    syncAgentOptions()
    // Takes effect on the next file rather than mid-transfer, which is the only sane moment to
    // change how many connections a download is using.
    downloadQueue.setConnections(next.downloads.connections)
    return next
  },
  'settings:set-password': (password: string) => {
    setPassword(password)
    return true
  },
  'settings:has-password': () => hasPassword(),
  'settings:set-hf-token': (token: string | null) => {
    setHfToken(token)
    downloadQueue.setToken(token)
    agent?.updateOptions({ hfToken: token })
    return true
  },
  'settings:set-freedns-token': (token: string | null) => {
    setFreednsToken(token)
    return true
  },

  // ---- hardware
  'hardware:get': async (refresh?: boolean) => {
    const hw = await getHardware(refresh)
    hardware = await refreshFreeVram(hw)
    return hardware
  },
  'runtime:missing-binaries': async () => missingBinaries((await getHardware()).backend),
  /**
   * What the bundled vendor tree looks like, for a backend of the caller's choosing.
   *
   * The argument was accepted and ignored. `verify:packaged` loops over cuda, vulkan and cpu and
   * reports a line for each — so the pre-ship check that exists to prove all three backends are
   * present was verifying the detected one three times, and an incomplete Vulkan tree would have
   * shipped green from a CUDA machine.
   */
  'runtime:vendor-diagnostics': async (backend?: Backend) =>
    vendorDiagnostics(backend ?? (await getHardware()).backend),
  'runtime:vendor-info': async () => ({
    exeDir: exeDir(),
    modelsDir: modelsDir(),
    missing: missingBinaries((await getHardware()).backend),
    embeddingModel: embeddingModelPath(),
    embeddingModelPresent: fs.existsSync(embeddingModelPath())
  }),

  // ---- library
  'library:scan': async () => {
    library = await scanLibrary(modelsDir())
    // Re-apply user metadata stored in the DB.
    for (const m of library) {
      const meta = all<{ favourite: number; tags: string | null; last_used_at: number | null }>(
        'SELECT favourite, tags, last_used_at FROM model_meta WHERE model_id = ?',
        m.id
      )[0]
      if (meta) {
        m.favourite = !!meta.favourite
        m.lastUsedAt = meta.last_used_at
      }
    }
    return library
  },
  'library:disk': () => libraryDiskUsage(modelsDir()),
  'library:set-favourite': (modelId: string, favourite: boolean) => {
    run(
      'INSERT INTO model_meta (model_id, favourite) VALUES (?, ?) ON CONFLICT(model_id) DO UPDATE SET favourite = excluded.favourite',
      modelId,
      favourite ? 1 : 0
    )
    const m = library.find((x) => x.id === modelId)
    if (m) m.favourite = favourite
    return true
  },
  'library:add-tag': (modelId: string, tag: string) => {
    const m = library.find((x) => x.id === modelId)
    if (m && !m.tags.includes(tag)) m.tags.push(tag)
    run(
      'INSERT INTO model_meta (model_id, tags) VALUES (?, ?) ON CONFLICT(model_id) DO UPDATE SET tags = excluded.tags',
      modelId,
      JSON.stringify(m?.tags ?? [])
    )
    return m?.tags ?? []
  },
  'library:delete-model': async (modelId: string) => {
    const m = library.find((x) => x.id === modelId)
    if (!m) throw new Error('Model not found')
    if (llama.loaded?.model.id === modelId) await llama.unload()
    await fsp.rm(m.path, { force: true })
    if (m.caps.mmprojPath) await fsp.rm(m.caps.mmprojPath, { force: true }).catch(() => undefined)
    library = library.filter((x) => x.id !== modelId)
    return true
  },
  'library:clean-partials': async () => downloadQueue.cleanPartials(modelsDir()),
  /**
   * Import GGUF files the user already has on disk.
   *
   * Three things matter here, all of them because these files are enormous:
   *   - A file already inside the models folder needs no work at all; the old code copied it
   *     onto itself.
   *   - On the same volume a hard link is instantaneous and costs no extra disk, where a copy
   *     of a 20 GB model costs 20 GB and several minutes of an apparently frozen window.
   *   - A name collision must not overwrite an existing model.
   *
   * Failures are collected per file rather than aborting the batch, so one unreadable file does
   * not silently discard the rest of the selection.
   */
  'library:import': async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'GGUF models', extensions: ['gguf'] }]
    })
    if (result.canceled) return { imported: [], skipped: [], failed: [], linked: 0 }

    const dest = modelsDir()
    await fsp.mkdir(dest, { recursive: true })

    const imported: string[] = []
    const skipped: string[] = []
    const failed: { file: string; error: string }[] = []
    let linked = 0

    for (const src of result.filePaths) {
      const name = path.basename(src)
      try {
        // Already in the library folder: nothing to do but rescan.
        if (path.resolve(path.dirname(src)).toLowerCase() === path.resolve(dest).toLowerCase()) {
          skipped.push(name)
          continue
        }

        const target = uniquePath(dest, path.parse(name).name, 'gguf', (candidate: string) => fs.existsSync(candidate))
        emit('library:import-progress', { file: name, phase: 'copying' })

        try {
          // Same volume: instantaneous, and the bytes are not duplicated.
          await fsp.link(src, target)
          linked++
        } catch {
          await fsp.copyFile(src, target)
        }
        imported.push(target)
      } catch (err) {
        failed.push({ file: name, error: err instanceof Error ? err.message : String(err) })
      }
    }

    library = await scanLibrary(dest)
    emit('library:import-progress', { file: '', phase: 'done' })
    return { imported, skipped, failed, linked }
  },

  // ---- auto-fit
  'autofit:plan': async (
    modelId: string,
    overrides?: Record<string, unknown>
  ): Promise<FitResult | { error: string }> => {
    const model = library.find((m) => m.id === modelId)
    if (!model) return { error: 'Model not found' }
    if (!model.arch) return { error: model.error ?? 'Model metadata could not be parsed' }

    const hw = await getHardware()
    hardware = await refreshFreeVram(hw)

    const s = loadSettings()
    return planFit(model.arch, hardware, {
      ...DEFAULT_CONSTRAINTS,
      minKvType: s.autoFit.minKvType,
      preferredKvType: s.autoFit.preferredKvType,
      targetContext: s.autoFit.targetContext,
      idealContext: s.autoFit.idealContext,
      headroomBytes: s.autoFit.headroomMb * 1024 * 1024,
      companionBytes: companionSize(model),
      overrides: (overrides ?? {}) as never
    })
  },

  // ---- model runtime
  'model:load': (modelId: string, plan?: FitPlan) => loadModelById(modelId, plan),
  'model:unload': async () => {
    await llama.unload()
    emit('model:status', null)
    return true
  },
  'model:status': () => {
    const l = llama.loaded
    return l
      ? {
          model: l.model.filename,
          modelId: l.model.id,
          port: l.port,
          plan: l.plan,
          startedAt: l.startedAt,
          caps: l.model.caps
        }
      : null
  },

  // ---- relocation
  'relocation:check': () => checkRelocation(),
  'relocation:keep': (from: string) => {
    keepInPlace(from)
    return true
  },
  'relocation:move': async (proposal: Parameters<typeof performMove>[0]) => {
    let cancelled = false
    const off = (): void => {
      cancelled = true
    }
    relocationCancel = off
    const result = await performMove(
      proposal,
      (p) => emit('relocation:progress', p),
      () => cancelled
    )
    relocationCancel = null
    return result
  },
  'relocation:cancel': () => {
    relocationCancel?.()
    return true
  },

  // ---- downloads
  'hf:search': async (query: string) => searchModels(query, getHfToken()),
  'hf:files': async (repo: string) => {
    const files = await listFiles(repo, getHfToken())
    const hw = await getHardware()
    const s = loadSettings()
    return { files, recommendation: recommendQuant(files, hw, s.autoFit.targetContext) }
  },
  'hf:download': async (repo: string, filename: string) => {
    const files = await listFiles(repo, getHfToken())
    const file = files.find((f) => f.filename === filename)
    if (!file) throw new Error(`${filename} not found in ${repo}`)

    const dir = path.join(modelsDir(), repo.replace('/', '__'))
    downloadQueue.setToken(getHfToken())
    const queued = [
      downloadQueue.enqueue({
        repo,
        filename: path.basename(file.filename),
        url: file.url,
        dest: path.join(dir, path.basename(file.filename)),
        bytesTotal: file.bytes,
        sha256: file.sha256
      })
    ]

    // Multimodal models are useless without their projector, so fetch it automatically.
    const mmproj = findMmprojFor(files)
    if (mmproj) {
      queued.push(
        downloadQueue.enqueue({
          repo,
          filename: path.basename(mmproj.filename),
          url: mmproj.url,
          dest: path.join(dir, path.basename(mmproj.filename)),
          bytesTotal: mmproj.bytes,
          sha256: mmproj.sha256
        })
      )
    }
    return queued
  },
  'downloads:list': () => downloadQueue.list(),
  'downloads:pause': (id: string) => {
    downloadQueue.pause(id)
    return true
  },
  'downloads:resume': (id: string) => {
    downloadQueue.resume(id)
    return true
  },
  'downloads:cancel': (id: string) => downloadQueue.cancel(id),
  'downloads:remove': (id: string) => {
    downloadQueue.remove(id)
    return true
  },

  // ---- chats
  'chat:list': (kind?: 'chat' | 'agent') => chats.listChats(kind),
  'chat:create': (opts: Parameters<typeof chats.createChat>[0]) => chats.createChat(opts ?? {}),
  'chat:load': (id: string) => chats.loadSession(id),
  'chat:rename': (id: string, title: string) => {
    chats.renameChat(id, title)
    return true
  },
  'chat:delete': async (id: string) => {
    chats.deleteChat(id)
    // Messages, tasks and attachments cascade out of the database; a session's file snapshots
    // live on disk and were left behind, holding a full copy of everything the agent had edited.
    await discardCheckpoints(id)
    return true
  },
  'chat:search': (query: string) => chats.searchChats(query),
  /**
   * Export a conversation to a file the user chooses.
   *
   * A save dialog, not a folder picker: the user gets to see and change the filename, and the
   * shell warns about overwriting. The written path is returned so the UI can say where it went
   * rather than appearing to do nothing.
   */
  'chat:export': async (id: string, format: 'md' | 'json') => {
    const title = get<{ title: string }>('SELECT title FROM chats WHERE id = ?', id)?.title ?? ''
    const result = await dialog.showSaveDialog({
      title: 'Export conversation',
      defaultPath: `${exportFilename(title, `chat-${id.slice(0, 8)}`)}.${format}`,
      filters: [format === 'md' ? { name: 'Markdown', extensions: ['md'] } : { name: 'JSON', extensions: ['json'] }]
    })
    if (result.canceled || !result.filePath) return null
    return offerReveal(writeExport(id, format, result.filePath))
  },
  'chat:export-text': (id: string, format: 'md' | 'json') => exportChat(id, format),

  /** Plain chat turn (no tools), streaming deltas to the UI. */
  'chat:send': async (
    chatId: string,
    text: string,
    attachments?: string[],
    collectionId?: string,
    reasoning?: ReasoningChoice
  ) => {
    const loaded = llama.loaded
    if (!loaded) throw new Error('No model is loaded')

    // A choice the loaded model cannot express is no choice — see sendableChoice. Narrowed here
    // rather than trusted, because the API server and a remote browser send this field too.
    const effort = sendableChoice(loaded.model.caps.reasoning, reasoning ?? null)

    const history = chats.loadMessages(chatId)
    const messages = history.map((m) => ({
      role: m.role === 'tool' ? ('user' as const) : (m.role as 'user' | 'assistant' | 'system'),
      content: m.content
    }))

    let userContent: string | Awaited<ReturnType<typeof buildContent>>['parts'] = text
    const notes: string[] = []

    if (attachments?.length) {
      const built = await buildContent(text, attachments, loaded.model.caps, await videoContext())
      userContent = built.parts
      notes.push(...built.notes)
    }

    // RAG: prepend retrieved context when a collection is attached.
    if (collectionId) {
      const hw = await getHardware()
      const hits = await rag.retrieve(text, hw.backend, { collectionId, chatId }, 6)
      const context = rag.formatContext(hits)
      if (context) messages.push({ role: 'system', content: context })
    }

    /*
     * The attachments have to survive in the transcript.
     *
     * Only the text was stored, so a conversation reopened later showed the question with no
     * sign that a file had been sent with it — and the export had the same hole. The names go
     * into the stored content, and the files themselves are recorded against the message.
     */
    const attachmentSummary = attachments?.length
      ? `\n\n[Attached: ${attachments.map((f) => path.basename(f)).join(', ')}]`
      : ''

    const userMsg = {
      id: `${Date.now().toString(36)}-u`,
      role: 'user' as const,
      content: `${text}${attachmentSummary}`,
      createdAt: Date.now()
    }
    chats.appendMessage(chatId, userMsg)
    for (const file of attachments ?? []) {
      try {
        recordAttachment(userMsg.id, { path: file, kind: classifyAttachment(file) })
      } catch (err) {
        // A missing attachment row must never cost the user their message — but it should not
        // vanish without trace either, or the only symptom is an export that quietly lost a file.
        logger.warn('chat', `could not record attachment ${file}`, err)
      }
    }
    chats.autoTitle(chatId)
    emit('chat:message', { chatId, message: userMsg })
    if (notes.length) emit('chat:notes', notes)

    const started = Date.now()
    let answer = ''
    let thinking = ''

    /*
     * A chat turn is interruptible.
     *
     * Only the agent could be stopped before, which is backwards for the surface where a wrong
     * turn is cheapest to abandon — a model that has misread the question and is settling in for
     * two thousand tokens about it should not have to be waited out. The controller is held
     * against the conversation so `chat:stop` can find it, and cleared in the finally below
     * whether the turn ended, failed or was cut short.
     */
    inFlightChats.get(chatId)?.abort()
    const abort = new AbortController()
    inFlightChats.set(chatId, abort)

    const request: CompletionOptions = {
      messages: [...messages, { role: 'user', content: userContent }],
      signal: abort.signal,
      // Dropped silently if the model never advertised the level — see reasoningRequestFields.
      ...reasoningRequestFields(loaded.model.caps.reasoning, effort)
    }

    /*
     * Ultra replaces the single request with several, then a pass that reads them together.
     *
     * The samples stream into their own boxes rather than the answer, so the transcript never
     * shows a draft in the place the real reply will appear. Only the synthesis pass writes to
     * chat:delta, which means everything downstream — the store, the caret, the markdown
     * renderer — sees exactly what it sees for an ordinary turn.
     */
    let stopped = false
    try {
      if (isUltra(effort)) {
        const cfg = loadSettings().ultra
        const { synthesis } = await ultraSamples(request, cfg, {
          onSampleStart: (index, total) => emit('chat:ultra-sample-start', { chatId, index, total }),
          onSampleDelta: (index, t) => emit('chat:ultra-sample-delta', { chatId, index, text: t }),
          onSampleReasoning: (index, t) => emit('chat:ultra-sample-reasoning', { chatId, index, text: t }),
          onSample: (sample) => emit('chat:ultra-sample', { chatId, sample }),
          onSynthesisStart: () => emit('chat:ultra-synthesis', { chatId })
        })
        request.messages = synthesis
      }

      for await (const ev of llama.streamEvents(request)) {
        if (ev.type === 'text') {
          answer += ev.text
          emit('chat:delta', { chatId, text: ev.text })
        }
        if (ev.type === 'reasoning') {
          thinking += ev.text
          emit('chat:reasoning', { chatId, text: ev.text })
        }
        if (ev.type === 'prompt_progress') {
          emit('chat:prompt-progress', {
            chatId,
            percent: ev.percent,
            processed: ev.processed,
            total: ev.total,
            cached: ev.cached
          })
        }
        if (ev.type === 'context') {
          emit('chat:context', { chatId, used: ev.used, max: ev.max })
          // The dashboard's meter reads the same number, so it is no longer a separate estimate
          // that can disagree with what the conversation is showing.
          setContextUsed(ev.used)
        }
      }
    } catch (err) {
      /*
       * A stopped turn keeps what it wrote.
       *
       * The user asked for it to stop, not for it to be undone — half an answer is usually still
       * worth reading, and discarding it would also lose the question that prompted it from the
       * visible transcript. Anything that is not the abort is a real failure and still throws.
       */
      if (!abort.signal.aborted) throw err
      stopped = true
    } finally {
      /*
       * Only clear the entry if it is still ours.
       *
       * A second turn on the same conversation replaces the controller before this one unwinds —
       * the API server and a remote browser can both send into a chat the desktop is already
       * using. Deleting unconditionally removed the *new* turn's controller, leaving it running
       * with no way to stop it.
       */
      if (inFlightChats.get(chatId) === abort) inFlightChats.delete(chatId)
    }

    const assistantMsg = {
      id: `${Date.now().toString(36)}-a`,
      role: 'assistant' as const,
      content: stopped ? `${answer}${answer.trim() ? '\n\n' : ''}_[stopped]_` : answer,
      reasoning: thinking || undefined,
      createdAt: Date.now()
    }
    chats.appendMessage(chatId, assistantMsg)
    emit('chat:message', { chatId, message: assistantMsg })

    const t = llama.timings
    if (t) {
      recordGeneration(loaded.model.filename, t.completionTokens, (Date.now() - started) / 1000)
      /*
       * The server's own count, not a guess from character lengths.
       *
       * This used to divide the total length of every message by four, which is a rule of thumb
       * for English prose and wrong for everything else — code, punctuation-heavy text and any
       * non-Latin script tokenise nothing like that. It also ignored the system prompt, the chat
       * template's own markup and any attachments, all of which occupy the window. The stream
       * reports the real prompt total, so the meter now agrees with what the model actually sees.
       */
      setContextUsed(t.promptTokens + t.completionTokens)
      chats.setContextUsed(chatId, t.promptTokens + t.completionTokens)
    }
    return assistantMsg
  },

  // ---- agent
  'agent:tools': () => {
    const a = getAgent()
    syncAgentOptions()
    return a.listTools()
  },
  'agent:run': async (sessionId: string, input: string, reasoning?: ReasoningChoice, attachments?: string[]) => {
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)

    /*
     * Text goes into the prompt, media goes to the model as content parts.
     *
     * Only the first user message of the turn carries parts; everything the loop appends after
     * it — assistant turns, tool results — stays plain text, which is the arrangement llama.cpp
     * expects and the reason the loop can feed its history back unchanged each iteration.
     */
    const turn = attachments?.length
      ? await attachmentTurn(input, attachments)
      : { text: input, media: [] as ContentPart[] }

    /*
     * What the transcript shows, as distinct from what the model is sent.
     *
     * The prompt carries inlined attachment text and, under Ultra, a chosen plan — none of which
     * the user typed. Recording it verbatim put the whole preamble inside their own message, so
     * a conversation opened with "Hello" appeared to have been sent with a numbered plan
     * attached. Attachments are named the way chat names them.
     */
    const displayText = attachments?.length
      ? `${input}\n\n[Attached: ${attachments.map((f) => path.basename(f)).join(', ')}]`
      : input

    // Same narrowing as the chat path: the stored level may belong to a model that is no longer
    // the one loaded.
    const effort = sendableChoice(llama.loaded?.model.caps.reasoning, reasoning ?? null)

    const a = getAgent()
    syncAgentOptions()
    a.updateOptions({ cwd: session.cwd, reasoningChoice: effort })

    /*
     * Show the user's message before doing anything slow with it.
     *
     * The loop is what normally announces the turn, and under Ultra the loop does not start
     * until several planning samples and a synthesis pass have finished — minutes during which
     * the transcript held no user message at all, so the empty state rendered and the sample box
     * sat alone at the bottom of an empty pane. The id is minted here and handed to the loop, so
     * what it stores is this message rather than a second one.
     */
    const userMessageId = `${Date.now().toString(36)}-u`
    emit('agent:message', {
      sessionId,
      message: { id: userMessageId, role: 'user', content: displayText, createdAt: Date.now() }
    })

    const before = session.messages.length
    activeAgentSessionId = sessionId
    const turnAbort = new AbortController()
    agentTurnAbort = turnAbort
    a.hydrate(session)
    // A prompt left unanswered by an earlier turn (window reloaded, remote tab closed) would
    // otherwise sit in the map forever and never be answerable again.
    drainPendingPermissions()
    try {
      /*
       * Ultra plans before it acts.
       *
       * Several passes investigate the task with only read tools available, so they can be run
       * over and over without touching anything. The winner is then handed to one ordinary run
       * that does the work once, with real tool results and the usual approval prompts — so the
       * thing that executes has seen the truth, rather than replaying calls that were planned
       * against results which never happened.
       */
      let prompt = turn.text
      let chosenPlan: string | undefined
      if (isUltra(effort)) {
        const cfg = loadSettings().ultra
        const count = Math.max(1, Math.min(8, Math.round(cfg.samples)))
        const temps = sampleTemperatures(count)
        const plans: string[] = []

        for (let i = 0; i < count; i++) {
          // Asked to stop, stop — rather than starting the next several minutes of planning.
          if (turnAbort.signal.aborted) break
          emit('agent:ultra-sample-start', { sessionId, index: i, total: count })
          const startedAt = Date.now()
          const plan = await a.planOnce(session, turn.text, turn.media, temps[i], {
            onDelta: (t) => emit('agent:ultra-sample-delta', { sessionId, index: i, text: t }),
            onReasoning: (t) => emit('agent:ultra-sample-reasoning', { sessionId, index: i, text: t })
          })
          plans.push(plan)
          emit('agent:ultra-sample', {
            sessionId,
            sample: {
              index: i,
              answer: plan,
              reasoning: '',
              continuations: 0,
              temperature: temps[i],
              ms: Date.now() - startedAt
            }
          })
        }

        emit('agent:ultra-synthesis', { sessionId })
        const chosen = turnAbort.signal.aborted
          ? ''
          : await llama.complete({
              messages: planSynthesisMessages(turn.text, plans),
              signal: turnAbort.signal
            })
        // A synthesis that comes back empty must not silently strip the user's own request.
        prompt = chosen.trim() ? `${turn.text}\n\n${planPreamble(chosen)}` : turn.text
        // Shown in a box of its own. It shapes everything that follows, so it should be
        // readable — just not by being pasted into the user's message.
        if (chosen.trim()) {
          chosenPlan = chosen.trim()
          emit('agent:ultra-plan', { sessionId, plan: chosenPlan })
        }
      }

      if (!turnAbort.signal.aborted) {
        await a.run(session, prompt, turn.media, displayText, { userMessageId, plan: chosenPlan })
      }
    } finally {
      drainPendingPermissions()
      // Persist on the way out either way. A turn that ends in an error still did real work —
      // the user's message, and any tool exchanges that already completed — and throwing that
      // away silently is worse than showing a transcript that stops mid-turn. Approvals the
      // user granted during the turn are worth keeping for the same reason.
      /*
       * A turn stopped during planning never reached the loop, so nothing recorded the message
       * that started it — it would have been on screen until the next reload and then gone.
       */
      if (session.messages.length === before) {
        session.messages.push({
          id: userMessageId,
          role: 'user',
          content: displayText,
          createdAt: Date.now()
        })
      }
      for (const m of session.messages.slice(before)) chats.appendMessage(sessionId, m)
      persistPermissionRules()
      if (agentTurnAbort === turnAbort) agentTurnAbort = null
    }

    chats.autoTitle(sessionId)
    return session
  },
  /** Cut a chat turn short. Whatever has been generated so far is kept. */
  'chat:stop': (chatId: string) => {
    inFlightChats.get(chatId)?.abort()
    return true
  },
  'agent:stop': () => {
    agentTurnAbort?.abort()
    agent?.stop()
    // Aborting the run does not settle a prompt the agent is already awaiting.
    drainPendingPermissions()
    return true
  },
  /**
   * Compact the session the user is looking at.
   *
   * This took no argument and compacted whatever happened to be in the agent's memory. Nothing
   * hydrates it outside of a turn, so clicking this after opening the app — before sending
   * anything — found an empty history, returned early, and reported success for work it had not
   * done. When it did find a history, it was whichever session last ran, not necessarily this one.
   */
  'agent:compact': async (sessionId?: string) => {
    const id = sessionId || activeAgentSessionId
    const session = id ? chats.loadSession(id) : null
    if (!session) {
      return { ok: false, message: 'Open a conversation before compacting it.' }
    }

    const a = getAgent()
    activeAgentSessionId = id
    a.hydrate(session)
    const report = await a.compactNow(session)

    /*
     * Re-measure, rather than leaving the old figure on screen.
     *
     * Compaction issues no request for the conversation itself, so nothing would otherwise
     * report its new size — the reading stayed at the pre-compaction number and the whole thing
     * looked as though it had achieved nothing. The estimate is the same one compaction sized
     * its own work with; the next real turn replaces it with the server's exact count.
     */
    if (report.ok && typeof report.afterTokens === 'number') {
      const max = llama.loaded?.plan.contextLength ?? 0
      if (max > 0) emit('agent:context', { sessionId: id, used: report.afterTokens, max })
      chats.setContextUsed(id, report.afterTokens)
      setContextUsed(report.afterTokens)
    }
    return report
  },
  'agent:set-cwd': async (sessionId?: string) => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
    if (result.canceled || !result.filePaths[0]) return null
    const dir = result.filePaths[0]
    getAgent().updateOptions({ cwd: dir })
    if (sessionId) chats.setChatCwd(sessionId, dir)
    return dir
  },
  'agent:answer': (id: string, answer: string) => {
    const resolve = pendingQuestions.get(id)
    if (resolve) {
      pendingQuestions.delete(id)
      resolve(String(answer ?? '').trim() || '(the user gave an empty answer)')
    }
    return true
  },
  'agent:permission-response': (id: string, decision: PermissionDecision) => {
    const resolve = pendingPermissions.get(id)
    if (resolve) {
      pendingPermissions.delete(id)
      resolve(decision)
    }
    return true
  },
  'agent:checkpoints': (sessionId: string) => listCheckpoints(sessionId),
  'agent:rewind': async (sessionId: string, checkpointId: string, messageId?: string) => {
    const result = await rewindTo(sessionId, checkpointId)
    if (messageId) chats.truncateFrom(sessionId, messageId)
    /*
     * A summary of work that has just been undone is worse than no summary.
     *
     * Rewinding removes messages, and the stored summary describes a stretch of them. Keeping it
     * would have the model told, every turn, about files it wrote and commands it ran that the
     * rewind has taken back. Dropping it costs only the room the summary was saving.
     */
    chats.clearSummary(sessionId)
    // The agent's rolling history still holds the rewound turns; the next run re-hydrates it.
    agent?.resetHistory()
    return result
  },
  'agent:memory': () => allMemory(),
  'agent:memory-add': (text: string) => addMemory(text, 'user'),
  'agent:memory-update': (id: string, text: string) => {
    updateMemory(id, text)
    return true
  },
  'agent:memory-delete': (id: string) => {
    deleteMemory(id)
    return true
  },
  'agent:permission-rules': () => getAgent().exportPermissionRules(),
  'agent:clear-permission-rules': () => {
    run('DELETE FROM permission_rules')
    getAgent().loadPermissionRules([])
    return true
  },

  // ---- MCP
  'mcp:list': () => mcpManager.status(),
  'mcp:add': (config: Parameters<typeof mcpManager.addServer>[0]) => mcpManager.addServer(config),
  'mcp:remove': (id: string) => {
    mcpManager.removeServer(id)
    return true
  },
  'mcp:set-enabled': (id: string, enabled: boolean) => {
    mcpManager.setEnabled(id, enabled)
    return true
  },
  'mcp:connect': () => mcpManager.connectAll(),

  // ---- attachments

  /**
   * Describe files the user is trying to attach.
   *
   * Classification and the capability check happen here rather than in the renderer, so the
   * warning shown next to a chip and the decision `buildContent` makes later cannot disagree.
   */
  'attachments:describe': async (paths: string[]) => {
    const caps = llama.loaded?.model.caps ?? null
    const out: AttachmentInfo[] = []

    for (const file of paths ?? []) {
      const kind = classifyAttachment(file)
      let bytes = -1
      try {
        bytes = (await fsp.stat(file)).size
      } catch {
        out.push({ path: file, name: path.basename(file), kind, bytes: -1, warning: 'File could not be read.' })
        continue
      }

      let warning: string | null = null
      if (!caps) warning = 'No model is loaded yet.'
      else if (kind === 'image' && !caps.vision)
        warning = 'This model has no vision projector, so the image will be skipped.'
      else if (kind === 'audio' && !caps.audio) warning = 'This model does not accept audio, so it will be skipped.'
      else if (kind === 'video' && !caps.videoPossible)
        warning = 'This model cannot read video without a vision projector.'

      out.push({ path: file, name: path.basename(file), kind, bytes, warning })
    }
    return out
  },

  /** Open a picker filtered to what the loaded model can actually take. */
  'attachments:pick': async () => {
    const caps = llama.loaded?.model.caps
    const filters = [
      { name: 'Everything supported', extensions: [...IMAGE_EXT, ...VIDEO_EXT, ...AUDIO_EXT, ...TEXT_EXT] },
      { name: 'Images', extensions: IMAGE_EXT },
      { name: 'Video', extensions: VIDEO_EXT },
      ...(caps?.audio ? [{ name: 'Audio', extensions: AUDIO_EXT }] : []),
      { name: 'Text and code', extensions: TEXT_EXT },
      { name: 'All files', extensions: ['*'] }
    ]
    const result = await dialog.showOpenDialog({ properties: ['openFile', 'multiSelections'], filters })
    if (result.canceled) return []
    return handlers['attachments:describe'](result.filePaths as never)
  },

  /**
   * Accept file bytes and stage them on disk.
   *
   * The desktop shell sends a path, which costs nothing. A remote browser has no path to send,
   * so it posts the contents and gets back a staged file the rest of the pipeline can treat
   * identically.
   */
  'attachments:prune': () => pruneStagedUploads(),

  'attachments:receive': async (name: string, base64: string) => {
    const safe = exportFilename(String(name ?? 'file'), 'attachment')
    const ext = path.extname(String(name ?? '')) || ''
    const dir = path.join(TOOL_OUTPUT_DIR, 'attachments')
    await fsp.mkdir(dir, { recursive: true })

    const dest = uniquePath(
      dir,
      `${Date.now().toString(36)}-${safe.replace(/\.[^.]*$/, '')}`,
      ext.replace(/^\./, '') || 'bin',
      (p) => fs.existsSync(p)
    )
    await fsp.writeFile(dest, Buffer.from(String(base64 ?? ''), 'base64'))
    const [info] = (await handlers['attachments:describe']([dest] as never)) as AttachmentInfo[]
    return info
  },

  // ---- RAG
  'rag:collections': () => rag.listCollections(),
  'rag:create-collection': (name: string) => rag.createCollection(name),
  'rag:delete-collection': (id: string) => {
    rag.deleteCollection(id)
    return true
  },
  'rag:documents': (filter: { collectionId?: string; chatId?: string }) => rag.listDocuments(filter ?? {}),
  'rag:delete-document': (id: string) => {
    rag.deleteDocument(id)
    return true
  },
  'rag:ingest': async (target: { collectionId?: string; chatId?: string }) => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Documents', extensions: ['txt', 'md', 'pdf', 'json', 'csv', 'ts', 'js', 'py', 'rs', 'go', 'java'] }
      ]
    })
    if (result.canceled) return []

    const hw = await getHardware()
    const out: unknown[] = []
    for (const file of result.filePaths) {
      emit('rag:progress', { file: path.basename(file), phase: 'ingesting' })
      try {
        out.push(await rag.ingestDocument(file, hw.backend, target ?? {}))
      } catch (err) {
        emit('rag:progress', {
          file: path.basename(file),
          phase: 'failed',
          error: err instanceof Error ? err.message : String(err)
        })
      }
    }
    emit('rag:progress', { phase: 'done' })
    return out
  },
  'rag:retrieve': async (query: string, scope: { collectionId?: string; chatId?: string }) => {
    const hw = await getHardware()
    return rag.retrieve(query, hw.backend, scope ?? {}, 8)
  },

  // ---- API server
  'server:start': async () => {
    const s = loadSettings()
    const port = await apiServer.start({
      port: s.server.port,
      apiKey: getApiKey(),
      jitLoad: s.server.jitLoad,
      exposeToNetwork: s.remote.enabled,
      loadModel: async (id) => {
        await loadModelById(id)
      },
      listModels: () => library.map((m) => ({ id: m.id, filename: m.filename, bytes: m.bytes }))
    })
    patchSettings({ server: { ...s.server, enabled: true } })
    emit('server:status', { running: true, port })
    return { running: true, port }
  },
  'server:stop': async () => {
    await apiServer.stop()
    const s = loadSettings()
    patchSettings({ server: { ...s.server, enabled: false } })
    emit('server:status', { running: false, port: null })
    return { running: false }
  },
  /*
   * Everything the Server view needs to describe the server honestly.
   *
   * `queue` used to be hardcoded to zero while requests were genuinely backed up. `hasApiKey`
   * is here because the view had no way to learn that a key already existed — it showed "none
   * set" on every launch, so the obvious move was to generate one, silently invalidating the key
   * every configured client was already using. It reports only whether a key exists, never the
   * key, because this handler answers remote callers too.
   */
  'server:status': () => ({
    running: apiServer.running,
    port: apiServer.port,
    queue: requestQueue.depth,
    hasApiKey: !!getApiKey(),
    networkBlocked: apiServer.exposureRefused
  }),
  'server:generate-key': () => generateApiKey(),
  'server:clear-key': () => {
    setApiKey(null)
    return true
  },
  'server:requests': (limit?: number) => requestLog(limit ?? 100),

  // ---- remote access
  'remote:status': () => ({
    tunnel: tunnel.current,
    web: remoteWeb.running,
    hasPassword: hasPassword(),
    settings: loadSettings().remote
  }),
  'remote:enable': async (mode: 'tunnel' | 'own-domain', domain?: string, email?: string) => {
    if (!hasPassword()) throw new Error('Set a password before enabling remote access.')

    const s = loadSettings()
    const webPort = s.server.port + 1

    if (mode === 'own-domain') {
      if (!domain) throw new Error('A domain is required for the self-hosted path.')
      const check = await verifyDomainPointsHere(domain)
      if (!check.ok) throw new Error(check.message)
      const cert = await certificates.obtain(domain, email ?? `admin@${domain}`)
      certificates.scheduleRenewal(domain, email ?? `admin@${domain}`)
      startDdnsUpdates()
      await remoteWeb.start({
        port: 443,
        invoke: invokeBridge,
        mode: 'own-domain',
        tls: { certPath: cert.certPath, keyPath: cert.keyPath }
      })
      patchSettings({ remote: { enabled: true, mode, domain } })
      emit('remote:status', { url: `https://${domain}`, mode })
      return { url: `https://${domain}` }
    }

    await remoteWeb.start({ port: webPort, invoke: invokeBridge, mode: 'tunnel' })
    const state = await tunnel.start(webPort)
    patchSettings({ remote: { enabled: true, mode: 'tunnel', domain: null } })
    emit('remote:status', { url: state.url, mode: 'tunnel' })
    return { url: state.url }
  },
  'remote:disable': async () => {
    await tunnel.stop()
    await remoteWeb.stop()
    certificates.clearRenewal()
    stopDdnsUpdates()
    const s = loadSettings()
    patchSettings({ remote: { ...s.remote, enabled: false } })
    emit('remote:status', { url: null })
    return true
  },
  'remote:ddns-update': () => updateFreedns(),
  'remote:check-domain': (domain: string) => verifyDomainPointsHere(domain),
  'remote:public-ip': () => publicIp(),
  'remote:cert-info': (domain: string) => certificates.existing(domain),

  // ---- stats & diagnostics
  'stats:live': async () => liveStats(hardware),
  'stats:history': () => historicalStats(),
  'stats:clear': () => {
    clearStats()
    return true
  },
  'diagnostics:build': async () => buildDiagnostics(),
  'diagnostics:reveal': async () => {
    const { path: p } = await buildDiagnostics()
    shell.showItemInFolder(p)
    return p
  },

  /**
   * Show a file this process wrote in the file manager.
   *
   * The allow-list above was already being maintained for this, but the handler it was written
   * for did not exist — so the "Show" button on an export toast called an unknown channel and
   * threw instead of opening anything. Only paths `offerReveal` has handed out are accepted, so
   * the argument is a token rather than a path the caller gets to choose.
   */
  'shell:reveal': (file: string) => {
    const resolved = path.resolve(String(file ?? ''))
    if (!revealable.has(resolved)) throw new Error('That file was not offered for reveal.')
    shell.showItemInFolder(resolved)
    return true
  },

  // ---- updates
  'update:check': () => checkForUpdate(),
  'update:apply': (url: string) => applyUpdate(url, (p) => emit('update:progress', p)),

  // ---- app
  'app:version': () => app.getVersion(),
  'app:paths': () => ({ exeDir: exeDir(), modelsDir: modelsDir() }),
  'app:quit': () => {
    app.quit()
    return true
  },
  'app:minimise-to-tray': () => {
    BrowserWindow.getAllWindows().forEach((w) => w.hide())
    return true
  }
}

let relocationCancel: (() => void) | null = null

/** Invoke a bridge handler by name — used by both IPC and the remote web server. */
export async function invokeBridge(channel: string, args: unknown[]): Promise<unknown> {
  const handler = handlers[channel]
  if (!handler) throw new Error(`Unknown channel: ${channel}`)
  return (handler as (...a: unknown[]) => unknown)(...args)
}

export async function shutdown(): Promise<void> {
  persistPermissionRules()
  killAllJobs()
  mcpManager.closeAll()
  await Promise.allSettled([
    llama.unload(),
    apiServer.stop(),
    remoteWeb.stop(),
    tunnel.stop(),
    closeBrowser(),
    rag.embeddings.stop()
  ])
  certificates.clearRenewal()
  stopDdnsUpdates()
}

export function setLibrary(models: ModelRecord[]): void {
  library = models
}

export function getLibrary(): ModelRecord[] {
  return library
}
