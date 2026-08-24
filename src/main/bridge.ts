/**
 * The bridge: one map of channel -> handler, used by BOTH the desktop IPC and the remote web
 * server. That is what makes "full parity" true by construction rather than by discipline —
 * there is no second, thinner implementation to drift.
 *
 * Remote gating happens in the web server (a deny-list of desktop-only channels), not here,
 * so a handler cannot forget to check.
 */

import { app, dialog, BrowserWindow } from 'electron'
import path from 'node:path'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import type {
  AppSettings,
  FitPlan,
  FitResult,
  HardwareSnapshot,
  ModelRecord,
  PermissionDecision,
  PermissionRequest
} from '@shared/types'
import { defaultModelsDir, exeDir } from './storage/paths'
import { loadSettings, patchSettings } from './storage/settings'
import { detectHardware, refreshFreeVram } from './hardware/gpu'
import { scanLibrary, libraryDiskUsage } from './models/library'
import { checkRelocation, keepInPlace, performMove } from './models/relocation'
import { DEFAULT_CONSTRAINTS, planFit, verifyPrediction } from './autofit/engine'
import { llama } from './runtime/llama'
import { missingBinaries, embeddingModelPath, vendorDiagnostics } from './runtime/binaries'
import { Agent } from './agent/loop'
import { killAllJobs } from './agent/tools/exec'
import { closeBrowser } from './agent/tools/browser'
import { listCheckpoints, rewindTo } from './agent/checkpoints'
import { mcpManager } from './agent/mcp'
import { addMemory, allMemory, deleteMemory, updateMemory } from './agent/memory'
import { searchModels, listFiles, recommendQuant, findMmprojFor } from './downloads/hf'
import { downloadQueue } from './downloads/queue'
import { apiServer } from './api/server'
import { tunnel } from './remote/tunnel'
import { remoteWeb } from './remote/web'
import { certificates, startDdnsUpdates, stopDdnsUpdates, updateFreedns, verifyDomainPointsHere, publicIp } from './remote/selfhost'
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
import { buildContent } from './chat/multimodal'
import { historicalStats, liveStats, requestLog, recordGeneration, clearStats, setContextUsed } from './stats'
import { buildDiagnostics, logger } from './log'
import { exportChat, writeExport, all, run } from './storage/db'
import { checkForUpdate, applyUpdate } from './update'

// ---------------------------------------------------------------- shared state

let hardware: HardwareSnapshot | null = null
let library: ModelRecord[] = []
let agent: Agent | null = null

const pendingPermissions = new Map<string, (d: PermissionDecision) => void>()

/**
 * The session the agent is currently working on.
 *
 * Every agent event carries this, so the renderer can attribute streamed text to the right
 * conversation even when that conversation's view is not mounted.
 */
let activeAgentSessionId = ''

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
        })
    })

    const sid = (): string => activeAgentSessionId
    agent.on('delta', (t: string) => emit('agent:delta', { sessionId: sid(), text: t }))
    agent.on('message', (m) => emit('agent:message', { sessionId: sid(), message: m }))
    agent.on('toolCall', (c) => emit('agent:tool-call', { sessionId: sid(), call: c }))
    agent.on('toolResult', (r) => emit('agent:tool-result', { sessionId: sid(), result: r }))
    agent.on('subToolCall', (c) => emit('agent:sub-tool-call', { sessionId: sid(), call: c }))
    agent.on('compacted', (info) => emit('agent:compacted', { sessionId: sid(), ...(info as object) }))
    agent.on('done', (reason) => emit('agent:done', { sessionId: sid(), reason }))
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
  run('INSERT OR REPLACE INTO model_meta (model_id, repo, favourite, tags, last_used_at) VALUES (?, ?, COALESCE((SELECT favourite FROM model_meta WHERE model_id = ?), 0), ?, ?)',
    model.id, model.repo, model.id, JSON.stringify(model.tags), Date.now())

  emit('model:status', { model: model.filename, port: loaded.port, plan: chosen })
  return { port: loaded.port, plan: chosen }
}

// ---------------------------------------------------------------- handlers

export const handlers: Record<string, (...args: never[]) => unknown> = {
  // ---- settings
  'settings:get': () => ({ ...loadSettings(), hfToken: getHfToken() ? '***set***' : null }),
  'settings:patch': (patch: Partial<AppSettings>) => patchSettings(patch),
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
  'runtime:vendor-diagnostics': async () => vendorDiagnostics((await getHardware()).backend),
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
  'library:import': async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'GGUF models', extensions: ['gguf'] }]
    })
    if (result.canceled) return []
    const dest = modelsDir()
    await fsp.mkdir(dest, { recursive: true })
    const imported: string[] = []
    for (const src of result.filePaths) {
      const target = path.join(dest, path.basename(src))
      await fsp.copyFile(src, target)
      imported.push(target)
    }
    library = await scanLibrary(dest)
    return imported
  },

  // ---- auto-fit
  'autofit:plan': async (modelId: string, overrides?: Record<string, unknown>): Promise<FitResult | { error: string }> => {
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
    return l ? { model: l.model.filename, modelId: l.model.id, port: l.port, plan: l.plan, startedAt: l.startedAt } : null
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
    const result = await performMove(proposal, (p) => emit('relocation:progress', p), () => cancelled)
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
        bytesTotal: file.bytes
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
          bytesTotal: mmproj.bytes
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
  'chat:delete': (id: string) => {
    chats.deleteChat(id)
    return true
  },
  'chat:search': (query: string) => chats.searchChats(query),
  'chat:export': async (id: string, format: 'md' | 'json') => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
    if (result.canceled || !result.filePaths[0]) return null
    return writeExport(id, format, result.filePaths[0])
  },
  'chat:export-text': (id: string, format: 'md' | 'json') => exportChat(id, format),

  /** Plain chat turn (no tools), streaming deltas to the UI. */
  'chat:send': async (chatId: string, text: string, attachments?: string[], collectionId?: string) => {
    const loaded = llama.loaded
    if (!loaded) throw new Error('No model is loaded')

    const history = chats.loadMessages(chatId)
    const messages = history.map((m) => ({
      role: m.role === 'tool' ? ('user' as const) : (m.role as 'user' | 'assistant' | 'system'),
      content: m.content
    }))

    let userContent: string | Awaited<ReturnType<typeof buildContent>>['parts'] = text
    const notes: string[] = []

    if (attachments?.length) {
      const built = await buildContent(text, attachments, loaded.model.caps)
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

    const userMsg = {
      id: `${Date.now().toString(36)}-u`,
      role: 'user' as const,
      content: text,
      createdAt: Date.now()
    }
    chats.appendMessage(chatId, userMsg)
    chats.autoTitle(chatId)
    emit('chat:message', { chatId, message: userMsg })
    if (notes.length) emit('chat:notes', notes)

    const started = Date.now()
    let answer = ''
    for await (const ev of llama.streamEvents({
      messages: [...messages, { role: 'user', content: userContent }]
    })) {
      if (ev.type === 'text') {
        answer += ev.text
        emit('chat:delta', { chatId, text: ev.text })
      }
    }

    const assistantMsg = {
      id: `${Date.now().toString(36)}-a`,
      role: 'assistant' as const,
      content: answer,
      createdAt: Date.now()
    }
    chats.appendMessage(chatId, assistantMsg)
    emit('chat:message', { chatId, message: assistantMsg })

    const t = llama.timings
    if (t) {
      recordGeneration(loaded.model.filename, t.completionTokens, (Date.now() - started) / 1000)
      setContextUsed(messages.reduce((a, m) => a + Math.ceil(m.content.length / 4), 0) + t.completionTokens)
    }
    return assistantMsg
  },

  // ---- agent
  'agent:tools': () => getAgent().listTools(),
  'agent:run': async (sessionId: string, input: string) => {
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)

    const s = loadSettings()
    const a = getAgent()
    a.updateOptions({
      cwd: session.cwd,
      planMode: s.agent.planMode,
      maxToolCallsPerTurn: s.agent.maxToolCallsPerTurn,
      commandTimeoutMs: s.agent.commandTimeoutMs,
      hardBlocksDisabled: s.agent.hardBlocksDisabled,
      compaction: s.agent.compaction,
      remoteToolsEnabled: s.agent.remoteToolsEnabled,
      hfToken: getHfToken()
    })

    const before = session.messages.length
    activeAgentSessionId = sessionId
    a.hydrate(session)
    await a.run(session, input)

    for (const m of session.messages.slice(before)) chats.appendMessage(sessionId, m)
    chats.autoTitle(sessionId)
    persistPermissionRules()
    return session
  },
  'agent:stop': () => {
    agent?.stop()
    return true
  },
  'agent:compact': async () => {
    await agent?.compactNow()
    return true
  },
  'agent:set-cwd': async (sessionId?: string) => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
    if (result.canceled || !result.filePaths[0]) return null
    const dir = result.filePaths[0]
    getAgent().updateOptions({ cwd: dir })
    if (sessionId) chats.setChatCwd(sessionId, dir)
    return dir
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
      filters: [{ name: 'Documents', extensions: ['txt', 'md', 'pdf', 'json', 'csv', 'ts', 'js', 'py', 'rs', 'go', 'java'] }]
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
  'server:status': () => ({ running: apiServer.running, port: apiServer.port, queue: 0 }),
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
        tls: { certPath: cert.certPath, keyPath: cert.keyPath }
      })
      patchSettings({ remote: { enabled: true, mode, domain } })
      emit('remote:status', { url: `https://${domain}`, mode })
      return { url: `https://${domain}` }
    }

    await remoteWeb.start({ port: webPort, invoke: invokeBridge })
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
    const { shell } = await import('electron')
    shell.showItemInFolder(p)
    return p
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
