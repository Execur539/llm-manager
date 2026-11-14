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
import os from 'node:os'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import type {
  AppSettings,
  Backend,
  FitConstraints,
  FitPlan,
  FitResult,
  HardwareSnapshot,
  KvType,
  ModelArchInfo,
  ModelRecord,
  AgentQuestion,
  PermissionDecision,
  PermissionRequest,
  AttachmentInfo,
  AgentMessage,
  AgentSessionState,
  ToolCall,
  ToolDefinition
} from '@shared/types'
import { defaultModelsDir, exeDir, TOOL_OUTPUT_DIR } from './storage/paths'
import { loadSettings, patchSettings } from './storage/settings'
import { detectHardware, mergeDetection, refreshFreeVram } from './hardware/gpu'
import { scanLibrary, libraryDiskUsage } from './models/library'
import { draftReserveBytes } from './models/draft-module'
import { resolveBatches } from './autofit/engine'
import { isInside, libraryRoots } from './models/roots'
import { checkRelocation, keepInPlace, performMove } from './models/relocation'
import { DEFAULT_CONSTRAINTS, gpuKey, usableDevices } from './autofit/engine'
import { describeLoad, extrasFromLoad, learnedExtras, planModel, recordSpeed, type ExtrasFeatures } from './autofit/planner'
import { llama, estimateTokens, type ChatMessage, type ContentPart, type CompletionOptions } from './runtime/llama'
import type { RereadReport } from './runtime/reuse-meter'
import {
  coverage,
  currentState,
  listStates,
  longestReusable,
  renderPrefix,
  requestText,
  prune,
  readMeta,
  removeState,
  restoreState,
  saveAsIs,
  saveState,
  type PartialMark,
  type SaveProgress,
  type StateMark
} from './runtime/state-cache'
import { stateCacheFolder } from './runtime/llama'
import crypto from 'node:crypto'
import { missingBinaries, embeddingModelPath, vendorDiagnostics, llamaServerPath } from './runtime/binaries'
import { cudaKvPairs, MATCHED_KV_PAIRS } from './runtime/fa-kernels'
import { deleteLogin, listLogins, saveLogin } from './agent/credentials'
import { Agent } from './agent/loop'
import { describeVideo, lookAtVideo, videoFacts } from './agent/tools/video'
import { makeResult, splitOutput, type ToolOutput } from './agent/tools/base'
import { rebuildHistory } from './agent/history'
import { scratchpad, scratchServer, idleMinutes, scratchpadChanged, onScratchpadChange, stopScratchpad } from './scratchpad'
import { CHAT_TOOLS, runChatTool } from './scratchpad/tools'
import { killAllJobs } from './agent/tools/exec'
import { closeBrowser } from './agent/tools/browser'
import { listCheckpoints, rewindTo, discardCheckpoints } from './agent/checkpoints'
import { mcpManager } from './agent/mcp'
import { addMemory, allMemory, deleteMemory, updateMemory } from './agent/memory'
import { searchModels, listFiles, groupVariants, recommendQuant, findMmprojFor, peekVariantArch } from './downloads/hf'
import { summarizer, type SummarizerHandle } from './runtime/summarizer'
import { chooseHelperPlacement, helperContext, helperVramNeed, keepBudget, keepRecent, type TranscriptItem } from './agent/compaction'
import { summariseTranscript, COMPACTION_FAILED } from './agent/summarise'
import { summariseInPlace } from './agent/summarise-in-place'
import { costOf, planQuick, quickView, standing, withPreface, type QuickPlan } from './chat/quick-compact'
import { RECALL_TOOLS, isRecallTool, recallRead, recallSearch } from './chat/recall'
import { setRecallSource, type RecallSource } from './agent/tools/recall'
import { samplingOptions } from '../shared/sampling'
import { persistSent, restoreSent, sentContentFor, sentTextFor, sentWithText } from './chat/media-cache'
import {
  adoptAttachments,
  deleteConversationFiles,
  keepPrepared,
  pruneOrphanAttachments,
  storeAttachment
} from './chat/attachment-store'
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
import { buildContent, type VideoContext, type PreparedMedia } from './chat/multimodal'
import { classifyAttachment, recordAttachment, toDataUrl, IMAGE_EXT, AUDIO_EXT, VIDEO_EXT, TEXT_EXT } from './chat/repo'
import { historicalStats, liveStats, requestLog, recordGeneration, clearStats, setContextUsed } from './stats'
import { buildDiagnostics, logger } from './log'
import { all, run, get } from './storage/db'
import { exportChat, writeExport } from './chat/export'
import * as protection from './protect/convert'
import { keyOf, osKeyStoreAvailable, readOpen, recordOf, statusOf, type Owner, type OwnerKind } from './protect/keys'
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
 * Refuse to rewrite the shared agent's history while a turn is actually using it.
 *
 * There is one `Agent` instance for every session (`getAgent()`), and its rolling history is
 * mutated in place by `resetHistory()` and `hydrate()`. Calling either while `agentTurnAbort` is
 * set does not just affect the session being edited — it pulls the history out from under
 * whichever turn is running, in any conversation, since it is the same object either way. Editing
 * or deleting a message in a conversation that is currently idle is still refused if a *different*
 * conversation is generating, because there is no way to tell the two apart at this level.
 */
function assertAgentIdle(): void {
  if (agentTurnAbort) {
    throw new Error('The model is busy with another turn right now — wait for it to finish first.')
  }
}

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

// What the scratchpad holds, for the window: every add, delete and sweep.
onScratchpadChange(() => emit('scratchpad:changed', scratchpad.totals()))

/*
 * A long wait before a reply is explained where it happens.
 *
 * The log has every request's count; this is for the one the person just sat through. Only the
 * agent view has somewhere to say it, and a sub-agent's reading is its parent's business.
 */
llama.on('reread', (r: RereadReport) => {
  if (!r.key?.startsWith('agent:') || r.key.startsWith('agent:sub-')) return
  emit('agent:notice', {
    sessionId: r.key.slice('agent:'.length),
    message: `Read ${r.read.toLocaleString()} of ${r.total.toLocaleString()} tokens before this reply: ${r.reason}.`
  })
})

/** Each reply's real decode speed, kept against the plan that produced it. */
llama.on('decodeSpeed', (d: { tokensPerSecond: number; tokens: number }) => {
  const loaded = llama.loaded
  if (!loaded) return
  try {
    recordSpeed(loaded.model.id, loaded.plan, d.tokensPerSecond, d.tokens)
  } catch (err) {
    logger.warn('autofit', 'could not keep a speed measurement', err)
  }
})

/**
 * The cards as they will be once the loaded model makes way.
 *
 * Planning another model, or the same one again, happens while the current one still holds its
 * VRAM. What its load cost was measured when it loaded, so that is given back to the free figure
 * rather than planning the next model into whatever happens to be left.
 */
function withLoadedReleased(hw: HardwareSnapshot): HardwareSnapshot {
  const loaded = llama.loaded
  if (!loaded) return hw
  const row = get<{ actual_vram: string | null }>('SELECT actual_vram FROM model_configs WHERE model_id = ?', loaded.model.id)
  try {
    const held = row?.actual_vram ? (JSON.parse(row.actual_vram) as { keys?: string[]; bytes?: number[] }) : null
    if (!held?.keys || !held.bytes) return hw
    return {
      ...hw,
      gpus: hw.gpus.map((g) => {
        const i = held.keys!.indexOf(gpuKey(g))
        return i >= 0 && g.freeIsMeasured ? { ...g, freeVram: Math.min(g.totalVram, g.freeVram + (held.bytes![i] ?? 0)) } : g
      })
    }
  } catch {
    return hw
  }
}

/** Stands in for the next message while a conversation is read ahead of it. */
const STAND_IN: ChatMessage = { role: 'user', content: '.' }

/** Whether the model is free to read ahead: loaded, allowed to, and nobody waiting on it. */
function canWarm(): boolean {
  return (
    !!llama.loaded &&
    loadSettings().runtime.warmConversations !== false &&
    !agentTurnAbort &&
    inFlightChats.size === 0 &&
    !llama.idleBusy
  )
}

/**
 * Read a conversation into the model now, so its next message only has to add to it.
 *
 * The request is the one the next turn will send, with a stand-in where the new message goes
 * and no reply asked for. The server checkpoints at the start of the last user message, so the
 * real message replaces the stand-in and everything before it is already read. Does nothing
 * when the server already holds the conversation. Returns whether anything was started.
 */
function warm(req: CompletionOptions): boolean {
  if (!req.messages.length || llama.holds(req)) return false
  void llama.runIdle(async (signal) => {
    /*
     * A saved state is a file read; whatever was said after it was saved is read on top of it,
     * so nothing after a save is lost, only read again. With nothing saved, the whole of it is read.
     */
    if (!(await restoreSaved(req, signal)) && !signal.aborted) await seedInstructions(req, signal, true)
    if (signal.aborted) return
    await readAhead(req, signal)
  })
  return true
}

/** The effort each conversation was last sent with; its state is saved and loaded under the same one. */
const lastEffort = new Map<string, ReasoningChoice | null>()

/** The request a conversation's next turn starts from, for a key such as chat:<id> or agent:<id>. */
async function conversationRequest(key: string): Promise<CompletionOptions | null> {
  const effort = effortFields(lastEffort.get(key))
  // A locked conversation cannot be read, so there is nothing to bring back or keep.
  if (stateSeal(key) === 'locked') return null
  if (key.startsWith('chat:')) {
    const id = key.slice('chat:'.length)
    const history = chats.loadMessages(id)
    if (!history.length) return null
    await restoreConversationMedia(id, history)
    const tools = chatToolsFor(id, history)
    return { messages: chatMessagesFor(id, history, loadSettings().reasoning.preserve), cacheKey: key, ...(tools ? { tools } : {}), ...effort }
  }
  if (key.startsWith('agent:')) {
    const session = chats.loadSession(key.slice('agent:'.length))
    if (!session?.messages.length) return null
    await restoreConversationMedia(session.id, session.messages)
    const a = getAgent()
    return { messages: a.buildHistory(session), tools: a.toolsFor(session), cacheKey: key, ...effort }
  }
  return null
}

/**
 * How a conversation's saved state is kept: under its key when it is protected, plain when it is
 * not, and not at all ('locked') while it is locked and nothing of it can be read or written.
 */
function stateSeal(key: string): Buffer | undefined | 'locked' {
  const m = /^(?:chat|agent):(.+)$/.exec(key)
  if (!m) return undefined
  const owner: Owner = { kind: 'chat', id: m[1] }
  if (!recordOf(owner)) return undefined
  return keyOf(owner) ?? 'locked'
}



function ownerOf(kind: OwnerKind, id: string): Owner {
  if ((kind !== 'chat' && kind !== 'collection') || typeof id !== 'string' || !id) throw new Error('Nothing to protect')
  return { kind, id }
}

/** Refuse to change a conversation that a reply is being written to. */
function refuseWhileBusy(owner: Owner): void {
  if (owner.kind !== 'chat') return
  if (inFlightChats.has(owner.id) || liveAgentSessions.has(owner.id)) {
    throw new Error('Wait for the reply to finish, or stop it, first.')
  }
}

function dropSavedStates(owner: Owner): void {
  if (owner.kind !== 'chat') return
  const dir = llama.stateStore?.dir ?? stateCacheFolder()
  for (const key of [`chat:${owner.id}`, `agent:${owner.id}`]) if (dir) removeState(dir, key)
}

function protectChanged(owner: Owner): void {
  emit('protect:changed', { kind: owner.kind, id: owner.id })
}

/**
 * Before a conversation is locked: nothing of it left readable in the app or the server. The
 * server's copy is erased (what was saved of it stays saved, encrypted); its scratchpad and the
 * frames it was shown go too.
 */
async function letGoOf(id: string): Promise<void> {
  const held = llama.heldKey
  if (held === `chat:${id}` || held === `agent:${id}`) {
    await llama.preemptIdle().catch(() => undefined)
    await llama.eraseHeld()
  }
  await scratchpad.clearChat(id)
  chatHeldMedia.delete(id)
  for (const key of [`chat:${id}`, `agent:${id}`]) recallVectors.delete(key)
  scratchpadChanged()
}

/** Bring the folder under its budget after a save, never removing a pinned conversation or the agent's instructions. */
function pruneStates(dir: string): string[] {
  const budget = Math.max(1, loadSettings().runtime.stateCache.budgetGb) * 1024 ** 3
  const keep = new Set(listStates(dir).filter((s) => s.key.startsWith(INSTRUCTIONS_PREFIX)).map((s) => s.key))
  return prune(dir, budget, keep)
}

/** A conversation's messages as they are now, for a key such as chat:<id> or agent:<id>. */
function conversationMessages(key: string): AgentMessage[] {
  const id = key.slice(key.indexOf(':') + 1)
  if (key.startsWith('agent:')) return liveAgentSessions.get(id)?.messages ?? chats.loadSession(id)?.messages ?? []
  return chats.loadMessages(id)
}

/**
 * What a save covers, message by message: each message's id and a hash of what the model reads of
 * it. Keyed by a protected conversation's own key, so the record beside its encrypted state cannot
 * be used to test guesses at what was said.
 */
function markHash(key: string, value: unknown): string {
  const seal = stateSeal(key)
  const keyed = Buffer.isBuffer(seal) ? seal : null
  return (keyed ? crypto.createHmac('sha256', keyed) : crypto.createHash('sha256')).update(JSON.stringify(value)).digest('hex').slice(0, 24)
}

function stateMarks(key: string, messages: AgentMessage[]): StateMark[] {
  return messages.map((m) => ({
    id: m.id,
    hash: markHash(key, [m.role, m.content, m.reasoning ?? null, m.plan ?? null, m.toolCalls ?? null, m.toolResult?.content ?? null, m.followUp ?? null])
  }))
}

/**
 * How much of a reply a save stopped part way holds. Measured from the first character that is not
 * white space, since a stored reply may have lost the line breaks it started with.
 */
function partialMark(key: string, reply: { content: string; reasoning: string }): PartialMark {
  const content = reply.content.trimStart()
  const reasoning = reply.reasoning.trimStart()
  return { content: content.length, reasoning: reasoning.length, hash: markHash(key, [reasoning, content]) }
}

/** Whether a reply, as it stands now, still begins with what a save holds of it. */
function replyBegins(key: string, m: { content: string; reasoning?: string }, mark: PartialMark): boolean {
  const content = m.content.trimStart().slice(0, mark.content)
  const reasoning = (m.reasoning ?? '').trimStart().slice(0, mark.reasoning)
  return content.length === mark.content && reasoning.length === mark.reasoning && markHash(key, [reasoning, content]) === mark.hash
}

/** Whether `key`'s own turn is under way. */
function turnInFlight(key: string): boolean {
  const id = key.slice(key.indexOf(':') + 1)
  return key.startsWith('agent:') ? liveAgentSessions.has(id) : inFlightChats.has(id)
}

/**
 * Save a conversation while its own turn is under way: the reply is stopped where it is, saved
 * with everything before it, and carries on (see llama.saveDuringTurn). While a tool runs, or a
 * call is being written, it waits for the model's next words; a turn that ends first gets an
 * ordinary save instead.
 */
async function saveMidTurn(key: string): Promise<SaveNowResult> {
  const store = llama.stateStore
  const loaded = llama.loaded
  if (!store || !loaded) return { ok: false, why: 'Load a model first: what is saved is what the model has read.' }
  const seal = stateSeal(key)
  if (seal === 'locked') return { ok: false, why: 'It is locked.' }
  const id = key.slice(key.indexOf(':') + 1)
  let result: SaveNowResult = { ok: false, why: 'Nothing was saved.' }
  emit('state:progress', { key, stage: 'waiting', percent: null })
  try {
    const outcome = await llama.saveDuringTurn(key, async ({ resume, reply }) => {
      const covers = {
        messages: stateMarks(key, conversationMessages(key)),
        summaryUpto: viewKey(id),
        canTrim: !((loaded.model.arch?.ssmLayers ?? 0) > 0),
        partial: partialMark(key, reply)
      }
      const earlier = listStates(store.dir).filter((s) => s.fingerprint === store.fingerprint && s.tokens > 0)
      const perToken = earlier.length ? earlier.reduce((n, s) => n + s.bytes / s.tokens, 0) / earlier.length : null
      const tokens = chats.getContextUsed(id)
      const expectedBytes = perToken && tokens ? Math.round(perToken * tokens) : null
      const weights = { writing: 1, encrypting: seal ? 0.25 : 0 }
      const total = weights.writing + weights.encrypting
      emit('state:progress', { key, stage: 'writing', percent: expectedBytes ? 0 : null, bytes: 0 })
      const saved = await saveAsIs({
        ...store,
        key,
        sent: resume,
        seal,
        covers,
        expectedBytes,
        onProgress: (p) => {
          const fraction = p.total ? Math.min(1, p.done / p.total) : null
          const before = p.stage === 'encrypting' ? weights.writing : 0
          emit('state:progress', {
            key,
            stage: p.stage,
            percent: fraction === null ? null : Math.round(((before + weights[p.stage] * fraction) / total) * 100),
            bytes: p.done
          })
        }
      })
      if (!saved.ok) throw new Error(saved.why)
      emit('state:saved', { key, tokens: saved.meta.tokens, savedAt: saved.meta.savedAt })
      const removed = pruneStates(store.dir)
      for (const gone of removed) emit('state:removed', { key: gone })
      logger.info('prompt', `saved ${key} to disk part way through a reply`, { tokens: saved.meta.tokens, bytes: saved.meta.bytes, removed })
      result = { ok: true, tokens: saved.meta.tokens, savedAt: saved.meta.savedAt }
    })
    if (outcome === 'missed') {
      emit('state:progress', { key, stage: null })
      return await saveNow(key)
    }
  } catch (err) {
    result = { ok: false, why: `Not saved: ${err instanceof Error ? err.message : String(err)}` }
  } finally {
    emit('state:progress', { key, stage: null })
  }
  return result
}

type SaveNowResult = { ok: true; tokens: number; savedAt: number; already?: boolean } | { ok: false; why: string }

/** Saves asked for and not finished, by conversation: a second click waits for the first. */
const savingNow = new Map<string, Promise<SaveNowResult>>()

/**
 * Save a conversation's state, because the user asked to. Nothing else saves one.
 *
 * When the server is holding another conversation, this one is read in first (what was saved of
 * it before is loaded, and anything after that read on top), so it can take as long as reading it
 * does. How far it has got is reported as it goes: reading, the server writing the file, and for a
 * protected conversation encrypting it. It runs as background work, so a message sent meanwhile
 * goes first and stops it.
 *
 * The record says which messages it covers, so the window can mark them, and whether the model can
 * cut a loaded state back, which decides what an edit inside the saved part costs (see coverage).
 */
async function saveNow(key: string): Promise<SaveNowResult> {
  if (!/^(chat|agent):[\w-]+$/.test(key)) return { ok: false, why: 'There is no conversation to save.' }
  const store = llama.stateStore
  const loaded = llama.loaded
  if (!loaded) return { ok: false, why: 'Load a model first: what is saved is what the model has read.' }
  if (!store) return { ok: false, why: 'This model cannot save its state. Load it again to try.' }
  // Its own reply being written is saved part way; another conversation's has the server.
  if (turnInFlight(key)) return saveMidTurn(key)
  if (agentTurnAbort || inFlightChats.size > 0) return { ok: false, why: 'Another conversation is being answered. Save once it has finished.' }
  const seal = stateSeal(key)
  if (seal === 'locked') return { ok: false, why: 'It is locked.' }
  const req = await conversationRequest(key)
  if (!req) return { ok: false, why: 'There is nothing in it to save yet.' }
  const id = key.slice(key.indexOf(':') + 1)
  const messages = conversationMessages(key)
  const covers = {
    messages: stateMarks(key, messages),
    summaryUpto: viewKey(id),
    canTrim: !((loaded.model.arch?.ssmLayers ?? 0) > 0)
  }
  try {
    // Nothing said since the last save: saving again would write the same thing.
    const current = await currentState({ ...store, key, req })
    if (current && coverage(current, covers.messages, covers.summaryUpto).count === messages.length) {
      return { ok: true, tokens: current.tokens, savedAt: current.savedAt, already: true }
    }
  } catch {
    /* found out by saving */
  }

  // How big the file will be, from earlier saves under this load: for a percentage while it is written.
  const earlier = listStates(store.dir).filter((s) => s.fingerprint === store.fingerprint && s.tokens > 0)
  const perToken = earlier.length ? earlier.reduce((n, s) => n + s.bytes / s.tokens, 0) / earlier.length : null
  const tokens = chats.getContextUsed(id)
  const expectedBytes = perToken && tokens ? Math.round(perToken * tokens) : null

  const reading = !llama.holds(req)
  const weights = { reading: reading ? 1 : 0, writing: 1, encrypting: seal ? 0.25 : 0 }
  const totalWeight = weights.reading + weights.writing + weights.encrypting
  const report = (stage: 'reading' | SaveProgress['stage'], fraction: number | null, bytes?: number): void => {
    const before = stage === 'reading' ? 0 : stage === 'writing' ? weights.reading : weights.reading + weights.writing
    const percent = fraction === null ? null : Math.round(((before + weights[stage] * Math.min(1, fraction)) / totalWeight) * 100)
    emit('state:progress', { key, stage, percent, bytes })
  }

  let result: SaveNowResult = { ok: false, why: 'A message was sent before it finished, so nothing was saved.' }
  try {
    await llama.runIdle(async (signal) => {
      try {
        if (reading) {
          report('reading', 0)
          if (!(await restoreSaved(req, signal)) && !signal.aborted) await seedInstructions(req, signal, true)
          if (signal.aborted) return
          await readAhead(req, signal, (done, total) => report('reading', total ? done / total : null))
          if (signal.aborted) return
        }
        report('writing', expectedBytes ? 0 : null, 0)
        const saved = await saveState({
          ...store,
          key,
          req,
          signal,
          seal,
          covers,
          expectedBytes,
          onProgress: (p) => report(p.stage, p.total ? p.done / p.total : null, p.done)
        })
        if (signal.aborted) return
        if (!saved.ok) {
          result = { ok: false, why: `Not saved: ${saved.why}.` }
          return
        }
        llama.assumeHeld(req)
        emit('state:saved', { key, tokens: saved.meta.tokens, savedAt: saved.meta.savedAt })
        const removed = pruneStates(store.dir)
        // Saves removed to stay within the disk budget: their messages show as unsaved again.
        for (const gone of removed) emit('state:removed', { key: gone })
        logger.info('prompt', `saved ${key} to disk`, { tokens: saved.meta.tokens, bytes: saved.meta.bytes, removed })
        result = { ok: true, tokens: saved.meta.tokens, savedAt: saved.meta.savedAt }
      } catch (err) {
        if (!signal.aborted) result = { ok: false, why: `Not saved: ${err instanceof Error ? err.message : String(err)}` }
      }
    })
  } finally {
    emit('state:progress', { key, stage: null })
  }
  return result
}

/** Agent sessions with a turn running, as the turn has them: stored only when it ends. */
const liveAgentSessions = new Map<string, AgentSessionState>()


/**
 * Load a conversation's saved state into the server. False when there is none that fits.
 *
 * Its own save first. With none (or none that still fits), the longest start of it that any other
 * save holds is borrowed: the system message conversations begin with, or the history a fork
 * shares with the conversation it was forked from. Anything after what was loaded is read on top.
 */
async function restoreSaved(req: CompletionOptions, signal?: AbortSignal, sent = false): Promise<boolean> {
  const store = llama.stateStore
  const key = req.cacheKey
  if (!store || !key) return false
  const seal = stateSeal(key)
  if (seal === 'locked') return false
  if (readMeta(store.dir, key) && (await restoreOwn(req, key, seal, signal, sent))) return true
  if (signal?.aborted) return false
  return restoreBorrowed(req, key, signal, sent)
}

async function restoreOwn(
  req: CompletionOptions,
  key: string,
  seal: Buffer | undefined,
  signal?: AbortSignal,
  sent = false
): Promise<boolean> {
  const store = llama.stateStore
  if (!store) return false
  try {
    const result = await restoreState({ ...store, key, req, signal, seal, sent })
    if (!result.ok) {
      logger.info('prompt', `${key} was not restored from disk: ${result.why}`)
      return false
    }
    llama.assumeHeld(req)
    logger.info('prompt', `restored ${key} from disk`, { tokens: result.meta.tokens, bytes: result.meta.bytes })
    return true
  } catch (err) {
    logger.warn('prompt', `could not restore ${key} from disk`, err)
    return false
  }
}

/**
 * The longest start of this conversation another conversation's save holds, loaded. A protected
 * conversation neither lends nor borrows: its saves are its own.
 */
async function restoreBorrowed(req: CompletionOptions, key: string, signal?: AbortSignal, sent = false): Promise<boolean> {
  const store = llama.stateStore
  if (!store || stateSeal(key) !== undefined) return false
  try {
    const prefix = sent ? await requestText(store.base, req, signal) : await renderPrefix(store.base, req, signal)
    const lender = longestReusable(store.dir, store.fingerprint, prefix, req.messages, (s) => s.key !== key && stateSeal(s.key) === undefined)
    if (!lender || signal?.aborted) return false
    const result = await restoreState({ ...store, key: lender.key, req, signal, borrowed: true, sent })
    if (!result.ok) {
      logger.info('prompt', `could not borrow ${lender.key} for ${key}: ${result.why}`)
      return false
    }
    llama.assumeHeld(req)
    logger.info('prompt', `loaded the start of ${key} from ${lender.key}`, { tokens: result.meta.tokens })
    return true
  } catch (err) {
    logger.warn('prompt', `could not look for a save to borrow for ${key}`, err)
    return false
  }
}

/**
 * Before a turn: bring this conversation's saved state back, if it has one. Anything said after it
 * was saved is read by the turn itself, on top of it.
 *
 * The same thing clicking into the message box does ahead of time, for a message sent without
 * that head start.
 */
async function resumeSaved(req: CompletionOptions, sent = false): Promise<void> {
  if (!llama.stateStore || !req.messages.length) return
  // A load already under way (clicking into the message box starts one) finishes first, and counts.
  await llama.preemptIdle()
  /*
   * Already held, in a way this request can use. A new message can follow a read-ahead (the server
   * keeps its place at the start of the stand-in message). Continue, or a rewound message sent
   * again, cannot: they end before the stand-in, and a model with recurrent layers cannot step
   * back to them, so after a read-ahead the save is loaded again.
   */
  if (llama.holds(req) && !(sent && llama.heldIsReadAhead)) return
  if (await restoreSaved(req, undefined, sent)) return
  // Someone is waiting: load the instructions if they are on disk, but do not stop to save them.
  await seedInstructions(req, new AbortController().signal, false)
}

/*
 * The agent's instructions, kept on disk.
 *
 * Every Agent conversation begins with the same system prompt and tool list, thousands of tokens
 * the model had to read again after every load. Their state is saved once, under a key made from
 * exactly that text and the effort it was rendered with, and loaded back in place of reading it
 * when a model has just been loaded. A loaded state only has to be how the conversation begins,
 * so the one file serves every conversation started with those instructions.
 */
const INSTRUCTIONS_PREFIX = 'agent-instructions:'
const INSTRUCTIONS_KEEP = 3

function instructionsRequest(req: CompletionOptions): CompletionOptions | null {
  if (!req.cacheKey?.startsWith('agent:') || req.messages[0]?.role !== 'system') return null
  const id = crypto
    .createHash('sha256')
    .update(JSON.stringify([req.messages[0], req.tools ?? [], req.reasoningEffort ?? null, req.chatTemplateKwargs ?? null]))
    .digest('hex')
    .slice(0, 16)
  return { ...req, messages: [req.messages[0]], cacheKey: INSTRUCTIONS_PREFIX + id }
}

/** With nothing held yet, load the instructions from disk, or read them and save them when `save`. */
async function seedInstructions(req: CompletionOptions, signal: AbortSignal, save: boolean): Promise<void> {
  const store = llama.stateStore
  // Anything already held began with the same instructions, and the server reuses that by itself.
  if (!store || loadSettings().runtime.keepInstructions === false || llama.heldKey) return
  const inst = instructionsRequest(req)
  if (!inst?.cacheKey) return
  try {
    if (readMeta(store.dir, inst.cacheKey)) {
      const result = await restoreState({ ...store, key: inst.cacheKey, req: inst, signal })
      if (result.ok) {
        llama.assumeHeld(inst)
        logger.info('prompt', 'restored the agent instructions from disk', { tokens: result.meta.tokens, bytes: result.meta.bytes })
        return
      }
      logger.info('prompt', `the agent instructions were not restored from disk: ${result.why}`)
    }
    if (!save || signal.aborted) return
    await readAhead(inst, signal)
    if (signal.aborted) return
    const saved = await saveState({ ...store, key: inst.cacheKey, req: inst, signal })
    if (!saved.ok) {
      logger.info('prompt', `the agent instructions were not saved to disk: ${saved.why}`)
      return
    }
    llama.assumeHeld(inst)
    const kept = listStates(store.dir)
      .filter((s) => s.key.startsWith(INSTRUCTIONS_PREFIX))
      .sort((a, b) => b.usedAt - a.usedAt)
    for (const old of kept.slice(INSTRUCTIONS_KEEP)) removeState(store.dir, old.key)
    logger.info('prompt', 'saved the agent instructions to disk', { tokens: saved.meta.tokens, bytes: saved.meta.bytes })
  } catch (err) {
    logger.warn('prompt', 'could not keep the agent instructions on disk', err)
  }
}

async function readAhead(
  req: CompletionOptions,
  signal: AbortSignal,
  onProgress?: (processed: number, total: number) => void
): Promise<void> {
  const events = llama.streamEvents({
    ...req,
    messages: [...req.messages, STAND_IN],
    processOnly: true,
    background: true,
    signal
  })
  for await (const ev of events) if (ev.type === 'prompt_progress') onProgress?.(ev.processed, ev.total)
}

/** The effort fields a conversation's requests carry, which anything read ahead must carry too. */
function effortFields(choice: ReasoningChoice | null | undefined): ReturnType<typeof reasoningRequestFields> {
  const support = llama.loaded?.model.caps.reasoning
  return reasoningRequestFields(support, sendableChoice(support, choice ?? null))
}

/*
 * Chats are named by the model.
 *
 * After the first reply, and every few turns after that, the model is asked for a title in the
 * background. The question is added to the end of the conversation exactly as a read-ahead's
 * stand-in is, so the server reuses everything before it and the next message reuses it again;
 * it costs a few tokens read and a few written, while the reply is being read. A later ask shows
 * the current title and lets the model keep it, so a name changes only when the conversation has.
 */
const RETITLE_EVERY = 3

function titleDue(chatId: string): boolean {
  const state = chats.titleState(chatId)
  if (!state || state.byUser || state.userTurns < 1) return false
  return state.userTurns === 1 || (state.userTurns - 1) % RETITLE_EVERY === 0
}

async function retitleChat(chatId: string, reasoning: ReasoningChoice | null, signal: AbortSignal): Promise<void> {
  const state = chats.titleState(chatId)
  if (!state || state.byUser) return
  const history = chats.loadMessages(chatId)
  if (!history.length) return
  await restoreConversationMedia(chatId, history)
  if (signal.aborted) return
  const conversation: CompletionOptions = {
    messages: chatMessagesFor(chatId, history, loadSettings().reasoning.preserve),
    cacheKey: `chat:${chatId}`,
    ...effortFields(reasoning)
  }
  const named = state.userTurns > 1
  const ask = named
    ? `(Not part of the conversation: this chat is titled "${state.title}". If that still says what it is about, ` +
      'reply with exactly that title. Otherwise reply with only a better one: 2 to 6 words, in the language of ' +
      'the conversation, no quotes.)'
    : '(Not part of the conversation: name this chat for the sidebar. Reply with only the title: 2 to 6 words, ' +
      'in the language of the conversation, no quotes.)'
  /*
   * No thinking, without touching the prompt start.
   *
   * Switching thinking off through the template (enable_thinking, an 'none' effort) renders the
   * system prompt differently on Qwen3.8, and the whole conversation was read again: measured,
   * 0 of 580 tokens reused. A budget of 0 alone did not stop it thinking. A reply that starts with
   * a closed <think></think> pair keeps every token before the question and answers in ~10 tokens.
   * Templates without those tags get room to think briefly instead.
   */
  const support = llama.loaded?.model.caps.reasoning
  const thinks = !!support && support.kind !== 'none'
  const prefill: ChatMessage[] = thinks && support?.thinkTags ? [{ role: 'assistant', content: '<think>\n\n</think>\n\n' }] : []
  try {
    const answer = await llama.complete({
      ...conversation,
      messages: [...conversation.messages, { role: 'user', content: ask }, ...prefill],
      continueFinal: prefill.length > 0,
      reasoningBudget: 0,
      temperature: 0.3,
      maxTokens: thinks && !prefill.length ? 256 : 32,
      background: true,
      signal
    })
    if (signal.aborted) return
    const applied = chats.setModelTitle(chatId, answer)
    if (applied) {
      // A protected conversation's title is not written to the log in the clear.
      logger.info('chat', `the model named ${chatId}`, recordOf({ kind: 'chat', id: chatId }) ? {} : { title: applied })
      emit('chat:titled', { chatId, title: applied })
    }
  } catch (err) {
    if (!signal.aborted) logger.info('chat', `could not name ${chatId}: ${err instanceof Error ? err.message : String(err)}`)
  } finally {
    // The server holds the conversation with the question after it, as after a read-ahead; the
    // next message reuses everything before the question.
    llama.assumeHeld(conversation)
  }
}

/** After a chat turn: name it if due. Nothing is summarised unasked; Compact does that. */
function afterChatTurn(chatId: string, reasoning: ReasoningChoice | null): void {
  const loaded = llama.loaded
  if (!loaded || agentTurnAbort || inFlightChats.size > 0 || llama.idleBusy) return
  if (!titleDue(chatId)) return
  void llama.runIdle((signal) => retitleChat(chatId, reasoning, signal))
}

export function modelsDir(): string {
  return loadSettings().modelsDir ?? defaultModelsDir()
}

/** Every folder the library lists models from: the models folder, then the user's extra folders. */
export function libraryFolders(): string[] {
  return libraryRoots(modelsDir(), loadSettings().extraModelDirs)
}

/**
 * The detection in flight, shared by everyone who asks while it runs.
 *
 * A detection is seconds of nvidia-smi and WMI, and at startup the dashboard, the library and the
 * downloads page all ask at once — they should wait on one run rather than start one each.
 */
let detecting: Promise<HardwareSnapshot> | null = null
/** Detections in a row that came back provisional, and when the next one may run. */
let provisionalDetections = 0
let nextDetectionAt = 0

function detect(): Promise<HardwareSnapshot> {
  detecting ??= detectHardware()
    .then((fresh) => {
      const merged = mergeDetection(hardware, fresh)
      hardware = merged
      if (merged.detection?.state === 'provisional') {
        provisionalDetections++
        // 5 s, 10 s, 20 s, 40 s, then once a minute: quick enough for a slow start, never a busy loop.
        nextDetectionAt = Date.now() + Math.min(60_000, 5_000 * 2 ** (provisionalDetections - 1))
        logger.warn('hardware', 'GPU detection got no answer; it will be retried', {
          reason: merged.detection.reason,
          attempt: provisionalDetections
        })
      } else {
        provisionalDetections = 0
      }
      return merged
    })
    .finally(() => {
      detecting = null
    })
  return detecting
}

/**
 * The hardware snapshot, detected on first use.
 *
 * A provisional one — a detection that timed out, as opposed to one that found nothing — is
 * detected again whenever a retry is due. The first snapshot used to be kept for the whole session,
 * so a driver slow to answer at startup meant a machine with GPUs was planned for as if it had none
 * until the app was restarted.
 */
async function getHardware(refresh = false): Promise<HardwareSnapshot> {
  if (!hardware || refresh) return detect()
  if (hardware.detection?.state === 'provisional' && Date.now() >= nextDetectionAt) return detect()
  return hardware
}

/**
 * The snapshot to load a model against, detected again first if the last one was a guess.
 *
 * Loading against a provisional snapshot is how a machine with two GPUs ends up running a model on
 * its CPU. One more detection costs a few seconds against that, so a load does not wait out the
 * backoff.
 */
async function settledHardware(): Promise<HardwareSnapshot> {
  if (!hardware || hardware.detection?.state === 'provisional') return detect()
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
        /*
         * A provisional snapshot is detected again, not refreshed.
         *
         * Refreshing re-reads the adapters already known, so a snapshot that missed the GPUs would
         * have stayed without them however long the session ran.
         */
        if (hardware.detection?.state === 'provisional') {
          if (Date.now() < nextDetectionAt) return
          await detect()
        } else {
          hardware = await refreshFreeVram(hardware)
        }
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
      sentContentFor: (messageId) => sentContentFor(messageId, llama.loaded?.model.caps),
      sentTextFor: (messageId) => sentTextFor(messageId),
      // Recorded against the session the turn belongs to, which is the one running when it sends.
      onSent: (messageId, content) => void persistSent(activeAgentSessionId, messageId, content),
      planMode: s.agent.planMode,
      maxToolCallsPerTurn: s.agent.maxToolCallsPerTurn,
      commandTimeoutMs: s.agent.commandTimeoutMs,
      hardBlocksDisabled: s.agent.hardBlocksDisabled,
      compaction: s.agent.compaction,
      compactionHelper: resolveCompactionHelper,
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
    agent.on('continuing', (c: { messageId: string }) => emit('agent:continuing', { sessionId: sid(), ...c }))
    agent.on('promptProgress', (p: { percent: number; processed: number; total: number; cached: number }) =>
      emit('agent:prompt-progress', { sessionId: sid(), ...p })
    )
    agent.on('contextUsed', (c: { used: number; max: number }) => {
      emit('agent:context', { sessionId: sid(), ...c })
      setContextUsed(c.used)
      // Kept for the end of the turn rather than written on every update — see the persist below.
      lastAgentContext = c.used
    })
    /*
     * Re-attach the files on the way out.
     *
     * The user's turn is announced twice by design — once here, before any slow preparation, and
     * again by the loop when it stores the turn for real. The renderer replaces by id, so the
     * loop's copy wins, and the loop does not know about attachments: the message it builds has
     * only text. The result was a transcript that showed the video for a moment and then dropped
     * back to naming it.
     *
     * Looked up rather than remembered, and only for the role and shape that can have any, so a
     * long agent turn does not query per assistant message.
     */
    agent.on('message', (m) => {
      const message =
        m.role === 'user' && !m.attachments?.length
          ? { ...m, attachments: chats.attachmentsForMessage(m.id) }
          : m
      emit('agent:message', { sessionId: sid(), message })
    })
    agent.on('toolCall', (c) => emit('agent:tool-call', { sessionId: sid(), call: c }))
    agent.on('speed', (p: { tokensPerSecond: number; tokens: number }) => emit('agent:speed', { sessionId: sid(), ...p }))
    agent.on('toolCallPartial', (p: { index: number; name: string; args: string }) =>
      emit('agent:tool-call-partial', { sessionId: sid(), ...p })
    )
    agent.on('toolResult', (r) => emit('agent:tool-result', { sessionId: sid(), result: r }))
    agent.on('toolFrames', (p: { messageId: string; frames: string[] }) => emit('agent:tool-frames', { sessionId: sid(), ...p }))
    agent.on('title', (t: { sessionId: string; title: string }) => emit('chat:titled', { chatId: t.sessionId, title: t.title }))
    agent.on('subToolCall', (c) => emit('agent:sub-tool-call', { sessionId: sid(), call: c }))
    agent.on('compacting', (info) => emit('agent:compacting', { sessionId: sid(), ...(info as object) }))
    agent.on('compacted', (info) => emit('agent:compacted', { sessionId: sid(), ...(info as object) }))
    agent.on('compactionProgress', (p) => emit('agent:compaction-progress', { sessionId: sid(), ...(p as object) }))
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
    agent.on('stepRetry', (n: { message: string }) => emit('agent:step-retry', { sessionId: sid(), ...n }))
    agent.on('notice', (n: { message: string }) => emit('agent:notice', { sessionId: sid(), ...n }))

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
    preserveReasoning: s.reasoning.preserve,
    sampling: s.generation.sampling,
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
/**
 * The small model compaction should use instead of the loaded one, started if need be.
 *
 * Null whenever the loaded model should do it after all: the option is off, nothing is picked,
 * the pick is not in the library or is the model already loaded, or the helper would not start.
 * Each of those falls back rather than failing, because the compaction has to happen either way.
 */
async function resolveCompactionHelper(): Promise<SummarizerHandle | null> {
  const s = loadSettings()
  const choice = s.agent.summarizer
  if (!choice?.enabled || !choice.modelId) return null
  const model = library.find((m) => m.id === choice.modelId || m.filename === choice.modelId)
  if (!model?.arch) {
    logger.warn('compaction', 'summariser model is not in the library; the loaded model will summarise', {
      modelId: choice.modelId
    })
    return null
  }
  if (llama.loaded?.model.path === model.path) return null

  try {
    const hw = await refreshFreeVram(await getHardware())
    hardware = hw
    // The user's headroom is kept for the desktop here too, exactly as the planner keeps it.
    const headroom = s.autoFit.headroomMb * 1024 * 1024
    const gpus =
      hw.backend === 'cuda'
        ? hw.gpus
            .filter((g) => g.vendor === 'nvidia' && g.freeIsMeasured)
            .map((g) => ({ index: g.index, name: g.name, freeBytes: g.freeVram - headroom }))
        : []
    const need = helperVramNeed(model.arch, model.bytes, helperContext(model.arch))
    const placement = chooseHelperPlacement(need, gpus, choice.placement)
    return await summarizer.ensure(model, placement, hw.backend)
  } catch (err) {
    logger.warn('compaction', 'summariser did not start; the loaded model will summarise', {
      error: err instanceof Error ? err.message : String(err)
    })
    return null
  }
}

/**
 * Media parts are charged a flat rate, as the agent charges them: an image is sent as a data URL,
 * and measuring that as text would value one photo at hundreds of thousands of tokens.
 */
const MEDIA_TOKEN_ESTIMATE = 1200

/**
 * The messages a chat's prompt is built from: its stored summary in place of the messages that
 * summary covers, then everything after them word for word.
 *
 * A summary whose last message is no longer in the history describes nothing that is, so it is
 * left out. It rides on the first kept message when that one is the user's, so the roles still
 * alternate — some chat templates refuse two user turns in a row.
 */
/**
 * Frames chat's watch_video returned, by conversation and by the message they belong to.
 *
 * In memory only, like the scratchpad they came out of: a rebuilt prompt sends them again for as
 * long as the app stays open, and after a restart says they are gone, rather than pictures of
 * someone's video being written to disk to survive it.
 */
const chatHeldMedia = new Map<string, Map<string, ContentPart[]>>()

function holdChatMedia(chatId: string, messageId: string, parts: ContentPart[]): void {
  let held = chatHeldMedia.get(chatId)
  if (!held) {
    held = new Map()
    chatHeldMedia.set(chatId, held)
  }
  held.set(messageId, parts)
  // A long session of looking at videos should not hold every frame it ever saw.
  if (held.size > 64) held.delete(held.keys().next().value as string)
}

const FRAMES_NOT_KEPT: ContentPart[] = [
  {
    type: 'text',
    text: '[The frames this look returned are not kept once the app closes. If you need them and the video is still in your scratchpad, look again.]'
  }
]

/** What each stored message of a chat sent beyond its text, for rebuilding its prompt. */
function chatSentContent(chatId: string, history: AgentMessage[]): (id: string) => string | ContentPart[] | undefined {
  const caps = llama.loaded?.model.caps
  const looks = new Set(
    history
      .filter((m) => m.role === 'tool' && m.toolCalls?.[0]?.name === 'watch_video' && /Showing \d+ frame/.test(m.content))
      .map((m) => m.id)
  )
  return (id) => chatHeldMedia.get(chatId)?.get(id) ?? (looks.has(id) ? FRAMES_NOT_KEPT : sentContentFor(id, caps))
}

/**
 * The tools a chat's next request offers: its scratchpad's, once the conversation has used it.
 *
 * Only from the first time they are needed, so an ordinary conversation's prompt is exactly what it
 * always was; and from then on every time, scratchpad empty or not, because tool definitions open
 * the prompt and changing them makes the server read the whole conversation again. Never for a
 * model that cannot call tools; those are shown an overview of a video instead.
 */
function chatToolsFor(chatId: string, history: AgentMessage[], adding = false): ToolDefinition[] | undefined {
  if (!llama.loaded?.model.caps.tools) return undefined
  const used =
    adding ||
    scratchpad.list(chatId).length > 0 ||
    history.some((m) => m.role === 'tool' || m.attachments?.some((a) => a.kind === 'video'))
  // Once part of the conversation is no longer sent in full, the model can look back into it.
  // A quick compact whose stretches were all rewound away has left nothing to look back into.
  const compacted = !!chats.getSummary(chatId) || standing(history, chats.getQuick(chatId)).length > 0
  const tools = [...(used ? CHAT_TOOLS : []), ...(compacted ? RECALL_TOOLS : [])]
  return tools.length ? tools : undefined
}

/** Vectors worked out for search by meaning, by conversation: in memory only, gone with the app. */
const recallVectors = new Map<string, Map<string, Float32Array[]>>()

/** What the recall tools search for a conversation: its transcript, and where the part still sent in full begins. */
function recallSourceFor(key: string): RecallSource {
  const id = key.slice(key.indexOf(':') + 1)
  const messages = conversationMessages(key).filter((m) => m.role !== 'system')
  const summary = chats.getSummary(id)
  const skip = summary ? messages.findIndex((m) => m.id === summary.uptoMessageId) + 1 : 0
  const stretches = standing(messages, chats.getQuick(id), skip)
  let vectors = recallVectors.get(key)
  if (!vectors) {
    vectors = new Map()
    recallVectors.set(key, vectors)
  }
  const caps = llama.loaded?.model.caps
  return {
    messages,
    visibleFrom: stretches.length ? stretches[stretches.length - 1].end + 1 : skip,
    sentText: (mid) => sentTextFor(mid),
    sentParts: (mid) => {
      const sent = sentContentFor(mid, caps)
      return Array.isArray(sent) ? sent : undefined
    },
    embed: async (texts) => rag.embeddings.embed(texts, (await getHardware()).backend),
    vectors
  }
}
setRecallSource((sessionId) => recallSourceFor(`agent:${sessionId}`))

async function runRecall(key: string, name: string, args: Record<string, unknown>, signal: AbortSignal): Promise<ToolOutput> {
  const from = recallSourceFor(key)
  return name === 'recall_search' ? recallSearch(from.messages, args, { ...from, signal }) : recallRead(from.messages, args, from)
}

/** What stands in for the summary and any quick compact when judging whether a save still fits. */
function viewKey(id: string): string | null {
  const summary = chats.getSummary(id)?.uptoMessageId ?? null
  const quick = chats.getQuick(id)
  return quick ? `${summary ?? ''}|${JSON.stringify(quick.segments)}` : summary
}

/** What a quick compact of a conversation would do, without doing it. */
async function quickPlanFor(key: string): Promise<QuickPlan | { ok: false; message: string }> {
  const loaded = llama.loaded
  if (!loaded) return { ok: false, message: 'Load a model first: how much to keep depends on its context window.' }
  if (!/^(chat|agent):[\w-]+$/.test(key)) return { ok: false, message: 'Open a conversation first.' }
  const id = key.slice(key.indexOf(':') + 1)
  if (chats.chatLocked(id)) return { ok: false, message: 'It is locked.' }
  const all = conversationMessages(key).filter((m) => m.role !== 'system')
  const summary = chats.getSummary(id)
  const upto = summary ? all.findIndex((m) => m.id === summary.uptoMessageId) : -1
  await restoreConversationMedia(id, all)
  let fixed = upto === -1 ? 0 : estimateTokens(summary?.summary ?? '')
  if (key.startsWith('agent:')) {
    const session = chats.loadSession(id)
    if (!session) return { ok: false, message: 'Open a conversation first.' }
    const a = getAgent()
    fixed += estimateTokens(String(a.buildHistory({ ...session, messages: [] })[0]?.content ?? '')) + estimateTokens(JSON.stringify(a.toolsFor(session)))
  } else {
    fixed += estimateTokens(String(loadSettings().generation.systemPrompt ?? '')) + estimateTokens(JSON.stringify(chatToolsFor(id, all) ?? []))
  }
  return planQuick(all, loaded.plan.contextLength || 8192, fixed, {
    skip: upto + 1,
    recall: !!loaded.model.caps.tools,
    sentText: (mid) => sentTextFor(mid),
    previous: chats.getQuick(id)
  })
}

/** How many rounds of tool calls one chat turn may make before it is made to answer. */
const CHAT_TOOL_ROUNDS = 24

function chatPromptHistory(chatId: string, history: AgentMessage[], preserveReasoning: boolean): ChatMessage[] {
  const stored = chats.getSummary(chatId)
  const upto = stored ? history.findIndex((m) => m.id === stored.uptoMessageId) : -1
  /*
   * An earlier message goes back exactly as it was sent, attachments included, and tool use goes
   * back as tool use: the calls on the reply that made them, each result after, and any frames it
   * returned after that, the shape the turn first had. The transcript has only the text.
   */
  // A quick compact changes how the older messages are sent, never the messages themselves.
  const canRecall = !!llama.loaded?.model.caps.tools
  const view = quickView(history, chats.getQuick(chatId), { skip: upto + 1, recall: canRecall })
  const sent = chatSentContent(chatId, history)
  const out = rebuildHistory(view.messages, {
    preserveReasoning,
    sentContentFor: (id) => (view.dropSent.has(id) ? undefined : withPreface(view, id, sent(id)))
  })
  if (stored && upto !== -1) {
    const note =
      `[Summary of the earlier conversation]\n${stored.summary}` +
      (canRecall ? '\n\n[The messages this summary stands in for are kept in full: recall_search finds anything in them and recall_read shows it.]' : '')
    const first = out[0]
    if (first?.role === 'user' && typeof first.content === 'string') {
      out[0] = { ...first, content: `${note}\n\n[The conversation continues]\n${first.content}` }
    } else if (first?.role === 'user' && Array.isArray(first.content)) {
      out[0] = { ...first, content: [{ type: 'text', text: `${note}\n\n[The conversation continues]` }, ...first.content] }
    } else {
      out.unshift({ role: 'user', content: note })
    }
  }
  return out
}

/** A chat's prompt before the new message: the parameters panel's system prompt, then its history. */
function chatMessagesFor(chatId: string, history: AgentMessage[], preserveReasoning: boolean): ChatMessage[] {
  const messages = chatPromptHistory(chatId, history, preserveReasoning)
  // The system prompt leads, ahead of any summary and the history itself.
  const systemPrompt = String(loadSettings().generation.systemPrompt ?? '')
  if (systemPrompt.trim()) messages.unshift({ role: 'system', content: systemPrompt })
  return messages
}

interface ChatCompactionReport {
  ok: boolean
  message: string
  beforeTokens?: number
  afterTokens?: number
}

/**
 * Summarise a chat's older messages so the conversation fits its window again.
 *
 * Only ever when the user presses Compact; nothing is summarised unasked. The most recent messages
 * are kept word for word: a quarter of the window, at most 32,000 tokens (see keepBudget).
 * A summary already stored is summarised again along with the messages after it, so each compaction
 * costs what has built up since the last one rather than the whole conversation.
 *
 * Never throws. A compaction that fails leaves the conversation as it was and the turn goes ahead —
 * the server's own refusal describes an overfull window better than a guess made here.
 */
async function compactChat(
  chatId: string,
  history: AgentMessage[],
  opts: {
    preserveReasoning: boolean
    signal?: AbortSignal
    /** The effort level the conversation's turns are sent with, so the summary request starts as they do. */
    reasoning?: ReasoningChoice | null
  }
): Promise<ChatCompactionReport> {
  const loaded = llama.loaded
  if (!loaded) return { ok: false, message: 'Load a model first — compacting needs one to summarise with.' }

  const stored = chats.getSummary(chatId)
  const upto = stored ? history.findIndex((m) => m.id === stored.uptoMessageId) : -1
  const summary = upto === -1 ? null : stored
  // A summary of messages that are no longer there describes nothing.
  if (stored && !summary) chats.clearSummary(chatId)

  const live = summary ? history.slice(upto + 1) : history
  const costs = live.map(
    (m) =>
      estimateTokens(sentTextFor(m.id) ?? m.content) +
      (opts.preserveReasoning && m.role === 'assistant' ? estimateTokens(m.reasoning ?? '') : 0)
  )
  // Sized as it is sent: part of it may be quick-compacted, and is then far smaller than its messages.
  const sentSize = (from: number): number => {
    const view = quickView(history, chats.getQuick(chatId), { skip: from })
    return view.messages.reduce((n, m) => n + costOf(m, (id) => (view.dropSent.has(id) ? undefined : sentTextFor(id))), 0)
  }
  const before = (summary ? estimateTokens(summary.summary) : 0) + sentSize(upto + 1)
  const windowTokens = loaded.plan.contextLength || 8192

  let cut = keepRecent(costs, keepBudget(windowTokens))
  // A tool's result stays with the call that produced it.
  while (cut > 0 && cut < live.length && live[cut].role === 'tool') cut--
  const older = live.slice(0, cut)
  if (older.length < (summary ? 1 : 2)) {
    return { ok: false, message: 'Nothing to compact yet — the recent messages are all that is in the window.' }
  }

  const items: TranscriptItem[] = [
    ...(summary ? [{ role: 'summary so far', text: summary.summary }] : []),
    // What each message sent, where that was more than its text, so attached documents are summarised too.
    ...older.map((m) => ({ role: m.role, text: sentTextFor(m.id) ?? m.content }))
  ]

  let boundary: string | null = null
  emit('chat:compacting', { chatId, automatic: false })
  try {
    // The requests this conversation's turns send, tools included: anything else starts the prompt
    // differently and the whole of it is read again just to be summarised.
    const tools = chatToolsFor(chatId, history)
    const messages = chatMessagesFor(chatId, history, opts.preserveReasoning)
    // Loaded from its save first, when the server is not holding it: a file read, not a reading.
    await resumeSaved({ messages, cacheKey: `chat:${chatId}`, ...(tools ? { tools } : {}), ...effortFields(opts.reasoning) })
    /*
     * Asked of the conversation itself first.
     *
     * The server is holding it, so the summary costs no reading; handing the transcript over as
     * text in a new prompt makes the model read the older half again before it writes anything.
     */
    // Not asked of the conversation itself once part of it is quick-compacted: the server holds
    // that part shortened, and a summary of one-line notes is no summary. The passes read the originals.
    const shortened = standing(history, chats.getQuick(chatId), upto + 1).length > 0
    const inPlace = shortened ? null : await summariseInPlace({
      kind: 'chat',
      messages,
      ...(tools ? { tools } : {}),
      cacheKey: `chat:${chatId}`,
      effort: effortFields(opts.reasoning),
      usedTokens: Math.max(before, chats.getContextUsed(chatId) ?? 0),
      signal: opts.signal,
      onProgress: (percent) => emit('chat:compaction-progress', { chatId, percent, chunk: 1, chunks: 1, stage: 'merge', helper: null })
    })
    if (opts.signal?.aborted) return { ok: false, message: 'Compaction was stopped.' }
    const text =
      inPlace ??
      (await summariseTranscript(items, {
        kind: 'chat',
        helper: await resolveCompactionHelper(),
        signal: opts.signal,
        onProgress: (p) => emit('chat:compaction-progress', { chatId, ...p })
      }))
    if (opts.signal?.aborted) return { ok: false, message: 'Compaction was stopped.' }
    if (text === COMPACTION_FAILED) {
      logger.warn('compaction', 'no pass produced a summary; the chat goes on with its full history', { chatId })
      return { ok: false, message: 'Compaction failed — no summary came back, so nothing was changed.' }
    }
    boundary = older[older.length - 1].id
    chats.setSummary(chatId, text, boundary)
    const after = estimateTokens(text) + sentSize(upto + 1 + cut)
    // An estimate until the next turn measures it, and far closer than the figure it replaces.
    chats.setContextUsed(chatId, after)
    emit('chat:context', { chatId, used: after, max: windowTokens })
    // What was saved of it held the conversation as it was; the edges say so.
    emit('state:removed', { key: `chat:${chatId}` })
    return { ok: true, message: `Summarised ${older.length} messages.`, beforeTokens: before, afterTokens: after }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    logger.warn('compaction', 'chat compaction failed; the chat goes on with its full history', { chatId, error: message })
    return { ok: false, message: `Compaction failed: ${message}` }
  } finally {
    emit('chat:compacted', { chatId, uptoMessageId: boundary })
  }
}

/**
 * Rebuild what a message from before sent content was recorded sent the model, from its kept files.
 *
 * Documents are read again, images and audio re-encoded from their kept copies, and a video sent as
 * the clip and stills made from it — never sampled again. Null when the message carried nothing
 * beyond its text; undefined when the loaded model cannot take something it carried, so it is looked
 * at again once one that can is loaded.
 */
async function reconstructSent(message: AgentMessage): Promise<ContentPart[] | null | undefined> {
  const rows = chats.attachmentRowsForMessage(message.id)
  if (!rows.length) return null
  const caps = llama.loaded?.model.caps
  if (!caps) return undefined
  const video = await videoContext()

  const files = rows.filter((r) => r.kind !== 'video' && fs.existsSync(r.path)).map((r) => r.path)
  if (files.some((f) => classifyAttachment(f) === 'image') && !caps.vision) return undefined
  if (files.some((f) => classifyAttachment(f) === 'audio') && !caps.audio) return undefined
  const parts: ContentPart[] = files.length ? (await buildContent('', files, caps, video)).parts : []

  for (const row of rows.filter((r) => r.kind === 'video')) {
    const clip = row.meta.optimised && fs.existsSync(row.meta.optimised) ? row.meta.optimised : null
    const stills = (row.meta.stills ?? []).filter((still) => fs.existsSync(still))
    if (!clip && !stills.length) continue
    if (!caps.vision || video.visionUnavailable) return undefined
    if (clip && video.serverTakesVideo) {
      parts.push({ type: 'input_video', input_video: { data: (await readOpen(clip)).toString('base64') } })
    }
    for (const still of stills) parts.push({ type: 'image_url', image_url: { url: await toDataUrl(still) } })
  }
  return parts.length ? [...parts, { type: 'text', text: message.content }] : null
}

/** A conversation's earlier messages, with everything they sent the model at hand for its history. */
async function restoreConversationMedia(chatId: string, messages: AgentMessage[]): Promise<void> {
  // Older attachments are brought into the app's own storage first, so a rebuild reads the kept copy.
  await adoptAttachments(chatId).catch(() => undefined)
  await restoreSent(chatId, messages, reconstructSent)
}

/*
 * A separate MTP head, when the model has one and it can be used.
 *
 * Only the CUDA build carries the patch that loads a head from its own file, and a model with a
 * head inside it needs no other. Charged for the model's full trained context, since the head
 * keeps a layer of cache for whatever context the plan settles on.
 */
function draftReserve(
  model: ModelRecord,
  backend: string,
  batches: { batchSize?: number; ubatchSize?: number } = {}
): { draftModule?: string; draftBytes: number } {
  const head = model.caps.mtpModule
  const arch = model.arch
  if (!head || !arch || backend !== 'cuda' || (arch.mtpLayers ?? 0) > 0) return { draftBytes: 0 }
  const spec = loadSettings().speculative
  if (!spec.enabled || spec.nMax <= 0 || !fs.existsSync(head.path)) return { draftBytes: 0 }
  // The draft context inherits the model's batch size, and its compute buffer grows with it.
  const { ubatch } = resolveBatches(batches.batchSize, batches.ubatchSize)
  return { draftModule: head.path, draftBytes: draftReserveBytes(head, arch, arch.contextLength || 262144, ubatch) }
}

function companionSize(model: ModelRecord): number {
  if (!model.caps.mmprojPath) return 0
  try {
    return fs.statSync(model.caps.mmprojPath).size
  } catch {
    return 0
  }
}

/** Load a model by id using a plan, or auto-fit one if no plan is supplied. */
/**
 * Placement overrides arrive from the renderer — or from a remote tab — and the planner hands
 * them on to a llama-server command line, so only well-formed values get through.
 */
function sanitizeOverrides(raw: Record<string, unknown> | undefined): FitConstraints['overrides'] {
  const o = raw ?? {}
  const int = (v: unknown): number | undefined =>
    typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.round(v) : undefined
  const kv = (v: unknown): KvType | undefined => (v === 'f16' || v === 'q8_0' || v === 'q4_0' ? v : undefined)
  const out: FitConstraints['overrides'] = {}
  if (int(o.contextLength)) out.contextLength = int(o.contextLength)
  if (int(o.gpuLayers) !== undefined) out.gpuLayers = int(o.gpuLayers)
  if (int(o.cpuMoeLayers) !== undefined) out.cpuMoeLayers = int(o.cpuMoeLayers)
  if (int(o.batchSize)) out.batchSize = int(o.batchSize)
  if (int(o.ubatchSize)) out.ubatchSize = int(o.ubatchSize)
  if (typeof o.flashAttention === 'boolean') out.flashAttention = o.flashAttention
  if (kv(o.kvType)) out.kvType = kv(o.kvType)
  if (kv(o.kvTypeV)) out.kvTypeV = kv(o.kvTypeV)
  if (typeof o.overrideTensors === 'string') {
    // One argv element to llama.cpp, so no control characters and nothing absurdly long.
    const t = o.overrideTensors.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 2000)
    if (t) out.overrideTensors = t
  }
  return out
}

/** The batch sizes set in Settings, as planner overrides; automatic ones are left out. */
function batchOverrides(s: AppSettings): FitConstraints['overrides'] {
  const out: FitConstraints['overrides'] = {}
  if (s.runtime.batchSize > 0) out.batchSize = s.runtime.batchSize
  if (s.runtime.ubatchSize > 0) out.ubatchSize = s.runtime.ubatchSize
  return out
}

/** The auto-fit constraints as the user's settings define them. */
function constraintsFromSettings(s: AppSettings): FitConstraints {
  return {
    ...DEFAULT_CONSTRAINTS,
    minKvType: s.autoFit.minKvType,
    preferredKvType: s.autoFit.preferredKvType,
    targetContext: s.autoFit.targetContext,
    idealContext: s.autoFit.idealContext,
    headroomBytes: s.autoFit.headroomMb * 1024 * 1024,
    expertCacheSlots: s.runtime.expertCache,
    allowRopeScaling: s.autoFit.allowRopeScaling,
    overrides: batchOverrides(s)
  }
}

/** What a load's extras depend on, for matching them to the last load's measurement. */
function extrasFeatures(model: ModelRecord, backend: Backend, overrides: FitConstraints['overrides']): ExtrasFeatures {
  return {
    mmproj: model.caps.mmprojPath ?? null,
    draft: draftReserve(model, backend, overrides).draftModule ?? null,
    ubatch: resolveBatches(overrides.batchSize, overrides.ubatchSize).ubatch
  }
}

/**
 * Everything the planner needs to plan this model on this machine now.
 *
 * Settings, the load dialog's own choices on top, the cache pairings the build is fast at, and what
 * the model's last load measured outside llama.cpp's buffers (falling back to an estimate from the
 * projector and head sizes when there is no such load yet).
 */
async function planConstraints(
  model: ModelRecord,
  hw: HardwareSnapshot,
  extra: FitConstraints['overrides'] = {}
): Promise<FitConstraints> {
  const s = loadSettings()
  const overrides = { ...batchOverrides(s), ...extra }
  const base = constraintsFromSettings(s)
  const arch = model.arch
  const head = model.caps.mtpModule
  const { ubatch } = resolveBatches(overrides.batchSize, overrides.ubatchSize)
  // The head's own cache follows the plan's context; what it costs at a given context, for moving a measurement.
  const draftKvAt = (ctx: number): number =>
    head && arch ? draftReserveBytes(head, arch, ctx, ubatch) - draftReserveBytes(head, arch, 0, ubatch) : 0
  const ceiling = arch?.contextLength && !s.autoFit.allowRopeScaling ? arch.contextLength : Infinity
  const expectedCtx = Math.min(overrides.contextLength ?? Math.max(s.autoFit.targetContext, s.autoFit.idealContext), ceiling)
  return {
    ...base,
    companionBytes: companionSize(model),
    draftBytes: draftReserve(model, hw.backend, overrides).draftBytes,
    extrasPerGpu: learnedExtras(model.id, usableDevices(hw), extrasFeatures(model, hw.backend, overrides), expectedCtx, draftKvAt),
    kvPairs: await kvPairsFor(hw.backend),
    overrides
  }
}

/**
 * The key/value cache pairings the planner may choose on a backend.
 *
 * Vulkan and CPU attention read any pairing directly. A CUDA build only does so for the pairs it
 * compiled, which ggml-cuda.dll names in its build features. When the file says nothing, matched
 * pairs are assumed, because that is what every CUDA build compiles unless told otherwise.
 */
async function kvPairsFor(backend: Backend): Promise<string[] | undefined> {
  if (backend !== 'cuda') return undefined
  const dll = path.join(path.dirname(llamaServerPath('cuda')), 'ggml-cuda.dll')
  return (await cudaKvPairs(dll)) ?? [...MATCHED_KV_PAIRS]
}

/**
 * The next plan to try when one did not fit.
 *
 * Cheapest first: GPU expert-cache slots (each one is a little speed), then routed experts to the
 * host (a token reads ten of five hundred), and whole layers last. Returns null when there is
 * nothing left to give up.
 */
async function moreConservative(plan: FitPlan, model: ModelRecord, hw: HardwareSnapshot): Promise<FitPlan | null> {
  const arch = model.arch
  if (!arch) return null
  // The batch sizes too: a retry that fell back to 512 would change the load's speed as well as its placement.
  const keep = {
    contextLength: plan.contextLength,
    kvType: plan.kvType,
    kvTypeV: plan.kvTypeV,
    batchSize: plan.batchSize,
    ...(plan.ubatchSize ? { ubatchSize: plan.ubatchSize } : {})
  }
  const take = (result: FitResult): FitPlan | null => {
    const next = result.chosen ?? result.alternatives[0]
    return next ? { ...next, draftMax: plan.draftMax, draftModule: plan.draftModule } : null
  }
  const replan = async (overrides: FitConstraints['overrides'], slots?: number): Promise<FitPlan | null> => {
    const constraints = await planConstraints(model, hw, overrides)
    return take(await planModel({ model, hw, constraints: slots === undefined ? constraints : { ...constraints, expertCacheSlots: slots }, measure: false }))
  }

  const slots = plan.expertCacheSlots ?? 0
  if (slots > 16) {
    const next = await replan({ ...keep, cpuMoeLayers: plan.cpuMoeLayers ?? arch.blockCount }, Math.floor(slots * 0.7))
    if (next) return next
  }

  const cpuMoe = plan.cpuMoeLayers ?? 0
  if (arch.expertCount > 0 && cpuMoe < arch.blockCount) {
    // A tenth of the stack at a time: small enough not to give away speed needlessly, large
    // enough that three retries cover a badly wrong estimate rather than inching towards it.
    const step = Math.max(2, Math.round(arch.blockCount / 10))
    const next = await replan({ ...keep, cpuMoeLayers: Math.min(arch.blockCount, cpuMoe + step) })
    if (next) return next
  }

  const layers = Math.floor(plan.gpuLayers * 0.9)
  if (layers > 0 && layers < plan.gpuLayers) {
    return replan({ ...keep, gpuLayers: layers, ...(arch.expertCount > 0 ? { cpuMoeLayers: arch.blockCount } : {}) }, 0)
  }
  return null
}

/**
 * Load, and give ground rather than give up when the card turns out to be fuller than predicted.
 *
 * The plan is an estimate of what llama.cpp will allocate, and on an architecture the engine
 * models imperfectly it can be a few hundred megabytes optimistic on one device — which surfaced
 * as a failed load and nothing else. A Qwen3.8-Flash-Next plan sized at 12.3 GB against 13.4 GB
 * free died on a 103 MiB recurrent-state cache, twice, with no way forward from the UI.
 */
async function loadWithBackoff(
  model: ModelRecord,
  plan: FitPlan,
  hw: HardwareSnapshot
): Promise<{ loaded: Awaited<ReturnType<typeof llama.load>>; plan: FitPlan }> {
  const arch = model.arch
  let current = plan
  for (let attempt = 0; ; attempt++) {
    try {
      return { loaded: await llama.load(model, current, hw.backend), plan: current }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      const outOfVram = /out of memory|failed to allocate|unable to allocate/i.test(message)
      const next = outOfVram && arch && attempt < 3 ? await moreConservative(current, model, hw) : null
      if (!next) throw err
      logger.warn('model', `${model.filename}: load ran out of VRAM, retrying with less on the GPUs`, {
        attempt: attempt + 1,
        from: { slots: current.expertCacheSlots ?? 0, cpuMoeLayers: current.cpuMoeLayers ?? 0, gpuLayers: current.gpuLayers, context: current.contextLength },
        to: { slots: next.expertCacheSlots ?? 0, cpuMoeLayers: next.cpuMoeLayers ?? 0, gpuLayers: next.gpuLayers, context: next.contextLength }
      })
      current = next
    }
  }
}

async function loadModelById(modelId: string, plan?: FitPlan): Promise<{ port: number; plan: FitPlan }> {
  const model = library.find((m) => m.id === modelId || m.filename === modelId)
  if (!model) throw new Error(`Model not found: ${modelId}`)
  if (!model.arch) throw new Error(model.error ?? 'Model metadata could not be parsed')

  // Loading replaces whatever is loaded. What the conversation held had of it is kept only if it was saved.
  await llama.preemptIdle()

  /*
   * The compaction helper goes before anything is measured.
   *
   * It sits in memory the outgoing model left free and was placed against that model. Left
   * running, it would count against the incoming model's free VRAM — a smaller context or more
   * experts in system RAM, to keep a helper that is not in use.
   */
  await summarizer.stop()

  /*
   * The loaded model goes before anything is measured, when a new plan is to be made.
   *
   * Its VRAM was counted as taken: loading a second model planned it into what the first one left,
   * then unloaded the first and loaded the second into a plan sized for a fraction of the cards.
   */
  if (!plan && llama.loaded) await llama.unload()

  const hw = await settledHardware()
  const fresh = await refreshFreeVram(hw)
  hardware = fresh

  let chosen = plan
  if (!chosen) {
    const result = await planModel({ model, hw: fresh, constraints: await planConstraints(model, fresh), measure: true })
    chosen = result.chosen ?? result.alternatives[0]
    if (!chosen) throw new Error('No workable configuration was found for this model on this hardware.')
  }

  /*
   * How far the model may draft ahead of itself, attached once the plan is settled.
   *
   * Only meaningful where the model carries an MTP head; llama.ts checks that rather than
   * trusting this, so a plan reused against a different model cannot ask for something it
   * cannot do.
   */
  const spec = loadSettings().speculative
  const draft = draftReserve(model, fresh.backend, { batchSize: chosen.batchSize, ubatchSize: chosen.ubatchSize })
  chosen = { ...chosen, draftMax: spec.enabled ? spec.nMax : 0, draftModule: draft.draftModule }

  const attempted = await loadWithBackoff(model, chosen, fresh)
  const loaded = attempted.loaded
  // What actually loaded, which after a backoff is not what was planned — and is what gets
  // verified against real VRAM, stored, and shown.
  chosen = attempted.plan
  logger.info('model', `loaded ${model.filename}`, {
    ctx: chosen.contextLength,
    layers: chosen.gpuLayers,
    cpuMoeLayers: chosen.cpuMoeLayers ?? 0
  })

  /*
   * Verify prediction against reality, card by card, and keep it for the next plan of this model.
   *
   * Measured over the cards the plan covers, in its order. The hardware list also holds devices
   * the backend cannot use (an integrated GPU under CUDA), and pairing by position only lined the
   * two up by luck. The keys are stored with the figures so learnedHeadroomFor can match them to
   * the cards present next time.
   */
  const after = await refreshFreeVram(fresh)
  const planned = usableDevices(fresh)
  const actual = planned.map((g) => {
    const now = after.gpus.find((x) => x.index === g.index)
    return now ? Math.max(0, g.freeVram - now.freeVram) : 0
  })
  /*
   * What the load cost outside llama.cpp's buffers is learned from this, exactly, rather than
   * widening a margin: the cards' loss, less the dry-run figure and the cache. A plan that was not
   * measured keeps the previous measurement, since its extras would absorb the estimate's error.
   */
  const s = loadSettings()
  const learned = extrasFromLoad(
    chosen,
    planned,
    actual,
    extrasFeatures(model, fresh.backend, { ...batchOverrides(s), batchSize: chosen.batchSize, ubatchSize: chosen.ubatchSize })
  )
  const previous = get<{ extras: string | null }>('SELECT extras FROM model_configs WHERE model_id = ?', model.id)?.extras ?? null
  const keptExtras = learned && (learned.exact || !previous) ? JSON.stringify(learned) : previous
  const comparison = describeLoad(chosen, planned, actual)
  logger.info('autofit', comparison ?? 'load matched its plan', {
    predicted: chosen.predictedVramPerGpu,
    actual,
    measured: !!chosen.measured,
    extras: learned?.bytes
  })
  if (comparison) emit('autofit:verified', { suggestion: comparison, predicted: chosen.predictedVramPerGpu, actual })

  run(
    'INSERT OR REPLACE INTO model_configs (model_id, plan, predicted_vram, actual_vram, extras, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
    model.id,
    JSON.stringify(chosen),
    JSON.stringify(chosen.predictedVramPerGpu),
    JSON.stringify({ keys: planned.map(gpuKey), bytes: actual }),
    keptExtras,
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
  // Kept attachment copies are not aged out, but ones no message refers to any more are removed.
  await pruneOrphanAttachments().catch(() => undefined)
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
    maxFps: s.video.maxFps,
    trackScale: s.video.trackScale,
    stillShare: s.video.stillShare,
    dropDuplicates: s.video.dropDuplicates,
    cropStatic: s.video.cropStatic,
    serverTakesVideo: modalities.video,
    visionUnavailable: llama.loaded?.visionUnavailable
  }
}

async function attachmentTurn(
  input: string,
  /** What is sent: the app's kept copies of images and audio, and any other file as attached. */
  files: string[],
  /** The user's own files, which are what the agent should act on if it opens one. */
  originals: string[],
  onStage?: (p: { file: string; stage: string }) => void
): Promise<{ text: string; media: ContentPart[]; prepared: PreparedMedia[] }> {
  const caps = llama.loaded?.model.caps
  if (!caps) return { text: input, media: [], prepared: [] }

  /*
   * A video is named, not shown.
   *
   * It used to be sampled here, before the model had read the question: a fixed budget of frames,
   * shrunk and packed into a clip, chosen without knowing what was being asked. Most of the
   * detail went, and what was kept was a guess. The agent has a tool for looking at a video
   * itself, so attaching one now only says which file it is and how long, and the model looks at
   * the parts it needs, as closely as it needs, exactly as if it had been given the path.
   */
  const videos = originals.filter((f) => classifyAttachment(f) === 'video')
  const sent = files.filter((f) => classifyAttachment(f) !== 'video')

  const built = await buildContent('', sent, caps, await videoContext(), onStage)
  const sections: string[] = []
  const media: ContentPart[] = []

  for (const file of videos) {
    const facts = await videoFacts(file).catch(() => null)
    sections.push(
      facts
        ? `Attached video: ${file}
${describeVideo(file, facts)}. It has not been shown to you. ` +
            'Look at it with watch_video: with no times first for an overview, then with start and end ' +
            'to look closer at whatever matters.'
        : `Attached video: ${file}
It could not be read as a video.`
    )
  }

  for (const part of built.parts) {
    if (part.type === 'text') sections.push(part.text ?? '')
    else media.push(part)
  }

  // Named whether or not they were also sent as media, so the agent can act on the file itself.
  const onDisk = originals.filter((f) => !['doc', 'video'].includes(classifyAttachment(f)))
  if (onDisk.length) {
    sections.push(`Attached files on disk:\n${onDisk.map((f) => `- ${f}`).join('\n')}`)
  }
  if (built.notes.length) sections.push(built.notes.join('\n'))

  return {
    text: sections.length ? `${sections.join('\n\n')}\n\n${input}` : input,
    media,
    prepared: built.media
  }
}

/** Rescan every library folder, then re-apply what the user set on each model and the DB keeps. */
async function rescanLibrary(): Promise<ModelRecord[]> {
  library = await scanLibrary(libraryFolders())
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
}

export const handlers: Record<string, (...args: never[]) => unknown> = {
  // ---- settings
  'settings:get': () => ({ ...loadSettings(), hfToken: getHfToken() ? '***set***' : null }),

  /**
   * What the current video settings would cost on a clip of a given length.
   *
   * Run through the real planner rather than reproduced in the renderer, because the numbers that
   * decide it live here: the loaded model's context window, whether this server can take video at
   * all, and the measured per-frame cost. A settings panel full of sliders whose effect you
   * cannot see until you upload something is a panel of guesses.
   */
  // ---- chat's scratchpad
  'scratchpad:list': (chatId: string) => scratchpad.list(chatId),
  'scratchpad:status': () => {
    const s = loadSettings().scratchpad
    return { ...scratchpad.totals(), location: s.location, folder: s.folder, idleMinutes: idleMinutes() }
  },
  'scratchpad:remove': async (chatId: string, name: string) => {
    const removed = await scratchpad.remove(chatId, name, 'the user deleted it')
    scratchpadChanged()
    return removed
  },
  'scratchpad:choose-folder': async () => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory', 'createDirectory'] })
    return result.canceled ? null : (result.filePaths[0] ?? null)
  },

  'settings:patch': (patch: Partial<AppSettings>) => {
    const helperBefore = JSON.stringify(loadSettings().agent.summarizer)
    const next = patchSettings(patch)
    // A changed helper choice applies at the next compaction, not to one chosen under the old setting.
    if (JSON.stringify(next.agent.summarizer) !== helperBefore) void summarizer.stop()
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
  'library:scan': () => rescanLibrary(),
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
    // Every part of a split model, not only the one llama.cpp is pointed at.
    for (const part of m.parts ?? [m.path]) await fsp.rm(part, { force: true })
    if (m.caps.mmprojPath) await fsp.rm(m.caps.mmprojPath, { force: true }).catch(() => undefined)
    library = library.filter((x) => x.id !== modelId)
    return true
  },
  'library:clean-partials': async () => downloadQueue.cleanPartials(modelsDir()),
  /**
   * Import GGUF files the user already has on disk.
   *
   * Three things matter here, all of them because these files are enormous:
   *   - A file already in the library needs no work at all; the old code copied it onto itself.
   *     That covers subfolders of the models folder and the extra folders, not only its top level.
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
    const folders = libraryFolders()

    const imported: string[] = []
    const skipped: string[] = []
    const failed: { file: string; error: string }[] = []
    let linked = 0

    for (const src of result.filePaths) {
      const name = path.basename(src)
      try {
        // Already in the library: nothing to do but rescan.
        if (folders.some((dir) => isInside(src, dir))) {
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

    await rescanLibrary()
    emit('library:import-progress', { file: '', phase: 'done' })
    return { imported, skipped, failed, linked }
  },
  /**
   * Add a folder whose models are listed and loaded where they are.
   *
   * For models kept outside the app's folder, on a drive set aside for them, say, where Import
   * would copy each one, often onto a drive with no room for a second 100 GB copy. Nothing is
   * written into the folder: downloads and imports still go to the models folder.
   */
  'library:add-folder': async (): Promise<{ added: string | null; coveredBy?: string; folders: string[] }> => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
    const current = loadSettings().extraModelDirs
    if (result.canceled || !result.filePaths[0]) return { added: null, folders: current }
    const dir = path.resolve(result.filePaths[0])
    // Already scanned, as the models folder, an extra folder, or inside one of them.
    const coveredBy = libraryFolders().find((root) => isInside(dir, root))
    if (coveredBy) return { added: null, coveredBy, folders: current }
    // A folder added earlier that sits inside the new one is now covered by it.
    const next = patchSettings({ extraModelDirs: [...current.filter((d) => !isInside(d, dir)), dir] })
    emit('library:update', await rescanLibrary())
    return { added: dir, folders: next.extraModelDirs }
  },
  /** Stop listing a folder's models. The folder and its files are left exactly as they are. */
  'library:remove-folder': async (dir: string): Promise<string[]> => {
    const next = patchSettings({ extraModelDirs: loadSettings().extraModelDirs.filter((d) => d !== dir) })
    emit('library:update', await rescanLibrary())
    return next.extraModelDirs
  },

  // ---- auto-fit
  /**
   * A plan for a model on this machine as it is now.
   *
   * `measure` confirms it with llama.cpp's own dry run: a few seconds the first time a configuration
   * is seen, nothing after. The library's badges leave it off and get the estimate, corrected by
   * whatever has been measured for that model before; the load dialog asks for it.
   */
  'autofit:plan': async (
    modelId: string,
    overrides?: Record<string, unknown>,
    measure?: boolean
  ): Promise<FitResult | { error: string }> => {
    const model = library.find((m) => m.id === modelId)
    if (!model) return { error: 'Model not found' }
    if (!model.arch) return { error: model.error ?? 'Model metadata could not be parsed' }

    const hw = await getHardware()
    const measuredNow = await refreshFreeVram(hw)
    hardware = measuredNow
    // Whatever is loaded gives its VRAM back before this plan would load.
    const fresh = withLoadedReleased(measuredNow)

    // The load dialog's own placement choices sit on top of the batch sizes from Settings.
    const result = await planModel({
      model,
      hw: fresh,
      constraints: await planConstraints(model, fresh, sanitizeOverrides(overrides)),
      measure: !!measure
    })
    /*
     * A plan made while detection is still a guess says so.
     *
     * Otherwise a CPU-only plan on a machine whose GPUs have not answered yet reads as the app's
     * verdict on that machine. The library asks again once detection settles.
     */
    const detection = fresh.detection
    return detection?.state === 'provisional'
      ? {
          ...result,
          notes: [
            `GPU detection has not finished (${detection.reason ?? 'no answer yet'}), so this plan may leave GPUs out. Detection is retried automatically.`,
            ...result.notes
          ]
        }
      : result
  },

  // ---- model runtime
  'model:load': (modelId: string, plan?: FitPlan) => loadModelById(modelId, plan),
  'model:sampling-defaults': () => llama.samplingDefaults(),
  'model:unload': async () => {
    await llama.preemptIdle()
    await llama.unload()
    // The helper was placed in what this model left free; it is placed again against the next one.
    await summarizer.stop()
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
    const variants = groupVariants(files)
    const hw = await getHardware()
    const s = loadSettings()
    // No header peeks here: this answers as soon as the file tree is known, and a peek per
    // variant is a range request per variant. `hf:variant-info` fills in MoE-awareness (and the
    // recommendation improves with it) once the renderer asks, a few at a time.
    return { files, variants, recommendation: recommendQuant(variants, hw, s.autoFit.targetContext) }
  },
  /**
   * A variant's own architecture facts, read from its header rather than assumed — specifically
   * whether it is a mixture-of-experts model (worth recommending even past free VRAM, since its
   * routed experts can live in system RAM) and whether it carries its own speculative-decoding
   * head. Split out from `hf:files` so opening a repo never blocks on N range requests before the
   * list of variants appears at all.
   */
  'hf:variant-info': async (repo: string, variantId: string) => {
    const files = await listFiles(repo, getHfToken())
    const variant = groupVariants(files).find((v) => v.id === variantId)
    if (!variant) throw new Error(`${variantId} not found in ${repo}`)
    return peekVariantArch(variant, getHfToken())
  },
  'hf:download': async (repo: string, variantId: string, options?: { skipProjector?: boolean }) => {
    const files = await listFiles(repo, getHfToken())
    const variant = groupVariants(files).find((v) => v.id === variantId)
    if (!variant) throw new Error(`${variantId} not found in ${repo}`)
    if (!variant.complete) {
      throw new Error(
        `Part${variant.missing.length === 1 ? '' : 's'} ${variant.missing.join(', ')} of ${variant.parts.length + variant.missing.length} ` +
          `${variant.missing.length === 1 ? 'is' : 'are'} missing from ${repo} — the upload may still be in progress.`
      )
    }

    const dir = path.join(modelsDir(), repo.replace('/', '__'))
    downloadQueue.setToken(getHfToken())
    // Every part of the variant, not just the one a caller happened to click — this is the fix
    // for "part 1 of 8" being its own downloadable-looking row with no indication that seven more
    // were needed before the model would actually load.
    const queued = variant.parts.map((file) =>
      downloadQueue.enqueue({
        repo,
        filename: path.basename(file.filename),
        url: file.url,
        dest: path.join(dir, path.basename(file.filename)),
        bytesTotal: file.bytes,
        sha256: file.sha256
      })
    )

    // Multimodal models are useless without their projector, so fetch it automatically.
    // Unless the caller has no use for one: a compaction helper only ever reads text.
    const mmproj = options?.skipProjector ? null : findMmprojFor(files)
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
  'chat:load': async (id: string) => {
    // Locked: what it is called and that it is locked, nothing more, until it is unlocked.
    if (chats.chatLocked(id)) {
      const locked: AgentSessionState = {
        id,
        title: chats.chatTitle(id) ?? 'Locked chat',
        cwd: '',
        planMode: false,
        messages: [],
        taskList: [],
        createdAt: 0,
        updatedAt: 0,
        locked: true
      }
      return locked
    }
    // Attachments from before copies were kept are brought in the first time a conversation opens.
    await adoptAttachments(id).catch(() => undefined)
    const stored = chats.loadSession(id)
    /*
     * An agent turn in progress is stored when it ends, so mid-turn the database is missing
     * everything the turn has said so far. Opening the conversation then (coming back from
     * another view, say) showed the tool calls, which come from elsewhere, but not the thinking
     * and words between them, until the turn finished. The live session has them.
     */
    const live = liveAgentSessions.get(id)
    return stored && live ? { ...stored, messages: [...live.messages] } : stored
  },
  'chat:rename': (id: string, title: string) => {
    chats.renameChat(id, title)
    return true
  },
  'chat:delete': async (id: string) => {
    chats.deleteChat(id)
    // Its saved state is gigabytes nothing will ever load again.
    const stateDir = llama.stateStore?.dir ?? stateCacheFolder()
    if (stateDir) for (const key of [`chat:${id}`, `agent:${id}`]) removeState(stateDir, key)
    // Messages, tasks and attachments cascade out of the database; a session's file snapshots
    // live on disk and were left behind, holding a full copy of everything the agent had edited.
    await discardCheckpoints(id)
    // Its kept attachment copies go with it; nothing else refers to them.
    await deleteConversationFiles(id)
    // And whatever its scratchpad still held, with the frames its model was shown.
    await scratchpad.clearChat(id)
    chatHeldMedia.delete(id)
    for (const key of [`chat:${id}`, `agent:${id}`]) recallVectors.delete(key)
    scratchpadChanged()
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
    const title = chats.chatTitle(id) ?? ''
    const result = await dialog.showSaveDialog({
      title: 'Export conversation',
      defaultPath: `${exportFilename(title, `chat-${id.slice(0, 8)}`)}.${format}`,
      filters: [format === 'md' ? { name: 'Markdown', extensions: ['md'] } : { name: 'JSON', extensions: ['json'] }]
    })
    if (result.canceled || !result.filePath) return null
    return offerReveal(writeExport(id, format, result.filePath))
  },
  'chat:export-text': (id: string, format: 'md' | 'json') => exportChat(id, format),

  /*
   * Protection, for conversations (chat and agent alike) and document collections; see protect/.
   *
   * A conversation is changed only while nothing is using it. Turning protection on or off removes
   * its saved state, written the old way, to be saved again the new way at the next quiet moment.
   * Locking it also makes the server let go of it, saving it encrypted first.
   */
  'protect:status': (kind: OwnerKind, id: string) => statusOf(ownerOf(kind, id)),
  'protect:available': () => osKeyStoreAvailable(),
  'protect:on': async (kind: OwnerKind, id: string, password?: string) => {
    const owner = ownerOf(kind, id)
    refuseWhileBusy(owner)
    const result = await protection.protect(owner, password || undefined)
    dropSavedStates(owner)
    protectChanged(owner)
    return result
  },
  'protect:off': async (kind: OwnerKind, id: string) => {
    const owner = ownerOf(kind, id)
    refuseWhileBusy(owner)
    await protection.unprotect(owner)
    dropSavedStates(owner)
    protectChanged(owner)
    return true
  },
  'protect:set-password': async (kind: OwnerKind, id: string, password: string) => {
    const owner = ownerOf(kind, id)
    // A first password re-encrypts everything under a new key, so nothing may be writing to it.
    refuseWhileBusy(owner)
    const result = await protection.setPassword(owner, String(password ?? ''))
    // Saved under the key it had; saved again under the new one at the next quiet moment.
    dropSavedStates(owner)
    protectChanged(owner)
    return result
  },
  'protect:remove-password': (kind: OwnerKind, id: string) => {
    const owner = ownerOf(kind, id)
    protection.removePassword(owner)
    protectChanged(owner)
    return true
  },
  'protect:unlock': async (kind: OwnerKind, id: string, secret: string) => {
    const owner = ownerOf(kind, id)
    const ok = await protection.unlock(owner, String(secret ?? ''))
    if (ok) protectChanged(owner)
    return ok
  },
  'protect:lock': async (kind: OwnerKind, id: string) => {
    const owner = ownerOf(kind, id)
    if (!recordOf(owner)?.pw) throw new Error('Only a conversation with a password can be locked.')
    refuseWhileBusy(owner)
    if (owner.kind === 'chat') await letGoOf(owner.id)
    protection.lock(owner)
    protectChanged(owner)
    return true
  },

  /** Plain chat turn (no tools), streaming deltas to the UI. */
  'chat:send': async (
    chatId: string,
    text: string,
    attachments?: string[],
    collectionId?: string,
    reasoning?: ReasoningChoice
  ) => {
    // Anything being done ahead of this message stops here, before the conversation is read.
    await llama.preemptIdle()
    const loaded = llama.loaded
    if (!loaded) throw new Error('No model is loaded')

    // A choice the loaded model cannot express is no choice — see sendableChoice. Narrowed here
    // rather than trusted, because the API server and a remote browser send this field too.
    const effort = sendableChoice(loaded.model.caps.reasoning, reasoning ?? null)

    const history = chats.loadMessages(chatId)
    /*
     * Nothing is summarised unasked. A window that is already full is said so, with what to do,
     * before the message is stored, and before the server refuses it with an error that says less.
     */
    if ((chats.getContextUsed(chatId) ?? 0) + estimateTokens(text) > loaded.plan.contextLength * 0.97) {
      throw new Error('The context window is full. Press Compact to summarise the older messages, then send this again.')
    }
    /*
     * Rebuilt from storage every turn, so this is the only way past reasoning can come back.
     *
     * With the setting on, each earlier answer carries the thinking that produced it, and
     * `--reasoning-preserve` keeps the template from dropping all but the latest. The messages are
     * built once any compaction has run — see chatPromptHistory.
     */
    const preserveReasoning = loadSettings().reasoning.preserve

    let userContent: string | Awaited<ReturnType<typeof buildContent>>['parts'] = text
    const notes: string[] = []
    /*
     * What each file became, keyed by the path it came in as.
     *
     * Hoisted out of the branch because the attachment rows are written further down, and a
     * video's sampled clip has to be recorded against the row or the transcript can only ever
     * describe the optimisation rather than show it.
     */
    let prepared = new Map<string, PreparedMedia>()

    // RAG: retrieved context, placed after the history once compaction has settled what that is.
    let ragContext = ''
    if (collectionId) {
      const hw = await getHardware()
      const hits = await rag.retrieve(text, hw.backend, { collectionId, chatId }, 6)
      ragContext = rag.formatContext(hits)
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

    const userMsg: AgentMessage = {
      id: `${Date.now().toString(36)}-u`,
      role: 'user',
      content: `${text}${attachmentSummary}`,
      createdAt: Date.now()
    }
    chats.appendMessage(chatId, userMsg)
    /*
     * Images and audio are copied into the app's own storage first, and the copy is what is recorded
     * and sent, so a restart, a moved original or a week-old paste can no longer change or lose what
     * the model saw — see attachment-store. Video and documents keep their own paths.
     */
    const sendFiles: string[] = []
    const attachmentIds = new Map<string, string>()
    for (const file of attachments ?? []) {
      const stored = await storeAttachment(chatId, file)
      sendFiles.push(stored)
      try {
        attachmentIds.set(
          stored,
          recordAttachment(
            userMsg.id,
            { path: stored, kind: classifyAttachment(file) },
            stored !== file ? { original: file } : undefined
          )
        )
      } catch (err) {
        // A missing attachment row must never cost the user their message — but it should not
        // vanish without trace either, or the only symptom is an export that quietly lost a file.
        logger.warn('chat', `could not record attachment ${file}`, err)
      }
    }
    chats.autoTitle(chatId)

    /*
     * Announced with its files before any of them are prepared.
     *
     * This event is what the transcript renders from until the conversation is next reopened, and
     * preparing a video is minutes of ffmpeg — sending it afterwards meant the question and the
     * video it was about only appeared once the wait was over.
     */
    const announce = (): void => {
      if (attachments?.length) userMsg.attachments = chats.attachmentsForMessage(userMsg.id)
      emit('chat:message', { chatId, message: userMsg })
    }
    announce()

    /*
     * A video goes into the conversation's scratchpad, and the model looks at it itself.
     *
     * It used to be cut down to a clip and a few stills before the model ever saw the question, by
     * settings that guessed in advance what would matter. Now it is copied into the scratchpad
     * (memory unless Settings say otherwise) and the model is told it is there: it takes an
     * overview with watch_video, looks closer at the moments the question is about, and deletes
     * the copy when it is done. Nothing outside the scratchpad is reachable from chat. A model that
     * cannot call tools is shown one overview instead, and the copy goes at once.
     */
    const videoFiles = sendFiles.filter((f) => classifyAttachment(f) === 'video')
    const otherFiles = sendFiles.filter((f) => classifyAttachment(f) !== 'video')
    const tools = chatToolsFor(chatId, history, videoFiles.length > 0)
    const scratchNotes: string[] = []
    const overview: ContentPart[] = []
    for (const file of videoFiles) {
      const label = path.basename(file)
      if (!loaded.model.caps.videoPossible) {
        notes.push(`${label} skipped: the loaded model has no vision projector, so video cannot be read.`)
        continue
      }
      emit('chat:media-progress', { chatId, file: label, stage: 'putting it in the scratchpad' })
      let item
      try {
        item = await scratchpad.addFile(chatId, file, 'video')
      } catch (err) {
        const why = err instanceof Error ? err.message : String(err)
        notes.push(`${label} could not be put in the scratchpad: ${why}`)
        scratchNotes.push(`[${label} was attached, but it could not be given to you: ${why}]`)
        continue
      }
      scratchpadChanged()
      const source = scratchpad.pathOf(item.id) ?? (await scratchServer.urlFor(item.id))
      if (tools) {
        const facts = await videoFacts(source).catch(() => null)
        scratchNotes.push(
          facts
            ? `Attached video: ${item.name}, now in your scratchpad. ${describeVideo(item.name, facts)}. It has not been shown ` +
                'to you. Look at it with watch_video: with no times first for an overview, then with start and end around ' +
                'the moments that matter. Delete it with scratchpad_delete when you have finished with it.'
            : `Attached video: ${item.name}, now in your scratchpad, but it could not be read as a video.`
        )
      } else {
        emit('chat:media-progress', { chatId, file: label, stage: 'picking frames' })
        const look = await lookAtVideo(
          source,
          item.name,
          { frames: 12 },
          {
            vision: !loaded.visionUnavailable,
            signal: AbortSignal.timeout(180_000),
            again: 'These frames are all of the video you can see: this model cannot call tools to look again.'
          }
        )
        const { text: about, media } = splitOutput(look)
        overview.push(...media)
        scratchNotes.push(`Attached video: ${about}`)
        await scratchpad.remove(chatId, item.name, 'it was shown to you once, as an overview')
        scratchpadChanged()
      }
    }

    if (otherFiles.length) {
      const built = await buildContent(text, otherFiles, loaded.model.caps, await videoContext(), (p) =>
        emit('chat:media-progress', { chatId, ...p })
      )
      userContent = built.parts
      notes.push(...built.notes)
      prepared = new Map(built.media.map((m) => [m.source, m]))
      for (const media of prepared.values()) {
        const id = attachmentIds.get(media.source)
        if (!id) continue
        const kept = await keepPrepared(chatId, media)
        chats.setAttachmentMeta(id, { optimised: kept.optimised, stills: kept.stills, note: media.note })
      }
    }
    if (scratchNotes.length || overview.length) {
      const extra: ContentPart[] = [...overview, { type: 'text', text: scratchNotes.join('\n\n') }]
      userContent = typeof userContent === 'string' ? [{ type: 'text', text: userContent }, ...extra] : [...userContent, ...extra]
    }
    /*
     * Recorded exactly as sent, so a rebuilt history sends the same. An overview's frames are the
     * exception: they stay in memory with the scratchpad they came from, and what is written down
     * in their place says they were there.
     */
    const recordSent = async (content: typeof userContent): Promise<void> => {
      if (!overview.length || typeof content === 'string') {
        await persistSent(chatId, userMsg.id, content)
        return
      }
      holdChatMedia(chatId, userMsg.id, content)
      await persistSent(chatId, userMsg.id, [
        ...content.filter((part) => part.type !== 'image_url'),
        { type: 'text', text: '[Frames of the attached video were shown here. They are not kept once the app closes.]' }
      ])
    }
    if (attachments?.length) {
      await recordSent(userContent)
      if (prepared.size) announce()
      emit('chat:media-progress', { chatId, file: '', stage: '' })
    }
    if (notes.length) emit('chat:notes', notes)

    /*
     * Retrieved excerpts go in the message they were retrieved for, and stay there.
     *
     * They used to be sent as a system message placed just before the question and stored
     * nowhere, so the next turn's prompt lacked them: the server had to go back to before that
     * point and read the last turn or two again, every turn. A system message anywhere but the
     * start is also something several chat templates refuse outright. Recorded with the message,
     * they are sent again in the same place, at the cost of staying in the window.
     */
    if (ragContext) {
      userContent =
        typeof userContent === 'string'
          ? `${ragContext}\n\n[The user's message]\n${userContent}`
          : [{ type: 'text' as const, text: ragContext }, ...userContent]
      await recordSent(userContent)
    }

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

    // What earlier messages sent is loaded first, so the prompt is built with their attachments.
    await restoreConversationMedia(chatId, history)
    const messages = chatMessagesFor(chatId, history, preserveReasoning)
    const generation = loadSettings().generation
    lastEffort.set(`chat:${chatId}`, effort)
    await resumeSaved({ messages, cacheKey: `chat:${chatId}`, ...(tools ? { tools } : {}), ...effortFields(effort) })

    const request: CompletionOptions = {
      messages: [...messages, { role: 'user', content: userContent }],
      cacheKey: `chat:${chatId}`,
      ...(tools ? { tools } : {}),
      // The parameters panel's settings. Ultra keeps them for every sample, stepping only the temperature.
      ...samplingOptions(generation.sampling),
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
    let toolRounds = 0
    try {
      // Ultra's drafts cannot call tools, so a turn with the scratchpad's tools is one model working.
      if (isUltra(effort) && !tools) {
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

      /*
       * One request, or a few when the model uses its scratchpad.
       *
       * A round that ends in tool calls has its words and thinking kept as a message of their own,
       * then each call is run against this conversation's scratchpad and its result recorded, and
       * the model is asked again with them, until it answers in words. Stored the way the agent
       * stores its steps, so a rebuilt prompt is the one this turn sent.
       */
      for (;;) {
        answer = ''
        thinking = ''
        const calls: ToolCall[] = []
        const batch = crypto.randomBytes(4).toString('hex')
        for await (const ev of llama.streamEvents(request)) {
          if (ev.type === 'text') {
            answer += ev.text
            emit('chat:delta', { chatId, text: ev.text })
          }
          if (ev.type === 'reasoning') {
            thinking += ev.text
            emit('chat:reasoning', { chatId, text: ev.text })
          }
          if (ev.type === 'tool_call_partial') {
            emit('chat:tool-call-partial', { chatId, index: ev.index, name: ev.name, args: ev.args })
          }
          if (ev.type === 'speed') emit('chat:speed', { chatId, tokensPerSecond: ev.tokensPerSecond, tokens: ev.tokens })
          if (ev.type === 'tool_call') {
            calls.push({
              id: ev.call.id,
              name: ev.call.name,
              args: ev.call.args,
              batch,
              ...(ev.call.invalid ? { invalid: ev.call.invalid } : {})
            })
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
        if (!calls.length || !tools) break
        toolRounds++

        if (answer.trim() || thinking.trim()) {
          const said: AgentMessage = {
            id: `${Date.now().toString(36)}-a${toolRounds}`,
            role: 'assistant',
            content: answer.trim(),
            reasoning: thinking || undefined,
            createdAt: Date.now()
          }
          chats.appendMessage(chatId, said)
          emit('chat:message', { chatId, message: said })
        }
        request.messages.push({
          role: 'assistant',
          content: answer.trim(),
          ...(preserveReasoning && thinking ? { reasoning_content: thinking } : {}),
          tool_calls: calls.map((c) => ({
            id: c.id,
            type: 'function' as const,
            function: { name: c.name, arguments: JSON.stringify(c.args) }
          }))
        })
        // Said and kept above; what follows the calls is a new answer.
        answer = ''
        thinking = ''

        const vision = !!loaded.model.caps.vision && !loaded.visionUnavailable
        for (const call of calls) {
          // Every announced call gets an answer, even one the user stopped before it ran.
          if (abort.signal.aborted) {
            request.messages.push({
              role: 'tool',
              content: 'Not run: the user stopped the turn before this call was reached.',
              tool_call_id: call.id,
              name: call.name
            })
            continue
          }
          emit('chat:tool-call', { chatId, call })
          const startedAt = Date.now()
          let out: ToolOutput
          let ok = !call.invalid
          try {
            out = call.invalid
              ? `The arguments for ${call.name} were not valid JSON, so it was not run: ${call.invalid}. Call it again with well-formed arguments.`
              : isRecallTool(call.name)
                ? await runRecall(`chat:${chatId}`, call.name, call.args, abort.signal)
                : await runChatTool(call.name, call.args, {
                  chatId,
                  pad: scratchpad,
                  urlFor: (id) => scratchServer.urlFor(id),
                  vision,
                  signal: abort.signal,
                  idleMinutes: idleMinutes()
                })
          } catch (err) {
            ok = false
            out = `${call.name} failed: ${err instanceof Error ? err.message : String(err)}`
          }
          const { text: resultText, media } = splitOutput(out)
          const result = makeResult(call.id, resultText, call.name, startedAt, ok)
          const toolMsg: AgentMessage = {
            id: crypto.randomBytes(6).toString('hex'),
            role: 'tool',
            content: result.content,
            toolCalls: [call],
            toolResult: result,
            createdAt: Date.now()
          }
          chats.appendMessage(chatId, toolMsg)
          request.messages.push({ role: 'tool', content: result.content, tool_call_id: call.id, name: call.name })
          if (media.length) {
            // Pictures arrive as a turn of their own after the result, as the agent sends them.
            const returned: ContentPart[] = [...media, { type: 'text', text: `[images returned by ${call.name}]` }]
            holdChatMedia(chatId, toolMsg.id, returned)
            request.messages.push({ role: 'user', content: returned })
            emit('chat:tool-frames', {
              chatId,
              messageId: toolMsg.id,
              frames: media.flatMap((part) => (part.type === 'image_url' && part.image_url ? [part.image_url.url] : []))
            })
          }
          emit('chat:message', { chatId, message: toolMsg })
          scratchpadChanged()
        }

        if (abort.signal.aborted) {
          stopped = true
          break
        }
        if (toolRounds >= CHAT_TOOL_ROUNDS) {
          answer = `_[stopped after ${CHAT_TOOL_ROUNDS} rounds of tool calls]_`
          break
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
      llama.missTurnSave(`chat:${chatId}`)
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
    if (!stopped) afterChatTurn(chatId, effort)
    return assistantMsg
  },

  // ---- agent
  'agent:tools': () => {
    const a = getAgent()
    syncAgentOptions()
    return a.listTools()
  },
  'agent:run': async (sessionId: string, input: string, reasoning?: ReasoningChoice, attachments?: string[]) => {
    // Anything being done ahead of this message stops here, before the session is read.
    await llama.preemptIdle()
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)
    liveAgentSessions.set(sessionId, session)

    /*
     * Text goes into the prompt, media goes to the model as content parts.
     *
     * Only the first user message of the turn carries parts; everything the loop appends after
     * it — assistant turns, tool results — stays plain text, which is the arrangement llama.cpp
     * expects and the reason the loop can feed its history back unchanged each iteration.
     */
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
    // Agent sessions never said which model ran them; a chat always has.
    if (llama.loaded) chats.setChatModel(sessionId, llama.loaded.model.id)

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

    /*
     * Announced before the files are prepared, not after.
     *
     * Sampling a video is minutes of ffmpeg on anything long or awkwardly encoded, and all of it
     * used to happen before the turn was announced — so the transcript showed a spinner with
     * nothing above it and no indication that anything was happening, for the whole wait.
     *
     * The message row is written first because foreign keys are on and an attachment naming a
     * message that does not exist is rejected. The loop stores this same id later and that write
     * is an upsert, so the row is updated in place and the attachments survive it.
     */
    chats.appendMessage(sessionId, {
      id: userMessageId,
      role: 'user',
      content: displayText,
      createdAt: Date.now()
    })

    // Kept copies first, recorded and sent in place of the originals — the same as the chat path.
    const sendFiles: string[] = []
    const attachmentIds = new Map<string, string>()
    for (const file of attachments ?? []) {
      const stored = await storeAttachment(sessionId, file)
      sendFiles.push(stored)
      try {
        attachmentIds.set(
          stored,
          recordAttachment(
            userMessageId,
            { path: stored, kind: classifyAttachment(file) },
            stored !== file ? { original: file } : undefined
          )
        )
      } catch (err) {
        logger.warn('agent', `could not record attachment ${file}`, err)
      }
    }

    const announce = (): void => {
      emit('agent:message', {
        sessionId,
        message: {
          id: userMessageId,
          role: 'user',
          content: displayText,
          createdAt: Date.now(),
          attachments: attachments?.length ? chats.attachmentsForMessage(userMessageId) : undefined
        }
      })
    }
    announce()

    /*
     * The slow part, now that the turn is on screen.
     *
     * Sampling a video is several ffmpeg passes over the source; on a long or awkwardly encoded
     * file that is minutes, and it all used to happen above this point with nothing shown.
     */
    const turn = attachments?.length
      ? await attachmentTurn(input, sendFiles, attachments, (p) => emit('agent:media-progress', { sessionId, ...p }))
      : { text: input, media: [] as ContentPart[], prepared: [] as PreparedMedia[] }

    /*
     * Re-announced now that the sampler has finished, which is what puts the toggle on the
     * player: until this point the row knows the file but not the clip that was made from it.
     */
    emit('agent:media-progress', { sessionId, file: '', stage: '' })
    if (turn.prepared.length) {
      for (const media of turn.prepared) {
        const id = attachmentIds.get(media.source)
        if (!id) continue
        // Out of the weekly-swept scratch folder before the row points at them.
        const kept = await keepPrepared(sessionId, media)
        chats.setAttachmentMeta(id, { optimised: kept.optimised, stills: kept.stills, note: media.note })
      }
      announce()
    }

    const before = session.messages.length
    activeAgentSessionId = sessionId
    const turnAbort = new AbortController()
    agentTurnAbort = turnAbort
    // Every turn rebuilds the history from storage, which keeps only text; the images come from here.
    await restoreConversationMedia(session.id, session.messages)
    a.hydrate(session)
    lastEffort.set(`agent:${sessionId}`, effort)
    await resumeSaved({
      messages: a.buildHistory(session),
      tools: a.toolsFor(session),
      cacheKey: `agent:${sessionId}`,
      ...effortFields(effort)
    })
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
      if (liveAgentSessions.get(sessionId) === session) liveAgentSessions.delete(sessionId)
      llama.missTurnSave(`agent:${sessionId}`)
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
  /**
   * Compact a chat now, whatever it currently costs.
   *
   * The same pass a turn runs when the window is nearly full, on demand — so a long conversation
   * can be made cheaper before the next question rather than while that question waits.
   */
  'chat:compact': async (chatId: string): Promise<ChatCompactionReport> => {
    // The model holds one conversation at a time; summarising another would turn out the one being answered.
    if (agentTurnAbort || inFlightChats.size > 0) {
      return { ok: false, message: 'A reply is still being written — stop it or let it finish first.' }
    }
    await llama.preemptIdle()
    const history = chats.loadMessages(chatId)
    // Loaded first, so the summary covers what attachments said rather than only their names.
    await restoreConversationMedia(chatId, history)
    return compactChat(chatId, history, {
      preserveReasoning: loadSettings().reasoning.preserve,
      reasoning: lastEffort.get(`chat:${chatId}`) ?? null
    })
  },
  /** What a quick compact would do to a conversation: sizes before and after, and how. Changes nothing. */
  'compact:preview': (key: string) => quickPlanFor(String(key ?? '')),
  /**
   * Quick compact: the older part of the conversation is sent in a much shorter form from now on,
   * worked out by rules with no model involved (see chat/quick-compact). Nothing is removed from the
   * transcript, and the recall tools reach everything it leaves out.
   */
  'compact:quick': async (key: string, level?: number) => {
    const k = String(key ?? '')
    if (agentTurnAbort || inFlightChats.size > 0) {
      return { ok: false, message: 'A reply is still being written — stop it or let it finish first.' }
    }
    await llama.preemptIdle()
    const plan = await quickPlanFor(k)
    if (!plan.ok || !('options' in plan)) return plan
    const chosen = plan.options.find((o) => o.level === Number(level)) ?? plan.options.find((o) => o.level === plan.recommended)
    if (!chosen) return { ok: false, message: 'There is no such strength.' }
    const id = k.slice(k.indexOf(':') + 1)
    chats.setQuick(id, chosen.spec)
    const done = { ok: true, message: chosen.name, beforeTokens: plan.beforeTokens, afterTokens: chosen.afterTokens }
    const max = llama.loaded?.plan.contextLength ?? 0
    if (k.startsWith('agent:')) {
      // The instructions and tool list are taken afresh at the next message: the tools now include
      // recall, and what the model had been told about the session stood in messages now shortened.
      getAgent().refreshPrompt(id)
      if (max > 0) emit('agent:context', { sessionId: id, used: done.afterTokens, max })
      setContextUsed(done.afterTokens)
    } else if (max > 0) {
      emit('chat:context', { chatId: id, used: done.afterTokens, max })
    }
    chats.setContextUsed(id, done.afterTokens)
    // What was saved of it held the conversation as it was sent before; the edges say so.
    emit('state:removed', { key: k })
    logger.info('compaction', `quick compact of ${k}`, { level: chosen.level, before: plan.beforeTokens, after: done.afterTokens })
    return done
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
  /**
   * Read a conversation into the model ahead of its next message.
   *
   * Called when a conversation's message box is clicked. Reading starts then rather than at
   * Send, so the time spent typing is time the model spends catching up: after a restart, a
   * model reload, or another conversation having had the model in between.
   */
  'chat:warm': async (chatId: string, reasoning?: ReasoningChoice) => {
    const loaded = llama.loaded
    if (!loaded || !chatId || !canWarm()) return false
    const history = chats.loadMessages(chatId)
    if (!history.length) return false
    await restoreConversationMedia(chatId, history)
    if (!canWarm()) return false
    lastEffort.set(`chat:${chatId}`, reasoning ?? null)
    const tools = chatToolsFor(chatId, history)
    return warm({
      messages: chatMessagesFor(chatId, history, loadSettings().reasoning.preserve),
      cacheKey: `chat:${chatId}`,
      ...(tools ? { tools } : {}),
      ...reasoningRequestFields(loaded.model.caps.reasoning, sendableChoice(loaded.model.caps.reasoning, reasoning ?? null))
    })
  },
  /**
   * The same for an agent session. With no session yet, what is read is the part every session
   * starts with, the system prompt and the tools, which is most of a first message's wait.
   */
  'agent:warm': async (sessionId?: string, reasoning?: ReasoningChoice) => {
    const loaded = llama.loaded
    if (!loaded || !canWarm()) return false
    const session: AgentSessionState = (sessionId ? chats.loadSession(sessionId) : null) ?? {
      id: 'draft',
      title: '',
      cwd: os.homedir(),
      planMode: false,
      messages: [],
      taskList: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }
    const a = getAgent()
    syncAgentOptions()
    if (session.messages.length) await restoreConversationMedia(session.id, session.messages)
    if (!canWarm()) return false
    lastEffort.set(`agent:${session.id}`, reasoning ?? null)
    return warm({
      messages: a.buildHistory(session),
      tools: a.toolsFor(session),
      cacheKey: `agent:${session.id}`,
      ...reasoningRequestFields(loaded.model.caps.reasoning, sendableChoice(loaded.model.caps.reasoning, reasoning ?? null))
    })
  },
  /**
   * Give a session the current system prompt and tools in place of the ones it started with.
   *
   * A session keeps what it first sent so the model never has to read it again; this is how it
   * picks up a tool added since, and it costs one full reading on the next message.
   */
  /** What is kept on disk of conversations the model has read, for the settings page and the pin. */
  'state:info': () => {
    const settings = loadSettings().runtime.stateCache
    const dir = llama.stateStore?.dir ?? stateCacheFolder()
    const states = dir ? listStates(dir) : []
    return {
      // Saving is always there now, by the Save state button; nothing saves on its own.
      enabled: true,
      active: !!llama.stateStore,
      modelLoaded: !!llama.loaded,
      folder: dir ?? '',
      budgetGb: settings?.budgetGb ?? 20,
      usedBytes: states.reduce((n, s) => n + s.bytes, 0),
      count: states.length,
      saved: states.map((s) => s.key)
    }
  },
  /**
   * Which of a conversation's messages its saved state still holds as they are: the last one it
   * covers, or null when nothing is saved that would load. A save made under another model is kept
   * but covers nothing while that model is not the one loaded. A save that an edit inside it has
   * made useless (a model with recurrent layers cannot be cut back) is removed here.
   */
  'state:coverage': (key: string) => {
    const k = String(key ?? '')
    const dir = llama.stateStore?.dir ?? stateCacheFolder()
    if (!dir || !/^(chat|agent):[\w-]+$/.test(k) || stateSeal(k) === 'locked') return null
    const meta = readMeta(dir, k)
    if (!meta) return null
    const store = llama.stateStore
    if (store && meta.fingerprint !== store.fingerprint) return { through: null, otherModel: true, tokens: meta.tokens, savedAt: meta.savedAt }
    const messages = conversationMessages(k)
    const id = k.slice(k.indexOf(':') + 1)
    const { count, dropped } = coverage(meta, stateMarks(k, messages), viewKey(id))
    let gone = dropped
    /*
     * Saved part way through a reply: the reply is the message after the last one covered. Still
     * being written, it is the reply on screen (no id yet); finished, it must still begin with
     * what the save holds of it.
     */
    let partial: { id: string | null; content: number; reasoning: number } | null = null
    const mark = meta.partial
    if (!gone && mark && count === (meta.messages?.length ?? 0)) {
      const next = messages[count]
      if (!next) partial = { id: null, content: mark.content, reasoning: mark.reasoning }
      else if (next.role === 'assistant' && replyBegins(k, next, mark)) partial = { id: next.id, content: mark.content, reasoning: mark.reasoning }
      else if (!meta.canTrim) gone = true
    }
    if (gone) {
      removeState(dir, k)
      return null
    }
    return { through: count ? messages[count - 1].id : null, partial, tokens: meta.tokens, savedAt: meta.savedAt }
  },
  /** Save a conversation's state now, whatever the app's own timing would have done. */
  'state:save': (key: string) => {
    const k = String(key ?? '')
    const running = savingNow.get(k)
    if (running) return running
    const saving = saveNow(k).finally(() => savingNow.delete(k))
    savingNow.set(k, saving)
    return saving
  },
  /** Remove every saved state. Pins are kept: they say what to keep once there is something again. */
  'state:clear': () => {
    const dir = llama.stateStore?.dir ?? stateCacheFolder()
    if (!dir) return 0
    const states = listStates(dir)
    for (const s of states) removeState(dir, s.key)
    return states.length
  },
  'agent:refresh-prompt': (sessionId: string) => {
    if (agentTurnAbort) return { ok: false, message: 'Wait for the reply to finish, then try again.' }
    if (!sessionId) return { ok: false, message: 'Open a conversation first.' }
    getAgent().refreshPrompt(sessionId)
    return {
      ok: true,
      message: 'This conversation will use the current tools and instructions from its next message. That message reads the whole conversation again once.'
    }
  },
  'agent:compact': async (sessionId?: string) => {
    // The model holds one conversation at a time; summarising another would turn out the one being answered.
    if (agentTurnAbort || inFlightChats.size > 0) {
      return { ok: false, message: 'A reply is still being written — stop it or let it finish first.' }
    }
    await llama.preemptIdle()
    const id = sessionId || activeAgentSessionId
    const session = id ? chats.loadSession(id) : null
    if (!session) {
      return { ok: false, message: 'Open a conversation before compacting it.' }
    }

    const a = getAgent()
    activeAgentSessionId = id
    // The history compaction leaves behind is the one the next turn uses, images included.
    await restoreConversationMedia(session.id, session.messages)
    a.hydrate(session)
    // Loaded from its save first, when the server is not holding it: a file read, not a reading.
    await resumeSaved({
      messages: a.buildHistory(session),
      tools: a.toolsFor(session),
      cacheKey: `agent:${id}`,
      ...effortFields(lastEffort.get(`agent:${id}`) ?? a.options.reasoningChoice)
    })
    let report: Awaited<ReturnType<typeof a.compactNow>>
    try {
      report = await a.compactNow(session)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      logger.warn('compaction', 'agent compaction failed; the session goes on with its full history', { sessionId: id, error: message })
      return { ok: false, message: `Compaction failed: ${message}` }
    }
    // What was saved of it held the conversation as it was; the edges say so.
    if (report.ok) emit('state:removed', { key: `agent:${id}` })

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
  /**
   * Rewrite what the model said, and make the model believe it.
   *
   * Editing a transcript that the model does not share would be theatre: the next turn is built
   * from the rolling history, so a change that only reaches storage is a change the model never
   * sees. Dropping the history forces it to be rebuilt from the edited session, which is the
   * only version that then exists.
   *
   * Reasoning is edited alongside the reply because the two are one message. Passing `undefined`
   * leaves it as it was; passing an empty string removes it, which is how a thinking block gets
   * deleted rather than blanked.
   */
  'agent:edit-message': (sessionId: string, messageId: string, content: string, reasoning?: string) => {
    assertAgentIdle()
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)
    const message = session.messages.find((m) => m.id === messageId)
    if (!message) throw new Error('That message is no longer in this conversation.')

    const edited: AgentMessage = {
      ...message,
      content,
      reasoning: reasoning === undefined ? message.reasoning : reasoning.trim() || undefined
    }
    chats.appendMessage(sessionId, edited)
    getAgent().resetHistory()
    return edited
  },

  /**
   * Rewrite something you said, and have the model answer it again.
   *
   * Unlike `agent:edit-message`, this is not "correct the record" — it is "I meant to ask this
   * instead." Everything that followed the original wording was an answer to words that no
   * longer exist, so it goes with it: the old reply, any tool calls it made, all of it.
   *
   * The new answer is produced by running a turn with `continueTurn: true` on a history that
   * hydrate() ends on the edited user message. That flag's actual job is "don't push another
   * user turn" — the loop only chooses to *resume* generation (`continue_final_message`) when
   * the history it hydrated also *ends* on an assistant turn, which this one does not. So the
   * model is asked a fresh question, not asked to keep talking, and the edited words are asked
   * exactly once rather than twice.
   */
  'agent:rewind-and-regenerate': async (sessionId: string, messageId: string, content: string, reasoning?: ReasoningChoice) => {
    assertAgentIdle()
    await llama.preemptIdle()
    /*
     * Checked before anything is touched, not after.
     *
     * Everything below this line is destructive — the edit is saved and the old reply deleted
     * before the new one exists. Discovering only afterwards that there is nothing to answer with
     * would leave exactly the damage this ordering avoids: the words changed, the old answer
     * gone, and nothing in its place.
     */
    if (!llama.loaded) throw new Error('No model is loaded')

    const before0 = chats.loadSession(sessionId)
    if (!before0) throw new Error(`No session ${sessionId}`)
    const idx = before0.messages.findIndex((m) => m.id === messageId)
    if (idx === -1) throw new Error('That message is no longer in this conversation.')
    const target = before0.messages[idx]
    if (target.role !== 'user') throw new Error('Only your own messages can be rewound like this.')

    /*
     * What the message sent is edited along with what it shows.
     *
     * A message can have sent more than its words: the app's note about the session above them,
     * or files beside them. That record is what a rebuilt history sends, so with only the
     * transcript changed the model was asked the old question again.
     */
    await restoreConversationMedia(sessionId, before0.messages)
    const typed = (s: string): string => s.replace(/\n\n\[Attached: [^\]]*\]$/, '')
    const resent = sentWithText(target.id, typed(target.content), typed(content))
    if (resent !== undefined) await persistSent(sessionId, target.id, resent)

    chats.appendMessage(sessionId, { ...target, content })

    const next = before0.messages[idx + 1]
    if (next) {
      chats.truncateFrom(sessionId, next.id)
      // Same reasoning as agent:rewind: a summary describing turns that no longer exist is
      // worse than none, and rebuilding it costs only the room it was saving.
      chats.clearSummary(sessionId)
    }

    // Reloaded rather than trimmed in memory: the edit and the truncation both happened in
    // storage above, and this is the one place both are reflected without recomputing either.
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)

    const effort = sendableChoice(llama.loaded?.model.caps.reasoning, reasoning ?? null)
    const a = getAgent()
    syncAgentOptions()
    a.updateOptions({ cwd: session.cwd, reasoningChoice: effort })

    activeAgentSessionId = sessionId
    const turnAbort = new AbortController()
    agentTurnAbort = turnAbort
    await restoreConversationMedia(session.id, session.messages)
    a.hydrate(session)
    // Whatever saved state still fits the rewound conversation, checked against the prompt it sends.
    await resumeSaved(
      { messages: a.buildHistory(session), tools: a.toolsFor(session), cacheKey: `agent:${sessionId}`, ...effortFields(effort) },
      true
    )
    drainPendingPermissions()

    const beforeRun = session.messages.length
    try {
      await a.run(session, '', [], '', { continueTurn: true })
    } finally {
      for (const m of session.messages.slice(beforeRun)) chats.appendMessage(sessionId, m)
      persistPermissionRules()
      if (agentTurnAbort === turnAbort) agentTurnAbort = null
    }
    return session
  },

  /**
   * Delete a message, and everything that came after it.
   *
   * Not just that one row: a later message can be an answer to this one, or a tool result this
   * one's call produced, and leaving those behind would show a transcript that no longer makes
   * sense — a reply with no question, a result with no call. `truncateFrom` already deletes a
   * message together with everything ordered after it, which is exactly this.
   *
   * The agent's rolling history is dropped rather than edited in place, same as an edit: the
   * next turn is built fresh from what storage now holds, and a stale summary describing turns
   * that no longer exist is worse than none.
   */
  'agent:delete-message': (sessionId: string, messageId: string) => {
    assertAgentIdle()
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)
    if (!session.messages.some((m) => m.id === messageId)) {
      throw new Error('That message is no longer in this conversation.')
    }
    const removed = chats.truncateFrom(sessionId, messageId)
    chats.clearSummary(sessionId)
    getAgent().resetHistory()
    return { removed }
  },

  /**
   * Carry on from where the last answer stopped.
   *
   * Only meaningful when the transcript ends with the model's own turn: with a user message after
   * it there is nothing to continue, and the ordinary send path is what is wanted.
   */
  'agent:continue': async (sessionId: string, reasoning?: ReasoningChoice) => {
    await llama.preemptIdle()
    const session = chats.loadSession(sessionId)
    if (!session) throw new Error(`No session ${sessionId}`)
    const last = session.messages.at(-1)
    if (!last || last.role === 'user') {
      throw new Error("There is nothing to continue — the last message is not from the model.")
    }

    const effort = sendableChoice(llama.loaded?.model.caps.reasoning, reasoning ?? null)
    const a = getAgent()
    syncAgentOptions()
    a.updateOptions({ cwd: session.cwd, reasoningChoice: effort })

    activeAgentSessionId = sessionId
    const turnAbort = new AbortController()
    agentTurnAbort = turnAbort
    /*
     * Hydrated deliberately, unlike the ordinary send path, which reuses the rolling window.
     * Continuing usually follows an edit, and the point of an edit is that the window is stale.
     */
    // With the images the earlier messages were sent with — without them the model resumes blind.
    await restoreConversationMedia(session.id, session.messages)
    a.hydrate(session)
    // What was saved of it, checked against the Continue prompt itself; it used to load nothing.
    await resumeSaved(
      { messages: a.buildHistory(session), tools: a.toolsFor(session), cacheKey: `agent:${sessionId}`, continueFinal: true, ...effortFields(effort) },
      true
    )
    drainPendingPermissions()

    const before = session.messages.length
    try {
      await a.run(session, '', [], '', { continueTurn: true })
    } finally {
      /*
       * From the resumed message onward, not from the end.
       *
       * A continuation usually adds no message at all: the loop folds the new text into the
       * answer it is continuing, so `messages.length` is unchanged and a slice from `before`
       * would store nothing. Starting one earlier catches the grown message, and re-storing an
       * untouched one costs nothing — appendMessage is an upsert keyed on the id.
       */
      for (const m of session.messages.slice(Math.max(0, before - 1))) chats.appendMessage(sessionId, m)
      persistPermissionRules()
      if (agentTurnAbort === turnAbort) agentTurnAbort = null
    }
    return session
  },

  'agent:rewind': async (sessionId: string, checkpointId: string, messageId?: string) => {
    await llama.preemptIdle()
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
  'credentials:list': () => listLogins(),
  'credentials:save': (input: { name: string; username?: string; password: string; url?: string; notes?: string }) =>
    saveLogin(input),
  'credentials:delete': (name: string) => deleteLogin(name),
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
  await llama.preemptIdle().catch(() => undefined)
  persistPermissionRules()
  killAllJobs()
  mcpManager.closeAll()
  await Promise.allSettled([
    llama.unload(),
    apiServer.stop(),
    remoteWeb.stop(),
    tunnel.stop(),
    closeBrowser(),
    rag.embeddings.stop(),
    summarizer.stop(),
    stopScratchpad()
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
