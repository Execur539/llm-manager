import { useEffect, useState } from 'react'
import type { AppSettings, ModelRecord } from '@shared/types'
import { RECOMMENDED_SUMMARIZER } from '@shared/summarizer'
import { invoke, on, fmtBytes } from '../lib/api'
import ConfirmDialog from '../components/ConfirmDialog'
import NumberField from '../components/NumberField'
import { Toolbar } from '../lib/shell'

interface McpStatus {
  id: string
  name: string
  connected: boolean
  tools: string[]
  error: string | null
}

interface MemoryEntry {
  id: string
  text: string
  updatedAt: number
}

const SECTIONS = [
  { id: 'context', label: 'Context and memory' },
  { id: 'speed', label: 'Speed' },
  { id: 'conversations', label: 'Conversations' },
  { id: 'agent', label: 'Agent' },
  { id: 'scratchpad', label: 'Scratchpad' },
  { id: 'mcp', label: 'MCP servers' },
  { id: 'memory', label: 'Agent memory' },
  { id: 'logins', label: 'Saved logins' },
  { id: 'hf', label: 'Hugging Face' },
  { id: 'folders', label: 'Folders' },
  { id: 'about', label: 'About' }
]

/**
 * Where everything on the page is, kept in view beside it.
 *
 * Settings run to several screens, and a list of where things are beats scrolling to find out.
 * The group under the top of the view is marked as you scroll; a click scrolls to one.
 */
function SettingsNav({ hasLogins }: { hasLogins: boolean }): JSX.Element {
  const sections = SECTIONS.filter((x) => hasLogins || x.id !== 'logins')
  const [active, setActive] = useState(sections[0].id)
  useEffect(() => {
    const root = document.querySelector<HTMLElement>('.main-scroll')
    if (!root) return
    const ids = SECTIONS.filter((x) => hasLogins || x.id !== 'logins').map((x) => x.id)
    const onScroll = (): void => {
      const line = root.getBoundingClientRect().top + 120
      let current = ids[0]
      for (const id of ids) {
        const el = document.getElementById(`settings-${id}`)
        if (el && el.getBoundingClientRect().top <= line) current = id
      }
      // The last groups are short, so their tops never reach the line; the end of the page means the last one.
      if (root.scrollTop > 0 && root.scrollTop + root.clientHeight >= root.scrollHeight - 2) current = ids[ids.length - 1]
      setActive(current)
    }
    onScroll()
    root.addEventListener('scroll', onScroll, { passive: true })
    return () => root.removeEventListener('scroll', onScroll)
  }, [hasLogins])
  return (
    <nav className="settings-nav" aria-label="Settings sections">
      {sections.map((x) => (
        <button
          key={x.id}
          className={`settings-nav-item${active === x.id ? ' active' : ''}`}
          aria-current={active === x.id ? 'true' : undefined}
          onClick={() => document.getElementById(`settings-${x.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
        >
          {x.label}
        </button>
      ))}
    </nav>
  )
}

export default function Settings({
  models = [],
  onModelsChanged
}: {
  models?: ModelRecord[]
  onModelsChanged?: () => Promise<void>
}): JSX.Element {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [missing, setMissing] = useState<string[]>([])
  const [paths, setPaths] = useState<{ exeDir: string; modelsDir: string } | null>(null)
  /** What is kept on disk of conversations the model has read. */
  const [stateInfo, setStateInfo] = useState<{
    active: boolean
    modelLoaded: boolean
    folder: string
    usedBytes: number
    count: number
  } | null>(null)
  useEffect(() => {
    void invoke<typeof stateInfo>('state:info')
      .then(setStateInfo)
      .catch(() => undefined)
  }, [])
  /** What chat's scratchpad holds right now, kept current as files arrive and go. */
  const [scratch, setScratch] = useState<{ files: number; bytes: number } | null>(null)
  useEffect(() => {
    const read = (): void => {
      void invoke<{ files: number; bytes: number }>('scratchpad:status')
        .then(setScratch)
        .catch(() => undefined)
    }
    read()
    return on('scratchpad:changed', read)
  }, [])
  const [mcp, setMcp] = useState<McpStatus[]>([])
  const [memory, setMemory] = useState<MemoryEntry[]>([])
  /** Saved logins, without their passwords. Null where they cannot be managed (a remote session). */
  const [logins, setLogins] = useState<{ name: string; username: string; url: string | null; notes: string | null }[] | null>(null)
  const [newLogin, setNewLogin] = useState({ name: '', username: '', password: '', url: '' })
  const [hfToken, setHfToken] = useState('')
  const [update, setUpdate] = useState<{ available: boolean; latestVersion: string | null; notes: string | null; downloadUrl: string | null; bytes?: number; error?: string; previousFailure?: string } | null>(null)
  const [version, setVersion] = useState('')
  const [info, setInfo] = useState<string | null>(null)
  /** Why the last attempt to install an update did not start, shown where it was pressed. */
  const [updateError, setUpdateError] = useState<string | null>(null)
  const [confirmHardBlocks, setConfirmHardBlocks] = useState(false)
  /** Bytes of the update fetched so far, or null when no download is running. */
  const [downloading, setDownloading] = useState<{ done: number; total: number } | null>(null)
  /** The recommended summariser's download, started from here. */
  const [helperDownload, setHelperDownload] = useState<'idle' | 'queued'>('idle')

  /*
   * The download reports progress; until now nobody listened.
   *
   * Subscribed for the life of the view rather than only while a download runs, so returning to
   * Settings mid-download picks the progress back up instead of showing an idle button for a
   * transfer that is still going.
   */
  useEffect(() => on<{ done: number; total: number }>('update:progress', setDownloading), [])

  // A finished summariser download rescans the library, so the model can be picked straight away.
  useEffect(() => {
    if (helperDownload !== 'queued') return
    return on<{ filename: string; status: string }[]>('downloads:update', (list) => {
      if (list.some((d) => d.filename === RECOMMENDED_SUMMARIZER.file && d.status === 'done')) {
        setHelperDownload('idle')
        void onModelsChanged?.()
      }
    })
  }, [helperDownload, onModelsChanged])

  // New MCP server form
  const [mcpName, setMcpName] = useState('')
  const [mcpCommand, setMcpCommand] = useState('')
  const [mcpArgs, setMcpArgs] = useState('')

  const refresh = async (): Promise<void> => {
    setSettings(await invoke<AppSettings>('settings:get'))
    setMissing(await invoke<string[]>('runtime:missing-binaries'))
    setPaths(await invoke('app:paths'))
    setMcp(await invoke<McpStatus[]>('mcp:list'))
    setMemory(await invoke<MemoryEntry[]>('agent:memory'))
    // Refused over the web interface, where this card is simply not shown.
    setLogins(await invoke<NonNullable<typeof logins>>('credentials:list').catch(() => null))
    setVersion(await invoke<string>('app:version'))
  }

  useEffect(() => {
    void refresh()
    const off = on<McpStatus[]>('mcp:update', setMcp)
    return off
  }, [])

  if (!settings) return <div className="empty">Loading…</div>

  const patch = async (p: Partial<AppSettings>): Promise<void> => {
    setSettings(await invoke<AppSettings>('settings:patch', p))
  }

  // Small models only: one near the size of a chat model would take the time it exists to save.
  const helperChoices = models
    .filter((m) => m.arch && !m.error && m.bytes <= 8 * 1024 ** 3)
    .sort((a, b) => a.bytes - b.bytes)
  const recommendedModel = models.find((m) => m.filename.toLowerCase() === RECOMMENDED_SUMMARIZER.file.toLowerCase())
  const helper = settings.agent.summarizer
  const setHelper = (next: Partial<AppSettings['agent']['summarizer']>): Promise<void> =>
    patch({ agent: { ...settings.agent, summarizer: { ...helper, ...next } } })
  const downloadRecommended = async (): Promise<void> => {
    try {
      await invoke('hf:download', RECOMMENDED_SUMMARIZER.repo, RECOMMENDED_SUMMARIZER.file, { skipProjector: true })
      setHelperDownload('queued')
      setInfo(`Downloading ${RECOMMENDED_SUMMARIZER.label}. Progress is under Find a model; it can be picked here once it finishes.`)
    } catch (err) {
      setInfo(`Could not start the download: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  return (
    <>
      <Toolbar title="Settings" />
      <div className="settings-layout">
        <SettingsNav hasLogins={!!logins} />
        <div className="settings-content">

          {info && <div className="card note">{info}</div>}

          {missing.length > 0 && (
            <div className="card" style={{ borderColor: '#5a4515' }}>
              <div className="card-title">
                Bundled components missing <span className="badge warn">{missing.length}</span>
              </div>
              <div className="dim">
                <div className="mono" style={{ margin: '6px 0' }}>{missing.join(', ')}</div>
                Run <code className="mono">npm run fetch-vendor</code> to download them. Features that need a missing
                component will say so rather than failing silently.
              </div>
            </div>
          )}

          <div className="card settings-group" id="settings-context">
            <h2 className="card-title">Context and memory</h2>
            <p className="settings-lede">
              The engine maximises context, prefers a Q8 KV cache, and never drops below Q4. It sizes against
              <strong> free</strong> VRAM and splits multi-GPU work by real capacity.
            </p>
            <div className="settings-rows">
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Target context</div>
                  <div className="setting-hint">Minimum to aim for</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.autoFit.targetContext}
                    min={512}
                    max={10_000_000}
                    onCommit={(n) => void patch({ autoFit: { ...settings.autoFit, targetContext: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Ideal context</div>
                  <div className="setting-hint">Stop growing here; the model's trained length still caps it</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.autoFit.idealContext}
                    min={512}
                    max={10_000_000}
                    onCommit={(n) => void patch({ autoFit: { ...settings.autoFit, idealContext: n } })}
                  />
                </div>
              </div>
              {/*
                * Off by default, and worth a sentence rather than a bare checkbox.
                *
                * Rope scaling genuinely extends the window, but llama.cpp applies it at every length
                * rather than only past the trained one, so short prompts are affected too — and the
                * long-context evaluations find most models degrading before their advertised limit
                * even without it.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Extend past trained context</div>
                  <div className="setting-hint">
                    Rope scaling, up to 4x — quality past the trained length is not guaranteed
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.autoFit.allowRopeScaling}
                    onChange={(e) =>
                      void patch({ autoFit: { ...settings.autoFit, allowRopeScaling: e.target.checked } })
                    }
                    data-testid="allow-rope-scaling"
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Preferred KV</div>
                  <div className="setting-hint">Aim for this first</div>
                </div>
                <div className="setting-control">
                  <select
                    className="kv-select"
                    value={settings.autoFit.preferredKvType}
                    onChange={(e) => void patch({ autoFit: { ...settings.autoFit, preferredKvType: e.target.value as 'f16' | 'q8_0' | 'q4_0' } })}
                  >
                    <option value="f16">f16 — best quality</option>
                    <option value="q8_0">q8_0 — recommended</option>
                    <option value="q4_0">q4_0 — smallest</option>
                  </select>
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">KV floor</div>
                  <div className="setting-hint">Never go below this</div>
                </div>
                <div className="setting-control">
                  <select
                    className="kv-select"
                    value={settings.autoFit.minKvType}
                    onChange={(e) => void patch({ autoFit: { ...settings.autoFit, minKvType: e.target.value as 'f16' | 'q8_0' | 'q4_0' } })}
                  >
                    <option value="q4_0">q4_0 — smallest</option>
                    <option value="q8_0">q8_0 — recommended</option>
                    <option value="f16">f16 — best quality</option>
                  </select>
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">VRAM headroom</div>
                  <div className="setting-hint">MB left free for the desktop</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.autoFit.headroomMb}
                    min={0}
                    max={65_536}
                    onCommit={(n) => void patch({ autoFit: { ...settings.autoFit, headroomMb: n } })}
                  />
                </div>
              </div>
              {/*
                * Only matters once part of a model lives in system RAM, which for a large
                * mixture-of-experts model is most of it.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Weight loading</div>
                  <div className="setting-hint">Mapped pages can fault to disk mid-answer</div>
                </div>
                <div className="setting-control">
                  <select
                    className="kv-select"
                    value={settings.runtime.loadMode}
                    onChange={(e) =>
                      void patch({ runtime: { ...settings.runtime, loadMode: e.target.value as 'auto' | 'ram' | 'mmap' } })
                    }
                    data-testid="load-mode"
                  >
                    <option value="auto">automatic — load into RAM when it fits</option>
                    <option value="ram">always load into RAM</option>
                    <option value="mmap">always memory-map</option>
                  </select>
                </div>
              </div>
              {/*
                * For mixture-of-experts models whose experts live in system RAM: the most recently used
                * experts are kept on the GPU and computed there. VRAM it takes comes out of the context.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">GPU expert cache</div>
                  <div className="setting-hint">
                    Experts kept on the GPU per layer · -1 = as many as fit · 0 = off · same answers · next load
                  </div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.runtime.expertCache}
                    min={-1}
                    max={512}
                    onCommit={(n) => void patch({ runtime: { ...settings.runtime, expertCache: n } })}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="card settings-group" id="settings-speed">
            <h2 className="card-title">Speed</h2>
            <div className="settings-rows">
              {/*
                * On by default, which is unusual for a performance setting and is the reason it is
                * worth a sentence: the model drafts its own next few tokens and checks them in the
                * same pass, so a wrong guess is discarded rather than accepted. Measured at 2x on a
                * Qwen3.8-27B with identical output. Models without the head simply do not get it.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Draft ahead (speculative decoding)</div>
                  <div className="setting-hint">
                    Faster generation on models with an MTP head, built in or in its own file (Flash-Next), same
                    answers
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.speculative.enabled}
                    onChange={(e) =>
                      void patch({ speculative: { ...settings.speculative, enabled: e.target.checked } })
                    }
                    data-testid="speculative-enabled"
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Tokens drafted ahead</div>
                  <div className="setting-hint">
                    A wrong guess discards everything after it · a head in its own file drafts at most 2
                  </div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.speculative.nMax}
                    min={1}
                    max={16}
                    onCommit={(n) => void patch({ speculative: { ...settings.speculative, nMax: n } })}
                  />
                </div>
              </div>
              {/*
                * Drafts taken from text already in the conversation, so any model can use it. It only
                * pays when the reply repeats what was read, such as a file being rewritten.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Draft from the conversation</div>
                  <div className="setting-hint">
                    Same answers · helps when the reply copies text it has read · next load
                  </div>
                </div>
                <div className="setting-control">
                  <select
                    className="kv-select"
                    value={settings.speculative.ngram}
                    onChange={(e) =>
                      void patch({
                        speculative: { ...settings.speculative, ngram: e.target.value as AppSettings['speculative']['ngram'] }
                      })
                    }
                    data-testid="speculative-ngram"
                  >
                    <option value="off">off</option>
                    <option value="ngram-mod">n-gram (mod)</option>
                    <option value="ngram-simple">n-gram (simple)</option>
                  </select>
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">CPU threads</div>
                  <div className="setting-hint">0 = automatic, which counts real cores rather than hyper-threads</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.runtime.threads}
                    min={0}
                    max={256}
                    onCommit={(n) => void patch({ runtime: { ...settings.runtime, threads: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Prompt threads</div>
                  <div className="setting-hint">--threads-batch · 0 = same as above · next load</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.runtime.threadsBatch}
                    min={0}
                    max={256}
                    onCommit={(n) => void patch({ runtime: { ...settings.runtime, threadsBatch: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Pin threads to cores</div>
                  <div className="setting-hint">Takes effect on the next load</div>
                </div>
                <div className="setting-control">
                  <select
                    className="kv-select"
                    value={settings.runtime.pinThreads}
                    onChange={(e) =>
                      void patch({ runtime: { ...settings.runtime, pinThreads: e.target.value as 'off' | 'close' | 'spread' } })
                    }
                    data-testid="pin-threads"
                  >
                    <option value="off">off — let Windows place them</option>
                    <option value="close">close — fill cores in order</option>
                    <option value="spread">spread — even across all cores</option>
                  </select>
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Re-lay CPU weights</div>
                  <div className="setting-hint">
                    Faster CPU kernels; off lets long prompts move that work to the GPU · next load
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.runtime.repack !== false}
                    onChange={(e) => void patch({ runtime: { ...settings.runtime, repack: e.target.checked } })}
                    data-testid="repack"
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Batch size</div>
                  <div className="setting-hint">--batch-size · 0 = automatic (512) · next load</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.runtime.batchSize}
                    min={0}
                    max={65536}
                    onCommit={(n) => void patch({ runtime: { ...settings.runtime, batchSize: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Micro-batch size</div>
                  <div className="setting-hint">
                    --ubatch-size · 0 = automatic (512) · larger reads prompts faster, uses more VRAM · never
                    above the batch size
                  </div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.runtime.ubatchSize}
                    min={0}
                    max={65536}
                    onCommit={(n) => void patch({ runtime: { ...settings.runtime, ubatchSize: n } })}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="card settings-group" id="settings-conversations">
            <h2 className="card-title">Conversations</h2>
            <div className="settings-rows">
              {/*
                * Past reasoning goes back to the model with each answer, so it can remember why it
                * decided something a few messages ago, not only what it said. It costs context in
                * proportion to how much the model thinks, which is why it is a switch.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Remember past reasoning</div>
                  <div className="setting-hint">
                    Keeps earlier turns' thinking in the context; uses more of it · applies on the next model load
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.reasoning.preserve}
                    onChange={(e) => void patch({ reasoning: { ...settings.reasoning, preserve: e.target.checked } })}
                    data-testid="reasoning-preserve"
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Keep agent instructions on disk</div>
                  <div className="setting-hint">
                    A freshly loaded model loads the Agent's instructions from a file instead of reading them
                    again · next load
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.runtime.keepInstructions !== false}
                    onChange={(e) => void patch({ runtime: { ...settings.runtime, keepInstructions: e.target.checked } })}
                    data-testid="keep-instructions"
                  />
                </div>
              </div>
              {/*
                * Both of these are about the wait before a reply starts, which is the model reading
                * the conversation rather than writing anything.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Read ahead</div>
                  <div className="setting-hint">
                    Start reading a conversation when its message box is clicked, and summarise a long one after a
                    reply rather than before the next message
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.runtime.warmConversations !== false}
                    onChange={(e) => void patch({ runtime: { ...settings.runtime, warmConversations: e.target.checked } })}
                    data-testid="warm-conversations"
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Saved conversations</div>
                  <div className="setting-hint">
                    What the model has read of a conversation, saved when you press Save state in it, so coming back
                    after another conversation, a model reload or a restart is a file read. Anything said after a save
                    is read again then, never lost · text-only conversations
                  </div>
                </div>
              </div>
                  <div className="setting-row">
                    <div className="setting-text">
                      <div className="setting-label">Disk budget</div>
                      <div className="setting-hint">
                        GB · past it, the least recently used saves are removed first
                      </div>
                    </div>
                    <div className="setting-control">
                      <NumberField
                        value={settings.runtime.stateCache.budgetGb}
                        min={1}
                        max={4000}
                        onCommit={(n) =>
                          void patch({
                            runtime: { ...settings.runtime, stateCache: { ...settings.runtime.stateCache, budgetGb: n } }
                          })
                        }
                      />
                    </div>
                  </div>
                  <div className="setting-row">
                    <div className="setting-text">
                      <div className="setting-label">Kept now</div>
                    </div>
                    <div className="setting-control">
                      <span className="faint" data-testid="state-cache-usage">
                        {stateInfo
                          ? `${stateInfo.count} conversation${stateInfo.count === 1 ? '' : 's'}, ${fmtBytes(stateInfo.usedBytes)}, in ${stateInfo.folder}`
                          : '…'}
                      </span>
                      {!!stateInfo?.count && (
                        <button
                          className="tiny"
                          style={{ marginLeft: 8 }}
                          onClick={() => {
                            void invoke('state:clear')
                              .then(() => invoke<typeof stateInfo>('state:info'))
                              .then(setStateInfo)
                              .catch(() => undefined)
                          }}
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>
            </div>
          </div>

          <div className="card settings-group" id="settings-agent">
            <h2 className="card-title">Agent</h2>
            <div className="settings-rows">
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Plan mode</div>
                  <div className="setting-hint">Read-only until a plan is approved</div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={settings.agent.planMode}
                    onChange={(e) => void patch({ agent: { ...settings.agent, planMode: e.target.checked } })}
                  />
                </div>
              </div>
              {/*
                * A small model for compaction, beside the loaded one.
                *
                * Off until chosen. Its summaries are what the loaded model works from afterwards, and a
                * 2B model writes them less well than a very large one: minutes of waiting against some
                * of that quality is the user's trade to make, not the app's. Shown whatever the agent's
                * strategy is, because chats compact with it too.
                */}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Summarise with a small model</div>
                  <div className="setting-hint">
                    Chats and agent sessions compact in about ten passes with a progress reading, instead of on
                    the loaded model
                  </div>
                </div>
                <div className="setting-control">
                  <input
                    type="checkbox"
                    checked={helper.enabled}
                    onChange={(e) => void setHelper({ enabled: e.target.checked })}
                    data-testid="summarizer-enabled"
                  />
                </div>
              </div>
              {helper.enabled && (
                <>
                  <div className="setting-row">
                    <div className="setting-text">
                      <div className="setting-label">Summariser model</div>
                    </div>
                    <div className="setting-control">
                      <select
                        className="kv-select"
                        value={helper.modelId ?? ''}
                        onChange={(e) => void setHelper({ modelId: e.target.value || null })}
                        data-testid="summarizer-model"
                      >
                        <option value="">Choose a model…</option>
                        {helperChoices.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.filename} · {fmtBytes(m.bytes)}
                            {m.id === recommendedModel?.id ? ' · recommended' : ''}
                          </option>
                        ))}
                      </select>
                      {!helper.modelId && <span className="faint">until one is picked, the loaded model summarises</span>}
                    </div>
                  </div>
                  <div className="setting-row">
                    <div className="setting-text">
                      <div className="setting-label">Runs on</div>
                    </div>
                    <div className="setting-control">
                      <select
                        className="kv-select"
                        value={helper.placement}
                        onChange={(e) => void setHelper({ placement: e.target.value as 'auto' | 'gpu' | 'cpu' })}
                      >
                        <option value="auto">automatic — a GPU with room, else the CPU</option>
                        <option value="gpu">GPU when there is room</option>
                        <option value="cpu">CPU only</option>
                      </select>
                    </div>
                  </div>
                  <div className="setting-row">
                    <div className="setting-text">
                      <div className="setting-label">Recommended</div>
                      <div className="setting-hint">
                        Long context on a small cache, and answers without thinking first
                      </div>
                    </div>
                    <div className="setting-control">
                      <span>
                        {RECOMMENDED_SUMMARIZER.label} · {fmtBytes(RECOMMENDED_SUMMARIZER.bytes)}
                      </span>
                      {recommendedModel ? (
                        helper.modelId !== recommendedModel.id && (
                          <button onClick={() => void setHelper({ modelId: recommendedModel.id })}>Use it</button>
                        )
                      ) : (
                        <button onClick={() => void downloadRecommended()} disabled={helperDownload === 'queued'}>
                          {helperDownload === 'queued' ? 'Downloading…' : 'Download'}
                        </button>
                      )}
                    </div>
                  </div>
                </>
              )}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Max tool calls / turn</div>
                  <div className="setting-hint">Runaway-loop ceiling</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.agent.maxToolCallsPerTurn}
                    min={1}
                    max={1000}
                    onCommit={(n) => void patch({ agent: { ...settings.agent, maxToolCallsPerTurn: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Command timeout</div>
                  <div className="setting-hint">In milliseconds</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.agent.commandTimeoutMs}
                    min={1000}
                    max={3_600_000}
                    onCommit={(n) => void patch({ agent: { ...settings.agent, commandTimeoutMs: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Approvals</div>
                </div>
                <div className="setting-control">
                  <button onClick={async () => { await invoke('agent:clear-permission-rules'); setInfo('Remembered approvals cleared.') }}>
                    Clear remembered approvals
                  </button>
                </div>
              </div>
            </div>

            <div className="danger-zone">
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <div>
                  <strong>Hard blocks</strong>
                  <div className="faint" style={{ fontSize: 11 }}>
                    Refuses disk formatting, system-root deletion, bootloader writes and disabling security tools —
                    in every permission mode.
                  </div>
                </div>
                <span className={`badge ${settings.agent.hardBlocksDisabled ? 'bad' : 'good'}`}>
                  {settings.agent.hardBlocksDisabled ? 'DISABLED' : 'enabled'}
                </span>
              </div>
              <button
                className="danger"
                style={{ marginTop: 10 }}
                data-testid="toggle-hard-blocks"
                onClick={() => {
                  // Re-enabling protection is safe and immediate; removing it is not.
                  if (settings.agent.hardBlocksDisabled) {
                    void patch({ agent: { ...settings.agent, hardBlocksDisabled: false } })
                  } else {
                    setConfirmHardBlocks(true)
                  }
                }}
              >
                {settings.agent.hardBlocksDisabled ? 'Re-enable hard blocks' : 'Disable hard blocks…'}
              </button>
            </div>
          </div>

          <div className="card settings-group" id="settings-scratchpad">
            <h2 className="card-title">Scratchpad</h2>
            {/*
              * Chat's only workspace. A chat cannot reach the user's files the way the agent can, so
              * what it is given to work on, a video above all, is copied here and its model reaches
              * it by name. In memory by default so it never touches a disk; the model deletes what it
              * has finished with, and anything left unused is deleted on a timer.
              */}
            <p className="settings-lede">
              Where a chat's model keeps what it is working on, such as a video you attach, which it watches a few frames at
              a time. Nothing in it is kept: the model deletes a file once it has finished with it, a file it has not used
              for a while is deleted anyway, and everything left is cleared when the app closes.
            </p>
            <div className="settings-rows">
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Location</div>
                  <div className="setting-hint">
                    {settings.scratchpad.location === 'folder' && settings.scratchpad.folder
                      ? 'In a folder: for videos too large to hold in memory'
                      : 'In memory: nothing is written to disk'}
                  </div>
                </div>
                <div className="setting-control">
                  <div className="segmented" role="group" aria-label="Scratchpad location">
                    <button
                      aria-pressed={settings.scratchpad.location !== 'folder' || !settings.scratchpad.folder}
                      onClick={() => void patch({ scratchpad: { ...settings.scratchpad, location: 'memory' } })}
                      data-testid="scratchpad-memory"
                    >
                      Memory
                    </button>
                    <button
                      aria-pressed={settings.scratchpad.location === 'folder' && !!settings.scratchpad.folder}
                      onClick={async () => {
                        const dir = settings.scratchpad.folder || (await invoke<string | null>('scratchpad:choose-folder'))
                        if (dir) await patch({ scratchpad: { ...settings.scratchpad, location: 'folder', folder: dir } })
                      }}
                      data-testid="scratchpad-folder"
                    >
                      Folder
                    </button>
                  </div>
                </div>
              </div>
              {settings.scratchpad.location === 'folder' && settings.scratchpad.folder && (
                <div className="setting-row">
                  <div className="setting-text">
                    <div className="setting-label">Folder</div>
                    <div className="setting-hint">
                      Its files go in a folder of the app&apos;s own inside this one, which is emptied when the app closes. Nothing
                      else here is touched.
                    </div>
                  </div>
                  <div className="setting-control">
                    <span className="faint truncate" title={settings.scratchpad.folder}>
                      {settings.scratchpad.folder}
                    </span>
                    <button
                      onClick={async () => {
                        const dir = await invoke<string | null>('scratchpad:choose-folder')
                        if (dir) await patch({ scratchpad: { ...settings.scratchpad, location: 'folder', folder: dir } })
                      }}
                    >
                      Change…
                    </button>
                  </div>
                </div>
              )}
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Delete unused files after</div>
                  <div className="setting-hint">Minutes since the model last read or wrote a file</div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.scratchpad.idleMinutes}
                    min={1}
                    max={1440}
                    testId="scratchpad-idle"
                    onCommit={(n) => void patch({ scratchpad: { ...settings.scratchpad, idleMinutes: n } })}
                  />
                </div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Holding now</div>
                </div>
                <div className="setting-control">
                  <span className="faint" data-testid="scratchpad-usage">
                    {!scratch || !scratch.files
                      ? 'Nothing'
                      : `${scratch.files} file${scratch.files === 1 ? '' : 's'}, ${fmtBytes(scratch.bytes)}`}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="card settings-group" id="settings-mcp">
            <h2 className="card-title">MCP servers</h2>
            <div className="dim" style={{ marginBottom: 10 }}>
              Connect any Model Context Protocol server to add its tools to the agent. Servers run as local
              processes with your permissions, so only add ones you trust.
            </div>

            {mcp.map((s) => (
              <div className="row row-card" key={s.id}>
                <span className={`badge ${s.connected ? 'good' : 'bad'}`}>{s.connected ? 'connected' : 'offline'}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="truncate">{s.name}</div>
                  <div className="faint" style={{ fontSize: 11 }}>
                    {s.connected ? `${s.tools.length} tools: ${s.tools.slice(0, 5).join(', ')}${s.tools.length > 5 ? '…' : ''}` : s.error ?? 'not connected'}
                  </div>
                </div>
                <button onClick={async () => { await invoke('mcp:remove', s.id); await refresh() }}>Remove</button>
              </div>
            ))}
            {!mcp.length && <div className="empty" style={{ fontSize: 12 }}>No MCP servers configured.</div>}

            <div className="row" style={{ marginTop: 12, flexWrap: 'wrap' }}>
              <input type="text" placeholder="Name" value={mcpName} onChange={(e) => setMcpName(e.target.value)} style={{ width: 120 }} />
              <input type="text" placeholder="Command (e.g. npx)" value={mcpCommand} onChange={(e) => setMcpCommand(e.target.value)} style={{ width: 160 }} />
              <input type="text" placeholder="Args (space separated)" value={mcpArgs} onChange={(e) => setMcpArgs(e.target.value)} style={{ flex: 1, minWidth: 160 }} />
              <button
                onClick={async () => {
                  if (!mcpName || !mcpCommand) return
                  await invoke('mcp:add', {
                    name: mcpName,
                    transport: 'stdio',
                    command: mcpCommand,
                    args: mcpArgs.split(/\s+/).filter(Boolean),
                    enabled: true
                  })
                  setMcpName('')
                  setMcpCommand('')
                  setMcpArgs('')
                  await invoke('mcp:connect')
                  await refresh()
                }}
              >
                Add
              </button>
              <button onClick={async () => { await invoke('mcp:connect'); await refresh() }}>Reconnect all</button>
            </div>
          </div>

          <div className="card settings-group" id="settings-memory">
            <h2 className="card-title">Agent memory</h2>
            <div className="dim" style={{ marginBottom: 10 }}>
              What the agent has chosen to remember across sessions. This is the only context carried between
              sessions — there is no per-folder instructions file.
            </div>
            {memory.map((m) => (
              <div className="row row-card" key={m.id}>
                <div style={{ flex: 1, minWidth: 0 }}>{m.text}</div>
                <button onClick={async () => { await invoke('agent:memory-delete', m.id); await refresh() }}>Forget</button>
              </div>
            ))}
            {!memory.length && <div className="empty" style={{ fontSize: 12 }}>Nothing remembered yet.</div>}
          </div>

          {logins && (
            <div className="card settings-group" id="settings-logins" data-testid="saved-logins">
              <h2 className="card-title">Saved logins</h2>
              <div className="dim" style={{ marginBottom: 10 }}>
                Usernames and passwords the agent can sign in with. It uses a password by reference when a tool
                runs and is not shown it, and you are asked before each new use. Passwords are encrypted on this
                computer.
              </div>
              {logins.map((l) => (
                <div className="row row-card" key={l.name}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <strong>{l.name}</strong>
                    <span className="dim"> · {l.username || 'no username'}{l.url ? ` · ${l.url}` : ''}</span>
                  </div>
                  <button onClick={async () => { await invoke('credentials:delete', l.name); await refresh() }}>Delete</button>
                </div>
              ))}
              {!logins.length && <div className="empty" style={{ fontSize: 12 }}>No logins saved yet.</div>}
              <div className="row" style={{ marginTop: 10, flexWrap: 'wrap' }}>
                <input
                  placeholder="Name (e.g. github)"
                  value={newLogin.name}
                  onChange={(e) => setNewLogin({ ...newLogin, name: e.target.value })}
                  style={{ flex: 1, minWidth: 120 }}
                  data-testid="login-name"
                />
                <input
                  placeholder="Username"
                  value={newLogin.username}
                  onChange={(e) => setNewLogin({ ...newLogin, username: e.target.value })}
                  style={{ flex: 1, minWidth: 120 }}
                  data-testid="login-username"
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={newLogin.password}
                  onChange={(e) => setNewLogin({ ...newLogin, password: e.target.value })}
                  style={{ flex: 1, minWidth: 120 }}
                  data-testid="login-password"
                />
                <input
                  placeholder="Site or app (optional)"
                  value={newLogin.url}
                  onChange={(e) => setNewLogin({ ...newLogin, url: e.target.value })}
                  style={{ flex: 1, minWidth: 120 }}
                />
                <button
                  disabled={!newLogin.name.trim() || !newLogin.password}
                  onClick={async () => {
                    try {
                      await invoke('credentials:save', newLogin)
                      setNewLogin({ name: '', username: '', password: '', url: '' })
                      await refresh()
                    } catch (err) {
                      setInfo(`Could not save the login: ${err instanceof Error ? err.message : String(err)}`)
                    }
                  }}
                  data-testid="login-save"
                >
                  Save
                </button>
              </div>
            </div>
          )}

          <div className="card settings-group" id="settings-hf">
            <h2 className="card-title">Hugging Face</h2>
            <div className="row">
              <input
                type="password"
                placeholder="Access token (for gated repos and higher rate limits)"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                style={{ flex: 1 }}
              />
              <button onClick={async () => { await invoke('settings:set-hf-token', hfToken || null); setHfToken(''); setInfo('Token saved.') }}>
                Save
              </button>
            </div>
            <div className="settings-rows" style={{ marginTop: 12 }}>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Connections per file</div>
                  <div className="setting-hint">
                    One connection rarely fills a fast line; more helps up to what it can carry
                  </div>
                </div>
                <div className="setting-control">
                  <NumberField
                    value={settings.downloads.connections}
                    min={1}
                    max={16}
                    onCommit={(n) => void patch({ downloads: { ...settings.downloads, connections: n } })}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="card settings-group" id="settings-folders">
            <h2 className="card-title">Folders</h2>
            <div className="settings-rows">
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">App folder</div>
                </div>
                <div className="setting-control truncate">{paths?.exeDir}</div>
              </div>
              <div className="setting-row">
                <div className="setting-text">
                  <div className="setting-label">Models folder</div>
                </div>
                <div className="setting-control truncate">{paths?.modelsDir}</div>
              </div>
            </div>
            <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
              Models live beside the app. If you move the app, it offers to bring them along on next launch.
            </div>

            <div className="dim" style={{ marginTop: 14, marginBottom: 8 }}>
              Other model folders. Their models are listed and loaded where they are, so nothing is copied;
              downloads and imports still go to the models folder.
            </div>
            {settings.extraModelDirs.map((dir) => (
              <div className="row row-card" key={dir}>
                <div className="truncate mono" style={{ flex: 1, minWidth: 0 }} title={dir}>{dir}</div>
                <button
                  data-testid="remove-model-folder"
                  onClick={async () => {
                    const folders = await invoke<string[]>('library:remove-folder', dir)
                    setSettings((s) => (s ? { ...s, extraModelDirs: folders } : s))
                    setInfo(`Models in ${dir} are no longer listed. The folder and its files were not touched.`)
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
            <div className="row" style={{ marginTop: 8 }}>
              <button
                data-testid="add-model-folder"
                onClick={async () => {
                  const r = await invoke<{ added: string | null; coveredBy?: string; folders: string[] }>('library:add-folder')
                  setSettings((s) => (s ? { ...s, extraModelDirs: r.folders } : s))
                  if (r.added) setInfo(`Models in ${r.added} now appear under My models.`)
                  else if (r.coveredBy) setInfo(`That folder is already in the library: it is inside ${r.coveredBy}.`)
                }}
              >
                Add folder…
              </button>
            </div>
          </div>

          <div className="card settings-group" id="settings-about">
            <h2 className="card-title">
              About
              <span className="badge">v{version}</span>
            </h2>
            <div className="row" style={{ flexWrap: 'wrap' }}>
              <button
                onClick={async () => {
                  const result = await invoke<typeof update>('update:check')
                  setUpdate(result)
                  if (result && !result.available) setInfo(result.error ?? 'You are on the latest version.')
                }}
              >
                Check for updates
              </button>
              <button onClick={async () => { const p = await invoke<string>('diagnostics:reveal'); setInfo(`Diagnostics written to ${p}`) }}>
                Copy diagnostics
              </button>
            </div>

            {/*
              * Shown above the offer, not folded into `error`.
              *
              * `error` describes this check and is only rendered when nothing is available. A swap
              * that failed last time needs saying precisely when an update *is* on offer, because
              * that offer looks identical to the one that already failed once — and without this the
              * app would simply be repeating a claim it did not honour.
              */}
            {update?.previousFailure && (
              <div className="card" style={{ marginTop: 12, borderColor: '#5c2626' }}>
                <div className="card-title">
                  The last update did not finish
                  <span className="badge bad">not applied</span>
                </div>
                <div className="dim">{update.previousFailure}</div>
              </div>
            )}

            {update?.available && (
              <div className="card" style={{ marginTop: 12, borderColor: 'var(--accent-dim)' }}>
                <div className="card-title">Version {update.latestVersion} is available</div>
                <pre style={{ maxHeight: 160 }}>{update.notes}</pre>
                {/*
                  * Nine hundred megabytes with nothing on screen is indistinguishable from a dead
                  * button. The main process was already reporting progress and the channel was
                  * already allowed through — nothing here had ever subscribed to it, so every byte
                  * of it was computed, sent across the bridge and dropped. The first sign the click
                  * had worked was the app restarting itself minutes later.
                  */}
                {downloading ? (
                  <div data-testid="update-progress">
                    <div className="row" style={{ justifyContent: 'space-between', fontSize: 12 }}>
                      <span>
                        {downloading.total > 0 && downloading.done >= downloading.total
                          ? `Checking version ${update.latestVersion} and starting it…`
                          : `Downloading version ${update.latestVersion}…`}
                      </span>
                      <span className="mono">
                        {downloading.total > 0 ? `${Math.round((downloading.done / downloading.total) * 100)}%` : fmtBytes(downloading.done)}
                      </span>
                    </div>
                    <div className="meter" style={{ marginTop: 6 }}>
                      <span style={{ width: downloading.total > 0 ? `${(downloading.done / downloading.total) * 100}%` : '100%' }} />
                    </div>
                    <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
                      {fmtBytes(downloading.done)}
                      {downloading.total > 0 && ` of ${fmtBytes(downloading.total)}`} · this version stays installed until the new one is running
                    </div>
                  </div>
                ) : (
                  <button
                    className="primary"
                    onClick={async () => {
                      // Set before awaiting: the first progress frame can be a second or more away
                      // on a slow link, and the button must stop looking clickable immediately.
                      setDownloading({ done: 0, total: update.bytes ?? 0 })
                      setUpdateError(null)
                      try {
                        const r = await invoke<{ ok: boolean; message: string }>('update:apply', update.downloadUrl)
                        if (!r?.ok) {
                          setDownloading(null)
                          setUpdateError(r?.message ?? 'The update could not be applied.')
                        }
                      } catch (err) {
                        setDownloading(null)
                        setUpdateError(err instanceof Error ? err.message : String(err))
                      }
                    }}
                    data-testid="update-apply"
                  >
                    Download and install
                  </button>
                )}
                {/* Here, not in the banner at the top of the page: that is a screen away from this
                    button, and a press that only changed something out of sight looked like nothing. */}
                {updateError && (
                  <div className="card error-card" style={{ marginTop: 10 }} data-testid="update-error">
                    <span className="badge bad">not installed</span> {updateError}
                  </div>
                )}
              </div>
            )}
          </div>

          <p className="settings-footnote">
            Stored in %APPDATA%\LLMManager, so settings survive the app being moved. No telemetry is collected, ever.
          </p>
        </div>
      </div>

      <ConfirmDialog
        open={confirmHardBlocks}
        danger
        title="Disable hard blocks?"
        confirmLabel="Disable hard blocks"
        requirePhrase="DISABLE HARD BLOCKS"
        body={
          <>
            <p>
              Hard blocks are the last thing standing between a confused model and an
              unrecoverable command. With them off the agent can format drives, delete system
              roots, overwrite the bootloader and disable security tooling.
            </p>
            <p className="faint">
              Approval prompts still apply. This only removes the refusals that no approval can
              override.
            </p>
          </>
        }
        onCancel={() => setConfirmHardBlocks(false)}
        onConfirm={() => {
          setConfirmHardBlocks(false)
          void patch({ agent: { ...settings.agent, hardBlocksDisabled: true } })
        }}
      />
    </>
  )
}
