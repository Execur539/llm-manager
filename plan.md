# LLM Manager — Plan

> Status: **Requirements gathering complete through Round 11 — awaiting go-ahead**
> Last updated: 2026-08-23
> Location: `D:\CODE\LLM Manager`

---

## 1. What this is

A portable Windows desktop app, shipped as a **single `.exe`**, that is a complete local-LLM
control panel:

- Searches HuggingFace, recommends the right quant for *your* hardware, and downloads models itself.
- Stores models in a folder **beside the exe**, and brings them along when the exe moves.
- Runs them via a **bundled llama.cpp** on CPU, Vulkan, or CUDA — chosen automatically.
- Fits them into VRAM with an **auto-fit engine designed to beat the competition's heuristics**.
- Chats with text, images, audio, video, and your own documents (RAG).
- Serves an **OpenAI-compatible API** to other tools on your machine.
- Hosts a **web UI reachable from anywhere**, over a tunnel or your own domain, behind a password.

- **Acts as an agentic harness** — the model can read and write files, run commands, browse the web,
  drive the OS, and extend itself through MCP servers, in a supervised loop.

LM Studio-like in spirit, but one file, fully portable, opinionated about getting the fit right —
and an agent platform, not just a chat window.

**Everything above is v1 scope.** Nothing is deferred to a later release.

---

## 2. Decision log

### Round 1 — identity
| Topic | Decision |
|-------|----------|
| Core job | All-in-one hub: local model management **and** chat **and** usage visibility. |
| Form factor | Electron desktop app. |
| Runtime | Bundled llama.cpp; the app owns model downloads and file management. |
| Audience | Shipped to other people. Setup-free, no hardcoded paths. |
| Distribution | One single `.exe` containing everything except models. |
| Model storage | Models live in a folder **beside the `.exe`**. |
| Portability rule | App remembers the last known models-folder path; if the exe moves, it offers to bring the models over. |

### Round 2 — mechanics
| Topic | Decision |
|-------|----------|
| State store | `%APPDATA%\LLMManager\` — settings, chat DB, last-known models path, logs. |
| Relocation UX | On launch from a new location: **prompt, then move**, showing the old path and total size, with a progress bar. |
| GPU backends | Bundle **CPU + Vulkan + CUDA**; runtime detection picks the best. ~1.5–2 GB exe accepted. |
| Model sources | **HuggingFace search** + **direct URL / local import**. No curated catalog, no Ollama registry. |

### Round 3 — features
| Topic | Decision |
|-------|----------|
| Chat scope | Core chat + history, presets & parameters, attachments & RAG, vision, **and video**. |
| API server | **Full server tab** — start/stop, port, live request log, copyable base URL. |
| Remote web UI | Browser-accessible **from outside the local network**. |
| Load config | **Intelligent auto-fit is the headline feature.** Honour user settings, spill to system RAM only when genuinely necessary, explicitly beat existing apps' heuristics. |
| Cloud APIs | **No — local only.** No API keys, no cloud chat, no cost tracking. |

### Round 4 — deployment & fitting
| Topic | Decision |
|-------|----------|
| Remote access | **Built-in tunnel** for zero-config access, **plus** a full bring-your-own-domain path via FreeDNS. |
| Web UI auth | **Password + signed session cookie.** Remote access cannot be enabled until a password is set. |
| Auto-fit goal | **Maximise context.** KV quant preferred at **Q8**, **never below Q4**. Target **≥64K**, ideally **128K**, where the model allows. |
| Packaging | **Extract-once, then cache.** Unpacks on first run only; later launches are instant. |

### Round 5 — self-hosting, RAG, UX
| Topic | Decision |
|-------|----------|
| Own domain | **Full self-host path**: FreeDNS dynamic-DNS updater, guided port forwarding, automatic TLS issuance + renewal. |
| TLS client | WACS (win-acme) investigated and **rejected** — see §6.1. Use in-process ACME (`acme-client`). |
| Embeddings | **Bundle a small embedding model** *and* let the user swap in another. Both, not either/or. |
| Auto-fit fallback | **Never silently degrade.** Present tradeoffs with predicted speed and context, let the user choose, remember per model. |
| UI direction | **LM Studio-like** — dark, dense, developer-focused. |

### Round 6 — engineering constraints
| Topic | Decision |
|-------|----------|
| TLS client | **In-process ACME** via `acme-client`. |
| Auto-fit priorities | P1 **free-VRAM sizing**, P2 **proportional multi-GPU splits**. Two further failure modes at lower priority (§5.1). |
| Hardware assumptions | **None.** Detect everything at runtime; no GPU/VRAM figures baked into defaults anywhere. |
| Updates | **In-app updater that asks first**, showing what changed. llama.cpp backends ship with the app bundle. |

### Round 7 — implementation
| Topic | Decision |
|-------|----------|
| Stack | **React + TypeScript + Vite**, shared by the desktop renderer and the remote web UI. |
| Model concurrency | **One model at a time, fast hot-swap.** Per-model fitted configs cached; optionally keep previous weights in system RAM. The embedding model is the one exception that stays resident. |
| Chat storage | **SQLite**, with per-chat **export to Markdown or JSON**. |
| Code signing | **Ship unsigned**, document the SmartScreen warning. Pipeline must not make signing hard to add later. |

### Round 8 — product surface
| Topic | Decision |
|-------|----------|
| First run | **Straight into the app.** No wizard. Empty library with a prominent "Find a model" entry point. Hardware detection still runs silently so compatibility badges are accurate immediately. |
| Library | **Disk management**, **tags & favourites**, **compatibility badges**. No folders/collections for models. |
| Concurrency | **Local user has priority.** Remote requests queue behind desktop activity. |
| Repo | **Local git now**, license decided later. |

### Round 9 — scope & operations
| Topic | Decision |
|-------|----------|
| MVP scope | **Everything.** Build order is internal sequencing only; the app must be runnable at every stage. |
| Downloads | **Resume after interruption**, **queue + background** with per-item pause/cancel, **HuggingFace token** support, **automatic mmproj fetching**. |
| Diagnostics | **Local logs only. No telemetry, ever.** Rotating logs + a "copy diagnostics" bundle for bug reports. |
| Branding | Name stays **LLM Manager**. Placeholder icon for now. |

### Round 10 — remote, RAG, observability
| Topic | Decision |
|-------|----------|
| Tunnel | Bundle **cloudflared** (Apache-2.0, free quick tunnels, no account, TLS handled, works behind CGNAT). |
| RAG | **Both**: quick per-chat file drops **and** named reusable collections attachable to any chat. |
| Voice | **Audio file attachments only.** No microphone capture, no TTS. |
| Observability | **All four**: live inference stats, hardware meters, server request log, historical stats. |

### Round 11 — runtime behaviour
| Topic | Decision |
|-------|----------|
| Tray behaviour | **Ask on first close** — "minimize to tray or quit?" — and remember the answer. Tray keeps the model loaded and the server serving. |
| Web UI scope | **Full parity, destructive actions gated.** Everything available remotely; deleting models and changing security settings require confirmation on the desktop app. |
| API server | **Configurable, JIT default.** Auto-load a requested model by default, with a setting to disable. API key optional, strongly recommended (and prompted for) when remote access is on. |
| Quant picking | **Recommend + show all.** Highlight the best quant for their hardware with a stated reason, list every variant with its own fit prediction. |

### Round 12 — agent harness
| Topic | Decision |
|-------|----------|
| Core tools | **All four groups**: filesystem, shell & processes, web (search/fetch/browser automation), and system/OS control (screenshots, clipboard, notifications, process list, registry read, input automation). |
| Advanced tools | **MCP client**, **agent primitives** (sub-agents, task lists, persistent memory, scheduled tasks), and **data & code execution** (HTTP, SQLite, parsing, scratch Python/Node interpreter). |
| Self-management | **Not selected.** The agent cannot drive LLM Manager itself (load/unload models, query fit, download). Recorded as out of scope; trivially addable later since the internal APIs exist anyway. |
| Permissions | **Tiered auto-approve.** Reads run freely; writes and shell prompt with *allow once / always for this tool / always for this exact command / deny*. Rules remembered per workspace. |
| Remote tools | **Full tools remotely, but opt-in and OFF by default.** Enabling it requires an explicit action, and should be gated behind a clear warning plus API key + strong password + rate limiting. |

### Round 13 — tool behaviour
| Topic | Decision |
|-------|----------|
| Scope | **Whole machine.** No workspace sandbox; any path the Windows user can reach. The tiered approval gate, hard-blocked list, and checkpoints are therefore the *only* containment — see §6.5 note. |
| Undo | **Automatic checkpoints + git integration.** Snapshot files before each turn for rewind; when the folder is a git repo, offer a commit before risky operations and show real diffs. |
| Extensibility | **MCP only.** No bespoke script-tool or HTTP-hook mechanism; custom capability means writing an MCP server. One extension path to maintain. |
| Tool-call reliability | **Grammar enforcement only.** GBNF constrains output so calls are structurally valid. No dedicated repair loop and no capability warnings in the library. *Note:* failed calls still return errors into the conversation, so the model gets a natural chance to correct itself — just without retry machinery. |

### Round 14 — agent loop
| Topic | Decision |
|-------|----------|
| Hard blocks | **Short hard-blocked list, overridable.** Disk formatting, recursive deletion of system roots, bootloader/firmware writes, and disabling security tooling are blocked by default; a buried setting with a **typed confirmation** can disable the list for users who genuinely need those operations. |
| Context management | **User's choice between two strategies**: *auto-compact* (summarize older turns, keep recent ones and task state verbatim) or *sliding window* (drop oldest). Auto-compact can also be **triggered manually** at any time. |
| Task start | **Plan mode toggle.** Off by default — the agent acts immediately. When on, it investigates read-only and proposes a plan for approval before touching anything. |
| Sub-agents | **Sequential, same model.** One at a time, sharing the loaded model. No extra VRAM, no context-slot splitting, consistent with the one-model-at-a-time decision. |

### Round 15 — agent UX
| Topic | Decision |
|-------|----------|
| Tool display | **Collapsed, expandable.** One-line summary per call ("Read main.ts", "Ran npm test — 2 failed"), expanding to full arguments and output on click. No live output streaming into the transcript; long commands show progress and reveal output on completion. |
| Large tool outputs | **Truncate + keep full on disk.** Big results are cut to a readable head/tail with a marker; the complete output is written to the session store so the agent can re-read specific parts on demand. |
| MCP management | **UI + config file.** Settings page listing servers, their discovered tools, enable/disable, and live connection status — backed by an editable JSON config for power users and sharing. No public server directory. |
| Instructions | **Memory only.** No per-folder instructions file. The agent maintains persistent notes across sessions, which the user can view and edit directly. |

### Round 16 — agent dependencies
| Topic | Decision |
|-------|----------|
| Browser automation | **Bundle Playwright Chromium.** Purpose-built automation browser with the best API and stealth handling. Cost: ~200–300 MB on top of an already-large exe. |
| Web search | **DuckDuckGo, no API key.** Works out of the box with no signup or cost; accepted tradeoffs are shallower results and possible rate-limiting under heavy use. |
| Code execution | **Bundle Python (embeddable, ~15–25 MB) + use Electron's built-in Node.** Both interpreters guaranteed on every install regardless of what the user has. No package installation into the environment. |
| Session persistence | **Full resume.** Conversation, tool history, task list, checkpoints, and pending approvals all persist and restore after a close or a crash. |

---

## 3. Research findings (verified 2026-08-23)

Two initial assumptions were wrong and were corrected by live lookup:

- **Qwen3.8-27B is real.** Dense 27B, Apache 2.0, weights released **2026-08-14**, natively
  multimodal over **text, images, video, diagrams, documents**. 262K native context, extendable to
  ~1M via YaRN. GGUF/llama.cpp community support was still rolling out at release — **verify current
  state before shipping**. ([source](https://www.orcarouter.ai/blog/qwen-3-8-27b-release-date))
- **llama.cpp supports video input.** Merged 2026-06-08 via PR #24269 into `mtmd`. It is
  **model-agnostic**: one `<__media__>` marker expands into decoded frames at tokenization time, so
  any vision model can consume video unmodified. Frame extraction **invokes FFmpeg as a subprocess**,
  which must be on PATH. ([PR](https://github.com/ggml-org/llama.cpp/pull/24269),
  [docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multimodal.md))
- `mtmd` modalities today: **image, audio, video**. Documented vision families: Gemma 3/4,
  Qwen 2/2.5 VL, Pixtral, Mistral Small 3.1, InternVL 2.5/3, Llama 4 Scout, SmolVLM, Moondream2.
  Audio: Ultravox, Voxtral, Qwen3-ASR. Mixed: Qwen2.5/Qwen3 Omni.

### Agentic capability findings (verified 2026-08-23)

- **llama.cpp has OpenAI-style function calling for ~any model**, via native handlers (Llama 3.1–3.3,
  Functionary v3.x, and others) plus a generic fallback. Enabled with the `--jinja` flag; implemented
  in `common/chat.h`.
  ([docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md))
- **JSON Schema is compiled into GBNF grammars enforced at sampling time.** This is significant:
  even a model with no tool-use training can be *constrained* to emit structurally valid tool calls.
  Combined with small local models being the weak link in agentic loops, grammar enforcement is a
  correctness feature, not an optimisation.
- **llama.cpp merged a full MCP client** into its web UI (2026-04). **500+ MCP servers** are publicly
  available covering databases, file storage, scraping, GitHub, Docker, Kubernetes, and more.
  ([source](https://jasonmoon.dev/blog/2026-04-15-mcp-v21-llama-cpp-local-agents/))
- **MCP v2.1 made tool sandboxing a mandatory spec requirement**, after security research found
  command-injection vulnerabilities in 43% of v2.0 implementations. Any MCP client we ship must
  honour that.
- **llama.cpp also speaks the Anthropic Messages API** (2026-01-19), with `tool_use` /
  `tool_result` content blocks. Our API server can therefore expose a Claude-compatible endpoint
  alongside the OpenAI one — meaning Claude-compatible clients, including Claude Code, could point
  at LLM Manager as their backend.
  ([source](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp))

### Consequences
1. **Bundle `ffmpeg.exe`** and inject its directory into the llama-server child's PATH. Without it,
   video input fails silently on most machines. Use an **LGPL** build to keep redistribution light.
2. **Capability detection** reads `mmproj` GGUF metadata. Since video is model-agnostic,
   *vision ⇒ video is possible* — but the UI distinguishes **native video** models (Qwen3.8,
   Qwen3-VL, Omni) from the **frames-as-images** fallback, with different frame-sampling defaults.
3. HuggingFace search must surface new releases immediately. **No hardcoded model list anywhere.**

---

## 4. Architecture

### 4.1 Filesystem layout

```
<wherever the user put it>/
├── LLM Manager.exe                  <- the single shipped file
├── .llmmanager-runtime/             <- unpacked payload, written on first run only
└── LLMManagerModels/                <- created on first run
    ├── <repo>/<file>.gguf
    ├── <repo>/<file>.mmproj.gguf
    └── .partial/                    <- resumable in-flight downloads

%APPDATA%\LLMManager\
├── settings.json                    <- all app settings
├── llmmanager.db                    <- SQLite: chats, models, stats, RAG
├── models-path.json                 <- breadcrumb: last known models folder
├── secrets.json                     <- password hash, API key, HF token (DPAPI-encrypted)
└── logs/                            <- rotating logs
```

### 4.2 Process model

- **Main (Node/Electron)** — window + tray lifecycle, filesystem, models-folder relocation, download
  manager, GPU detection, **auto-fit engine**, llama-server supervision, HTTP server for the remote
  web UI, cloudflared supervision, ACME + DDNS, SQLite access.
- **Preload** — narrow typed IPC bridge; `contextIsolation` on, `nodeIntegration` off.
- **Renderer (React + TS)** — the entire UI. Built once, served two ways: loaded locally by the
  Electron window, and served over HTTP to remote browsers.
- **llama-server child** — bundled llama.cpp, OpenAI-compatible HTTP bound to `127.0.0.1` on a
  random free port. Lifetime owned by main; killed on quit and on crash-restart.
- **cloudflared child** — spawned only when tunnelling is enabled.

### 4.3 Bundled payload and size budget

| Component | Approx size | Notes |
|-----------|------------|-------|
| Electron runtime | ~180 MB | |
| llama.cpp CPU + Vulkan | ~60 MB | Vulkan covers NVIDIA/AMD/Intel |
| llama.cpp CUDA + CUDA runtime libs | ~400 MB–1.2 GB | Dominates the payload; cuBLAS is the bulk |
| FFmpeg (LGPL) | ~80 MB | Required for video |
| cloudflared | ~40 MB | Tunnel |
| Embedding model (GGUF) | ~60–120 MB | Bundled default for RAG |
| **Playwright Chromium** | ~200–300 MB | Agent browser automation |
| **Python (embeddable)** | ~15–25 MB | Agent code execution |
| **ripgrep** | ~5 MB | Fast agent file search |
| App code + assets | ~10 MB | |
| **Total** | **~1.2–2.0 GB** | At the top of the accepted ~2 GB ceiling. If the CUDA payload lands at the high end, revisit which CUDA architectures ship. |

### 4.4 Module breakdown (main process)

```
src/main/
├── index.ts                 app + window + tray lifecycle
├── ipc/                     typed IPC handlers
├── storage/                 settings, secrets (DPAPI), SQLite migrations
├── models/
│   ├── gguf.ts              GGUF header parser (metadata, tensors, arch)
│   ├── library.ts           scan, index, tag, disk usage, dedup, orphan cleanup
│   ├── relocation.ts        breadcrumb, move-on-launch, progress, cancel
│   └── capabilities.ts      modality detection from model + mmproj metadata
├── hardware/
│   ├── gpu.ts               adapter enumeration, free/total VRAM, utilisation
│   └── backend.ts           CPU/Vulkan/CUDA selection
├── autofit/
│   ├── budget.ts            per-device memory budgeting
│   ├── kv.ts                exact KV cache maths
│   ├── plan.ts              candidate configs + tradeoff options
│   └── verify.ts            post-load validation against prediction
├── runtime/
│   ├── server.ts            spawn/supervise llama-server, health, hot-swap
│   └── queue.ts             priority queue (local > remote)
├── downloads/
│   ├── hf.ts                HF search, file listing, token auth
│   ├── queue.ts             resumable, pausable, concurrent downloads
│   └── recommend.ts         quant recommendation per hardware
├── rag/
│   ├── extract.ts           PDF/txt/code/docx text extraction
│   ├── chunk.ts             chunking + overlap
│   ├── embed.ts             embedding model runner
│   └── search.ts            vector search
├── api/
│   ├── openai.ts            /v1/* endpoints, JIT loading, API key
│   └── weblog.ts            request logging
├── remote/
│   ├── web.ts               HTTP(S) server for the web UI
│   ├── auth.ts              password hash, sessions, gating
│   ├── tunnel.ts            cloudflared lifecycle
│   ├── ddns.ts              FreeDNS updater
│   └── acme.ts              in-process Let's Encrypt via acme-client
├── stats/                   live + historical metrics
├── update/                  ask-first updater
└── log/                     rotating logs + diagnostics bundle
```

---

## 5. The auto-fit engine (headline feature)

**Objective:** maximise context length, subject to KV quant ≥ Q4 (prefer Q8), targeting ≥64K and
ideally 128K where the model's trained context allows — without ever OOMing and without ever
silently degrading behind the user's back.

### Inputs
- GGUF header: `n_layer`, `n_embd`, `n_head_kv`, `head_dim`, vocab, per-tensor sizes, quant type,
  trained context length.
- **Free** VRAM per adapter, measured immediately before spawn (not total, not cached).
- Exact KV cache size: `n_layer × n_head_kv × head_dim × ctx × 2 × bytes(kv_type)`.
- Compute/scratch buffers, CUDA context overhead, and desktop/display VRAM already committed.
- System RAM available, for spillover decisions.
- **Every user-set value is respected, never silently overridden.**

### Output
A ranked set of candidate configurations, each with predicted context, predicted tokens/sec, and
where each layer lives. The best one loads automatically; if the target cannot be met, the user is
shown the tradeoffs and picks — and that choice is remembered per model.

### 5.1 Failure modes we must beat

Ranked by user priority. Each becomes an explicit test case.

**P1 — Sizing against total VRAM instead of free VRAM.**
Other apps budget from the card's *total* capacity, ignoring the 1–3 GB the desktop compositor and
browser already hold. The load then OOMs or thrashes into shared memory at a fraction of the speed.
*Our approach:* query **free** VRAM per adapter (NVML on NVIDIA, DXGI `QueryVideoMemoryInfo` as the
vendor-neutral path), re-check immediately before spawning, and budget against the delta.

**P2 — Naive even splits across mismatched GPUs.**
A 24 GB + 8 GB pair gets a 50/50 tensor split, so the small card OOMs while the big one idles.
*Our approach:* split proportional to **actual free capacity per device**, and show the computed
ratio in the UI rather than hiding it.

**P3 — Over-conservative offload.**
Leaving 4–6 GB unused and pushing layers to CPU "to be safe", destroying throughput for no reason.
*Our approach:* fill to a measured headroom margin, not a hardcoded percentage.

**P4 — Ignoring KV cache growth.**
Fitting at load time, then OOMing 20K tokens into a long chat. *Our approach:* reserve the **full
configured context's** KV cache up front, so a load that succeeds cannot die mid-conversation.

**Verification loop:** after each load, compare actual VRAM consumption against the prediction and
record the delta. Persistent error feeds back into the headroom margin — the engine gets more
accurate on each machine over time.

**Nothing is assumed.** Because this ships to unknown hardware, no VRAM figure, GPU model, or layer
count may be hardcoded anywhere in the codebase.

---

## 6. Remote access & security

Two distinct mechanisms sharing one settings page:

**Tunnel path (default, zero-config)** — `cloudflared` produces a public HTTPS URL. No router
configuration, works behind CGNAT, TLS terminated by Cloudflare.

**Own-domain path (full self-host)** — FreeDNS subdomain + built-in dynamic-DNS updater + guided
port forwarding + **in-process ACME** for a real Let's Encrypt certificate, renewed automatically
while the app runs.

Security rules:
- Remote access **cannot be enabled without a password being set first**.
- Password stored as a salted hash; sessions are signed cookies with expiry.
- Destructive actions (delete model, change security settings) are **gated to the desktop app**.
- API key optional but prompted for whenever remote access is enabled.
- Rate limiting and lockout on repeated failed logins.
- The llama-server child **only ever binds to `127.0.0.1`** — it is never directly exposed; all
  outside traffic goes through our authenticated layer.

### 6.1 TLS: why not WACS (win-acme)

Investigated at the user's suggestion and **rejected**, on these findings:

- win-acme **requires administrator rights**.
- It registers a **Windows Scheduled Task running as SYSTEM** for renewals.
- That task **invokes `wacs.exe` by absolute path** — if the folder moves or is deleted, renewals
  fail silently. ([source](https://www.win-acme.com/manual/automatic-renewal),
  [admin rights issue](https://github.com/win-acme/win-acme/issues/992))

Disqualifying here specifically: this app is *designed to be relocated* — the entire models-relocation
feature exists because the exe moves. WACS would break on every move and leave an orphaned SYSTEM
scheduled task behind, violating the portable, no-admin, no-trace design.

**Chosen instead:** [`acme-client`](https://github.com/publishlab/node-acme-client) — pure Node, no
admin, no external binary, no scheduled task. The app already owns the web server, so the HTTP-01
challenge is served directly and renewal runs in-process. Same Let's Encrypt certificates, none of
the fragility. Requires port 80 reachable; DNS-01 via the FreeDNS API is the fallback if the ISP
blocks it.

---

## 6.5 The agent harness

The app is not only a chat client: the model runs in a **supervised agentic loop** with a broad tool
set, in the spirit of Claude Code.

### Why this is buildable now
Grammar-constrained tool calling (§3) means even weak local models can be *forced* to emit valid
tool calls. And rather than hand-writing every tool forever, an **MCP client** opens the app to
500+ existing servers — that is what actually satisfies "anything it could ever want to do".

### Proposed tool catalog

| Group | Tools |
|-------|-------|
| **Filesystem** | read, write, edit (exact-match replace), multi-edit, glob, grep/ripgrep, list, stat, move, copy, delete, diff |
| **Shell & process** | run command, run in background, stream output, kill, environment inspection, working-directory management |
| **Development** | git (status/diff/log/commit/branch), run tests, run linters, package managers, build commands |
| **Web** | search, fetch URL → markdown, headless browser automation (navigate, click, type, read DOM, screenshot) |
| **System / OS** | process list, screenshots, clipboard read/write, notifications, registry read (write gated), installed-app enumeration, mouse/keyboard automation |
| **Data** | HTTP requests, SQLite queries, JSON/CSV/XML parsing, file-format conversion |
| ~~Self-management~~ | *Excluded by decision.* The agent cannot drive LLM Manager itself. |
| **Agent primitives** | sub-agent delegation, task lists, persistent memory, scheduled/recurring tasks |
| **MCP client** | connect to any MCP server (stdio + HTTP), discover its tools/resources/prompts, honour v2.1 sandboxing requirements |

### Agent runtime behaviour

**Permission tiers.** Read-class tools (read, list, glob, grep, stat, search, fetch, screenshot,
process list, clipboard read) run without prompting. Write-class and execute-class tools (write,
edit, move, copy, delete, shell, input automation, registry write, HTTP with side effects) prompt
with *allow once / always for this tool / always for this exact command / deny*. Decisions persist
per folder. Hard-blocked commands are refused above all of this, unless the buried override is on.

**Plan mode.** Off by default. When enabled, the agent is restricted to read-class tools until it
has produced a written plan the user approves; approving unlocks the full tier set for that task.

**Checkpoints.** Before each turn, every file the agent is about to modify is snapshotted to the
session store. "Rewind to before this message" restores them. In a git repo, the agent additionally
offers a commit before risky operations and renders real diffs.

**Context management.** User selects *auto-compact* (summarize older turns, preserve recent turns
and task state verbatim) or *sliding window* (drop oldest). Auto-compact is also manually
triggerable. Independently, large tool results are truncated in-context with the full output written
to disk and re-readable on demand — this is the single biggest lever on how long a session survives.

**Sub-agents.** Sequential only, sharing the loaded model. A sub-agent gets its own context and
returns a result to the parent, which keeps the parent's context clean on large subtasks.

**Memory.** The agent keeps persistent notes across sessions. No per-folder instructions file;
memory is the only carried context, and it is user-viewable and user-editable.

**Tool-call generation.** JSON Schema for every tool is compiled to GBNF and enforced at sampling
time, so calls are always structurally valid regardless of the model's tool training. Argument-level
mistakes return errors into the conversation for the model to react to; there is no dedicated retry
loop by decision.

### Security analysis

Adding arbitrary code execution to an app that is **deliberately internet-reachable** is a material
change to the threat model, not an incremental feature:

1. **Remote shell exposure.** The web UI is reachable from outside the LAN with full parity. If the
   agent can run shell commands, a leaked or brute-forced password becomes **remote code execution
   as the logged-in Windows user**. *Decision:* remote tool use is supported but **off by default**,
   enabled only by explicit user action behind a clear warning, and requires an API key, a strong
   password, and rate limiting before it can be turned on.
2. **Prompt injection into a shell.** The agent reads files and web pages. Any of that content can
   contain text aimed at the model. With shell access, injection escalates from "wrong answer" to
   "arbitrary command". Mitigations: treat all tool output as untrusted data, never as instruction;
   require approval for state-changing calls; keep an allowlist rather than a blocklist.
3. **Weak local models are the operator.** Unlike a frontier model, a 7B quant may call `delete` when
   it meant `list`. Guardrails matter *more* here, not less.
4. **MCP servers are third-party code.** They run on the user's machine with the user's rights.
   MCP v2.1 mandates sandboxing; we honour it and surface clearly what each server can do.

Baseline safeguards regardless of the Round 12 answers:
- Every destructive tool call is **reversible or confirmable** — file edits are checkpointed so any
  turn can be rolled back.
- **No workspace sandbox** (decided): the agent operates machine-wide. Containment therefore rests
  entirely on the tiered approval gate, the hard-blocked list, and checkpoints. Paths are still
  canonicalised before use so that approval prompts show the *real* resolved target rather than a
  misleading relative or symlinked path.
- A **hard-blocked command list** covering disk formatting, system-wide recursive deletion,
  bootloader and firmware writes, and disabling security tooling. Blocked in every permission mode
  by default, but **overridable** via a deliberately buried setting requiring a typed confirmation.
- Full **audit log** of every tool call, its arguments, and its result, in the diagnostics bundle.
- The agent may never read `secrets.json`, the password hash, the API key, or the HF token.

## 7. Feature list

### Model library
- Auto-scan the models folder; index GGUF metadata (arch, params, quant, trained context, modality).
- **Compatibility badges** computed before loading: *fits fully at 128K* / *partial offload* / *too large*.
- Disk management: size per model, library total, free space, sort by size, delete with confirmation.
- Detect and clean partial or orphaned downloads.
- User tags + favourites; auto-tags from metadata (size, quant, modality, context).
- Import existing GGUF files from anywhere on disk.

### Model acquisition
- HuggingFace search with quant listing and per-file size.
- **Quant recommendation** for the user's actual hardware, with a stated reason, alongside every variant.
- Resumable downloads (HTTP range), background queue, per-item pause/cancel, combined progress.
- Optional HuggingFace token for gated repos and rate limits.
- Automatic `mmproj` companion fetching for multimodal models.
- Direct URL download and local file import.

### Loading
- Automatic backend selection (CUDA → Vulkan → CPU) from detected hardware.
- Auto-fit as described in §5, with a full advanced-override panel.
- Fast hot-swap between models using cached per-model configs.
- Explicit tradeoff dialog whenever the target cannot be met.

### Chat
- Streaming responses; stop generation.
- Conversation history: rename, delete, search, export to Markdown or JSON.
- Regenerate, edit-and-resend, branch conversations.
- Per-chat system prompt; named presets (temperature, top-p, top-k, min-p, repeat penalty).
- Per-model remembered defaults.
- **Attachments:** images, audio files, video (FFmpeg frame extraction), documents.
- Frame-sampling controls for video, defaulted by whether the model is natively video-trained.

### RAG
- Per-chat file drops for one-offs.
- Named, reusable document collections attachable to any chat.
- Bundled embedding model, swappable for any other embedding-capable GGUF.
- Chunking with overlap; vector search; cited sources in answers.

### API server
- OpenAI-compatible: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`.
- Start/stop, port selection, copyable base URL.
- **JIT model loading** on request (default on, configurable off).
- Optional API key.
- Live request log: endpoint, model, tokens in/out, duration, local vs remote client.

### Remote access
- cloudflared tunnel with one-click enable, URL display, and QR code.
- Own-domain path: FreeDNS DDNS updater, port-forward guidance, automatic Let's Encrypt TLS.
- Password + session auth; API key; rate limiting.
- Full-parity web UI with destructive actions gated to the desktop.

### Agent harness
- Supervised tool-calling loop with GBNF-enforced call structure.
- Tiered permissions: reads free, writes/shell prompted, decisions remembered per folder.
- Overridable hard-block list for catastrophic operations.
- Plan-mode toggle: read-only investigation, then an approved plan before acting.
- Automatic checkpoints with rewind; git-aware commits and diffs.
- Collapsed, expandable tool cards; large outputs truncated with the full text kept on disk.
- Context compaction (auto or sliding window, manually triggerable).
- Sequential sub-agents, task lists, persistent editable memory, scheduled tasks.
- MCP client with management UI and JSON config.
- Full session resume across close and crash.
- Complete audit log of every tool call in the diagnostics bundle.

### Observability
- Live inference stats: tokens/sec, time to first token, context used vs available, KV fill.
- Hardware meters: per-GPU VRAM and utilisation, system RAM, CPU.
- Historical stats: totals, hours of use, most-used models, average speed per model.

### App
- Tray behaviour with first-close prompt; model and server stay alive when minimized.
- Models-folder relocation on move, with size, progress, cancel, and a keep-in-place option.
- Ask-first in-app updater showing what changed.
- Rotating local logs + one-click diagnostics bundle. **No telemetry.**

---

## 8. Data model (SQLite)

```
models          id, repo, filename, path, bytes, sha, arch, params_b, quant,
                ctx_train, has_vision, has_audio, native_video, mmproj_path,
                added_at, last_used_at, favourite
model_tags      model_id, tag
model_configs   model_id, ctx, n_gpu_layers, tensor_split, kv_type, batch,
                flash_attn, chosen_tradeoff, predicted_vram, actual_vram, updated_at

chats           id, title, model_id, preset_id, system_prompt, created_at, updated_at
messages        id, chat_id, parent_id, role, content, tokens, timings_json, created_at
attachments     id, message_id, kind(image|audio|video|doc), path, meta_json
presets         id, name, temperature, top_p, top_k, min_p, repeat_penalty, system_prompt

collections     id, name, created_at
documents       id, collection_id, chat_id, filename, path, mime, added_at
chunks          id, document_id, ord, text, tokens
embeddings      chunk_id, dim, vector BLOB, embed_model

downloads       id, repo, filename, url, dest, bytes_total, bytes_done, status, error, added_at
requests        id, ts, endpoint, model_id, tokens_in, tokens_out, ms, client(local|remote), ip
stats_daily     date, model_id, tokens_in, tokens_out, active_seconds
```

Settings live in `settings.json`; secrets (password hash, API key, HF token) in `secrets.json`,
encrypted with Windows DPAPI so they are unreadable if copied to another machine.

---

## 9. Build order

Everything is in v1 scope. This is sequencing, not scope reduction — the app stays runnable at every
stage.

| Phase | Deliverable |
|-------|-------------|
| **0** | Scaffold: Electron + Vite + React + TS, IPC bridge, settings store, SQLite + migrations, logging, `git init`. |
| **1** | Hardware detection + backend selection; bundled llama.cpp; spawn/supervise llama-server; health checks. |
| **2** | GGUF parser; model library; folder management; relocation-on-move; disk management; tags/favourites. |
| **3** | **Auto-fit engine** + tradeoff UI + compatibility badges + post-load verification loop. |
| **4** | Chat: streaming, history, presets, parameters, branching, export. |
| **5** | Downloads: HF search, quant recommendation, resumable queue, mmproj auto-fetch, token support. |
| **6** | Multimodal: images, audio, video via FFmpeg, capability detection, frame-sampling controls. |
| **7** | RAG: embedding model, extraction, chunking, vector search, per-chat drops + collections. |
| **8** | API server: endpoints, JIT loading, API key, request log, local-priority queue. |
| **9** | Remote access: web UI serving, password + sessions, cloudflared, FreeDNS DDNS, ACME TLS, gating. |
| **10** | Observability: live stats, hardware meters, historical stats. |
| **11** | **Agent core**: tool-call loop, GBNF schema enforcement, tiered permission engine, hard-block list, collapsed tool UI, audit log. |
| **12** | **Core tools**: filesystem, shell + background processes, ripgrep, git, checkpoints with rewind, truncate-with-full-on-disk. |
| **13** | **Reach tools**: Playwright browser automation, DuckDuckGo search, URL→markdown fetch, OS control (screenshots, clipboard, input automation), Python/Node execution, HTTP/SQLite/parsing. |
| **14** | **Agent platform**: MCP client + management UI, sub-agents, task lists, persistent memory, scheduled tasks, plan mode, context compaction, full session resume. |
| **15** | Packaging: extract-once portable exe, updater, diagnostics bundle, docs. |

---

## 10. Risks and things to verify before/while building

| Risk | Mitigation |
|------|-----------|
| **Extract-once portable exe** — electron-builder's `portable` target re-extracts to `%TEMP%` every launch. There is no built-in cached mode. | Needs a custom bootstrapper (7-Zip SFX or a small NSIS stub) that unpacks to `.llmmanager-runtime/` once and re-uses it, with a version check. Prototype this early — it is the riskiest packaging assumption. |
| **CUDA payload size** | Ship a single reasonable CUDA arch set rather than a fat binary; measure real size in Phase 1 and revisit if it blows past ~2 GB. |
| **Self-extracting 2 GB exe triggers antivirus heuristics** | Expected with unsigned binaries. Document it; revisit if signing happens. |
| **ISP blocks port 80** — breaks ACME HTTP-01 | Fall back to DNS-01 via the FreeDNS update API, or tell the user to use the tunnel path. |
| **Qwen3.8-27B GGUF support** may not have landed in llama.cpp yet | Verify at Phase 1; the app is model-list-agnostic either way, so this affects testing, not design. |
| **HuggingFace API rate limits** | Cache search results; support the optional token; back off politely. |
| **Video via FFmpeg subprocess** | Bundle FFmpeg and inject PATH for the child process; verify the exact invocation llama.cpp expects at Phase 6. |
| **Moving 100 GB across drives** | Progress + cancel + resume-safe; never delete the source until the copy verifies. Same-drive moves use an instant rename. |
| **Payload now ~2 GB** after Playwright Chromium + Python | Measure early. If CUDA lands at the high end, cut CUDA architectures before cutting features. |
| **Small models are poor agents.** A 7B quant may call the wrong tool or mangle arguments. Grammar enforcement guarantees *valid* calls, not *correct* ones — and no repair loop was chosen. | Approval gates and checkpoints are the real safety net. Worth revisiting the repair-loop decision after seeing real behaviour in Phase 11. |
| **Prompt injection with whole-machine shell access** | Tool output is framed as untrusted data, never instruction. Approval prompts show the fully resolved path or command, so injected intent is visible before it runs. |
| **DuckDuckGo rate-limiting** under heavy agent use | Cache results, back off politely, and surface a clear error rather than silently returning nothing. Optional API key remains an easy later addition. |
| **MCP servers are third-party code** running with the user's rights | Honour MCP v2.1 sandboxing; show clearly what each server exposes; make enable/disable one click. |

---

## 11. Non-goals

- Cloud API providers, API-key management for them, spend tracking.
- Ollama registry integration.
- Curated/hardcoded model catalog.
- Microphone capture and text-to-speech.
- Multiple chat models resident in VRAM simultaneously.
- Telemetry or analytics of any kind.
- macOS and Linux builds.
- Agent self-management of LLM Manager (loading models, downloading, querying fit).
- Custom script tools or HTTP-hook tools — MCP is the single extension path.
- Per-folder agent instructions files; memory is the only carried context.
- A dedicated tool-call repair/retry loop.
- Parallel sub-agents.
- Package installation (pip/npm) by the agent into its execution environment.

---

## 12. Environment notes

- Windows 11, Node v22.14.0, Python 3.13.2, git 2.49.0, npm (no pnpm/bun detected).
- Prior related projects in `D:\CODE` worth mining for reference:
  - `RunLLM` — `gguf_reader.py` is a useful reference for GGUF metadata parsing.
  - `LLM_Chat PROTOTYPE` — includes `LM-Studio-Reference-Schema.txt`.
  - `OllamaGUI` — Node + Express + `public/` frontend patterns.
- Relationship to those projects is **undecided** — this may supersede them or stand alone.

---

## 13. Still open (low priority — can be decided during build)

- Vector store implementation: `sqlite-vec` extension vs. brute-force cosine in JS.
- Whether this supersedes `OllamaGUI` / `RunLLM` / `LLM_Chat PROTOTYPE`.
- Open-source license choice.
- Final icon and accent colour.
- Whether to keep previous model weights in system RAM on hot-swap (memory cost vs. swap speed).
- Chat import from other tools (LM Studio, Ollama, ChatGPT exports).

---

## 14. Changelog

- 2026-08-23 — File created; Round 1 asked.
- 2026-08-23 — Round 1 answered (identity + distribution model). Round 2 asked.
- 2026-08-23 — Round 2 answered (APPDATA state, prompt-then-move, CPU+Vulkan+CUDA, HF + direct URL). Round 3 asked.
- 2026-08-23 — Round 3 answered (full chat + RAG + vision + video, server tab + remote web UI, smart auto-fit, local only).
- 2026-08-23 — Verified Qwen3.8-27B and llama.cpp video support via web; added research findings and FFmpeg bundling requirement. Round 4 asked.
- 2026-08-23 — Round 4 answered (tunnel + own domain, password auth, max-context auto-fit with Q8/Q4 KV floor, extract-once packaging). Round 5 asked.
- 2026-08-23 — Round 5 answered (full self-host path, bundled+swappable embeddings, never-silently-degrade auto-fit, LM Studio-like UI). Investigated WACS and rejected it with evidence (§6.1). Round 6 asked.
- 2026-08-23 — Round 6 answered (in-process ACME, free-VRAM + multi-GPU split as P1/P2, zero hardware assumptions, ask-first updater). Wrote §5.1. Round 7 asked.
- 2026-08-23 — Round 7 answered (React+TS+Vite, one model with hot-swap, SQLite + export, ship unsigned). Round 8 asked.
- 2026-08-23 — Round 8 answered (no wizard, disk+tags+badges library, local-user priority, local git). Round 9 asked.
- 2026-08-23 — Round 9 answered (full scope, resumable queued downloads, local logs only, name kept). Round 10 asked.
- 2026-08-23 — Round 10 answered (cloudflared, per-chat + collections RAG, audio attachments only, full observability). Round 11 asked.
- 2026-08-23 — Round 11 answered (ask-on-close tray, full-parity gated web UI, JIT default + configurable, recommend+show-all quants).
- 2026-08-23 — **Plan fully written out**: architecture, module breakdown, auto-fit spec, security model, feature list, data model, build order, risks.
- 2026-08-23 — **Scope expanded: agentic harness.** Verified llama.cpp function calling, GBNF grammar enforcement, MCP client support, MCP v2.1 mandatory sandboxing, and the Anthropic Messages API endpoint. Added §6.5. Round 12 asked.
- 2026-08-23 — Round 12 answered (all core tool groups, MCP + agent primitives + code execution, tiered auto-approve, remote tools opt-in off by default; self-management excluded). Round 13 asked.
- 2026-08-23 — Round 13 answered (whole-machine scope, checkpoints + git, MCP-only extensibility, grammar enforcement only). Round 14 asked.
- 2026-08-23 — Round 14 answered (overridable hard blocks, user-selectable compaction strategy with manual trigger, plan-mode toggle, sequential sub-agents). Round 15 asked.
- 2026-08-23 — Round 15 answered (collapsed tool display, truncate-with-full-on-disk, MCP UI + config, memory-only instructions). Wrote agent runtime behaviour spec. Round 16 asked.
- 2026-08-23 — Round 16 answered (bundle Playwright Chromium, DuckDuckGo no-key search, bundle Python + Node, full session resume). Updated size budget to ~1.2–2.0 GB, extended build order to 15 phases, added agent risks and non-goals.
