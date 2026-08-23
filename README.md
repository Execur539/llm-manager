# LLM Manager

An all-in-one local LLM control panel for Windows: it finds and downloads models, works out how
to fit them on your actual hardware, runs them, chats with them, serves them over an API, exposes
them securely to the internet, and drives them as an agent with real tools.

No telemetry. No cloud providers. Everything runs on your machine.

---

## What it does

**Models.** Search HuggingFace, get a quant recommended for *your* free VRAM, download with a
resumable queue, and see at a glance which models fit before loading anything. Models live in a
folder beside the app; move the app and it offers to bring them along.

**Auto-fit.** The centrepiece. It maximises context length subject to a KV-cache quality floor,
and is built specifically to beat four things other apps get wrong:

| | Failure | What we do |
|---|---|---|
| P1 | Sizing against *total* VRAM, ignoring what the desktop already holds | Measure **free** VRAM per adapter, re-checked immediately before every load |
| P2 | Splitting evenly across mismatched GPUs so the small card OOMs | Split proportional to **real free capacity**, and show the ratio |
| P3 | Leaving 4–6 GB unused "to be safe" | Fill to a measured margin that self-corrects from observed loads |
| P4 | Fitting at load time, then OOMing 20K tokens into a chat | Reserve the **entire** context's KV cache up front |

When the target genuinely cannot be met, it never silently degrades — it shows the tradeoffs with
predicted speed and context and lets you choose.

**Chat.** Streaming conversations with history, search and export. Images, audio and video
attachments on capable models. Document collections for retrieval-grounded answers.

**Agent.** A supervised tool-calling loop in the spirit of Claude Code: read and write files, run
commands and background jobs, search and browse the web, control the desktop, execute Python and
JavaScript, and connect any MCP server for everything else. Reads run freely; writes and commands
ask first, and the prompt shows the fully-resolved target — not the model's description of it.
Every turn is checkpointed so it can be rewound.

**Serve.** An OpenAI-compatible API with just-in-time model loading, plus an Anthropic Messages
endpoint — so Claude-compatible clients can point at a local model. Local requests always take
priority over remote ones.

**Remote.** Reach your machine from anywhere: a zero-config Cloudflare tunnel, or your own domain
with dynamic DNS and auto-renewing Let's Encrypt certificates issued in-process. Password + session
auth, rate limiting, and destructive actions gated to the desktop.

---

## Getting started

```bash
npm install
npm run fetch-vendor
npm run dev
```

`fetch-vendor` downloads the third-party binaries the app bundles — llama.cpp (CPU/Vulkan/CUDA),
FFmpeg, cloudflared, Python, ripgrep, Chromium and an embedding model. It is roughly **1.5 GB** and
resumes if interrupted.

```bash
node scripts/fetch-vendor.mjs --list      # see the components and their sizes
node scripts/fetch-vendor.mjs llama       # fetch just one
```

Without them the app still runs — it just reports which features are unavailable rather than
failing mysteriously.

## Building

```bash
npm run typecheck        # both tsconfigs
npm test                 # 78 assertions over the pure logic
npm run pack:installer   # NSIS installer
npm run pack:portable    # single-file portable exe
```

The portable build sets `portable.unpackDirName` to a fixed name, which makes electron-builder
unpack to a stable directory and **reuse it** on subsequent launches. Without it the default is a
fresh uuid-named temp directory per launch, which with a multi-gigabyte payload would mean
re-extracting everything every single time.

Builds are unsigned, so Windows SmartScreen warns on first run.

---

## Where things live

```
<app folder>/
├── LLM Manager.exe
├── LLMManagerModels/          models, created on first run
└── vendor/                    bundled binaries (dev: ./vendor)

%APPDATA%\LLMManager\
├── settings.json              all settings
├── secrets.json               DPAPI-encrypted: password hash, API key, tokens
├── llmmanager.db              chats, sessions, stats, RAG, memory
├── models-path.json           breadcrumb that makes relocation possible
├── checkpoints/               per-turn file snapshots for rewind
├── tool-output/               full text of truncated tool results
└── logs/                      rotating logs
```

## Layout

```
src/
├── main/
│   ├── autofit/      the fit engine
│   ├── models/       GGUF parser, library, relocation
│   ├── hardware/     GPU + backend detection
│   ├── runtime/      llama-server supervision, bundled binary resolution
│   ├── agent/        tool loop, permissions, tools/, MCP, memory, checkpoints
│   ├── downloads/    HuggingFace search + resumable queue
│   ├── rag/          extraction, chunking, embeddings, retrieval
│   ├── api/          OpenAI + Anthropic endpoints, priority queue
│   ├── remote/       auth, tunnel, DDNS, ACME, web server
│   ├── chat/         persistence, multimodal message building
│   ├── stats/ log/ update/ storage/
│   └── bridge.ts     one handler map shared by desktop IPC and remote HTTP
├── preload/          the only renderer↔main surface
├── renderer/         React + TypeScript UI (also served to remote browsers)
└── shared/           types crossing the boundary
```

The bridge is deliberately one map. Desktop IPC and the remote web UI call the same handlers, so
"full parity" is true by construction rather than by discipline.

## Security posture

The agent runs machine-wide with your permissions. What stands between a confused model and your
filesystem:

- **Tiered approval** — writes and commands prompt, with the resolved path or command shown.
- **Hard blocks** — disk formatting, system-root deletion, bootloader writes and disabling security
  tooling are refused in every mode. Overridable only via a buried setting requiring typed confirmation.
- **Checkpoints** — every write is snapshotted first; any turn can be rewound.
- **Tool output is data** — the system prompt is explicit that instructions found in files, command
  output or web pages are never to be followed.
- **Remote tools are off by default** — remote sessions can read and chat but not write or execute
  unless you explicitly enable it.
- **The inference server never binds beyond loopback.** Everything from outside goes through the
  authenticated layer.

See [BUILD_STATUS.md](BUILD_STATUS.md) for what is verified, what is untested, and what is known
to be incomplete.
