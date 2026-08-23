# Build status — 2026-08-23 (session 2)

Honest account of what exists. The spec is in [plan.md](plan.md); this tracks reality against it.

**Every phase in the plan now has an implementation.** What that does *not* mean is that
everything has been run: no llama.cpp binary has been executed, because `vendor/` is empty on this
machine and the download is ~1.5 GB. So the code paths that require a live model — loading,
generation, the agent loop end to end, RAG embedding, multimodal — are **written and typechecked
but never executed**. Treat that as the headline caveat.

## Verified

- `npm run typecheck` — clean on both tsconfigs.
- `npm run build` — main, preload and renderer all bundle.
- `npm test` — **94 assertions, 94 passing**, covering the auto-fit engine, GBNF compiler,
  permission engine and hard-block list, the GGUF parser (against a synthesised GGUF file),
  quant recommendation, and adapter filtering.
- **The app launches.** Verified on this machine: it initialised `%APPDATA%\LLMManager`, created
  the SQLite schema, wrote the relocation breadcrumb, scanned the (empty) library, and detected
  real hardware — `backend=cuda`, RTX 5080 + RTX 4070 Ti + AMD iGPU.
- **The auto-fit engine was run against that real hardware** (`node scripts/hw-check.mjs`) and
  produced sane plans for 8B, 27B and 70B models.

What the tests actually prove:

```
KV maths ............... closed-form match; linear in context; q4 ≈ 53% of q8
flash attention ........ no-FA attention buffer is >10x larger at 128K (why FA defaults on)
P1 free-vs-total ....... 24 GB card with 6 GB free gets no full-GPU plan
P2 multi-GPU ........... 24+8 GB splits 78/22; three-way split is proportional
P4 KV reservation ...... plan.kvBytes equals the whole configured context
never degrades ......... cramped card returns needsUserChoice + rationale-carrying options
overrides .............. kv type, batch, flash-attn and context overrides all survive planning
trained ceiling ........ never plans past the model's trained context, and says why
verification ........... over/under-prediction both flagged; silence when close
GBNF ................... literal tool names, one branch per tool, required vs optional correct
hard blocks ............ 7 destructive commands blocked, 5 lookalikes allowed
                         (`rm -rf /` blocked, `rm -rf ./node_modules` allowed)
permission tiers ....... reads silent, writes prompt, allow-tool remembered, denial explained
prompts ................ relative paths resolved before display
GGUF ................... real parse of a synthesised file: header, kv, elided arrays, tensors,
                         head_dim derivation, per-layer vs non-layer weight split
recommendation ......... picks best quant that fits, steps down on smaller cards, never an mmproj
mixed rigs ............. CUDA excludes an AMD iGPU; Vulkan skips it only when discrete cards exist
phantom adapters ....... Parsec / Virtual Desktop / IDD / Basic Display rejected; real GPUs kept
```

### Two bugs found by running against real hardware

Worth recording, because both were invisible to synthetic tests:

1. **Virtual display adapters counted as GPUs.** Screen-sharing tools (Parsec, Virtual Desktop)
   install adapters that WMI reports like graphics cards. They appeared in the device list with
   0 VRAM.
2. **The AMD iGPU was given a 2.6% tensor split under the CUDA backend** — where it cannot
   participate at all. That would have produced a broken `llama-server` invocation.

Together they were stealing budget share: Qwen3.8-27B planned **20,480 tokens** of context before
the fix and **89,088** after, on the same machine. Both are now filtered, with the exclusion stated
in the plan's notes rather than applied silently, and both are covered by regression tests.

## Implemented

| Area | Module | Notes |
|---|---|---|
| GGUF parsing | `models/gguf.ts` | Sliding-window reads; a 40 GB model costs a few MB. Tested against a synthesised file, **not a real one**. |
| Hardware detection | `hardware/gpu.ts` | nvidia-smi for true free VRAM; registry `qwMemorySize` fallback avoids the 4 GB `AdapterRAM` uint32 lie. |
| Auto-fit | `autofit/engine.ts` | Fully tested. Includes post-load verification that feeds back into the headroom margin. |
| Model library | `models/library.ts` | Scan, cache by size+mtime, capability detection from mmproj, auto-tags, disk usage. |
| Relocation | `models/relocation.ts` | Same-volume rename fast path; cross-volume copy with verify-before-delete. **Untested against real data.** |
| Persistence | `storage/db.ts` | `node:sqlite` (Electron 38 / Node 22) — no native module to rebuild. 2 migrations. |
| Chat | `chat/repo.ts` | Chats and agent sessions in one table; history, search, export to MD/JSON, auto-titling, full session resume. |
| Multimodal | `chat/multimodal.ts` | Images and audio as content parts; video via FFmpeg frame sampling, denser for natively-video-trained models. |
| Downloads | `downloads/` | HF search, quant recommendation, resumable queue (range requests), pause/cancel, mmproj auto-fetch, token support. |
| llama runtime | `runtime/llama.ts` | Spawn/health/unload, SSE streaming, **native `tools` parameter**, tool-call fragment accumulation, timings, tokenizer count. |
| Agent loop | `agent/loop.ts` | Native tool calls with prose-JSON fallback, tiered permissions, checkpoints, compaction, sequential depth-limited sub-agents. |
| Tools | `agent/tools/` | **44 built-in**: 10 filesystem, 7 exec, 3 web, 10 system/OS, 8 browser, 3 data, 7 agentic (some overlap in counts by group). |
| MCP client | `agent/mcp.ts` | stdio + streamable HTTP, JSON-RPC implemented directly, namespaced tools, tier from server annotations. |
| Memory | `agent/memory.ts` | Persistent, user-editable, injected into the system prompt. |
| RAG | `rag/index.ts` | Separate embedding server, PDF/text extraction, overlap chunking, Float32 blobs, brute-force cosine. |
| API server | `api/server.ts` | `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/messages` (Anthropic), JIT loading, API key with constant-time compare, priority queue. |
| Remote | `remote/` | Password+scrypt+HMAC sessions, lockout, cloudflared tunnel, FreeDNS DDNS, in-process ACME, SSE event fan-out, desktop-only action gate. |
| Observability | `stats/index.ts` | Live tok/s, TTFT, KV fill, request log, daily totals. |
| Logging | `log/index.ts` | Rotating logs, crash handlers, redacting diagnostics bundle. |
| Updater | `update/index.ts` | Ask-first, staged swap via a post-exit helper script. |
| Packaging | `scripts/make-portable.mjs` | 7-Zip SFX with `OverwriteMode=2` for genuine extract-once. |
| Vendor fetch | `scripts/fetch-vendor.mjs` | Resumable downloads of all 7 components, per-component selection. |

## Not verified / known gaps

**Nothing involving a live model has been run.** In particular:
- llama-server spawn, health-polling and argument construction.
- Streaming, tool-call accumulation, and the whole agent loop.
- RAG embedding (needs the embedding model and a second llama-server).
- Multimodal message construction and FFmpeg frame extraction.
- The tunnel, ACME issuance, and the remote web UI end to end.

**Untested code paths that don't need a model:**
- Cross-volume relocation of a large models folder.
- The portable SFX build (needs 7-Zip's `7zSD.sfx`, which ships with the LZMA SDK, not the installer).
- The updater's exe-swap script.
- MCP against a real server.

**Known imperfections:**
- The GGUF parser has still never seen a file produced by a real converter. Expect at least one
  metadata key surprise.
- `fetch-vendor` matches llama.cpp release asset names by regex; upstream renaming those assets
  breaks it until the pattern is updated.
- The close-to-tray dialog treats the choice as sticky rather than offering a checkbox, because
  `showMessageBoxSync` does not return checkbox state.
- Chromium revision is read from `playwright-core/browsers.json` with a hardcoded fallback that
  will drift.
- PDF extraction handles text PDFs only; scanned documents are rejected with a clear message rather
  than silently returning noise. No OCR.
- No unit tests for the queue, MCP wire protocol, or the remote auth flow.

## Deliberately not built

Recorded so they read as decisions, not omissions:
- Cloud API providers, API-key management for them, spend tracking (Round 3).
- Agent self-management of LLM Manager (Round 12).
- Custom script tools — MCP is the single extension path (Round 13).
- Per-folder instructions files; memory is the only carried context (Round 15).
- A dedicated tool-call repair/retry loop (Round 13).
- Parallel sub-agents (Round 14).
- Microphone capture and TTS (Round 10).
- Telemetry of any kind (Round 9).

## Next

1. `npm run fetch-vendor`, then load a real model — that exercises most of the untested surface at once.
2. Fix whatever the GGUF parser gets wrong on first contact with a real file.
3. End-to-end agent run: one read, one approved write, one command.
4. Then the tunnel, then packaging.
