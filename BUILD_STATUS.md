# Build status — 2026-08-23 (session 2)

Honest account of what exists. The spec is in [plan.md](plan.md); this tracks reality against it.

**Every phase has an implementation, and the core path now runs on a real model.**
Qwen3.8-27B-Q4_K_M was fetched, parsed, fitted, loaded, generated from, tool-called, and
unloaded on real hardware. What remains unverified is listed under "Not verified" below —
principally RAG, the tunnel, and the remote UI.

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
- **End-to-end run with a real model** (`scripts/load-test.ts`, executed inside Electron so it
  exercises the shipping code paths, not mocks):

```
model    ggml-org/Qwen3.8-27B-GGUF, Q4_K_M, 17.67 GB + 0.59 GB mmproj
parse    arch=qwen35, 64 blocks (16 attention + 48 SSM), vision+video+tools detected,
         mmproj auto-paired, tags auto-assigned
fit      131,072 ctx, 64/64 layers on GPU, split 51.4% / 48.6%
load     llama-server ready in 8.8s
verify   predicted 10.56 + 10.01 GB   actual 10.63 + 11.23 GB   (1.12x, self-correction fired)
generate 60 tokens, TTFT 1668 ms, 44.9 tok/s, coherent answer
tools    native tools parameter produced list_dir({"path":"C:Windows"})
unload   llama.loaded === null, VRAM returned to 13.36 + 11.71 GB free, no stray processes
```

Note the tool call dropped a backslash (`C:Windows`). That is the documented limit of grammar
enforcement: it guarantees a structurally valid call, not correct arguments. The tool would have
returned an error, which the loop feeds back to the model.

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
| Packaging | `build/portable.nsi` + `scripts/make-portable.mjs` | Custom NSIS launcher giving genuine extract-once: unpack to a versioned LOCALAPPDATA dir, marker written last, old runtimes cleaned on upgrade. electron-builder's own `portable` target was measured re-extracting 3 GB on every launch. |
| Vendor fetch | `scripts/fetch-vendor.mjs` | Resumable downloads of all 7 components, per-component selection. |

## Not verified / known gaps

Verified by the end-to-end run: llama-server spawn and argument construction, health polling,
streaming, tool-call accumulation, prediction verification, and unload.

**Packaging verified.** `LLM-Manager-0.1.0-portable.exe`, 0.86 GB, built and launch-tested:

```
payload        1.9 GB unpacked (vendor/.cache excluded — it held the archives the
               vendor files were extracted from, nearly doubling the exe)
compressed     0.86 GB, 45% of payload
first run      51.1s  (extracts to %LOCALAPPDATA%\LLMManager
untime-0.1.0, marker written)
second run      1.9s
third run       1.9s   -> 27x speedup, extraction genuinely skipped
on exit        runtime dir survives (electron-builder's portable target deletes it)
```

**A data-loss bug found by actually running the portable build.** `exeDir()` returned
`path.dirname(app.getPath('exe'))`, which for a portable build is the *extraction cache*, not the
folder holding the exe the user launched. So "models live beside the exe" resolved to
`%LOCALAPPDATA%\LLMManageruntime-0.1.0\LLMManagerModels`, and the relocation feature
faithfully moved an 18 GB library into it — filling the C: drive to 99%. Worse, the launcher's
upgrade cleanup deletes old `runtime-*` directories, so the next version bump would have taken
the models with it.

Fixed in three layers, because one guard is not enough for something unrecoverable:
1. The NSIS launcher exports `LLMM_PORTABLE_DIR=$EXEDIR`, and `exeDir()` prefers it.
2. `defaultModelsDir()` falls back to Documents if anything still resolves inside the cache.
3. `checkRelocation()` refuses to propose a move into the cache, and the launcher's cleanup
   skips any runtime directory containing a model library.

Covered by `scripts/paths-check.mjs` (16 assertions).

Two further corrections this exposed, both mine:
1. electron-builder's `portable` target does **not** give extract-once. `unpackDirName` only
   stabilises the directory *name*; the payload is re-extracted every launch (~24s measured) and
   the directory is deleted on exit. Replaced with a custom NSIS launcher.
2. NSIS cannot emit an output above ~2 GB. The first attempt failed at 1.94 GB, which is what
   surfaced the `.cache` packaging bug.

Still unexercised:
- The full agent loop (the tool-call *mechanism* is verified; the loop around it is not).
- RAG embedding — needs the second llama-server for the embedding model.
- Multimodal message construction and FFmpeg frame extraction.
- The tunnel, ACME issuance, and the remote web UI end to end.
- The API server's HTTP surface.

**Untested code paths that don't need a model:**
- Cross-volume relocation of a large models folder.
- The updater's exe-swap script.
- MCP against a real server.

**Known imperfections:**
- `fetch-vendor` matches llama.cpp release asset names by regex; upstream renaming those assets
  breaks it until the pattern is updated.
- The recurrent-state and mmproj footprints use empirical correction factors
  (`SSM_STATE_SAFETY`, `MMPROJ_OVERHEAD`) fitted to one model on one machine. They err high on
  purpose, but they are calibration, not derivation.
- Reserving for the projector cost enough budget to drop the chosen KV type from q8_0 to q4_0
  on this rig. Still the full 131K context, but a real tradeoff worth surfacing better in the UI.
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
