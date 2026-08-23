# Build status — 2026-08-23

Honest account of what exists after the first autonomous build session.
The full spec is in [plan.md](plan.md); this file tracks reality against it.

**Bottom line: this is a working foundation, not a finished app.** The plan describes 15 phases.
Roughly phases 0–3 are implemented and verified, phase 11 (agent core) is implemented but
unverified end-to-end, and phases 4–10 and 12–15 are not built. It compiles, builds, and the
core engine is tested — but it cannot yet run a model, because no llama.cpp binary is bundled.

---

## Verified working

Both TypeScript projects typecheck clean and `electron-vite build` succeeds.
`node scripts/smoke.mjs` runs 16 assertions against the auto-fit engine and GBNF compiler —
**16 passed, 0 failed**.

| Area | State | Notes |
|------|-------|-------|
| **Project scaffold** | Done | Electron + Vite + React 18 + TypeScript, strict mode, path aliases, electron-builder config pending. |
| **GGUF parser** (`src/main/models/gguf.ts`) | Done, untested on real files | Full metadata + tensor directory parse via a sliding-window reader, so a 40 GB model costs a few MB of reads. Huge arrays (`tokenizer.ggml.tokens`) are walked and discarded. Layer vs non-layer weight split computed for the fit engine. **Not yet run against an actual .gguf** — no model file was available. |
| **Hardware detection** (`src/main/hardware/gpu.ts`) | Done | nvidia-smi for exact free/total VRAM; WMI + display-class registry `qwMemorySize` as the vendor-neutral fallback (avoids the 4 GB `AdapterRAM` uint32 lie). Every device carries `freeIsMeasured`. |
| **Auto-fit engine** (`src/main/autofit/engine.ts`) | Done, unit-tested | See below. |
| **Model library** (`src/main/models/library.ts`) | Done | Scan, parse, cache by size+mtime, capability detection from mmproj metadata, auto-tags, disk usage. |
| **Relocation** (`src/main/models/relocation.ts`) | Done, untested | Breadcrumb comparison, same-volume rename fast path, cross-volume copy with progress/cancel, size verification before the source is removed. |
| **Permission engine** (`src/main/agent/permissions.ts`) | Done | Tiered auto-approve, remembered rules per folder, hard-block list, path canonicalisation so prompts show real targets. |
| **GBNF compiler** (`src/main/agent/gbnf.ts`) | Done, unit-tested | JSON Schema → GBNF, one branch per tool with the name pinned as a literal. |
| **Agent tools** | Done, untested | 20 tools: 10 filesystem, 7 exec, 3 web. |
| **Agent loop** (`src/main/agent/loop.ts`) | Done, untested | Tool-call parse, authorise, checkpoint, execute, feed result back. Plan mode narrows the catalog to read-tier. |
| **Checkpoints** (`src/main/agent/checkpoints.ts`) | Done, untested | Snapshot-before-write, rewind restores files and deletes ones the agent created. |
| **llama-server supervision** | Written, unrunnable | Spawn/health-poll/unload and SSE streaming client are complete; needs the binary. |
| **UI** | Minimal but real | Dashboard (live hardware meters), Models (fit plans + compatibility badges), Agent (collapsed tool cards, streaming), Settings. Permission modal wired end-to-end. |

### What the smoke tests actually prove

```
KV cache maths ......... closed-form match at 128K q8_0; q4_0 ≈ half
P1 free-vs-total VRAM .. 24 GB card with 6 GB free does NOT get a full-GPU plan
P2 multi-GPU split ..... 24 GB + 8 GB pair splits 78/22, not 50/50
never degrade silently . cramped 12 GB card returns needsUserChoice + real alternatives
unmeasured VRAM ........ AMD path notes the estimate rather than pretending
KV floor ............... never goes below q4_0
P4 KV reservation ...... plan.kvBytes equals full-context KV, so no mid-chat OOM
GBNF ................... root rule, literal tool names, both branches, primitives
```

A design consequence worth recording: **flash attention defaults on**, because without it the
attention buffer is `batch x context x heads x 2` — over 4 GB at 128K context on a 32-head model.
At the context targets in the plan, FA is not an optimisation, it is a precondition.

---

## Not built yet

Listed so nothing is quietly missing.

**Blocking a first real run:**
- **No bundled binaries.** `vendor/` is empty — no llama.cpp (CUDA/Vulkan/CPU), ffmpeg, cloudflared,
  Python embeddable, or ripgrep. `src/main/runtime/binaries.ts` knows where they go and
  `missingBinaries()` surfaces them in Settings, but nothing fetches them. **Until this is done the
  app cannot load a model, and the agent cannot generate.**
- No `electron-builder` target configuration, so no exe is produced yet.

**Specified but unimplemented:**
- Downloads: HuggingFace search, quant recommendation, resumable queue, mmproj auto-fetch, HF token.
- Chat: SQLite persistence, history, presets, export. (Types exist; storage layer does not.)
- Multimodal: image/audio/video attachment handling, frame-sampling controls.
- RAG: embedding model, extraction, chunking, vector search, collections.
- API server: `/v1/*` endpoints, JIT loading, API key, request log, local-priority queue.
- Remote: web UI serving, password + sessions, cloudflared, FreeDNS DDNS, ACME TLS, action gating.
- Observability: live inference stats, historical stats.
- Agent platform: MCP client + UI, sub-agents, task lists, persistent memory, scheduled tasks,
  context compaction, session resume.
- OS-control tools (screenshots, clipboard, input automation) and Playwright browser automation.
- Tray behaviour, updater, diagnostics bundle, packaging.

**Known gaps in what *is* built:**
- The GGUF parser has never seen a real file. Expect at least one field-name surprise on first run.
- `collectPaths` in the agent loop takes an unused `cwd` argument and does not resolve relative
  paths before checkpointing — relative-path writes may checkpoint the wrong location.
- The agent loop streams without the grammar attached, then parses JSON opportunistically. The
  grammar is compiled and exposed but not yet passed to the sampler, so tool calls are currently
  *parsed* rather than *enforced*. This is the single most important correctness gap in the agent.
- No context compaction, so long agent sessions will hit the context limit and fail.
- Relocation and checkpoint code paths are untested against real data.

---

## Next steps, in order

1. **Fetch vendor binaries** — llama.cpp release builds for CUDA/Vulkan/CPU, LGPL ffmpeg,
   cloudflared, Python embeddable, ripgrep. This unblocks everything else.
2. **Test the GGUF parser against a real model** and fix what it gets wrong.
3. **Attach the grammar to the sampler** in `LlamaRuntime.stream` so tool calls are enforced,
   not merely parsed.
4. First end-to-end run: load a model, generate, run one filesystem tool through the approval gate.
5. SQLite persistence, then downloads, then the remaining phases.

## Running it now

```bash
npm run dev
```

The window opens, hardware detection runs, Settings lists the missing binaries. The Models tab
will be empty until there are `.gguf` files beside the exe (dev mode: `./LLMManagerModels/`).
