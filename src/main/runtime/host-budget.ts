/**
 * How much system RAM to let llama.cpp use for its prompt cache and context checkpoints.
 *
 * Costs no VRAM and buys back the thing that actually hurts on a long conversation: a hybrid
 * model cannot truncate its cache to resume part-way, so it resumes from a checkpoint or it
 * reprocesses the entire prompt. Each checkpoint is on the order of 150 MB, and the default
 * budget of 8 GB is a handful of them.
 *
 * Measured against the RAM that will still be free once the model is loaded, not the RAM free
 * now. The figure is taken before llama-server starts, and a model whose routed experts live in
 * system RAM is about to take tens of gigabytes of it: Qwen3.8-Flash-Next's 74 GB of experts left
 * a quarter of the pre-load figure, 26 GB of prompt cache, to be squeezed into what was left over.
 * A file-backed per-layer embedding table is not resident, but the rows a long session reads are
 * held in the operating system's file cache, so a share of it is kept back as well.
 *
 * Capped, and a machine with little to spare gets llama.cpp's default left alone: the figure is a
 * ceiling llama.cpp fills opportunistically rather than an allocation, other things on the
 * machine need room, and past a few dozen gigabytes the cache is holding conversations nobody
 * will return to.
 */

const MiB = 1024 * 1024
const DEFAULT_MB = 8192
const CAP_MB = 49152
/** Below this there is nothing to give, and taking a quarter of it would hurt. */
const MIN_SPARE_MB = 12288
/** Most of the file cache worth reserving for a file-backed table, however large the table. */
const FILE_CACHE_RESERVE_CAP = 8192 * MiB

export function hostCacheBudgetMb(freeBytes: number, residentHostBytes = 0, fileBackedBytes = 0): number {
  const fileCacheReserve = Math.min(Math.max(0, fileBackedBytes) / 4, FILE_CACHE_RESERVE_CAP)
  const spareMb = Math.floor((freeBytes - Math.max(0, residentHostBytes) - fileCacheReserve) / MiB)
  if (spareMb < MIN_SPARE_MB) return 0
  const quarter = Math.floor(spareMb / 4)
  return Math.max(DEFAULT_MB, Math.min(quarter, CAP_MB))
}
