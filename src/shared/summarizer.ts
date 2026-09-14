/**
 * The helper model the app suggests for compaction.
 *
 * Chosen for this job rather than for chat. Qwen3.5-2B holds 262,144 tokens natively and only six
 * of its twenty-four layers keep a KV cache, so a whole chunk costs a few hundred megabytes beside
 * a large model rather than gigabytes. It does not think by default, which matters in a summariser:
 * one that reasoned first would spend the time it exists to save. Its model card warns the 2B is
 * more prone to repetition loops than larger Qwen3.5 sizes, which is why every helper summary is
 * checked for one and handed back to the loaded model when a loop is caught.
 */
export const RECOMMENDED_SUMMARIZER = {
  repo: 'unsloth/Qwen3.5-2B-GGUF',
  file: 'Qwen3.5-2B-Q4_K_M.gguf',
  label: 'Qwen3.5-2B (Q4_K_M)',
  bytes: 1_280_835_840
} as const
