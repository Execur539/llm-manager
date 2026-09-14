/**
 * Fold what a Continue produced into the message it continues.
 *
 * Shared by the agent loop, which stores the result, and the agent view, which draws it while it
 * streams — so the message on screen is the same text before and after the finished copy arrives.
 *
 * A message stopped inside its thinking has reasoning and no answer. llama.cpp resumes such a
 * message inside the thought, so the new reasoning carries straight on from the old, possibly
 * mid-word, and is joined with nothing between. Where the message already had an answer, any new
 * thinking is a separate episode and gets a paragraph break.
 */
export function continuedMessage(
  base: { content: string; reasoning?: string },
  text: string,
  thinking: string
): { content: string; reasoning?: string } {
  if (!base.content && base.reasoning) {
    return { content: text, reasoning: base.reasoning + thinking }
  }
  const more = thinking.trim()
  const reasoning = more ? (base.reasoning ? `${base.reasoning}\n\n${more}` : more) : base.reasoning
  return { content: base.content + text, reasoning: reasoning || undefined }
}
