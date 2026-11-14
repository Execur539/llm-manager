/**
 * The Compact button with its second way of compacting, a card tucked under it that comes down
 * when the button is hovered or focused.
 *
 * Compact asks the model for a summary. Quick compact involves no model: the older part of the
 * conversation is sent in a shorter form from then on, worked out by rules in an instant, and the
 * model can look back into what was left out with its recall tools. Because nothing is generated,
 * what each strength would leave is known before anything is done, so the strength is chosen from
 * a list that shows it.
 */

import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { invoke } from '../lib/api'
import { toast } from '../lib/store'
import ConfirmDialog from './ConfirmDialog'

interface QuickOption {
  level: number
  name: string
  detail: string
  afterTokens: number
}

interface QuickPlan {
  ok: boolean
  message: string
  beforeTokens?: number
  floorTokens?: number
  targetTokens?: number
  windowTokens?: number
  messages?: number
  options?: QuickOption[]
  recommended?: number
}

const n = (v: number | undefined): string => Math.round(v ?? 0).toLocaleString()

export default function CompactMenu({
  stateKey,
  disabled,
  children
}: {
  /** chat:<id> or agent:<id>. */
  stateKey: string | null
  disabled?: boolean
  /** The Compact button itself. */
  children: React.ReactNode
}): JSX.Element {
  const [plan, setPlan] = useState<QuickPlan | null>(null)
  const [level, setLevel] = useState(0)
  const [working, setWorking] = useState(false)
  /*
   * Whether the second card is down. Kept in state rather than left to :hover, so that while it is
   * tucked away it is also out of reach of the keyboard and of screen readers: a button nobody can
   * see should not take focus or be announced.
   */
  const [down, setDown] = useState(false)
  // Where nothing can be hovered, it is simply shown beside Compact.
  const [touch, setTouch] = useState(false)
  useEffect(() => {
    const q = window.matchMedia('(hover: none)')
    const read = (): void => setTouch(q.matches)
    read()
    q.addEventListener('change', read)
    return () => q.removeEventListener('change', read)
  }, [])
  const shown = down || touch
  /*
   * Closing waits a moment. The pointer on its way down to the card crosses a gap, and can get
   * there before the card has finished sliding; closing at once pulled the card away from it.
   */
  const closing = useRef<ReturnType<typeof setTimeout> | null>(null)
  const open = (): void => {
    if (closing.current) clearTimeout(closing.current)
    closing.current = null
    setDown(true)
  }
  const close = (): void => {
    if (closing.current) clearTimeout(closing.current)
    closing.current = setTimeout(() => setDown(false), 350)
  }
  useEffect(() => () => void (closing.current && clearTimeout(closing.current)), [])

  const preview = async (): Promise<void> => {
    if (!stateKey || working) return
    // Let go of the card, or it stays down after the click.
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    setDown(false)
    setWorking(true)
    try {
      const p = await invoke<QuickPlan>('compact:preview', stateKey)
      if (p.ok && p.options?.length) {
        setLevel(p.recommended ?? p.options[0].level)
        setPlan(p)
      } else toast(p.message, 'info')
    } catch (err) {
      toast(`Quick compact failed: ${err instanceof Error ? err.message : String(err)}`, 'error')
    } finally {
      setWorking(false)
    }
  }

  const apply = async (): Promise<void> => {
    setPlan(null)
    if (!stateKey) return
    try {
      const r = await invoke<{ ok: boolean; message: string; beforeTokens?: number; afterTokens?: number }>('compact:quick', stateKey, level)
      toast(
        r.ok ? `Compacted: about ${n(r.beforeTokens)} tokens down to about ${n(r.afterTokens)}.` : r.message,
        r.ok ? 'success' : 'info'
      )
    } catch (err) {
      toast(`Quick compact failed: ${err instanceof Error ? err.message : String(err)}`, 'error')
    }
  }

  const share = (tokens: number): number => (plan?.windowTokens ? Math.round((tokens / plan.windowTokens) * 100) : 0)

  return (
    <span
      className={`compact-menu${shown ? ' down' : ''}`}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) close()
      }}
    >
      {children}
      {stateKey && (
        /*
         * Tucked away, the card is only a shape: its button exists once it is down, so nothing
         * unseen can take focus, be announced, or sit under the Compact button as a second control.
         */
        <span className="compact-drop" role={shown ? 'menu' : undefined} aria-hidden={!shown || undefined}>
          {shown && (
            <button
              role="menuitem"
              disabled={disabled || working}
              onClick={() => void preview()}
              title="Shorten the older part of the conversation by rules, with no model involved: instant, and the model can still look back into what is left out"
              data-testid="quick-compact"
            >
              Quick compact
            </button>
          )}
        </span>
      )}
      {/* Outside the toolbar: a full-window overlay inside it can be confined to it by the toolbar's own styling. */}
      {createPortal(
      <ConfirmDialog
        open={!!plan}
        title="Quick compact"
        confirmLabel="Compact"
        body={
          plan && (
            <>
              <p>
                Now about <strong>{n(plan.beforeTokens)}</strong> tokens ({share(plan.beforeTokens ?? 0)}% of the context window).
                Choose how hard to shorten the older {n(plan.messages)} messages:
              </p>
              <div className="quick-options" role="radiogroup" aria-label="How hard to compact">
                {plan.options?.map((o) => (
                  <label key={o.level} className={`quick-option${o.level === level ? ' chosen' : ''}`} data-testid={`quick-level-${o.level}`}>
                    <input type="radio" name="quick-level" checked={o.level === level} onChange={() => setLevel(o.level)} />
                    <span className="quick-option-text">
                      <span className="quick-option-name">
                        {o.name}
                        {o.level === plan.recommended && <span className="badge">suggested</span>}
                      </span>
                      <span className="faint">{o.detail}</span>
                    </span>
                    <span className="quick-option-size">
                      {n(o.afterTokens)}
                      <span className="faint"> · {share(o.afterTokens)}%</span>
                    </span>
                  </label>
                ))}
              </div>
              <p>
                Nothing is deleted. What you said, which files were changed and what failed are always kept, and the model can
                look back into everything else. The most recent messages stay word for word.
              </p>
              <p>The next message reads the shortened conversation once. A saved model state will no longer match it.</p>
            </>
          )
        }
        onConfirm={() => void apply()}
        onCancel={() => setPlan(null)}
      />,
      document.body
      )}
    </span>
  )
}
