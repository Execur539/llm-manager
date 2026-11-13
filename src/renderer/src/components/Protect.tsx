/**
 * Protecting a conversation or a document collection: the toolbar button, its dialog, and the
 * panel a locked one shows in place of what it holds.
 *
 * Protected means stored encrypted (see main/protect). Without a password it opens by itself, in
 * this app, on this Windows account. With one it stays closed until its password or recovery key
 * is given, and closes again when locked or when the app quits.
 */

import { useEffect, useRef, useState } from 'react'
import { invoke } from '../lib/api'
import { toast } from '../lib/store'
import Icon from './Icon'

export type ProtectKind = 'chat' | 'collection'

export interface ProtectState {
  protected?: boolean
  locked?: boolean
  password?: boolean
}

const nounOf = (kind: ProtectKind, agent?: boolean): string =>
  kind === 'collection' ? 'collection' : agent ? 'session' : 'conversation'

function errorText(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err)
  // Electron prefixes what a handler threw with where it came from; only the reason is for people.
  return raw.replace(/^Error invoking remote method '[^']+': (Error: )?/, '')
}

type View = 'main' | 'password' | 'recovery' | 'stop'

export function ProtectButton({
  kind,
  id,
  state,
  agent,
  disabled
}: {
  kind: ProtectKind
  id: string | null
  state: ProtectState | undefined
  /** An agent session, so the dialog says "session". */
  agent?: boolean
  disabled?: boolean
}): JSX.Element | null {
  const [open, setOpen] = useState(false)
  if (!id) return null
  const on = !!state?.protected
  return (
    <>
      <button
        className={on ? 'active' : undefined}
        disabled={disabled || !!state?.locked}
        onClick={() => setOpen(true)}
        title={
          on
            ? state?.password
              ? 'Stored encrypted, and opens only with its password'
              : 'Stored encrypted; opens by itself in this app'
            : `Store this ${nounOf(kind, agent)} encrypted, so other programs cannot read it`
        }
        data-testid="protect-button"
      >
        <Icon name="lock" size={14} />
        {on ? 'Protected' : 'Protect'}
      </button>
      {open && <ProtectDialog kind={kind} id={id} state={state ?? {}} agent={agent} onClose={() => setOpen(false)} />}
    </>
  )
}

function ProtectDialog({
  kind,
  id,
  state,
  agent,
  onClose
}: {
  kind: ProtectKind
  id: string
  state: ProtectState
  agent?: boolean
  onClose: () => void
}): JSX.Element {
  const noun = nounOf(kind, agent)
  const [view, setView] = useState<View>('main')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [recoveryKey, setRecoveryKey] = useState('')
  const [saved, setSaved] = useState(false)
  const firstField = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (view === 'password') firstField.current?.focus()
  }, [view])

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      // The recovery key is shown once; it is not dismissed by a stray key.
      if (e.key === 'Escape' && view !== 'recovery') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view, onClose])

  const run = async (fn: () => Promise<void>): Promise<void> => {
    setWorking(true)
    setError(null)
    try {
      await fn()
    } catch (err) {
      setError(errorText(err))
    } finally {
      setWorking(false)
    }
  }

  const protectNow = (): Promise<void> =>
    run(async () => {
      await invoke('protect:on', kind, id)
      toast(`Protected. This ${noun} is stored encrypted.`, 'success')
      onClose()
    })

  const submitPassword = (): Promise<void> =>
    run(async () => {
      if (password.length < 8) throw new Error('Use at least 8 characters.')
      if (password !== confirm) throw new Error('The two passwords are not the same.')
      const result = await invoke<{ recoveryKey: string }>('protect:set-password', kind, id, password)
      setPassword('')
      setConfirm('')
      setRecoveryKey(result.recoveryKey)
      setView('recovery')
    })

  let body: JSX.Element
  if (view === 'password') {
    body = (
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void submitPassword()
        }}
      >
        <h2>{state.password ? 'Change the password' : 'Set a password'}</h2>
        <div className="confirm-body">
          <p>
            The {noun} will open only with this password, or with a recovery key you are shown next. Programs running as you
            cannot read it while it is locked.
          </p>
        </div>
        <label className="protect-field">
          <span className="faint">Password</span>
          <input
            ref={firstField}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            data-testid="protect-password"
          />
        </label>
        <label className="protect-field">
          <span className="faint">The same again</span>
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            data-testid="protect-password-confirm"
          />
        </label>
        {error && <div className="protect-error" role="alert">{error}</div>}
        <div className="actions">
          <button type="button" onClick={() => setView('main')} disabled={working}>
            Back
          </button>
          <button type="submit" className="primary" disabled={working || !password} data-testid="protect-password-save">
            {working ? 'Setting…' : 'Set password'}
          </button>
        </div>
      </form>
    )
  } else if (view === 'recovery') {
    body = (
      <>
        <h2>Your recovery key</h2>
        <div className="confirm-body">
          <p>
            It opens this {noun} if the password is forgotten. Keep it somewhere safe, away from this computer. It is shown only
            now, and without the password or this key the {noun} cannot be opened by anyone.
          </p>
        </div>
        <div className="recovery-key" data-testid="recovery-key">
          <code className="mono">{recoveryKey}</code>
          <button
            className="tiny"
            onClick={() => {
              void navigator.clipboard.writeText(recoveryKey).then(
                () => toast('Recovery key copied.', 'success'),
                () => toast('Could not copy it; write it down instead.', 'error')
              )
            }}
          >
            <Icon name="copy" size={13} /> Copy
          </button>
        </div>
        <label className="protect-check">
          <input type="checkbox" checked={saved} onChange={(e) => setSaved(e.target.checked)} data-testid="recovery-saved" />
          I have kept the recovery key somewhere safe
        </label>
        <div className="actions">
          <button className="primary" disabled={!saved} onClick={onClose} data-testid="recovery-done">
            Done
          </button>
        </div>
      </>
    )
  } else if (view === 'stop') {
    body = (
      <>
        <h2>Stop protecting this {noun}?</h2>
        <div className="confirm-body">
          <p>Everything it holds is decrypted and stored as plain text again, readable by any program that opens the app's files.</p>
        </div>
        {error && <div className="protect-error" role="alert">{error}</div>}
        <div className="actions">
          <button onClick={() => setView('main')} disabled={working}>
            Cancel
          </button>
          <button
            className="danger"
            disabled={working}
            onClick={() =>
              void run(async () => {
                await invoke('protect:off', kind, id)
                toast(`This ${noun} is no longer protected.`, 'info')
                onClose()
              })
            }
            data-testid="protect-stop-confirm"
          >
            {working ? 'Decrypting…' : 'Stop protecting'}
          </button>
        </div>
      </>
    )
  } else if (!state.protected) {
    body = (
      <>
        <h2>
          <Icon name="lock" size={17} /> Protect this {noun}
        </h2>
        <div className="confirm-body">
          <p>
            {kind === 'collection'
              ? 'Its documents, their text and their index are stored encrypted.'
              : 'Its messages and thinking, its title, its attachments and the model’s saved state of it are stored encrypted.'}{' '}
            Programs that read the app’s files directly see nothing they can use.
          </p>
          <p>
            It opens by itself here, in this app on your Windows account. Add a password if it should stay closed even to
            programs running as you.
          </p>
        </div>
        {error && <div className="protect-error" role="alert">{error}</div>}
        <div className="actions">
          <button onClick={onClose} disabled={working}>
            Cancel
          </button>
          <button onClick={() => setView('password')} disabled={working} data-testid="protect-with-password">
            Protect with a password…
          </button>
          <button className="primary" onClick={() => void protectNow()} disabled={working} data-testid="protect-confirm">
            {working ? 'Encrypting…' : 'Protect'}
          </button>
        </div>
      </>
    )
  } else {
    body = (
      <>
        <h2>
          <Icon name="lock" size={17} /> {state.password ? 'Protected with a password' : 'Protected'}
        </h2>
        <div className="confirm-body">
          <p>
            {state.password
              ? `Stored encrypted, and opens only with its password or recovery key. It stays open until you lock it or close the app.`
              : `Stored encrypted. It opens by itself here, in this app on your Windows account.`}
          </p>
        </div>
        {error && <div className="protect-error" role="alert">{error}</div>}
        <div className="actions protect-actions">
          <button onClick={() => setView('stop')} disabled={working} data-testid="protect-stop">
            Stop protecting
          </button>
          {state.password && (
            <button
              disabled={working}
              onClick={() =>
                void run(async () => {
                  await invoke('protect:remove-password', kind, id)
                  toast('Password removed. It opens by itself in this app again.', 'info')
                  onClose()
                })
              }
              data-testid="protect-remove-password"
            >
              Remove password
            </button>
          )}
          <button onClick={() => setView('password')} disabled={working} data-testid="protect-set-password">
            {state.password ? 'Change password…' : 'Set a password…'}
          </button>
          {state.password ? (
            <button
              className="primary"
              disabled={working}
              onClick={() =>
                void run(async () => {
                  await invoke('protect:lock', kind, id)
                  onClose()
                })
              }
              data-testid="protect-lock"
            >
              {working ? 'Locking…' : 'Lock now'}
            </button>
          ) : (
            <button className="primary" onClick={onClose} disabled={working}>
              Done
            </button>
          )}
        </div>
      </>
    )
  }

  return (
    <div className="overlay" onClick={() => view !== 'recovery' && !working && onClose()} data-testid="protect-dialog">
      <div className="modal confirm-modal" role="dialog" aria-modal="true" aria-label={`Protect this ${noun}`} onClick={(e) => e.stopPropagation()}>
        {body}
      </div>
    </div>
  )
}

/** In place of a locked conversation's or collection's contents: its password or recovery key, asked for. */
export function LockedPanel({
  kind,
  id,
  agent,
  password = true
}: {
  kind: ProtectKind
  id: string
  agent?: boolean
  /** False when it has no password and Windows could not open its key: there is nothing to type. */
  password?: boolean
}): JSX.Element {
  const [secret, setSecret] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const field = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setSecret('')
    setError(null)
    field.current?.focus()
  }, [id])

  const unlock = async (): Promise<void> => {
    if (!secret) return
    setWorking(true)
    setError(null)
    try {
      const ok = await invoke<boolean>('protect:unlock', kind, id, secret)
      if (!ok) {
        setError('That is not its password or its recovery key.')
        field.current?.select()
      } else {
        setSecret('')
      }
    } catch (err) {
      setError(errorText(err))
    } finally {
      setWorking(false)
    }
  }

  if (!password) {
    return (
      <div className="locked-panel" data-testid="locked-panel">
        <div className="locked-icon">
          <Icon name="lock" size={22} />
        </div>
        <h2>This {nounOf(kind, agent)} cannot be opened here</h2>
        <p className="faint">
          It was protected without a password, so only the Windows account it was protected on can open it, and this account
          cannot. Sign in to that account and open it there.
        </p>
      </div>
    )
  }

  return (
    <div className="locked-panel" data-testid="locked-panel">
      <div className="locked-icon">
        <Icon name="lock" size={22} />
      </div>
      <h2>This {nounOf(kind, agent)} is locked</h2>
      <p className="faint">Enter its password, or its recovery key, to open it.</p>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void unlock()
        }}
      >
        <input
          ref={field}
          type="password"
          value={secret}
          onChange={(e) => setSecret(e.target.value)}
          placeholder="Password or recovery key"
          autoComplete="current-password"
          aria-label="Password or recovery key"
          data-testid="unlock-input"
        />
        <button type="submit" className="primary" disabled={working || !secret} data-testid="unlock-button">
          {working ? 'Unlocking…' : 'Unlock'}
        </button>
      </form>
      {error && (
        <div className="protect-error" role="alert" data-testid="unlock-error">
          {error}
        </div>
      )}
    </div>
  )
}
