/**
 * Attachment handling for the composer: a picker, drag-and-drop, and the chips that show what
 * is staged.
 *
 * Two ways in, because there are two shells. The desktop app gets the real path of a dropped
 * file from the preload, which costs nothing even for a 4 GB video. A remote browser has no path
 * to give, so it reads the bytes and posts them; the main process stages the file and the rest
 * of the pipeline cannot tell the difference.
 */

import { useCallback, useRef, useState } from 'react'
import type { AttachmentInfo } from '@shared/types'
import { invoke, fmtBytes, isDesktop } from '../lib/api'
import { toast } from '../lib/store'
import Icon, { type IconName } from './Icon'

/** Anything larger has to come in by path; posting it as base64 would blow up memory. */
const MAX_UPLOAD_BYTES = 48 * 1024 * 1024

const KIND_ICON: Record<AttachmentInfo['kind'], IconName> = {
  image: 'documents',
  video: 'documents',
  audio: 'documents',
  doc: 'documents'
}

const KIND_LABEL: Record<AttachmentInfo['kind'], string> = {
  image: 'image',
  video: 'video',
  audio: 'audio',
  doc: 'text'
}

export function useAttachments(): {
  items: AttachmentInfo[]
  busy: boolean
  pick: () => Promise<void>
  addFiles: (files: FileList | File[]) => Promise<void>
  remove: (path: string) => void
  clear: () => void
} {
  const [items, setItems] = useState<AttachmentInfo[]>([])
  const [busy, setBusy] = useState(false)
  /** `pick` is defined above `addFiles`, and in the remote shell it needs to call it. */
  const addFilesRef = useRef<((files: FileList | File[]) => Promise<void>) | null>(null)

  const merge = useCallback((incoming: AttachmentInfo[]) => {
    setItems((prev) => {
      const seen = new Set(prev.map((a) => a.path))
      return [...prev, ...incoming.filter((a) => a && !seen.has(a.path))]
    })
  }, [])

  /*
   * Two pickers, because there are two shells.
   *
   * The desktop opens a native dialog through the bridge, which is the only way to get a real
   * path. A remote browser cannot: that dialog would open on the host machine, in front of
   * whoever is sitting at it, and the request would hang until they dismissed it. The browser's
   * own file input goes through the upload path instead, which is what drag-and-drop already
   * used — until now the paperclip was the one way in that a remote session could not use.
   */
  const pick = useCallback(async () => {
    if (!isDesktop) {
      const input = document.createElement('input')
      input.type = 'file'
      input.multiple = true
      input.onchange = () => {
        if (input.files?.length) void addFilesRef.current?.(input.files)
      }
      input.click()
      return
    }

    setBusy(true)
    try {
      merge((await invoke<AttachmentInfo[]>('attachments:pick')) ?? [])
    } catch (err) {
      toast(`Could not attach: ${String(err)}`, 'error')
    } finally {
      setBusy(false)
    }
  }, [merge])

  const addFiles = useCallback(
    async (files: FileList | File[]) => {
      const list = [...files]
      if (!list.length) return

      setBusy(true)
      try {
        const paths: string[] = []
        const uploads: File[] = []

        for (const file of list) {
          const p = isDesktop ? (window.api?.pathForFile?.(file) ?? '') : ''
          if (p) paths.push(p)
          else uploads.push(file)
        }

        const described: AttachmentInfo[] = []
        if (paths.length) {
          described.push(...((await invoke<AttachmentInfo[]>('attachments:describe', paths)) ?? []))
        }

        for (const file of uploads) {
          if (file.size > MAX_UPLOAD_BYTES) {
            toast(`${file.name} is too large to attach over the network (${fmtBytes(file.size)}).`, 'error')
            continue
          }
          const base64 = await fileToBase64(file)
          const info = await invoke<AttachmentInfo>('attachments:receive', file.name, base64)
          if (info) described.push(info)
        }

        merge(described)
      } catch (err) {
        toast(`Could not attach: ${String(err)}`, 'error')
      } finally {
        setBusy(false)
      }
    },
    [merge]
  )

  addFilesRef.current = addFiles

  return {
    items,
    busy,
    pick,
    addFiles,
    remove: (p) => setItems((prev) => prev.filter((a) => a.path !== p)),
    clear: () => setItems([])
  }
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('read failed'))
    // The result is a data URL; the payload starts after the comma.
    reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '')
    reader.readAsDataURL(file)
  })
}

/** The staged files, above the composer. */
export function AttachmentBar({
  items,
  onRemove,
  disabled
}: {
  items: AttachmentInfo[]
  onRemove: (path: string) => void
  disabled?: boolean
}): JSX.Element | null {
  if (!items.length) return null

  return (
    <div className="attachments" data-testid="attachment-bar">
      {items.map((a) => (
        <div
          key={a.path}
          className={`attachment${a.warning ? ' warned' : ''}`}
          title={a.warning ? `${a.path}\n\n${a.warning}` : a.path}
          data-testid="attachment"
        >
          <Icon name={a.warning ? 'alert' : KIND_ICON[a.kind]} size={13} />
          <span className="attachment-name truncate">{a.name}</span>
          <span className="attachment-meta">
            {KIND_LABEL[a.kind]}
            {a.bytes >= 0 && ` · ${fmtBytes(a.bytes)}`}
          </span>
          <button
            className="attachment-remove"
            onClick={() => onRemove(a.path)}
            disabled={disabled}
            aria-label={`Remove ${a.name}`}
            title="Remove"
            data-testid="attachment-remove"
          >
            <Icon name="close" size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}

/**
 * Wraps a region so files can be dropped anywhere on it.
 *
 * Drag events fire on every child, so a naive enter/leave pair flickers as the pointer crosses
 * elements. Counting enters and leaves is what makes the overlay stable.
 */
export function DropZone({
  onFiles,
  disabled,
  children
}: {
  onFiles: (files: FileList | File[]) => void
  disabled?: boolean
  children: React.ReactNode
}): JSX.Element {
  const [over, setOver] = useState(false)
  const depth = useRef(0)

  const reset = (): void => {
    depth.current = 0
    setOver(false)
  }

  return (
    <div
      className="dropzone"
      onDragEnter={(e) => {
        if (disabled || !e.dataTransfer?.types?.includes('Files')) return
        e.preventDefault()
        depth.current += 1
        setOver(true)
      }}
      onDragOver={(e) => {
        if (disabled || !e.dataTransfer?.types?.includes('Files')) return
        // Without this the browser navigates to the dropped file instead of firing onDrop.
        e.preventDefault()
        e.dataTransfer.dropEffect = 'copy'
      }}
      onDragLeave={(e) => {
        // Guarded exactly as the enter is. Without the same test, dragging something that is not
        // a file across the pane decremented a counter nothing had incremented, and the overlay
        // then flickered off partway through a real drag.
        if (disabled || !e.dataTransfer?.types?.includes('Files')) return
        e.preventDefault()
        depth.current -= 1
        if (depth.current <= 0) reset()
      }}
      onDrop={(e) => {
        if (disabled) return
        e.preventDefault()
        reset()
        const files = e.dataTransfer?.files
        if (files?.length) onFiles(files)
      }}
    >
      {children}
      {over && (
        <div className="drop-overlay" data-testid="drop-overlay">
          <div className="drop-card">
            <Icon name="download" size={22} strokeWidth={1.5} />
            <div className="drop-title">Drop to attach</div>
            <div className="drop-hint">Images, video, audio, and text or code files</div>
          </div>
        </div>
      )}
    </div>
  )
}
