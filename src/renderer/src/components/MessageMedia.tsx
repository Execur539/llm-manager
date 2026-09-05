/**
 * The files that were sent with a message, shown rather than named.
 *
 * A turn used to read `[Attached: 2025-11-19 19-09-41.mp4]` and stop there, which is a poor
 * showing for an app that just spent real work looking at the thing. Images, audio and video all
 * have players; a document stays a chip, because inlining its text is already what happened.
 *
 * Video gets the interesting one. What the model was actually shown is not the file the user
 * picked — it is sampled, cropped, decimated and re-encoded, and the only account of that was a
 * sentence claiming a frame rate. The toggle plays the real thing.
 */

import { useState } from 'react'
import type { MessageAttachment } from '@shared/types'
import { mediaUrl, fmtBytes } from '../lib/api'
import Icon from './Icon'

export default function MessageMedia({ items }: { items: MessageAttachment[] }): JSX.Element | null {
  if (!items.length) return null
  return (
    <div className="msg-media" data-testid="message-media">
      {items.map((a) => (
        <Attachment key={a.id} item={a} />
      ))}
    </div>
  )
}

function Attachment({ item }: { item: MessageAttachment }): JSX.Element {
  const [failed, setFailed] = useState(false)

  // A file attached from removable media, or a scratch clip that was cleaned up. Falling back to
  // the chip is honest; a broken player pretending to be a video is not.
  if (failed || item.kind === 'doc') return <FileChip item={item} missing={failed} />

  if (item.kind === 'image') {
    return (
      <figure className="media-frame" data-testid="media-image">
        <img src={mediaUrl(item.id)} alt={item.name} loading="lazy" onError={() => setFailed(true)} />
        <figcaption className="media-caption truncate" title={item.name}>
          {item.name}
        </figcaption>
      </figure>
    )
  }

  if (item.kind === 'audio') {
    return (
      <div className="media-frame media-audio" data-testid="media-audio">
        <div className="media-caption truncate" title={item.name}>
          {item.name}
        </div>
        <audio controls preload="metadata" src={mediaUrl(item.id)} onError={() => setFailed(true)} />
      </div>
    )
  }

  return <VideoAttachment item={item} onBroken={() => setFailed(true)} />
}

function VideoAttachment({ item, onBroken }: { item: MessageAttachment; onBroken: () => void }): JSX.Element {
  const [variant, setVariant] = useState<'source' | 'optimised'>('source')
  const showing = item.optimised ? variant : 'source'

  return (
    <figure className="media-frame media-video" data-testid="media-video">
      <div className="media-stage">
        {/*
         * Keyed on the variant so switching swaps the element rather than mutating `src`.
         *
         * The two files are different lengths — the sampled clip is a few seconds of frames drawn
         * from minutes of source — so carrying a playhead across would land somewhere arbitrary,
         * and some browsers keep showing the old frame until the new one decodes.
         */}
        <video
          key={showing}
          controls
          preload="metadata"
          playsInline
          src={mediaUrl(item.id, showing)}
          onError={() => (showing === 'source' ? onBroken() : setVariant('source'))}
        />
        {item.optimised && (
          <div className="media-toggle" role="group" aria-label="Which version to play">
            <button
              className={showing === 'source' ? 'on' : ''}
              onClick={() => setVariant('source')}
              data-testid="media-view-original"
            >
              Original
            </button>
            <button
              className={showing === 'optimised' ? 'on' : ''}
              onClick={() => setVariant('optimised')}
              title="The sampled, cropped clip the model was actually shown"
              data-testid="media-view-optimised"
            >
              <Icon name="sparkle" size={12} />
              What the model saw
            </button>
          </div>
        )}
      </div>

      <figcaption className="media-caption truncate" title={item.name}>
        {item.name}
        {item.bytes !== undefined && <span className="media-size">{fmtBytes(item.bytes)}</span>}
      </figcaption>

      {/*
       * The sampler's own sentence, shown only against the clip it describes.
       *
       * Under the original it would be describing something the viewer is not looking at.
       */}
      {showing === 'optimised' && item.note && <div className="media-note">{item.note}</div>}
    </figure>
  )
}

function FileChip({ item, missing }: { item: MessageAttachment; missing?: boolean }): JSX.Element {
  return (
    <div className={`media-chip${missing ? ' missing' : ''}`} data-testid="media-chip">
      <Icon name={missing ? 'alert' : 'documents'} size={14} />
      <span className="truncate" title={item.name}>
        {item.name}
      </span>
      {missing ? (
        <span className="media-size">unavailable</span>
      ) : (
        item.bytes !== undefined && <span className="media-size">{fmtBytes(item.bytes)}</span>
      )}
    </div>
  )
}

/**
 * The message text without the trailing list of filenames.
 *
 * That line exists so the transcript, the export and the model all know a file was sent, and it
 * has to stay in what is stored. On screen it is redundant the moment the files are rendered
 * underneath — and worse, it is the only part of the message the user did not write.
 */
export function stripAttachmentLine(content: string, hasMedia: boolean): string {
  if (!hasMedia) return content
  return content.replace(/\n*\[Attached:[^\]\n]*\]\s*$/, '').trimEnd()
}
