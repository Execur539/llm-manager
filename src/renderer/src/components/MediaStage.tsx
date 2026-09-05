/**
 * What the app is doing to an attachment before the turn can begin.
 *
 * Preparing a video is several complete passes over the source file, and on a long or awkwardly
 * encoded one — a 120 fps AV1 capture was the case that prompted this — that is minutes with
 * nothing on screen but a spinner. The passes are named rather than counted because their
 * durations are wildly uneven: finding what moves takes under a second, sampling the frames takes
 * most of the wait, and a progress bar split evenly between them would be a lie.
 */

import { Spinner } from './Spinner'

export default function MediaStage({ file, stage }: { file: string; stage: string }): JSX.Element {
  return (
    <div className="media-stage-note" data-testid="media-stage">
      <Spinner size={13} />
      <span className="media-stage-what">{stage}</span>
      {file && <span className="media-stage-file truncate">{file}</span>}
    </div>
  )
}
