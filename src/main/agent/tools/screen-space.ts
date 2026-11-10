/**
 * One coordinate system for everything that points at the screen.
 *
 * Positions are real screen pixels, with the primary display's top-left corner at 0,0. That needed
 * saying, because three different answers were in use: screenshots were taken at the display's
 * scaled size, PowerShell placed the cursor in scaled units, and Windows' accessibility interface
 * reports real pixels. On a display scaled to 200% those disagree by a factor of two, so a click
 * aimed from a screenshot landed a quarter of the way to its target.
 *
 * A screenshot is also shrunk before the model sees it, so a position read off the picture is in
 * the picture's pixels, not the screen's. The last capture's placement is remembered here so the
 * mouse tools can convert one to the other instead of asking the model to do the arithmetic.
 */

/**
 * Makes the PowerShell process see real pixels, on every display.
 *
 * Without it Windows hands a script scaled coordinates for the cursor and for window rectangles,
 * which do not match what the accessibility interface reports for the same things. Per-monitor
 * awareness where the system has it, so a second display at a different scale is right too.
 */
export const DPI_AWARE = `
Add-Type -Namespace Llmm -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr v); [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
try { if (-not [Llmm.Dpi]::SetProcessDpiAwarenessContext([IntPtr](-4))) { [void][Llmm.Dpi]::SetProcessDPIAware() } } catch { try { [void][Llmm.Dpi]::SetProcessDPIAware() } catch { } }
`

interface Capture {
  /** Screen position of the picture's top-left corner. */
  left: number
  top: number
  /** Screen pixels per picture pixel. */
  scale: number
  width: number
  height: number
}

let lastCapture: Capture | null = null

/** Remember where the picture just shown to the model sits on the screen. */
export function recordCapture(capture: Capture): void {
  lastCapture = capture
}

/**
 * A position as the tools were given it, in screen pixels.
 *
 * `coords: "image"` means the numbers were read off the last screenshot; anything else means they
 * are already screen pixels, which is what the window and control tools report.
 */
export function toScreen(x: number, y: number, coords: unknown): { x: number; y: number } {
  if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error('x and y must be numbers.')
  if (String(coords ?? 'screen').toLowerCase() !== 'image') return { x: Math.round(x), y: Math.round(y) }
  if (!lastCapture) {
    throw new Error('coords: "image" refers to the last screenshot, and none has been taken yet. Take a screenshot first.')
  }
  if (x < 0 || y < 0 || x > lastCapture.width || y > lastCapture.height) {
    throw new Error(
      `(${x}, ${y}) is outside the last screenshot, which was shown at ${lastCapture.width}x${lastCapture.height}.`
    )
  }
  return {
    x: Math.round(lastCapture.left + x * lastCapture.scale),
    y: Math.round(lastCapture.top + y * lastCapture.scale)
  }
}
