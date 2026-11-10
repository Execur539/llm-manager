/**
 * Working an application through its controls rather than its pixels.
 *
 * Windows keeps, for every window, a tree of what is in it — buttons, menus, text boxes, list
 * rows — with each one's name, type, state and position. Screen readers are built on it. Reading
 * that tree tells the model what can be pressed and what it is called, and pressing a control by
 * name does not depend on finding it in a screenshot and guessing where its middle is.
 *
 * The mouse tools are here too, for the applications that draw their own interface and publish no
 * tree: games, canvases, some older programs.
 *
 * All of it runs as PowerShell against .NET's UI Automation client, so nothing is compiled at
 * install time and nothing is bundled.
 */

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import type { Tool, ToolContext } from './base'
import { schema, str, int, bool } from './base'
import { agentScratchDir, runPowerShell } from './exec'
import { DPI_AWARE, toScreen } from './screen-space'

/**
 * The one script behind the three control tools.
 *
 * Its arguments arrive as a JSON file named in the environment, never as text spliced into the
 * script, so a window title or a string to type cannot be mistaken for PowerShell.
 *
 * An element is identified by its window's handle and its position in the tree: "132456:0.3.1" is
 * the second child of the fourth child of the first child of window 132456. The path is only
 * meaningful until the window rearranges itself, which is why every action reports what it acted
 * on by name.
 */
const UI_SCRIPT = String.raw`
$ErrorActionPreference = 'Stop'
${DPI_AWARE}
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$a = Get-Content -Raw -Encoding UTF8 -LiteralPath $env:LLMM_UI_ARGS | ConvertFrom-Json
$AE = [System.Windows.Automation.AutomationElement]
$Auto = [System.Windows.Automation.Automation]
$Scope = [System.Windows.Automation.TreeScope]
$View = $Auto::ControlViewCondition
$Walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$Root = $AE::RootElement
$NotSupported = $AE::NotSupported

function Fail($m) { Write-Output "ERROR: $m"; exit 0 }
function Short($s, $n) { $t = ([string]$s) -replace '\s+', ' '; if ($t.Length -gt $n) { $t.Substring(0, $n) + '...' } else { $t } }

function Get-TopWindow($el) {
  $cur = $el
  while ($cur) {
    $p = $Walker.GetParent($cur)
    if (-not $p -or $Auto::Compare($p, $Root)) { return $cur }
    $cur = $p
  }
  return $el
}

function Window-List {
  $out = @()
  foreach ($w in @($Root.FindAll($Scope::Children, $View))) {
    $n = $w.Current.Name
    if ($n) { $out += ('"' + (Short $n 60) + '"') }
  }
  return ($out -join ', ')
}

function Find-Window($spec) {
  $s = ([string]$spec).Trim()
  if (-not $s -or $s -match '^(focused|active|foreground|current)$') { return Get-TopWindow $AE::FocusedElement }
  if ($s -match '^\d{4,}$') { try { return $AE::FromHandle([IntPtr][int64]$s) } catch { } }
  $wins = @($Root.FindAll($Scope::Children, $View))
  if ($s -match '^pid\s*:?\s*(\d+)$') {
    $want = [int]$Matches[1]
    foreach ($w in $wins) { if ($w.Current.ProcessId -eq $want) { return $w } }
    return $null
  }
  $low = $s.ToLower()
  foreach ($w in $wins) { if ($w.Current.Name -eq $s) { return $w } }
  foreach ($w in $wins) { $n = $w.Current.Name; if ($n -and $n.ToLower().Contains($low)) { return $w } }
  foreach ($w in $wins) {
    $pn = ''
    try { $pn = (Get-Process -Id $w.Current.ProcessId -ErrorAction Stop).ProcessName } catch { }
    if ($pn -and $pn.ToLower().Contains($low)) { return $w }
  }
  return $null
}

function Resolve-Id($id) {
  $parts = ([string]$id) -split ':', 2
  if ($parts[0] -notmatch '^\d+$') { Fail "'$id' is not an element id. Ids come from ui_inspect and look like 132456:0.3.1." }
  try { $el = $AE::FromHandle([IntPtr][int64]$parts[0]) } catch { Fail "The window that element id belongs to has closed. Run ui_inspect again." }
  if ($parts.Count -gt 1 -and $parts[1]) {
    foreach ($i in ($parts[1] -split '\.')) {
      $kids = $el.FindAll($Scope::Children, $View)
      if ([int]$i -ge $kids.Count) { Fail "Element $id is no longer there: the window has changed. Run ui_inspect again." }
      $el = $kids[[int]$i]
    }
  }
  return $el
}

function Type-Name($ct) { if ($ct) { $ct.ProgrammaticName -replace '^ControlType\.', '' } else { 'Unknown' } }

# Older programs (Windows Forms, plain Win32) often publish every control as a bare "Pane". Each is
# still a window of its own, and the window's class says what it is.
function Class-Kind($class) {
  $c = ([string]$class).ToUpper()
  if ($c -match 'BUTTON') { return 'Button' }
  if ($c -match 'RICHEDIT|EDIT') { return 'Edit' }
  if ($c -match 'STATIC') { return 'Text' }
  if ($c -match 'COMBOBOX') { return 'ComboBox' }
  if ($c -match 'LISTBOX') { return 'List' }
  return $null
}
function Kind-Of($type, $class) { if ($type -eq 'Pane') { $k = Class-Kind $class; if ($k) { return $k } }; return $type }

function Native-Ready {
  if (-not ('LlmmNative' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class LlmmNative {
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr SendMessageTimeout(IntPtr h, uint msg, IntPtr w, string l, uint flags, uint timeout, out IntPtr result);
}
'@
  }
}

function Describe-Current($el) {
  $c = $el.Current
  $t = Kind-Of (Type-Name $c.ControlType) $c.ClassName
  if ($c.Name) { "$t " + '"' + (Short $c.Name 80) + '"' } else { $t }
}

# The first control under $root whose name is $name: exactly if there is one, containing it if not.
function Find-Named($root, $name, $type) {
  $fits = { param($e) (-not $type) -or ((Kind-Of (Type-Name $e.Current.ControlType) $e.Current.ClassName) -ieq $type) }
  $cond = New-Object System.Windows.Automation.PropertyCondition($AE::NameProperty, [string]$name, [System.Windows.Automation.PropertyConditionFlags]::IgnoreCase)
  $exact = @($root.FindAll($Scope::Descendants, $cond)) | Where-Object { & $fits $_ }
  # Several things can carry one name: a row and the label inside it. Prefer the one that can be acted on.
  $usable = { param($e) $q = $null; $e.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$q) -or $e.TryGetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern, [ref]$q) -or $e.TryGetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern, [ref]$q) -or $e.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$q) -or $e.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$q) }
  $pick = $exact | Where-Object { $_.Current.IsEnabled -and -not $_.Current.IsOffscreen -and (& $usable $_) } | Select-Object -First 1
  if (-not $pick) { $pick = $exact | Where-Object { $_.Current.IsEnabled -and -not $_.Current.IsOffscreen } | Select-Object -First 1 }
  if (-not $pick) { $pick = $exact | Select-Object -First 1 }
  if ($pick) { return $pick }
  $low = ([string]$name).ToLower()
  $queue = New-Object System.Collections.Generic.Queue[object]
  $queue.Enqueue($root)
  $seen = 0
  while ($queue.Count -gt 0 -and $seen -lt 4000) {
    $el = $queue.Dequeue(); $seen++
    foreach ($k in @($el.FindAll($Scope::Children, $View))) {
      $n = $k.Current.Name
      if ($n -and $n.ToLower().Contains($low) -and (& $fits $k)) { return $k }
      $queue.Enqueue($k)
    }
  }
  return $null
}

function Resolve-Target($needControl) {
  if ($a.element) { return Resolve-Id $a.element }
  $w = Find-Window $a.window
  if (-not $w) { Fail ("No window matches '" + $a.window + "'. Open windows: " + (Window-List)) }
  if ($a.name) {
    $el = Find-Named $w $a.name $a.type
    if (-not $el) { Fail ("Nothing named '" + $a.name + "' was found in " + (Describe-Current $w) + ". Run ui_inspect on it to see what it contains.") }
    return $el
  }
  if ($needControl) { Fail 'Say which control: give element (an id from ui_inspect) or name.' }
  return $w
}

function Mouse-Ready {
  if (-not ('LlmmPointer' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Threading;
public static class LlmmPointer {
  [DllImport("user32.dll")] static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] static extern void mouse_event(uint f, uint dx, uint dy, int data, IntPtr extra);
  public static void Click(int x, int y, bool right, int times) {
    SetCursorPos(x, y); Thread.Sleep(60);
    uint d = right ? 0x08u : 0x02u, u = right ? 0x10u : 0x04u;
    for (int i = 0; i < times; i++) { mouse_event(d, 0, 0, 0, IntPtr.Zero); mouse_event(u, 0, 0, 0, IntPtr.Zero); if (i + 1 < times) Thread.Sleep(70); }
  }
}
'@
  }
}

function Press-At($el, $right, $times) {
  # Set by the checks, so that a test can never move the real mouse.
  if ($env:LLMM_UI_NO_MOUSE) { Fail ('MOUSE NEEDED for ' + (Describe-Current $el) + '; patterns: ' + (($el.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName }) -join ',')) }
  try { $el.SetFocus() } catch { }
  Start-Sleep -Milliseconds 120
  $pt = New-Object System.Windows.Point
  if (-not $el.TryGetClickablePoint([ref]$pt)) {
    $r = $el.Current.BoundingRectangle
    if ($r.IsEmpty) { Fail ((Describe-Current $el) + ' has no position on screen, so it cannot be clicked.') }
    $pt = New-Object System.Windows.Point(($r.X + $r.Width / 2), ($r.Y + $r.Height / 2))
  }
  Mouse-Ready
  [LlmmPointer]::Click([int]$pt.X, [int]$pt.Y, $right, $times)
  return ('at ' + [int]$pt.X + ',' + [int]$pt.Y)
}

$mode = [string]$a.mode

if ($mode -eq 'click') {
  $el = Resolve-Target $true
  $what = Describe-Current $el
  if (-not $el.Current.IsEnabled) { Fail "$what is disabled, so it cannot be used right now." }
  $how = if ($a.action) { ([string]$a.action).ToLower() } else { 'click' }
  $p = $null
  switch ($how) {
    'click' {
      if ($el.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$p)) { $p.Invoke(); Write-Output "Pressed $what."; break }
      if ($el.TryGetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern, [ref]$p)) { $p.Select(); Write-Output "Selected $what."; break }
      if ($el.TryGetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern, [ref]$p)) { $p.Toggle(); Write-Output ("Toggled $what; it is now " + $p.Current.ToggleState + '.'); break }
      if ($el.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$p)) {
        if ($p.Current.ExpandCollapseState -eq 'Expanded') { $p.Collapse(); Write-Output "Collapsed $what." } else { $p.Expand(); Write-Output "Expanded $what." }
        break
      }
      # A control that is its own window and says it is a button can be pressed by message, with no mouse.
      $hwnd = $el.Current.NativeWindowHandle
      if ($hwnd -ne 0 -and (Class-Kind $el.Current.ClassName) -eq 'Button') {
        Native-Ready
        [void][LlmmNative]::PostMessage([IntPtr]$hwnd, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)
        Write-Output "Pressed $what."
        break
      }
      $where = Press-At $el $false 1
      Write-Output "Clicked $what with the mouse $where."
    }
    'double_click' { $where = Press-At $el $false 2; Write-Output "Double-clicked $what $where." }
    'right_click' { $where = Press-At $el $true 1; Write-Output "Right-clicked $what $where." }
    'focus' { $el.SetFocus(); Write-Output "Focused $what." }
    'expand' { if ($el.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$p)) { $p.Expand(); Write-Output "Expanded $what." } else { Fail "$what cannot be expanded." } }
    'collapse' { if ($el.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$p)) { $p.Collapse(); Write-Output "Collapsed $what." } else { Fail "$what cannot be collapsed." } }
    default { Fail "Unknown action '$how'. Use click, double_click, right_click, focus, expand or collapse." }
  }
  exit 0
}

if ($mode -eq 'set_text') {
  $el = Resolve-Target $true
  $what = Describe-Current $el
  if (-not $el.Current.IsEnabled) { Fail "$what is disabled, so nothing can be typed into it." }
  $text = [string]$a.text
  $p = $null
  if ($el.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$p)) {
    if ($p.Current.IsReadOnly) { Fail "$what is read-only." }
    $p.SetValue($text)
    Write-Output ("Set $what to " + $text.Length + ' characters.')
    exit 0
  }
  $hwnd = $el.Current.NativeWindowHandle
  if ($hwnd -ne 0 -and (Class-Kind $el.Current.ClassName) -eq 'Edit') {
    Native-Ready
    $done = [IntPtr]::Zero
    [void][LlmmNative]::SendMessageTimeout([IntPtr]$hwnd, 0x000C, [IntPtr]::Zero, $text, 2, 5000, [ref]$done)
    Write-Output ("Set $what to " + $text.Length + ' characters.')
    exit 0
  }
  # No way to set it directly: focus it and type, replacing what is there.
  $el.SetFocus()
  Start-Sleep -Milliseconds 150
  Add-Type -AssemblyName System.Windows.Forms
  $keys = ([regex]::Replace($text, '[+^%~(){}\[\]]', '{$0}')) -replace "\r?\n", '{ENTER}'
  [System.Windows.Forms.SendKeys]::SendWait('^a')
  [System.Windows.Forms.SendKeys]::SendWait($keys)
  Write-Output ("Typed " + $text.Length + " characters into $what.")
  exit 0
}

# ---- inspect

$target = Resolve-Target $false
$top = Get-TopWindow $target
$handle = $top.Current.NativeWindowHandle
$prefix = ''
if ($a.element) { $prefix = (([string]$a.element) -split ':', 2)[1] }
elseif ($a.name) {
  # Work out where the named control sits, so the ids printed below it are usable.
  $chain = @()
  $cur = $target
  while ($cur -and -not $Auto::Compare($cur, $top)) {
    $parent = $Walker.GetParent($cur)
    if (-not $parent) { break }
    $sibs = $parent.FindAll($Scope::Children, $View)
    $at = -1
    for ($i = 0; $i -lt $sibs.Count; $i++) { if ($Auto::Compare($sibs[$i], $cur)) { $at = $i; break } }
    if ($at -lt 0) { break }
    $chain = ,$at + $chain
    $cur = $parent
  }
  $prefix = ($chain -join '.')
}

if ($a.text) {
  $p = $null
  $body = ''
  if ($target.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$p)) { $body = $p.DocumentRange.GetText(20000) }
  elseif ($target.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$p)) { $body = [string]$p.Current.Value }
  if (-not $body) {
    $bits = New-Object System.Collections.Generic.List[string]
    $cond = New-Object System.Windows.Automation.OrCondition(
      (New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, [System.Windows.Automation.ControlType]::Text)),
      (New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, [System.Windows.Automation.ControlType]::Edit)),
      (New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, [System.Windows.Automation.ControlType]::Document)))
    foreach ($e in @($target.FindAll($Scope::Descendants, $cond))) {
      $n = $e.Current.Name
      if ($n -and $bits.Count -lt 400) { $bits.Add($n) }
    }
    $body = ($bits -join [Environment]::NewLine)
  }
  Write-Output ('Text of ' + (Describe-Current $target) + ':')
  if ($body) { Write-Output $body } else { Write-Output '(no text)' }
  exit 0
}

$maxLines = if ($a.max) { [int]$a.max } else { 120 }
$maxDepth = if ($a.depth) { [int]$a.depth } else { 30 }
$filter = if ($a.filter) { ([string]$a.filter).ToLower() } else { $null }
$Interactive = @('Button','CheckBox','ComboBox','Edit','Hyperlink','ListItem','MenuItem','RadioButton','Slider','Spinner','SplitButton','TabItem','TreeItem','Document','DataItem')

$cr = New-Object System.Windows.Automation.CacheRequest
$ValueProp = [System.Windows.Automation.ValuePattern]::ValueProperty
$ToggleProp = [System.Windows.Automation.TogglePattern]::ToggleStateProperty
$SelectedProp = [System.Windows.Automation.SelectionItemPattern]::IsSelectedProperty
$ExpandProp = [System.Windows.Automation.ExpandCollapsePattern]::ExpandCollapseStateProperty
foreach ($prop in @($AE::NameProperty, $AE::ControlTypeProperty, $AE::AutomationIdProperty, $AE::BoundingRectangleProperty,
    $AE::IsEnabledProperty, $AE::IsOffscreenProperty, $AE::IsPasswordProperty, $AE::ClassNameProperty, $ValueProp, $ToggleProp, $SelectedProp, $ExpandProp)) { $cr.Add($prop) }

$script:lines = New-Object System.Collections.Generic.List[string]
$script:visited = 0
$script:more = 0

function Walk($el, $path, $depth) {
  if ($script:visited -gt 5000) { return }
  $kids = $el.FindAll($Scope::Children, $View)
  for ($i = 0; $i -lt $kids.Count; $i++) {
    $k = $kids[$i]
    $script:visited++
    $kp = if ($path) { "$path.$i" } else { "$i" }
    if ($k.GetCachedPropertyValue($AE::IsOffscreenProperty) -eq $true -and -not $a.offscreen) { continue }
    $name = [string]$k.GetCachedPropertyValue($AE::NameProperty)
    $type = Kind-Of (Type-Name ($k.GetCachedPropertyValue($AE::ControlTypeProperty))) ([string]$k.GetCachedPropertyValue($AE::ClassNameProperty))
    $autoId = [string]$k.GetCachedPropertyValue($AE::AutomationIdProperty)
    $show = if ($filter) { ("$name $type $autoId").ToLower().Contains($filter) } else { $name -or ($Interactive -contains $type) }
    if ($show) {
      if ($script:lines.Count -ge $maxLines) { $script:more++ }
      else {
        $line = $handle.ToString() + ':' + $kp + ' ' + $type
        if ($name) { $line += ' "' + (Short $name 80) + '"' }
        $v = $k.GetCachedPropertyValue($ValueProp, $true)
        if (-not [object]::ReferenceEquals($v, $NotSupported) -and $v -and $v -ne $name) {
          if ($k.GetCachedPropertyValue($AE::IsPasswordProperty) -eq $true) { $line += ' value=(hidden)' } else { $line += ' value="' + (Short $v 60) + '"' }
        }
        $tg = $k.GetCachedPropertyValue($ToggleProp, $true)
        if (-not [object]::ReferenceEquals($tg, $NotSupported)) { if ("$tg" -eq 'On') { $line += ' [checked]' } elseif ("$tg" -eq 'Off') { $line += ' [unchecked]' } }
        $sel = $k.GetCachedPropertyValue($SelectedProp, $true)
        if (-not [object]::ReferenceEquals($sel, $NotSupported) -and $sel -eq $true) { $line += ' [selected]' }
        $ex = $k.GetCachedPropertyValue($ExpandProp, $true)
        if (-not [object]::ReferenceEquals($ex, $NotSupported)) { if ("$ex" -eq 'Expanded') { $line += ' [expanded]' } elseif ("$ex" -eq 'Collapsed') { $line += ' [collapsed]' } }
        if ($k.GetCachedPropertyValue($AE::IsEnabledProperty) -eq $false) { $line += ' [disabled]' }
        $r = $k.GetCachedPropertyValue($AE::BoundingRectangleProperty)
        if ($r -and -not $r.IsEmpty) { $line += ' at ' + [int]($r.X + $r.Width / 2) + ',' + [int]($r.Y + $r.Height / 2) }
        $script:lines.Add($line)
      }
    }
    if ($depth -lt $maxDepth) { Walk $k $kp ($depth + 1) }
  }
}

# Not $scope: PowerShell names ignore case, and that would overwrite $Scope above.
$activeCache = $cr.Activate()
try { Walk $target $prefix 0 } finally { $activeCache.Dispose() }

$procName = ''
try { $procName = (Get-Process -Id $top.Current.ProcessId -ErrorAction Stop).ProcessName } catch { }
$r = $top.Current.BoundingRectangle
$head = 'Window ' + $handle + ' "' + (Short $top.Current.Name 80) + '" (' + $procName + ')'
if (-not $r.IsEmpty) { $head += ' at ' + [int]$r.X + ',' + [int]$r.Y + ', ' + [int]$r.Width + 'x' + [int]$r.Height }
Write-Output $head
if ($a.element -or $a.name) { Write-Output ('Inside ' + (Describe-Current $target) + ':') }
if ($script:lines.Count -eq 0) {
  if ($filter) { Write-Output ("Nothing in it matches '" + $a.filter + "'.") }
  else { Write-Output '(this window publishes no controls; it may draw its own interface, in which case use screenshot and the mouse tools)' }
} else {
  foreach ($l in $script:lines) { Write-Output $l }
  if ($script:more -gt 0) { Write-Output ("... and " + $script:more + " more. Narrow it with filter, or give element to list what is inside one control.") }
}
`

/** Moving, scrolling and dragging with the real mouse. */
const POINTER_SCRIPT = String.raw`
$ErrorActionPreference = 'Stop'
${DPI_AWARE}
$a = Get-Content -Raw -Encoding UTF8 -LiteralPath $env:LLMM_UI_ARGS | ConvertFrom-Json
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Threading;
public static class LlmmMouse {
  [StructLayout(LayoutKind.Sequential)] public struct P { public int X; public int Y; }
  [DllImport("user32.dll")] static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] static extern bool GetCursorPos(out P p);
  [DllImport("user32.dll")] static extern void mouse_event(uint f, uint dx, uint dy, int data, IntPtr extra);
  public static string Where() { P p; GetCursorPos(out p); return p.X + "," + p.Y; }
  public static void Move(int x, int y) { SetCursorPos(x, y); Thread.Sleep(60); }
  public static void Click(int x, int y, bool right, int times) {
    Move(x, y);
    uint d = right ? 0x08u : 0x02u, u = right ? 0x10u : 0x04u;
    for (int i = 0; i < times; i++) { mouse_event(d, 0, 0, 0, IntPtr.Zero); mouse_event(u, 0, 0, 0, IntPtr.Zero); if (i + 1 < times) Thread.Sleep(70); }
  }
  public static void Scroll(int x, int y, int notches, bool sideways) {
    Move(x, y);
    int step = notches > 0 ? 1 : -1;
    for (int i = 0; i < Math.Abs(notches); i++) { mouse_event(sideways ? 0x01000u : 0x0800u, 0, 0, step * 120, IntPtr.Zero); Thread.Sleep(20); }
  }
  public static void Drag(int x1, int y1, int x2, int y2, bool right) {
    Move(x1, y1);
    uint d = right ? 0x08u : 0x02u, u = right ? 0x10u : 0x04u;
    mouse_event(d, 0, 0, 0, IntPtr.Zero); Thread.Sleep(90);
    // In steps: a program watching for a drag ignores a pointer that simply appears somewhere else.
    for (int i = 1; i <= 24; i++) { SetCursorPos(x1 + (x2 - x1) * i / 24, y1 + (y2 - y1) * i / 24); Thread.Sleep(12); }
    Thread.Sleep(90); mouse_event(u, 0, 0, 0, IntPtr.Zero);
  }
}
'@
switch ([string]$a.op) {
  'where' { Write-Output ([LlmmMouse]::Where()) }
  'move' { [LlmmMouse]::Move([int]$a.x, [int]$a.y); Write-Output 'ok' }
  'click' { [LlmmMouse]::Click([int]$a.x, [int]$a.y, [bool]$a.right, [int]$a.times); Write-Output 'ok' }
  'scroll' { [LlmmMouse]::Scroll([int]$a.x, [int]$a.y, [int]$a.notches, [bool]$a.sideways); Write-Output 'ok' }
  'drag' { [LlmmMouse]::Drag([int]$a.x, [int]$a.y, [int]$a.x2, [int]$a.y2, [bool]$a.right); Write-Output 'ok' }
  default { Write-Output 'ERROR: unknown operation' }
}
`

/** Run one of the scripts above with its arguments in a file, and return what it printed. */
async function runScript(
  script: string,
  args: Record<string, unknown>,
  ctx: ToolContext,
  timeoutMs: number
): Promise<{ out: string; timedOut: boolean }> {
  const file = path.join(agentScratchDir(), `ui-args-${crypto.randomBytes(6).toString('hex')}.json`)
  fs.writeFileSync(file, JSON.stringify(args), 'utf8')
  try {
    const r = await runPowerShell(script, { timeoutMs, signal: ctx.signal, env: { LLMM_UI_ARGS: file } })
    const out = r.stdout.trim()
    if (r.timedOut) return { out, timedOut: true }
    if (out.startsWith('ERROR: ')) throw new Error(out.slice('ERROR: '.length))
    if (!out && r.stderr.trim()) {
      const reason = r.stderr.trim().split(/\r?\n/)[0]
      throw new Error(
        /access is denied|E_ACCESSDENIED/i.test(r.stderr)
          ? 'Windows refused: that window belongs to a program running as administrator, which an ordinary program may not control.'
          : reason
      )
    }
    return { out, timedOut: false }
  } finally {
    fs.rmSync(file, { force: true })
  }
}

export const pointerOp = (args: Record<string, unknown>, ctx: ToolContext): Promise<{ out: string; timedOut: boolean }> =>
  runScript(POINTER_SCRIPT, args, ctx, 20_000)

const WINDOW_HELP =
  'Which window: part of its title, a program name, "pid 1234", a window number from an earlier ui_inspect, or leave out for the window in front'

const uiInspect: Tool = {
  name: 'ui_inspect',
  description:
    'List what is in an application window: its buttons, menus, text boxes, list rows and labels, ' +
    'each with an id, its type, name, state and the screen position of its middle. This is how to ' +
    'find out what can be pressed and what it is called, and it is more reliable than reading a ' +
    'screenshot. Use the ids with ui_click and ui_set_text. Set text to read what a control or ' +
    'window says instead of listing it.',
  tier: 'read',
  parameters: schema({
    window: str(WINDOW_HELP),
    element: str('List only what is inside this control (an id from an earlier ui_inspect)'),
    name: str('List only what is inside the control with this name'),
    filter: str('Show only controls whose name or type contains this'),
    text: bool('Return the text of the window or control instead of a list'),
    max: int('Most lines to return (default 120)'),
    offscreen: bool('Include controls that are scrolled out of view')
  }),
  async run(args, ctx) {
    const { out, timedOut } = await runScript(UI_SCRIPT, { ...args, mode: 'inspect' }, ctx, 45_000)
    if (timedOut) {
      return (
        (out ? `${out}\n` : '') +
        'The window took too long to describe itself. Narrow it with filter, or give element to look inside one control.'
      )
    }
    return out || 'Nothing was returned.'
  }
}

const uiClick: Tool = {
  name: 'ui_click',
  description:
    'Press a control in an application window: a button, menu item, tab, checkbox, list row or ' +
    'link. Say which by element (an id from ui_inspect) or by window and name. It presses the ' +
    'control directly where the application allows it, and with the mouse where it does not. ' +
    'Controls the real desktop.',
  tier: 'execute',
  parameters: schema({
    element: str('Id of the control, from ui_inspect'),
    window: str(WINDOW_HELP),
    name: str('Name of the control, as ui_inspect shows it'),
    type: str('Its type, such as Button or MenuItem, when several controls share the name'),
    action: str("'click' (default), 'double_click', 'right_click', 'focus', 'expand' or 'collapse'")
  }),
  async run(args, ctx) {
    const { out, timedOut } = await runScript(UI_SCRIPT, { ...args, mode: 'click' }, ctx, 15_000)
    if (timedOut) {
      // Pressing a button that opens a dialog does not return, in some applications, until the
      // dialog is closed. That is a press that worked, not one that failed.
      return 'The press was sent, but the application has not answered yet. It has most likely opened a dialog. Use ui_inspect to see what is on screen now.'
    }
    return out || 'Done.'
  }
}

const uiSetText: Tool = {
  name: 'ui_set_text',
  description:
    'Put text into a text box in an application window, replacing what is there. Say which by ' +
    'element (an id from ui_inspect) or by window and name. Controls the real desktop.',
  tier: 'execute',
  parameters: schema(
    {
      text: str('The text to put in'),
      element: str('Id of the text box, from ui_inspect'),
      window: str(WINDOW_HELP),
      name: str('Name of the text box, as ui_inspect shows it')
    },
    ['text']
  ),
  async run(args, ctx) {
    const { out, timedOut } = await runScript(UI_SCRIPT, { ...args, mode: 'set_text' }, ctx, 20_000)
    if (timedOut) return 'The text was sent, but the application has not answered yet. Use ui_inspect to see what it shows now.'
    return out || 'Done.'
  }
}

const COORDS_HELP = "'screen' (default) for screen pixels, as ui_inspect and window_bounds give them, or 'image' for a position read off the last screenshot"

const clickMouse: Tool = {
  name: 'click_mouse',
  description:
    'Move the mouse to a position and click. Controls the real desktop, and the user sees it ' +
    'happen. Prefer ui_click, which presses a control by name; use this for programs that draw ' +
    'their own interface. For a position read off a screenshot, pass coords: "image".',
  tier: 'execute',
  parameters: schema(
    {
      x: int('X position'),
      y: int('Y position'),
      button: str("'left' (default) or 'right'"),
      double: bool('Double-click'),
      coords: str(COORDS_HELP)
    },
    ['x', 'y']
  ),
  async run(args, ctx) {
    const at = toScreen(Number(args.x), Number(args.y), args.coords)
    const right = String(args.button ?? 'left') === 'right'
    await pointerOp({ op: 'click', x: at.x, y: at.y, right, times: args.double ? 2 : 1 }, ctx)
    return `${args.double ? 'Double-clicked' : 'Clicked'} ${right ? 'right' : 'left'} at (${at.x}, ${at.y}).`
  }
}

const moveMouse: Tool = {
  name: 'move_mouse',
  description:
    'Move the mouse to a position without clicking, to hover over something: a tooltip, a menu ' +
    'that opens on hover. Controls the real desktop.',
  tier: 'execute',
  parameters: schema({ x: int('X position'), y: int('Y position'), coords: str(COORDS_HELP) }, ['x', 'y']),
  async run(args, ctx) {
    const at = toScreen(Number(args.x), Number(args.y), args.coords)
    await pointerOp({ op: 'move', x: at.x, y: at.y }, ctx)
    return `Moved the mouse to (${at.x}, ${at.y}).`
  }
}

const scrollMouse: Tool = {
  name: 'scroll_mouse',
  description:
    'Turn the mouse wheel over a position, to scroll whatever is under it. Controls the real desktop.',
  tier: 'execute',
  parameters: schema(
    {
      x: int('X position to scroll over'),
      y: int('Y position to scroll over'),
      direction: str("'down' (default), 'up', 'left' or 'right'"),
      amount: int('How many notches of the wheel (default 3)'),
      coords: str(COORDS_HELP)
    },
    ['x', 'y']
  ),
  async run(args, ctx) {
    const at = toScreen(Number(args.x), Number(args.y), args.coords)
    const direction = String(args.direction ?? 'down').toLowerCase()
    const amount = Math.max(1, Math.min(100, Math.round(Number(args.amount ?? 3)) || 3))
    const sideways = direction === 'left' || direction === 'right'
    // The wheel counts away from the user as positive: up, and to the right.
    const notches = direction === 'up' || direction === 'right' ? amount : -amount
    await pointerOp({ op: 'scroll', x: at.x, y: at.y, notches, sideways }, ctx)
    return `Scrolled ${direction} ${amount} notch${amount === 1 ? '' : 'es'} at (${at.x}, ${at.y}).`
  }
}

const dragMouse: Tool = {
  name: 'drag_mouse',
  description:
    'Press the mouse button at one position, move to another and release: to drag a file, move a ' +
    'slider, resize a window or select a region. Controls the real desktop.',
  tier: 'execute',
  parameters: schema(
    {
      from_x: int('X position to press at'),
      from_y: int('Y position to press at'),
      to_x: int('X position to release at'),
      to_y: int('Y position to release at'),
      button: str("'left' (default) or 'right'"),
      coords: str(COORDS_HELP)
    },
    ['from_x', 'from_y', 'to_x', 'to_y']
  ),
  async run(args, ctx) {
    const from = toScreen(Number(args.from_x), Number(args.from_y), args.coords)
    const to = toScreen(Number(args.to_x), Number(args.to_y), args.coords)
    const right = String(args.button ?? 'left') === 'right'
    await pointerOp({ op: 'drag', x: from.x, y: from.y, x2: to.x, y2: to.y, right }, ctx)
    return `Dragged from (${from.x}, ${from.y}) to (${to.x}, ${to.y}).`
  }
}

export const uiTools: Tool[] = [uiInspect, uiClick, uiSetText, clickMouse, moveMouse, scrollMouse, dragMouse]
