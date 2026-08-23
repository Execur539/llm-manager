; Extract-once portable launcher.
;
; electron-builder's own `portable` target re-extracts the entire payload to TEMP on every
; launch and deletes it on exit — measured at ~24s per launch with this app's 3 GB payload.
; `portable.unpackDirName` only makes the directory *name* stable; it does not make the
; extraction skippable.
;
; This script produces a single exe that unpacks to a versioned directory under LOCALAPPDATA
; and writes a completion marker. Subsequent launches see the marker and skip straight to
; running the app, so only the first run pays the extraction cost.
;
; The marker is written last and includes the version, so a partial extraction (killed midway,
; out of disk) is never mistaken for a finished one, and a new build never reuses an old payload.

Unicode true
ManifestDPIAware true
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!include "FileFunc.nsh"
!include "LogicLib.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef PAYLOAD
  !error "PAYLOAD must be defined: the directory to embed"
!endif
!ifndef APPEXE
  !define APPEXE "LLM Manager.exe"
!endif

Name "LLM Manager ${VERSION}"
OutFile "${OUTFILE}"
SilentInstall silent
XPStyle on

Var RuntimeDir
Var Marker

Section
  StrCpy $RuntimeDir "$LOCALAPPDATA\LLMManager\runtime-${VERSION}"
  StrCpy $Marker "$RuntimeDir\.unpacked-${VERSION}"

  ; Fast path: a completed extraction of *this* version already exists.
  ${If} ${FileExists} "$Marker"
    ${AndIf} ${FileExists} "$RuntimeDir\${APPEXE}"
      Exec '"$RuntimeDir\${APPEXE}"'
      Return
  ${EndIf}

  ; Slow path, first run only. Clear any half-finished attempt before unpacking.
  RMDir /r "$RuntimeDir"
  CreateDirectory "$RuntimeDir"

  ; A banner is worth it here: this is a multi-gigabyte unpack and silence looks like a hang.
  Banner::show /NOUNLOAD "Setting up LLM Manager — this happens once..."

  SetOutPath "$RuntimeDir"
  File /r "${PAYLOAD}\*.*"

  Banner::destroy

  ; Marker written only after every file has landed.
  FileOpen $0 "$Marker" w
  FileWrite $0 "${VERSION}"
  FileClose $0

  ; Old versions leave their own runtime-<version> directories behind; remove the previous one
  ; so upgrades do not silently accumulate several gigabytes each.
  Call CleanOldRuntimes

  Exec '"$RuntimeDir\${APPEXE}"'
SectionEnd

Function CleanOldRuntimes
  FindFirst $0 $1 "$LOCALAPPDATA\LLMManager\runtime-*"
  loop:
    StrCmp $1 "" done
    StrCmp $1 "runtime-${VERSION}" next
    RMDir /r "$LOCALAPPDATA\LLMManager\$1"
  next:
    FindNext $0 $1
    Goto loop
  done:
  FindClose $0
FunctionEnd
