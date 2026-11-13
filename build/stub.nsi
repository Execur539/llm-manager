; The entry point that never moves: %LOCALAPPDATA%\LLMManager\LLM Manager.exe
;
; Shortcuts and taskbar pins aim here rather than at an unpacked copy of the app or at the
; portable exe. Unpacked copies live in folders named after their version and are removed when the
; next version unpacks, and the portable exe is wherever the user last put it. This stub stays put
; and starts whichever portable exe ran last, which in turn starts the newest unpacked version.
;
; The launcher writes this file on every run and records its own path in the registry, so the stub
; follows the portable exe when it is moved, as soon as it has been run once from its new place.
; The same file also stands in for the app inside an old version's folder after that version is
; cleaned up, so a pin made to an old version starts the newest one instead of nothing.
;
; Built by scripts/make-portable.mjs and carried inside the portable launcher.

Unicode true
ManifestDPIAware true
RequestExecutionLevel user
SetCompressor /SOLID lzma

!include "FileFunc.nsh"
!include "LogicLib.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef VERSION4
  !define VERSION4 "0.0.0.0"
!endif
!ifndef COPYRIGHT
  !define COPYRIGHT "LLM Manager"
!endif
!ifndef PORTABLE_NAME
  !define PORTABLE_NAME "LLM-Manager-portable.exe"
!endif

Name "LLM Manager"
OutFile "${OUTFILE}"
SilentInstall silent
XPStyle on

; The icon a pin shows is the icon of the file it points at, so this carries the app's own.
!ifdef ICON
  Icon "${ICON}"
!endif

VIProductVersion "${VERSION4}"
VIAddVersionKey "ProductName" "LLM Manager"
VIAddVersionKey "FileDescription" "LLM Manager"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "CompanyName" "Aidan Stafford"
VIAddVersionKey "LegalCopyright" "${COPYRIGHT}"

Var Launcher
Var Params

Section
  ${GetParameters} $Params

  ; Written by every launcher run that is not itself a download waiting to install.
  ReadRegStr $Launcher HKCU "Software\LLM Manager" "Launcher"

  ${If} $Launcher != ""
  ${AndIf} ${FileExists} "$Launcher"
    ; The portable exe's own folder as the working directory, as when it is double-clicked.
    ${GetParent} "$Launcher" $0
    SetOutPath "$0"
    Exec '"$Launcher" $Params'
    Quit
  ${EndIf}

  ${If} $Launcher == ""
    MessageBox MB_ICONEXCLAMATION "LLM Manager has not been started from ${PORTABLE_NAME} on this computer yet.$\n$\nStart it from that file once, and this shortcut will open it from then on."
  ${Else}
    MessageBox MB_ICONEXCLAMATION "LLM Manager could not find ${PORTABLE_NAME} where it was last started:$\n$\n$Launcher$\n$\nIt may have been moved or deleted. Start it from wherever it is now, and this shortcut will follow it."
  ${EndIf}
SectionEnd
