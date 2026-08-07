param(
    [string]$ComfyPortable = "D:\ComfyUI\ComfyUI_windows_portable"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $ComfyPortable "python_embeded\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ComfyUI embedded Python was not found: $python"
}

& $python (Join-Path $workspace "scripts\render_previews.py")
exit $LASTEXITCODE

