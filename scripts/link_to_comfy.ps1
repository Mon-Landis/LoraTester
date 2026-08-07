param(
    [string]$ComfyPortable = "D:\ComfyUI\ComfyUI_windows_portable",
    [string]$PluginName = "LoraTester"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$customNodes = Join-Path $ComfyPortable "ComfyUI\custom_nodes"

if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
    throw "ComfyUI custom_nodes directory was not found: $customNodes"
}

$target = Join-Path $customNodes $PluginName
if (Test-Path -LiteralPath $target) {
    $item = Get-Item -Force -LiteralPath $target
    $existingTarget = @($item.Target) | Select-Object -First 1
    if ($item.LinkType -eq "Junction" -and $existingTarget) {
        $resolvedTarget = [System.IO.Path]::GetFullPath([string]$existingTarget)
        if ($resolvedTarget -eq [System.IO.Path]::GetFullPath($workspace)) {
            Write-Output "Development junction already exists: $target -> $workspace"
            exit 0
        }
    }
    throw "Refusing to replace an existing custom node path: $target"
}

$created = New-Item -ItemType Junction -Path $target -Target $workspace
Write-Output "Created development junction: $($created.FullName) -> $workspace"

