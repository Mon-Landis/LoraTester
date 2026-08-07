param(
    [string]$ComfyPortable = "D:\ComfyUI\ComfyUI_windows_portable"
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$comfyRoot = Join-Path $ComfyPortable "ComfyUI"
$python = Join-Path $ComfyPortable "python_embeded\python.exe"

if (-not (Test-Path -LiteralPath $comfyRoot -PathType Container)) {
    throw "ComfyUI source directory was not found: $comfyRoot"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ComfyUI embedded Python was not found: $python"
}

Write-Output "Workspace: $workspace"
Write-Output "ComfyUI:   $comfyRoot"
Write-Output "Python:    $python"
Write-Output ""

& $python -c "import PIL, numpy, torch, sys; print('Python', sys.version.split()[0]); print('Pillow', PIL.__version__); print('NumPy', numpy.__version__); print('Torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('CUDA runtime', torch.version.cuda); print('Device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

$comfyVersion = & git -C $comfyRoot log -1 --format="%h / %cs / %s"
Write-Output "ComfyUI $comfyVersion"

$node = Get-Command node -ErrorAction SilentlyContinue
if ($node) {
    Write-Output "Node.js $(node --version)"
} else {
    Write-Output "Node.js not installed (not required for the current Python-only stage)"
}

$parent = Split-Path $workspace -Parent
$folderName = Split-Path $workspace -Leaf
& $python -c "import sys; sys.path.insert(0, r'$parent'); plugin=__import__('$folderName'); print('Plugin import OK; registered nodes:', len(plugin.NODE_CLASS_MAPPINGS))"

