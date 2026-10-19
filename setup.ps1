# AutoLabeler - environment setup
# Creates a Python 3.12 virtual environment with all dependencies.
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
# Options: -Cpu (CPU-only torch)  -Cuda (CUDA torch for NVIDIA GPU)  -NoPt (skip .pt support)

param(
    [switch]$Cpu,
    [switch]$Cuda,
    [switch]$NoPt
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Root = $PSScriptRoot

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Find-BootstrapPython {
    $candidates = @()
    $onPath = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
    $candidates += Get-ChildItem "$env:ProgramFiles\Python*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    $candidates += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }

    foreach ($exe in $candidates) {
        try {
            $version = & $exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match '^\d+\.\d+$') {
                return $exe
            }
        } catch { }
    }
    return $null
}

Write-Step "Looking for a Python interpreter"
$python = Find-BootstrapPython
if (-not $python) {
    Write-Host "No Python installation found." -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/ and run this script again."
    exit 1
}
Write-Host "Found: $python"

Write-Step "Preparing uv (fast package manager)"
$uv = $null
$uvOnPath = Get-Command uv.exe -ErrorAction SilentlyContinue
if ($uvOnPath) {
    $uv = $uvOnPath.Source
} else {
    & $python -m pip install --user --upgrade uv --quiet
    $userBase = (& $python -m site --user-base).Trim()
    $candidate = Join-Path $userBase "Scripts\uv.exe"
    if (Test-Path -LiteralPath $candidate) {
        $uv = $candidate
    } else {
        $fallback = Join-Path (Split-Path $python) "Scripts\uv.exe"
        if (Test-Path -LiteralPath $fallback) { $uv = $fallback }
    }
}
if (-not $uv) {
    Write-Host "Could not install uv automatically." -ForegroundColor Red
    exit 1
}
Write-Host "uv: $uv"

Write-Step "Creating virtual environment (.venv, Python 3.12)"
& $uv venv "$Root\.venv" --python 3.12 --allow-existing
if (-not $?) { exit 1 }

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"

Write-Step "Installing core dependencies (PyQt6, OpenCV, ONNX Runtime, ...)"
& $uv pip install --python $venvPython -r "$Root\requirements.txt"
if (-not $?) { exit 1 }

$installTorch = $false
if (-not $NoPt) {
    if ($Cuda) {
        $installTorch = $true
        $cuda = $true
    } elseif ($Cpu) {
        $installTorch = $true
        $cuda = $false
    } else {
        Write-Host ""
        $answer = Read-Host "Install PyTorch + Ultralytics for .pt models? (NVIDIA GPU detected = recommended) [Y/n]"
        if ($answer -eq "" -or $answer -match '^(y|yes)$') {
            $installTorch = $true
            if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
                $cudaAnswer = Read-Host "Use CUDA build (GPU, ~2.5 GB download)? [Y/n]"
                $cuda = ($cudaAnswer -eq "" -or $cudaAnswer -match '^(y|yes)$')
            } else {
                $cuda = $false
            }
        }
    }
}

if ($installTorch) {
    if ($cuda) {
        Write-Step "Installing PyTorch with CUDA 12.6 (GPU)"
        & $uv pip install --python $venvPython torch --index-url https://download.pytorch.org/whl/cu126
        if (-not $?) { exit 1 }
    } else {
        Write-Step "Installing PyTorch (CPU)"
    }
    Write-Step "Installing Ultralytics"
    & $uv pip install --python $venvPython -r "$Root\requirements-pt.txt"
    if (-not $?) { exit 1 }
} else {
    Write-Host ""
    Write-Host "Skipping .pt support - only .onnx models will work." -ForegroundColor Yellow
    Write-Host "You can add it later:  $uv pip install --python `"$venvPython`" -r requirements-pt.txt"
}

Write-Step "Verifying installation"
& $venvPython -c "import PyQt6, cv2, numpy, PIL, yaml, onnxruntime; print('core ok')"
if ($LASTEXITCODE -ne 0) { exit 1 }
if ($installTorch) {
    & $venvPython -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ultralytics ok')"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start the app with:  run.bat"
