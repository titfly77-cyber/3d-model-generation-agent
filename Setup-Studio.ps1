param([string]$PythonPath = 'D:\Python312\python.exe')
$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
Set-Location -LiteralPath $studioRoot
foreach ($folder in @('tmp','cache\pip','cache\npm','cache\pycache','data')) { New-Item -ItemType Directory -Path (Join-Path $studioRoot $folder) -Force | Out-Null }
$env:TEMP = Join-Path $studioRoot 'tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPYCACHEPREFIX = Join-Path $studioRoot 'cache\pycache'
$env:PIP_CACHE_DIR = Join-Path $studioRoot 'cache\pip'
$env:npm_config_cache = Join-Path $studioRoot 'cache\npm'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { & $PythonPath -m venv .venv; if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' } }
$requirements = if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
& '.\.venv\Scripts\python.exe' -m pip install -r $requirements --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Push-Location -LiteralPath (Join-Path $studioRoot 'frontend')
try {
    & npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Node dependency installation failed.' }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Output 'Setup complete. Run Start-Studio.ps1.'
