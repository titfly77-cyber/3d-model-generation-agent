param([switch]$Rebuild, [switch]$Open)
$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
Set-Location -LiteralPath $studioRoot
foreach ($folder in @('tmp','cache','data')) { New-Item -ItemType Directory -Path (Join-Path $studioRoot $folder) -Force | Out-Null }
$env:TEMP = Join-Path $studioRoot 'tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPYCACHEPREFIX = Join-Path $studioRoot 'cache\pycache'
$env:PIP_CACHE_DIR = Join-Path $studioRoot 'cache\pip'
$env:npm_config_cache = Join-Path $studioRoot 'cache\npm'
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $studioRoot 'cache\playwright'
$python = Join-Path $studioRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run Setup-Studio.ps1 first.' }
if ($Rebuild -or -not (Test-Path -LiteralPath (Join-Path $studioRoot 'frontend\dist\index.html'))) {
    Push-Location -LiteralPath (Join-Path $studioRoot 'frontend')
    try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' } } finally { Pop-Location }
}
$alreadyRunning = $false
try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; $alreadyRunning = $health.app -eq 'forma-studio' } catch {}
if (-not $alreadyRunning) {
    & $python (Join-Path $studioRoot 'scripts\start_server.py')
    if ($LASTEXITCODE -ne 0) { throw 'Could not launch the Studio server.' }
    $ready = $false
    foreach ($attempt in 1..30) {
        Start-Sleep -Milliseconds 500
        try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; if ($health.app -eq 'forma-studio') { $ready = $true; break } } catch {}
    }
    if (-not $ready) { throw 'Server did not start. See data/server.stderr.log.' }
}
Write-Output 'FORMA Studio is running: http://127.0.0.1:8765'
if ($Open) { Start-Process 'http://127.0.0.1:8765' }
