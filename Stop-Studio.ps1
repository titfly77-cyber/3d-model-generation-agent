$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
$pidFile = Join-Path $studioRoot 'data\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Output 'No saved server PID.'; exit }
$studioProcessId = [int](Get-Content -LiteralPath $pidFile)
$server = Get-CimInstance Win32_Process -Filter "ProcessId = $studioProcessId" -ErrorAction SilentlyContinue
if (-not $server) { Write-Output 'Studio is already stopped.'; exit }
if ($server.CommandLine -notmatch 'uvicorn.*backend\.main:app.*8765') { throw 'Saved PID does not belong to this Studio server.' }
$allProcesses = Get-CimInstance Win32_Process
function Stop-StudioChildren([int]$parentProcessId) {
    foreach ($child in $allProcesses | Where-Object { $_.ParentProcessId -eq $parentProcessId }) {
        Stop-StudioChildren $child.ProcessId
        Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Stop-StudioChildren $studioProcessId
Stop-Process -Id $studioProcessId -Force
Write-Output 'FORMA Studio stopped. Unfinished tasks can be resumed on next launch.'
