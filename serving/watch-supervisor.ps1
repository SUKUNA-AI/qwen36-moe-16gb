param([Parameter(Mandatory=$true)][int]$OwnerPid,[Parameter(Mandatory=$true)][string]$OwnerStarted,
      [ValidatePattern('^Local\\Qwen36WatchReady-[a-f0-9]{32}$')][string]$ReadyEvent='',
      [string]$Root='C:\path\to\LLM\qwen36-api')
$ErrorActionPreference='Stop'
$statePath=Join-Path $Root 'run\supervisor.json'
function Same-Process([int]$ProcessId,[string]$Started){
    if($ProcessId-le 0 -or -not $Started){return $null}
    try {
        $p=Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if($p -and $p.StartTime.ToUniversalTime().ToString('o') -eq $Started){return $p}
    } catch {}
    return $null
}
function Write-Event([string]$Message){
    try {Add-Content -LiteralPath (Join-Path $Root 'logs\supervisor.log') -Value "$(Get-Date -Format o) $Message"} catch {}
}
if($ReadyEvent -and (Same-Process $OwnerPid $OwnerStarted)){
    # Confirm the exact published owner before the supervisor joins its job.
    # This watcher never opens/retains the lifetime job handle.
    $initial=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
    if($initial.SupervisorPid-ne $OwnerPid -or $initial.SupervisorStarted-ne $OwnerStarted){exit 1}
    $ready=[Threading.EventWaitHandle]::OpenExisting($ReadyEvent)
    try{$null=$ready.Set()}finally{$ready.Dispose()}
}
while(Same-Process $OwnerPid $OwnerStarted){Start-Sleep -Seconds 3}
$mutex=[Threading.Mutex]::new($false,'Global\Qwen36ApiSupervisor')
$acquired=$false
try {$acquired=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$acquired=$true}
if(-not $acquired){$mutex.Dispose();exit 0}
try {
    if(-not(Test-Path -LiteralPath $statePath)){exit 0}
    $s=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
    if($s.SupervisorPid-ne $OwnerPid -or $s.SupervisorStarted-ne $OwnerStarted){exit 0}
    # This mutex fences the named service stop as well as Windows PID cleanup.
    foreach($name in @('Tunnel','HostMetrics')){
        $p=Same-Process ([int]$s.($name+'Pid')) ([string]$s.($name+'Started'))
        if($p){try{$p.Kill();$p.WaitForExit(5000)|Out-Null}catch{}}
    }
    # New supervisors publish model-start intent before any WSL start. If job
    # setup failed first, there is no Linux service of this generation to stop.
    # Legacy owner state lacks this field and retains the previous safe cleanup.
    $needsServiceStop=-not $s.PSObject.Properties['ModelStartRequested'] -or [bool]$s.ModelStartRequested
    $stopped=-not $needsServiceStop
    if($needsServiceStop){
        for($attempt=0;$attempt-lt 3;$attempt++){
            & wsl.exe -d archlinux -- systemctl --user stop qwen36-inference.service
            if($LASTEXITCODE-eq 0){$stopped=$true;break}
            Start-Sleep -Seconds 3
        }
    }
    if(-not $stopped){Write-Event 'Owner died; service stop unconfirmed after 3 attempts; state retained';exit 1}
    $p=Same-Process ([int]$s.KeepalivePid) ([string]$s.KeepaliveStarted)
    if($p){try{$p.Kill();$p.WaitForExit(5000)|Out-Null}catch{}}
    Write-Event 'Owner process died; owned service and identified children stopped'
    $current=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
    if($current.SupervisorPid-eq $OwnerPid -and $current.SupervisorStarted-eq $OwnerStarted){Remove-Item -LiteralPath $statePath}
} finally {$mutex.ReleaseMutex();$mutex.Dispose()}
