param([ValidateSet('start','stop','status','restart','metrics')][string]$Action='start',
      [ValidateSet('','main128','long256','baseline64')][string]$Profile='',
      [string]$Root='C:\path\to\LLM\qwen36-api')
$ErrorActionPreference='Stop'
$statePath=Join-Path $Root 'run\supervisor.json'
$stopPath=Join-Path $Root 'run\stop.request'
function Read-SharedText([string]$Path,[int]$Attempts=5){
    # FileShare ReadWrite|Delete: this reader can never make the supervisor's
    # File.Replace heartbeat fail; a reader IOException during Replace is retried.
    $last=$null
    for($i=0;$i-lt $Attempts;$i++){
        try{
            $fs=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
            try{$reader=[IO.StreamReader]::new($fs,[Text.UTF8Encoding]::new($false));try{return $reader.ReadToEnd()}finally{$reader.Dispose()}}
            finally{$fs.Dispose()}
        }catch [IO.FileNotFoundException]{$last='FileNotFound'}
        catch [IO.DirectoryNotFoundException]{return $null}
        catch [IO.IOException]{$last='IOException'}
        catch [UnauthorizedAccessException]{$last='UnauthorizedAccess'}
        Start-Sleep -Milliseconds 50
    }
    if($last-eq 'FileNotFound'){return $null}
    throw ('Shared read failed after retries: '+$last)
}
function Read-State {
    if(-not(Test-Path -LiteralPath $statePath)){return $null}
    $raw=Read-SharedText $statePath
    if($null-eq $raw){return $null}
    return $raw|ConvertFrom-Json
}
function Same-Process([int]$ProcessId,[string]$Started){
    if($ProcessId-le 0 -or -not $Started){return $null}
    try {
        $p=Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if($p -and $p.StartTime.ToUniversalTime().ToString('o') -eq $Started){return $p}
    } catch {}
    return $null
}
function Current-Supervisor {
    $s=Read-State
    if($s){return Same-Process ([int]$s.SupervisorPid) ([string]$s.SupervisorStarted)}
    return $null
}
function Cleanup-Orphan {
    # The command mutex serializes profile selection; the owner mutex fences stop.
    $ownerMutex=[Threading.Mutex]::new($false,'Global\Qwen36ApiSupervisor')
    $held=$false
    try {
        try {$held=$ownerMutex.WaitOne(75000)} catch [Threading.AbandonedMutexException] {$held=$true}
        if(-not $held){throw 'Supervisor cleanup is still active after 75 seconds'}
        $s=Read-State
        if($s -and (Same-Process ([int]$s.SupervisorPid) ([string]$s.SupervisorStarted))){
            throw 'A supervisor became active; refusing to clean another generation'
        }
        foreach($name in @('Tunnel','HostMetrics','Watchdog')){
            if($s){
                $p=Same-Process ([int]$s.($name+'Pid')) ([string]$s.($name+'Started'))
                if($p){$p.Kill();$p.WaitForExit(5000)|Out-Null}
            }
        }
        & wsl.exe -d archlinux -- systemctl --user stop qwen36-inference.service
        if($LASTEXITCODE-ne 0){throw 'Owned inference stop failed; state retained'}
        if($s){
            $p=Same-Process ([int]$s.KeepalivePid) ([string]$s.KeepaliveStarted)
            if($p){$p.Kill();$p.WaitForExit(5000)|Out-Null}
            Remove-Item -LiteralPath $statePath -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $stopPath -ErrorAction SilentlyContinue
    } finally {if($held){$ownerMutex.ReleaseMutex()};$ownerMutex.Dispose()}
}
if($Action-eq 'status'){
    $p=Current-Supervisor
    Write-Output "Supervisor running: $([bool]$p)"
    if($p){$raw=Read-SharedText $statePath;if($null-ne $raw){Write-Output $raw}}
    & wsl.exe -d archlinux -- systemctl --user status qwen36-inference.service --no-pager
    if($p){
        & wsl.exe -d archlinux -- /home/USER/qwen36-api/.venv/bin/python /home/USER/qwen36-api/readiness.py --edge
    }else{
        & wsl.exe -d archlinux -- /home/USER/qwen36-api/.venv/bin/python /home/USER/qwen36-api/readiness.py
        exit 1
    }
    exit $LASTEXITCODE
}
if($Action-eq 'metrics'){
    & wsl.exe -d archlinux -- /home/USER/qwen36-api/.venv/bin/python /home/USER/qwen36-api/metrics_summary.py
    exit $LASTEXITCODE
}
$control=[Threading.Mutex]::new($false,'Global\Qwen36ApiControl')
$controlHeld=$false
try {
    try {$controlHeld=$control.WaitOne(10000)} catch [Threading.AbandonedMutexException] {$controlHeld=$true}
    if(-not $controlHeld){throw 'Another qwen36 start/stop/restart command is active'}
    if($Action-eq 'stop' -or $Action-eq 'restart'){
        New-Item -ItemType Directory -Force -Path (Join-Path $Root 'run')|Out-Null
        $s=Read-State
        if($s -and (Current-Supervisor)){
            $request=@{SupervisorPid=$s.SupervisorPid;SupervisorStarted=$s.SupervisorStarted}
            $temp=$stopPath+'.'+$PID+'.tmp'
            [IO.File]::WriteAllText($temp,($request|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
            if(Test-Path -LiteralPath $stopPath){[IO.File]::Replace($temp,$stopPath,[NullString]::Value)}else{[IO.File]::Move($temp,$stopPath)}
            $deadline=(Get-Date).AddSeconds(75)
            while((Current-Supervisor) -and (Get-Date)-lt $deadline){Start-Sleep -Milliseconds 250}
            if(Current-Supervisor){throw 'Supervisor did not stop within 75 seconds; inspect logs'}
        }
        Cleanup-Orphan
        if($Action-eq 'stop'){Write-Output 'Owned service and supervisor stopped';exit 0}
    }
    & wsl.exe -d archlinux -- systemctl --user is-active --quiet qwen36-experiment.service
    if($LASTEXITCODE-eq 0){throw 'The isolated experiment is active. Stop it before starting the public backend.'}
    if(Current-Supervisor){
        if($Profile){throw 'Stop the running service before changing its profile.'}
        & wsl.exe -d archlinux -- systemctl --user reset-failed qwen36-inference.service
        if($LASTEXITCODE-ne 0){
            # "not loaded" (garbage-collected inactive unit): nothing to reset; anything else stays fatal
            $active=& wsl.exe -d archlinux -- systemctl --user show qwen36-inference.service --property=ActiveState --value
            if($LASTEXITCODE-ne 0 -or ([string]$active).Trim()-ne 'inactive'){throw 'Could not reset owned inference service'}
        }
        & wsl.exe -d archlinux -- systemctl --user start qwen36-inference.service
        if($LASTEXITCODE-ne 0){throw 'Inference start failed'}
        Write-Output 'Supervisor is running; inference start requested; check status for readiness'
        exit 0
    }
    if(Test-Path -LiteralPath $statePath){Cleanup-Orphan}
    if(-not $Profile -and $Action-eq 'start'){$Profile='main128'}
    if($Profile){
        & wsl.exe -d archlinux -- /home/USER/qwen36-api/.venv/bin/python /home/USER/qwen36-api/profile-runtime.py select $Profile
        if($LASTEXITCODE-ne 0){throw 'Profile selection failed'}
    }
    $script=Join-Path $Root 'supervisor.ps1'
    $launcher=Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-File',('"'+$script+'"'),'-Root',('"'+$Root+'"')) -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(15)
    while(-not(Current-Supervisor) -and -not $launcher.HasExited -and (Get-Date)-lt $deadline){Start-Sleep -Milliseconds 100}
    if(-not(Current-Supervisor)){throw 'Supervisor did not publish its owner identity; inspect logs'}
    Write-Output 'Supervisor ownership established; model loading and route readiness are checked by qwen36 status'
} finally {if($controlHeld){$control.ReleaseMutex()};$control.Dispose()}
