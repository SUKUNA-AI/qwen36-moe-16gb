[CmdletBinding()]
param(
    [ValidateSet('Supervisor','Watchdog')][string]$Target='Supervisor',
    [ValidatePattern('^([a-f0-9]{32})?$')][string]$ExpectGeneration='',
    [switch]$Live,
    [ValidateRange(30,300)][int]$CleanupTimeoutSeconds=150,
    [ValidateRange(0,120)][int]$NoResurrectionSeconds=15,
    [ValidatePattern('^[A-Za-z0-9_.-]{1,48}$')][string]$Label='owner-fault',
    [string]$OutDir=''
)
# Owner-death fault for the qwen36 live acceptance runbook (phase C2 / E6).
# Without -Live it is READ-ONLY (PLAN_ONLY): it checks the preconditions and
# names the exact process it would terminate. With -Live it terminates exactly
# one process - the identity-verified (PID + StartTime, handle held) Supervisor
# or its outside-job Watchdog of the expected generation - under the same
# control mutex as qwen36 start/stop, and then only OBSERVES the cleanup done by
# the watcher (or by the supervisor): owner state removed, every recorded owner
# identity gone, the named inference unit inactive/failed with MainPID=0, and no
# resurrection for a hold period. It never starts or stops units, never kills by
# name, never writes or deletes owner state and never reads secrets. GPU release
# is judged afterwards by ops-live-snapshot.ps1 -CompareStoppedBaseline before
# any new start.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$Root='C:\path\to\LLM\qwen36-api'

function Read-SharedText([string]$Path,[int]$Attempts=5){
    # FileShare ReadWrite|Delete: this reader cannot break the supervisor's File.Replace (F-01).
    $last=$null
    for($i=0;$i-lt $Attempts;$i++){
        try{
            $fs=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
            try{$reader=[IO.StreamReader]::new($fs,[Text.UTF8Encoding]::new($false));try{return $reader.ReadToEnd()}finally{$reader.Dispose()}}
            finally{$fs.Dispose()}
        }catch [IO.FileNotFoundException]{$last='FileNotFound'}
        catch [IO.DirectoryNotFoundException]{return $null}
        catch [IO.IOException]{$last='IOException'}
        Start-Sleep -Milliseconds 50
    }
    if($last-eq 'FileNotFound'){return $null}
    throw ('Shared read failed after retries: '+$last)
}
function Get-Prop($Object,[string]$Name){
    if($null-ne $Object -and $Object.PSObject.Properties[$Name]){return $Object.$Name}
    return $null
}
function Read-OwnerState([string]$StatePath){
    $raw=Read-SharedText $StatePath
    if($null-eq $raw){return $null}
    $state=$raw|ConvertFrom-Json
    if($null-eq $state){throw 'Owner state is empty or invalid'}
    return $state
}
function Get-OwnerIdentities($State){
    $rows=[ordered]@{}
    foreach($name in @('Supervisor','Tunnel','Keepalive','Watchdog','HostMetrics')){
        $id=Get-Prop $State ($name+'Pid');$started=Get-Prop $State ($name+'Started')
        $rows[$name]=[ordered]@{Pid=$(if($null-ne $id){[int]$id}else{0});Started=$(if($null-ne $started){[string]$started}else{''})}
    }
    return $rows
}
function Open-ExactProcess([int]$ProcessId,[string]$Started){
    # Returns the process only when PID and creation time both match; the handle
    # is opened first, so the PID cannot be reused between the check and a Kill.
    if($ProcessId-le 0 -or -not $Started){return $null}
    $p=$null
    try{
        $p=Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if($p){
            $null=$p.Handle
            if(-not $p.HasExited -and $p.StartTime.ToUniversalTime().ToString('o')-eq $Started){return $p}
        }
    }catch{}
    if($p){$p.Dispose()}
    return $null
}
function Test-ExactAlive([int]$ProcessId,[string]$Started){
    $p=Open-ExactProcess $ProcessId $Started
    if($null-eq $p){return $false}
    $p.Dispose()
    return $true
}
function Invoke-NativeLines([string]$File,[string[]]$Arguments){
    # Windows PowerShell 5.1 turns redirected native stderr into a terminating
    # error under ErrorActionPreference=Stop; judge by the exit code instead.
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    try{$lines=@(& $File @Arguments 2>$null);$code=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
    return [pscustomobject]@{Lines=@($lines|ForEach-Object {[string]$_});Code=$code}
}
function Get-InferenceUnit {
    # Read-only.
    $r=Invoke-NativeLines 'wsl.exe' @('-d','archlinux','--','systemctl','--user','show','qwen36-inference.service','--property=ActiveState','--property=MainPID')
    if($r.Code-ne 0){throw 'systemctl show failed'}
    $unit=[ordered]@{}
    foreach($line in $r.Lines){
        $text=([string]$line).TrimEnd("`r")
        if(-not $text){continue}
        $pair=$text -split '=',2
        if($pair.Count-ne 2 -or $unit.Contains($pair[0])){throw 'Malformed systemctl show output'}
        $unit[$pair[0]]=$pair[1]
    }
    if(-not $unit.Contains('ActiveState') -or -not $unit.Contains('MainPID')){throw 'Incomplete unit state'}
    return $unit
}
function Test-UnitStopped($Unit){
    return ([string]$Unit['ActiveState']-in @('inactive','failed')) -and ([string]$Unit['MainPID']-eq '0')
}
function Test-UnitServing($Unit){
    return ([string]$Unit['ActiveState']-eq 'active') -and ([string]$Unit['MainPID']-ne '0') -and ([string]$Unit['MainPID']-ne '')
}
function Set-Outcome($Result,[string]$Outcome,[string]$Reason){
    $Result.Result=$Outcome;$Result.Reason=$Reason
    return $Result
}
function Invoke-OwnerFault {
    param([string]$StatePath,[ValidateSet('Supervisor','Watchdog')][string]$Target,[string]$ExpectGeneration,[bool]$Live,
          [int]$CleanupTimeoutSeconds=150,[int]$NoResurrectionSeconds=15,[double]$PollSeconds=2,
          [string]$ControlMutexName='Global\Qwen36ApiControl',[int]$MutexWaitMs=10000)
    $result=[ordered]@{Schema=1;Tool='ops-live-owner-fault';Utc=[DateTime]::UtcNow.ToString('o');Target=$Target;
        Mode=$(if($Live){'LIVE'}else{'PLAN_ONLY'});ExpectGeneration=$ExpectGeneration;Generation=$null;Result=$null;Reason=$null;
        SignalsSent=0;ServiceActions=0;OwnerStateWrites=0;Preconditions=[ordered]@{};Identities=$null;TargetIdentity=$null;
        GpuRelease='NOT_JUDGED_HERE: ops-live-snapshot.ps1 -CompareStoppedBaseline before any new start';ExternalEndToEnd='NOT_RUN'}
    try{
        if($Live -and $ExpectGeneration-notmatch '^[a-f0-9]{32}$'){return (Set-Outcome $result 'REFUSED' 'expect_generation_required_for_live')}
        $pre=$result.Preconditions
        $state=Read-OwnerState $StatePath
        $pre.OwnerStatePresent=$null-ne $state
        if(-not $state){return (Set-Outcome $result 'REFUSED' 'precondition_failed:OwnerStatePresent')}
        $generation=[string](Get-Prop $state 'Generation')
        $result.Generation=$generation
        $pre.GenerationMatches=$(if($ExpectGeneration){$generation-eq $ExpectGeneration}else{'NOT_REQUESTED'})
        $ids=Get-OwnerIdentities $state
        $result.Identities=$ids
        $result.TargetIdentity=$ids[$Target]
        $pre.SupervisorAlive=Test-ExactAlive $ids['Supervisor'].Pid $ids['Supervisor'].Started
        $pre.WatchdogAlive=Test-ExactAlive $ids['Watchdog'].Pid $ids['Watchdog'].Started
        $pre.LifetimeJobArmed=(Get-Prop $state 'LifetimeJobArmed')-eq $true
        $pre.ModelStartRequested=(Get-Prop $state 'ModelStartRequested')-eq $true
        $pre.PhaseServing=[string](Get-Prop $state 'Phase')-eq 'model-loading-or-serving'
        try{$pre.InferenceServing=Test-UnitServing (Get-InferenceUnit)}catch{$pre.InferenceServing=$false;$result.UnitQueryError=$_.Exception.GetType().Name}
        $failed=@($pre.Keys|Where-Object {$pre[$_]-is [bool] -and -not $pre[$_]})
        if($failed.Count){return (Set-Outcome $result 'REFUSED' ('precondition_failed:'+($failed -join ',')))}
        if(-not $Live){return (Set-Outcome $result 'PLAN_ONLY' 'preconditions_met_no_signal_sent')}

        $mutex=[Threading.Mutex]::new($false,$ControlMutexName);$held=$false
        try{
            try{$held=$mutex.WaitOne($MutexWaitMs)}catch [Threading.AbandonedMutexException]{$held=$true}
            if(-not $held){return (Set-Outcome $result 'REFUSED' 'control_mutex_busy')}
            # Re-read the persisted owner and the target identity right before the
            # only intentional signal of this tool. Never enumerate by executable name.
            $again=Read-OwnerState $StatePath
            if(-not $again -or [string](Get-Prop $again 'Generation')-ne $ExpectGeneration){return (Set-Outcome $result 'REFUSED' 'owner_changed_before_signal')}
            $now=(Get-OwnerIdentities $again)[$Target]
            if($now.Pid-ne $ids[$Target].Pid -or $now.Started-ne $ids[$Target].Started){return (Set-Outcome $result 'REFUSED' 'target_changed_before_signal')}
            $process=Open-ExactProcess $now.Pid $now.Started
            if(-not $process){return (Set-Outcome $result 'REFUSED' 'target_not_alive_before_signal')}
            try{
                $process.Kill()
                $result.SignalsSent=1
                $result.SignalUtc=[DateTime]::UtcNow.ToString('o')
                if(-not $process.WaitForExit(5000)){return (Set-Outcome $result 'FAIL' 'target_did_not_exit')}
            }finally{$process.Dispose()}
        }finally{if($held){$mutex.ReleaseMutex()};$mutex.Dispose()}

        # Observation only from here on.
        $clock=[Diagnostics.Stopwatch]::StartNew()
        $cleanup=[ordered]@{TimeoutSeconds=$CleanupTimeoutSeconds;StateRemovedS=$null;IdentitiesGoneS=$null;UnitStoppedS=$null;
            CompleteS=$null;RemainingAlive=@();LastUnit=$null;UnitQueryFailures=0}
        $result.Cleanup=$cleanup
        $complete=$false
        while($true){
            $s=Read-OwnerState $StatePath
            if($s -and [string](Get-Prop $s 'Generation')-ne $ExpectGeneration){
                return (Set-Outcome $result 'FAIL' 'new_owner_appeared_during_cleanup')
            }
            $alive=@(foreach($name in @($ids.Keys)){$i=$ids[$name];if($i.Pid-gt 0 -and (Test-ExactAlive $i.Pid $i.Started)){$name}})
            $stopped=$false
            try{
                $unit=Get-InferenceUnit
                $stopped=Test-UnitStopped $unit
                $cleanup.LastUnit=[ordered]@{ActiveState=[string]$unit['ActiveState'];MainPID=[string]$unit['MainPID']}
            }catch{$cleanup.UnitQueryFailures++}
            $t=[math]::Round($clock.Elapsed.TotalSeconds,1)
            if(-not $s -and $null-eq $cleanup.StateRemovedS){$cleanup.StateRemovedS=$t}
            if($alive.Count-eq 0 -and $null-eq $cleanup.IdentitiesGoneS){$cleanup.IdentitiesGoneS=$t}
            if($stopped -and $null-eq $cleanup.UnitStoppedS){$cleanup.UnitStoppedS=$t}
            $cleanup.RemainingAlive=$alive
            if(-not $s -and $alive.Count-eq 0 -and $stopped){$cleanup.CompleteS=$t;$complete=$true;break}
            if($clock.Elapsed.TotalSeconds-ge $CleanupTimeoutSeconds){break}
            Start-Sleep -Milliseconds ([int]($PollSeconds*1000))
        }
        if(-not $complete){return (Set-Outcome $result 'FAIL' 'cleanup_deadline_reached')}

        $hold=[Diagnostics.Stopwatch]::StartNew()
        $watch=[ordered]@{Seconds=$NoResurrectionSeconds;Observations=0}
        $result.NoResurrection=$watch
        while($true){
            $s=Read-OwnerState $StatePath
            try{$unit=Get-InferenceUnit}catch{return (Set-Outcome $result 'FAIL' 'unit_state_unavailable_during_no_resurrection')}
            $watch.Observations++
            if($s){return (Set-Outcome $result 'FAIL' 'owner_state_reappeared')}
            if(-not (Test-UnitStopped $unit)){return (Set-Outcome $result 'FAIL' 'unit_resurrected_after_cleanup')}
            if($hold.Elapsed.TotalSeconds-ge $NoResurrectionSeconds){break}
            Start-Sleep -Milliseconds ([int]($PollSeconds*1000))
        }
        return (Set-Outcome $result 'PASS' 'owner_death_cleanup_observed')
    }catch{
        $result.ErrorType=$_.Exception.GetType().Name
        return (Set-Outcome $result $(if($result.SignalsSent){'FAIL'}else{'REFUSED'}) 'unexpected_exception')
    }
}
function Write-OwnerFaultReceipt($Report,[string]$Directory,[string]$Name){
    New-Item -ItemType Directory -Force -Path $Directory|Out-Null
    $path=Join-Path $Directory $Name
    $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($Report|ConvertTo-Json -Depth 8));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
    return $path
}
function Get-OwnerFaultExit([string]$Outcome){
    switch($Outcome){'PASS'{return 0}'PLAN_ONLY'{return 0}'FAIL'{return 1}default{return 2}}
}

$report=Invoke-OwnerFault -StatePath (Join-Path $Root 'run\supervisor.json') -Target $Target -ExpectGeneration $ExpectGeneration -Live ([bool]$Live) `
    -CleanupTimeoutSeconds $CleanupTimeoutSeconds -NoResurrectionSeconds $NoResurrectionSeconds
if(-not $OutDir){$OutDir=Join-Path $PSScriptRoot 'ops-live-receipts'}
$name='ops-live-owner-fault-'+$Label+'-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+[Guid]::NewGuid().ToString('N').Substring(0,6)+'.json'
$receipt=Write-OwnerFaultReceipt $report $OutDir $name
$exit=Get-OwnerFaultExit ([string]$report.Result)
[pscustomobject]@{Receipt=$receipt;Result=$report.Result;Reason=$report.Reason;Target=$Target;Generation=$report.Generation;
    SignalsSent=$report.SignalsSent;Exit=$exit}|ConvertTo-Json
exit $exit
