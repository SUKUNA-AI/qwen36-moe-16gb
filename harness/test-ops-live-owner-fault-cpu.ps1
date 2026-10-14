[CmdletBinding()]
param()
# CPU tests for ops-live-owner-fault.ps1. Only its function definitions are
# loaded; the script's top level (D:\ owner state, WSL, real mutex) never runs.
# Every process terminated here is an own synthetic powershell.exe started by
# this test (exact Process objects are kept and cleaned up), the inference unit
# is an own fixture file and the control mutex is a private Local\ name.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$path=Join-Path $PSScriptRoot 'ops-live-owner-fault.ps1'
$tokens=$null;$errors=$null
$tree=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
if($errors){throw 'Owner-fault parser errors'}
foreach($definition in $tree.FindAll({param($n) $n-is [Management.Automation.Language.FunctionDefinitionAst]},$false)){
    . ([scriptblock]::Create($definition.Extent.Text))
}
$checks=[Collections.Generic.List[string]]::new()
function Check([string]$Name,[bool]$Condition){if(-not $Condition){throw $Name};$checks.Add($Name)}

# --- static safety properties of the tool -------------------------------------
$source=[IO.File]::ReadAllText($path)
Check 'static_exactly_one_kill_call' ([regex]::Matches($source,'\.Kill\(\)').Count-eq 1)
$forbidden=@('taskkill','Stop-Process','Remove-Item','reset-failed','qwen36-api/secrets','qwen36-api\secrets','--shutdown','--terminate',
             'WriteAllText','Move-Item','Set-Content','Out-File','Copy-Item','Qwen36ApiSupervisor')
Check 'static_no_forbidden_actions_or_owner_mutex' (@($forbidden|Where-Object {$source.IndexOf($_,[StringComparison]::OrdinalIgnoreCase)-ge 0}).Count-eq 0)
Check 'static_no_unit_mutation' (-not ($source -match "systemctl[^\r\n]*'(stop|start|restart|kill|reset-failed)'"))
Check 'static_uses_management_control_mutex' ($source.Contains("[string]`$ControlMutexName='Global\Qwen36ApiControl'"))
Check 'unit_parsers' ((Test-UnitStopped ([ordered]@{ActiveState='failed';MainPID='0'})) -and -not (Test-UnitStopped ([ordered]@{ActiveState='inactive';MainPID='77'})) -and
    -not (Test-UnitStopped ([ordered]@{ActiveState='deactivating';MainPID='0'})) -and (Test-UnitServing ([ordered]@{ActiveState='active';MainPID='9'})) -and
    -not (Test-UnitServing ([ordered]@{ActiveState='active';MainPID='0'})))
Check 'exit_code_mapping' ((Get-OwnerFaultExit 'PASS')-eq 0 -and (Get-OwnerFaultExit 'PLAN_ONLY')-eq 0 -and (Get-OwnerFaultExit 'FAIL')-eq 1 -and (Get-OwnerFaultExit 'REFUSED')-eq 2)

# --- synthetic fixture ------------------------------------------------------
$workRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\work\agents\ops-live'))
$fixture=Join-Path $workRoot ('owner-fault-fixture-'+[Guid]::NewGuid().ToString('N'))
if(-not $fixture.StartsWith($workRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Fixture escaped workspace'}
New-Item -ItemType Directory -Path $fixture|Out-Null
$script:UnitFile=Join-Path $fixture 'unit.txt'
$statePath=Join-Path $fixture 'supervisor.json'
$mutexName='Local\Qwen36OwnerFaultTest-'+[Guid]::NewGuid().ToString('N')
$own=[Collections.Generic.List[Diagnostics.Process]]::new()

# Test override: the unit state comes from the own fixture file, never from WSL.
function Get-InferenceUnit {
    for($i=0;$i-lt 10;$i++){
        try{$t=@(([IO.File]::ReadAllText($script:UnitFile)).Trim() -split '\s+');if($t.Count-eq 2){return [ordered]@{ActiveState=$t[0];MainPID=$t[1]}}}catch{}
        Start-Sleep -Milliseconds 20
    }
    throw 'fixture unit unavailable'
}
function Set-Unit([string]$Text){[IO.File]::WriteAllText($script:UnitFile,$Text)}
function Start-Own([string[]]$Arguments){
    $p=Start-Process -FilePath powershell.exe -ArgumentList $Arguments -WindowStyle Hidden -PassThru
    $own.Add($p)
    $fresh=Get-Process -Id $p.Id
    return [pscustomobject]@{Process=$p;Pid=$p.Id;Started=$fresh.StartTime.ToUniversalTime().ToString('o')}
}
function Start-Sleeper {Start-Own @('-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 90')}
function Write-FixtureFile([string]$Name,[string]$Text){$file=Join-Path $fixture $Name;[IO.File]::WriteAllText($file,$Text,[Text.UTF8Encoding]::new($false));return $file}

# Synthetic watcher: waits for the exact watched identity to die, then (normal)
# kills the given own victims by exact identity, marks the fixture unit stopped
# and removes the fixture state; or does nothing (none); or publishes another
# generation (newowner). Bounded lifetime 60 s.
$cleaner=Write-FixtureFile 'cleaner.ps1' @'
param([int]$WatchPid,[string]$WatchStarted,[string]$Mode,[string]$StatePath,[string]$UnitFile,[string]$Victims)
$ErrorActionPreference='Stop'
function Exact([int]$Id,[string]$Started){try{$p=Get-Process -Id $Id -ErrorAction SilentlyContinue;if($p -and $p.StartTime.ToUniversalTime().ToString('o')-eq $Started){return $p}}catch{};return $null}
$deadline=(Get-Date).AddSeconds(60)
while((Get-Date)-lt $deadline -and (Exact $WatchPid $WatchStarted)){Start-Sleep -Milliseconds 100}
if((Get-Date)-ge $deadline){exit 3}
if($Mode-eq 'none'){exit 0}
if($Mode-eq 'newowner'){
    $s=[IO.File]::ReadAllText($StatePath)|ConvertFrom-Json;$s.Generation=('f'*32)
    $tmp=$StatePath+'.new.tmp';[IO.File]::WriteAllText($tmp,($s|ConvertTo-Json));[IO.File]::Replace($tmp,$StatePath,[NullString]::Value);exit 0
}
foreach($v in ($Victims -split ';')){if($v){$pair=$v -split '\|',2;$p=Exact ([int]$pair[0]) $pair[1];if($p){$p.Kill();$null=$p.WaitForExit(5000)}}}
[IO.File]::WriteAllText($UnitFile,'inactive 0')
[IO.File]::Delete($StatePath)
exit 0
'@
# Synthetic resurrection outside the owner identities: once the state is gone,
# mark the unit active again after one second.
$resurrector=Write-FixtureFile 'resurrector.ps1' @'
param([string]$StatePath,[string]$UnitFile)
$deadline=(Get-Date).AddSeconds(60)
while((Get-Date)-lt $deadline -and [IO.File]::Exists($StatePath)){Start-Sleep -Milliseconds 100}
Start-Sleep -Milliseconds 1000
[IO.File]::WriteAllText($UnitFile,'active 5555')
'@
$holder=Write-FixtureFile 'holder.ps1' @'
param([string]$Name,[string]$Ready)
$m=[Threading.Mutex]::new($true,$Name)
[IO.File]::WriteAllText($Ready,'1')
Start-Sleep -Seconds 60
'@

function Write-State($Ids,[string]$Generation,[hashtable]$Override=@{}){
    $state=[ordered]@{Generation=$Generation;Phase='model-loading-or-serving';LifetimeJobArmed=$true;ModelStartRequested=$true;
        KeepaliveLeaseId=('c'*32);HeartbeatSequence=1}
    foreach($name in @('Supervisor','Tunnel','Keepalive','Watchdog','HostMetrics')){$state[$name+'Pid']=$Ids[$name].Pid;$state[$name+'Started']=$Ids[$name].Started}
    foreach($k in $Override.Keys){$state[$k]=$Override[$k]}
    [IO.File]::WriteAllText($statePath,($state|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
}
function New-Owner([ValidateSet('Supervisor','Watchdog')][string]$Target,[string]$Mode='normal'){
    $ids=[ordered]@{Tunnel=(Start-Sleeper);Keepalive=(Start-Sleeper);HostMetrics=(Start-Sleeper)}
    $victims=(@('Tunnel','Keepalive','HostMetrics')|ForEach-Object {[string]$ids[$_].Pid+'|'+$ids[$_].Started}) -join ';'
    $targetProcess=Start-Sleeper
    $cleanerProcess=Start-Own @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$cleaner,'-WatchPid',[string]$targetProcess.Pid,
        '-WatchStarted',$targetProcess.Started,'-Mode',$Mode,'-StatePath',$statePath,'-UnitFile',$script:UnitFile,'-Victims',$victims)
    if($Target-eq 'Supervisor'){$ids.Supervisor=$targetProcess;$ids.Watchdog=$cleanerProcess}
    else{$ids.Watchdog=$targetProcess;$ids.Supervisor=$cleanerProcess}
    $generation=[Guid]::NewGuid().ToString('N')
    Write-State $ids $generation
    Set-Unit 'active 4242'
    return [pscustomobject]@{Ids=$ids;Generation=$generation}
}
function Invoke-Fixture([string]$Target,[string]$Generation,[bool]$Live,[int]$Cleanup=30,[int]$Hold=2,[int]$MutexWaitMs=2000){
    return Invoke-OwnerFault -StatePath $statePath -Target $Target -ExpectGeneration $Generation -Live $Live -CleanupTimeoutSeconds $Cleanup `
        -NoResurrectionSeconds $Hold -PollSeconds 0.2 -ControlMutexName $mutexName -MutexWaitMs $MutexWaitMs
}
function Alive($Identity){return Test-ExactAlive $Identity.Pid $Identity.Started}
function Stop-AllOwn {
    # Reverse start order: synthetic watchers die before the processes they watch.
    for($i=$own.Count-1;$i-ge 0;$i--){$p=$own[$i];try{if(-not $p.HasExited){$p.Kill();$null=$p.WaitForExit(5000)}}catch{}}
}

try{
    # --- preconditions and refusals: nothing may be signalled -----------------
    $o=New-Owner 'Supervisor'
    $r=Invoke-Fixture 'Supervisor' $o.Generation $false
    Check 'plan_only_names_target_and_sends_nothing' ($r.Result-eq 'PLAN_ONLY' -and $r.SignalsSent-eq 0 -and $r.TargetIdentity.Pid-eq $o.Ids.Supervisor.Pid -and (Alive $o.Ids.Supervisor))
    $r=Invoke-Fixture 'Supervisor' '' $false
    Check 'plan_without_generation_reports_it' ($r.Result-eq 'PLAN_ONLY' -and $r.Preconditions.GenerationMatches-eq 'NOT_REQUESTED' -and $r.Generation-eq $o.Generation)
    $r=Invoke-Fixture 'Supervisor' '' $true
    Check 'live_requires_expected_generation' ($r.Result-eq 'REFUSED' -and $r.Reason-eq 'expect_generation_required_for_live' -and (Alive $o.Ids.Supervisor))
    $r=Invoke-Fixture 'Supervisor' ('e'*32) $true
    Check 'live_refuses_other_generation' ($r.Result-eq 'REFUSED' -and $r.Reason-like '*GenerationMatches*' -and $r.SignalsSent-eq 0 -and (Alive $o.Ids.Supervisor))
    Write-State $o.Ids $o.Generation @{SupervisorStarted='2000-01-01T00:00:00.0000000Z'}
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    Check 'live_refuses_reused_pid_identity' ($r.Result-eq 'REFUSED' -and $r.Reason-like '*SupervisorAlive*' -and (Alive $o.Ids.Supervisor))
    $dead=Start-Sleeper;$dead.Process.Kill();$null=$dead.Process.WaitForExit(5000)
    Write-State $o.Ids $o.Generation @{WatchdogPid=$dead.Pid;WatchdogStarted=$dead.Started}
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    Check 'live_refuses_without_live_watcher' ($r.Result-eq 'REFUSED' -and $r.Reason-like '*WatchdogAlive*' -and (Alive $o.Ids.Supervisor))
    Write-State $o.Ids $o.Generation @{LifetimeJobArmed=$false}
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    Check 'live_refuses_unarmed_lifetime_job' ($r.Result-eq 'REFUSED' -and $r.Reason-like '*LifetimeJobArmed*' -and (Alive $o.Ids.Supervisor))
    Write-State $o.Ids $o.Generation
    Set-Unit 'inactive 0'
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    Check 'live_refuses_when_unit_not_serving' ($r.Result-eq 'REFUSED' -and $r.Reason-like '*InferenceServing*' -and (Alive $o.Ids.Supervisor))
    Set-Unit 'active 4242'
    $ready=Join-Path $fixture 'holder.ready'
    $mutexHolder=Start-Own @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$holder,'-Name',$mutexName,'-Ready',$ready)
    $deadline=(Get-Date).AddSeconds(20);while(-not [IO.File]::Exists($ready) -and (Get-Date)-lt $deadline){Start-Sleep -Milliseconds 100}
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true -MutexWaitMs 300
    Check 'live_refuses_while_management_mutex_held' ($r.Result-eq 'REFUSED' -and $r.Reason-eq 'control_mutex_busy' -and $r.SignalsSent-eq 0 -and (Alive $o.Ids.Supervisor))
    # Keep one unowned handle so the kernel object survives the holder's death:
    # the private mutex is then abandoned (as after a killed qwen36 command).
    $keepMutex=[Threading.Mutex]::new($false,$mutexName)
    $mutexHolder.Process.Kill();$null=$mutexHolder.Process.WaitForExit(5000)
    [IO.File]::Delete($statePath)
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    Check 'live_refuses_absent_owner' ($r.Result-eq 'REFUSED' -and $r.Reason-like '*OwnerStatePresent*' -and (Alive $o.Ids.Supervisor))
    Stop-AllOwn;$own.Clear()

    # --- live: supervisor death observed through the synthetic watcher ---------
    $o=New-Owner 'Supervisor'
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    $gone=@($o.Ids.Keys|Where-Object {Alive $o.Ids[$_]}).Count-eq 0
    Check 'live_supervisor_death_cleanup_pass' ($r.Result-eq 'PASS' -and $r.SignalsSent-eq 1 -and $null-ne $r.Cleanup.CompleteS -and
        $r.NoResurrection.Observations-ge 1 -and $gone -and -not [IO.File]::Exists($statePath))
    Check 'live_pass_after_abandoned_management_mutex' ($r.Result-eq 'PASS')
    $receiptName='owner-fault-fixture-receipt.json'
    $receipt=Write-OwnerFaultReceipt $r $fixture $receiptName
    $doc=[IO.File]::ReadAllText($receipt)|ConvertFrom-Json
    $second=$false;try{$null=Write-OwnerFaultReceipt $r $fixture $receiptName}catch{$second=$true}
    Check 'receipt_create_new_and_parseable' ($doc.Result-eq 'PASS' -and $doc.Target-eq 'Supervisor' -and $second -and
        ([IO.File]::ReadAllText($receipt)) -notmatch 'KeepaliveLeaseId')
    Stop-AllOwn;$own.Clear()

    $o=New-Owner 'Supervisor' 'none'
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true -Cleanup 3
    $left=@($r.Cleanup.RemainingAlive)
    Check 'live_missing_watcher_cleanup_fails_bounded' ($r.Result-eq 'FAIL' -and $r.Reason-eq 'cleanup_deadline_reached' -and $r.SignalsSent-eq 1 -and
        $left-contains 'Tunnel' -and $left-notcontains 'Supervisor' -and $null-eq $r.Cleanup.StateRemovedS -and [IO.File]::Exists($statePath))
    Stop-AllOwn;$own.Clear()

    $o=New-Owner 'Supervisor' 'newowner'
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true
    Check 'live_new_owner_during_cleanup_fails' ($r.Result-eq 'FAIL' -and $r.Reason-eq 'new_owner_appeared_during_cleanup')
    Stop-AllOwn;$own.Clear()

    $o=New-Owner 'Supervisor'
    $null=Start-Own @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$resurrector,'-StatePath',$statePath,'-UnitFile',$script:UnitFile)
    $r=Invoke-Fixture 'Supervisor' $o.Generation $true -Hold 5
    Check 'live_resurrection_after_cleanup_fails' ($r.Result-eq 'FAIL' -and $r.Reason-eq 'unit_resurrected_after_cleanup' -and $null-ne $r.Cleanup.CompleteS)
    Stop-AllOwn;$own.Clear()

    # --- live: watcher death, supervisor performs the cleanup (runbook E6) -------
    $o=New-Owner 'Watchdog'
    $r=Invoke-Fixture 'Watchdog' $o.Generation $true
    $gone=@($o.Ids.Keys|Where-Object {Alive $o.Ids[$_]}).Count-eq 0
    Check 'live_watcher_death_cleanup_pass' ($r.Result-eq 'PASS' -and $r.SignalsSent-eq 1 -and $r.TargetIdentity.Pid-eq $o.Ids.Watchdog.Pid -and $gone)
}finally{
    Stop-AllOwn
    if(Get-Variable -Name keepMutex -Scope Script -ErrorAction SilentlyContinue){$keepMutex.Dispose()}
    for($i=0;$i-lt 10 -and (Test-Path -LiteralPath $fixture);$i++){
        try{[IO.Directory]::Delete($fixture,$true)}catch{Start-Sleep -Milliseconds 300}
    }
}
[pscustomobject]@{Result='PASS';Count=$checks.Count;Checks=$checks.ToArray();Live='NOT_RUN';ServiceActions=0;
    OwnProcessesOnly=$true}|ConvertTo-Json -Depth 3
