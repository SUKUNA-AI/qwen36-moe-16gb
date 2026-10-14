[CmdletBinding()]
param(
    [switch]$Live,
    [switch]$IncludeFaults,
    [ValidateSet('main128','long256','baseline64')][string]$Profile='main128',
    [ValidateRange(60,1200)][int]$ReadyTimeoutSeconds=480,
    [string]$Root='C:\path\to\LLM\qwen36-api',
    [string]$ReportPath=''
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$sourceRoot=Split-Path $PSScriptRoot -Parent
$rootFull=[IO.Path]::GetFullPath($Root).TrimEnd('\')
if($rootFull-ine 'C:\path\to\LLM\qwen36-api'){throw 'This acceptance script is scoped to C:\path\to\LLM\qwen36-api only'}
$statePath=Join-Path $rootFull 'run\supervisor.json'
$stopPath=Join-Path $rootFull 'run\stop.request'
$managePath=Join-Path $rootFull 'manage.ps1'
$testId=[Guid]::NewGuid().ToString('N')
$events=[Collections.Generic.List[object]]::new()
$captured=$null
$ownedIdentities=[Collections.Generic.List[object]]::new()
$createdStopToken=$false
$failure=$null
$started=[DateTime]::UtcNow

function Record([string]$Case,[string]$Result,$Data=$null){
    $item=[ordered]@{Case=$Case;Result=$Result;Utc=[DateTime]::UtcNow.ToString('o')}
    if($null-ne $Data){$item.Data=$Data}
    $events.Add($item)
}
function Parse-Script([string]$Path){
    $tokens=$null;$errors=$null
    [Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$errors)|Out-Null
    if($errors){throw 'PowerShell source parser failed'}
}
function Read-SharedText([string]$Path,[int]$Attempts=5){
    # FileShare ReadWrite|Delete (F-01/F-02): polling must never make the
    # supervisor's File.Replace heartbeat fail; reader IOExceptions are retried.
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
function Invoke-NativeLines([string]$File,[string[]]$Arguments){
    # F-11: Windows PowerShell 5.1 turns redirected native stderr into a
    # terminating error under Stop. Judge natives by exit code; stderr dropped.
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    try{$lines=@(& $File @Arguments 2>$null);$code=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
    return [pscustomobject]@{Lines=@($lines|ForEach-Object {[string]$_});Code=$code}
}
function Same-Process([int]$ProcessId,[string]$CreationTime){
    if($ProcessId-le 0 -or -not $CreationTime){return $null}
    try{
        $process=Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if($process){
            # Keep an open process handle across identity validation and Kill.
            $null=$process.Handle
            if($process.StartTime.ToUniversalTime().ToString('o')-eq $CreationTime){return $process}
        }
    }catch{}
    return $null
}
function Same-Owner($First,$Second){
    return $null-ne $First -and $null-ne $Second -and
        $First.SupervisorPid-eq $Second.SupervisorPid -and
        $First.SupervisorStarted-eq $Second.SupervisorStarted -and
        $First.Generation-eq $Second.Generation
}
function Require-Owner {
    $current=Read-State
    if(-not(Same-Owner $script:captured $current)){throw 'Acceptance owner generation changed; refusing mutation'}
    if(-not(Same-Process ([int]$current.SupervisorPid) ([string]$current.SupervisorStarted))){throw 'Acceptance owner is no longer alive'}
    return $current
}
function With-ControlLock([scriptblock]$Body){
    $mutex=[Threading.Mutex]::new($false,'Global\Qwen36ApiControl');$held=$false
    try{
        try{$held=$mutex.WaitOne(10000)}catch [Threading.AbandonedMutexException]{$held=$true}
        if(-not $held){throw 'Another management command is active'}
        # manage.ps1 is invoked in this PowerShell thread, so its same mutex is
        # reentrant. Do not replace the call with a child powershell process.
        & $Body
    }finally{if($held){$mutex.ReleaseMutex()};$mutex.Dispose()}
}
function Remember-Identities($State){
    foreach($name in @('Supervisor','Tunnel','Keepalive','Watchdog','HostMetrics')){
        $id=[int]$State.($name+'Pid');$creation=[string]$State.($name+'Started')
        if($id-gt 0 -and $creation){
            $script:ownedIdentities.Add([pscustomobject]@{Name=$name;Pid=$id;Started=$creation;Generation=$State.Generation})
        }
    }
}
function Invoke-Management([string]$Action,[string]$Selected=''){
    # Only stdout is discarded (F-11): redirecting the script's error stream
    # would turn any native stderr line inside manage.ps1 into a terminating error.
    if($Action-eq 'start' -or $Action-eq 'restart'){
        & $managePath -Action $Action -Profile $Selected -Root $rootFull | Out-Null
    }else{& $managePath -Action $Action -Root $rootFull | Out-Null}
    $code=$LASTEXITCODE
    if($code-ne 0){throw ('Management command failed: '+$Action)}
}
function Unit-State([string]$Unit='qwen36-inference.service'){
    $r=Invoke-NativeLines 'wsl.exe' @('-d','archlinux','--','systemctl','--user','show',$Unit,'--property=ActiveState','--value')
    if($r.Code-ne 0){throw 'Cannot establish named service state'}
    return ($r.Lines -join '').Trim()
}
function Assert-ManualConfiguration {
    $r=Invoke-NativeLines 'wsl.exe' @('-d','archlinux','--','systemctl','--user','is-enabled','qwen36-inference.service')
    $enabled=($r.Lines -join '').Trim()
    if($enabled-notin @('disabled','static')){throw 'Inference unit is enabled or its enablement is unknown'}
    $runKey=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Microsoft\Windows\CurrentVersion\Run',$false)
    try{
        if($runKey -and $runKey.GetValueNames()-contains 'Qwen36Api'){throw 'Known Qwen36 login entry exists'}
    }finally{if($runKey){$runKey.Dispose()}}
    Record 'manual_only_known_startup_paths' 'PASS' @{UserUnit=$enabled;KnownLoginEntry=$false;
        ScheduledTasks='NOT_AUDITED_ALL_TASK_NAMES';ColdBoot='NOT_RUN'}
}
function Read-Ready {
    $r=Invoke-NativeLines 'wsl.exe' @('-d','archlinux','--','/home/USER/qwen36-api/.venv/bin/python','/home/USER/qwen36-api/readiness.py','--edge')
    try{$doc=($r.Lines -join "`n")|ConvertFrom-Json}catch{throw 'Readiness did not return its sanitized JSON schema'}
    return [pscustomobject]@{Code=$r.Code;Document=$doc}
}
function Wait-Ready([string]$Case,[string]$DifferentInstance=''){
    $clock=[Diagnostics.Stopwatch]::StartNew()
    while($clock.Elapsed.TotalSeconds-lt $ReadyTimeoutSeconds){
        $ownerState=Require-Owner
        $check=Read-Ready
        if($check.Code-eq 0 -and $check.Document.ready-eq $true -and $check.Document.model-eq 'qwen36-35b'){
            if(-not $ownerState.LifetimeJobArmed -or -not $ownerState.ModelStartRequested){throw 'Ready service lacks lifetime ownership or model-start intent'}
            if($check.Document.profile-ne $Profile){throw 'Selected profile does not match acceptance profile'}
            # F-05: identity comparisons below are meaningless without an id.
            $instance=if($check.Document.PSObject.Properties['instance_id']){$check.Document.instance_id}else{$null}
            if($instance-isnot [string] -or -not $instance){throw 'Ready service lacks a model instance identity'}
            # F-06: the selection file alone does not prove the served profile.
            if(-not $check.Document.PSObject.Properties['profile_verified'] -or $check.Document.profile_verified-ne $true){throw 'Served context does not prove the acceptance profile'}
            if($DifferentInstance -and $instance-eq $DifferentInstance){throw 'Expected a new loaded model generation'}
            Record $Case 'PASS' @{ElapsedSeconds=$clock.Elapsed.TotalSeconds;InstanceId=$check.Document.instance_id;
                Profile=$check.Document.profile;BackendReady=$check.Document.backend_ready;
                TunnelReady=$check.Document.route.tunnel.ready;HttpsReady=$check.Document.route.https.ready;
                LifetimeJobArmed=$true;
                ExternalEndToEnd='NOT_RUN_CADDY_IS_HOME_LAN'}
            return $check.Document
        }
        Start-Sleep -Seconds 3
    }
    throw ('Readiness deadline reached: '+$Case)
}
function Assert-ChildrenGone($Identities){
    foreach($identity in $Identities){
        if(Same-Process ([int]$identity.Pid) ([string]$identity.Started)){throw ('An identified owned process remains: '+$identity.Name)}
    }
}
function Wait-Stopped([string]$Case,$Identities,[int]$Timeout=150){
    $deadline=[DateTime]::UtcNow.AddSeconds($Timeout)
    while([DateTime]::UtcNow-lt $deadline){
        $current=Read-State
        if($current -and -not(Same-Owner $script:captured $current)){throw 'New owner appeared during cleanup; refusing intervention'}
        $remaining=@($Identities|Where-Object {Same-Process ([int]$_.Pid) ([string]$_.Started)})
        if(-not $current -and $remaining.Count-eq 0 -and (Unit-State)-in @('inactive','failed')){
            Record $Case 'PASS' @{IdentifiedProcessesGone=$Identities.Count;UnitState=(Unit-State)}
            return
        }
        Start-Sleep -Seconds 2
    }
    throw ('Owned cleanup deadline reached: '+$Case)
}
function Start-New([string]$Case){
    With-ControlLock {
        if(Read-State){throw 'Start requires an absent owner state; no existing session is adopted'}
        if((Unit-State)-notin @('inactive','failed')){throw 'Start requires stopped inference'}
        if((Unit-State 'qwen36-experiment.service')-notin @('inactive','failed')){throw 'Isolated experiment is active'}
        # Passing Profile makes manage reject any preexisting live owner.
        Invoke-Management 'start' $Profile
        $script:captured=Read-State
        if(-not $script:captured -or -not $script:captured.Generation){throw 'New management generation was not published'}
        $null=Require-Owner
    }
    return Wait-Ready $Case
}
function Stop-Current([string]$Case){
    With-ControlLock {
        $current=Require-Owner;Remember-Identities $current
        Invoke-Management 'stop'
    }
    Wait-Stopped $Case @($script:ownedIdentities.ToArray())
    $script:captured=$null
}
function Crash-Owned([ValidateSet('Tunnel','HostMetrics','Supervisor')][string]$Name){
    With-ControlLock {
        $state=Require-Owner;Remember-Identities $state
        $id=[int]$state.($Name+'Pid');$creation=[string]$state.($Name+'Started')
        $process=Same-Process $id $creation
        if(-not $process){throw ('Identified target is not alive: '+$Name)}
        # Re-read the persisted owner and target immediately before the only
        # intentional kill in this harness. Never enumerate by executable name.
        $again=Require-Owner
        if($again.($Name+'Pid')-ne $id -or $again.($Name+'Started')-ne $creation){throw 'Target changed before fault injection'}
        $process.Kill()
        if(-not $process.WaitForExit(5000)){throw 'Identified target did not exit'}
        Record ('crash_'+$Name) 'INJECTED' @{Pid=$id;Started=$creation;Generation=$state.Generation}
        return [pscustomobject]@{Pid=$id;Started=$creation;Generation=$state.Generation}
    }
}
function Wait-ChildReplacement([string]$Name,$Previous){
    $deadline=[DateTime]::UtcNow.AddSeconds(45)
    while([DateTime]::UtcNow-lt $deadline){
        $state=Require-Owner
        $id=[int]$state.($Name+'Pid');$creation=[string]$state.($Name+'Started')
        if(($id-ne $Previous.Pid -or $creation-ne $Previous.Started) -and (Same-Process $id $creation)){
            Remember-Identities $state
            Record ('recovered_'+$Name) 'PASS' @{Pid=$id;Started=$creation;Generation=$state.Generation}
            return
        }
        Start-Sleep -Seconds 2
    }
    throw ('Child replacement deadline reached: '+$Name)
}
function Observe-TunnelFault([string]$ExpectedInstance){
    $null=Require-Owner
    $check=Read-Ready
    if(-not $check.Document.backend_ready -or $check.Document.instance_id-ne $ExpectedInstance){
        throw 'Tunnel fault unexpectedly disturbed the loaded backend'
    }
    if(-not $check.Document.route.ready){
        if($check.Code-eq 0 -or $check.Document.ready){throw 'Readiness treated a failed route as ready'}
        Record 'tunnel_outage_readiness' 'PASS' @{BackendReady=$true;RouteReady=$false;OverallReady=$false}
    }else{
        # Supervisor polls every five seconds. A replacement can recover before
        # this SSH probe; do not fabricate an observed outage in that case.
        Record 'tunnel_outage_readiness' 'NOT_OBSERVED_ALREADY_RECOVERED' @{BackendReady=$true;RouteReady=$true}
    }
}
function Get-TunnelOutageObserved($Items){
    # F-07: a tunnel replacement that recovered before the probe is not an
    # observed outage; expose this next to the top-level Result.
    $rows=@($Items|Where-Object {$_.Case-eq 'tunnel_outage_readiness'})
    if($rows.Count-eq 0){return 'NOT_RUN'}
    return @($rows|Where-Object {$_.Result-ne 'PASS'}).Count-eq 0
}
function Remove-TestStopToken {
    if(-not $script:createdStopToken){return}
    if(Test-Path -LiteralPath $stopPath){
        $value=Get-Content -LiteralPath $stopPath -Raw|ConvertFrom-Json
        if($value.PSObject.Properties['AcceptanceTestId'] -and $value.AcceptanceTestId-eq $testId){Remove-Item -LiteralPath $stopPath}
    }
    $script:createdStopToken=$false
}

# Default mode checks source syntax and presents a plan. It cannot invoke WSL,
# get a live process, read a secret, mutate run state, or start/stop anything.
$parsers=@($PSCommandPath,(Join-Path $sourceRoot 'manage.ps1'),(Join-Path $sourceRoot 'supervisor.ps1'),
    (Join-Path $sourceRoot 'watch-supervisor.ps1'),(Join-Path $sourceRoot 'host-metrics.ps1'))
foreach($path in $parsers){Parse-Script $path}
if(-not $Live){
    [pscustomobject]@{Result='STATIC_PASS';Parsers=$parsers.Count;Live='NOT_RUN';
        Plan=@('Require stopped own service and experiment','Verify installed management hashes',
               'Manual start + backend/tunnel/HTTPS readiness','Status + repeated start same generation',
               'Manual restart with changed supervisor and model identities','Ignore previous-generation stop token',
               'Optional crash tunnel and host metrics; keep model identity',
               'Optional crash supervisor; watcher cleanup, no automatic resurrection',
               'Manual start after owner death','Final stop; all captured PID identities gone');
        FaultsRequested=[bool]$IncludeFaults;AutoStartChanges=$false}|ConvertTo-Json -Depth 5
    exit 0
}
if(-not $ReportPath){$ReportPath=Join-Path $PSScriptRoot ('ops-v2-live-'+$started.ToString('yyyyMMddTHHmmss')+'-'+$testId.Substring(0,8)+'.json')}
if(Test-Path -LiteralPath $ReportPath){throw 'Refusing to overwrite an existing acceptance report'}
try{
    foreach($name in @('manage.ps1','supervisor.ps1','watch-supervisor.ps1','host-metrics.ps1')){
        $expected=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $sourceRoot $name)).Hash
        $actual=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $rootFull $name)).Hash
        if($actual-ne $expected){throw ('Installed management source differs: '+$name)}
    }
    Record 'installed_control_hashes' 'PASS'
    Assert-ManualConfiguration
    $first=Start-New 'manual_start_ready'
    With-ControlLock {
        $same=Require-Owner
        Invoke-Management 'status'
        Invoke-Management 'start'
        $null=Require-Owner
        if(-not(Same-Owner $same (Read-State))){throw 'Repeated start replaced the owner'}
    }
    $sameReady=Wait-Ready 'idempotent_start_ready'
    if($sameReady.instance_id-ne $first.instance_id){throw 'Repeated start unexpectedly replaced the model'}
    $oldOwner=$captured
    With-ControlLock {
        $state=Require-Owner;Remember-Identities $state
        Invoke-Management 'restart' $Profile
        $script:captured=Read-State
        if(Same-Owner $oldOwner $script:captured){throw 'Restart did not change supervisor identity'}
        $null=Require-Owner
    }
    $second=Wait-Ready 'manual_restart_ready' $first.instance_id
    Assert-ChildrenGone @($ownedIdentities.ToArray())
    With-ControlLock {
        $null=Require-Owner
        if(Test-Path -LiteralPath $stopPath){throw 'Unexpected stop request; refusing to overwrite it'}
        $token=@{SupervisorPid=$oldOwner.SupervisorPid;SupervisorStarted=$oldOwner.SupervisorStarted;AcceptanceTestId=$testId}
        [IO.File]::WriteAllText($stopPath,($token|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
        $script:createdStopToken=$true
        try{
            Start-Sleep -Seconds 12
            $null=Require-Owner
            Record 'old_generation_stop_ignored' 'PASS'
        }finally{Remove-TestStopToken}
    }
    $afterStale=Wait-Ready 'old_generation_stop_backend_still_ready'
    if($afterStale.instance_id-ne $second.instance_id){throw 'Stale stop token unexpectedly replaced the model'}
    if($IncludeFaults){
        foreach($name in @('Tunnel','HostMetrics')){
            $previous=Crash-Owned $name
            if($name-eq 'Tunnel'){Observe-TunnelFault $second.instance_id}
            Wait-ChildReplacement $name $previous
            $ready=Wait-Ready ($name.ToLowerInvariant()+'_recovery_ready')
            if($ready.instance_id-ne $second.instance_id){throw 'Helper recovery unexpectedly replaced the loaded model'}
        }
        $null=Crash-Owned 'Supervisor'
        Wait-Stopped 'owner_death_watcher_cleanup' @($ownedIdentities.ToArray())
        $captured=$null
        Start-Sleep -Seconds 15
        if((Read-State) -or (Unit-State)-notin @('inactive','failed')){throw 'Unexpected automatic resurrection after owner death'}
        Record 'no_resurrection_after_owner_death_15s' 'PASS'
        $third=Start-New 'manual_start_after_owner_death'
        if($third.instance_id-eq $second.instance_id){throw 'Owner-death recovery reused the old model identity'}
    }else{Record 'controlled_faults' 'NOT_RUN'}
    Stop-Current 'final_stop'
    & $managePath -Action status -Root $rootFull | Out-Null
    if($LASTEXITCODE-eq 0){throw 'Stopped status incorrectly reported readiness'}
    Record 'stopped_status_nonzero' 'PASS'
    Assert-ManualConfiguration
    Record 'lifecycle_acceptance' 'PASS'
}catch{
    # Only script-generated errors are exposed. Do not publish arbitrary native
    # stderr or the content of configuration/credential files in this report.
    $failure=$_.Exception.GetType().FullName
    Record 'lifecycle_acceptance' 'FAIL' @{ExceptionType=$failure;Line=$_.InvocationInfo.ScriptLineNumber}
}finally{
    try{With-ControlLock {Remove-TestStopToken}}catch{$failure='StopTokenCleanupFailed';Record 'test_stop_token_cleanup' 'FAIL'}
    if($captured){
        try{
            $current=Read-State
            if((Same-Owner $captured $current) -and (Same-Process ([int]$current.SupervisorPid) ([string]$current.SupervisorStarted))){
                Stop-Current 'failure_cleanup_owned_generation'
            }elseif($current){$failure='CleanupRefused';Record 'failure_cleanup' 'REFUSED_DIFFERENT_OR_DEAD_OWNER'}
        }catch{$failure='OwnedCleanupFailed';Record 'failure_cleanup' 'FAIL'}
    }
    $summary=[ordered]@{Schema=1;TestId=$testId;StartedUtc=$started.ToString('o');FinishedUtc=[DateTime]::UtcNow.ToString('o');
        Result=$(if($failure){'FAIL'}else{'PASS'});TunnelOutageObserved=(Get-TunnelOutageObserved $events.ToArray());
        Profile=$Profile;FaultsRequested=[bool]$IncludeFaults;
        Events=$events.ToArray();ColdWindowsBoot='NOT_RUN';ExternalEndToEnd='NOT_RUN';
        RestartLimitExhaustion='NOT_RUN';WorkerCrash='NOT_RUN';InferenceGeneration='SEPARATE_SMOKE_REQUIRED';
        StartupEnablementAudit='KNOWN_UNIT_AND_RUN_ENTRY_ONLY';AtomicCreateProcessGap='WINDOWS_JOB_MEMBERSHIP_IMPLEMENTED_OWN_CHILD_TESTS_PASS';
        FinalProcesses=$ownedIdentities.ToArray()}
    $reportStream=[IO.File]::Open([IO.Path]::GetFullPath($ReportPath),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{
        $reportBytes=[Text.UTF8Encoding]::new($false).GetBytes(($summary|ConvertTo-Json -Depth 9))
        $reportStream.Write($reportBytes,0,$reportBytes.Length)
    }finally{$reportStream.Dispose()}
    [pscustomobject]@{Result=$summary.Result;Report=$ReportPath;Events=$events.Count}|ConvertTo-Json
}
if($failure){exit 1}
