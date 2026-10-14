[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$path=Join-Path $PSScriptRoot 'test-ops-v2.ps1'
$tokens=$null;$errors=$null
$tree=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
if($errors){throw 'Acceptance parser errors'}
# Load function definitions only. The acceptance script's live/top-level branch
# is never evaluated. Process and state lookups below are deterministic mocks.
$functions=$tree.FindAll({param($node) $node-is [Management.Automation.Language.FunctionDefinitionAst]},$false)
foreach($definition in $functions){. ([scriptblock]::Create($definition.Extent.Text))}
$events=[Collections.Generic.List[object]]::new()
$ownedIdentities=[Collections.Generic.List[object]]::new()
$checks=[Collections.Generic.List[string]]::new()
$script:killCalls=0;$script:waitCalls=0;$script:readIndex=0;$script:stateSequence=@()
$script:fakeProcesses=@{}
$script:createdStopToken=$false
$script:captured=$null;$script:current=$null;$script:check=$null
function Check([string]$Name,[bool]$Condition){if(-not $Condition){throw $Name};$checks.Add($Name)}
function Must-Refuse([string]$Name,[scriptblock]$Body){
    $failed=$false
    try{& $Body|Out-Null}catch{$failed=$true}
    Check $Name $failed
}
function Get-Process {
    param([int]$Id,[string]$ErrorAction)
    return $script:fakeProcesses[$Id]
}
function Read-State {
    if($script:readIndex-lt $script:stateSequence.Count){
        $answer=$script:stateSequence[$script:readIndex];$script:readIndex++;return $answer
    }
    return $script:current
}
function With-ControlLock([scriptblock]$Body){& $Body}
function Read-Ready{return $script:check}
function New-Identity([int]$Id){
    $process=[pscustomobject]@{Id=$Id;Handle=999;StartTime=[DateTime]::Parse('2026-01-01T00:00:00Z').ToUniversalTime()}
    $process|Add-Member ScriptMethod Kill {$script:killCalls++}
    $process|Add-Member ScriptMethod WaitForExit {param($Timeout) $script:waitCalls++;return $true}
    $script:fakeProcesses[$Id]=$process
    return $process
}
$supervisor=New-Identity 10;$tunnel=New-Identity 20
$creation=$supervisor.StartTime.ToUniversalTime().ToString('o')
$script:current=[pscustomobject]@{SupervisorPid=10;SupervisorStarted=$creation;Generation='fixture-a';
    TunnelPid=20;TunnelStarted=$creation;KeepalivePid=0;KeepaliveStarted=$null;
    WatchdogPid=0;WatchdogStarted=$null;HostMetricsPid=0;HostMetricsStarted=$null}
$script:captured=$script:current
Check 'same_owner_matches' (Same-Owner $captured $current)
Check 'null_owner_rejected' (-not(Same-Owner $null $current))
$other=$current|ConvertTo-Json|ConvertFrom-Json;$other.Generation='fixture-b'
Check 'different_generation_rejected' (-not(Same-Owner $captured $other))
Check 'same_process_matches_creation' ($null-ne (Same-Process 20 $creation))
Check 'reused_pid_different_creation_rejected' ($null-eq (Same-Process 20 '1900-01-01T00:00:00.0000000Z'))
$got=Crash-Owned 'Tunnel'
Check 'exact_owned_tunnel_killed_once' ($script:killCalls-eq 1 -and $script:waitCalls-eq 1 -and $got.Pid-eq 20)
$script:readIndex=0;$script:stateSequence=@($current,$other)
Must-Refuse 'owner_changed_between_reads_no_kill' {Crash-Owned 'Tunnel'}
Check 'owner_race_did_not_kill' ($script:killCalls-eq 1)
$otherTarget=$current|ConvertTo-Json|ConvertFrom-Json;$otherTarget.TunnelStarted='other'
$script:readIndex=0;$script:stateSequence=@($current,$otherTarget)
Must-Refuse 'target_changed_between_reads_no_kill' {Crash-Owned 'Tunnel'}
Check 'target_race_did_not_kill' ($script:killCalls-eq 1)
$script:stateSequence=@();$script:readIndex=0
$script:check=[pscustomobject]@{Code=1;Document=[pscustomobject]@{backend_ready=$true;instance_id='model-a';ready=$false;route=[pscustomobject]@{ready=$false}}}
Observe-TunnelFault 'model-a'
Check 'backend_ready_tunnel_down_reports_unready' ($events[$events.Count-1].Case-eq 'tunnel_outage_readiness' -and $events[$events.Count-1].Result-eq 'PASS')
$script:check.Document.ready=$true
Must-Refuse 'false_ready_during_tunnel_loss_rejected' {Observe-TunnelFault 'model-a'}
$script:check.Document.ready=$true;$script:check.Document.route.ready=$true;$script:check.Code=0
Observe-TunnelFault 'model-a'
Check 'fast_tunnel_recovery_is_not_claimed_as_observed_outage' ($events[$events.Count-1].Result-eq 'NOT_OBSERVED_ALREADY_RECOVERED')
$script:check.Document.instance_id='different'
Must-Refuse 'model_replacement_after_tunnel_fault_rejected' {Observe-TunnelFault 'model-a'}
# F-05/F-06: Wait-Ready refuses a ready document without instance identity or
# without a served-context proof of the acceptance profile.
$ReadyTimeoutSeconds=60;$Profile='main128'
$readyState=$current|ConvertTo-Json|ConvertFrom-Json
$readyState|Add-Member NoteProperty LifetimeJobArmed $true;$readyState|Add-Member NoteProperty ModelStartRequested $true
$script:current=$readyState;$script:captured=$readyState
function New-ReadyDocument([hashtable]$Change,[string[]]$Drop=@()){
    $doc=[ordered]@{ready=$true;model='qwen36-35b';backend_ready=$true;instance_id='model-a';profile='main128';profile_verified=$true;
        route=[ordered]@{ready=$true;tunnel=[ordered]@{ready=$true};https=[ordered]@{ready=$true}}}
    foreach($k in $Change.Keys){$doc[$k]=$Change[$k]}
    foreach($k in $Drop){$doc.Remove($k)}
    return ($doc|ConvertTo-Json -Depth 4|ConvertFrom-Json)
}
$script:check=[pscustomobject]@{Code=0;Document=(New-ReadyDocument @{})}
$doc=Wait-Ready 'fixture_ready'
Check 'ready_with_identity_and_verified_profile_passes' ($doc.instance_id-eq 'model-a' -and $events[$events.Count-1].Result-eq 'PASS')
foreach($case in @(@{Name='empty';Change=@{instance_id=''};Drop=@()},@{Name='null';Change=@{instance_id=$null};Drop=@()},@{Name='missing';Change=@{};Drop=@('instance_id')})){
    $script:check=[pscustomobject]@{Code=0;Document=(New-ReadyDocument $case.Change $case.Drop)}
    Must-Refuse ('ready_without_instance_identity_refused_'+$case.Name) {Wait-Ready 'fixture_no_identity'}
}
foreach($case in @(@{Name='false';Change=@{profile_verified=$false};Drop=@()},@{Name='missing';Change=@{};Drop=@('profile_verified')})){
    $script:check=[pscustomobject]@{Code=0;Document=(New-ReadyDocument $case.Change $case.Drop)}
    Must-Refuse ('ready_without_served_profile_proof_refused_'+$case.Name) {Wait-Ready 'fixture_unverified_profile'}
}
$script:check=[pscustomobject]@{Code=0;Document=(New-ReadyDocument @{})}
Must-Refuse 'same_instance_when_new_generation_expected_refused' {Wait-Ready 'fixture_restart' 'model-a'}
# F-07: top-level observation flag next to Result.
Check 'tunnel_outage_observed_true' ((Get-TunnelOutageObserved @(@{Case='tunnel_outage_readiness';Result='PASS'}))-eq $true)
Check 'tunnel_outage_not_observed_is_false' ((Get-TunnelOutageObserved @(@{Case='tunnel_outage_readiness';Result='NOT_OBSERVED_ALREADY_RECOVERED'}))-eq $false)
Check 'tunnel_outage_not_run' ((Get-TunnelOutageObserved @(@{Case='controlled_faults';Result='NOT_RUN'}))-eq 'NOT_RUN')
# F-11: native stderr under Stop is judged by exit code only (own cmd.exe child).
$r=Invoke-NativeLines 'cmd.exe' @('/c','echo synthetic-native-stderr 1>&2 & exit /b 3')
Check 'native_stderr_under_stop_is_not_terminating' ($r.Code-eq 3)
$script:fakeProcesses.Remove(10)
Must-Refuse 'dead_supervisor_refuses_fault' {Crash-Owned 'Tunnel'}
Check 'dead_owner_did_not_kill' ($script:killCalls-eq 1)
$scratch=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\work'))
$fixture=Join-Path $scratch ('ops-exit-fixture-'+[Guid]::NewGuid().ToString('N'))
if(-not([IO.Path]::GetFullPath($fixture).StartsWith($scratch+'\',[StringComparison]::OrdinalIgnoreCase))){throw 'Fixture escaped workspace'}
New-Item -ItemType Directory -Path $fixture|Out-Null
$managePath=Join-Path $fixture 'synthetic-manage.ps1'
$rootFull=$fixture
try{
    [IO.File]::WriteAllText($managePath,@'
param([string]$Action,[string]$Profile,[string]$Root)
$ErrorActionPreference='Stop'
& cmd.exe /c "echo synthetic-native-stderr 1>&2"
if($Action-eq 'status'){exit 3}
exit 0
'@,[Text.UTF8Encoding]::new($false))
    Invoke-Management 'start'
    Check 'nested_script_exit_zero_returns_to_harness' ($LASTEXITCODE-eq 0)
    Check 'nested_script_native_stderr_is_not_terminating' ($LASTEXITCODE-eq 0)
    Must-Refuse 'nested_script_exit_nonzero_is_detected' {Invoke-Management 'status'}
}finally{
    Remove-Item -LiteralPath $managePath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $fixture -ErrorAction SilentlyContinue
}
[pscustomobject]@{Result='PASS';Count=$checks.Count;Checks=$checks.ToArray();Live='NOT_RUN';RealSignals=0;NetworkCalls=0}|ConvertTo-Json -Depth 4
