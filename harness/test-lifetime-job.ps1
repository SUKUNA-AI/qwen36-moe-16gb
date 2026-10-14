[CmdletBinding()]
param(
    [switch]$RunOwnChildren,
    [ValidateSet('controller','owner','child')][string]$Role='controller',
    [ValidateSet('normal','crash','nested','assignment_refused','full_assignment_refused','stale_watcher','before_job')][string]$Case='crash',
    [string]$FixtureRoot='',
    [ValidateSet('working','grandchild')][string]$ChildName='working'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$source=Join-Path (Split-Path $PSScriptRoot -Parent) 'supervisor.ps1'
$watchSource=Join-Path (Split-Path $PSScriptRoot -Parent) 'watch-supervisor.ps1'
$workspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$scratch=Join-Path $workspace 'work'
$scriptPath=$PSCommandPath
function Json-File([string]$Path,$Value){
    [IO.File]::WriteAllText($Path,($Value|ConvertTo-Json -Depth 9),[Text.UTF8Encoding]::new($false))
}
function Identity($Process){
    $null=$Process.Handle
    return @{Pid=$Process.Id;Started=$Process.StartTime.ToUniversalTime().ToString('o')}
}
function Exact-Process($Value){
    if(-not $Value){return $null}
    try{
        $p=Get-Process -Id ([int]$Value.Pid) -ErrorAction SilentlyContinue
        if($p){$null=$p.Handle;if($p.StartTime.ToUniversalTime().ToString('o')-eq $Value.Started){return $p}}
    }catch{}
    return $null
}
function Spawn([string[]]$Arguments){
    return Start-Process -FilePath 'powershell.exe' -ArgumentList (@('-NoProfile','-File',('"'+$scriptPath+'"'))+$Arguments) -WindowStyle Hidden -PassThru
}
function Wait-File([string]$Path,[int]$Seconds=25){
    $until=[DateTime]::UtcNow.AddSeconds($Seconds)
    while([DateTime]::UtcNow-lt $until){
        if(Test-Path -LiteralPath $Path){
            try{return Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json}catch{}
        }
        Start-Sleep -Milliseconds 100
    }
    throw ('Fixture receipt deadline: '+[IO.Path]::GetFileName($Path))
}
function Wait-Gone($Value,[int]$Seconds=10){
    $until=[DateTime]::UtcNow.AddSeconds($Seconds)
    while([DateTime]::UtcNow-lt $until){if(-not(Exact-Process $Value)){return};Start-Sleep -Milliseconds 100}
    throw 'An exact synthetic process identity remains alive'
}
$tokens=$null;$parseErrors=$null
$tree=[Management.Automation.Language.Parser]::ParseFile($source,[ref]$tokens,[ref]$parseErrors)
if($parseErrors){throw 'Supervisor parser failure'}
$functions=$tree.FindAll({param($node) $node-is [Management.Automation.Language.FunctionDefinitionAst]},$false)
$native=@($functions|Where-Object Name -eq 'Initialize-LifetimeNative')[0]
if($Role-ne 'controller'){
    $FixtureRoot=[IO.Path]::GetFullPath($FixtureRoot)
    if(-not $FixtureRoot.StartsWith($scratch+'\',[StringComparison]::OrdinalIgnoreCase) -or -not(Test-Path -LiteralPath $FixtureRoot)){
        throw 'Synthetic process fixture must be a precreated workspace work directory'
    }
}
if($Role-eq 'child'){
    Json-File (Join-Path $FixtureRoot ($ChildName+'.json')) (Identity (Get-Process -Id $PID))
    if($ChildName-eq 'working'){
        $null=Spawn @('-Role','child','-ChildName','grandchild','-FixtureRoot',('"'+$FixtureRoot+'"'))
    }
    # Independent maximum lifetime if a test is interrupted before job setup.
    Start-Sleep -Seconds 90
    exit 0
}
if($Role-eq 'owner'){
    $watchReady=$null
    try{
        if($Case-eq 'full_assignment_refused'){
            & (Join-Path $FixtureRoot 'full-supervisor-fixture.ps1') -Root $FixtureRoot
            $fullCode=$LASTEXITCODE
            Json-File (Join-Path $FixtureRoot 'ready.json') @{Result='FULL_SOURCE_RETURNED';ExitCode=$fullCode;
                StateAbsent=(-not(Test-Path -LiteralPath (Join-Path $FixtureRoot 'run\supervisor.json')));
                NoWSL=(-not(Test-Path -LiteralPath (Join-Path $FixtureRoot 'FORBIDDEN-WSL')))}
            exit 0
        }
        if($Case-in @('nested','assignment_refused')){
            $literal=@($native.FindAll({param($node) $node-is [Management.Automation.Language.StringConstantExpressionAst] -and $node.Value.Contains('public static class Qwen36LifetimeJob')},$true))[0].Value
            $outer=$literal.Replace('Qwen36Power','Qwen36FixturePower').Replace('Qwen36LifetimeJob','Qwen36FixtureParentJob').Replace('limits.Basic.Flags = 0x00002000;','limits.Basic.Flags = 0;')
            Add-Type $outer
            [Qwen36FixtureParentJob]::ArmForCurrentProcess()
        }
        foreach($name in @('Initialize-LifetimeNative','Write-Event','Same-Process','Save-State','Start-OwnedChild')){
            $definition=@($functions|Where-Object Name -eq $name)[0]
            . ([scriptblock]::Create($definition.Extent.Text))
        }
        if($Case-eq 'assignment_refused'){
            # Fixture-only invalid process handle forces a real Win32 assignment
            # failure. It exercises the same fail-closed path as unsupported
            # nested jobs without altering an existing or foreign job object.
            $forced=$native.Extent.Text.Replace('AssignProcessToJobObject(job, GetCurrentProcess())','AssignProcessToJobObject(job, IntPtr.Zero)')
            . ([scriptblock]::Create($forced))
        }
        Initialize-LifetimeNative
        $Root=$FixtureRoot;$runDir=Join-Path $Root 'run';$logDir=Join-Path $Root 'logs'
        New-Item -ItemType Directory -Force -Path $runDir,$logDir|Out-Null
        $statePath=Join-Path $runDir 'supervisor.json'
        $ownerStarted=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
        $generation=[Guid]::NewGuid().ToString('N');$phase='synthetic-before-job';$children=@{};$restartTimes=@{}
        $jobArmed=$false;$modelStartRequested=$false
        $heartbeatSequence=0L;$keepaliveLeaseId=$null
        Save-State
        $readyName='Local\Qwen36WatchReady-'+$generation
        $watchReady=[Threading.EventWaitHandle]::new($false,[Threading.EventResetMode]::ManualReset,$readyName)
        $watchArgs=@('-NoProfile','-File',('"'+(Join-Path $Root 'watch-fixture.ps1')+'"'),'-OwnerPid',[string]$PID,
            '-OwnerStarted',$ownerStarted,'-ReadyEvent',$readyName,'-Root',('"'+$Root+'"'))
        Start-OwnedChild 'Watchdog' 'powershell.exe' $watchArgs
        $watcher=Same-Process $children['Watchdog'].Id $children['Watchdog'].Started
        Json-File (Join-Path $Root 'watcher-identity.json') (Identity $watcher)
        if(-not $watchReady.WaitOne(10000)){throw 'Fixture watcher handshake failed'}
        if($Case-ne 'before_job'){
            [Qwen36LifetimeJob]::ArmForCurrentProcess();$jobArmed=$true
            if([Qwen36LifetimeJob]::Contains($watcher.Handle)){throw 'Watcher unexpectedly joined owner job'}
            $phase='synthetic-job-ready';Save-State
            # Deliberately suppress the publication after a real child launch.
            # The production Start-OwnedChild and its job membership check run
            # unchanged; owner death is injected at its Save-State boundary.
            function Save-State {}
            Start-OwnedChild 'Tunnel' 'powershell.exe' @('-NoProfile','-File',('"'+$scriptPath+'"'),
                '-Role','child','-FixtureRoot',('"'+$Root+'"'),'-ChildName','working')
            $working=Wait-File (Join-Path $Root 'working.json')
            $grandchild=Wait-File (Join-Path $Root 'grandchild.json')
            $workingProcess=Exact-Process $working;$grandProcess=Exact-Process $grandchild
            if(-not $workingProcess -or -not $grandProcess){throw 'Synthetic descendants exited too soon'}
            $state=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
            Json-File (Join-Path $Root 'ready.json') @{
                Result='READY';Owner=(Identity (Get-Process -Id $PID));Watcher=(Identity $watcher);
                Working=$working;Grandchild=$grandchild;ModelStartRequested=$false;
                OwnerInJob=$true;WatcherInJob=[Qwen36LifetimeJob]::Contains($watcher.Handle);
                WorkingInJob=[Qwen36LifetimeJob]::Contains($workingProcess.Handle);
                GrandchildInJob=[Qwen36LifetimeJob]::Contains($grandProcess.Handle);
                HandleInheritable=[Qwen36LifetimeJob]::HandleIsInheritable();
                HasAncestorJob=[Qwen36LifetimeJob]::InheritedParentJob;
                ChildUnpublished=($null-eq $state.TunnelPid)
            }
        }else{
            Json-File (Join-Path $Root 'ready.json') @{Result='READY_BEFORE_JOB';Owner=(Identity (Get-Process -Id $PID));Watcher=(Identity $watcher);ModelStartRequested=$false}
        }
        $until=[DateTime]::UtcNow.AddSeconds(50)
        while([DateTime]::UtcNow-lt $until -and -not(Test-Path -LiteralPath (Join-Path $Root 'exit.request'))){Start-Sleep -Milliseconds 100}
        exit 0
    }catch{
        Json-File (Join-Path $FixtureRoot 'ready.json') @{Result='REFUSED';ErrorType=$_.Exception.GetBaseException().GetType().FullName;
            ModelStartRequested=$false;Owner=(Identity (Get-Process -Id $PID))}
        exit 1
    }finally{if($watchReady){$watchReady.Dispose()}}
}
if(-not $RunOwnChildren){
    [pscustomobject]@{Result='STATIC_PASS';OwnChildren='NOT_RUN';WSL='NOT_RUN';GPU='NOT_RUN';
        Cases=@('normal','crash','nested','assignment_refused','full_assignment_refused','stale_watcher','before_job')}|ConvertTo-Json
    exit 0
}
$fixture=Join-Path $scratch ('lifetime-job-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixture|Out-Null
$receipts=[Collections.Generic.List[object]]::new();$identities=[Collections.Generic.List[object]]::new()
$failure=$null
try{
    $caseNames=if($PSBoundParameters.ContainsKey('Case')){@($Case)}else{@('normal','crash','nested','assignment_refused','full_assignment_refused','stale_watcher','before_job')}
    foreach($caseName in $caseNames){
        $caseRoot=Join-Path $fixture $caseName
        New-Item -ItemType Directory -Path $caseRoot|Out-Null
        $watchText=[IO.File]::ReadAllText($watchSource)
        $fixtureMutex='Local\Qwen36Fixture-'+[Guid]::NewGuid().ToString('N')
        $watchText=$watchText.Replace('Global\Qwen36ApiSupervisor',$fixtureMutex)
        # Local synthetic copy: command trap makes any accidental WSL call fail.
        # The real source and its state/identity/cleanup logic are unchanged.
        $fixtureTrap=@'
$ErrorActionPreference='Stop'
function wsl.exe { [IO.File]::WriteAllText((Join-Path $Root 'FORBIDDEN-WSL'),'attempted'); throw 'Fixture forbids WSL' }
function Start-Process {
    param($FilePath,$ArgumentList,$WindowStyle,[switch]$PassThru,$RedirectStandardError)
    if($FilePath-ine 'powershell.exe'){
        [IO.File]::WriteAllText((Join-Path $Root 'FORBIDDEN-WSL'),'forbidden child');throw 'Fixture permits only its synthetic PowerShell watcher'
    }
    Microsoft.PowerShell.Management\Start-Process @PSBoundParameters
}
'@
        $watchText=$watchText.Replace("`$ErrorActionPreference='Stop'",$fixtureTrap)
        $watchText=$watchText.Replace('} finally {$mutex.ReleaseMutex();$mutex.Dispose()}',@'
} finally {
    $mutex.ReleaseMutex();$mutex.Dispose()
    [IO.File]::WriteAllText((Join-Path $Root 'watcher-finished.marker'),'outside watcher executed after owner death')
}
'@)
        [IO.File]::WriteAllText((Join-Path $caseRoot 'watch-fixture.ps1'),$watchText,[Text.UTF8Encoding]::new($false))
        if($caseName-eq 'full_assignment_refused'){
            [IO.File]::WriteAllText((Join-Path $caseRoot 'watch-supervisor.ps1'),$watchText,[Text.UTF8Encoding]::new($false))
            $fullText=[IO.File]::ReadAllText($source).Replace('Global\Qwen36ApiSupervisor',$fixtureMutex)
            $fullText=$fullText.Replace('AssignProcessToJobObject(job, GetCurrentProcess())','AssignProcessToJobObject(job, IntPtr.Zero)')
            $fullText=$fullText.Replace("`$ErrorActionPreference='Stop'",$fixtureTrap)
            [IO.File]::WriteAllText((Join-Path $caseRoot 'full-supervisor-fixture.ps1'),$fullText,[Text.UTF8Encoding]::new($false))
        }
        $owner=Spawn @('-Role','owner','-Case',$caseName,'-FixtureRoot',('"'+$caseRoot+'"'))
        $ownerIdentity=Identity $owner;$identities.Add($ownerIdentity)
        $row=Wait-File (Join-Path $caseRoot 'ready.json')
        if($caseName-eq 'full_assignment_refused'){
            if($row.Result-ne 'FULL_SOURCE_RETURNED' -or $row.ExitCode-ne 1 -or -not $row.StateAbsent -or -not $row.NoWSL){throw 'Full supervisor did not fail before WSL and clean its state'}
            Wait-Gone $ownerIdentity
            $receipts.Add(@{Case=$caseName;Result='PASS';FullSource=$row;WSL='NOT_RUN';GPU='NOT_RUN'})
            continue
        }
        $watcherIdentity=Wait-File (Join-Path $caseRoot 'watcher-identity.json');$identities.Add($watcherIdentity)
        if($row.PSObject.Properties['Working']){$identities.Add($row.Working);$identities.Add($row.Grandchild)}
        if($caseName-eq 'assignment_refused'){
            if($row.Result-ne 'REFUSED' -or $row.ErrorType-ne 'System.ComponentModel.Win32Exception'){throw 'Win32 assignment failure did not fail closed'}
            if(Test-Path -LiteralPath (Join-Path $caseRoot 'working.json')){throw 'A worker started despite failed job setup'}
        }elseif($caseName-eq 'before_job'){
            if($row.Result-ne 'READY_BEFORE_JOB'){throw 'Before-job fixture failed'}
        }else{
            if($row.Result-ne 'READY' -or $row.WatcherInJob -or -not $row.WorkingInJob -or -not $row.GrandchildInJob -or $row.HandleInheritable -or -not $row.ChildUnpublished){throw 'Lifetime membership invariant failed'}
            if($caseName-eq 'nested' -and -not $row.HasAncestorJob){throw 'Nested case has no ancestor job'}
        }
        if($row.ModelStartRequested){throw 'Synthetic fixture attempted model start'}
        $statePath=Join-Path $caseRoot 'run\supervisor.json'
        if($caseName-eq 'stale_watcher'){
            $changed=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
            $changed.SupervisorPid=$PID;$changed.SupervisorStarted=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o');$changed.Generation='new-fixture-owner'
            Json-File $statePath $changed
        }
        if($caseName-eq 'normal'){[IO.File]::WriteAllText((Join-Path $caseRoot 'exit.request'),'normal')}
        elseif($caseName-ne 'assignment_refused'){
            $exact=Exact-Process $ownerIdentity
            if(-not $exact){throw 'Synthetic owner identity changed before kill'}
            $exact.Kill();$null=$exact.WaitForExit(5000)
        }
        Wait-Gone $ownerIdentity
        if($caseName-notin @('assignment_refused','before_job')){Wait-Gone $row.Working;Wait-Gone $row.Grandchild}
        Wait-Gone $watcherIdentity 12
        if(-not(Test-Path -LiteralPath (Join-Path $caseRoot 'watcher-finished.marker'))){throw 'Watcher did not run outside-job cleanup'}
        if($caseName-eq 'stale_watcher'){
            $after=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
            if($after.Generation-ne 'new-fixture-owner'){throw 'Old watcher modified a new generation'}
        }elseif(Test-Path -LiteralPath $statePath){throw 'Watcher did not remove its own synthetic state'}
        if(Test-Path -LiteralPath (Join-Path $caseRoot 'FORBIDDEN-WSL')){throw 'Synthetic watcher attempted WSL'}
        $receipts.Add(@{Case=$caseName;Result='PASS';JobMembership=$row;WatcherCleanupAfterOwnerExit=$true;
            OwnedDescendantsGone=$true;WSL='NOT_RUN';GPU='NOT_RUN'})
    }
}catch{$failure=$_.Exception.GetType().FullName;$receipts.Add(@{Case='harness';Result='FAIL';ErrorType=$failure;Line=$_.InvocationInfo.ScriptLineNumber})}
finally{
    # Cleanup only exact identities from this fixture, never executable-name scans.
    foreach($identity in $identities){
        $p=Exact-Process $identity
        if($p){try{$p.Kill();$null=$p.WaitForExit(5000)}catch{$failure='SyntheticCleanupFailed'}}
    }
    $summary=@{Schema=1;Result=$(if($failure){'FAIL'}else{'PASS'});Cases=$receipts.ToArray();
        Scope='OWN_SYNTHETIC_WINDOWS_PROCESSES_ONLY';LiveService='NOT_RUN';Fixture=$fixture;
        SupervisorSha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash;
        WatcherSha256=(Get-FileHash -LiteralPath $watchSource -Algorithm SHA256).Hash}
    Json-File (Join-Path $fixture 'results.json') $summary
    $summary|ConvertTo-Json -Depth 9
}
if($failure){exit 1}
