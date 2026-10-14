param([string]$Workspace='C:\path\to\qwen36-workspace')
$ErrorActionPreference='Stop'
$base=Resolve-Path $Workspace
$out=Join-Path $base 'outputs\qwen36-api'
$scratch=Join-Path $base 'work\cuda-transport\ops-test-scratch'
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$results=[Collections.Generic.List[object]]::new()
function Assert([bool]$Condition,[string]$Name){if(-not $Condition){throw "FAILED: $Name"};$results.Add(@{Name=$Name;Pass=$true})}
foreach($file in @('manage.ps1','supervisor.ps1','watch-supervisor.ps1','host-metrics.ps1','optimization\release-management.ps1')){
    $tokens=$null;$errors=$null
    [Management.Automation.Language.Parser]::ParseFile((Join-Path $out $file),[ref]$tokens,[ref]$errors)|Out-Null
    Assert (@($errors).Count-eq 0) ('parse '+$file)
}
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $out 'supervisor.ps1'),[ref]$tokens,[ref]$errors)
foreach($fn in $ast.FindAll({param($node)$node-is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
    . ([ScriptBlock]::Create($fn.Extent.Text))
}
$statePath=Join-Path $scratch 'state.json'
$stopPath=Join-Path $scratch 'stop.json'
$ownerStarted='2026-10-13T00:00:00.0000000Z'
$generation='cpu-fixture'
$phase='starting'
$jobArmed=$false;$modelStartRequested=$false
$heartbeatSequence=0L;$keepaliveLeaseId=$null
$children=@{}
$logDir=Join-Path $scratch 'logs'
$fakeProcess=[pscustomobject]@{Id=1234;StartTime=[DateTime]::Parse('2026-10-13T00:00:00Z').ToUniversalTime();Handle=[IntPtr]::Zero}
$script:kills=0;$script:waits=0;$script:spawns=0
$fakeProcess|Add-Member ScriptMethod Kill {$script:kills++}
$fakeProcess|Add-Member ScriptMethod WaitForExit {param($timeout)$script:waits++;return $true}
function Get-Process {param($Id,$ErrorAction)if($Id-eq 1234){return $fakeProcess};return $null}
Assert ($null-eq (Same-Process 1234 '2026-10-12T00:00:00.0000000Z')) 'reject reused PID creation time'
Assert ($null-ne (Same-Process 1234 $ownerStarted)) 'accept exact PID creation time'
$children['Tunnel']=@{Id=1234;Started='2026-10-12T00:00:00.0000000Z'}
Stop-OwnedChild 'Tunnel'
Assert ($kills-eq 0) 'never kill reused child PID'
$children['Tunnel'].Started=$ownerStarted
Stop-OwnedChild 'Tunnel'
Assert ($kills-eq 1 -and $waits-eq 1) 'kill only mocked owned process and await exit'
Save-State
$saved=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
Assert ($saved.SupervisorPid-eq $PID -and $saved.TunnelPid-eq 1234 -and $saved.Generation-eq $generation) 'publish owner and child identities'
Assert ($saved.HeartbeatSequence-eq 1) 'first published heartbeat is numeric and positive'
$phase='model-loading-or-serving';Save-State
Assert ((Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json).Phase-eq $phase) 'atomic replacement preserves valid JSON'
Assert ((Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json).HeartbeatSequence-eq 2) 'atomic replacement advances heartbeat'
# F-01: a Windows reader without FILE_SHARE_DELETE (held by an in-process
# handle released from a .NET thread) must not stop the supervisor.
$stateWriteAttempts=20;$heartbeatFailureSeconds=20
Add-Type @'
using System; using System.IO; using System.Threading;
public static class Qwen36CpuHold {
    public static FileStream Open(string path) { return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite); }
    public static void ReleaseAfter(FileStream stream, int ms) { var t = new Thread(() => { Thread.Sleep(ms); stream.Dispose(); }); t.IsBackground = true; t.Start(); }
}
'@
$held=[Qwen36CpuHold]::Open($statePath);[Qwen36CpuHold]::ReleaseAfter($held,300)
$clock=[Diagnostics.Stopwatch]::StartNew();Save-State;$clock.Stop()
Assert ((Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json).HeartbeatSequence-eq 3 -and $clock.ElapsedMilliseconds-ge 200) 'state replace retries a transient sharing violation'
$stateWriteAttempts=3
$held=[Qwen36CpuHold]::Open($statePath)
try{
    $threw=$false;$clock=[Diagnostics.Stopwatch]::StartNew()
    try{Save-State}catch{$threw=$_.Exception.GetBaseException()-is [IO.IOException]}
    $clock.Stop()
    Assert ($threw -and $clock.ElapsedMilliseconds-lt 3000) 'persistent sharing violation fails closed after bounded retries'
    $lastHeartbeatWrite=[Diagnostics.Stopwatch]::StartNew()
    $tolerated=$true;try{Save-Heartbeat}catch{$tolerated=$false}
    Assert $tolerated 'heartbeat write failure tolerated while last heartbeat is younger than 20 s'
    $heartbeatFailureSeconds=0
    $threw=$false;try{Save-Heartbeat}catch{$threw=$true}
    Assert $threw 'heartbeat write failure beyond the tolerance window stops fail-closed'
}finally{$held.Dispose();$stateWriteAttempts=20;$heartbeatFailureSeconds=20}
Save-Heartbeat
Assert ((Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json).HeartbeatSequence-gt 3) 'heartbeat resumes after the reader releases'
# F-01/F-02 stress: own synthetic writer process runs the real Save-State every
# 20 ms; this process alternates an old-style Get-Content reader (no
# FILE_SHARE_DELETE) with manage.ps1 Read-State (shared read + retry).
$manageAst=[Management.Automation.Language.Parser]::ParseFile((Join-Path $out 'manage.ps1'),[ref]$tokens,[ref]$errors)
foreach($fn in $manageAst.FindAll({param($node)$node-is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
    if($fn.Name-in @('Read-SharedText','Read-State')){. ([ScriptBlock]::Create($fn.Extent.Text))}
}
$stress=Join-Path $scratch ('stress-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stress|Out-Null
$stressState=Join-Path $stress 'supervisor.json'
$writer=Join-Path $stress 'writer.ps1'
[IO.File]::WriteAllText($writer,@'
param([string]$Source,[string]$State,[int]$Seconds)
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)
foreach($fn in $ast.FindAll({param($n)$n-is [Management.Automation.Language.FunctionDefinitionAst]},$true)){if($fn.Name-eq 'Save-State'){. ([ScriptBlock]::Create($fn.Extent.Text))}}
$statePath=$State;$generation='stress-fixture';$ownerStarted='2026-10-14T00:00:00.0000000Z';$phase='stress'
$jobArmed=$true;$modelStartRequested=$false;$heartbeatSequence=0L;$keepaliveLeaseId=$null;$children=@{};$stateWriteAttempts=20
$fail=0;$n=0;$clock=[Diagnostics.Stopwatch]::StartNew()
while($clock.Elapsed.TotalSeconds-lt $Seconds){$n++;try{Save-State}catch{$fail++};Start-Sleep -Milliseconds 20}
"$n $fail"
'@,[Text.UTF8Encoding]::new($false))
$savedState=$statePath
try{
    $statePath=$stressState
    $writerOut=Join-Path $stress 'writer.out'
    $proc=Start-Process -FilePath powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$writer+'"'),'-Source',('"'+(Join-Path $out 'supervisor.ps1')+'"'),'-State',('"'+$stressState+'"'),'-Seconds','6') -NoNewWindow -PassThru -RedirectStandardOutput $writerOut
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    while(-not(Test-Path -LiteralPath $stressState) -and [DateTime]::UtcNow-lt $deadline){Start-Sleep -Milliseconds 20}
    $sharedReads=0;$sharedFailures=0;$plainReads=0;$plainFailures=0;$clock=[Diagnostics.Stopwatch]::StartNew()
    while($clock.Elapsed.TotalSeconds-lt 5 -and -not $proc.HasExited){
        try{$null=(Get-Content -LiteralPath $stressState -Raw|ConvertFrom-Json).HeartbeatSequence;$plainReads++}catch{$plainFailures++}
        try{$s=Read-State;if($null-eq $s){throw 'absent'};$sharedReads++}catch{$sharedFailures++}
        Start-Sleep -Milliseconds 5
    }
    if(-not $proc.WaitForExit(30000)){throw 'Own synthetic writer did not exit'}
    $parts=((Get-Content -LiteralPath $writerOut -Raw).Trim() -split ' ')
    $stressFacts=@{Writes=[int]$parts[0];WriterFailures=[int]$parts[1];PlainReads=$plainReads;PlainReaderIOExceptions=$plainFailures;SharedReads=$sharedReads;SharedFailures=$sharedFailures}
    Assert ([int]$parts[0]-gt 50 -and [int]$parts[1]-eq 0) 'supervisor Save-State survives concurrent non-delete-sharing readers'
    Assert ($sharedFailures-eq 0 -and $sharedReads-gt 50) 'manage Read-State never fails during supervisor Replace'
}finally{
    $statePath=$savedState
    Remove-Item -LiteralPath $stress -Recurse -Force -ErrorAction SilentlyContinue
}
$statePath=Join-Path $scratch ('absent-'+[Guid]::NewGuid().ToString('N')+'.json')
Assert ($null-eq (Read-State)) 'manage Read-State returns null for absent state'
$statePath=$savedState
[IO.File]::WriteAllText($stopPath,'{"SupervisorPid":1234,"SupervisorStarted":"unrelated"}')
Assert (-not(Stop-Requested)) 'ignore unrelated generation stop request'
@{SupervisorPid=$PID;SupervisorStarted=$ownerStarted}|ConvertTo-Json|Set-Content -LiteralPath $stopPath
Assert (Stop-Requested) 'accept matching generation stop request'
[IO.File]::WriteAllText($stopPath,'partial JSON')
Assert (-not(Stop-Requested)) 'ignore malformed stop request'
function Add-Content {throw [IO.IOException]::new('synthetic ENOSPC')}
Write-Event 'test only'
Assert $true 'disk-full log failure does not throw'
function Start-Process {param($FilePath,$ArgumentList,$WindowStyle,$PassThru,$RedirectStandardError)$script:spawns++;return $fakeProcess}
$restartTimes=@{Watchdog=[Collections.Generic.List[datetime]]::new()}
for($i=0;$i-lt 6;$i++){$restartTimes.Watchdog.Add((Get-Date))}
$limited=$false
try{Start-OwnedChild 'Watchdog' 'mock.exe' @('mock')}catch{$limited=$_.Exception.Message-like '*restart limit*'}
Assert ($limited -and $spawns-eq 0) 'six starts per fifteen minutes enforced before spawn'
$restartTimes.Watchdog.Clear()
for($i=0;$i-lt 6;$i++){$restartTimes.Watchdog.Add((Get-Date).AddMinutes(-16))}
Start-OwnedChild 'Watchdog' 'mock.exe' @('mock')
Assert ($spawns-eq 1 -and $restartTimes.Watchdog.Count-eq 1) 'expired restart window permits one mocked spawn'
$jobArmed=$true
$refused=$false;try{Start-OwnedChild 'Watchdog' 'mock.exe' @('mock')}catch{$refused=$_.Exception.Message-like '*cannot be restarted*'}
Assert ($refused -and $spawns-eq 1) 'watcher cannot inherit job through a respawn'
$jobArmed=$false
$refused=$false;try{Start-OwnedChild 'Tunnel' 'mock.exe' @('mock')}catch{$refused=$_.Exception.Message-like '*before lifetime job*'}
Assert ($refused -and $spawns-eq 1) 'working child cannot start before ownership is armed'
$source=Get-Content -LiteralPath (Join-Path $out 'supervisor.ps1') -Raw
$start=$source.IndexOf('# Ownership and the death watcher exist before any model or tunnel.')
$body=$source.Substring($start)
Assert ($body.IndexOf('Save-State')-lt $body.IndexOf("Start-OwnedChild 'Watchdog'")) 'structural owner publication precedes watcher'
Assert ($body.IndexOf("Start-OwnedChild 'Watchdog'")-lt $body.IndexOf('systemctl --user start')) 'structural watcher precedes model startup'
Assert ($body.IndexOf('$watchReady.WaitOne(10000)')-lt $body.IndexOf('[Qwen36LifetimeJob]::ArmForCurrentProcess()')) 'watcher handshake precedes self job membership'
Assert ($body.IndexOf('[Qwen36LifetimeJob]::ArmForCurrentProcess()')-lt $body.IndexOf('Start-Keepalive')) 'lifetime job precedes all working children'
Assert ($body.IndexOf('$modelStartRequested=$true;Save-State')-lt $body.IndexOf('systemctl --user start')) 'model intent is published before WSL start'
$loop=$body.Substring($body.IndexOf('while(-not(Stop-Requested)){'))
Assert (([regex]::Matches($source,'Save-Heartbeat')).Count-eq 2 -and $loop.IndexOf('Save-Heartbeat')-ge 0 -and $loop.IndexOf('Save-Heartbeat')-lt $loop.IndexOf('Start-Sleep -Seconds 5')) 'only the periodic loop heartbeat is failure tolerant'
$ownerAssignment=$ast.Find({param($n)$n-is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text-eq '$ownerArgs'},$true)
$Root=$scratch
. ([ScriptBlock]::Create($ownerAssignment.Extent.Text))
$hostAst=[Management.Automation.Language.Parser]::ParseFile((Join-Path $out 'host-metrics.ps1'),[ref]$tokens,[ref]$errors)
$hostParameters=@($hostAst.ParamBlock.Parameters|ForEach-Object{$_.Name.VariablePath.UserPath})
$sharedParameterNames=@($ownerArgs|Where-Object{$_-match '^-[A-Za-z]+$'}|ForEach-Object{$_.Substring(1)})
Assert (@($sharedParameterNames|Where-Object{$_-notin $hostParameters}).Count-eq 0) 'shared child arguments bind to host-metrics declared parameters'
Assert ($ownerArgs-notcontains '-ReadyEvent' -and $body-match "\+\`$ownerArgs\+@\('-ReadyEvent',\`$readyName\)") 'ready event is passed exclusively to watchdog'
Assert (-not($source-match 'Register-ScheduledTask|CurrentVersion\\Run|systemctl.*enable')) 'no autostart installer in supervisor'
$probe=[Threading.Mutex]::new($false,('Global\Qwen36ApiCpuTest-'+[Guid]::NewGuid().ToString('N')))
$probeHeld=$false
try{$probeHeld=$probe.WaitOne(0);Assert $probeHeld 'Global mutex creation permitted for current account'}finally{if($probeHeld){$probe.ReleaseMutex()};$probe.Dispose()}
$releaseAst=[Management.Automation.Language.Parser]::ParseFile((Join-Path $out 'optimization\release-management.ps1'),[ref]$tokens,[ref]$errors)
foreach($fn in $releaseAst.FindAll({param($node)$node-is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
    if($fn.Name-in @('Hash','Target','Validate-Rollback')){. ([ScriptBlock]::Create($fn.Extent.Text))}
}
$root=Join-Path $scratch 'release'
$allowed=@('manage.ps1','supervisor.ps1','watch-supervisor.ps1','host-metrics.ps1','bin\qwen36.cmd')
$fixtureManifestPath=Join-Path $root 'backups\fixture\manifest.json'
$rows=@()
foreach($name in $allowed){
    $dest=Join-Path $root $name;$saved=Join-Path (Split-Path $fixtureManifestPath) ('files\'+$name)
    New-Item -ItemType Directory -Force -Path (Split-Path $dest),(Split-Path $saved)|Out-Null
    [IO.File]::WriteAllText($dest,'after');[IO.File]::WriteAllText($saved,'before')
    $rows+=@{Path=$name;Before=(Hash $saved);After=(Hash $dest)}
}
$manifest=@{Files=$rows}
Validate-Rollback $manifest $fixtureManifestPath
Assert $true 'complete rollback manifest preflight passes'
$rejected=$false;try{Target '..\unrelated.ps1'}catch{$rejected=$true}
Assert $rejected 'rollback rejects unowned path'
[IO.File]::WriteAllText((Join-Path $root 'manage.ps1'),'later foreign edit')
$rejected=$false;try{Validate-Rollback $manifest $fixtureManifestPath}catch{$rejected=$_.Exception.Message-like '*Later edit*'}
Assert $rejected 'rollback preserves later foreign edit'
[IO.File]::WriteAllText((Join-Path $root 'manage.ps1'),'after')
[IO.File]::WriteAllText((Join-Path (Split-Path $fixtureManifestPath) 'files\bin\qwen36.cmd'),'damaged backup')
$rejected=$false;try{Validate-Rollback $manifest $fixtureManifestPath}catch{$rejected=$_.Exception.Message-like '*Backup hash mismatch*'}
Assert $rejected 'all backup hashes validated before rollback mutations'
$oldStatePath=$statePath
$statePath='C:\path\to\LLM\space dir\run\supervisor.json'
Assert ((Wsl-StatePath)-eq '/path/to/LLM/space dir/run/supervisor.json') 'drive-backed state converts to WSL without shell interpolation'
$statePath='\\server\share\supervisor.json'
$rejected=$false;try{Wsl-StatePath}catch{$rejected=$true}
Assert $rejected 'unsupported UNC state refuses keepalive launch'
$statePath=$oldStatePath
function Start-OwnedChild([string]$Name,[string]$File,[string[]]$Arguments,[string]$ErrorLog=''){
    $script:keepaliveCaptured=@{Name=$Name;File=$File;Arguments=$Arguments;Published=(Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json)}
}
Start-Keepalive
$firstLease=$keepaliveLeaseId
Assert ($firstLease-match '^[a-f0-9]{32}$' -and $keepaliveCaptured.Published.KeepaliveLeaseId-eq $firstLease) 'keepalive lease is published before child spawn'
Assert ($keepaliveCaptured.Arguments-contains '/home/USER/qwen36-api/keepalive.py' -and $keepaliveCaptured.Arguments-contains '--lease-id' -and $keepaliveCaptured.Arguments-notcontains 'infinity') 'keepalive launches bounded helper with explicit lease identity'
Start-Keepalive
Assert ($keepaliveLeaseId-ne $firstLease -and $keepaliveCaptured.Published.KeepaliveLeaseId-eq $keepaliveLeaseId) 'replacement child invalidates previous helper lease'
$report=@{Result='PASS';PowerShell=$PSVersionTable.PSVersion.ToString();Tests=$results;Mocked=$true;StateStress=$stressFacts;
    RealProcessKill='NOT_RUN';RealServiceLifecycle='NOT_RUN';RealNetwork='NOT_RUN'}
$dest=Join-Path $out 'optimization\ops-control-tests.json'
[IO.File]::WriteAllText($dest,($report|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$report|ConvertTo-Json -Depth 6
