[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_.-]{1,48}$')][string]$Label,
    [string]$OutDir='',
    [string]$CompareStoppedBaseline='',
    [string]$RunningSnapshot='',
    [ValidateRange(64,8192)][int]$ToleranceMiB=1024,
    [ValidateRange(1000,16000)][int]$MinReleasedMiB=6000,
    [ValidateRange(50,5000)][int]$SupervisorLogTail=600
)
# READ-ONLY operational snapshot for the qwen36 live acceptance runbook.
# It never starts/stops units or processes, never sends signals, never reads
# /home/USER/qwen36-api/secrets and never prints request/response content.
# supervisor.json and supervisor.log are opened with FileShare ReadWrite|Delete
# so this reader cannot make the supervisor's File.Replace heartbeat fail.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$Root='C:\path\to\LLM\qwen36-api'
$statePath=Join-Path $Root 'run\supervisor.json'
$logPath=Join-Path $Root 'logs\supervisor.log'
$sourceRoot=Split-Path $PSScriptRoot -Parent
$Units=@('qwen36-inference.service','qwen36-experiment.service')
$ControlFiles=[ordered]@{'manage.ps1'=(Join-Path $sourceRoot 'manage.ps1');'supervisor.ps1'=(Join-Path $sourceRoot 'supervisor.ps1');
    'watch-supervisor.ps1'=(Join-Path $sourceRoot 'watch-supervisor.ps1');'host-metrics.ps1'=(Join-Path $sourceRoot 'host-metrics.ps1');
    'bin\qwen36.cmd'=(Join-Path $PSScriptRoot 'qwen36.cmd')}
# All 11 release MANAGED_FILES (profile-runtime.py / release.py FILES).
$NativeFiles=@('readiness.py','keepalive.py','profile-runtime.py','guard.py','ingress.py','serve_guarded.py','start-inference.sh','request_metrics.py','placement.py','metrics_summary.py','profiles.json')
$NativeRoot='/home/USER/qwen36-api'

function Read-SharedText([string]$Path,[int]$Attempts=5){
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
function Same-Process([int]$ProcessId,[string]$Started){
    if($ProcessId-le 0 -or -not $Started){return $false}
    try{
        $p=Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        return [bool]($p -and $p.StartTime.ToUniversalTime().ToString('o')-eq $Started)
    }catch{return $false}
}
function ConvertFrom-OwnerState($Raw){
    # Untyped on purpose: a [string] parameter would turn $null (absent file) into ''.
    if($null-eq $Raw){return [ordered]@{Present=$false}}
    $s=$Raw|ConvertFrom-Json
    if($null-eq $s){throw 'Owner state is empty or invalid'}
    $owner=[ordered]@{Present=$true;Generation=[string](Get-Prop $s 'Generation');Phase=[string](Get-Prop $s 'Phase');
        LifetimeJobArmed=(Get-Prop $s 'LifetimeJobArmed');ModelStartRequested=(Get-Prop $s 'ModelStartRequested');
        HeartbeatSequence=(Get-Prop $s 'HeartbeatSequence');KeepaliveLeaseId=[string](Get-Prop $s 'KeepaliveLeaseId');
        Updated=[string](Get-Prop $s 'Updated');Processes=[ordered]@{}}
    foreach($name in @('Supervisor','Tunnel','Keepalive','Watchdog','HostMetrics')){
        $id=Get-Prop $s ($name+'Pid');$started=Get-Prop $s ($name+'Started')
        $id=if($null-ne $id){[int]$id}else{0};$started=if($null-ne $started){[string]$started}else{''}
        $owner.Processes[$name]=[ordered]@{Pid=$id;Started=$started;Alive=(Same-Process $id $started)}
    }
    return $owner
}
function ConvertFrom-UnitShow([string[]]$Lines){
    $result=[ordered]@{};$current=$null
    foreach($line in @($Lines)+@('')){
        $text=([string]$line).TrimEnd("`r")
        if($text-eq ''){
            if($current){
                if(-not $current.Contains('Id')){throw 'Unit block without Id'}
                if($result.Contains($current['Id'])){throw 'Duplicate unit block'}
                $result[$current['Id']]=$current;$current=$null
            }
            continue
        }
        $pair=$text -split '=',2
        if($pair.Count-ne 2){throw 'Malformed systemctl show line'}
        if(-not $current){$current=[ordered]@{}}
        if($current.Contains($pair[0])){throw 'Duplicate systemctl property'}
        $current[$pair[0]]=$pair[1]
    }
    foreach($unit in $Units){
        if(-not $result.Contains($unit)){throw ('Missing unit block: '+$unit)}
        foreach($k in 'LoadState','ActiveState','SubState','MainPID'){if(-not $result[$unit].Contains($k)){throw 'Incomplete unit state'}}
    }
    return $result
}
function Invoke-Native([string]$File,[string[]]$Arguments){
    # Windows PowerShell 5.1 turns redirected native stderr into a terminating
    # error under ErrorActionPreference=Stop; use the exit code instead.
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    try{$lines=@(& $File @Arguments 2>$null);$code=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
    return [pscustomobject]@{Lines=@($lines|ForEach-Object {[string]$_});Code=$code}
}
function Invoke-UnitShow {
    $r=Invoke-Native 'wsl.exe' (@('-d','archlinux','--','systemctl','--user','show')+$Units+@('--property=Id','--property=LoadState','--property=ActiveState','--property=SubState','--property=MainPID','--property=NRestarts','--property=InvocationID'))
    if($r.Code-ne 0){throw 'systemctl show failed'}
    return ConvertFrom-UnitShow $r.Lines
}
function ConvertFrom-GpuCsv([string[]]$Lines){
    $rows=@()
    foreach($line in $Lines){
        if(-not $line -or -not $line.Trim()){continue}
        $f=@($line -split ',' | ForEach-Object {$_.Trim()})
        if($f.Count-ne 4){throw 'Unexpected nvidia-smi row'}
        $rows+=[ordered]@{Index=[int]$f[0];Name=$f[1];UsedMiB=[int]$f[2];TotalMiB=[int]$f[3]}
    }
    if($rows.Count-lt 1){throw 'No GPU rows'}
    return $rows
}
function Invoke-Gpu {
    $r=Invoke-Native 'nvidia-smi.exe' @('--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits')
    if($r.Code-ne 0){throw 'nvidia-smi failed'}
    return @(ConvertFrom-GpuCsv $r.Lines)
}
function Get-ControlHashes {
    $rows=[ordered]@{}
    foreach($name in $ControlFiles.Keys){
        $installed=Join-Path $Root $name;$staged=$ControlFiles[$name]
        $i=if(Test-Path -LiteralPath $installed -PathType Leaf){(Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash.ToLower()}else{$null}
        $s=if(Test-Path -LiteralPath $staged -PathType Leaf){(Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash.ToLower()}else{$null}
        $rows[$name]=[ordered]@{Installed=$i;Staged=$s;Match=($null-ne $i -and $i-eq $s)}
    }
    return $rows
}
function ConvertFrom-HashLines([string[]]$Lines){
    # Case-sensitive keys: Linux paths that differ only by case stay distinct.
    $map=[Collections.Hashtable]::new([StringComparer]::Ordinal)
    foreach($line in $Lines){if($line -match '^([0-9a-f]{64})\s+\*?(\S+)$'){$map[$Matches[2]]=$Matches[1]}}
    return $map
}
function Get-NativeHashes {
    # sha256sum is executed directly (no shell, no nested quoting). Missing
    # files only produce stderr; they are reported as null, never as a match.
    $drive=$sourceRoot.Substring(0,1).ToLowerInvariant()
    $wslSource='/mnt/'+$drive+'/'+$sourceRoot.Substring(3).Replace('\','/')
    $paths=@()
    foreach($f in $NativeFiles){$paths+='/home/USER/qwen36-api/'+$f;$paths+=$wslSource+'/'+$f}
    $r=Invoke-Native 'wsl.exe' (@('-d','archlinux','--','sha256sum')+$paths)
    $map=ConvertFrom-HashLines $r.Lines
    $rows=[ordered]@{}
    foreach($f in $NativeFiles){
        $n=$map['/home/USER/qwen36-api/'+$f];$s=$map[$wslSource+'/'+$f]
        $rows[$f]=[ordered]@{Native=$n;Staged=$s;Match=($null-ne $n -and $n-eq $s)}
    }
    return $rows
}
function Compare-ReleaseTree([string[]]$HashLines,[string[]]$NonRegularLines,$QualificationRaw,$NativeRows,$StagedQualificationSha){
    # Installed release == its own receipt: every release/qualification.json
    # source_files entry (11 managed files, release/python/**, release/lib/*.so)
    # has the receipt sha256, release/ holds no extra or non-regular entry
    # (__pycache__/*.pyc excluded, as profile-runtime.py), and the installed
    # receipt equals the staged one. Relative names only (<=20 per list).
    $releasePrefix=$NativeRoot+'/release/'
    $qualificationPath=$releasePrefix+'qualification.json'
    $native=ConvertFrom-HashLines $HashLines
    $nativeQualification=$native[$qualificationPath]
    $r=[ordered]@{Match=$false;Reason=$null;QualificationPresent=$false;QualificationNativeSha256=$nativeQualification;
        QualificationStagedSha256=$StagedQualificationSha;
        QualificationStagedMatch=($null-ne $nativeQualification -and $nativeQualification-eq $StagedQualificationSha);
        ReceiptFiles=0;ReleaseFilesNative=0;ManagedSetMatches=$false;NonRegular=@($NonRegularLines|Where-Object {$_}).Count;
        Mismatched=@();Missing=@();Extra=@()}
    if($null-eq $QualificationRaw){$r.Reason='QUALIFICATION_ABSENT';return $r}
    $sources=Get-Prop ($QualificationRaw|ConvertFrom-Json) 'source_files'
    if($null-eq $sources -or $sources-isnot [Management.Automation.PSCustomObject]){$r.Reason='RECEIPT_WITHOUT_SOURCE_FILES';return $r}
    $expected=[Collections.Hashtable]::new([StringComparer]::Ordinal)
    foreach($p in $sources.PSObject.Properties){
        if($p.Value-isnot [string] -or $p.Value-cnotmatch '^[0-9a-f]{64}$'){$r.Reason='RECEIPT_HASH_INVALID';return $r}
        $expected[$p.Name]=$p.Value
    }
    $r.QualificationPresent=$true;$r.ReceiptFiles=$expected.Count
    $mismatched=[Collections.Generic.List[string]]::new();$missing=[Collections.Generic.List[string]]::new()
    foreach($name in @($expected.Keys|Sort-Object)){
        $actual=$null
        if($name.StartsWith('release/')){$actual=$native[$NativeRoot+'/'+$name]}
        elseif($null-ne $NativeRows -and $NativeRows.Contains($name)){$actual=$NativeRows[$name]['Native']}
        if($null-eq $actual){$missing.Add($name)}elseif($actual-cne $expected[$name]){$mismatched.Add($name)}
    }
    $nativeRelease=@($native.Keys|Where-Object {$_.StartsWith($releasePrefix) -and $_-cne $qualificationPath}|ForEach-Object {$_.Substring($NativeRoot.Length+1)})
    $r.ReleaseFilesNative=$nativeRelease.Count
    $extra=@($nativeRelease|Where-Object {-not $expected.ContainsKey($_)}|Sort-Object)
    $managed=@($expected.Keys|Where-Object {-not $_.StartsWith('release/')}|Sort-Object)
    $r.ManagedSetMatches=(($managed -join '|')-ceq (@($NativeFiles|Sort-Object) -join '|'))
    $r.Mismatched=@($mismatched|Select-Object -First 20);$r.Missing=@($missing|Select-Object -First 20);$r.Extra=@($extra|Select-Object -First 20)
    $r.Match=($mismatched.Count-eq 0 -and $missing.Count-eq 0 -and $extra.Count-eq 0 -and $r.NonRegular-eq 0 -and
        $r.ManagedSetMatches -and $r.QualificationStagedMatch)
    if(-not $r.Match){$r.Reason='RELEASE_TREE_DIFFERS_FROM_RECEIPT'}
    return $r
}
function Get-ReleaseTree($NativeRows){
    # wsl.exe -e: arguments reach find/cat verbatim (no Linux shell, no globbing).
    $release=$NativeRoot+'/release'
    $staged=Join-Path $sourceRoot 'qualification.json'
    $stagedSha=if(Test-Path -LiteralPath $staged -PathType Leaf){(Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash.ToLower()}else{$null}
    $dir=Invoke-Native 'wsl.exe' @('-d','archlinux','-e','test','-d',$release)
    if($dir.Code-ne 0){return [ordered]@{Match=$false;Reason='RELEASE_DIR_ABSENT';QualificationStagedSha256=$stagedSha}}
    $h=Invoke-Native 'wsl.exe' @('-d','archlinux','-e','find',$release,'-name','__pycache__','-prune','-o','-type','f','!','-name','*.pyc','-exec','sha256sum','{}','+')
    if($h.Code-ne 0){throw 'release tree hash failed'}
    $n=Invoke-Native 'wsl.exe' @('-d','archlinux','-e','find',$release,'-mindepth','1','-type','l','-print','-o','-name','__pycache__','-prune','-o','!','-type','f','!','-type','d','-print')
    if($n.Code-ne 0){throw 'release tree scan failed'}
    $q=Invoke-Native 'wsl.exe' @('-d','archlinux','-e','cat',($release+'/qualification.json'))
    $raw=if($q.Code-eq 0){$q.Lines -join "`n"}else{$null}
    return Compare-ReleaseTree $h.Lines $n.Lines $raw $NativeRows $stagedSha
}
function Get-LogSummary($Raw,[int]$Tail){
    if($null-eq $Raw){return [ordered]@{Present=$false}}
    $lines=@($Raw -split "`r?`n" | Where-Object {$_})
    if($lines.Count-gt $Tail){$lines=$lines[($lines.Count-$Tail)..($lines.Count-1)]}
    $count={param($pattern) @($lines|Where-Object {$_ -like $pattern}).Count}
    # Classification only: no raw log text is stored in the receipt.
    return [ordered]@{Present=$true;LinesExamined=$lines.Count;
        KeepaliveSpawned=(& $count '*Keepalive spawned*');TunnelSpawned=(& $count '*Tunnel spawned*');
        HostMetricsSpawned=(& $count '*HostMetrics spawned*');WatchdogSpawned=(& $count '*Watchdog spawned*');
        SupervisorStopped=(& $count '*Supervisor stopped:*');ReplaceFailureSeen=((& $count '*Supervisor stopped:*Replace*')-gt 0);
        RestartLimitSeen=((& $count '*restart limit reached*')-gt 0);WatcherExitSeen=((& $count '*Outside-job watcher exited*')-gt 0);
        StopUnconfirmedSeen=((& $count '*stop unconfirmed*')-gt 0);OwnerDeathCleanup=(& $count '*Owner process died; owned service*')}
}
function Compare-StoppedRelease($Now,$Baseline,$Running,[int]$Tolerance,[int]$MinReleased){
    $checks=[ordered]@{}
    $checks.OwnerStateAbsent=-not $Now.Owner.Present
    $inf=$Now.Units['qwen36-inference.service'];$exp=$Now.Units['qwen36-experiment.service']
    $checks.InferenceStopped=($inf['ActiveState']-in @('inactive','failed')) -and ([string]$inf['MainPID']-eq '0')
    $checks.ExperimentNotActive=$exp['ActiveState']-in @('inactive','failed')
    $nowUsed=[int]$Now.Gpu[0].UsedMiB;$baseUsed=[int]$Baseline.Gpu[0].UsedMiB
    $checks.GpuWithinBaselineTolerance=$nowUsed-le ($baseUsed+$Tolerance)
    $facts=[ordered]@{NowUsedMiB=$nowUsed;BaselineUsedMiB=$baseUsed;ToleranceMiB=$Tolerance}
    if($null-ne $Running){
        $runUsed=[int]$Running.Gpu[0].UsedMiB
        $checks.GpuReleasedFromRunning=($runUsed-$nowUsed)-ge $MinReleased
        $facts.RunningUsedMiB=$runUsed;$facts.ReleasedMiB=$runUsed-$nowUsed;$facts.MinReleasedMiB=$MinReleased
    }
    $pass=@($checks.Values|Where-Object {$_ -ne $true}).Count-eq 0
    return [ordered]@{Pass=$pass;Checks=$checks;Facts=$facts;
        Note='Windows GPU memory includes other applications; per-process WSL usage is not visible. Tolerance is relative to the pre-start baseline.'}
}
function ConvertFrom-SavedSnapshot([string]$Raw){
    # Minimal adapter so Compare-StoppedRelease accepts a saved snapshot file.
    $o=$Raw|ConvertFrom-Json
    $units=[ordered]@{}
    $saved=Get-Prop $o 'Units'
    if($null-ne $saved){foreach($p in $saved.PSObject.Properties){$units[$p.Name]=[ordered]@{ActiveState=(Get-Prop $p.Value 'ActiveState');MainPID=(Get-Prop $p.Value 'MainPID')}}}
    $gpu=@(Get-Prop $o 'Gpu')
    if($gpu.Count-lt 1 -or $null-eq $gpu[0]){throw 'Saved snapshot lacks GPU data'}
    $owner=Get-Prop $o 'Owner'
    return [ordered]@{Owner=[ordered]@{Present=[bool](Get-Prop $owner 'Present')};Units=$units;Gpu=@([ordered]@{UsedMiB=[int](Get-Prop $gpu[0] 'UsedMiB')})}
}

$errors=[ordered]@{}
$snapshot=[ordered]@{Schema=1;Label=$Label;Utc=[DateTime]::UtcNow.ToString('o');ReadOnly=$true;SignalsSent=0;ServiceActions=0;
    ExternalEndToEnd='NOT_RUN'}
try{$snapshot.Owner=ConvertFrom-OwnerState (Read-SharedText $statePath)}catch{$errors.Owner=$_.Exception.GetType().Name}
try{$snapshot.Units=Invoke-UnitShow}catch{$errors.Units=$_.Exception.GetType().Name}
try{$snapshot.NativeHashes=Get-NativeHashes}catch{$errors.NativeHashes=$_.Exception.GetType().Name}
try{
    if(-not $snapshot.Contains('NativeHashes')){throw 'Native managed hashes unavailable'}
    $snapshot.ReleaseTree=Get-ReleaseTree $snapshot.NativeHashes
}catch{$errors.ReleaseTree=$_.Exception.GetType().Name}
try{$snapshot.Gpu=@(Invoke-Gpu)}catch{$errors.Gpu=$_.Exception.GetType().Name}
try{$snapshot.ControlHashes=Get-ControlHashes}catch{$errors.ControlHashes=$_.Exception.GetType().Name}
try{$snapshot.SupervisorLog=Get-LogSummary (Read-SharedText $logPath) $SupervisorLogTail}catch{$errors.SupervisorLog=$_.Exception.GetType().Name}
$snapshot.Errors=$errors
$exit=if($errors.Count){2}else{0}
if($CompareStoppedBaseline){
    try{
        if($errors.Count){throw 'Current snapshot incomplete'}
        $base=ConvertFrom-SavedSnapshot (Read-SharedText $CompareStoppedBaseline)
        $run=if($RunningSnapshot){ConvertFrom-SavedSnapshot (Read-SharedText $RunningSnapshot)}else{$null}
        $verdict=Compare-StoppedRelease $snapshot $base $run $ToleranceMiB $MinReleasedMiB
        $verdict.Baseline=[IO.Path]::GetFileName($CompareStoppedBaseline)
        if($RunningSnapshot){$verdict.Running=[IO.Path]::GetFileName($RunningSnapshot)}
    }catch{$verdict=[ordered]@{Pass=$false;Reason='COMPARISON_INPUT_INVALID';ErrorType=$_.Exception.GetType().Name}}
    $snapshot.StoppedReleaseVerdict=$verdict
    if(-not $verdict.Pass){$exit=1}
}
if(-not $OutDir){$OutDir=Join-Path $PSScriptRoot 'ops-live-receipts'}
New-Item -ItemType Directory -Force -Path $OutDir|Out-Null
$name='ops-live-snapshot-'+$Label+'-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+[Guid]::NewGuid().ToString('N').Substring(0,6)+'.json'
$path=Join-Path $OutDir $name
$stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($snapshot|ConvertTo-Json -Depth 8));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
$ownerPresent=$null;if($snapshot.Contains('Owner')){$ownerPresent=$snapshot.Owner.Present}
$releasePass=$null;if($snapshot.Contains('StoppedReleaseVerdict')){$releasePass=$snapshot.StoppedReleaseVerdict.Pass}
[pscustomobject]@{Snapshot=$path;Exit=$exit;Errors=@($errors.Keys);OwnerPresent=$ownerPresent;StoppedReleasePass=$releasePass}|ConvertTo-Json
exit $exit
