[CmdletBinding()]
param()
# CPU tests for ops-live-snapshot.ps1. Only function definitions are loaded; the
# snapshot's top-level (WSL, nvidia-smi, D:\ state) is never executed. One check
# runs an own synthetic writer process against an own temp file under work/.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$path=Join-Path $PSScriptRoot 'ops-live-snapshot.ps1'
$tokens=$null;$errors=$null
$tree=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
if($errors){throw 'Snapshot parser errors'}
foreach($definition in $tree.FindAll({param($n) $n-is [Management.Automation.Language.FunctionDefinitionAst]},$false)){
    . ([scriptblock]::Create($definition.Extent.Text))
}
$Units=@('qwen36-inference.service','qwen36-experiment.service')
$checks=[Collections.Generic.List[string]]::new()
function Check([string]$Name,[bool]$Condition){if(-not $Condition){throw $Name};$checks.Add($Name)}
function Must-Throw([string]$Name,[scriptblock]$Body){$t=$false;try{& $Body|Out-Null}catch{$t=$true};Check $Name $t}

$show=@('Id=qwen36-inference.service','LoadState=loaded','ActiveState=inactive','SubState=dead','MainPID=0','NRestarts=0','InvocationID=','',
        'Id=qwen36-experiment.service','LoadState=not-found','ActiveState=inactive','SubState=dead','MainPID=0','NRestarts=0','InvocationID=')
$units=ConvertFrom-UnitShow $show
Check 'unit_show_parses_both_blocks' ($units['qwen36-inference.service']['ActiveState']-eq 'inactive' -and $units['qwen36-experiment.service']['LoadState']-eq 'not-found')
Must-Throw 'unit_show_duplicate_property_refused' {ConvertFrom-UnitShow (@('Id=qwen36-inference.service','ActiveState=active','ActiveState=inactive')+$show)}
Must-Throw 'unit_show_missing_unit_refused' {ConvertFrom-UnitShow $show[0..6]}
Must-Throw 'unit_show_malformed_line_refused' {ConvertFrom-UnitShow (@('garbage')+$show)}
Must-Throw 'unit_show_incomplete_unit_refused' {ConvertFrom-UnitShow @('Id=qwen36-inference.service','ActiveState=inactive','','Id=qwen36-experiment.service','LoadState=x','ActiveState=y','SubState=z','MainPID=0')}

$gpu=@(ConvertFrom-GpuCsv @('0, NVIDIA GeForce RTX 5070 Ti, 3255, 16303'))
Check 'gpu_csv_parsed' ($gpu.Count-eq 1 -and $gpu[0].UsedMiB-eq 3255 -and $gpu[0].TotalMiB-eq 16303)
Must-Throw 'gpu_csv_malformed_refused' {ConvertFrom-GpuCsv @('0, name, [N/A]')}
Must-Throw 'gpu_csv_empty_refused' {ConvertFrom-GpuCsv @('')}

$validLine=('a'*64)+'  /home/USER/qwen36-api/readiness.py'
$hashes=ConvertFrom-HashLines @($validLine,'sha256sum: /x: No such file or directory','not a hash line')
Check 'hash_lines_only_valid_rows' ($hashes.Count-eq 1 -and $hashes['/home/USER/qwen36-api/readiness.py']-eq ('a'*64))

function Get-Process {param([int]$Id,[string]$ErrorAction) if($Id-eq 4242){return [pscustomobject]@{StartTime=[DateTime]::Parse('2026-10-13T10:00:00Z').ToUniversalTime()}};return $null}
$alive=[DateTime]::Parse('2026-10-13T10:00:00Z').ToUniversalTime().ToString('o')
Check 'absent_state_is_not_present' (-not (ConvertFrom-OwnerState $null).Present)
Must-Throw 'empty_state_is_not_absent' {ConvertFrom-OwnerState ''}
$legacy=ConvertFrom-OwnerState ('{"SupervisorPid":4242,"SupervisorStarted":"'+$alive+'","TunnelPid":7,"TunnelStarted":"x"}')
Check 'legacy_state_without_new_fields_is_read_under_strict_mode' ($legacy.Present -and $null-eq $legacy.LifetimeJobArmed -and $legacy.Processes.Supervisor.Alive -and -not $legacy.Processes.Tunnel.Alive)
$reused=ConvertFrom-OwnerState ('{"SupervisorPid":4242,"SupervisorStarted":"2000-01-01T00:00:00.0000000Z","Generation":"g"}')
Check 'reused_pid_is_not_alive_identity' (-not $reused.Processes.Supervisor.Alive)

function New-Snap([bool]$Owner,[string]$Active,[string]$MainPid,[int]$Used,[string]$Experiment='inactive'){
    return [ordered]@{Owner=[ordered]@{Present=$Owner};
        Units=[ordered]@{'qwen36-inference.service'=[ordered]@{ActiveState=$Active;MainPID=$MainPid};'qwen36-experiment.service'=[ordered]@{ActiveState=$Experiment;MainPID='0'}};
        Gpu=@([ordered]@{UsedMiB=$Used})}
}
$baseline=New-Snap $false 'inactive' '0' 3000
$running=New-Snap $true 'active' '777' 15500
Check 'stopped_release_pass' ((Compare-StoppedRelease (New-Snap $false 'inactive' '0' 3400) $baseline $running 1024 6000).Pass)
Check 'gpu_above_tolerance_fails' (-not (Compare-StoppedRelease (New-Snap $false 'inactive' '0' 4100) $baseline $null 1024 6000).Pass)
Check 'owner_state_present_fails' (-not (Compare-StoppedRelease (New-Snap $true 'inactive' '0' 3000) $baseline $null 1024 6000).Pass)
Check 'active_or_deactivating_unit_fails' (-not (Compare-StoppedRelease (New-Snap $false 'deactivating' '55' 3000) $baseline $null 1024 6000).Pass)
Check 'experiment_active_fails' (-not (Compare-StoppedRelease (New-Snap $false 'inactive' '0' 3000 'active') $baseline $null 1024 6000).Pass)
Check 'insufficient_release_from_running_fails' (-not (Compare-StoppedRelease (New-Snap $false 'inactive' '0' 3900) $baseline (New-Snap $true 'active' '1' 6000) 1024 6000).Pass)
$saved=ConvertFrom-SavedSnapshot (([ordered]@{Owner=[ordered]@{Present=$false};Units=$baseline.Units;Gpu=@([ordered]@{Index=0;Name='x';UsedMiB=3000;TotalMiB=16303})})|ConvertTo-Json -Depth 6)
Check 'saved_snapshot_roundtrip_compares' ((Compare-StoppedRelease (New-Snap $false 'inactive' '0' 3100) $saved $null 1024 6000).Pass)
Must-Throw 'saved_snapshot_without_gpu_refused' {ConvertFrom-SavedSnapshot '{"Owner":{"Present":false},"Units":{}}'}

# RP-1: all 11 MANAGED_FILES and the release/ tree against the installed receipt.
foreach($name in @('NativeFiles','NativeRoot')){
    $assignment=$tree.Find({param($n)$n-is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text-eq ('$'+$name)},$false)
    . ([scriptblock]::Create($assignment.Extent.Text))
}
$runtimeSource=Get-Content -LiteralPath (Join-Path (Split-Path $PSScriptRoot -Parent) 'profile-runtime.py') -Raw
$managedBlock=[regex]::Match($runtimeSource,'MANAGED_FILES = frozenset\(\{([^}]*)\}\)').Groups[1].Value
$managedNames=@([regex]::Matches($managedBlock,"'([^']+)'")|ForEach-Object {$_.Groups[1].Value}|Sort-Object)
Check 'native_files_equal_release_managed_files' ($NativeFiles.Count-eq 11 -and $managedNames.Count-eq 11 -and ((@($NativeFiles|Sort-Object) -join '|')-ceq ($managedNames -join '|')))
function New-Hash([string]$Seed){return ([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($Seed)))).Replace('-','').ToLower()}
function New-ReleaseFixture{
    $f=@{Sources=[ordered]@{};Rows=[ordered]@{};Lines=[Collections.Generic.List[string]]::new()}
    foreach($n in $NativeFiles){$h=New-Hash $n;$f.Sources[$n]=$h;$f.Rows[$n]=[ordered]@{Native=$h;Staged=$h;Match=$true}}
    foreach($n in @('release/python/freetoken/core.py','release/lib/libqwen_attention_gate.so')){$h=New-Hash $n;$f.Sources[$n]=$h;$f.Lines.Add($h+'  '+$NativeRoot+'/'+$n)}
    $f.Receipt=(@{schema=1;source_files=$f.Sources}|ConvertTo-Json -Depth 4)
    $f.QSha=New-Hash $f.Receipt
    $f.Lines.Add($f.QSha+'  '+$NativeRoot+'/release/qualification.json')
    return $f
}
function Tree($f,[string[]]$NonRegular=@(),$Receipt='use-fixture',$Staged='use-fixture'){
    if($Receipt-is [string] -and $Receipt-ceq 'use-fixture'){$Receipt=$f.Receipt}
    if($Staged-is [string] -and $Staged-ceq 'use-fixture'){$Staged=$f.QSha}
    return Compare-ReleaseTree @($f.Lines) $NonRegular $Receipt $f.Rows $Staged
}
$f=New-ReleaseFixture;$t=Tree $f
Check 'release_tree_matching_receipt_passes' ($t.Match -and $t.ReceiptFiles-eq 13 -and $t.ReleaseFilesNative-eq 2 -and $t.ManagedSetMatches -and $t.QualificationStagedMatch)
$f=New-ReleaseFixture;$f.Lines[0]=(New-Hash 'tampered')+'  '+$NativeRoot+'/release/python/freetoken/core.py';$t=Tree $f
Check 'release_runtime_byte_change_fails' (-not $t.Match -and $t.Mismatched-contains 'release/python/freetoken/core.py')
$f=New-ReleaseFixture;$f.Lines.RemoveAt(1);$t=Tree $f
Check 'release_library_missing_fails' (-not $t.Match -and $t.Missing-contains 'release/lib/libqwen_attention_gate.so')
$f=New-ReleaseFixture;$f.Lines.Add((New-Hash 'x')+'  '+$NativeRoot+'/release/lib/libstale.so');$t=Tree $f
Check 'release_extra_unreceipted_file_fails' (-not $t.Match -and $t.Extra-contains 'release/lib/libstale.so')
$f=New-ReleaseFixture;$f.Lines[0]=$f.Sources['release/python/freetoken/core.py']+'  '+$NativeRoot+'/release/python/freetoken/Core.py';$t=Tree $f
Check 'release_paths_are_case_sensitive' (-not $t.Match -and $t.Missing-contains 'release/python/freetoken/core.py' -and $t.Extra-contains 'release/python/freetoken/Core.py')
$f=New-ReleaseFixture;$f.Rows['guard.py']['Native']=New-Hash 'other';$t=Tree $f
Check 'managed_file_differs_from_receipt_fails' (-not $t.Match -and $t.Mismatched-contains 'guard.py')
$f=New-ReleaseFixture;$f.Rows.Remove('profiles.json');$t=Tree $f
Check 'managed_file_without_native_hash_fails' (-not $t.Match -and $t.Missing-contains 'profiles.json')
$f=New-ReleaseFixture;$f.Sources.Remove('metrics_summary.py');$f.Receipt=(@{source_files=$f.Sources}|ConvertTo-Json -Depth 4);$t=Tree $f -Staged $null
Check 'receipt_without_all_managed_files_fails' (-not $t.Match -and -not $t.ManagedSetMatches)
$f=New-ReleaseFixture;$t=Tree $f @('/home/USER/qwen36-api/release/python/freetoken/link.py')
Check 'release_symlink_or_special_entry_fails' (-not $t.Match -and $t.NonRegular-eq 1)
$f=New-ReleaseFixture;$t=Tree $f -Staged (New-Hash 'other receipt')
Check 'installed_receipt_differs_from_staged_fails' (-not $t.Match -and -not $t.QualificationStagedMatch)
$f=New-ReleaseFixture;$t=Tree $f -Receipt $null
Check 'absent_installed_receipt_fails' (-not $t.Match -and $t.Reason-eq 'QUALIFICATION_ABSENT')
$f=New-ReleaseFixture;$t=Tree $f -Receipt '{"source_files":{"guard.py":"NOT-A-HASH"}}'
Check 'invalid_receipt_hash_fails' (-not $t.Match -and $t.Reason-eq 'RECEIPT_HASH_INVALID')
$t=Tree (New-ReleaseFixture) -Receipt '{"schema":1}'
Check 'receipt_without_source_files_fails' (-not $t.Match -and $t.Reason-eq 'RECEIPT_WITHOUT_SOURCE_FILES')

$log=@('2026-10-13T10:00:00 Keepalive spawned PID=1','2026-10-13T10:00:01 Tunnel spawned PID=2','2026-10-13T10:00:02 Keepalive spawned PID=3',
       '2026-10-13T10:00:03 Supervisor stopped: Exception calling "Replace" with "3" argument(s): PRIVATE_DETAIL_MARKER') -join "`r`n"
$summary=Get-LogSummary $log 600
Check 'log_summary_classifies_without_raw_text' ($summary.KeepaliveSpawned-eq 2 -and $summary.ReplaceFailureSeen -and ($summary|ConvertTo-Json) -notmatch 'PRIVATE_DETAIL_MARKER')
Check 'log_summary_absent_log' (-not (Get-LogSummary $null 600).Present)

# Shared reader vs. the supervisor's exact WriteAllText+File.Replace pattern in a
# separate own process. Neither side may fail (sharing violation regression).
$workRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\work'))
$fixture=Join-Path $workRoot ('ops-live-snapshot-fixture-'+[Guid]::NewGuid().ToString('N'))
if(-not $fixture.StartsWith($workRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Fixture escaped workspace'}
New-Item -ItemType Directory -Path $fixture|Out-Null
try{
    $state=Join-Path $fixture 'supervisor.json'
    [IO.File]::WriteAllText($state,'{"HeartbeatSequence":0}')
    $writer=Join-Path $fixture 'writer.ps1'
    [IO.File]::WriteAllText($writer,@'
param([string]$State,[int]$Seconds)
$ErrorActionPreference='Stop';$fail=0;$n=0L;$clock=[Diagnostics.Stopwatch]::StartNew()
while($clock.Elapsed.TotalSeconds-lt $Seconds){
  $n++;$temp=$State+'.gen.tmp'
  try{[IO.File]::WriteAllText($temp,(@{HeartbeatSequence=$n;Pad=('x'*300)}|ConvertTo-Json),[Text.UTF8Encoding]::new($false));[IO.File]::Replace($temp,$State,[NullString]::Value)}catch{$fail++}
  Start-Sleep -Milliseconds 20
}
"$n $fail"
'@,[Text.UTF8Encoding]::new($false))
    $out=Join-Path $fixture 'writer.out'
    $proc=Start-Process -FilePath powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$writer,'-State',$state,'-Seconds','6') -NoNewWindow -PassThru -RedirectStandardOutput $out
    $readFailures=0;$reads=0;$clock=[Diagnostics.Stopwatch]::StartNew()
    while($clock.Elapsed.TotalSeconds-lt 5.5){
        try{$null=((Read-SharedText $state)|ConvertFrom-Json).HeartbeatSequence;$reads++}catch{$readFailures++}
        Start-Sleep -Milliseconds 5
    }
    if(-not $proc.WaitForExit(30000)){throw 'Own synthetic writer did not exit'}
    $parts=((Get-Content -LiteralPath $out -Raw).Trim() -split ' ')
    Check 'shared_reader_never_breaks_supervisor_replace' ([int]$parts[0]-gt 50 -and [int]$parts[1]-eq 0 -and $readFailures-eq 0 -and $reads-gt 100)
}finally{
    Remove-Item -LiteralPath $fixture -Recurse -Force -ErrorAction SilentlyContinue
}
[pscustomobject]@{Result='PASS';Count=$checks.Count;Checks=$checks.ToArray();Live='NOT_RUN';ServiceActions=0}|ConvertTo-Json -Depth 3
