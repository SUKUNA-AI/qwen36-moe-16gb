param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$workspace=Split-Path (Split-Path (Split-Path $PSScriptRoot))
$root=Join-Path $workspace ('work\release-cpu-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root|Out-Null
$allowed=@('manage.ps1','supervisor.ps1','watch-supervisor.ps1','host-metrics.ps1','bin\qwen36.cmd')
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'release-management.ps1'),[ref]$tokens,[ref]$errors)
if($errors){throw 'Release script parser error'}
foreach($fn in $ast.FindAll({param($n)$n-is [Management.Automation.Language.FunctionDefinitionAst]},$true)){
    . ([ScriptBlock]::Create($fn.Extent.Text))
}
$checks=[Collections.Generic.List[object]]::new()
function Assert([bool]$Condition,[string]$Name){if(-not $Condition){throw ('FAILED: '+$Name)};$checks.Add(@{Name=$Name;Pass=$true})}
function Refuses([scriptblock]$Body,[string]$Name){$rejected=$false;try{& $Body}catch{$rejected=$true};Assert $rejected $Name}
$script:stateLines=@('LoadState=loaded','ActiveState=inactive','SubState=dead')
$script:stateExit=0;$script:stateCalls=0;$script:experimentActive=$false
# Explicitly intercept every WSL invocation. No systemctl/WSL/native process runs.
function wsl.exe {
    # PowerShell consumes the standalone -- token when a function shadows exe.
    if(($args -join ' ') -notmatch '^\-d archlinux (?:-- )?systemctl --user show qwen36-(inference|experiment)\.service '){throw 'Unexpected mocked WSL command'}
    $script:stateCalls++
    $global:LASTEXITCODE=$script:stateExit
    if($script:experimentActive -and $args-contains 'qwen36-experiment.service'){
        return @('LoadState=loaded','ActiveState=activating','SubState=start')
    }
    return $script:stateLines
}
foreach($case in @(@('loaded','inactive','dead'),@('loaded','failed','failed'),@('masked','inactive','dead'),@('not-found','inactive','dead'))){
    $script:stateLines=@(('LoadState='+$case[0]),('ActiveState='+$case[1]),('SubState='+$case[2]));$script:stateCalls=0
    Require-Stopped
    Assert ($script:stateCalls-eq 2) ('accept verified stopped both units '+($case-join '/'))
}
foreach($value in @('active','activating','deactivating','reloading','refreshing','maintenance','')){
    $script:stateLines=@('LoadState=loaded',('ActiveState='+$value),'SubState=dead')
    Refuses {Require-Stopped} ('reject state '+$value)
}
foreach($lines in @(@('LoadState=not-found','ActiveState=inactive','SubState=running'),
                   @('LoadState=error','ActiveState=inactive','SubState=dead'),
                   @('LoadState=loaded','ActiveState=inactive'),
                   @('LoadState=loaded','ActiveState=inactive','SubState=dead','ActiveState=active'))){
    $script:stateLines=$lines;Refuses {Require-Stopped} ('reject invalid state '+($lines-join '/'))
}
$script:stateLines=@('LoadState=loaded','ActiveState=inactive','SubState=dead')
foreach($code in @(1,3,127)){$script:stateExit=$code;Refuses {Require-Stopped} ('reject failed systemctl '+$code)}
$script:stateExit=0;$script:experimentActive=$true
Refuses {Require-Stopped} 'reject activating experiment after inactive inference'
$script:experimentActive=$false

$backup=Join-Path $root 'backups\fixture'
$manifestPath=Join-Path $backup 'manifest.json'
$rows=@()
foreach($name in $allowed){
    $dest=Target $name;$saved=Join-Path $backup ('files\'+$name)
    New-Item -ItemType Directory -Force -Path (Split-Path $dest),(Split-Path $saved)|Out-Null
    [IO.File]::WriteAllText($dest,('after-'+$name));[IO.File]::WriteAllText($saved,('before-'+$name))
    $rows+=@{Path=$name;Before=(Hash $saved);After=(Hash $dest)}
}
$manifest=[pscustomobject]@{Files=$rows}
function Target-Snapshot {return (($allowed|ForEach-Object{Hash (Target $_)})-join ',')}
$snapshot=Target-Snapshot
$lastBackup=Join-Path $backup ('files\'+$allowed[-1])
[IO.File]::WriteAllText($lastBackup,'corrupt last backup')
Refuses {Apply-Rollback $manifest $manifestPath} 'later corrupt backup refuses before first copy'
Assert ((Target-Snapshot)-eq $snapshot) 'corrupt backup caused no partial target mutation'
[IO.File]::WriteAllText($lastBackup,('before-'+$allowed[-1]))
$lastTarget=Target $allowed[-1];[IO.File]::WriteAllText($lastTarget,'later owner edit')
$edited=Target-Snapshot
Refuses {Apply-Rollback $manifest $manifestPath} 'later destination edit refuses before first copy'
Assert ((Target-Snapshot)-eq $edited) 'later edit preserved every target'
[IO.File]::WriteAllText($lastTarget,('after-'+$allowed[-1]))
$script:experimentActive=$true
Refuses {Apply-Rollback $manifest $manifestPath} 'active experiment prevents rollback'
Assert ((Target-Snapshot)-eq $snapshot) 'active experiment caused no target mutation'
$script:experimentActive=$false
Apply-Rollback $manifest $manifestPath
Assert (@($rows|Where-Object{(Hash (Target $_.Path))-ne $_.Before}).Count-eq 0) 'successful synthetic rollback restores every byte'
Apply-Rollback $manifest $manifestPath
Assert (@($rows|Where-Object{(Hash (Target $_.Path))-ne $_.Before}).Count-eq 0) 'synthetic rollback is idempotent'
$report=@{Pass=$true;Checks=$checks;Count=$checks.Count;Scope='CPU synthetic files and mocked WSL only';Live='NOT_RUN'}
$report|ConvertTo-Json -Depth 5
