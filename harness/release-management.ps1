param([switch]$CheckOnly,[string]$RollbackManifest='')
$ErrorActionPreference='Stop'
$root='C:\path\to\LLM\qwen36-api'
$output=Split-Path $PSScriptRoot
$allowed=@('manage.ps1','supervisor.ps1','watch-supervisor.ps1','host-metrics.ps1','bin\qwen36.cmd')
$releaseLocks=@()
function Hash([string]$Path){if(Test-Path -LiteralPath $Path -PathType Leaf){return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash};return $null}
function Target([string]$Relative){
    if($allowed-notcontains $Relative){throw 'Path is not an owned management release file'}
    $resolved=[IO.Path]::GetFullPath((Join-Path $root $Relative))
    if(-not $resolved.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Target escapes service root'}
    return $resolved
}
function Lock-Release {
    foreach($name in @('Global\Qwen36ApiControl','Global\Qwen36ApiSupervisor')){
        $m=[Threading.Mutex]::new($false,$name);$held=$false
        try {$held=$m.WaitOne(0)} catch [Threading.AbandonedMutexException] {$held=$true}
        if(-not $held){$m.Dispose();throw 'A qwen36 control operation or supervisor is active'}
        $script:releaseLocks+=,$m
    }
}
function Unlock-Release {
    for($i=$releaseLocks.Count-1;$i-ge 0;$i--){$releaseLocks[$i].ReleaseMutex();$releaseLocks[$i].Dispose()}
}
function Require-Stopped {
    foreach($unit in @('qwen36-inference.service','qwen36-experiment.service')){
        $lines=& wsl.exe -d archlinux -- systemctl --user show $unit --no-pager --property=LoadState --property=ActiveState --property=SubState
        if($LASTEXITCODE-ne 0){throw ('Cannot verify '+$unit+'; management files unchanged')}
        $state=@{}
        foreach($line in $lines){
            $pair=([string]$line) -split '=',2
            if($pair.Count-ne 2 -or $state.ContainsKey($pair[0])){throw 'Invalid systemctl state response'}
            $state[$pair[0]]=$pair[1]
        }
        if($state.Count-ne 3 -or -not $state.ContainsKey('LoadState') -or -not $state.ContainsKey('ActiveState') -or -not $state.ContainsKey('SubState')){throw 'Incomplete systemctl state response'}
        $missing=$state.LoadState-eq 'not-found' -and $state.ActiveState-eq 'inactive' -and $state.SubState-eq 'dead'
        $loaded=$state.LoadState-in @('loaded','masked') -and $state.ActiveState-in @('inactive','failed')
        if(-not($missing -or $loaded)){throw ('Stop '+$unit+' before replacing management files')}
    }
    if(Test-Path -LiteralPath (Join-Path $root 'run\supervisor.json')){throw 'Supervisor state remains; run qwen36 stop first'}
}
function Validate-Rollback($Manifest,[string]$ManifestPath){
    $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach($row in $Manifest.Files){
        $dest=Target ([string]$row.Path)
        if(-not $seen.Add([string]$row.Path)){throw 'Duplicate rollback entry'}
        if($row.After-notmatch '^[0-9a-fA-F]{64}$' -or ($row.Before -and $row.Before-notmatch '^[0-9a-fA-F]{64}$')){throw 'Invalid rollback hashes'}
        $now=Hash $dest
        if($now-ne $row.Before -and $now-ne $row.After){throw ('Later edit detected; rollback refused: '+$row.Path)}
        if($row.Before){
            $saved=Join-Path (Split-Path $ManifestPath) ('files\'+$row.Path)
            if((Hash $saved)-ne $row.Before){throw ('Backup hash mismatch: '+$row.Path)}
        }
    }
    if($seen.Count-ne $allowed.Count){throw 'Incomplete rollback manifest'}
}
function Apply-Rollback($Manifest,[string]$ManifestPath){
    # Caller holds both release mutexes. Verify every target and backup before
    # changing even the first file; do not discover corrupt later backups midway.
    Require-Stopped
    Validate-Rollback $Manifest $ManifestPath
    foreach($row in $Manifest.Files){
        $dest=Target $row.Path
        if((Hash $dest)-eq $row.Before){continue}
        if($row.Before){Copy-Item -LiteralPath (Join-Path (Split-Path $ManifestPath) ('files\'+$row.Path)) -Destination $dest -Force}
        else{Remove-Item -LiteralPath $dest}
    }
}
if($RollbackManifest){
    $manifestPath=[IO.Path]::GetFullPath($RollbackManifest)
    if(-not $manifestPath.StartsWith($root+'\backups\',[StringComparison]::OrdinalIgnoreCase)){throw 'Manifest is outside protected backups'}
    $manifest=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
    Validate-Rollback $manifest $manifestPath
    if($CheckOnly){Write-Output 'Management rollback hash check PASS';exit}
    & (Join-Path $root 'manage.ps1') stop
    if($LASTEXITCODE-ne 0){throw 'Manual stop failed; rollback not applied'}
    try {
        Lock-Release
        Apply-Rollback $manifest $manifestPath
    } finally {Unlock-Release}
    Write-Output 'Management rollback complete; service remains stopped';exit
}
$sources=@{}
foreach($name in $allowed){
    $sources[$name]=if($name-eq 'bin\qwen36.cmd'){Join-Path $PSScriptRoot 'qwen36.cmd'}else{Join-Path $output $name}
}
foreach($source in $sources.Values){
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){throw ('Missing source: '+$source)}
    if([IO.Path]::GetExtension($source)-eq '.ps1'){
        $tokens=$null;$errors=$null
        [Management.Automation.Language.Parser]::ParseFile($source,[ref]$tokens,[ref]$errors)|Out-Null
        if($errors){throw ('PowerShell parse error: '+$source)}
    }
}
if($CheckOnly){Write-Output ('Management release check PASS: '+$sources.Count+' files; no services or startup settings changed');exit}
try {
    Lock-Release
    Require-Stopped
    $backup=Join-Path $root ('backups\optimization-management-'+(Get-Date -Format yyyyMMddTHHmmss)+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $backup|Out-Null
    $acl=Get-Acl -LiteralPath $backup
    $acl.SetAccessRuleProtection($true,$false)
    foreach($identity in @([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.Principal.SecurityIdentifier]::new('S-1-5-18'))){
        $rule=[Security.AccessControl.FileSystemAccessRule]::new($identity,'FullControl','ContainerInherit,ObjectInherit','None','Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $backup -AclObject $acl
    $rows=@()
    foreach($name in $sources.Keys){
        $dest=Target $name;$before=Hash $dest
        if($before){
            $saved=Join-Path $backup ('files\'+$name)
            New-Item -ItemType Directory -Force -Path (Split-Path $saved)|Out-Null
            Copy-Item -LiteralPath $dest -Destination $saved
        }
        $rows+=@{Path=$name;Before=$before;After=(Hash $sources[$name])}
    }
    $manifestPath=Join-Path $backup 'manifest.json'
    @{Files=$rows;AutoStart=$false}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $manifestPath
    foreach($name in $sources.Keys){
        $dest=Target $name
        New-Item -ItemType Directory -Force -Path (Split-Path $dest)|Out-Null
        Copy-Item -LiteralPath $sources[$name] -Destination $dest -Force
    }
    Write-Output ('Management release installed. Rollback manifest: '+$manifestPath)
} finally {Unlock-Release}
