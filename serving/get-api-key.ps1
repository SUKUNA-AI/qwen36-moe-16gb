# Owner-only retrieval. Key is captured, never printed or placed in process arguments.
#   qwen36 key               copy the owner key
#   qwen36 key NAME          copy the user key NAME
#   qwen36 key add NAME      issue a user key NAME and copy it
#   qwen36 key remove NAME   revoke the user key NAME (effective on the next request, no restart)
#   qwen36 key list          list user key names
# User keys live in secrets/users.env (name=key, 0600); guard.py re-reads it on change.
param([string]$Command='',[string]$Name='')
$ErrorActionPreference='Stop'
$users='/home/USER/qwen36-api/secrets/users.env'
function Read-Users {
    $lines=& wsl.exe -d archlinux --exec sh -c "test ! -e $users || cat $users"
    if($LASTEXITCODE-ne 0){throw 'Unable to read owner-only user keys'}
    # Comma keeps an empty result an empty array instead of $null.
    return ,@($lines | ForEach-Object {"$_".Trim().TrimStart([char]0xFEFF)} | Where-Object {$_ -and -not $_.StartsWith('#')})
}
function Write-Users($lines) {
    $text=(@($lines)-join "`n")+"`n"
    # Raw ASCII bytes on stdin, never arguments; umask keeps the new file owner-only,
    # mv replaces it atomically. The stdin writer takes the console input encoding,
    # whose UTF-8 preamble would otherwise become a BOM in the file.
    $start=New-Object Diagnostics.ProcessStartInfo 'wsl.exe'
    $start.Arguments="-d archlinux --exec sh -c `"umask 077 && cat > $users.new && mv -f $users.new $users`""
    $start.UseShellExecute=$false
    $start.RedirectStandardInput=$true
    $saved=[Console]::InputEncoding
    [Console]::InputEncoding=New-Object Text.UTF8Encoding $false
    try{$process=[Diagnostics.Process]::Start($start)}finally{[Console]::InputEncoding=$saved}
    $bytes=[Text.Encoding]::ASCII.GetBytes($text)
    $process.StandardInput.BaseStream.Write($bytes,0,$bytes.Length)
    $process.StandardInput.Close()
    $process.WaitForExit()
    [Array]::Clear($bytes,0,$bytes.Length)
    if($process.ExitCode-ne 0){throw 'Unable to write owner-only user keys'}
}
function Assert-Name {
    if($Name-cnotmatch '^[a-z0-9][a-z0-9_-]{0,31}$' -or $Name-eq 'owner'){throw 'User name must match [a-z0-9][a-z0-9_-]{0,31} and not be owner'}
}
function Copy-Key($value,$label) {
    Set-Clipboard -Value $value
    Write-Output "$label copied to your local clipboard. Do not paste it into chat or logs."
}
if($Command-eq ''){
    $line=& wsl.exe -d archlinux -- cat /home/USER/qwen36-api/secrets/api.env
    if($LASTEXITCODE-ne 0){throw 'Unable to read owner-only API secret'}
    if($line-notmatch '^QWEN36_API_KEY=([A-Za-z0-9_-]{43,})$'){throw 'Unexpected secret file format'}
    Copy-Key $Matches[1] 'API key'
    $line=$null;$Matches.Clear()
}elseif($Command-eq 'list'){
    $current=Read-Users
    $names=@($current | ForEach-Object {($_ -split '=',2)[0]})
    if($names.Count){$names}else{Write-Output 'No user keys issued.'}
}elseif($Command-eq 'add'){
    Assert-Name
    $current=Read-Users
    if(@($current | Where-Object {$_.StartsWith("$Name=")}).Count){throw "User key $Name already exists; remove it first"}
    $bytes=New-Object byte[] 32
    $rng=[Security.Cryptography.RandomNumberGenerator]::Create()
    try{$rng.GetBytes($bytes)}finally{$rng.Dispose()}
    $value=[Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
    Write-Users (@($current)+"$Name=$value")
    Copy-Key $value "User key $Name"
    $value=$null;$current=$null
}elseif($Command-eq 'remove'){
    Assert-Name
    $current=Read-Users
    $kept=@($current | Where-Object {-not $_.StartsWith("$Name=")})
    if($kept.Count-eq @($current).Count){throw "No user key $Name"}
    Write-Users $kept
    Write-Output "User key $Name revoked."
    $current=$null;$kept=$null
}else{
    $Name=$Command
    Assert-Name
    $current=Read-Users
    $entry=@($current | Where-Object {$_.StartsWith("$Name=")})
    if($entry.Count-ne 1){throw "No user key $Name"}
    Copy-Key ($entry[0] -split '=',2)[1] "User key $Name"
    $entry=$null;$current=$null
}
