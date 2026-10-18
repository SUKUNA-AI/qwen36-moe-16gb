$ErrorActionPreference='Stop'
$root='C:\path\to\LLM\qwen36-api'
$bin=Join-Path $root 'bin'
$existing=Get-Command qwen36 -ErrorAction SilentlyContinue
if($existing -and $existing.Source-ne (Join-Path $bin 'qwen36.cmd')){throw 'An unrelated qwen36 command exists; refusing to shadow it'}
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$launcher=@'
@echo off
if "%~1"=="key" (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\get-api-key.ps1" "%~2" "%~3"
) else if "%~1"=="" (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\manage.ps1" start
) else if "%~2"=="--profile" (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\manage.ps1" -Action "%~1" -Profile "%~3"
) else (
  powershell.exe -NoProfile -File "C:\path\to\LLM\qwen36-api\manage.ps1" %*
)
'@
[IO.File]::WriteAllText((Join-Path $bin 'qwen36.cmd'),$launcher+"`r`n",[Text.Encoding]::ASCII)
$reg=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment',$true)
try{
    $old=[string]$reg.GetValue('Path','',[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $parts=@($old -split ';' | Where-Object {$_})
    if($parts -notcontains $bin){
        $backup=Join-Path $root ('backups\user-path-before-qwen36-'+(Get-Date -Format yyyyMMddTHHmmss)+'.json')
        @{Value=$old;Kind=[string]$reg.GetValueKind('Path')} | ConvertTo-Json | Set-Content -LiteralPath $backup
        $kind=$reg.GetValueKind('Path')
        $reg.SetValue('Path',(($parts+$bin)-join ';'),$kind)
    }
}finally{$reg.Dispose()}
if(($env:Path-split ';')-notcontains $bin){$env:Path += ';'+$bin}
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Qwen36EnvironmentNotice {
 [DllImport("user32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 public static extern IntPtr SendMessageTimeout(IntPtr h, uint m, UIntPtr w, string l, uint f, uint t, out UIntPtr r);
}
'@
$result=[UIntPtr]::Zero
[Qwen36EnvironmentNotice]::SendMessageTimeout([IntPtr]65535,26,[UIntPtr]::Zero,'Environment',2,2000,[ref]$result) | Out-Null
@{AutomaticStartup=$false;Command='qwen36';Bin=$bin;DefaultAction='start';Actions=@('start','stop','status','restart','key','metrics')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'run\manual-command.json')
Write-Output 'Manual qwen36 command installed in user PATH. No login startup. Open a new terminal for the refreshed PATH.'
