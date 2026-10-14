# Owner-only retrieval. Key is captured, never printed or placed in process arguments.
$ErrorActionPreference='Stop'
$line=& wsl.exe -d archlinux -- cat /home/USER/qwen36-api/secrets/api.env
if($LASTEXITCODE-ne 0){throw 'Unable to read owner-only API secret'}
if($line-notmatch '^QWEN36_API_KEY=([A-Za-z0-9_-]{43,})$'){throw 'Unexpected secret file format'}
$value=$Matches[1]
Set-Clipboard -Value $value
$value=$null;$line=$null;$Matches.Clear()
Write-Output 'API key copied to your local clipboard. Do not paste it into chat or logs.'
