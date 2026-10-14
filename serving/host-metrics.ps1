param([Parameter(Mandatory=$true)][ValidateRange(1,2147483647)][int]$OwnerPid,
      [Parameter(Mandatory=$true)][string]$OwnerStarted,[string]$Root='C:\path\to\LLM\qwen36-api')
$ErrorActionPreference='Stop'
$file=Join-Path $Root 'logs\host-memory.jsonl'
$sampleErrors=0
while($true){
    try{
        $owner=Get-Process -Id $OwnerPid -ErrorAction SilentlyContinue
        if(-not $owner -or $owner.StartTime.ToUniversalTime().ToString('o')-ne $OwnerStarted){break}
    }catch{break}
    try{
        $os=Get-CimInstance Win32_OperatingSystem
        $memory=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
        $vm=@(Get-Process -Name 'VmmemWSL' -ErrorAction SilentlyContinue)
        $row=@{ts=[DateTime]::UtcNow.ToString('o');windows_available_bytes=[int64]$os.FreePhysicalMemory*1024;
            windows_total_bytes=[int64]$os.TotalVisibleMemorySize*1024;windows_commit_bytes=[int64]$memory.CommittedBytes;
            windows_commit_limit_bytes=[int64]$memory.CommitLimit;wsl_vm_working_set_bytes=[int64](($vm|Measure-Object WorkingSet64 -Sum).Sum);
            sample_errors_total=$sampleErrors}
        if((Test-Path -LiteralPath $file) -and (Get-Item -LiteralPath $file).Length-gt 2MB){
            for($i=3;$i-ge 1;$i--){if(Test-Path -LiteralPath "$file.$i"){Move-Item -LiteralPath "$file.$i" -Destination "$file.$($i+1)" -Force}}
            Move-Item -LiteralPath $file -Destination "$file.1" -Force
        }
        [IO.File]::AppendAllText($file,($row|ConvertTo-Json -Compress)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
    }catch{$sampleErrors++}
    Start-Sleep -Seconds 5
}
