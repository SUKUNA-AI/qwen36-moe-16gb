param([string]$Root='C:\path\to\LLM\qwen36-api')
$ErrorActionPreference='Stop'
$runDir=Join-Path $Root 'run'
$logDir=Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $runDir,$logDir | Out-Null
$mutex=[Threading.Mutex]::new($false,'Global\Qwen36ApiSupervisor')
$acquired=$false
try {$acquired=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$acquired=$true}
if(-not $acquired){$mutex.Dispose();exit 0}
$statePath=Join-Path $runDir 'supervisor.json'
$stopPath=Join-Path $runDir 'stop.request'
$ownerStarted=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$generation=[Guid]::NewGuid().ToString('N')
$children=@{}
$restartTimes=@{}
$phase='starting'
$exitCode=0
$jobArmed=$false
$modelStartRequested=$false
$watchReady=$null
$heartbeatSequence=0L
$keepaliveLeaseId=$null
$stateWriteAttempts=20
$heartbeatFailureSeconds=20
$lastHeartbeatWrite=$null
function Wsl-StatePath {
    $full=[IO.Path]::GetFullPath($statePath)
    if($full-notmatch '^([A-Za-z]):\\'){throw 'Keepalive requires a local drive-backed Windows state path'}
    return '/mnt/'+$Matches[1].ToLowerInvariant()+'/'+$full.Substring(3).Replace('\','/')
}
function Start-Keepalive {
    $linuxState=Wsl-StatePath
    $script:keepaliveLeaseId=[Guid]::NewGuid().ToString('N')
    Save-State
    Start-OwnedChild 'Keepalive' "$env:WINDIR\System32\wsl.exe" @('-d','archlinux','--',
        '/home/USER/qwen36-api/.venv/bin/python','/home/USER/qwen36-api/keepalive.py',
        '--state',('"'+$linuxState+'"'),'--generation',$generation,'--lease-id',$keepaliveLeaseId,
        '--owner-pid',[string]$PID,'--owner-started',$ownerStarted)
}
function Initialize-LifetimeNative {
Add-Type @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class Qwen36Power {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
public static class Qwen36LifetimeJob {
    // Intentionally retained until process termination. Closing the last handle
    // while this process is a member would kill the owner before its finally.
    // No SafeHandle finalizer or public Dispose can close this handle early.
    private static IntPtr ownedJob = IntPtr.Zero;
    public static bool InheritedParentJob { get; private set; }
    [StructLayout(LayoutKind.Sequential)] private struct BasicLimits {
        public long PerProcessUserTime, PerJobUserTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IoCounters {
        public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] private struct ExtendedLimits {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool GetHandleInformation(IntPtr handle, out uint flags);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
    [DllImport("kernel32.dll")] private static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);
    private static void Require(bool ok, string operation) {
        if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error(), operation);
    }
    public static void ArmForCurrentProcess() {
        if (ownedJob != IntPtr.Zero) throw new InvalidOperationException("Lifetime job already created");
        bool inherited;
        Require(IsProcessInJob(GetCurrentProcess(), IntPtr.Zero, out inherited), "Query parent-job membership");
        InheritedParentJob = inherited;
        // NULL security attributes and name: only this process owns an unnamed,
        // non-inheritable job handle. Children inherit membership, not ownership.
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(), "Create lifetime job");
        bool assigned = false;
        try {
            var limits = new ExtendedLimits();
            limits.Basic.Flags = 0x00002000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            // No BREAKAWAY flags, UI limits or changes to any ancestor job.
            Require(SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits))), "Set lifetime job limits");
            Require(SetHandleInformation(job, 1, 0), "Disable lifetime handle inheritance");
            uint flags;
            Require(GetHandleInformation(job, out flags), "Check lifetime handle flags");
            if ((flags & 1) != 0) throw new InvalidOperationException("Lifetime handle is inheritable");
            Require(AssignProcessToJobObject(job, GetCurrentProcess()), "Join lifetime job (nested-job compatibility required)");
            assigned = true;
            ownedJob = job;
            RequireCurrentMember();
        } finally {
            // Before assignment the empty job is safe to close. After assignment
            // even a validation failure must leave its handle for process exit.
            if (!assigned) CloseHandle(job);
        }
    }
    public static bool Contains(IntPtr process) {
        if (ownedJob == IntPtr.Zero) throw new InvalidOperationException("Lifetime job not armed");
        bool member;
        Require(IsProcessInJob(process, ownedJob, out member), "Check lifetime membership");
        return member;
    }
    public static void RequireCurrentMember() {
        if (!Contains(GetCurrentProcess())) throw new InvalidOperationException("Owner is outside lifetime job");
    }
    public static bool HandleIsInheritable() {
        if (ownedJob == IntPtr.Zero) throw new InvalidOperationException("Lifetime job not armed");
        uint flags;
        Require(GetHandleInformation(ownedJob, out flags), "Check lifetime handle flags");
        return (flags & 1) != 0;
    }
}
'@
}
function Write-Event([string]$Message){
    try {
        $file=Join-Path $logDir 'supervisor.log'
        if((Test-Path -LiteralPath $file) -and (Get-Item -LiteralPath $file).Length -gt 2MB){
            for($i=3;$i-ge 1;$i--){
                $old="$file.$i"; $next="$file.$($i+1)"
                if(Test-Path -LiteralPath $old){Move-Item -LiteralPath $old -Destination $next -Force}
            }
            Move-Item -LiteralPath $file -Destination "$file.1" -Force
        }
        Add-Content -LiteralPath $file -Value "$(Get-Date -Format o) $Message"
    } catch {Write-Verbose 'Supervisor log write failed; lifecycle cleanup remains active'}
}
function Same-Process([int]$ProcessId,[string]$Started){
    if($ProcessId-le 0 -or -not $Started){return $null}
    try {
        $p=Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if($p -and $p.StartTime.ToUniversalTime().ToString('o') -eq $Started){return $p}
    } catch {}
    return $null
}
function Stop-OwnedChild([string]$Name){
    $child=$children[$Name]
    if($child){$p=Same-Process $child.Id $child.Started;if($p){try{$p.Kill();$p.WaitForExit(5000)|Out-Null}catch{}}}
}
function Save-State {
    $script:heartbeatSequence++
    $state=@{SupervisorPid=$PID;SupervisorStarted=$ownerStarted;Generation=$generation;Phase=$phase;
        Distro='archlinux';Listener='Caddy loopback 127.0.0.1:1919';Updated=[DateTime]::UtcNow.ToString('o');
        LifetimeJobArmed=[bool]$jobArmed;ModelStartRequested=[bool]$modelStartRequested;
        HeartbeatSequence=[long]$heartbeatSequence;KeepaliveLeaseId=$keepaliveLeaseId}
    foreach($name in @('Tunnel','Keepalive','Watchdog','HostMetrics')){
        $child=$children[$name]
        $state[$name+'Pid']=if($child){$child.Id}else{$null}
        $state[$name+'Started']=if($child){$child.Started}else{$null}
    }
    $temp=$statePath+'.'+$generation+'.tmp'
    $json=$state|ConvertTo-Json
    # A concurrent Windows reader without FILE_SHARE_DELETE makes Replace fail
    # with a sharing violation. Retry briefly (bounded, 20 x 50 ms); a persistent
    # failure still throws to the caller.
    for($attempt=1;;$attempt++){
        try{
            [IO.File]::WriteAllText($temp,$json,[Text.UTF8Encoding]::new($false))
            if(Test-Path -LiteralPath $statePath){[IO.File]::Replace($temp,$statePath,[NullString]::Value)}else{[IO.File]::Move($temp,$statePath)}
            break
        }catch [IO.IOException],[UnauthorizedAccessException]{
            if($attempt-ge $stateWriteAttempts){throw}
            Start-Sleep -Milliseconds 50
        }
    }
    $script:lastHeartbeatWrite=[Diagnostics.Stopwatch]::StartNew()
}
function Save-Heartbeat {
    # Only the periodic heartbeat tolerates write failures, and only while the
    # last published heartbeat is younger than 20 s (keepalive's stale bound is
    # 30 s). Ownership, intent and child identity writes stay strict.
    try{Save-State}
    catch{
        if($null-eq $lastHeartbeatWrite -or $lastHeartbeatWrite.Elapsed.TotalSeconds-ge $heartbeatFailureSeconds){throw}
        Write-Event ('Heartbeat write deferred: '+$_.Exception.GetBaseException().GetType().Name)
    }
}
function Stop-Requested {
    if(-not(Test-Path -LiteralPath $stopPath)){return $false}
    try {
        $request=Get-Content -LiteralPath $stopPath -Raw|ConvertFrom-Json
        return $request.SupervisorPid-eq $PID -and $request.SupervisorStarted-eq $ownerStarted
    } catch {return $false}
}
function Start-OwnedChild([string]$Name,[string]$File,[string[]]$Arguments,[string]$ErrorLog=''){
    if($Name-eq 'Watchdog'){
        if($jobArmed){throw 'Outside-job watcher cannot be restarted from inside the lifetime job'}
    }else{
        if(-not $jobArmed){throw 'Working child cannot start before lifetime job ownership'}
        [Qwen36LifetimeJob]::RequireCurrentMember()
    }
    $now=Get-Date
    if(-not $restartTimes.ContainsKey($Name)){$restartTimes[$Name]=[Collections.Generic.List[datetime]]::new()}
    $history=$restartTimes[$Name]
    for($i=$history.Count-1;$i-ge 0;$i--){if(($now-$history[$i]).TotalMinutes-gt 15){$history.RemoveAt($i)}}
    if($history.Count-ge 6){throw "$Name restart limit reached (6 / 15 minutes)"}
    $history.Add($now)
    $options=@{FilePath=$File;ArgumentList=$Arguments;WindowStyle='Hidden';PassThru=$true}
    if($ErrorLog){$options.RedirectStandardError=$ErrorLog}
    $p=Start-Process @options
    $null=$p.Handle
    if($Name-ne 'Watchdog' -and -not [Qwen36LifetimeJob]::Contains($p.Handle)){
        # This is an explicit failure, not a post-creation Assign workaround.
        # Own process handle is retained; never terminate by a reused numeric PID.
        try{$p.Kill();$p.WaitForExit(5000)|Out-Null}catch{}
        throw 'Working child did not inherit lifetime job membership'
    }
    $children[$Name]=@{Id=$p.Id;Started=$p.StartTime.ToUniversalTime().ToString('o')}
    Save-State
    Write-Event "$Name spawned PID=$($p.Id)"
}
try {
    Initialize-LifetimeNative
    if(Test-Path -LiteralPath $statePath){throw 'Unclean previous owner state; use qwen36 stop before starting'}
    Remove-Item -LiteralPath $stopPath -ErrorAction SilentlyContinue
    # Ownership and the death watcher exist before any model or tunnel.
    Save-State
    $readyName='Local\Qwen36WatchReady-'+$generation
    $watchReady=[Threading.EventWaitHandle]::new($false,[Threading.EventResetMode]::ManualReset,$readyName)
    $ownerArgs=@('-OwnerPid',[string]$PID,'-OwnerStarted',$ownerStarted,'-Root',('"'+$Root+'"'))
    Start-OwnedChild 'Watchdog' 'powershell.exe' (@('-NoProfile','-File',('"'+(Join-Path $Root 'watch-supervisor.ps1')+'"'))+$ownerArgs+@('-ReadyEvent',$readyName))
    if(-not $watchReady.WaitOne(10000)){throw 'Outside-job watcher did not confirm readiness before model start'}
    $watcher=Same-Process $children['Watchdog'].Id $children['Watchdog'].Started
    if(-not $watcher){throw 'Outside-job watcher exited before model start'}
    $null=$watcher.Handle
    [Qwen36LifetimeJob]::ArmForCurrentProcess()
    $jobArmed=$true
    if([Qwen36LifetimeJob]::Contains($watcher.Handle)){throw 'Death watcher must remain outside the lifetime job'}
    $phase='ownership-ready';Save-State
    [Qwen36Power]::SetThreadExecutionState([uint32]2147483649)|Out-Null
    Start-Keepalive
    if(-not(Stop-Requested)){
        # Publish intent before sending start. The outside-job watcher can stop
        # the named unit even if this process dies inside the WSL request.
        $modelStartRequested=$true;Save-State
        & wsl.exe -d archlinux -- systemctl --user reset-failed qwen36-inference.service
        if($LASTEXITCODE-ne 0){
            # systemd refuses reset-failed for a unit it has garbage-collected ("not loaded": inactive and
            # unreferenced, e.g. the first start after idle); there is no failed state to clear then.
            $active=& wsl.exe -d archlinux -- systemctl --user show qwen36-inference.service --property=ActiveState --value
            if($LASTEXITCODE-ne 0 -or ([string]$active).Trim()-ne 'inactive'){throw 'Could not reset owned inference service'}
        }
        & wsl.exe -d archlinux -- systemctl --user start qwen36-inference.service
        if($LASTEXITCODE-ne 0){throw 'Could not start owned inference service'}
    }
    $phase='model-loading-or-serving';Save-State
    Write-Event 'Model loading requested; readiness is checked separately'
    while(-not(Stop-Requested)){
        [Qwen36Power]::SetThreadExecutionState([uint32]2147483649)|Out-Null
        Save-Heartbeat
        if(-not(Same-Process $children['Watchdog'].Id $children['Watchdog'].Started)){
            # A replacement launched here would inherit the job and die with us.
            # Stop safely; a new manual start establishes a new outside watcher.
            throw 'Outside-job watcher exited; stopping owned service'
        }
        if(-not(Same-Process $children['Keepalive'].Id $children['Keepalive'].Started)){
            Start-Keepalive
        }
        if(-not(Same-Process $children['Tunnel'].Id $children['Tunnel'].Started)){
            $sshLog=Join-Path $logDir 'ssh.log'
            if(Test-Path -LiteralPath $sshLog){Move-Item -LiteralPath $sshLog -Destination "$sshLog.previous" -Force}
            Start-OwnedChild 'Tunnel' "$env:WINDIR\System32\OpenSSH\ssh.exe" @('-N','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','ConnectTimeout=10','-R','127.0.0.1:1919:127.0.0.1:1919','EDGE_SSH_HOST') $sshLog
        }
        if(-not(Same-Process $children['HostMetrics'].Id $children['HostMetrics'].Started)){
            Start-OwnedChild 'HostMetrics' 'powershell.exe' (@('-NoProfile','-File',('"'+(Join-Path $Root 'host-metrics.ps1')+'"'))+$ownerArgs)
        }
        Start-Sleep -Seconds 5
    }
    Write-Event 'Owner-scoped stop requested'
} catch {
    Write-Event ('Supervisor stopped: '+$_.Exception.Message)
    $exitCode=1
} finally {
    # The ownership mutex also fences the named systemd stop against new owners.
    $mine=$false
    try {
        if(Test-Path -LiteralPath $statePath){
            $s=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
            $mine=$s.SupervisorPid-eq $PID -and $s.SupervisorStarted-eq $ownerStarted
        }
    } catch {}
    Stop-OwnedChild 'Tunnel'
    Stop-OwnedChild 'HostMetrics'
    $stopped=-not $modelStartRequested
    if($mine -and $modelStartRequested){
        try {& wsl.exe -d archlinux -- systemctl --user stop qwen36-inference.service;$stopped=$LASTEXITCODE-eq 0}catch{}
    }
    if($stopped -and $mine){
        Stop-OwnedChild 'Keepalive'
        Stop-OwnedChild 'Watchdog'
        Remove-Item -LiteralPath $statePath -ErrorAction SilentlyContinue
        if(Stop-Requested){Remove-Item -LiteralPath $stopPath -ErrorAction SilentlyContinue}
    } elseif($mine){Write-Event 'Service stop unconfirmed; state retained for the owner-death watcher';$exitCode=1}
    if('Qwen36Power'-as [type]){[Qwen36Power]::SetThreadExecutionState([uint32]2147483648)|Out-Null}
    if($watchReady){$watchReady.Dispose()}
    # Do not close the job here: this supervisor is a member. OS process teardown
    # closes its sole handle after this finally and kills every inherited child,
    # including one created just before an interrupted Save-State.
    $mutex.ReleaseMutex();$mutex.Dispose()
}
exit $exitCode
