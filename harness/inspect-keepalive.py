"""Read-only verification of one generation/lease's private Linux process receipt."""
import argparse,json,os,pathlib,re,select,stat,sys

ROOT=pathlib.Path('/home/USER/qwen36-api/run/keepalive')

def starttime(pid):
    raw=pathlib.Path(f'/proc/{pid}/stat').read_text()
    return int(raw[raw.rfind(')')+2:].split()[19])

def private(path,directory):
    info=path.lstat()
    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise ValueError('receipt_permissions')
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('receipt_type')

def record_for(generation,lease,root):
    if not re.fullmatch(r'[a-f0-9]{32}',generation) or not re.fullmatch(r'[a-f0-9]{32}',lease):
        raise ValueError('invalid_identity')
    directory=root/generation;private(root,True);private(directory,True)
    paths=list(directory.glob('*.json'))
    if len(paths)>64:raise ValueError('receipt_count_limit')
    records=[]
    for path in paths:
        private(path,False)
        with path.open('rb') as handle:raw=handle.read(16385)
        if len(raw)>16384:raise ValueError('receipt_size_limit')
        value=json.loads(raw)
        if value.get('generation')!=generation or value.get('lease_id')!=lease:continue
        pid=value.get('linux_pid');ticks=value.get('linux_starttime_ticks')
        if type(pid) is not int or pid<1 or type(ticks) is not int or ticks<1 or value.get('linux_uid')!=os.getuid():
            raise ValueError('invalid_receipt_identity')
        if path.name!=f'{pid}-{ticks}.json':raise ValueError('receipt_filename_identity')
        records.append(value)
    if len(records)!=1:raise ValueError('missing_or_ambiguous_receipt')
    return records[0]

def verify(generation,lease,expect='running',wait_seconds=0,root=ROOT):
    if sys.platform!='linux' or os.getuid()==0:raise ValueError('ordinary_linux_owner_required')
    if expect not in ('running','gone') or not 0<=wait_seconds<=60:raise ValueError('invalid_expectation')
    record=record_for(generation,lease,root);pid=record['linux_pid'];ticks=record['linux_starttime_ticks']
    fd=None;alive=False
    try:
        try:
            same=starttime(pid)==ticks and pathlib.Path(f'/proc/{pid}').stat().st_uid==os.getuid()
            if same:
                fd=os.pidfd_open(pid)
                same=starttime(pid)==ticks
                if same:
                    args=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                    def argument(name):
                        index=args.index(name.encode());return args[index+1].decode()
                    if (argument('--generation')!=generation or argument('--lease-id')!=lease or
                        not any(pathlib.PurePosixPath(arg.decode()).name=='keepalive.py' for arg in args if arg)):
                        raise ValueError('process_role_mismatch')
                    poll=select.poll();poll.register(fd,select.POLLIN)
                    alive=not bool(poll.poll(int(wait_seconds*1000) if expect=='gone' else 0))
        except (FileNotFoundError,ProcessLookupError):alive=False
        return {'pass':alive if expect=='running' else not alive,'expect':expect,'exact_identity_alive':alive,
            'generation':generation,'lease_id':lease,'linux_pid':pid,'linux_starttime_ticks':ticks,
            'read_only':True,'signals_sent':0,'wait_seconds':wait_seconds}
    finally:
        if fd is not None:os.close(fd)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation',required=True);p.add_argument('--lease-id',required=True)
    p.add_argument('--expect',choices=('running','gone'),default='running');p.add_argument('--wait-seconds',type=int,default=0)
    args=p.parse_args()
    try:result=verify(args.generation,args.lease_id,args.expect,args.wait_seconds)
    except Exception as error:result={'pass':False,'error_type':type(error).__name__,'read_only':True,'signals_sent':0}
    print(json.dumps(result));return 0 if result['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
