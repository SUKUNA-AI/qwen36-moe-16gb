"""Lossless exchanges between mapped pinned RAM and GPU expert slots.

One host slot is reused in place: its promoted bytes are read into registers
before the victim overwrites them. No host-to-host copy or runtime disk tier.
Ownership commits on the same CUDA stream after all bank tiles finish.
"""
import triton
import triton.language as tl


@triton.jit
def frequency_plan(IDS,MAP,OWNERS,FREQ,AT,STEP,SRC,DST,COUNT,STATS,
                   BASE:tl.constexpr,SLOTS:tl.constexpr,N:tl.constexpr,HALFLIFE:tl.constexpr,BLOCK:tl.constexpr):
    slots=tl.arange(0,BLOCK);valid=slots<SLOTS
    owners=tl.load(OWNERS+slots,valid,0)
    step=tl.load(STEP)+1;tl.store(STEP,step)
    age=(step-tl.load(AT+owners,valid,0)).to(tl.float32)
    score=tl.load(FREQ+owners,valid,0)*tl.exp2(-age/HALFLIFE)
    protected=~valid
    for i in range(N):
        gid=BASE+tl.load(IDS+i)
        protected=protected | (owners==gid)
    count=0
    for i in range(N):
        local=tl.load(IDS+i);gid=BASE+local;slot=tl.load(MAP+gid)
        if slot<0:
            slot=tl.argmin(tl.where(protected,float('inf'),score),axis=0)
            victim=tl.load(OWNERS+slot)
            tl.store(MAP+victim,-1);tl.store(MAP+gid,slot);tl.store(OWNERS+slot,gid)
            tl.store(SRC+count,local);tl.store(DST+count,slot);count+=1
            protected=protected | (slots==slot)
        value=tl.load(FREQ+gid)*tl.exp2(-(step-tl.load(AT+gid)).to(tl.float32)/HALFLIFE)+1.
        tl.store(FREQ+gid,value);tl.store(AT+gid,step);tl.store(IDS+i,slot)
    tl.store(COUNT,count)
    tl.store(STATS,tl.load(STATS)+N)
    tl.store(STATS+1,tl.load(STATS+1)+count)
    tl.store(STATS+2,tl.load(STATS+2)+1)


@triton.jit
def exchange(HOST, GPU, FEAT, HOST_SLOT, OLD_IDS, SRC, DST, COUNT,
             BASE:tl.constexpr, OFFSET:tl.constexpr, BLOCK:tl.constexpr):
    bank, item, tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    if item < tl.load(COUNT):
        gid=BASE+tl.load(SRC+item)
        hslot=tl.load(HOST_SLOT+gid)
        relative=tl.load(DST+item)
        n=tl.load(FEAT+bank)//4
        for start in range(tile*BLOCK,n,8*BLOCK):
            x=start+tl.arange(0,BLOCK)
            hp=tl.load(HOST+bank).to(tl.pointer_type(tl.int32))+hslot.to(tl.int64)*n+x
            gp=tl.load(GPU+bank).to(tl.pointer_type(tl.int32))+(relative.to(tl.int64)+OFFSET)*n+x
            promoted=tl.load(hp,x<n,0)
            evicted=tl.load(gp,x<n,0)
            tl.store(hp,evicted,x<n)
            tl.store(gp,promoted,x<n)


@triton.jit
def commit(HOST_SLOT, OLD_IDS, SRC, DST, COUNT, BASE:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    valid=i<tl.load(COUNT)
    gid=BASE+tl.load(SRC+i,valid,0)
    slot=tl.load(DST+i,valid,0)
    victim=tl.load(OLD_IDS+slot,valid,0)
    hslot=tl.load(HOST_SLOT+gid,valid,0)
    tl.store(HOST_SLOT+victim,hslot,valid)
    tl.store(HOST_SLOT+gid,-1,valid)


@triton.jit
def gather_layer(HOST,GPU,FEAT,HOST_SLOT,GPU_SLOT,BASE:tl.constexpr,
                 OFFSET:tl.constexpr,DEST:tl.constexpr,BLOCK:tl.constexpr):
    bank,expert,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    gid=BASE+expert
    gs=tl.load(GPU_SLOT+gid)
    hs=tl.load(HOST_SLOT+gid)
    n=tl.load(FEAT+bank)//4
    gpu=tl.load(GPU+bank).to(tl.pointer_type(tl.int32))
    host=tl.load(HOST+bank).to(tl.pointer_type(tl.int32))
    for start in range(tile*BLOCK,n,8*BLOCK):
        x=start+tl.arange(0,BLOCK)
        if gs>=0:
            values=tl.load(gpu+(gs.to(tl.int64)+OFFSET)*n+x,x<n,0)
        else:
            values=tl.load(host+hs.to(tl.int64)*n+x,x<n,0)
        tl.store(gpu+(DEST+expert).to(tl.int64)*n+x,values,x<n)


@triton.jit
def fixed_plan(IDS,GPU_SLOT,SRC,DST,COUNT,BASE:tl.constexpr,OFFSET:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.arange(0,BLOCK);valid=x<N
    ids=tl.load(IDS+x,valid,0)
    gs=tl.load(GPU_SLOT+BASE+ids,valid,-1)
    misses=valid & (gs<0)
    out=tl.cumsum(misses.to(tl.int32))-1
    tl.store(SRC+out,ids,misses)
    tl.store(DST+out,out,misses)
    tl.store(IDS+x,tl.where(gs>=0,gs+OFFSET,out),valid)
    tl.store(COUNT,tl.sum(misses.to(tl.int32)))


@triton.jit
def fixed_copy(HOST,GPU,FEAT,HOST_SLOT,SRC,DST,COUNT,BASE:tl.constexpr,BLOCK:tl.constexpr):
    bank,item,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    if item<tl.load(COUNT):
        hs=tl.load(HOST_SLOT+BASE+tl.load(SRC+item))
        ds=tl.load(DST+item)
        n=tl.load(FEAT+bank)//4
        for start in range(tile*BLOCK,n,8*BLOCK):
            x=start+tl.arange(0,BLOCK)
            hp=tl.load(HOST+bank).to(tl.pointer_type(tl.int32))+hs.to(tl.int64)*n+x
            gp=tl.load(GPU+bank).to(tl.pointer_type(tl.int32))+ds.to(tl.int64)*n+x
            tl.store(gp,tl.load(hp,x<n,0),x<n)
