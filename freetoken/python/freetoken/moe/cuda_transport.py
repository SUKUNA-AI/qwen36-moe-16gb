"""Opt-in standalone CUDA transport; no implicit runtime integration."""
import ctypes
import pathlib
import torch

ROOT = pathlib.Path.home() / 'qwen36-api/optimization/cuda-transport'

class Transport:
    def __init__(self, sizes, threads=256, unroll=4, policy='balanced', cap=8):
        if threads not in (128,256,512,1024) or unroll not in (1,4): raise ValueError('Unsupported launch shape')
        if threads==1024 and policy!='persistent':raise ValueError('1024 threads requires persistent grid')
        if any(s <= 0 or s % 16 for s in sizes): raise ValueError('Rows must have 16-byte alignment')
        if not 1 <= cap <= 64: raise ValueError('Invalid CTA cap')
        self.threads, self.unroll = threads, unroll
        tasks=[]
        for bank, size in enumerate(sizes):
            n = cap if policy in ('uniform','persistent') else max(1,min(cap,(size+65535)//65536))
            vectors=size//16;shift=vectors.bit_length()-1 if vectors&(vectors-1)==0 else -1
            tasks.extend((bank,i,n,shift) for i in range(n))
        self.tasks=torch.tensor(tasks,dtype=torch.int32,device='cuda')
        self.lib=ctypes.CDLL(str(ROOT/'libtransport.so'))
        p,i,u=ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t
        prefix='qwen_transport_flat_' if policy=='persistent' else 'qwen_transport_'
        self.exchange_function=getattr(self.lib,prefix+'exchange')
        self.gather_function=getattr(self.lib,prefix+'gather')
        self.exchange_function.argtypes=[p]*8+[i]*7+[u]
        self.gather_function.argtypes=[p]*6+[i]*7+[u]
        self.exchange_function.restype=self.gather_function.restype=i

    def exchange(self,h,g,b,hs,src,dst,count,base=0,offset=0,exchange=True):
        rc=self.exchange_function(*[x.data_ptr() for x in (h,g,b,hs,src,dst,count,self.tasks)],
            len(self.tasks),len(src),base,offset,int(exchange),self.threads,self.unroll,torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(f'CUDA transport launch {rc}')

    def gather(self,h,g,b,hs,gs,base=0,offset=0,destination=0,experts=256):
        rc=self.gather_function(*[x.data_ptr() for x in (h,g,b,hs,gs,self.tasks)],
            len(self.tasks),experts,base,offset,destination,self.threads,self.unroll,torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(f'CUDA gather launch {rc}')
