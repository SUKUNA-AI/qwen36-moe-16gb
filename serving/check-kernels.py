import torch
import flashinfer.sampling as sampling
p=torch.softmax(torch.randn(1,248320,device='cuda'),dim=-1)
for operation in [lambda:sampling.sampling_from_probs(p),lambda:sampling.top_k_top_p_sampling_from_probs(p,20,.95)]:
    value=operation();torch.cuda.synchronize();assert 0<=value.item()<p.shape[-1]
print('FLASHINFER_SAMPLING_GPU_PASS')
