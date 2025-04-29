import torch
from torch.nn import functional as F
# pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from time import time

def measure_performance(fn, input, evals, warmups, **kwargs):

    for i in range(warmups):
        _ = fn(*input, **kwargs)
    t0 = time()
    for i in range(evals):
        _ = fn(*input, **kwargs)
    torch.cuda.synchronize()
    dt = time() - t0
    print(f"{fn.__name__}: {evals/dt} evals/s")

device = torch.device("cuda:0")
B, T, C = 16, 256, 1024
nheads = 32
headdim = C // nheads
dtype = torch.bfloat16

qkv = torch.randn((B, T, 3, nheads, headdim), device=device, dtype=dtype, requires_grad=True)
q = qkv[:,:,0].transpose(1,2).contiguous()
k = qkv[:,:,1].transpose(1,2).contiguous()
v = qkv[:,:,2].transpose(1,2).contiguous()

out1 = F.scaled_dot_product_attention(q, k, v, scale=headdim**-0.5)
out1 = out1.transpose(1,2).contiguous()
out2 = flash_attn_qkvpacked_func(qkv, softmax_scale=headdim**-0.5)

print("Norm diff:", (out1 - out2).norm().item())

measure_performance(F.scaled_dot_product_attention, (q, k, v), evals=1000, warmups=100, scale=headdim**-0.5)
measure_performance(flash_attn_qkvpacked_func, (qkv,), evals=1000, warmups=100, softmax_scale=headdim**-0.5)
measure_performance(flash_attn_func, (q, k, v), evals=1000, warmups=100, softmax_scale=headdim**-0.5)
from code import interact
interact(local=locals())