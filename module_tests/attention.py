import torch
from torch import nn, Tensor
import custom_modules_cpp as F2
from code import interact
from time import time
# torch.set_printoptions(sci_mode=False)
    
class attention_cpp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V:Tensor, num_heads:int):
        attn_out = F2.mha(Q, K, V, num_heads)
        return attn_out
    
    @staticmethod
    def backward(ctx, output_grad):
        raise NotImplemented
    

class Attention(nn.Module):
    """Self attention or cross-attention"""

    def __init__(self, attn_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_heads = attn_heads
    
    def forward(self, Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
        B,T,C = K.shape; L = Q.size(1)

        k = K.view(B, T, self.attn_heads, C//self.attn_heads).transpose(1,2)
        q = Q.view(B, L, self.attn_heads, C//self.attn_heads).transpose(1,2)
        v = V.view(B, T, self.attn_heads, C//self.attn_heads).transpose(1,2)
        y = nn.functional.scaled_dot_product_attention(q, k, v)
        
        return y.transpose(1,2).flatten(2, -1)


def measure_performance(fn, input, evals, warmups, return_output=True, **kwargs):

    for i in range(warmups):
        _ = fn(*input, **kwargs)
    t0 = time()
    for i in range(evals):
        output = fn(*input, **kwargs)
    dt = time() - t0
    print(f"{type(fn).__name__}: {evals/dt} evals/s")

    if return_output: return output


if __name__ == "__main__":
    B, T, L, C = 2, 128, 96, 512
    num_heads = 8
    torch.manual_seed(1)
    Q = torch.randn((B,T,C), requires_grad=True)
    K = torch.randn((B,L,C), requires_grad=True)
    V = torch.randn((B,L,C), requires_grad=True)

    # Pytorch written module
    attn = Attention(num_heads)
    
    # Pytorch API
    mha = nn.MultiheadAttention(C, num_heads, 0.0, batch_first=True)
    # We assume K,Q,V projections are already done, so remove them
    mha.in_proj_weight.data = torch.cat([torch.eye(C), torch.eye(C),torch.eye(C)], 0)
    mha.out_proj.weight.data = torch.eye(C)

    out1, _ = measure_performance(mha, (Q, K, V), evals=1000, warmups=10, need_weights=False)

    # our version
    out2 = measure_performance(attention_cpp.apply, (Q, K, V, num_heads), evals=1000, warmups=10)

    out3 = measure_performance(attn, (Q, K, V), evals=1000, warmups=10)


    # PyTorch's version
    # attn_out = nn.functional.scaled_dot_product_attention(Q, K, V)
    # # Our version
    # attn_out2 = attention_cpp.apply(Q, K, V)

    print(torch.norm(out1 - out2).item())
    print(torch.norm(out2 - out3).item())
    print(torch.norm(out1 - out3).item())
    interact(local=locals())
    # fakeloss = (attn_out * dout).sum()
    # fakeloss.backward()
