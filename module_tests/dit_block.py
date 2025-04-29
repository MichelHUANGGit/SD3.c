import torch as pt
from torch.nn.functional import silu
import custom_modules as F
from code import interact
from inspect import signature
from modules import MM_DiT_Block

# Parameters
B,H,W = 16, 64, 64
Tc, Tx = 77+77, 512
emb_dim = 768
attn_heads = 12
mlp_expand = 2
discard_context = False
use_dual_attention = True
use_kq_norm = True
# init_rstd = 0.01
init_range = (-emb_dim**-0.5, emb_dim**-0.5)

# initialization
DiT_block = MM_DiT_Block(emb_dim, attn_heads, mlp_expand, discard_context, use_dual_attention, use_kq_norm)
d = DiT_block
params = {}
for name, tensor in DiT_block.named_parameters():
    params[name] = tensor.data.clone() if tensor.dim() != 2 else tensor.data.T.contiguous()

# inputs
x = pt.randn((B, Tx, emb_dim))
y = silu(pt.randn((B, emb_dim)))
c = pt.randn((B, Tc, emb_dim))
c_clone = c.clone()
x_clone = x.clone()
y_clone = y.clone()

# CPU Reference
with pt.no_grad():
    c_out_true, x_out_true = DiT_block.forward(x, c, y)

# C++ CPU implementation
c_out, x_out = F.Dit_block(
    y, c, x, 
    # context
    params,
    # others
    B, Tc, Tx, emb_dim, attn_heads, mlp_expand, use_dual_attention, use_kq_norm, discard_context
)

if not discard_context:
    print("c_out c++ Max diff: ", (c_out - c_out_true).abs().max().item())
print("x_out c++ Max diff: ", (x_out - x_out_true).abs().max().item())

# cuda implementation
c_clone = c_clone.to("cuda")
x_clone = x_clone.to("cuda")
y_clone = y_clone.to("cuda")
params_cuda = {}
for name, tensor in DiT_block.named_parameters():
    params_cuda[name] = tensor.data.to("cuda") if tensor.dim() != 2 else tensor.data.T.contiguous().to("cuda")

c_out_cuda, x_out_cuda = F.DiT_block_cuda(
    y_clone, c_clone, x_clone, 
    # context
    params_cuda,
    # others
    B, Tc, Tx, emb_dim, attn_heads, mlp_expand, use_dual_attention, use_kq_norm, discard_context
)

if not discard_context:
    print("c_out cuda Max diff: ", (c_out_cuda.cpu() - c_out_true).abs().max().item())
print("x_out cuda Max diff: ", (x_out_cuda.cpu() - x_out_true).abs().max().item())