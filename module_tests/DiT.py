import torch as pt
import custom_modules_cpp as F
from code import interact
from inspect import signature
from modules import MMDiT

# Parameters
B,H,W = 2, 64, 64
Tc = 77+77
captions_dim = 2048
pooled_dim = 2048
in_channels = 16
emb_dim = 256
attn_heads = 2
mlp_expand = 4
use_qk_norm = True
patch_size = 2
pos_embed_max_size = 96
base_height = 64
num_layers = 3
dual_attention_layers = (0,)
init_range = lambda emb_dim: (-emb_dim**-0.5, emb_dim**-0.5)

model = MMDiT(num_layers, dual_attention_layers, in_channels, emb_dim, pooled_dim, captions_dim, patch_size, attn_heads, mlp_expand, pos_embed_max_size, base_height, use_qk_norm)
params = dict()
for name, tensor in model.named_parameters():
    if tensor.dim() != 2:
        params[name] = tensor.data.clone()
    else:
        # linear weights
        params[name] = tensor.data.T.contiguous()


# INPUTS
latent = pt.randn((B, in_channels, H, W))
captions = pt.randn((B, Tc, captions_dim))
pooled_captions = pt.randn((B, pooled_dim))
timesteps = pt.rand((B,))

x_out = F.DiT(
    pooled_captions, captions, latent, timesteps,
    params,
    # others
    num_layers, set(dual_attention_layers), 
    pooled_dim, captions_dim, in_channels, emb_dim, attn_heads, mlp_expand,
    patch_size, pos_embed_max_size, base_height, use_qk_norm
)

out = model.forward(latent, captions, pooled_captions, timesteps)

assert (x_out - out).abs().max().item() < 1e-5

print("Test passed successfully!")