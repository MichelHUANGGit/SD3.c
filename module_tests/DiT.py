import torch as pt
import custom_modules_cpp as F
from code import interact
from inspect import signature
from modules import MMDiT

interact(local=locals())

# Parameters
B,H,W = 2, 64, 64
Tc = 77+77
captions_dim = 2048
pooled_dim = 2048
in_channels = 16
emb_dim = 256
attn_heads = 2
mlp_expand = 4
discard_context = False
use_dual_attention = False
use_kq_norm = True
patch_size = 2
pos_embed_max_size = 96
base_height = 64
init_range = lambda emb_dim: (-emb_dim**-0.5, emb_dim**-0.5)

model = MMDiT(1, (0,), in_channels, emb_dim, pooled_dim, captions_dim, patch_size, attn_heads, mlp_expand, pos_embed_max_size, base_height, use_kq_norm)

# INPUTS
latent = pt.randn((B, in_channels, H, W))
captions = pt.randn((B, Tc, captions_dim))
pooled_captions = pt.randn((B, pooled_dim))
timesteps = pt.rand((B,))

# WEIGHTS
COPY_WEIGHTS = True
if COPY_WEIGHTS:
    time_mlp_W1 = model.timestep_mlp[0].weight.data.T.contiguous()
    time_mlp_b1 = model.timestep_mlp[0].bias.data.clone()
    time_mlp_W2 = model.timestep_mlp[2].weight.data.T.contiguous()
    time_mlp_b2 = model.timestep_mlp[2].bias.data.clone()

    pooled_mlp_W1 = model.pooled_text_mlp[0].weight.data.T.contiguous()
    pooled_mlp_b1 = model.pooled_text_mlp[0].bias.data.clone()
    pooled_mlp_W2 = model.pooled_text_mlp[2].weight.data.T.contiguous()
    pooled_mlp_b2 = model.pooled_text_mlp[2].bias.data.clone()

    captions_W = model.context_linear.weight.data.T.contiguous()
    captions_b = model.context_linear.bias.data.clone()

    kernel_W = model.to_patches.to_patch.weight.data.clone()
    kernel_b = model.to_patches.to_patch.bias.data.clone()
    
else:
    time_mlp_W1 = pt.empty((256, emb_dim)).uniform_(*init_range(emb_dim))
    time_mlp_b1 = pt.empty((emb_dim)).uniform_(*init_range(emb_dim))
    time_mlp_W2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range(emb_dim))
    time_mlp_b2 = pt.empty((emb_dim)).uniform_(*init_range(emb_dim))

    pooled_mlp_W1 = pt.empty((pooled_dim, emb_dim)).uniform_(*init_range(pooled_dim))
    pooled_mlp_b1 = pt.empty((emb_dim)).uniform_(*init_range(pooled_dim))
    pooled_mlp_W2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range(emb_dim))
    pooled_mlp_b2 = pt.empty((emb_dim)).uniform_(*init_range(emb_dim))

    captions_W = pt.empty((captions_dim, emb_dim)).uniform_(*init_range(captions_dim))
    captions_b = pt.empty((emb_dim)).uniform_(*init_range(captions_dim))

    kernel_W = pt.empty((emb_dim, in_channels, patch_size, patch_size)).uniform_(*init_range(in_channels))
    kernel_b = pt.empty((emb_dim)).uniform_(*init_range(in_channels))

x_out = F.DiT(
    pooled_captions, captions, latent, timesteps,
    time_mlp_W1, time_mlp_b1, time_mlp_W2, time_mlp_b2,
    pooled_mlp_W1, pooled_mlp_b1, pooled_mlp_W2, pooled_mlp_b2,
    captions_W, captions_b,
    kernel_W, kernel_b,
    # others
    B, Tc, pooled_dim, captions_dim, in_channels, emb_dim, attn_heads, mlp_expand, 
    patch_size, pos_embed_max_size, base_height, use_dual_attention, use_kq_norm, discard_context
)

out = model.forward(latent, captions, pooled_captions, timesteps)

interact(local=locals())