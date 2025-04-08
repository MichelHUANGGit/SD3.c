import torch as pt
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Tuple
from code import interact
from time import time
from math import log

class AdaLN_Zero(nn.Module):

    def __init__(self, emb_dim, chunks=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks = chunks
        self.linear = nn.Linear(emb_dim, chunks*emb_dim)
    
    def forward(self, y:Tensor) -> Tuple:
        y = self.linear(y)
        # create a Tokens dimension for broadcasting later, and chunk
        return y[:, None, :].chunk(self.chunks, dim=-1)

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels, base_height, patch_size, pos_embed_max_size=96, use_layer_norm=False, use_positional_embeddings=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.use_layer_norm = use_layer_norm
        self.use_positional_embeddings = use_positional_embeddings
        self.base_size = base_height // patch_size

        self.to_patch = nn.Conv2d(in_channels, out_channels, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        if use_layer_norm:
            self.lnorm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
        if use_positional_embeddings:
            self.register_buffer(
                "pos_enc_2D",
                positional_encoding_2D_HF(self.out_channels, pos_embed_max_size, pos_embed_max_size, self.base_size).unsqueeze(0).to(pt.float32),
                persistent=True
            )
    
    def crop_pos_enc(self, height, width):
        # retrieve 2D positional encoding save in buffer. (pos_embed_max_size * pos_embed_max_size, emb_dim)
        curr_pos_enc_2D = self.get_buffer("pos_enc_2D")
        # crop to (height * width, emb_dim) (where pos_embed_max_size >= height, width)
        offset_h = (self.pos_embed_max_size - height) // 2
        offset_w = (self.pos_embed_max_size - width) // 2
        cropped_pos_enc_2D = curr_pos_enc_2D \
            .view(self.pos_embed_max_size, self.pos_embed_max_size, self.out_channels) \
            [offset_h:offset_h+height, offset_w:offset_w+width, :] \
            .flatten(0, 1)
        return cropped_pos_enc_2D # (1, height*width, emb_dim)
        

    def forward(self, x:Tensor) -> Tensor:
        """x (B,C,H,W): noised latent"""
        
        patches = self.to_patch(x)
        B,E,h,w = patches.shape
        patches = patches.flatten(2).transpose(1,2) # (B,T,E) where T=h*w, E=emb_dim, h,w = height, width in patches (not pixels) resolution
        if self.use_layer_norm:
            patches = self.lnorm(patches)

        if self.use_positional_embeddings:
            pos_emb = self.crop_pos_enc(h, w)
            patches = patches + pos_emb # (B,T,E)
        return patches

def get_sinusoidal_embedding(t:Tensor, emb_dim:int) -> Tensor:
    # t tensor of dim (B,)
    t = t[:, None]
    log_10_000 = 9.2103403719761836
    half_dim =  emb_dim // 2
    frequencies = pt.exp(-log_10_000 * pt.arange(half_dim, dtype=pt.float32) / half_dim)[None, :].to(t.device)
    sinusoidal = pt.cat([pt.cos(t * frequencies), pt.sin(t * frequencies)], dim=1)
    return sinusoidal # (B,C)

def positional_encoding_2D_HF(emb_dim, height, width, base_size):
    """simplified huggingface implementation"""
    h = pt.arange(height, dtype=pt.float64) / (height / base_size)
    w = pt.arange(width, dtype=pt.float64) / (width / base_size)
    pos_h, pos_w = pt.meshgrid(h, w, indexing="xy")
    pos_h = pos_h.reshape(-1)
    pos_w = pos_w.reshape(-1)

    # Using torch.linspace would make more sense:
    # div_term = 1.0 / (10000.0 ** pt.linspace(0.0, 1.0, steps=emb_dim//4, dtype=pt.float64))
    # But in huggingface, this is how it's done:
    div_term = 1.0 / (10000.0 ** (pt.arange(emb_dim // 4, dtype=pt.float64) / (emb_dim / 4.0)))
    
    out = pt.cat([
        pt.sin(pos_h.outer(div_term)),
        pt.cos(pos_h.outer(div_term)),
        pt.sin(pos_w.outer(div_term)),
        pt.cos(pos_w.outer(div_term)),
    ], dim=-1)

    return out

def positional_encoding_2D(emb_dim, height, width):
    """Taken and adapted from https://github.com/wzlxjtu/PositionalEncoding2D"""
    if emb_dim % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(emb_dim))
    log_10_000 = 9.2103403719761836
    pe = pt.zeros(emb_dim, height, width)
    # Each dimension use half of d_model
    dim = emb_dim // 2
    # exp(-log(x)) = exp(log(1/x)) = 1/x
    # exp(-log(10000)) = 1/10000
    # exp(a * -log(10000/b)) = exp(log( (b/10000)**a ))
    div_term = pt.exp(pt.arange(0., dim, 2) * -(log_10_000 / dim))[:, None]
    pos_w = pt.arange(0., width)[None, :]
    pos_h = pt.arange(0., height)[None, :]
    pe[0:dim:2, :, :] = pt.sin(pos_w * div_term)[:, None, :].repeat(1, height, 1)
    pe[1:dim:2, :, :] = pt.cos(pos_w * div_term)[:, None, :].repeat(1, height, 1)
    pe[dim::2, :, :] = pt.sin(pos_h * div_term)[:, :, None].repeat(1, 1, width)
    pe[dim + 1::2, :, :] = pt.cos(pos_h * div_term)[:, :, None].repeat(1, 1, width)

    return pe


class TimestepEmbeddings(nn.Module):

    def __init__(self, timestep_emb_dim, out_emb_dim):
        super().__init__()
        self.timestep_emb_dim = timestep_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, out_emb_dim),
            nn.SiLU(),
            nn.Linear(out_emb_dim, out_emb_dim)
        )
    
    def forward(self, timesteps:Tensor) -> Tensor:
        temb = get_sinusoidal_embedding(timesteps, self.timestep_emb_dim)
        temb = self.mlp(temb)
        return temb


class Attention(nn.Module):
    """Self attention or cross-attention"""

    def __init__(self, attn_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_heads = attn_heads
    
    def forward(self, Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
        B,Tk,C = K.shape; Tq = Q.size(1)

        k = K.view(B, Tk, self.attn_heads, C//self.attn_heads).transpose(1,2)
        q = Q.view(B, Tq, self.attn_heads, C//self.attn_heads).transpose(1,2)
        v = V.view(B, Tk, self.attn_heads, C//self.attn_heads).transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v)
        
        return y.transpose(1,2).flatten(2, -1)
    
class MLP(nn.Module):

    def __init__(self, emb_dim, expand_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(emb_dim, emb_dim*expand_factor)
        self.gelu = nn.GELU("tanh")
        self.lin2 = nn.Linear(emb_dim*expand_factor, emb_dim)

    def forward(self, x:Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(pt.ones(dim))
            if bias:
                self.bias = nn.Parameter(pt.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(pt.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * pt.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [pt.float16, pt.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

class MM_DiT_Block(nn.Module):
    """https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/mmdit-x.png"""
    
    def __init__(self, emb_dim, attn_heads, mlp_expand, discard_context=False, use_dual_attention=True, use_kq_norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.emb_dim = emb_dim
        self.attn_heads = attn_heads
        assert emb_dim % attn_heads == 0, f"Embedding dimension (={emb_dim}) must be divisible by the number of attention heads (={attn_heads})"
        self.head_dim = emb_dim // attn_heads
        self.mlp_expand = mlp_expand
        self.discard_context = discard_context # whether or not to discard the context embeddings after the attention module, True for the last block
        self.use_dual_attention = use_dual_attention
        self.use_kq_norm = use_kq_norm

        # Context layers
        self.context_lnorm1 = nn.LayerNorm(emb_dim, eps=1e-6, elementwise_affine=False)
        self.context_ada_lnorm = AdaLN_Zero(emb_dim, chunks=6                           if not(discard_context) else 2)
        self.context_to_kqv = nn.Linear(emb_dim, 3*emb_dim)
        if use_kq_norm:
            self.context_rmsnorm_key = RMSNorm(self.head_dim, eps=1e-6)
            self.context_rmsnorm_query = RMSNorm(self.head_dim, eps=1e-6)
        self.context_attn_Wout = nn.Linear(emb_dim, emb_dim)                            if not(discard_context) else nn.Identity()
        self.context_lnorm2 = nn.LayerNorm(emb_dim, eps=1e-6, elementwise_affine=False) if not(discard_context) else nn.Identity()
        self.context_mlp = MLP(emb_dim, mlp_expand)                                     if not(discard_context) else nn.Identity()
        
        # latent vector layers
        self.latent_lnorm1 = nn.LayerNorm(emb_dim, eps=1e-6, elementwise_affine=False)
        self.latent_ada_lnorm = AdaLN_Zero(emb_dim, chunks=9 if use_dual_attention else 6) 
        self.latent_to_kqv = nn.Linear(emb_dim, 3*emb_dim)
        if use_kq_norm:
            self.latent_rmsnorm_key = RMSNorm(self.head_dim, eps=1e-6)
            self.latent_rmsnorm_query = RMSNorm(self.head_dim, eps=1e-6)
        self.latent_attn_Wout = nn.Linear(emb_dim, emb_dim)
        self.latent_lnorm2 = nn.LayerNorm(emb_dim, eps=1e-6, elementwise_affine=False)
        self.latent_mlp = MLP(emb_dim, mlp_expand)

        # if dual, extra latent layers
        if use_dual_attention:
            self.latent_to_kqv_x2 = nn.Linear(emb_dim, 3*emb_dim)
            if use_kq_norm:
                self.latent_rmsnorm_key_x2 = RMSNorm(self.head_dim, eps=1e-6)
                self.latent_rmsnorm_query_x2 = RMSNorm(self.head_dim, eps=1e-6)
            self.latent_attn_Wout_x2 = nn.Linear(emb_dim, emb_dim)
        
        self.B_id = 0
        self.T_id = 1
        self.C_id = 2


    def forward(self, x:Tensor, c:Tensor, y:Tensor):
        """ 
        x (B,Tx,C): noised latent (where Tx = the number of patches of the latent vector)
        c (B,Tc,C): context embeddings (where Tc = 77+77 tokens coming from CLIP and T5 text encoders in SD3.5)
        y (B,C): timestep+text mixed embeddings
        """
        B,Tc,C = c.shape; Tx = x.size(1)

        ##################################### Context embeddings c process before attention ####################################
        if not(self.discard_context):
            shift_attn_c, scale_attn_c, gate_attn_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.context_ada_lnorm(y)
        else:
            # NOTE: Scale and shift's are in swapped order on purpose to mimic exactly the huggingface implementation
            scale_attn_c, shift_attn_c = self.context_ada_lnorm(y)

        c_out = self.context_lnorm1(c) # (B,77+77,C)
        c_out = (1.0 + scale_attn_c) * c_out + shift_attn_c # scale, shift
        qc, kc, vc = self.context_to_kqv(c_out).chunk(3, dim=-1)
        self.register_buffer("tempc", qc)
        qc = qc.view(B, Tc, self.attn_heads, C//self.attn_heads)
        kc = kc.view(B, Tc, self.attn_heads, C//self.attn_heads)
        vc = vc.view(B, Tc, self.attn_heads, C//self.attn_heads)

        ##################################### Latent vector x1 (and x2 if dual) process before attention ####################################

        if self.use_dual_attention:
            shift_attn_x1, scale_attn_x1, gate_attn_x1, shift_mlp_x, scale_mlp_x, gate_mlp_x, shift_attn_x2, scale_attn_x2, gate_attn_x2 = self.latent_ada_lnorm(y)
        else:
            shift_attn_x1, scale_attn_x1, gate_attn_x1, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.latent_ada_lnorm(y)
        x_out = self.latent_lnorm1(x) # (B,Tx,C)

        # x1
        x1 = (1.0 + scale_attn_x1) * x_out + shift_attn_x1 # scale, shift
        qx, kx, vx = self.latent_to_kqv(x1).chunk(3, dim=-1)
        qx = qx.view(B, Tx, self.attn_heads, C//self.attn_heads)
        kx = kx.view(B, Tx, self.attn_heads, C//self.attn_heads)
        vx = vx.view(B, Tx, self.attn_heads, C//self.attn_heads)

        # Optional rmsnorm along head_dim (last axis)
        if self.use_kq_norm:
            kc = self.context_rmsnorm_key(kc)
            qc = self.context_rmsnorm_query(qc)
            kx = self.latent_rmsnorm_key(kx)
            qx = self.latent_rmsnorm_query(qx)

        ############################################### ATTENTION ####################################################

        # Concatenate latent and context embeddings for self+cross attention, along tokens dimension
        K = pt.concat([kc, kx], dim=self.T_id).transpose(1,2) # (B, attn_heads, Tx+Tc, C//attn_heads)
        Q = pt.concat([qc, qx], dim=self.T_id).transpose(1,2) # (B, attn_heads, Tx+Tc, C//attn_heads)
        V = pt.concat([vc, vx], dim=self.T_id).transpose(1,2) # (B, attn_heads, Tx+Tc, C//attn_heads)

        attn_output = F.scaled_dot_product_attention(Q, K, V) # self attention + cross attention between context and image
        attn_output = attn_output.transpose(1,2).flatten(2, -1) # (B,Tx+Tc,C)

        if not(self.discard_context):
            attn_output_c = attn_output[:,:Tc,:] # first Tc tokens
            attn_output_c = self.context_attn_Wout(attn_output_c)
            attn_output_c = attn_output_c * gate_attn_c # gating: the gamma_c vector
            c = c + attn_output_c  # residual

        attn_output_x1 = attn_output[:,Tc:,:] # remaining tokens
        attn_output_x1 = self.latent_attn_Wout(attn_output_x1) # Wout projection in the attention mechanism
        attn_output_x1 = attn_output_x1 * gate_attn_x1 # gating: the gamma_x1 vector
        x = x + attn_output_x1 # residual

        if self.use_dual_attention:
            x2 = (1.0 + scale_attn_x2) * x_out + shift_attn_x2
            qx2, kx2, vx2 = self.latent_to_kqv_x2(x2).chunk(3, dim=-1)
            qx2 = qx2.view(B, Tx, self.attn_heads, C//self.attn_heads).transpose(1,2)
            kx2 = kx2.view(B, Tx, self.attn_heads, C//self.attn_heads).transpose(1,2)
            vx2 = vx2.view(B, Tx, self.attn_heads, C//self.attn_heads).transpose(1,2)
            if self.use_kq_norm:
                qx2 = self.latent_rmsnorm_query_x2(qx2)
                kx2 = self.latent_rmsnorm_key_x2(kx2)

            attn_output_x2 = F.scaled_dot_product_attention(qx2, kx2, vx2) # self attention image
            attn_output_x2 = attn_output_x2.transpose(1,2).flatten(2, -1) # (B,Tx,C)
            attn_output_x2 = self.latent_attn_Wout_x2(attn_output_x2)
            attn_output_x2 = attn_output_x2 * gate_attn_x2
            x = x + attn_output_x2 # residual
        
        ##################################### Latent vector x1 (and x2 if dual) MLP ####################################

        x_out = self.latent_lnorm2(x)
        x_out = x_out * (1.0 + scale_mlp_x) + shift_mlp_x # shift scale
        x_out = self.latent_mlp(x_out) # MLP
        x_out = gate_mlp_x * x_out # gating: eta vector
        x_out = x_out + x # residual 

        ##################################### context embeddings c MLP ####################################

        if not(self.discard_context):
            c_out = self.context_lnorm2(c)
            c_out = c_out * (1.0 + scale_mlp_c) + shift_mlp_c # shift scale
            c_out = self.context_mlp(c_out) # MLP
            c_out = gate_mlp_c * c_out
            c_out = c_out + c
        else:
            c_out = None

        return c_out, x_out
    
class MMDiT(nn.Module):
    """Architecture: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/mmdit-x.png"""

    def __init__(self, num_layers, dual_attention_layers, in_dim, emb_dim, pooled_dim, captions_dim, patch_size, attn_heads, mlp_expand, pos_embed_max_size, base_height, use_kq_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.dual_attention_layers = dual_attention_layers
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.pooled_dim = pooled_dim
        self.captions_dim = captions_dim
        self.patch_size = patch_size
        self.attn_heads = attn_heads
        self.mlp_expand = mlp_expand
        self.use_kq_norm = use_kq_norm

        self.timestep_mlp = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.pooled_text_mlp = nn.Sequential(
            nn.Linear(pooled_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.context_linear = nn.Linear(captions_dim, emb_dim)
        self.to_patches = PatchEmbedding(in_dim, emb_dim, base_height, patch_size, pos_embed_max_size, use_positional_embeddings=True)

        self.transformer = nn.ModuleList([
            MM_DiT_Block(emb_dim, attn_heads, mlp_expand, discard_context=False, use_dual_attention=(layer_id in dual_attention_layers), use_kq_norm=use_kq_norm)
            for layer_id in range(num_layers-1)
        ])
        # last DiT block discards context embeddings mid way (after the attention)
        self.transformer.append(MM_DiT_Block(emb_dim, attn_heads, mlp_expand, discard_context=True, use_dual_attention=False, use_kq_norm=use_kq_norm))

        self.lnorm_out = nn.LayerNorm(emb_dim, elementwise_affine=False, bias=False, eps=1e-5)
        self.ada_ln_out = AdaLN_Zero(emb_dim, chunks=2)
        self.linear_out = nn.Linear(emb_dim, patch_size*patch_size*in_dim)

    def forward(self, noisy_latent:Tensor, coarse_captions:Tensor, pooled_captions:Tensor, timesteps:Tensor|float) -> Tensor:
        batch_size, in_dim, height, width = noisy_latent.shape
        if isinstance(timesteps, float):
            timesteps = pt.full((B,), fill_value=timesteps).to(noisy_latent.device)
        assert timesteps.dim() == 1

        timesteps_embeddings = get_sinusoidal_embedding(timesteps, emb_dim=256)
        timesteps_embeddings = self.timestep_mlp(timesteps_embeddings)
        pooled_text = self.pooled_text_mlp(pooled_captions)

        y = F.silu(timesteps_embeddings + pooled_text)
        c = self.context_linear(coarse_captions)
        x = self.to_patches(noisy_latent)

        self.register_buffer("x", x)

        for block in self.transformer:
            c, x = block(x, c, y)

        scale, shift = self.ada_ln_out(y) # NOTE: scale & shift are swapped to mimic HF
        x_out = self.lnorm_out(x)
        x_out = x_out * (1.0 + scale) + shift
        x_out = self.linear_out(x_out) # (B, Tx, patch_size * patch_size * in_dim), where Tx = number of patches

        # patch to pixels
        height_p, width_p = height//self.patch_size, width//self.patch_size # resolution in patches
        x_out = x_out.view(batch_size, height_p, width_p, self.patch_size, self.patch_size, in_dim)
        x_out = pt.einsum("bhwpqc->bchpwq", x_out) # (B, in_dim, height_p, patch_size, width_p, patch_size)
        x_out = x_out.flatten(2,3).flatten(3,4) # flatten height with patch dimension, and width with patch dimension

        return x_out
    
def print_params(mod: nn.Module):
    print(f"{mod.__class__.__name__} - Number of parameters {sum(p.numel() for p in mod.parameters()):,}")

def count_params(mod: nn.Module):
    return sum(p.numel() for p in mod.parameters())

def measure_performance(model, input, evals, warmups, return_output=True, **kwargs):

    for i in range(warmups):
        _ = model(*input, **kwargs)
    t0 = time()
    for i in range(evals):
        output = model(*input, **kwargs)
    pt.cuda.synchronize()
    dt = time() - t0
    print(f"{type(model).__name__}: {evals/dt} evals/s")

    if return_output: return output

if __name__ == "__main__":
    measure_performance(positional_encoding_2D, (128, 128, 128), evals=100, warmups=5, return_output=False)

    ####################### ATTENTION TEST ########################
    
    attn = Attention(2)
    # Tk and Tq can be different in a cross attention setting,
    # e.g. key = image embeddings, query = text embeddings, -> The text "attends" to the image
    B,Tk,Tq,C = 4,128,196,512
    Q = pt.randn((B,Tq,C))
    K = pt.randn((B,Tk,C))
    V = pt.randn_like(K)
    
    # Compare nn.mha
    mha = nn.MultiheadAttention(C, 2, 0.0, batch_first=True)
    # We assume K,Q,V projections are already done, so remove them
    mha.in_proj_weight.data = pt.cat([pt.eye(C), pt.eye(C),pt.eye(C)], 0)
    mha.out_proj.weight.data = pt.eye(C)

    attn_out = measure_performance(attn, (Q,K,V), evals=100, warmups=5)
    attn_out2, _ = mha.forward(Q,K,V, need_weights=False)
    
    assert pt.allclose(attn_out, attn_out2)
    del attn, mha, attn_out, attn_out2

    ###################### MLP TEST ##########################

    B,T,C = 4,128,512
    mlp = MLP(C, expand_factor=4)
    x = pt.randn((B,T,C))
    y = mlp(x)
    assert y.shape == (B,T,C)
    del x,y, mlp



    ############################################################## Full architecture TEST #####################################################################

    # Parameters
    B,H,W = 2, 64, 64
    Tc,C = 77+77,512 #context embeddings 77+77tokens, latent embeddings: 128 patches/tokens
    attn_heads = 2
    patch_size = 2
    num_patches = H*W // (patch_size**2)
    in_channels = 16
    captions_dim = 2048
    pooled_dim = 2048
    pos_embed_max_size = 160
    num_layers = 10
    dual_attention_layers = (0, 1, 2, 3, 4, 5)

    # Compare with official huggingface implementation
    from diffusers import SD3Transformer2DModel
    SD3 = SD3Transformer2DModel(
        num_layers=num_layers,
        in_channels=in_channels,
        num_attention_heads=attn_heads, # use same dim for comparison
        attention_head_dim=C//attn_heads, # use same dim for comparison
        pooled_projection_dim=pooled_dim,
        caption_projection_dim=C,
        joint_attention_dim=captions_dim,
        qk_norm=None,
        dual_attention_layers=dual_attention_layers,
        patch_size=patch_size,
        sample_size=H,
        pos_embed_max_size=pos_embed_max_size,
    )

    model = MMDiT(
        num_layers=num_layers,
        in_dim=in_channels,
        emb_dim=C,
        attn_heads=attn_heads,
        pooled_dim=pooled_dim,
        captions_dim=captions_dim,
        use_kq_norm=False,
        dual_attention_layers=dual_attention_layers,
        patch_size=patch_size,
        base_height=H,
        mlp_expand=4,
        pos_embed_max_size=pos_embed_max_size,
    )

    assert count_params(model) == count_params(SD3)
    original_params_count = count_params(SD3)

    # (NOTHING INTERESTING HERE) Copying exact weights for comparison (NOTHING INTERESTING HERE) (NOTHING INTERESTING HERE)
    # to patch
    SD3.pos_embed.proj.weight.data = model.to_patches.to_patch.weight.data.clone()
    SD3.pos_embed.proj.bias.data = model.to_patches.to_patch.bias.data.clone()

    # timestep embedding
    SD3.time_text_embed.timestep_embedder.linear_1.weight.data = model.timestep_mlp[0].weight.data.clone()
    SD3.time_text_embed.timestep_embedder.linear_1.bias.data = model.timestep_mlp[0].bias.data.clone()
    SD3.time_text_embed.timestep_embedder.linear_2.weight.data = model.timestep_mlp[2].weight.data.clone()
    SD3.time_text_embed.timestep_embedder.linear_2.bias.data = model.timestep_mlp[2].bias.data.clone()

    # pooled captions
    SD3.time_text_embed.text_embedder.linear_1.weight.data = model.pooled_text_mlp[0].weight.data.clone()
    SD3.time_text_embed.text_embedder.linear_1.bias.data = model.pooled_text_mlp[0].bias.data.clone()
    SD3.time_text_embed.text_embedder.linear_2.weight.data = model.pooled_text_mlp[2].weight.data.clone()
    SD3.time_text_embed.text_embedder.linear_2.bias.data = model.pooled_text_mlp[2].bias.data.clone()

    # coarse-grained captions (a.k.a. context or 'c')
    SD3.context_embedder.weight.data = model.context_linear.weight.data.clone()
    SD3.context_embedder.bias.data = model.context_linear.bias.data.clone()

    # DiT blocks
    for i in range(num_layers):
        dit_block_SD3 = SD3.transformer_blocks[i]
        dit_block_imp = model.transformer[i]

        # Ada layer norm
        dit_block_SD3.norm1.linear.weight.data = dit_block_imp.latent_ada_lnorm.linear.weight.data.clone()
        dit_block_SD3.norm1.linear.bias.data = dit_block_imp.latent_ada_lnorm.linear.bias.data.clone()
        dit_block_SD3.norm1_context.linear.weight.data = dit_block_imp.context_ada_lnorm.linear.weight.data.clone()
        dit_block_SD3.norm1_context.linear.bias.data = dit_block_imp.context_ada_lnorm.linear.bias.data.clone()


        # Wkqv, Wo latent vector x1
        dit_block_SD3.attn.to_q.weight.data = dit_block_imp.latent_to_kqv.weight.data[:C,:].clone()
        dit_block_SD3.attn.to_q.bias.data = dit_block_imp.latent_to_kqv.bias.data[:C].clone()
        dit_block_SD3.attn.to_k.weight.data = dit_block_imp.latent_to_kqv.weight.data[C:2*C,:].clone()
        dit_block_SD3.attn.to_k.bias.data = dit_block_imp.latent_to_kqv.bias.data[C:2*C].clone()
        dit_block_SD3.attn.to_v.weight.data = dit_block_imp.latent_to_kqv.weight.data[2*C:,:].clone()
        dit_block_SD3.attn.to_v.bias.data = dit_block_imp.latent_to_kqv.bias.data[2*C:].clone()
        dit_block_SD3.attn.to_out[0].weight.data = dit_block_imp.latent_attn_Wout.weight.data.clone()
        dit_block_SD3.attn.to_out[0].bias.data = dit_block_imp.latent_attn_Wout.bias.data.clone()

        # Wkqv, Wo context
        dit_block_SD3.attn.add_q_proj.weight.data = dit_block_imp.context_to_kqv.weight.data[:C,:].clone()
        dit_block_SD3.attn.add_q_proj.bias.data = dit_block_imp.context_to_kqv.bias.data[:C].clone()
        dit_block_SD3.attn.add_k_proj.weight.data = dit_block_imp.context_to_kqv.weight.data[C:2*C,:].clone()
        dit_block_SD3.attn.add_k_proj.bias.data = dit_block_imp.context_to_kqv.bias.data[C:2*C].clone()
        dit_block_SD3.attn.add_v_proj.weight.data = dit_block_imp.context_to_kqv.weight.data[2*C:,:].clone()
        dit_block_SD3.attn.add_v_proj.bias.data = dit_block_imp.context_to_kqv.bias.data[2*C:].clone()
        if dit_block_SD3.attn.to_add_out is not None:
            dit_block_SD3.attn.to_add_out.weight.data = dit_block_imp.context_attn_Wout.weight.data.clone()
            dit_block_SD3.attn.to_add_out.bias.data = dit_block_imp.context_attn_Wout.bias.data.clone()

        if i in dual_attention_layers:
            # Wkqv, x2
            dit_block_SD3.attn2.to_q.weight.data = dit_block_imp.latent_to_kqv_x2.weight.data[:C,:].clone()
            dit_block_SD3.attn2.to_q.bias.data = dit_block_imp.latent_to_kqv_x2.bias.data[:C].clone()
            dit_block_SD3.attn2.to_k.weight.data = dit_block_imp.latent_to_kqv_x2.weight.data[C:2*C,:].clone()
            dit_block_SD3.attn2.to_k.bias.data = dit_block_imp.latent_to_kqv_x2.bias.data[C:2*C].clone()
            dit_block_SD3.attn2.to_v.weight.data = dit_block_imp.latent_to_kqv_x2.weight.data[2*C:,:].clone()
            dit_block_SD3.attn2.to_v.bias.data = dit_block_imp.latent_to_kqv_x2.bias.data[2*C:].clone()
            dit_block_SD3.attn2.to_out[0].weight.data = dit_block_imp.latent_attn_Wout_x2.weight.data.clone()
            dit_block_SD3.attn2.to_out[0].bias.data = dit_block_imp.latent_attn_Wout_x2.bias.data.clone()

        # mlp
        dit_block_SD3.ff.net[0].proj.weight.data = dit_block_imp.latent_mlp.lin1.weight.data.clone()
        dit_block_SD3.ff.net[0].proj.bias.data = dit_block_imp.latent_mlp.lin1.bias.data.clone()
        dit_block_SD3.ff.net[2].weight.data = dit_block_imp.latent_mlp.lin2.weight.data.clone()
        dit_block_SD3.ff.net[2].bias.data = dit_block_imp.latent_mlp.lin2.bias.data.clone()
        if dit_block_SD3.ff_context is not None:
            dit_block_SD3.ff_context.net[0].proj.weight.data = dit_block_imp.context_mlp.lin1.weight.data.clone()
            dit_block_SD3.ff_context.net[0].proj.bias.data = dit_block_imp.context_mlp.lin1.bias.data.clone()
            dit_block_SD3.ff_context.net[2].weight.data = dit_block_imp.context_mlp.lin2.weight.data.clone()
            dit_block_SD3.ff_context.net[2].bias.data = dit_block_imp.context_mlp.lin2.bias.data.clone()

        # ada ln out
        SD3.norm_out.linear.weight.data = model.ada_ln_out.linear.weight.data.clone()
        SD3.norm_out.linear.bias.data = model.ada_ln_out.linear.bias.data.clone()
        SD3.proj_out.weight.data = model.linear_out.weight.data.clone()
        SD3.proj_out.bias.data = model.linear_out.bias.data.clone()

    assert count_params(SD3) == original_params_count


    device = pt.device("cuda")
    pt.manual_seed(1)
    latent = pt.randn((B, in_channels, H, W)).to(device)
    coarse_captions = pt.randn((B, Tc, captions_dim)).to(device)
    pooled_captions = pt.randn((B, pooled_dim)).to(device)
    timesteps = pt.rand((B,)).to(device)
    SD3.to(device).eval()
    model.to(device).eval()


    def forward_hook_SD3(self, args, kwargs):
        self.register_buffer("x", kwargs["hidden_states"].detach().cpu())
        self.register_buffer("c", kwargs["encoder_hidden_states"].detach().cpu())

    def forward_hook_dit(self, inputs):
        self.register_buffer("x", inputs[0].detach().cpu())
        self.register_buffer("c", inputs[1].detach().cpu())

    def forward_post_hook(self, foward_args, forward_kwargs, output):
        self.register_buffer("x_out", output[1].detach().cpu())

    for i in range(num_layers):
        SD3.transformer_blocks[i].register_forward_pre_hook(forward_hook_SD3, with_kwargs=True)
        model.transformer[i].register_forward_pre_hook(forward_hook_dit)

    with pt.no_grad():
        out_imp = measure_performance(model, (latent, coarse_captions, pooled_captions, timesteps), evals=50, warmups=10)
        out_SD3 = measure_performance(SD3, (latent, coarse_captions, pooled_captions, timesteps), evals=50, warmups=10)[0]

    a = SD3.transformer_blocks
    b = model.transformer

    for i in range(num_layers):
        assert a[i].x.shape == b[i].x.shape == (B, num_patches, C)
        assert a[i].c.shape == b[i].c.shape == (B, Tc, C)
        assert (a[i].x - b[i].x).abs().max().item() < 1e-5
        assert (a[i].c - b[i].c).abs().max().item() < 1e-5
        print(f"Block {i}: OK")

    assert out_imp.shape == out_SD3.shape == (B, in_channels, H, W)
    assert (out_SD3 - out_imp).abs().max().item() < 1e-4, interact(local=locals())

    # Gradients calculations
    out_imp = model(latent, coarse_captions, pooled_captions, timesteps)
    loss_imp = out_imp.sum()
    loss_imp.backward()
    first_layer_grad_imp = model.to_patches.to_patch.weight.grad

    out_SD3 = SD3(latent, coarse_captions, pooled_captions, timesteps)
    loss_SD3 = out_SD3[0].sum()
    loss_SD3.backward()
    first_layer_grad_SD3 = SD3.pos_embed.proj.weight.grad
    
    assert pt.allclose(first_layer_grad_SD3, first_layer_grad_imp, atol=1e-3), interact(local=locals())

    print("All tests passed successfully!")
    