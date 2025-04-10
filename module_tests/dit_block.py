import torch as pt
from torch.nn.functional import silu
import custom_modules_cpp as F
from code import interact
from inspect import signature
from modules import MM_DiT_Block

# Parameters
B,H,W = 2, 64, 64
Tc, Tx = 77+77, 156
emb_dim = 256
attn_heads = 4
mlp_expand = 2
discard_context = False
use_dual_attention = False
use_kq_norm = True
# init_rstd = 0.01
init_range = (-emb_dim**-0.5, emb_dim**-0.5)
COPY_WEIGHTS = True

DiT_block = MM_DiT_Block(emb_dim, attn_heads, mlp_expand, discard_context, use_dual_attention, use_kq_norm)

x = pt.randn((B, Tx, emb_dim))
y = silu(pt.randn((B, emb_dim)))
c = pt.randn((B, Tc, emb_dim))
c_clone = c.clone()
x_clone = x.clone()
y_clone = y.clone()


if COPY_WEIGHTS:
    c_adalnormW = DiT_block.context_ada_lnorm.linear.weight.data.T.contiguous()
    c_adalnormb = DiT_block.context_ada_lnorm.linear.bias.data.clone()
    x_adalnormW = DiT_block.latent_ada_lnorm.linear.weight.data.T.contiguous()
    x_adalnormb = DiT_block.latent_ada_lnorm.linear.bias.data.clone()
    if use_kq_norm:
        rmsnorm_weight = DiT_block.latent_rmsnorm_key.weight.data.clone()
    else:
        rmsnorm_weight = pt.ones((emb_dim//attn_heads))

    # context
    c_Wqkv = DiT_block.context_to_kqv.weight.data.T.contiguous()
    c_bqkv = DiT_block.context_to_kqv.bias.data.clone()

    c_Wout = DiT_block.context_attn_Wout.weight.data.T.contiguous() if not discard_context else pt.empty((emb_dim, emb_dim))
    c_bout = DiT_block.context_attn_Wout.bias.data.clone() if not discard_context else pt.empty((emb_dim,))

    # latent
    x_Wqkv = DiT_block.latent_to_kqv.weight.data.T.contiguous()
    x_bqkv = DiT_block.latent_to_kqv.bias.data.clone()

    x_Wout = DiT_block.latent_attn_Wout.weight.data.T.contiguous()
    x_bout = DiT_block.latent_attn_Wout.bias.data.clone()

    # dual
    if use_dual_attention:
        x2_Wqkv = DiT_block.latent_to_kqv_x2.weight.data.T.contiguous()
        x2_bqkv = DiT_block.latent_to_kqv_x2.bias.data.clone()

        x2_Wout = DiT_block.latent_attn_Wout_x2.weight.data.T.contiguous()
        x2_bout = DiT_block.latent_attn_Wout_x2.bias.data.clone()
    else:
        x2_Wqkv = pt.empty((emb_dim, 3*emb_dim)).uniform_(*init_range)
        x2_Wout = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)

        x2_bqkv = pt.empty((3*emb_dim,)).uniform_(*init_range)
        x2_bout = pt.empty((emb_dim,)).uniform_(*init_range)

    # mlp
    if not discard_context:
        c_mlp_W1 = DiT_block.context_mlp.lin1.weight.data.T.contiguous()
        c_mlp_b1 = DiT_block.context_mlp.lin1.bias.data.clone()
        c_mlp_W2 = DiT_block.context_mlp.lin2.weight.data.T.contiguous()
        c_mlp_b2 = DiT_block.context_mlp.lin2.bias.data.clone()
    else:
        c_mlp_W1 = pt.empty((emb_dim, mlp_expand*emb_dim)).uniform_(*init_range)
        c_mlp_b1 = pt.empty((mlp_expand*emb_dim)).uniform_(*init_range)
        c_mlp_W2 = pt.empty((mlp_expand*emb_dim, emb_dim)).uniform_(-(mlp_expand*emb_dim)**-0.5, (mlp_expand*emb_dim)**-0.5)
        c_mlp_b2 = pt.empty((emb_dim)).uniform_(-(mlp_expand*emb_dim)**-0.5, (mlp_expand*emb_dim)**-0.5)

    mlp_W1 = DiT_block.latent_mlp.lin1.weight.data.T.contiguous()
    mlp_b1 = DiT_block.latent_mlp.lin1.bias.data.clone()
    mlp_W2 = DiT_block.latent_mlp.lin2.weight.data.T.contiguous()
    mlp_b2 = DiT_block.latent_mlp.lin2.bias.data.clone()


else:
    context_chunks = 6 if not discard_context else 2

    c_adalnormW = pt.empty((emb_dim, context_chunks*emb_dim)).uniform_(*init_range)
    c_adalnormb = pt.empty((context_chunks*emb_dim)).uniform_(*init_range)
    chunks = 9 if use_dual_attention else 6
    x_adalnormW = pt.empty((emb_dim, chunks*emb_dim)).uniform_(*init_range)
    # print([pt.max(chunk).item() for chunk in x_adalnormW.chunk(chunks, -1)])
    x_adalnormb = pt.empty((chunks*emb_dim)).uniform_(*init_range)

    rmsnorm_weight = pt.ones((emb_dim//attn_heads))


    c_Wq = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    c_Wk = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    c_Wv = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    c_Wo = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)

    c_bq = pt.empty((emb_dim,)).uniform_(*init_range)
    c_bk = pt.empty((emb_dim,)).uniform_(*init_range)
    c_bv = pt.empty((emb_dim,)).uniform_(*init_range)
    c_bo = pt.empty((emb_dim,)).uniform_(*init_range)

    Wq = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    Wk = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    Wv = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    Wo = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)

    bq = pt.empty((emb_dim,)).uniform_(*init_range)
    bk = pt.empty((emb_dim,)).uniform_(*init_range)
    bv = pt.empty((emb_dim,)).uniform_(*init_range)
    bo = pt.empty((emb_dim,)).uniform_(*init_range)
    # if use_dual_attention:
    Wq2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    Wk2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    Wv2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
    Wo2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)

    bq2 = pt.empty((emb_dim,)).uniform_(*init_range)
    bk2 = pt.empty((emb_dim,)).uniform_(*init_range)
    bv2 = pt.empty((emb_dim,)).uniform_(*init_range)
    bo2 = pt.empty((emb_dim,)).uniform_(*init_range)

    c_mlp_W1 = pt.empty((emb_dim, mlp_expand*emb_dim)).uniform_(*init_range)
    c_mlp_b1 = pt.empty((mlp_expand*emb_dim)).uniform_(*init_range)
    c_mlp_W2 = pt.empty((mlp_expand*emb_dim, emb_dim)).uniform_(-(mlp_expand*emb_dim)**-0.5, (mlp_expand*emb_dim)**-0.5)
    c_mlp_b2 = pt.empty((emb_dim)).uniform_(-(mlp_expand*emb_dim)**-0.5, (mlp_expand*emb_dim)**-0.5)

    mlp_W1 = pt.empty((emb_dim, mlp_expand*emb_dim)).uniform_(*init_range)
    mlp_b1 = pt.empty((mlp_expand*emb_dim)).uniform_(*init_range)
    mlp_W2 = pt.empty((mlp_expand*emb_dim, emb_dim)).uniform_(-(mlp_expand*emb_dim)**-0.5, (mlp_expand*emb_dim)**-0.5)
    mlp_b2 = pt.empty((emb_dim)).uniform_(-(mlp_expand*emb_dim)**-0.5, (mlp_expand*emb_dim)**-0.5)

with pt.no_grad():
    c_out_true, x_out_true = DiT_block.forward(x, c, y)


c_out, x_out = F.Dit_block(
    y, c, x, 
    # context
    c_adalnormW, c_adalnormb,
    c_Wqkv, c_bqkv,
    rmsnorm_weight, rmsnorm_weight,
    c_Wout, c_bout,
    c_mlp_W1, c_mlp_b1, c_mlp_W2, c_mlp_b2,
    # latent
    x_adalnormW, x_adalnormb,
    x_Wqkv, x_bqkv,
    rmsnorm_weight, rmsnorm_weight, 
    x_Wout, x_bout,
    # dual latent
    x2_Wqkv, x2_bqkv,
    rmsnorm_weight, rmsnorm_weight,
    x2_Wout, x2_bout,
    mlp_W1, mlp_b1, mlp_W2, mlp_b2,
    # others
    B, Tc, Tx, emb_dim, attn_heads, mlp_expand, use_dual_attention, use_kq_norm, discard_context
)

d = DiT_block


if not discard_context:
    print("c_out Max diff: ", (c_out - c_out_true).abs().max().item())
print("x_out Max diff: ", (x_out - x_out_true).abs().max().item())
interact(local=locals())