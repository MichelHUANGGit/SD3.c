import torch as pt
import custom_modules_cpp as F
from code import interact
from inspect import signature
from modules import MM_DiT_Block

# Parameters
B,H,W = 2, 64, 64
Tc, Tx = 77+77, 128
emb_dim = 512
attn_heads = 2
mlp_expand = 4
discard_context = False
use_dual_attention = True
use_kq_norm = True
# init_rstd = 0.01
init_range = (-emb_dim**-0.5, emb_dim**-0.5)
COPY_WEIGHTS = True

DiT_block = MM_DiT_Block(emb_dim, attn_heads, mlp_expand, discard_context, use_dual_attention, use_kq_norm)

x = pt.randn((B, Tx, emb_dim))
y = pt.randn((B, emb_dim))
c = pt.randn((B, Tc, emb_dim))
c_clone = c.clone()
x_clone = x.clone()
y_clone = y.clone()

# interact(local=locals())

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
    c_Wq = DiT_block.context_to_kqv.weight.data[:emb_dim, :].T.contiguous()
    c_Wk = DiT_block.context_to_kqv.weight.data[emb_dim:2*emb_dim, :].T.contiguous()
    c_Wv = DiT_block.context_to_kqv.weight.data[2*emb_dim:, :].T.contiguous()
    c_Wo = DiT_block.context_attn_Wout.weight.data.T.contiguous() if not discard_context else pt.empty((emb_dim, emb_dim))

    c_bq = DiT_block.context_to_kqv.bias.data[:emb_dim].clone()
    c_bk = DiT_block.context_to_kqv.bias.data[emb_dim:2*emb_dim].clone()
    c_bv = DiT_block.context_to_kqv.bias.data[2*emb_dim:].clone()
    c_bo = DiT_block.context_attn_Wout.bias.data.clone() if not discard_context else pt.empty((emb_dim,))

    # latent
    Wq = DiT_block.latent_to_kqv.weight.data[:emb_dim, :].T.contiguous()
    Wk = DiT_block.latent_to_kqv.weight.data[emb_dim:2*emb_dim, :].T.contiguous()
    Wv = DiT_block.latent_to_kqv.weight.data[2*emb_dim:, :].T.contiguous()
    Wo = DiT_block.latent_attn_Wout.weight.data.T.contiguous()

    bq = DiT_block.latent_to_kqv.bias.data[:emb_dim].clone()
    bk = DiT_block.latent_to_kqv.bias.data[emb_dim:2*emb_dim].clone()
    bv = DiT_block.latent_to_kqv.bias.data[2*emb_dim:].clone()
    bo = DiT_block.latent_attn_Wout.bias.data.clone()

    # dual
    if use_dual_attention:
        Wq2 = DiT_block.latent_to_kqv_x2.weight.data[:emb_dim, :].T.contiguous()
        Wk2 = DiT_block.latent_to_kqv_x2.weight.data[emb_dim:2*emb_dim, :].T.contiguous()
        Wv2 = DiT_block.latent_to_kqv_x2.weight.data[2*emb_dim:, :].T.contiguous()
        Wo2 = DiT_block.latent_attn_Wout_x2.weight.data.T.contiguous()

        bq2 = DiT_block.latent_to_kqv_x2.bias.data[:emb_dim].clone()
        bk2 = DiT_block.latent_to_kqv_x2.bias.data[emb_dim:2*emb_dim].clone()
        bv2 = DiT_block.latent_to_kqv_x2.bias.data[2*emb_dim:].clone()
        bo2 = DiT_block.latent_attn_Wout_x2.bias.data.clone()
    else:
        Wq2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
        Wk2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
        Wv2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)
        Wo2 = pt.empty((emb_dim, emb_dim)).uniform_(*init_range)

        bq2 = pt.empty((emb_dim,)).uniform_(*init_range)
        bk2 = pt.empty((emb_dim,)).uniform_(*init_range)
        bv2 = pt.empty((emb_dim,)).uniform_(*init_range)
        bo2 = pt.empty((emb_dim,)).uniform_(*init_range)

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
    c_adalnormW = pt.empty((emb_dim, 6*emb_dim)).uniform_(*init_range)
    c_adalnormb = pt.empty((6*emb_dim)).uniform_(*init_range)
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
    c_Wq, c_Wk, c_Wv, c_bq, c_bk, c_bv,
    rmsnorm_weight, rmsnorm_weight,
    c_Wo, c_bo,
    c_mlp_W1, c_mlp_b1, c_mlp_W2, c_mlp_b2,
    # latent
    x_adalnormW, x_adalnormb,
    Wq, Wk, Wv, bq, bk, bv,
    Wq2, Wk2, Wv2, bq2, bk2, bv2,
    rmsnorm_weight, rmsnorm_weight, rmsnorm_weight, rmsnorm_weight,
    Wo, bo, Wo2, bo2,
    mlp_W1, mlp_b1, mlp_W2, mlp_b2,
    # others
    B, Tc, Tx, emb_dim, attn_heads, mlp_expand, use_dual_attention, use_kq_norm, discard_context
)

d = DiT_block


if not discard_context:
    print("c_out Max diff: ", (c_out - c_out_true).abs().max().item())
print("x_out Max diff: ", (x_out - x_out_true).abs().max().item())
interact(local=locals())