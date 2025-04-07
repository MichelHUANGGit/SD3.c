import torch as pt
from torch import Tensor
import custom_modules_cpp as F2


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


if __name__ == "__main__":
    B, C, H, W = 16, 256, 40, 30
    t = pt.rand((B,))
    temb = get_sinusoidal_embedding(t, C)
    temb2 = F2.sinusoidal_embedding(t, C)
    print(f"sinusodial embeddings diff: {pt.norm(temb - temb2)}")

    PE = positional_encoding_2D_HF(C, H, W, 1)
    PE2 = F2.positional_embedding(C, H, W, 1)
    print(f"Positional embeddings diff: {pt.norm(PE - PE2)}")
    import code; code.interact(local=locals())