import torch
import custom_modules as F2
from code import interact

class LayerNorm:

    @staticmethod
    def forward(x, w, b):
        B, T, C = x.size()
        eps = 1e-5
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        B,T,C = x.shape
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db
    
class LayerNormCPP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, eps:float=1e-5):
        C = input.size(-1)
        if weight is None:
            weight = torch.ones(1,1,C)
        if bias is None:
            bias = torch.zeros(1,1,C)
        output, mean, rstd = F2.layer_norm(input, weight, bias, eps, True)
        ctx.save_for_backward(input, weight, mean, rstd)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_grad, weight_grad, bias_grad = F2.layer_norm_backward(grad_output, *ctx.saved_tensors)
        return input_grad, weight_grad, bias_grad


if __name__ == "__main__":
    # create a small dummy example and check w.r.t PyTorch backward
    B = 2
    T = 3
    C = 4
    torch.manual_seed(1)
    x = torch.randn(B, T, C, requires_grad=True)
    w = torch.randn(C, requires_grad=True)
    b = torch.randn(C, requires_grad=True)
    dout = torch.randn(B, T, C)

    # PyTorch's version
    out = torch.nn.functional.layer_norm(x, (C,), w, b, 1e-5)
    fakeloss = (out * dout).sum()
    fakeloss.backward()

    dx = x.grad.clone()
    dw = w.grad.clone()
    db = b.grad.clone()

    # Our version
    x.grad = None
    w.grad = None
    b.grad = None
    out2 = LayerNormCPP.apply(x, w, b)
    fakeloss = (out2 * dout).sum()
    fakeloss.backward()

    print("out error:", (out2 - out).abs().max().item())
    print("dx error:", (x.grad - dx).abs().max().item())
    print("dw error:", (w.grad - dw).abs().max().item())
    print("db error:", (b.grad - db).abs().max().item())
