import torch
import custom_modules_cpp as F2
from code import interact
    
class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        output, rms = F2.rms_norm(input, weight, 1e-5, True)
        ctx.save_for_backward(input, rms, weight)
        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        input, rms, weight = ctx.saved_tensors
        input_grad, weight_grad = F2.rms_norm_backward(output_grad, input, rms, weight)
        return input_grad, weight_grad


if __name__ == "__main__":
    # create a small dummy example and check w.r.t PyTorch backward
    B,T,C = 2,3,4
    torch.manual_seed(1)
    x = torch.randn(B, T, C, requires_grad=True)
    w = torch.randn((C,), requires_grad=True)
    dout = torch.randn(B, T, C)

    # PyTorch's version
    out = torch.nn.functional.rms_norm(x, (C,), w, 1e-5)
    fakeloss = (out * dout).sum()
    fakeloss.backward()

    dx = x.grad.clone()
    dw = w.grad.clone()

    # Our version
    x.grad = None
    w.grad = None
    out2 = RMSNorm.apply(x, w)
    fakeloss = (out2 * dout).sum()
    fakeloss.backward()

    print("out error:", (out2 - out).abs().max().item())
    print("dx error:", (x.grad - dx).abs().max().item())
    print("dw error:", (w.grad - dw).abs().max().item())