import torch
import custom_modules_cpp as F2

    
class SILU_CPP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = F2.silu(input)
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        input_grad = F2.silu_backward(output_grad, *ctx.saved_tensors)
        return input_grad


if __name__ == "__main__":
    # create a small dummy example and check w.r.t PyTorch backward
    B,T,C = 2,3,128
    torch.manual_seed(1)
    x = torch.randn(B, T, C, requires_grad=True)
    dout = torch.randn(B, T, C)

    # PyTorch's version
    out = torch.nn.functional.silu(x)
    fakeloss = (out * dout).mean()
    fakeloss.backward()

    dx = x.grad.clone()

    # Our version
    x.grad = None
    out2 = SILU_CPP.apply(x)
    fakeloss = (out2 * dout).mean()
    fakeloss.backward()

    print("out error:", (out2 - out).abs().max().item())
    print("dx error:", (x.grad - dx).abs().max().item())
