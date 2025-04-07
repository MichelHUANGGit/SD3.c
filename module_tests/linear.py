import torch
import custom_modules_cpp as F2

    
class Linear_CPP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input:torch.Tensor, weight:torch.Tensor, bias:torch.Tensor):
        output = F2.linear(input, weight, bias)

        # pass these tensors for backward
        ctx.save_for_backward(input, weight)
        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        input, weight = ctx.saved_tensors
        
        input_grad, weight_grad, bias_grad = F2.linear_backward(output_grad, input, weight)
        return input_grad, weight_grad, bias_grad


if __name__ == "__main__":
    # create a small dummy example and check w.r.t PyTorch backward
    B = 2
    T = 4
    C_in, C_out = 512,128
    torch.manual_seed(1)
    x = torch.randn(B, T, C_in, requires_grad=True)
    W = torch.randn((C_in, C_out), requires_grad=True)
    b = torch.zeros((C_out,), requires_grad=True)


    # PyTorch's version
    out = torch.nn.functional.linear(x, W.T, b)
    dout = torch.randn_like(out)
    fakeloss = (out * dout).sum()
    fakeloss.backward()

    dx = x.grad.clone()
    dW = W.grad.clone()
    db = b.grad.clone()

    # Our version
    x.grad = None
    W.grad = None
    b.grad = None
    out2 = Linear_CPP.apply(x, W, b)
    fakeloss = (out2 * dout).sum()
    fakeloss.backward()

    print("out error:", (out2 - out).abs().max().item())
    print("dx error:", (x.grad - dx).abs().max().item())
    print("dW error:", (W.grad - dW).abs().max().item())
    print("db error:", (b.grad - db).abs().max().item())