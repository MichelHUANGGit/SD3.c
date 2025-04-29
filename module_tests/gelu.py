import torch
import custom_modules as F2

    
class GELU_CPP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = F2.gelu(input)
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        input_grad = F2.gelu_backward(output_grad, *ctx.saved_tensors)
        return input_grad


# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
torch.manual_seed(1)
x = torch.randn(B, T, C, requires_grad=True)
dout = torch.randn(B, T, C)

# PyTorch's version
out = torch.nn.functional.gelu(x, approximate="none")
fakeloss = (out * dout).sum()
fakeloss.backward()

dx = x.grad.clone()

# Our version
x.grad = None
out2 = GELU_CPP.apply(x)
fakeloss = (out2 * dout).sum()
fakeloss.backward()

print("out error:", (out2 - out).abs().max().item())
print("dx error:", (x.grad - dx).abs().max().item())



"""def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
with open('ln.bin', 'wb') as file:
    write(x, file) # (B, T, C)
    write(w, file) # (C, )
    write(b, file) # (C, )
    write(out, file) # (B, T, C)
    write(mean, file) # (B, T)
    write(rstd, file) # (B, T)
    write(dout, file) # (B, T, C)
    write(dx, file) # (B, T, C)
    write(dw, file) # (C, )
    write(db, file) # (C, )"""