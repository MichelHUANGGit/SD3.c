import torch
import custom_modules_cpp as F2

class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        # Compute softmax
        y = F2.softmax(x, axis)
        ctx.save_for_backward(y, torch.tensor(axis))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, axis = ctx.saved_tensors
        axis = axis.item()

        # Compute gradient: g * y - y * (y^T g)
        grad_input = F2.softmax_backward(grad_output, y, axis)
        return grad_input, None  # Second None corresponds to 'dim' which does not require gradients

# Example usage
x = torch.randn((3,4,5), requires_grad=True)
dim = -1

y = SoftmaxFunction.apply(x, dim)
dout = torch.randn_like(y)
loss = (y * dout).sum()
loss.backward()
candidate_grad = x.grad.clone()

x.grad = None
y = torch.softmax(x, dim=dim)
loss = (y * dout).sum()
loss.backward()
true_grad = x.grad

print("True gradients: \n", true_grad)
print("Max Diff:", torch.max(torch.abs((true_grad - candidate_grad))).item())

from code import interact; interact(local=locals())