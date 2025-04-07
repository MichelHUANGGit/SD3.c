import torch
import torch.nn.functional as F

# Input tensor
N, Cin, H, W = 1, 3, 5, 5  # Batch, Input channels, Height, Width
Cout, Kh, Kw = 2, 3, 3  # Output channels, Kernel height, Kernel width

X = torch.randn(N, Cin, H, W, requires_grad=True)
K = torch.randn(Cout, Cin, Kh, Kw, requires_grad=True)
b = torch.randn(Cout, requires_grad=True)

# Standard convolution
Y = F.conv2d(X, K, b)
Y.retain_grad()
loss = Y.sum()
loss.backward()

# Compute transposed convolution (same as gradient w.r.t input)
Y_grad = torch.ones_like(Y)  # Simulating dL/dY
dX = F.conv_transpose2d(Y_grad, K)  # This is equivalent to ∂L/∂X

print("Shape of dX (Gradient w.r.t. Input X):", dX.shape)

import code; code.interact(local=locals())