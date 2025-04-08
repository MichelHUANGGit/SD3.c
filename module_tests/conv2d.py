import torch as pt
from torch.nn import functional as F
import custom_modules_cpp as F2
from code import interact
# pt.set_printoptions(precision=8, sci_mode=False)

def get_size(H, W, kernel_size, stride, padding):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    H_out = (H + 2*padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (W + 2*padding[1] - kernel_size[0]) // stride[0] + 1
    return H_out, W_out

def main(B, C_in, C_out, H, W, kernel_size, stride, padding):
    # Tensors and modules
    x = pt.randn((B,C_in,H,W))
    kernel = pt.randn((C_out, C_in, *kernel_size))
    bias = pt.randn((C_out,))
    # conv2d = pt.nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
    # conv2d.weight.data = kernel.clone()
    # conv2d.bias.data = bias.clone()


    out = F.conv2d(x, kernel, bias, stride, padding)
    out2 = F2.conv2d(x, kernel, bias, stride, padding)

    # out3 = conv2d(x)

    print((out - out2).abs().max().item())
    # print(pt.norm(out2 - out3))
    # print(pt.norm(out - out3))


if __name__ == "__main__":

    # Parameters
    main(
        B=2, C_in=1, C_out=1, H=32, W=32, kernel_size=(2,3), stride=1, padding=0
    )
    main(
        B=2, C_in=3, C_out=1, H=32, W=32, kernel_size=(4,2), stride=1, padding=0
    )
    main(
        B=2, C_in=1, C_out=1, H=32, W=32, kernel_size=(3,3), stride=1, padding=3
    )
    main(
        B=2, C_in=1, C_out=2, H=6, W=6, kernel_size=(3,3), stride=2, padding=1
    )
    main(
        B=2, C_in=16, C_out=32, H=128, W=128, kernel_size=(3,5), stride=2, padding=2
    )