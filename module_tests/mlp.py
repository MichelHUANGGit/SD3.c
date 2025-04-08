import torch as pt
from torch import nn, Tensor
import custom_modules_cpp


if __name__ == "__main__":
    B,T = 2,144
    C_in, C_hid, C_out = 256,1024,256
    activation = "GELU"
    kwargs = {"approximate" : "tanh"} if activation == "GELU" else {}

    lin1 = nn.Linear(C_in, C_hid)
    act = getattr(nn, activation)(**kwargs)
    # act = nn.GELU(approximate="tanh")
    lin2 = nn.Linear(C_hid, C_out)

    x = pt.randn((B, C_in))
    y_true = lin2(act(lin1(x)))


    W1 = lin1.weight.data.T.contiguous()
    b1 = lin1.bias.data
    W2 = lin2.weight.data.T.contiguous()
    b2 = lin2.bias.data

    y = custom_modules_cpp.mlp(x, W1, b1, W2, b2, activation, B, C_in, C_hid, C_out)
    # y = y.view((B, C_out))

    print("Max diff: ", (y - y_true).abs().max().item())

    # import code; code.interact(local=locals())