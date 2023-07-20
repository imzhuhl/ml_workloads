import torch
import torch.nn as nn
import sys
import logging


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class Net(nn.Module):
    def __init__(self, intermediate_size, embed_dim):
        super().__init__()
        embed_dim = 256
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
    

embed_dim = 256
model = Net(512, embed_dim)
model.eval()

x = torch.randn((2, 50, embed_dim))

# torch._logging.set_logs(aot_graphs=True)

model = torch.compile(model)

model.to(torch.bfloat16)
x = x.to(torch.bfloat16)

with torch.no_grad():
    y = model(x)
    

# if CAST:
#     with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
#         with torch.no_grad():
#             y = model(x)
#             print(y.shape)
# else:
#     with torch.no_grad():
#         y = model(x)
#         print(y.shape)

