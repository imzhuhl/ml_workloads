import torch
import torch.nn as nn
import sys
import logging

CAST = 'cast' in sys.argv


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
    
    def forward(self, x):
        x = self.fc1(x)
        return x
    

model = Net()
model.eval()

x = torch.randn((2, 64))

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

