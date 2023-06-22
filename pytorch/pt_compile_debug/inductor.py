
import torch
import torch.nn as nn
import torch._dynamo as torchdynamo

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


def fn(x):
    return torch.softmax(x, -1)

# @torchdynamo.optimize()
# def addrelu(a, b):
#     return torch.relu(torch.add(a, b))

# print(addrelu(torch.randn(128, 128), torch.randn(128, 128)))

torch._inductor.config.debug = True

model = Net().eval()
model = torch.compile(model)

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    for _ in range(3):
        y = model(x)

# print(y.shape)

# opt_fn = torch.compile(fn)
# x = torch.randn(2048, 1024)
# y = opt_fn(x)



