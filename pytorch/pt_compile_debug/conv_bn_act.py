import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    

model = Net()
model.eval()

x = torch.randn((2, 3, 32, 32))

# options={'trace.enabled':True, 'trace.graph_diagram':True}
model = torch.compile(model)

with torch.no_grad():
    y = model(x)
    print(y.shape)
    

