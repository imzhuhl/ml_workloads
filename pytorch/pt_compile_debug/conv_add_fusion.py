import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.add(x1, x2)
        return x
    

model = Net()
# model.eval()

model = torch.compile(model)

with torch.no_grad():
    x = torch.randn((2, 3, 32, 32))
    for _ in range(3):
        y = model(x)
    print(y.shape)
    

