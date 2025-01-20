import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayLayer(nn.Module):
    def __init__(self, input_dim):
        super(HighwayLayer, self).__init__()
        
        # 变换部分 H(x) 使用线性层和激活函数
        self.H = nn.Linear(input_dim, input_dim)
        
        # 门控部分 T(x) 使用线性层 + sigmoid
        self.T = nn.Linear(input_dim, input_dim)
        
        # 初始化门控偏置，让网络更容易学习保持信息
        self.T.bias.data.fill_(-1)  # 偏置设为负值，确保初始时门控更倾向于传递输入

    def forward(self, x):
        H_out = F.relu(self.H(x))        # 变换后的值 H(x)
        T_out = torch.sigmoid(self.T(x)) # 门控值 T(x)
        
        # Highway 公式: y = H(x) * T(x) + x * (1 - T(x))
        out = H_out * T_out + x * (1 - T_out)
        return out

class HighwayNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(HighwayNetwork, self).__init__()
        
        # 初始的线性层，将输入映射到一个更高的维度
        self.initial_linear = nn.Linear(input_dim, input_dim)
        
        # 使用多个 Highway Layer
        self.highway_layers = nn.ModuleList([HighwayLayer(input_dim) for _ in range(num_layers)])
        
        # 最后一层线性层，将输出映射回所需的维度
        self.output_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        # 通过初始线性层
        out = F.relu(self.initial_linear(x))
        
        # 依次通过 Highway 层
        for highway_layer in self.highway_layers:
            out = highway_layer(out)
        
        # 输出层
        out = self.output_layer(out)
        return out

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='M2A_AMAL/student_tp2/data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建HighwayNetwork模型
input_dim = 28 * 28  # 输入图像是28x28
model = HighwayNetwork(input_dim=input_dim, num_layers=3)

# 使用 GPU 加速 (但是我没有)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard 集成
# 设置保存路径
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp2/runs/Highway/{n}")

# 训练循环
for epoch in range(5):  # 训练5个epochs
    model.train()
    for data, target in train_loader:
        data = data.view(data.size(0), -1)  # 将图片展平为向量
        data, target = data, target
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    writer.add_scalar('Highway Loss', loss, epoch)

writer.close()