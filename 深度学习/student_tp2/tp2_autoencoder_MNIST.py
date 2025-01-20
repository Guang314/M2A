import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os

# 预数据处理：归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # 对灰度值进行标准化
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='M2A_AMAL/student_tp2/data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='M2A_AMAL/student_tp2/data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 定义自动编码器类
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder部分
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),     # 输入 28x28 的图像被展平为 784 维向量
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),         # 将特征压缩到32维
            nn.ReLU()
        )

        # Decoder部分
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()               # 输出值在[0,1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# 使用 GPU 加速 (但是我没有)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TensorBoard 集成
# 设置保存路径
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp2/runs/auto_MNIST/{n}")

# 模型检查点 (Checkpointing)

# 保存训练过程中最好的模型
def save_checkpoint(model, epoch, loss, path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, f"{path}/checkpoint_epoch_{epoch}.pth")


# 训练与验证循环
num_epochs = 15

for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
 
    # 平均训练损失
    train_loss /= len(train_loader)
    
    # 验证
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()

    val_loss /= len(test_loader)

    # 打印并记录损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)

    # 保存模型检查点
    save_checkpoint(model, epoch+1, val_loss)

writer.close()