import torch
import torch.nn as nn
import torch.optim as optim

# 定义自动编码器类
class Autoencoder(nn.Module):
    
    # 定义初始化函数
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()

        # 编码器部分：线性层 -> Relu
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 解码器部分：线性层(权重为编码器的转置) -> Sigmoid
        # 将会在后面手动共享权重
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码阶段：输入 -> 线性变换 -> ReLU
        encoded = self.relu(self.encoder(x))

        # 解码阶段：编码结果 -> 线性变换 -> Sigmoid
        decoded = self.sigmoid(self.decoder(encoded))

        return decoded
    
    def tie_weights(self):
        # 将解码器权重设置为编码器权重的转置
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())


# 定义网络的输入维度和隐藏层维度
input_dim = 784              # 对于 MNIST, 28x28 的图像, 784 维
hidden_dim = 64

# 创建自动编码器实例
autoencoder = Autoencoder(input_dim, hidden_dim)

# 调用共享权重函数，使得解码器权重为编码器权重的转置
autoencoder.tie_weights()

# 打印网络结构，验证权重是否共享
print(autoencoder)

# 测试一下网络是否正常工作
x = torch.randn((1, input_dim))
output = autoencoder(x)

print(x)

print("Input shape: ", x.shape)
print("Output shape: ", output.shape)

print(output)