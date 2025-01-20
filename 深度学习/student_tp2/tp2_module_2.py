import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 初始化
n_x = 5
n_h = 8
n_y = 4
N_sample = 50
epsilon = 0.05
nb_epoch = 400

X = torch.randn(N_sample, n_x)
Y = torch.randn(N_sample, n_y)

# 神经网络模型 (计算出预测值)
module_2 = nn.Sequential(
    nn.Linear(n_x, n_h),
    nn.Tanh(),
    nn.Linear(n_h, n_y),
)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optim_Adam = optim.Adam(                 # 使用 Adam 算法
    params = module_2.parameters(),      # .parameters()可以直接得到参数
    lr = epsilon
)
optim_Adam.zero_grad()

# 迭代梯度下降
writer = SummaryWriter('M2A_AMAL/student_tp2/runs/Module_2')

for n_iter in range(nb_epoch):

    # 计算预测值
    y_hat = module_2(X)

    # 计算损失
    loss = criterion(y_hat, Y)

    # tensorboard --lodir runs/
    writer.add_scalar('Module_2 loss', loss, n_iter)

    # 直接输出
    print(f"Itération {n_iter}: loss {loss}")

    # 反向传播：计算梯度
    loss.backward()

    # 更新参数
    optim_Adam.step()

    # 梯度清零
    optim_Adam.zero_grad()

writer.close()