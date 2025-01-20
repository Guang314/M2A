import torch
from torch.utils.tensorboard import SummaryWriter

# 初始化
n_x = 5
n_h = 8
n_y = 4
N_sample = 50
epsilon = 0.05
nb_epoch = 400

# 生成数据集
X = torch.randn(N_sample, n_x)
Y = torch.randn(N_sample, n_y)

# 初始化神经层函数 (手动)
linear_1 = torch.nn.Linear(n_x, n_h)
tanh = torch.nn.Tanh()
linear_2 = torch.nn.Linear(n_h, n_y)
mse = torch.nn.MSELoss()

# 使用 Adam 优化算法
optim_Adam = torch.optim.Adam(
    params = list(linear_1.parameters()) + list(linear_2.parameters()), 
    lr = epsilon
)
optim_Adam.zero_grad()

# 使用 optim 梯度下降
writer = SummaryWriter('M2A_AMAL/student_tp2/runs/Module_1')
for n_iter in range(nb_epoch):

    # Linear_1 层
    h_tilde = linear_1(X)

    # 激活层 (tanh)
    h = tanh(h_tilde)

    # Linear_2 层
    y_hat = linear_2(h)

    # 损失计算层
    loss = mse(y_hat, Y)

    # tensorboard --lodir runs/
    writer.add_scalar('Module_1 loss', loss, n_iter)

    # 直接输出
    print(f"Itération {n_iter}: loss {loss}")

    # 反向传播：计算梯度
    loss.backward()

    # 参数更新
    optim_Adam.step()

    # 梯度清零
    optim_Adam.zero_grad()

writer.close()
