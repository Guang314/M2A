import torch
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

# 定义神经网络类
class MonModuleLineare(torch.nn.Module):

    # 定义初始化函数
    def __init__(self):
        super(MonModuleLineare, self).__init__()
        # 神经层
        self.un = torch.nn.Linear(n_x, n_h)
        self.deux = torch.nn.Tanh()
        self.trois = torch.nn.Linear(n_h, n_y)

    # 定义前向传播函数
    def forward(self, x):

        # 依次运行计算
        prediction = self.trois(self.deux(self.un(x)))

        return prediction
    
# 生成模型
Module_3 = MonModuleLineare()

# 查看模型的参数
# print(list(Module_3.parameters()))

# 定义损失函数
citerion = torch.nn.MSELoss()

# 定义优化器
optim_Adam = torch.optim.Adam(
    params=Module_3.parameters(),
    lr=epsilon
)
optim_Adam.zero_grad()

# 迭代梯度下降
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp2/runs/Module_3/{n}")

for n_iter in range(nb_epoch):

    # 计算预测值
    y_hat = Module_3(X)

    # 计算损失函数
    loss = citerion(y_hat, Y)

    # tensorboard --lodir runs/
    writer.add_scalar('Module_3 loss', loss, n_iter)

    # 直接输出
    print(f"Itération {n_iter}: loss {loss}")

    # 反向传播：计算梯度
    loss.backward()

    # 更新参数
    optim_Adam.step()

    # 梯度清零
    optim_Adam.zero_grad()

writer.close()
