from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Nombre de stations utilisé
CLASSES = 10

# Longueur des séquences
LENGTH = 20

# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2

# Taille du batch
BATCH_SIZE = 32

# PATH = "../data/"
PATH = "M2A_AMAL/student_tp3/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles

DIM_HIDDEN = 256

# 生成 RNN 模型
Model_RNN_m2m = RNN(
    input_dim=DIM_INPUT,
    latent_dim=DIM_HIDDEN,
    output_dim=DIM_INPUT
)

# GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model_RNN = Model_RNN_m2m.to(device)

# 定义损失函数
citerion = torch.nn.MSELoss()

# 定义优化器
optim_Adam = torch.optim.Adam(
    params=Model_RNN.parameters(),
    lr=0.001
)
optim_Adam.zero_grad()

# Tensorboard集成
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp3/runs/time_series/{n}")

# 训练与验证
num_epochs = 50

for epoch in range(num_epochs):

    Model_RNN.train()

    # 训练集
    train_loss = 0.0
    total = 0

    for data, labels in data_train:

        # 处理数据
        batch_size = data.size(0)
        data = data.permute(1,0,2,3).to(device)
        labels = labels.permute(1,0,2,3).to(device)
        h0 = torch.zeros(size=(batch_size, CLASSES, DIM_HIDDEN)).to(device)

        # 向前传播
        outputs, hidden_states = Model_RNN(data, h0)

        # 计算误差
        loss = citerion(outputs, labels)

        # 反向传播
        optim_Adam.zero_grad()
        loss.backward()
        optim_Adam.step()

        # 计算训练损失
        train_loss += loss.item() * batch_size
        total += batch_size

    train_loss /= total
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}")

    writer.add_scalar('Exo3_Training Loss', train_loss, epoch)

    # 验证集
    Model_RNN.eval()
    val_loss = 0.0
    total = 0

    with torch.no_grad():
        for data, labels in data_test:

            # 处理数据
            batch_size = data.size(0)
            data = data.permute(1,0,2,3).to(device)
            labels = labels.permute(1,0,2,3).to(device)

            # 前向传播
            h0 = torch.zeros(size=(batch_size, CLASSES, DIM_HIDDEN)).to(device)
            outputs, hidden_states = Model_RNN(data, h0)

            # 计算损失
            loss = citerion(outputs, labels)

            # 计算验证损失
            val_loss += loss.item() * batch_size
            total += batch_size

    val_loss /= total
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

    writer.add_scalar('Exo3_Validation Loss', val_loss, epoch)

writer.close()