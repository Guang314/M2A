from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Nombre de stations utilisé 站点数量
CLASSES = 10

# Longueur des séquences 序列长度
LENGTH = 20

# Dimension de l'entrée (1 (in) ou 2 (in/out)) 输入维度
DIM_INPUT = 2

# Taille du batch 批次大小
BATCH_SIZE = 32

# PATH = "../data/"
PATH = "M2A_AMAL/student_tp3/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)



#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

# 初始化
DIM_HIDDEN = 256
softmax = torch.nn.Softmax()

# 生成 RNN 模型
Model_RNN = RNN(
    input_dim=DIM_INPUT,
    latent_dim=DIM_HIDDEN,
    output_dim=CLASSES
)

# 使用 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model_RNN = Model_RNN.to(device)

# 定义损失函数
citerion = torch.nn.CrossEntropyLoss()

# 定义优化器
optim_Adam = torch.optim.Adam(
    params=Model_RNN.parameters(),
    lr=0.001
)
optim_Adam.zero_grad()

# Tensorboard集成
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp3/runs/HZ_metro/{n}")

# 训练与验证
num_epochs = 50
for epoch in range(num_epochs):

    Model_RNN.train()
    
    # 训练集
    train_loss = 0.0
    correct = 0
    total = 0

    for data, labels in data_train:

        # 处理数据
        batch_size = labels.size(0)
        data, labels = data.permute(1,0,2).to(device), labels.to(device)       # DataLoader读取的是 batch*length*dim , 而 Model 里面是 length*batch*dim
        h0 = torch.zeros(batch_size, DIM_HIDDEN).to(device)

        # Forward Pass
        outputs, hidden_states = Model_RNN(data, h0)
        outputs = outputs[-1]

        loss = citerion(outputs, labels)

        # Backward Pass
        optim_Adam.zero_grad()
        loss.backward()
        optim_Adam.step()

        # 计算训练损失和准确率
        train_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()

    train_loss /= total
    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

    writer.add_scalar('Exo2_Training Loss', train_loss, epoch)
    writer.add_scalar('Exo2_Training Accuracy', train_accuracy, epoch)

    # 验证集
    Model_RNN.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data,labels in data_test:
            
            # 处理数据
            batch_size = labels.size(0)
            data, labels = data.permute(1,0,2).to(device), labels.to(device)       # DataLoader读取的是 batch*length*dim , 而 Model 里面是 length*batch*dim

            # Forward Pass 计算损失
            h0 = torch.zeros(batch_size, DIM_HIDDEN).to(device)
            outputs, hidden_states = Model_RNN(data, h0)   
            outputs = outputs[-1]

            loss = citerion(outputs, labels)

            # 计算训练损失
            val_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()


    val_loss /= total
    val_accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    writer.add_scalar('Exo2_Validation Loss', val_loss, epoch)
    writer.add_scalar('Exo2_Validation Accuracy', val_accuracy, epoch)

writer.close()




