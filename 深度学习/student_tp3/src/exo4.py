import string
import numpy as np
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO
import torch.nn.functional as F

# 路径
PATH = "M2A_AMAL/student_tp3/data/"

# 超参数
input_dim = len(LETTRES) + 1
latent_dim = 128
output_dim = len(LETTRES) + 1

# 载入数据集
dataset = TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000)
data_trump = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

# 初始化模型、损失函数和优化器
model = RNN(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# Tensorboard集成
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp3/runs/Trump_speech/{n}")

# 训练循环
def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            # 将输入和标签移动到设备上
            x_batch = F.one_hot(x_batch.to(device), num_classes=input_dim).float()
            y_batch = y_batch.to(device)

            # 初始化隐藏状态
            h = torch.zeros(x_batch.size(1), latent_dim).to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs, _ = model(x_batch, h)

            # 调整输出和目标形状以适应 CrossEntropyLoss
            outputs = outputs.view(-1, output_dim)
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算损失
            epoch_loss += loss

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}")

# 定义生成文本的函数
def generate_text(model, start_str, gen_length=50):
    model.eval()      # 设置模型为评估模式
    generate_text = start_str
    input_seq = string2code(generate_text).unsqueeze(0).to(device)

    h = torch.zeros(1, latent_dim).to(device)

    for _ in range(gen_length):
        x = F.one_hot(input_seq, num_classes=input_dim).float().to(device)
        output, h = model(x, h)

        # 从输出中选取最可能的字符的索引
        output_char_idx = torch.argmax(output[-1], dim=-1).item()

        # 终止符号
        if output_char_idx == 0:
            break

        # 将输出字符添加到生成文本中
        generate_text += id2lettre[output_char_idx]

        # 更新输出序列，使用生成的字符
        output_char_idx_tensor = torch.tensor([[output_char_idx]]).to(device)
        input_seq = torch.cat([input_seq, output_char_idx_tensor], dim=-1)

    return generate_text


# 训练模型
train_model(model=model, dataloader=data_trump, optimizer=optimizer, criterion=criterion, num_epochs=10)


# 训练完成后保存模型
torch.save(model.state_dict(), 'trump_rnn_model.pth')

# 使用示例
start_text = "I will make America great again"
generate_word = generate_text(model, start_text, gen_length=100)
print(generate_word)