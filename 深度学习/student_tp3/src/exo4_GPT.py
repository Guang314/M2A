import string
import numpy as np
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN_exo4, device

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



#  TODO: (ChatGPT做的, 有问题)
import torch.nn.functional as F

# 路径
PATH = "M2A_AMAL/student_tp3/data/"

# 超参数
embedding_dim = 128  # 嵌入维度
hidden_dim = 256     # 隐藏层维度
num_epochs = 20      # 训练轮数
batch_size = 64      # 批次大小
learning_rate = 0.001 # 学习率

# 载入数据集
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = RNN_exo4(vocab_size=len(lettre2id), embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# Tensorboard集成
n = str(torch.randint(1,1000,(1,)).item())
writer = SummaryWriter(f"M2A_AMAL/student_tp3/runs/Trump_speech/{n}")

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(data_trump):
        inputs, targets = inputs.to(device), targets.to(device)  # 移动到设备
        optimizer.zero_grad()  # 清零梯度

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs.view(-1, len(lettre2id)), targets.view(-1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if batch_idx % 100 == 0:  # 每100个批次打印一次损失
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
            writer.add_scalar('Loss/train', loss.item(), epoch * len(data_trump) + batch_idx)

# 训练完成后保存模型
torch.save(model.state_dict(), 'trump_rnn_model.pth')
writer.close()

def generate_text(model, start_str, gen_length=100):
    model.eval()  # 设置模型为评估模式
    generated = [lettre2id[c] for c in normalize(start_str)]  # 开始字符串转为索引
    input_tensor = torch.tensor(generated).unsqueeze(0).to(device)  # 转换为张量并添加批次维度

    with torch.no_grad():
        for _ in range(gen_length):
            output = model(input_tensor)  # 前向传播
            next_char_prob = F.softmax(output[:, -1, :], dim=-1)  # 取最后一个时间步的输出并应用softmax
            next_char = torch.multinomial(next_char_prob, 1).item()  # 根据概率分布进行采样
            generated.append(next_char)  # 追加生成的字符
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_char]]).to(device)), dim=1)  # 更新输入

    return code2string(generated)

# 使用示例
print(generate_text(model, "I will make America great again", gen_length=50))