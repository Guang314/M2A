
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

DATA_PATH = "../student_tp3/data/trump_full_speech.txt"

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    计算交叉熵损失，同时忽略填充字符
    :param output: Tensor, 形状为 (length, batch, output_dim)
    :param target: Tensor, 形状为 (length, batch)
    :param padcar: int, 填充字符的索引
    :return: Tensor, 损失值
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    
    # 1.调整 output 和 target 的形状，以适应 CrossEntropyLoss
    # 需要变成 (batch_size, length, output_dim)
    output = output.view(-1, output.size(2))
    target = target.view(-1)

    # 2.计算掩码
    mask = (target != padcar).float()

    # 3.计算非缩减的交叉熵损失
    loss = CrossEntropyLoss(output, target, reduce=None)

    # 4.通过掩码忽略填充字符的损失
    masked_loss = loss * mask

    # 5.计算掩码总和，用于归一化
    return masked_loss.sum() / mask.sum()

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, vocab_size, embedding_dim, latent_dim, output_dim):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.hidden_layer = nn.Linear(embedding_dim + latent_dim, latent_dim)                  # 隐藏层

        self.output_layer = nn.Linear(latent_dim, output_dim)                 # 解码层

        self.relu = nn.ReLU()                                                # 激活函数

    def one_step(self, x, h):

        combine = torch.cat((x,h), dim=-1)

        h_next = self.relu(self.hidden_layer(combine))

        return h_next
    
    def forward(self, x, h=None):

        # x 的形状是 (seq_length, batch_size)，需要嵌入转换
        x = self.embedding(x)

        if h is None:
            h = torch.zeros(x.size(1), self.hidden_layer.out_features)

        hidden_states = []      # 储存所有时间步的隐藏状态
        outputs = []            # 储存所有时间步的预测输出

        for t in range(x.size(0)):

            h = self.one_step(x[t], h)
            hidden_states.append(h.unsqueeze(0))

            out = self.decode(h)
            outputs.append(out.unsqueeze(0))

        hidden_states = torch.cat(hidden_states, dim=0)
        outputs = torch.cat(outputs, dim=0)

        return outputs, hidden_states
    
    def decode(self, h):

        output = self.output_layer(h)

        return output
    
# 用于 exo4 的 RNN 模型
""" class RNN_exo4(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN_exo4, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 嵌入层
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # RNN层
        self.fc = nn.Linear(hidden_dim, vocab_size)  # 全连接层

    def forward(self, x):
        # x: 输入的形状为 (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        rnn_out, _ = self.rnn(embedded)  # (batch_size, seq_len, hidden_dim)
        output = self.fc(rnn_out)  # (batch_size, seq_len, vocab_size)
        return output """


class LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # LSTM 门
        self.forget_gate = nn.Linear(input_dim + latent_dim, latent_dim)
        self.input_gate = nn.Linear(input_dim + latent_dim, latent_dim)
        self.output_gate = nn.Linear(input_dim + latent_dim, latent_dim)
        self.cell_state_update = nn.Linear(input_dim + latent_dim, latent_dim)
        
        self.output_layer = nn.Linear(latent_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h=None, C=None):
        if h is None:
            h = torch.zeros(x.size(1), self.latent_dim)
        if C is None:
            C = torch.zeros(x.size(1), self.latent_dim)

        hidden_states, cell_states, outputs = [], [], []
        
        for t in range(x.size(0)):
            x_t = x[t]

            combined = torch.cat((x_t, h), dim=-1)
            
            # LSTM 门计算
            f_t = self.sigmoid(self.forget_gate(combined))
            i_t = self.sigmoid(self.input_gate(combined))
            o_t = self.sigmoid(self.output_gate(combined))
            C_tilde = self.tanh(self.cell_state_update(combined))

            C = f_t * C + i_t * C_tilde  # 更新细胞状态
            h = o_t * self.tanh(C)       # 更新隐藏状态

            hidden_states.append(h.unsqueeze(0))
            cell_states.append(C.unsqueeze(0))
            
            # 输出层
            output = self.output_layer(h)
            outputs.append(output.unsqueeze(0))

        hidden_states = torch.cat(hidden_states, dim=0)
        outputs = torch.cat(outputs, dim=0)
        return outputs, hidden_states, C


class GRU(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # GRU 门
        self.update_gate = nn.Linear(input_dim + latent_dim, latent_dim)
        self.reset_gate = nn.Linear(input_dim + latent_dim, latent_dim)
        self.new_gate = nn.Linear(input_dim + latent_dim, latent_dim)
        
        self.output_layer = nn.Linear(latent_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(1), self.latent_dim)

        hidden_states, outputs = [], []
        
        for t in range(x.size(0)):
            x_t = x[t]

            combined = torch.cat((x_t, h), dim=-1)
            
            # GRU 门计算
            z_t = self.sigmoid(self.update_gate(combined))
            r_t = self.sigmoid(self.reset_gate(combined))
            
            combined_reset = torch.cat((x_t, r_t * h), dim=-1)
            h_tilde = self.tanh(self.new_gate(combined_reset))
            
            h = (1 - z_t) * h + z_t * h_tilde  # 更新隐藏状态

            hidden_states.append(h.unsqueeze(0))
            
            # 输出层
            output = self.output_layer(h)
            outputs.append(output.unsqueeze(0))

        hidden_states = torch.cat(hidden_states, dim=0)
        outputs = torch.cat(outputs, dim=0)
        return outputs, hidden_states


# 数据导入
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

PATH = "M2A_AMAL/student_tp3/data/"
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size=32, shuffle=True)



# 训练函数
def train_model(rnn_model, train_data, padcar, epochs=100, lr=0.001, clip_value=5):
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
    writer = SummaryWriter()  # TensorBoard 监控

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_data):
            rnn_model.train()
            optimizer.zero_grad()

            outputs, hidden_states = rnn_model(inputs)  # 前向传播
            loss = maskedCrossEntropy(outputs, targets, padcar)  # 计算 masked CrossEntropy
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()

            # TensorBoard 监控梯度和门值
            for name, param in rnn_model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(name, param.grad, epoch)

        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')
        writer.add_scalar('Training Loss', avg_loss, epoch)

    writer.close()

# 使用 Embedding 而不是 one-hot 的模型训练

# 主训练过程
def main():
    # 数据加载
    train_loader = data_trump
    padcar = 0

    # 模型初始化
    vocab_size = len(id2lettre)  # 字符表的大小
    embedding_dim = 128          # 嵌入维度
    latent_dim = 256             # 隐状态的维度
    output_dim = vocab_size      # 输出维度等于字符表大小

    # 使用 RNN/LSTM/GRU 初始化模型
    # model = RNN(vocab_size, embedding_dim, latent_dim, output_dim)
    # 或者选择 LSTM 或 GRU
    # model = LSTM(embedding_dim, latent_dim, output_dim)
    model = GRU(embedding_dim, latent_dim, output_dim)

    # 开始训练
    train_model(model, train_loader, padcar)

if __name__ == "__main__":
    main()

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot

