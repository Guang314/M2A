import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 RNN 模型
class RNN(nn.Module):
    
    def __init__(self, input_dim, latent_dim, output_dim):
        super(RNN, self).__init__()
        
        self.hidden_layer = nn.Linear(input_dim + latent_dim, latent_dim)                  # 隐藏层

        self.output_layer = nn.Linear(latent_dim, output_dim)                 # 解码层

        self.relu = nn.ReLU()                                                # 激活函数

    def one_step(self, x, h):

        combine = torch.cat((x,h), dim=-1)

        h_next = self.relu(self.hidden_layer(combine))

        return h_next
    
    def forward(self, x, h):

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

# 用于Exo4的 RNN 模型
class RNN_exo4(nn.Module):
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
        return output

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station


class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]