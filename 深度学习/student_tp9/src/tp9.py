
import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import time
import logging
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

GLOVE_PATH = Path("data/glove")
DATASET_PATH = Path("data/aclImdb")
IMDB_CLASSES  = ['neg','pos']


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")
    glove_fn = open(GLOVE_PATH / ("glove.6B.%dd.txt" % embedding_size))
    words, embeddings = [], []
    for line in glove_fn:
        values = line.split()
        words.append(values[0])
        embeddings.append([float(x) for x in values[1:]])

    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")


    return word2id, embeddings, FolderText(IMDB_CLASSES, DATASET_PATH /"train", tokenizer, load=False), FolderText(IMDB_CLASSES, DATASET_PATH / "test", tokenizer, load=False)




MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)

#  TODO: 
    ##  TODO: 

# Question 1

class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.scale = np.sqrt(output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        Q = self.query(x)   # [batch_size, seq_len, output_dim]
        K = self.key(x)     # [batch_size, seq_len, output_dim]
        V = self.value(x)   # [batch_size, seq_len, output_dim]

        # compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale    # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # apply attention weights to values
        context = torch.matmul(attention_weights, V)    # [batch_size, seq_len, output_dim]
        return context
    
class BaseSelfAttentionModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, output_dim):
        super(BaseSelfAttentionModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)       # [batch_size, seq_len, embedding_dim]

        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        
        # mean pooling over the sequence
        x = torch.mean(x, dim=1)     # [batch_size, hidden_dim]
        x = self.fc(x)               # [batch_size, output_dim]

        return x

# 训练模型
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    自定义的 collate 函数，用于处理批次内的文本长度不一致问题。
    """
    texts, labels = zip(*batch)
    # 填充文本序列到批次内的最大长度
    texts = [torch.tensor(text) for text in texts]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)  # 0 表示填充值
    labels = torch.tensor(labels)  # 转为张量
    return padded_texts, labels

if __name__ == "__main__":
    # 加载数据 与 模型训练
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    word2id, embeddings, train_data, test_data = get_imdb_data()

    # 使用自定义的 collate_fn 创建数据加载器
    batch_size = 32  # 批大小
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    hidden_dim = 128
    num_layers = 3
    output_dim = 2

    # 模型训练
    model = BaseSelfAttentionModel(embeddings, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 开始训练
    for epoch in range(5):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        eval_correct = evaluate_model(model, test_loader, device)
        print(f"\nEpoch: {epoch+1}\t Train_loss: {train_loss:.4f}\t\t Accuracy: {eval_correct}\n")

# [[/STUDENT]]
