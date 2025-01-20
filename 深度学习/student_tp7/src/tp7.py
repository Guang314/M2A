import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp7_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
test_size = 10000
train_size = len(train) - val_size -test_size
train, val, test = torch.utils.data.random_split(train, [train_size, val_size,test_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO: 

# --- Algorithm trivial
class BaselineModel(nn.Module):
    def __init__(self, num_classes=3):
        super(BaselineModel, self).__init__()
        self.num_classes = num_classes
        # 根据训练集计算多数类别（0、1 或 2）
        class_counts = [0, 0, 0]  # 每个类别的计数
        for text, label in train:
            class_counts[label] += 1
        self.majority_class = class_counts.index(max(class_counts))

    def forward(self, x):
        # 始终预测多数类别
        return torch.full((x.size(0),), self.majority_class, dtype=torch.long)
    
# --- CNN model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, kernel_size=3, num_filters=128, num_classes=3):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size, stride=1)
        self.fc = nn.Linear(num_filters*2, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)  # 输入序列的嵌入
        x = x.transpose(1, 2)  # 将维度转换为（batch, channels, length）
        x = self.conv1(x)  # 第一层卷积层
        x = self.pool(x)   # 最大池化层
        x = self.conv2(x)  # 第二层卷积层
        x = x.max(dim=2)[0]  # 对序列长度进行全局最大池化
        x = self.fc(x)  # 全连接层进行分类
        return x
    
# --- 训练和评估模型
def train_model(model, train_iter, val_iter, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_iter:
            text, labels = batch
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 日志记录
        epoch_loss = running_loss / len(train_iter)
        epoch_acc = correct / total
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # 验证
        validate_model(model, val_iter)
        
    writer.close()

def validate_model(model, val_iter):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_iter:
            text, labels = batch
            outputs = model(text)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    logging.info(f"验证准确率：{val_acc:.4f}")

# --- 主执行
baseline_model = BaselineModel(num_classes=3)
cnn_model = CNNModel(vocab_size=vocab_size, num_classes=3)

# 训练 CNN 模型
train_model(cnn_model, train_iter, val_iter, num_epochs=5)

# 评估基线模型
baseline_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_iter:
        text, labels = batch
        outputs = baseline_model(text)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

baseline_acc = correct / total
logging.info(f"基线模型准确率：{baseline_acc:.4f}")