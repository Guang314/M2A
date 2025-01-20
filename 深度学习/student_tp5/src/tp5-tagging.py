import itertools
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse
logging.basicConfig(level=logging.INFO)

DATA_PATH = "M2A_AMAL/student_tp5/data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu")
raw_train = [parse(x)[0] for x in data_file if len(x)>1]
data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu")
raw_dev = [parse(x)[0] for x in data_file if len(x)>1]
data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu")
raw_test = [parse(x)[0] for x in data_file if len(x)>1]

train_data = TaggingDataset(raw_train, words, tags, True)
dev_data = TaggingDataset(raw_dev, words, tags, True)
test_data = TaggingDataset(raw_test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)



#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

# Seq2Seq 模型实现函数
class POSSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(POSSeq2SeqModel, self).__init__()
        # 定义嵌入层，将词ID映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Vocabulary.PAD)
        # 定义LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 定义输出层，将LSTM的隐藏状态映射到标签空间
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentences):
        # 输入：sentences (batch_size, seq_len)
        embeds = self.embedding(sentences)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # lstm_out: (batch_size, seq_len, hidden_dim)
        tag_space = self.fc(lstm_out)  # (batch_size, seq_len, tagset_size)
        return tag_space

# OOV 随机替换
def replace_with_oov(sentences, oov_id, replace_prob=0.1):
    """在训练过程中随机将部分词语替换为OOV token"""
    for sentence in sentences:
        for i in range(len(sentence)):
            if torch.rand(1).item() < replace_prob:  # 以 replace_prob 概率替换词语
                sentence[i] = oov_id
    return sentences

# 模型训练函数
def train_model(model, train_loader, dev_loader, epochs, learning_rate, replace_prob):
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sentences, tags in train_loader:
            # 随机将部分输入词替换为OOV
            sentences = replace_with_oov(sentences, Vocabulary.OOVID, replace_prob)
            
            # 前向传播
            tag_scores = model(sentences)
            tag_scores = tag_scores.view(-1, tag_scores.shape[-1])  # 将输出展平
            tags = tags.view(-1)  # 展平真实标签
            
            # 计算损失
            loss = criterion(tag_scores, tags)
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 打印每个epoch的损失
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        # 在验证集上评估模型
        evaluate_model(model, dev_loader)

# 模型评估函数
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sentences, tags in data_loader:
            tag_scores = model(sentences)
            predictions = torch.argmax(tag_scores, dim=-1)
            
            # 忽略填充值部分
            mask = (tags != Vocabulary.PAD)
            correct += (predictions[mask] == tags[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    print(f'Accuracy: {accuracy:.4f}')

# 定义超参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
LEARNING_RATE = 0.001
REPLACE_PROB = 0.1  # OOV 替换概率

# 初始化模型
vocab_size = len(words)
tagset_size = len(tags)
model = POSSeq2SeqModel(vocab_size, tagset_size, EMBEDDING_DIM, HIDDEN_DIM)

# 启动训练
train_model(model, train_loader, dev_loader, EPOCHS, LEARNING_RATE, REPLACE_PROB)

def visualize_results(model, sentence, words_vocab, tags_vocab):
    model.eval()
    with torch.no_grad():
        word_ids = torch.LongTensor([words_vocab[word] for word in sentence])
        word_ids = word_ids.unsqueeze(0)  # 增加batch维度
        tag_scores = model(word_ids)
        predictions = torch.argmax(tag_scores, dim=-1).squeeze(0)  # 去掉batch维度
        
        predicted_tags = [tags_vocab.getword(tag_id) for tag_id in predictions]
        print(f'Sentence: {" ".join(sentence)}')
        print(f'Predicted Tags: {" ".join(predicted_tags)}')

# 可视化示例
example_sentence = ["Je", "mange", "une", "pomme"]
visualize_results(model, example_sentence, words, tags)