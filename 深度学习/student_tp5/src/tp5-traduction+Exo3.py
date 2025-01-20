import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import time
import re
import sentencepiece as spm

# 设置日志
logging.basicConfig(level=logging.INFO)

# 文件路径
FILE = "M2A_AMAL/student_tp5/data/en-fra.txt"
# writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()

class Vocabulary:
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]

class TradDataset():
    def __init__(self, data, vocOrig, vocDest, sp_model, adding=True, max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1: continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len or len(dest) > max_len: continue
            
            # 使用 SentencePiece 模型进行分词
            orig_tokens = sp_model.encode(orig, out_type=str)
            dest_tokens = sp_model.encode(dest, out_type=str)
            
            # 将分词后的单词转化为词汇表中的ID
            orig_tensor = torch.tensor([vocOrig.get(o) for o in orig_tokens] + [Vocabulary.EOS])
            dest_tensor = torch.tensor([vocDest.get(d) for d in dest_tokens] + [Vocabulary.EOS])
            self.sentences.append((orig_tensor, dest_tensor))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]

def collate_fn(batch):
    orig, dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig), o_len, pad_sequence(dest), d_len

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8 * len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN = 100
BATCH_SIZE = 100

# 训练 SentencePiece 模型
def train_sentencepiece_model(input_file, model_prefix, vocab_size=1000):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=['<pad>', '<eos>', '<sos>', '<unk>']
    )

# 执行训练
train_sentencepiece_model(FILE, 'spm_model')

# 加载训练好的模型
sp_model = spm.SentencePieceProcessor(model_file='spm_model.model')

# 创建数据集
datatrain = TradDataset("".join(lines[:idxTrain]), vocEng, vocFra, sp_model, max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]), vocEng, vocFra, sp_model, max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        output, hidden = self.gru(packed)
        return hidden

# 解码器
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        embedded = self.embedding(x)         # (1, batch_size, embed_size)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output.squeeze(0))  # (batch_size, output_size)
        output = self.softmax(output)
        return output, hidden

    # 解码器的 generate 功能
    def generate(self, hidden, max_length=20):
        batch_size = hidden.size(1)
        input_token = torch.tensor([Vocabulary.SOS] * batch_size, device=device).unsqueeze(0)  # SOS 作为第一个输入
        outputs = []
        
        for _ in range(max_length):
            output, hidden = self.forward(input_token, hidden)
            top1 = output.argmax(1)  # 获取概率最大的词
            outputs.append(top1.item())
            if top1.item() == Vocabulary.EOS:  # 如果是 EOS，则停止生成
                break
            input_token = top1.unsqueeze(0)

        return outputs

# 训练循环函数
def train_model(encoder, decoder, dataloader, encoder_optimizer, decoder_optimizer, criterion, num_epochs=10, teacher_forcing_ratio=0.5):
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 获取数据
            src, src_len, tgt, tgt_len = batch
            src, tgt = src.to(device), tgt.to(device)

            # 初始化优化器和梯度
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # 编码器前向传递
            hidden = encoder(src, src_len)

            # 解码器初始化
            decoder_input = torch.tensor([Vocabulary.SOS] * BATCH_SIZE, device=device).unsqueeze(0)
            loss = 0

            # 随机决定是否使用教师强制
            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

            # 解码器前向传递
            if use_teacher_forcing:
                for t in range(tgt.size(0)):
                    output, hidden = decoder(decoder_input, hidden)
                    decoder_input = tgt[t].unsqueeze(0)  # 下一步的输入是当前目标词
                    loss += criterion(output, tgt[t])
            else:
                for t in range(tgt.size(0)):
                    output, hidden = decoder(decoder_input, hidden)
                    top1 = output.argmax(1)  # 获取概率最大的词
                    decoder_input = top1.unsqueeze(0)  # 下一步的输入是生成的词
                    loss += criterion(output, tgt[t])

            # 反向传播和优化
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 评估模型函数
def evaluate(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            src, src_len, tgt, tgt_len = batch
            src = src.to(device)

            # 编码器前向传递
            hidden = encoder(src, src_len)

            # 解码器生成翻译
            output_sentence = decoder.generate(hidden)
            print(f"Generated: {vocFra.getwords(output_sentence)}")
            print(f"Target: {vocFra.getwords(tgt)}")

# 初始化模型、损失函数和优化器
encoder = Encoder(input_size=len(vocEng), embed_size=256, hidden_size=512).to(device)
decoder = Decoder(output_size=len(vocFra), embed_size=256, hidden_size=512).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# 训练模型
train_model(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion, num_epochs=10)

# 测试模型
evaluate(encoder, decoder, test_loader)
