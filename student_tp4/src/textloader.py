import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ## NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)     # 将法语的重音字符，转换为一般英语字符

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])                       # 将每个字符串的字母，转成 id 

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)                                         # 将 tensor 数字 转换为字符串


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])

def pad_collate_fn(samples: List[List[int]]):
    #  TODO:  Renvoie un batch à partir d'une liste de listes d'indexes (de phrases) qu'il faut padder.

    # 1.为每个句子增加 EOS 符号
    eos_value = 1
    samples = [sample.tolist() + [eos_value] for sample in samples]
    
    # 2.找到最大句子长度
    max_len = max(len(sample) for sample in samples)

    # 3.将每个句子填充 0 直到长度和最大句子相等
    pad_value = 0
    padded_samples = []

    for sample in samples:
        padded_sample = sample + [pad_value] * (max_len - len(sample))
        padded_samples.append(padded_sample)

    # 4.将其转换为 最大长度*句子数量 的 tensor 形式
    padded_samples = torch.tensor(padded_samples, dtype=torch.long).transpose(0,1)

    return padded_samples


if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
    data = next(iter(loader))
    # 检查 print(data)
    print("Chaîne à code : ", test)
    # Longueur maximum
    assert data.shape == (7, 3)
    print("Shape ok")
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    print("encodage OK")
    # Token EOS présent
    assert data[5,2] == EOS_IX
    print("Token EOS ok")
    # BLANK présent
    assert (data[4:,1]==0).sum() == data.shape[0]-4
    print("Token BLANK ok")
    # les chaînes sont identiques
    s_decode = " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    print("Chaîne décodée : ", s_decode)
    assert test == s_decode
    # " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
