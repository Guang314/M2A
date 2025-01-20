from textloader import  string2code, id2lettre
import math
import torch
import numpy as np

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles

    # 1. 初始化隐藏状态
    h = None                    # 隐状态初始化为 None
    generate_sequence = []      # 用于储存生成的字符索引

    # 2. 将 start 转换为字符串索引
    if start:
        input_seq = string2code(start)    # 将开始字符串转换为编码
        generate_sequence.extend(input_seq.tolist())   # 添加到生成序列
        # 获取最后一个字符的嵌入
        x = rnn.embedding(input_seq[-1].unsqueeze(0)).unsqueeze(0)    # (1, 1, input_dim)
    else:
        # 如果没有 start 字符串，则使用 EOS 的输入
        x = rnn.embedding(torch.tensor([eos]).unsqueeze(0)).unsqueeze(0)   # (1, 1, input_dim)

    # 3. 生成字符串序列
    for _ in range(maxlen):
        # 传递当前输入和隐状态到 RNN
        output, h = rnn(x, h)        # output: (1, batch_size, output_size)

        # 使用 decoder 获取 Logits
        logits = decoder(output[-1])    # 获取最后时间步的输出 (1, output_dim)

        # 4. 从 Logits 中选取下一个字符
        next_char_logits = logits.squeeze(0)     # (output_dim, )

        # 确定生成方式 (随机生成/确定性生成)
        # 使用 softmax 进行概率分布
        probs = torch.softmax(next_char_logits, dim=-1).detach().numpy()
        next_char = np.random.choice(len(probs), p=probs)      # 随机选择下一个字符

        # 将选中的字符添加到生成序列中
        generate_sequence.append(next_char)

        # 如果生成的是 EOS，终止生成
        if next_char == eos:
            break

        # 准备下一个输入的嵌入
        x = rnn.embedding(torch.tensor([next_char]).unsqueeze(0)).unsqueeze(0)   # (1, 1, input_dim)

    # 5. 将生成的序列转换为字符串
    generated_string = ''.join(id2lettre[i] for i in generate_sequence)

    return generated_string
        

import torch.nn.functional as F

def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
    使用束搜索生成序列：在每次迭代中，对于每个候选序列，探索 k 个最可能的符号；
    然后仅保留所有生成的序列中概率最高的 k 个，作为下一步的候选序列。
    
    Args:
        rnn: RNN 网络
        emb: 词嵌入层
        decoder: 解码器，用于生成下一个 token 的概率分布
        eos: 结束符号的 token ID
        k: 束搜索的宽度
        start: 初始字符串
        maxlen: 生成序列的最大长度
        
    Returns:
        最终生成的序列
    """

    # 1. 初始化
    h = None  # 隐状态初始化
    sequences = [(start, 0)]  # 序列和其对应的对数概率 (sequence, log_probability)
    finished_sequences = []  # 保存已经完成的序列

    # 2. 将 start 转换为字符串索引并获取嵌入表示
    if start:
        input_seq = string2code(start)  # 将开始字符串转换为编码
        x = emb(input_seq[-1].unsqueeze(0)).unsqueeze(0)  # 获取最后一个字符的嵌入
    else:
        # 如果没有 start 字符串，则使用 EOS 作为输入
        x = emb(torch.tensor([eos]).unsqueeze(0)).unsqueeze(0)  # 开始符号的嵌入

    # 3. 生成序列
    for _ in range(maxlen):
        all_candidates = []

        # 对每一个候选序列进行扩展
        for seq, score in sequences:
            # 如果序列已经生成了结束符号，直接保存
            if seq and seq[-1] == eos:
                finished_sequences.append((seq, score))
                continue

            # 传递当前输入和隐状态到 RNN
            output, h = rnn(x, h)
            
            # 使用 decoder 获取下一个符号的 logits
            logits = decoder(output[-1])  # 获取最后时间步的输出 (1, output_dim)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # 计算 log 概率

            # 选择前 k 个最高概率的符号
            topk_probs, topk_indices = torch.topk(log_probs, k)

            # 对于每个符号扩展当前序列
            for i in range(k):
                next_char = topk_indices[i].item()
                next_score = score + topk_probs[i].item()  # 更新对数概率
                
                # 生成新的候选序列
                candidate = (seq + [next_char], next_score)
                all_candidates.append(candidate)

        # 选择 k 个概率最高的序列
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:k]

        # 准备下一个输入的嵌入
        next_char = sequences[0][0][-1]
        x = emb(torch.tensor([next_char]).unsqueeze(0)).unsqueeze(0)

        # 如果所有序列都已经生成结束符号，提前终止
        if all([seq[-1] == eos for seq, _ in sequences]):
            break

    # 如果没有生成结束符的序列，将所有的序列视为完成的
    if not finished_sequences:
        finished_sequences = sequences

    # 返回对数概率最高的序列
    best_sequence = max(finished_sequences, key=lambda x: x[1])[0]

    return ''.join(id2lettre[i] for i in best_sequence)


# p_nucleus

def p_nucleus(decoder, alpha: float):
    """
    返回一个核采样概率分布函数，该函数根据当前 RNN 的隐藏状态计算输出的概率分布。
    
    Args:
        * decoder: 给定 RNN 状态后返回 logits 的函数
        * alpha (float): 覆盖的概率质量阈值（如 0.95）
    
    Returns:
        compute: 核采样的计算函数
    """
    
    def compute(h):
        """
        计算核采样概率分布。
        
        Args:
            * h (torch.Tensor): RNN 的隐藏状态
            
        Returns:
            核采样概率分布，基于 logits 的概率子集。
        """
        # 1. 使用 decoder 获取 logits 并转换为概率
        logits = decoder(h)  # (1, output_dim)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # 获取概率分布 (output_dim,)
        
        # 2. 对概率进行降序排列
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 3. 累积概率直到超过 alpha，确定候选子集 I_alpha
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, alpha).item() + 1  # 超过 alpha 的最小集合
        
        # 4. 截取前 cutoff_index 个符号，形成候选集合 I_alpha
        nucleus_probs = sorted_probs[:cutoff_index]
        nucleus_indices = sorted_indices[:cutoff_index]
        
        # 5. 重新归一化这些概率
        nucleus_probs /= nucleus_probs.sum()
        
        # 6. 从这个重新归一化的分布中采样
        next_char = torch.multinomial(nucleus_probs, 1).item()
        
        # 返回采样的字符 ID
        return nucleus_indices[next_char].item()
    
    return compute

def generate_beam_nucleus(rnn, emb, decoder, eos, k, alpha, start="", maxlen=200):
    """
    使用核采样和束搜索生成序列。
    
    Args:
        * rnn: RNN 网络
        * emb: 词嵌入层
        * decoder: 解码器，用于生成下一个 token 的概率分布
        * eos: 结束符号的 token ID
        * k: 束搜索的宽度
        * alpha: 核采样的概率阈值
        * start: 初始字符串
        * maxlen: 生成序列的最大长度
        
    Returns:
        最终生成的序列
    """
    
    # 初始化
    h = None  # 隐状态初始化
    sequences = [(start, 0)]  # 序列和其对应的对数概率 (sequence, log_probability)
    finished_sequences = []  # 保存已经完成的序列

    # 初始化嵌入输入
    if start:
        input_seq = string2code(start)  # 将开始字符串转换为编码
        x = emb(input_seq[-1].unsqueeze(0)).unsqueeze(0)  # 获取最后一个字符的嵌入
    else:
        # 如果没有 start 字符串，则使用 EOS 作为输入
        x = emb(torch.tensor([eos]).unsqueeze(0)).unsqueeze(0)  # 开始符号的嵌入

    # 获取核采样的概率计算函数
    compute_nucleus_prob = p_nucleus(decoder, alpha)
    
    # 生成序列
    for _ in range(maxlen):
        all_candidates = []

        # 对每一个候选序列进行扩展
        for seq, score in sequences:
            # 如果序列已经生成了结束符号，直接保存
            if seq and seq[-1] == eos:
                finished_sequences.append((seq, score))
                continue

            # 传递当前输入和隐状态到 RNN
            output, h = rnn(x, h)
            
            # 使用核采样计算下一个符号
            next_char = compute_nucleus_prob(output[-1])  # 核采样生成字符

            # 更新对数概率（暂时使用 score 不变的情况）
            next_score = score  # 核采样的概率目前不用于更新总分

            # 生成新的候选序列
            candidate = (seq + [next_char], next_score)
            all_candidates.append(candidate)

        # 选择 k 个概率最高的序列
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:k]

        # 准备下一个输入的嵌入
        next_char = sequences[0][0][-1]
        x = emb(torch.tensor([next_char]).unsqueeze(0)).unsqueeze(0)

        # 如果所有序列都已经生成结束符号，提前终止
        if all([seq[-1] == eos for seq, _ in sequences]):
            break

    # 如果没有生成结束符的序列，将所有的序列视为完成的
    if not finished_sequences:
        finished_sequences = sequences

    # 返回对数概率最高的序列
    best_sequence = max(finished_sequences, key=lambda x: x[1])[0]

    return ''.join(id2lettre[i] for i in best_sequence)
