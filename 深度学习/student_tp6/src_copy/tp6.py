import logging  # 导入 logging 模块，用于记录日志
logging.basicConfig(level=logging.INFO)  # 配置日志记录的基本设置，设置日志级别为 INFO

import os  # 导入 os 模块，提供与操作系统交互的功能
from pathlib import Path  # 导入 Path 类，用于处理文件路径
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数式 API
from torch.utils.data import DataLoader, Dataset, random_split  # 导入数据加载和处理工具
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 日志记录工具
from tqdm import tqdm  # 导入 tqdm，用于显示进度条
import click  # 导入 click 模块，用于命令行界面构建

from sklearn.datasets import fetch_openml  # 从 scikit-learn 导入数据集获取工具

# 数据路径设置
DATA_PATH = "tmp/mnist"  # 指定数据存储路径
# DATA_PATH = "M2A_AMAL/student_tp6/tmp/minist"

# 数据集划分比例
TRAIN_RATIO = 0.05  # 训练集占总数据集的比例
TEST_RATIO = 0.2  # 测试集占总数据集的比例

def store_grad(var):
    """存储反向传播中的梯度

    对于张量 x, 在调用 `loss.backward` 之前调用 `store_grad(x)`。
    梯度将可通过 `x.grad` 获取。
    """
    def hook(grad):  # 定义一个钩子函数来存储梯度
        var.grad = grad  # 将梯度存储在 var 的 grad 属性中
    var.register_hook(hook)  # 注册钩子函数
    return var  # 返回变量

# [[STUDENT]] 需要实现的部分

BATCH_SIZE = 300  # 设置批处理大小
NUM_CLASSES = 10  # 设置分类数量（数字 0-9）
ITERATIONS = 1000  # 设置迭代次数
DEFAULT_DIMS = [100, 100, 100]  # 设置默认的隐藏层维度
RANDOM_SEED = 14  # 设置随机种子，保证可重复性

# 设置设备为 GPU（如果可用），否则使用 CPU
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 下载 MNIST 数据集
all_x, all_y = fetch_openml("mnist_784", return_X_y=True, as_frame=False, data_home=DATA_PATH)  # 从 OpenML 获取 MNIST 数据
all_x = torch.tensor(all_x).view(-1, all_x.shape[1]).float() / 255.  # 转换特征数据为张量并归一化到 [0, 1]
all_y = torch.tensor(all_y.astype(int)).long()  # 转换标签数据为长整型张量

# 划分训练集和测试集
test_length = int(TEST_RATIO * all_x.shape[0])  # 计算测试集长度
train_images, train_labels = all_x[:test_length].reshape(-1, 28, 28), all_y[:test_length]  # 获取训练图像和标签
test_images, test_labels = all_x[test_length:].reshape(-1, 28, 28), all_y[test_length:]  # 获取测试图像和标签

# 输入图像的维度（展平）
INPUT_DIM = train_images.shape[1] * train_images.shape[2]  # 输入维度为 28*28

# -- Exo 1: 数据加载器

class MnistDataset(Dataset):  # 定义 MNIST 数据集类，继承自 PyTorch 的 Dataset
    def __init__(self, images, labels):  # 构造函数，接受图像和标签
        self.images = images  # 存储图像
        self.labels = labels  # 存储标签

    def __getitem__(self, index):  # 根据索引获取图像和标签
        return self.images[index], self.labels[index]  # 返回图像和对应标签

    def __len__(self):  # 返回数据集的大小
        return len(self.images)  # 返回图像的数量

# 设置随机种子以保证可重复性
torch.manual_seed(RANDOM_SEED)
train_images = torch.FloatTensor(train_images) / 255.  # 将训练图像转换为 FloatTensor 并归一化
ds = MnistDataset(train_images, train_labels)  # 创建训练数据集实例
test_images = torch.FloatTensor(test_images) / 255.  # 将测试图像转换为 FloatTensor 并归一化
test_data = MnistDataset(test_images, test_labels)  # 创建测试数据集实例

# 划分训练集和验证集
train_length = int(len(ds) * TRAIN_RATIO)  # 根据训练比例计算训练集长度
train_data, val_data = random_split(ds, (train_length, len(ds) - train_length))  # 随机划分训练集和验证集
logging.info("Kept %d samples out of %d for training", train_length, len(ds))  # 记录日志，显示训练样本数量

class State:  # 定义状态类，用于保存和加载模型状态
    def __init__(self, path: Path, model, optim):  # 构造函数，接受模型路径、模型和优化器
        self.path = path  # 存储模型路径
        self.model = model  # 存储模型
        self.optim = optim  # 存储优化器
        self.epoch = 0  # 当前轮次
        self.iteration = 0  # 当前迭代次数

    @staticmethod
    def load(path: Path):  # 静态方法，加载模型状态
        if path.is_file():  # 如果文件存在
            with path.open("rb") as fp:  # 以二进制方式打开文件
                state = torch.load(fp, map_location=DEVICE)  # 加载模型状态
                logging.info("Starting back from epoch %d", state.epoch)  # 记录日志，显示从哪个轮次开始
                return state  # 返回加载的状态
        return State(path, None, None)  # 如果文件不存在，返回一个新状态

    def save(self):  # 保存当前状态的方法
        savepath_tmp = self.path.parent / ("%s.tmp" % self.path.name)  # 创建临时保存路径
        with savepath_tmp.open("wb") as fp:  # 以二进制方式打开临时文件
            torch.save(self, fp)  # 保存当前状态到临时文件
        os.rename(savepath_tmp, self.path)  # 重命名临时文件为目标路径


# 定义不同的归一化方法
NORMALIZATIONS = {
    "identity": None,  # 不使用归一化
    "batchnorm": lambda dim: nn.BatchNorm1d(dim),  # 批量归一化
    "layernorm": lambda dim: nn.LayerNorm(dim)  # 层归一化
}

class Model(nn.Module):  # 定义模型类，继承自 nn.Module
    def __init__(self, in_features, out_features, dims, dropouts, normalization_str="identity"):  # 构造函数
        super().__init__()  # 调用父类构造函数

        layers = []  # 初始化层列表
        normalization = NORMALIZATIONS[normalization_str]  # 根据字符串选择归一化方式

        self.id = f"n{normalization_str}"  # 初始化模型 ID
        self.trackedlayers = set()  # 用于跟踪存储梯度的层
        dim = in_features  # 输入特征数量

        for newdim, p in zip(dims, dropouts):  # 遍历维度和 dropout 概率
            layers.append(nn.Linear(dim, newdim))  # 添加线性层
            dim = newdim  # 更新当前维度
            self.id += f"-{dim}_{p}"  # 更新模型 ID

            if p > 0:  # 如果 dropout 概率大于 0
                layers.append(nn.Dropout(p))  # 添加 dropout 层

            if normalization:  # 如果有归一化方法
                layers.append(normalization(dim))  # 添加归一化层

            self.trackedlayers.add(layers[-1])  # 跟踪此层

            layers.append(nn.ReLU())  # 添加 ReLU 激活层

        layers.append(nn.Linear(dim, out_features))  # 添加输出层
        self.layers = nn.Sequential(*layers)  # 将所有层组合为一个顺序容器
    
    def forwards(self, input):  # 定义前向传播的函数
        outputs = []  # 存储输出
        for module in self.layers:  # 遍历所有层
            input = module(input)  # 将输入传递给当前层
            if module in self.trackedlayers:  # 如果当前层是跟踪的层
                outputs.append(store_grad(input))  # 存储该层的输出梯度

        return input, outputs  # 返回最终输出和跟踪的输出

    def forward(self, input):  # 定义标准前向传播
        return self.layers(input)  # 直接通过所有层返回输出


def run(iterations, model, l1, l2):  # 定义运行模型的函数
    """运行模型""" 
    model_id = model.id  # 获取模型 ID
    if l1 > 0:  # 如果 L1 正则化系数大于 0
        model_id += f"-l1_{l1:.2g}"  # 更新模型 ID
    if l2 > 0:  # 如果 L2 正则化系数大于 0
        model_id += f"-l2_{l2:.2g}"  # 更新模型 ID
    cumloss = torch.tensor(0)  # 初始化累积损失
    savepath = Path(f"models/model-{model_id}.pth")  # 定义模型保存路径
    writer = SummaryWriter(f"runs/{model_id}")  # 创建 TensorBoard 记录器

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)  # 创建训练数据加载器
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)  # 创建测试数据加载器

    state = State.load(savepath)  # 加载模型状态
    if state.model is None:  # 如果没有模型
        state.model = model.to(DEVICE)  # 将模型转移到设备
        state.optim = torch.optim.Adam(state.model.parameters(), lr=1e-4)  # 使用 Adam 优化器

    it = 0  # 初始化迭代计数器
    model = state.model  # 获取当前模型
    loss = nn.CrossEntropyLoss()  # 定义交叉熵损失
    loss_nagg = nn.CrossEntropyLoss(reduction='sum')  # 定义交叉熵损失（总和）

    def batches(loader):  # 定义批处理生成器
        for x, y in loader:  # 遍历数据加载器
            x = x.to(DEVICE).reshape(x.shape[0], INPUT_DIM)  # 将数据转移到设备并展平
            y = y.long().to(DEVICE)  # 将标签转移到设备
            yield x, y  # 生成批次

    for epoch in tqdm(range(state.epoch, iterations)):  # 遍历每个轮次
        # 遍历批次
        model.train()  # 设置模型为训练模式
        for x, y in batches(train_loader):  # 遍历训练数据批次
            state.optim.zero_grad()  # 清空梯度
            l = loss(model(x), y)  # 计算损失

            total_loss = l  # 初始化总损失

            if l1 > 0:  # 如果 L1 正则化系数大于 0
                l1_loss = 0  # 初始化 L1 损失
                for name, value in model.named_parameters():  # 遍历模型参数
                    if name.endswith(".weight"):  # 如果参数是权重
                        l1_loss += value.abs().sum()  # 计算 L1 正则化损失

                l1_loss *= l1  # 乘以 L1 正则化系数
                total_loss += l1_loss  # 加入总损失
                writer.add_scalar('loss/l1', l1_loss, state.iteration)  # 记录 L1 损失到 TensorBoard
                
            if l2 > 0:  # 如果 L2 正则化系数大于 0
                l2_loss = 0  # 初始化 L2 损失
                for name, value in model.named_parameters():  # 遍历模型参数
                    if name.endswith(".weight"):  # 如果参数是权重
                        l2_loss += (value ** 2).sum()  # 计算 L2 正则化损失
                l2_loss *= l2  # 乘以 L2 正则化系数
                total_loss += l2_loss  # 加入总损失
                writer.add_scalar('loss/l2', l2_loss, state.iteration)  # 记录 L2 损失到 TensorBoard

            total_loss.backward()  # 反向传播计算梯度
            state.optim.step()  # 更新模型参数

            writer.add_scalar('loss/train', l, state.iteration)  # 记录训练损失到 TensorBoard
            state.iteration += 1  # 增加迭代计数器

            if state.iteration % 500 == 0:  # 每 500 次迭代进行日志记录
                logprobs, outputs = model.forwards(x)  # 前向传播并获取输出
                with torch.no_grad():  # 不计算梯度
                    probs = nn.functional.softmax(logprobs, dim=1)  # 计算概率分布
                    writer.add_histogram(f'entropy', -(probs * probs.log()).sum(1), state.iteration)  # 记录熵到 TensorBoard

                l = loss(logprobs, y)  # 计算损失
                l.backward()  # 反向传播计算梯度
                for ix, output in enumerate(outputs):  # 遍历输出
                    writer.add_histogram(f'output/{ix}', output, state.iteration)  # 记录输出到 TensorBoard
                    writer.add_histogram(f'grads/{ix}', output.grad, state.iteration)  # 记录梯度到 TensorBoard

                ix = 0  # 初始化索引
                for module in model.layers:  # 遍历模型层
                    if isinstance(module, nn.Linear):  # 如果是线性层
                        writer.add_histogram(f'linear/{ix}/weight', module.weight, state.iteration)  # 记录权重到 TensorBoard
                        ix += 1  # 增加索引

        # 评估模型
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度
            cumloss = 0  # 初始化累积损失
            cumcorrect = 0  # 初始化累积正确预测数量
            count = 0  # 初始化计数器
            for x, y in batches(test_loader):  # 遍历测试数据批次
                logprobs = model(x)  # 前向传播计算输出
                cumloss += loss_nagg(logprobs, y)  # 计算损失并累加
                cumcorrect += (logprobs.argmax(1) == y).sum()  # 计算正确预测数量并累加
                count += x.shape[0]  # 更新样本计数

            writer.add_scalar('loss/test', cumloss.item() / count, state.iteration)  # 记录测试损失到 TensorBoard
            writer.add_scalar('correct/test', cumcorrect.item() / count, state.iteration)  # 记录测试准确率到 TensorBoard

        state.epoch = epoch + 1  # 更新当前轮次
        state.save()  # 保存当前状态

    # 返回最后一次测试损失
    return cumloss.item() / count  # 返回平均测试损失


def model(dims, dropouts, normalization="identity", l1=0, l2=0):  # 定义创建模型的函数
    return Model(INPUT_DIM, NUM_CLASSES, dims, dropouts, normalization_str=normalization), l1, l2  # 返回模型实例和正则化参数


@click.group()  # 定义点击命令行组
@click.option('--iterations', default=ITERATIONS, help='Number of iterations.')  # 定义迭代次数参数
@click.option('--device', default='cpu', help='Device for computation')  # 定义计算设备参数
@click.pass_context  # 允许上下文传递
def cli(ctx, iterations, device):  # 定义命令行界面函数
    global DEVICE  # 声明全局变量
    ctx.obj["iterations"] = iterations  # 将迭代次数存储在上下文中
    DEVICE = torch.device(device)  # 根据参数设置设备

@cli.command()  # 定义子命令 vanilla
@click.pass_context  # 允许上下文传递
def vanilla(ctx):  # 定义 vanilla 命令的函数
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0]))  # 运行模型，使用默认参数

@cli.command()  # 定义子命令 l2
@click.pass_context  # 允许上下文传递
def l2(ctx):  # 定义 l2 命令的函数
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0], l2=1e-3))  # 运行模型，使用 L2 正则化

@cli.command()  # 定义子命令 l1
@click.pass_context  # 允许上下文传递
def l1(ctx):  # 定义 l1 命令的函数
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0], l1=1e-4))  # 运行模型，使用 L1 正则化

@cli.command()  # 定义子命令 dropout
@click.pass_context  # 允许上下文传递
def dropout(ctx):  # 定义 dropout 命令的函数
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0.2,0.2,0.2]))  # 运行模型，使用 dropout

@cli.command()  # 定义子命令 batchnorm
@click.pass_context  # 允许上下文传递
def batchnorm(ctx):  # 定义 batchnorm 命令的函数
    run(ctx.obj["iterations"], *model(DEFAULT_DIMS, [0,0,0], 'batchnorm'))  # 运行模型，使用 Batch Normalization

if __name__ == '__main__':  # 当脚本作为主程序执行时
    cli(obj={})  # 运行命令行界面


# [[/STUDENT]]
