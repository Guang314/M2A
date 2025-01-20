import torch  # 导入 PyTorch 库
from torch.functional import norm  # 从 PyTorch 导入 norm 功能
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 函数式神经网络模块
from torch.utils.data import DataLoader, random_split, TensorDataset  # 导入 PyTorch 数据加载和处理工具
from pathlib import Path  # 导入路径操作模块
import time  # 导入时间模块，用于时间管理
import pytorch_lightning as pl  # 导入 PyTorch Lightning 框架
from pytorch_lightning.loggers import TensorBoardLogger  # 导入 TensorBoard 日志记录器

BATCH_SIZE = 311  # 设置批处理大小
TRAIN_RATIO = 0.8  # 训练数据占比
TEST_RATIO = 0.1  # 测试数据占比
LOG_PATH = "/tmp/runs/lightning_logs"  # 日志保存路径

# 更改数据路径
DATA_PATH = "/tmp/mnist"  # 数据存储路径

from sklearn.datasets import fetch_openml  # 从 scikit-learn 导入数据集获取工具

class Lit2Layer(pl.LightningModule):  # 定义一个继承自 LightningModule 的神经网络模型
    def __init__(self, dim_in, l, dim_out, learning_rate=1e-3):
        super().__init__()  # 调用父类构造函数
        # 定义一个包含两层线性变换的模型，第二层后接 ReLU 激活函数
        self.model = nn.Sequential(nn.Linear(dim_in, l), nn.ReLU(), nn.Linear(l, dim_out))
        self.learning_rate = learning_rate  # 学习率
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.name = "exemple-lightning"  # 模型名称
        self.valid_outputs = []  # 存储验证输出
        self.training_outputs = []  # 存储训练输出

    def forward(self, x):  # 定义前向传播的行为
        """ Définit le comportement forward du module"""
        x = self.model(x)  # 通过模型进行前向传播
        return x  # 返回模型输出
         
    def configure_optimizers(self):  # 定义优化器
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 使用 Adam 优化器
        return optimizer  # 返回优化器
    
    def training_step(self, batch, batch_idx):  # 定义训练步骤
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss), 
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch  # 解包批数据
        yhat = self(x)  # 获取模型预测
        loss = self.loss(yhat, y)  # 计算损失
        acc = (yhat.argmax(1) == y).sum()  # 计算准确率
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}  # 记录损失、准确率和样本数量
        self.log("accuracy", acc / len(x), on_step=False, on_epoch=True)  # 记录准确率
        self.valid_outputs.append({"loss": loss, "accuracy": acc, "nb": len(x)})  # 记录输出
        return logs  # 返回日志信息
    
    def validation_step(self, batch, batch_idx):  # 定义验证步骤
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch  # 解包批数据
        yhat = self(x)  # 获取模型预测
        loss = self.loss(yhat, y)  # 计算损失
        acc = (yhat.argmax(1) == y).sum()  # 计算准确率
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}  # 记录损失、准确率和样本数量
        self.log("val_accuracy", acc / len(x), on_step=False, on_epoch=True)  # 记录验证准确率
        self.valid_outputs.append({"loss": loss, "accuracy": acc, "nb": len(x)})  # 记录输出
        return logs  # 返回日志信息
    
    def test_step(self, batch, batch_idx):  # 定义测试步骤
        """ une étape de test """
        x, y = batch  # 解包批数据
        yhat = self(x)  # 获取模型预测
        loss = self.loss(yhat, y)  # 计算损失
        acc = (yhat.argmax(1) == y).sum()  # 计算准确率
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}  # 记录损失、准确率和样本数量
        return logs  # 返回日志信息
    
    def log_x_end(self, outputs, phase):  # 记录每个阶段的结果
        total_acc = sum([o['accuracy'] for o in outputs])  # 计算总准确率
        total_nb = sum([o['nb'] for o in outputs])  # 计算样本总数
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)  # 计算平均损失
        total_acc = total_acc / total_nb  # 计算平均准确率
        self.log_dict({f"loss/{phase}": total_loss, f"acc/{phase}": total_acc})  # 记录损失和准确率
        # self.logger.experiment.add_scalar(f'loss/{phase}', total_loss, self.current_epoch)  # 注释掉的：使用 logger 记录损失
        # self.logger.experiment.add_scalar(f'acc/{phase}', total_acc, self.current_epoch)  # 注释掉的：使用 logger 记录准确率
    
    def on_training_epoch_end(self):  # 训练周期结束后的钩子
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""
        self.log_x_end(self.training_outputs, 'train')  # 记录训练输出
        self.training_outputs.clear()  # 清空训练输出
        
    def on_validation_epoch_end(self):  # 验证周期结束后的钩子
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        self.log_x_end(self.valid_outputs, 'valid')  # 记录验证输出
        self.valid_outputs.clear()  # 清空验证输出
        
    def on_test_epoch_end(self):  # 测试周期结束后的钩子
        pass  # 目前不执行任何操作



class LitMnistData(pl.LightningDataModule):  # 定义一个数据模块，用于处理 MNIST 数据
    
    def __init__(self, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO):
        super().__init__()  # 调用父类构造函数
        self.dim_in = None  # 输入维度初始化
        self.dim_out = None  # 输出维度初始化
        self.batch_size = batch_size  # 批处理大小
        self.train_ratio = train_ratio  # 训练集比例
        self.test_ratio = test_ratio  # 测试集比例

    def prepare_data(self):  # 准备数据的方法
        ### Do not use "self" here.
        # S'il faut charger d'abord les données ou autres.
        pass  # 目前不执行任何操作
            
    def setup(self, stage=None):  # 设置数据的方法，根据阶段加载数据
        x, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False, data_home=DATA_PATH)  # 从 OpenML 获取 MNIST 数据集
        x = torch.tensor(x).view(-1, x.shape[1]).float() / 255.  # 将特征数据转换为张量并归一化
        y = torch.tensor(y.astype(int)).long()  # 将标签数据转换为长整型张量
        train_length = int(x.shape[0] * self.train_ratio)  # 计算训练集长度
        test_length = int(x.shape[0] * self.test_ratio)  # 计算测试集长度
        val_length = x.shape[0] - train_length - test_length  # 计算验证集长度
        if stage == "fit" or stage is None:  # 如果处于训练或验证阶段
            x, y = x[:x.shape[0] - test_length], y[:x.shape[0] - test_length]  # 划分训练和验证数据
            self.dim_in = x.shape[1]  # 设置输入维度
            self.dim_out = len(set(y))  # 设置输出维度（类别数量）
            ds_train = TensorDataset(x, y)  # 创建训练数据集
            self.mnist_train, self.mnist_val = random_split(ds_train, [train_length, x.shape[0] - train_length])  # 划分训练集和验证集
        if stage == "test":  # 如果处于测试阶段
            x, y = x[x.shape[0] - test_length:], y[x.shape[0] - test_length:]  # 划分测试数据
            self.mnist_test = TensorDataset(x, y)  # 创建测试数据集
            
    def train_dataloader(self):  # 返回训练数据加载器
        return DataLoader(self.mnist_train, batch_size=self.batch_size)  # 创建并返回训练数据加载器
    def val_dataloader(self):  # 返回验证数据加载器
        return DataLoader(self.mnist_val, batch_size=self.batch_size)  # 创建并返回验证数据加载器
    def test_dataloader(self):  # 返回测试数据加载器
        return DataLoader(self.mnist_test, batch_size=self.batch_size)  # 创建并返回测试数据加载器
    

data = LitMnistData()  # 创建 MNIST 数据模块实例

# 这些方法需要手动调用，因为需要输入维度。否则调用是自动的
data.prepare_data()  # 准备数据
data.setup(stage="fit")  # 设置数据，准备训练和验证集

model = Lit2Layer(data.dim_in, 10, data.dim_out, learning_rate=1e-3)  # 创建模型实例，输入维度、隐藏层大小和输出维度

logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)  # 创建 TensorBoard 日志记录器

trainer = pl.Trainer(default_root_dir=LOG_PATH, logger=logger, max_epochs=100)  # 创建训练器，设置日志路径和最大训练轮数
trainer.fit(model, datamodule=data)  # 训练模型
trainer.test(model, datamodule=data)  # 测试模型
