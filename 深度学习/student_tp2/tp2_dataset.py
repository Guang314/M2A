import torch
from torch.utils.data import Dataset, DataLoader

# 定义 Mondataset 类
class MonDataset(Dataset):
    
    # 定义初始化函数
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    # 定义 __getitem__ 方法
    def __getitem__(self, index):

        # 指定索引的 样本和标签
        image = self.data[index]
        label = self.labels[index]

        # 将图像转为张量并进行归一化处理
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        return image, label

    # 定义 __len__ 方法
    def __len__(self):

        # 返回数据长度
        return len(self.data)    

# 下载数据集
from sklearn.datasets import fetch_openml
x, y = fetch_openml(
    'mnist_784', 
    return_X_y = True,
    as_frame = False,
    data_home = "M2A_AMAL/student_tp2/data"
)

# 打印 labels
# print(y)         # mnist_784 数据集的 labels 是由 str 组成的列表

# 打印 data
# print(x)

# 创建数据集对象
mnist_784_dataset = MonDataset(
    data = x, 
    labels = y
)

# 创建数据加载器
dataloader = DataLoader(
    dataset = mnist_784_dataset,
    shuffle = True,
    batch_size = 100
)

# 测试 dataloader
print(f"Batch size: 100")

for batch_images, batch_labels in dataloader:
    print(f"Images batch shape: {batch_images.shape}")
    print(f"Labels batch shape: {len(batch_labels)}")
