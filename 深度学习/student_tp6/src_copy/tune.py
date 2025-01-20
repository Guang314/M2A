import optuna  # 导入 Optuna 库，用于超参数优化
import torch.nn as nn  # 导入 PyTorch 的神经网络模块


def objective(trial):  # 定义目标函数，该函数将在每个试验中被优化
    from tp6 import Model, run, NUM_CLASSES, INPUT_DIM  # 从 tp6 模块中导入模型、运行函数、类别数量和输入维度

    iterations = 200  # 设置训练的迭代次数为 200
    dims = [100, 100, 100]  # 定义隐藏层的维度

    # 使用 Optuna 建议的分类参数，选择归一化方法
    norm_type = trial.suggest_categorical('normalization', ["identity", "batchnorm", "layernorm"])
    normalization = norm_type  # 将建议的归一化方法存储在变量中

    # 为每个隐藏层建议一个 dropout 概率（使用对数均匀分布）
    dropouts = [trial.suggest_loguniform('dropout_p%d' % ix, 1e-2, 0.5) for ix in range(len(dims))]

    # 在区间 [0, 1] 中为 L2 正则化参数建议一个值
    l2 = trial.suggest_uniform('l2', 0, 1)
    # 在区间 [0, 1] 中为 L1 正则化参数建议一个值
    l1 = trial.suggest_uniform('l1', 0, 1)

    # 创建模型实例，传入输入维度、类别数量、隐藏层维度、dropout 概率和归一化方法
    model = Model(INPUT_DIM, NUM_CLASSES, dims, dropouts, normalization)
    # 运行训练，并返回损失
    return run(iterations, model, l1, l2)  # 运行训练并返回损失


# 创建一个 Optuna 研究对象，默认采用 TPE（Tree-structured Parzen Estimator）算法
study = optuna.create_study()
# 优化目标函数，进行 20 次试验
study.optimize(objective, n_trials=20)
# 输出最佳超参数
print(study.best_params)  # 打印出最佳参数配置
