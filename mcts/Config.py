#!usr/bin/python
# coding=utf8
from matplotlib import pyplot as plt
from matplotlib import colors


class ConfigMCTS(object):
    def __init__(self) -> None:
        super().__init__()
        # int: [1...)
        # 优化预算
        self.total_evals = 400
        # int: [leaf_size...)
        # 初始样本点数
        self.init_evals = 40
        # int: [1...)
        # 固定树划分不变，执行 selects 次分支选择及模拟
        self.selects = 20
        # int: [1...)
        # 每次选择将选取排序前 beam 个分支同时进行模拟
        self.beam = 1
        # bool
        # 使用传统的 MCTS 选择策略还是将所有叶子节点同时排序
        # 进行 multi-beam 搜索时，这一项必须为 True; 如果为 False，self.beam将被置为1
        self.multi_beam = False
        # int: [1...)
        # 每次模拟采样的样本点数
        self.batch_evals = 1
        # bool
        # 是否使用核密度估计，当数据集较大时会导致性能下降
        self.use_kde = False
        # int: [1...)
        # 允许节点划分的最小样本点数
        self.leaf_size = 10
        # int: [1...)
        # 最大树深度，根节点为 0
        self.max_depth = 5
        # float
        # 划分出的子空间的均值最小相对提升，用于控制划分进行
        # TODO: 功能未实现
        self.min_lift = 0
        # float: [0...)
        # UCB的探索因子
        self.cp = 0.1
        # bool
        # 是否对最小化问题进行了最大化重构，影响命令行输出和绘图效果
        self.inverse = False
        # int: [1...)
        # 每隔 dump_step 个 iteration 进行一次保存
        self.dump_step = 1
        # bool
        # 是否显示更多命令行信息
        self.verbose = False
        # ['sobol', 'random', 'rs', 'mc', 'uniform']
        # 初始化时使用的全局采样器类型
        self.global_sampler_type = "sobol"
        # ['sobol', 'random', 'rs', 'mc', 'uniform']
        # 模拟时使用的局部采样器类型
        self.local_sampler_type = "sobol"


class ConfigInfoScissor(object):
    def __init__(self) -> None:
        super().__init__()
        # float
        # 损失函数中空间熵项的系数
        self.c_ent = 1
        # float
        # 损失函数中平衡项系数
        self.c_ub = 0.01
        # float: 0 ~ 1
        # 最高容忍不平衡程度
        self.max_ub = 0.99
        # float
        # 学习率
        self.lr = 2e-3
        # int
        # 余弦退火半周期
        self.T_0 = 20
        # int
        # 余弦退火衰退率
        self.T_mult = 2
        # int
        # 最高训练epochs
        self.train_epochs = 2500
        # int
        # 当损失函数这么多epochs未更新时，结束训练
        self.no_update_epochs = 10
        # float
        # 判断损失函数未更新的残差阈值
        self.tol = 1e-3


class ConfigKDE(object):
    def __init__(self) -> None:
        super().__init__()
        # int
        # ANN-KDE 近似近邻核密度估计器中，近邻向量数量
        self.k = 20
        # ['gaussian', ]
        # KDE中核函数
        self.kernel_type = "gaussian"
        # ['Flat L2', ]
        # faiss向量检索使用的 Index 方式
        self.index_type = "Flat L2"
        # int
        # faiss向量检索加速结构更新周期，单位为 #samples
        self.update_period = 1


class ConfigMetric(object):
    def __init__(self) -> None:
        super().__init__()
        # int
        # 是否需要评估算法效果，以及多久评估一次，单位是 iteration, -1不评估
        self.eval_metrics_step = 1
        # bool
        # 是否对最小化问题进行了最大化重构，影响命令行输出和绘图效果
        self.inverse = False
        # int or tuple
        # reshape真值，计算帧差异时的分辨率
        self.dpi = 100
        # ['interpolate', ]
        # 评价算法时建立回归面的方法
        self.regression_method = "interpolate"
        # ['nearest','cubic','linear']
        # 插值方式，当回归方法选择为 interpolate 时生效
        self.interpolate_method = "nearest"
        # (['<', '<=', '>', '>=', '=='], float) or ('in', [float, float])
        # 提取 IR（感兴趣区域）的方法，由两项构成
        self.IR_rule = (">=", 0.9)


class ConfigPlot(object):
    def __init__(self) -> None:
        super().__init__()
        # int
        # 是否需要绘制划分图，以及每隔多少迭代轮次绘制一次，-1不绘制
        self.plot_partition_step = 4
        # ["mean", "min", "max", "median", "regression"]
        # 绘制切分时的模式
        self.plot_partition_mode = "mean"
        # bool
        # 是否绘制选择区域图，注意划分图和选择区域图一一对应，绘制频率继承
        self.plot_selected = True
        # bool
        # 是否在绘制划分图时绘制数据点
        self.plot_scatter = True
        # bool
        # 是否在划分图和选择区域图中标出本轮采集的样本点
        self.plot_scatter_selected = True
        # int
        # 是否绘制插值图，以及每隔多少迭代轮次绘制一次，-1不绘制
        self.plot_interpolate_step = 4
        # ['nearest', 'linear', 'cubic']
        # 绘制插值图的模式
        self.plot_interpolate_mode = "nearest"
        # int
        # 是否绘制采样动力学图，以及每隔多少迭代轮次绘制一次，-1不绘制
        self.plot_dynamics_step = 4
        # int
        # 是否绘制密度分布图，以及每隔多少迭代轮次绘制一次，-1不绘制
        self.plot_density_step = 4
        # int
        # 是否绘制收敛曲线，以及收敛曲线多少迭代轮次绘制一次，-1不绘制
        self.plot_curve_step = 4
        # {'index': ['color', float]}
        # 绘制收敛曲线时选择哪几个指标
        self.curve_index_color_width = {
            "recall": ["indianred", 1.0],
            "precision": ["peru", 1.0],
            "F2 score": ["royalblue", 2.0],
            "TNR": ["mediumpurple", 1.0],
        }

        # [int, (int, int)]
        # 绘制等高线图、热力图等时的采样精度
        self.dpi = 300
        # colors.Norm
        # 等高线图、散点图等的归一化器
        self.norm = colors.Normalize(vmin=0, vmax=20)
        # int or list
        # 等高线图的阶梯数目
        self.levels = 50
        # plt.cm.CMap
        # 色彩映射
        self.cmap = plt.cm.viridis
        # float: [0...1]
        # 等高线图颜色填充透明度
        self.contourf_alpha = 0.7
        # float: [0...1]
        # 散点图散点透明度
        self.scatter_alpha = 1.0
        # float: [0...)
        # 散点大小
        self.scatter_s = 1.0
