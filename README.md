# 基于 pytorch的通用型训练框架
## 项目结构
```
├── configs
│   ├── cls/*.yaml # 分类模型配置文件
│   └── det/*.yaml # 检测模型配置文件

├── tools
│   ├── train.py # 训练入口
│   ├── test.py  # 测试入口
│   └── val      # 验证入口
├── train_data   # 训练数据 + 验证数据 (可选+ 测试数据)
├── README.md    # 本工程介绍
└── train_utility  # 训练框架相关的核心代码
    ├── data     # 数据处理相关
    ├── engine   # 模型训练框架相关
    ├── losses   # 损失函数相关
    ├── metrics  # 评价指标相关
    ├── modelling  # 模型搭建相关
    |      ├── architectures # 模型结构连接相关的
    |      ├── neck          # 模型颈部相关
    |      ├── backbones     # 模型骨架相关
    |      └── heads         # 模型头部相关
    └── optimizer # 优化器相关
    └── postprocess # 后处理相关(模型输出结果处理)
```
## 目标
旨在方便快速的进行数学建模, 便于复现, 工程更加规范化！并且可以快速的接入 huggingface平台上的先进模型!!!