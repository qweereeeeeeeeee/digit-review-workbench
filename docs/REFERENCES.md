# 参考与创新边界

1. 考核原文：灵境竞赛组-后端AI方向-第一阶段考核任务.pdf。要求自主选题、Python/NumPy/Pandas/Matplotlib、手写 ResNet 图像分类、仓库历史、说明和运行视频。
2. 数据官方说明：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
3. 数据加载源码：https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_base.py
4. 原始数据页面：https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
5. 残差网络官方参考：https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
6. ResNet 论文：https://arxiv.org/abs/1512.03385
7. 温度缩放方法：https://proceedings.mlr.press/v70/guo17a.html
8. PyTorch 可复现性说明：https://docs.pytorch.org/docs/stable/notes/randomness.html

本项目未直接复制参考项目，也未调用 torchvision 的 ResNet。残差学习、温度缩放、拒识和遮挡敏感度均为已有方法，不宣称算法首创。项目级创新在于将这些方法与独立校准集、坏例展示、输入校验、模型校验值和实时界面结合，形成可核查的端到端交付。

UCI Digits 仅含 1797 个样本，且是原始数据集的一个子集。本项目重新拆分这 1797 个样本，不应与其他论文使用的原始 train/test 划分成绩直接比较。没有可靠的书写者 ID，因此采用样本级分层拆分；不能宣称跨书写者泛化。

没有执行迁移学习对照。考核将迁移学习列为理解内容，不能替代手写 ResNet。本项目理解其含义（复用预训练权重、冻结/微调），但为小尺寸单通道数据选择从零训练，避免将预训练骨干冒充手写实现。

## 数据署名与修改说明

Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49 。UCI 页面注明数据为 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。本项目通过 scikit-learn 获取其 1797 张子集，将像素除以 16，并新增独立划分与派生扰动；原始数据作者不为本项目结论背书。
