# 这次具体看了什么

上一版的问题不在于训练数据造假，而是只做到了分类演示。单次测试分数高，不等于用户能完成一批图片的处理。这次先看成熟项目如何处理模型之外的部分。

| 仓库 / 实际读过的位置 | 值得带过来的设计 | 这次不照搬的部分 |
|---|---|---|
| [pytorch/vision](https://github.com/pytorch/vision/blob/main/references/classification/train.py)：train_one_epoch、evaluate、checkpoint；utils.py 的指标累计 | 按样本数量累计损失；训练和验证明确分开；保存可复查配置 | ImageNet 多 GPU、EMA 等先不引入，本机 CPU 和数据规模用不上 |
| [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML/blob/main/madewithml/evaluate.py)：get_per_class_metrics、get_slice_metrics；tests/code/test_data.py | 不只看总体分数，要看类别和困难子集；数据划分本身也需要检查 | 不引入 Ray 集群和整套 NLP 工作流 |
| [Label Studio](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/tasks/models.py)：Task、prediction、annotation、model_version、锁相关方法 | 预测和人工标签不是同一种事实；修改要有记录；任务可能被多人同时打开 | 不复制 Django 大型应用；本次用 SQLite 的事务和版本字段处理本地并发冲突 |
| [FastAPI 官方模板](https://github.com/fastapi/full-stack-fastapi-template/blob/master/backend/app/api/routes/items.py)：分页、资源查找、事务提交与错误响应 | 后端先有稳定数据模型，接口对状态负责 | 不为单机演示增加没有使用场景的账户系统；明确只绑定本机 |
| [Cleanlab](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/filter.py)：find_label_issues 的样本外概率要求 | 不把“模型不同意标签”当作标签一定错误；训练集评估不能冒充样本外证据 | 不宣称复现 confident learning；本次只是低置信度优先的复核队列 |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR)、[docTR](https://github.com/mindee/doctr)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的 README | OCR 包含文本定位、识别和结构化输出，范围远大于单数字分类 | 本次不宣传整页 OCR，不用其预训练模型替代手写 ResNet |
| [DVC](https://github.com/iterative/dvc) 的项目介绍 | 数据与模型版本应能追踪 | 小项目先用固定划分、文件哈希与模型指纹，不安装重型数据平台 |

也调研过 TrashNet 的数据说明和许可证：它的白底六类照片和当前数字项目是不同问题。用户明确选择继续做数字识别，因此没有下载该数据、没有改做垃圾分类。

这些是设计参考，未复制上述项目源代码。残差网络、置信度排序、SQLite 乐观并发控制都不是新算法。能否称为有用的改进，要看队列实验和真实操作是否成立。
