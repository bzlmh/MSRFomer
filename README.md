# MSRFomer: Multi-Scale Refinement Transformer for Handwritten Chinese Character Recognition

本项目提出了一种结合 CNN 和多尺度 Transformer（MSA + Multi-scale ViT）结构的手写汉字识别模型，采用“两阶段训练”策略：先使用 CNN 进行预训练，再通过融合 Transformer 层微调以提升字符识别效果。

---

## 🧩 项目结构

MSRFomer/
├── pretrain_code/ # 预训练 CNN 特征提取器的代码
│ └── train.py # 基于 ResNet 的预训练脚本
├── MSRFomer/ # 主模型训练代码目录
│ ├── train.py # 微调阶段主训练脚本
│ └── Pretrained_pth/ # 存放预训练模型权重
└── README.md # 项目说明文档


---

## 🚀 第一步：预训练 CNN 提取特征

使用 `pretrain_code/train.py` 在初始数据集上训练一个 CNN（如 ResNet），用于提取字符图像的初始特征。

### 🧪 运行步骤

```bash
cd pretrain_code           # 进入预训练目录
python train.py            # 开始 CNN 模型训练
mv checkpoint.pth ../MSRFomer/Pretrained_pth/
cd ..                      # 返回项目根目录
python MSRFomer/train.py   # 开始微调主模型
