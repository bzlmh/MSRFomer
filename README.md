# MSRFomer: Multi-Scale Refinement Transformer for Handwritten Chinese Character Recognition

本项目提出了一种结合 CNN 和多尺度 Transformer（MSA + Multi-scale ViT）结构的手写汉字识别模型，采用“两阶段训练”策略：先使用 CNN 进行预训练，再通过融合 Transformer 层微调以提升字符识别效果。

---

## 🧩 项目结构

# 创建主目录
mkdir MSRFomer

# 进入主目录
cd MSRFomer

# 创建预训练代码目录及文件
mkdir pretrain_code
touch pretrain_code/train.py

# 创建主模型训练目录及子文件
mkdir MSRFomer
touch MSRFomer/train.py
mkdir MSRFomer/Pretrained_pth

# 创建README.md文档
touch README.md

# 返回上级目录
cd ..

使用 `pretrain_code/train.py` 在初始数据集上训练一个 CNN（如 ResNet），用于提取字符图像的初始特征。

### 🧪 运行步骤

```bash
cd pretrain_code           # 进入预训练目录
python train.py            # 开始 CNN 模型训练
mv checkpoint.pth ../MSRFomer/Pretrained_pth/
cd ..                      # 返回项目根目录
python MSRFomer/train.py   # 开始微调主模型
