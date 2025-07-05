# MSRFomer: Multi-Scale Refinement Transformer for Handwritten Chinese Character Recognition

本项目提出了一种结合 CNN 和多尺度 Transformer（MSA + Multi-scale ViT）结构的手写汉字识别模型，采用“两阶段训练”策略：先使用 CNN 进行预训练，再通过融合 Transformer 层微调以提升字符识别效果。

---

## 🧩 项目结构

### 🧪 运行步骤

```bash
cd pretrain_code           # 进入预训练目录
python train.py            # 开始 CNN 模型训练
mv checkpoint.pth ../MSRFomer/Pretrained_pth/
cd ..                      # 返回项目根目录
python MSRFomer/train.py   # 开始微调主模型
