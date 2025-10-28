# MSRFomer: Ancient Handwritten Chinese Character Recognition via Multi-Style Attention and Feature Fusion

This paper proposes a handwritten Chinese character recognition model that integrates Convolutional Neural Networks with a multi-scale Transformer architecture. The model adopts a two-stage training strategy: in the first stage, CNN layers are pre-trained to capture local features, and in the second stage, Transformer layers are incorporated for fine-tuning, enhancing the overall recognition performance. This work has been accepted by AIAHPC 2025 (EI).

---

## 🧩 项目结构

### 🧪 运行步骤

```bash
cd pretrain_code           # 进入预训练目录
python train.py            # 开始 CNN 模型训练
mv checkpoint.pth ../MSRFomer/Pretrained_pth/
cd ..                      # 返回项目根目录
python MSRFomer/train.py   # 开始微调主模型
