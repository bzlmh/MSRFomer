# Ancient Handwritten Chinese Character Recognition via Multi-Style Attention and Feature Fusion

A deep learning framework for ancient handwritten Chinese character recognition, integrating Convolutional Neural Networks (CNNs) with a multi-scale Transformer architecture. The proposed model is designed to capture both fine-grained local stroke patterns and global semantic dependencies, enabling robust recognition across diverse handwriting styles.

---

## Overview

Ancient handwritten Chinese character recognition remains a challenging task due to style diversity, structural complexity, and limited annotated resources. To address these issues, this project introduces a hybrid recognition framework that combines CNN-based local feature extraction with Transformer-based global context modeling.

The model adopts a **two-stage training strategy**:

### Stage 1 — CNN Pretraining
The CNN backbone is trained independently to learn local structural features and stroke-level representations.

### Stage 2 — Transformer Fine-Tuning
Multi-scale Transformer layers are introduced for global semantic interaction and feature refinement, further improving recognition accuracy.

This strategy enhances both convergence stability and overall performance.

---

## Features

- **Multi-style attention mechanism** for handling handwriting diversity  
- **Multi-scale feature fusion** between CNN and Transformer representations  
- **Two-stage optimization** for stable and efficient training  
- **Improved recognition accuracy** on ancient handwritten Chinese character datasets  

---

## Project Structure

```bash
.
├── pretrain_code/              # CNN pretraining stage
├── MSRFomer/                   # Main model training and fine-tuning
│   ├── Pretrained_pth/         # Directory for pretrained checkpoints
│   └── train.py
└── README.md

```

## Citation

<a name="citation"></a>

If you find this work helpful, please cite our paper:

```bibtex
@INPROCEEDINGS{AHCCR,
  author    = {Zhang, Tianyi and Liu, Menghui and Yang, Yilan and Zuo, Fang and Wang, Guanghui},
  booktitle = {2025 5th International Conference on Artificial Intelligence, Automation and High Performance Computing (AIAHPC)},
  title     = {Ancient Handwritten Chinese Character Recognition via Multi-Style Attention and Feature Fusion},
  year      = {2025},
  pages     = {29--32},
}

```
