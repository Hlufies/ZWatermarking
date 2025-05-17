<h1 align='Center'>Disentangled Style Domain for Implicit  $z$-Watermark Towards Copyright Protection</h1>
<div align='Center'>
    <a href='' target='_blank'>Junqiang Huang</a>&emsp;
    <a href='' target='_blank'>Zhaojun Guo</a>&emsp;
    <a href='' target='_blank'>Ge Luo</a>&emsp;
    <a href='' target='_blank'>Zhenxing Qian</a>&emsp;
    <a href='' target='_blank'>Sheng Li</a>&emsp;
    <a href='' target='_blank'>Xinpeng Zhang</a>&emsp;
</div>
<div align='Center'>
    Fudan University
</div>
<div align='Center'>
<i><strong><a href='[[https://eccv2024.ecva.net](https://neurips.cc/)](https://neurips.cc/)' target='_blank'>NeurIPS 2024</a></strong></i>
</div>


<div align='Center'>
    <a href='https://github.com/Hlufies/ZWatermarking'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>


## Introduction
Text-to-image models have shown surprising performance in high-quality image generation, while also raising intensified concerns about the unauthorized usage of personal dataset in training and personalized fine-tuning. In this paper, we introduce a novel implicit Zero-Watermarking scheme that first utilizes the disentangled style domain to detect unauthorized dataset usage in text-to-image models.

Based on your request for a fully English README optimization and referencing the provided search results, here's an enhanced version that aligns with international open-source standards, incorporating technical precision and developer-friendly conventions:

---


## 🚀 Quick Start  
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-green) 

Environment Configuration  
```bash
# Create conda environment (Python 3.10 required)
conda create -n zwatermark python=3.10 -y  
conda activate zwatermark

# Install project dependencies
pip install -r requirements.txt

# Clone repository
git clone https://github.com/Hlufies/ZWatermarking.git
cd ZWatermarking
```

Project Structure  
```markdown
ZWatermarking/
├── StyleDomain(IP)/          # Core watermarking algorithm implementation
│   ├── config/               # Model configuration files (YAML format)
│   ├── model/                # Network architectures
│   ├── pretrainedModel/      # Pre-trained weights directory
│   ├── dataset.py            # Data loading & preprocessing
│   ├── utils.py              # General utilities
│   ├── train.py              # Main training script
│   ├── train.sh              # One-click training automation
│   ├── test.py               # Model validation script
│   └── README.md             # Module-specific documentation
├── ZModel/                   # Auxiliary model components
├── utils.py                  # Global helper functions
├── train_utils.py            # Training pipeline utilities
├── valid_utils.py            # Validation metrics implementation
└── README.md                 # Main project documentation
```

## 运行
### 第一步 获取预训练的style domain encoder
cd StyleDomain(IP)
参考该文件夹下的Readme.md进行操作
### 第二步 训练ZModel以及版权推理

todolist
1. 更新训练文件脚本
2. 更新训练Readme.md部分内容
3. 更新版权推理标本



Here's the polished English version with professional technical terminology and standard open-source documentation practices:

---

🚀 Quick Start  
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-green) 

Environment Configuration  
```bash
# Create conda environment (Python 3.10 required)
conda create -n zwatermark python=3.10 -y  
conda activate zwatermark

# Install project dependencies
pip install -r requirements.txt

# Clone repository
git clone https://github.com/Hlufies/ZWatermarking.git
cd ZWatermarking
```

Project Structure  
```markdown
ZWatermarking/
├── StyleDomain_IP/           # Core watermark embedding/extraction module
│   ├── config/               # Model configuration files (YAML format)
│   ├── model/                # Network architecture implementations
│   ├── pretrainedModel/      # Pre-trained model weights
│   ├── dataset.py            # Data pipeline and preprocessing
│   ├── utils.py              # Common utility functions
│   ├── train.py              # Main training entry point
│   ├── train.sh              # Automated training script
│   ├── test.py               # Model validation and testing
│   └── README.md             # Module documentation
├── ZModel/                   # Ownership verification networks
├── Identifier.py             # Generate identifier embeddings
├── Identifier.txt            # Identifier.txt
├── utils.py                  # Global helper functions
├── train_utils.py            # Training pipeline components
├── valid_utils.py            # Validation metrics implementation
└── README.md                 # Project documentation hub
```

## 🛠️ Execution Workflow

Step 1: Obtain Pre-trained Style Domain Encoder
```bash
cd StyleDomain(IP)
# Follow instructions in the module's README.md for:
# - Model pretraining
# - Latent space configuration
# - Disentanglement parameter tuning
```
Step 2: Domain-Specific Identifier Injection

z serves as the key or special bias of the style domain. Identifier z can be the spatial embedding vector (e.g., image, text, audio, model, etc.). In this paper, we set the text **swz** to be converted into text feature embeddings by CLIP as z, embedding it into ZModel. This is achieved by maximizing the offset via identifier z, ensuring nonoverlap.

```bash
# Generate identifier embeddings and save model weights Identifier.pth
python Identifier.py
```


Step 2: Train Ownership Verification Model
```bash
# Navigate to ZModel directory
cd ../ZModel
```

📜 Citation  
```bibtex
@article{huang2024disentangled,
  title={Disentangled Style Domain for Implicit $ z $-Watermark Towards Copyright Protection},
  author={Huang, Junqiang and Guo, Zhaojun and Luo, Ge and Qian, Zhenxing and Li, Sheng and Zhang, Xinpeng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={55810--55830},
  year={2024}
}
```







