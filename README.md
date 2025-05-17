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
```

Project Structure  
```markdown
ZWatermarking/
├── StyleDomain/               # Core watermarking algorithm implementation
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


📜 Citation  
```bibtex
@article{zwatermark2025,
  title={Disentangled Style Encoding via Self-Decoupled Diffusion},
  author={Anonymous},
  journal={Submitted to CVPR},
  year={2025}
}
```







