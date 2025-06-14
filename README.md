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
├── StyleDomain(IP)/          # Core watermark embedding/extraction module
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

#### Step 1: Obtain Pre-trained Style Domain Encoder
```bash
cd StyleDomain(IP)

# Follow instructions in the module's README.md
```


#### Step 2: Domain-Specific Identifier Injection

```bash
# In this paper, we set 'swz' to be converted into vector by CLIP as z, embedding it into ZModel. 

# Generate identifier embeddings and save model weights Identifier.pth
python Identifier.py
```

#### Step 3: Training & Test Sets Preparation Guidelines

1. Prepare an unauthorized protected dataset​​ (50-100 samples). Rationale: "unauthorized" clarifies data ownership; "samples" specifies unit.
2. Fine-tune a surrogate generative model​​ using the unauthorized dataset, then generate similar images via targeted prompts. Recommendation: Apply ​​DreamBooth​​ for fine-tuning with ​​rare tokens​​ to enhance model learning and output quality. Note: Fidelity of the surrogate model directly correlates with generation efficacy.
3. ​​Generate mimicry images​​ for each protected sample using the surrogate model, producing 10-100 similar images per original input.
4. Construct paired data​​ of unauthorized samples and corresponding mimicry samples, then encode into ​​latent representations​​ (shape: [B，4， 64， 64]) via VAE.
5. Watermark Embedding for Protected Data Units​​
   1. Assign a ​​128-bit encoded zero watermark​​ to each protected data unit (e.g., image or bounding box). Annotate with a ​​unique index​​ and positive sample identifier z (e.g., label: "swz"). 128-bit length aligns with watermarking standards for copyright protection. Indexing supports traceability of leakage sources.
   2. Global Negative Sample Construction​​：​Randomly sample ​​background images​​ from COCO (e.g., images without objects or erased annotations) and Generate noise tensors with shape [B, 64, 4, 4]. Set watermark to all zeros (0x000...000) and label as non_z identifier.

   
#### Step 4: Train Ownership Verification Model
```bash
cd ../ZModel
```

## ⏳ TODO & Project Roadmap & Current Status  

• [ ] Improve the documentation for Step 3 and Step 4 

• [ ] Refine the code implementation for Step 3 and Step 4
  




## 📜 Citation  
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







