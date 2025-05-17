## Self-Decoupled Diffusion Model for Style Domain Encoder Pre-training on COCO Dataset  

🧠 Core Methodology  
1. **COCO Data Preprocessing**  
• VAE-based Latent Encoding:  

  Compress images to [Batch, 4, 64, 64] latent space using a β-VAE architecture:  
  ```python 
  vae = VAE()
  latent_tensor = vae.encode(images)  # [Batch,4,64,64]
  ```
  And then, save latent_tensor as 'coco_latent.pth'.

2. **Self-Decoupled Diffusion Training**  
Training Pipeline (`train.sh`):  
```bash
python train.py --config "/config/IP.yaml" 
```
3.**Get Pretrained-Style domain encoder**

```bash
python split.py 
```



