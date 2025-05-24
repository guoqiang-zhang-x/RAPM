# High-Quality Diffusion Distillation on a Single GPU with Relative and Absolute Position Matching

### [paper link](https://arxiv.org/abs/2503.20744), 2025, by Guoqiang Zhang, N. Kenta, J. P. Lewis, C. Mesnage, and W. Bastiaan Kleijn

### Description 

### Code for generating images from pre-trained models 
```
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "guoqiang-x/RAPM_SDXL"
ckpt_name = "RAPM_SDXL.pt"

hf_hub_download(repo_id=repo_name, filename=ckpt_name, local_dir="./")

```

### Citation

If you find our work useful in your research, please cite:

```
@MISC{guoqiang2025rapm,
  title={High-Quality Diffusion Distillation on a Single GPU with Relative and Absolute Position Matching},
  author={G. Zhang and N. Kenta and J. P. Lewis and C. Mesnage and W. B. Kleijn},
  howpublished={arXiv:2503.20744v1},
  year={2025}
}
```


