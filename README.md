# High-Quality Diffusion Distillation on a Single GPU with Relative and Absolute Position Matching

### [paper link](https://arxiv.org/abs/2503.20744), 2025, by Guoqiang Zhang, N. Kenta, J. P. Lewis, C. Mesnage, and W. Bastiaan Kleijn

<a href="URL_REDIRECT" target="blank"><img align="center" src="https://github.com/guoqiang-zhang-x/RAPM/blob/main/image_examples/RAPM_images.png" width="900" /></a>



### Description
Existing diffusion distillation methods such as DMD2 and PCM are both effective and resource-demanding in terms of the number of GPUs. In this project, we propose a new diffusion distillation method, named as __relative and absolute position matching (RAPM)__.   

<a href="URL_REDIRECT" target="blank"><img align="center" src="https://github.com/guoqiang-zhang-x/RAPM/blob/main/image_examples/my_awesome.gif" width="300" /></a>

| ![my_awesome.gif]([http://www.storywarren.com/wp-content/uploads/2016/09/space-1.jpg](https://github.com/guoqiang-zhang-x/RAPM/blob/main/image_examples/my_awesome.gif)) | 
|:--:| 
| *Space* |

### News
__2025-05-24:__ the LoRA weights of RAPM after distillating SDXL was uploaded to [Huggingface](https://huggingface.co/guoqiang-x/RAPM_SDXL).




### Code for generating images from pre-trained models 
```ruby

import os
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "guoqiang-x/RAPM_SDXL"
lora_weight_name = "RAPM_SDXL.pt"

hf_hub_download(repo_id=repo_name, filename=lora_weight_name, local_dir="./")
lora_state_dict = torch.load(lora_weight_name)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    scheduler=DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        timestep_spacing="trailing",
        clip_sample=False, 
        set_alpha_to_one=False,
    ), 
    revision=None,
    torch_dtype=torch.float16,
)

pipeline = pipeline.to('cuda')

pipeline.load_lora_weights(lora_state_dict)
pipeline.fuse_lora()

prompt = "a photo of a woman"
seed = 50
generator = torch.Generator(device='cuda').manual_seed(seed)
image = pipeline(
        prompt=prompt,
        generator=generator,
        num_inference_steps=4,
        num_images_per_prompt=1,
        guidance_scale=1.0,
    ).images[0]

image.save(os.path.join("./",'sample.jpg')) 

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


