# High-Quality Diffusion Distillation on a Single GPU with Relative and Absolute Position Matching

### [paper link](https://arxiv.org/abs/2503.20744), 2025, by Guoqiang Zhang, N. Kenta, J. P. Lewis, C. Mesnage, and W. Bastiaan Kleijn

<a href="URL_REDIRECT" target="blank"><img align="center" src="https://github.com/guoqiang-zhang-x/RAPM/blob/main/image_examples/RAPM_images.png" width="900" /></a>



### Description
Existing diffusion distillation methods such as DMD2 and PCM are both effective and resource-demanding in terms of the number of GPUs. In this project, we propose a new diffusion distillation method, named as __relative and absolute position matching (RAPM)__. In brief, RAPM sucessfully distillates SDXL over a single GPU with batchsize=1 without using real images. From a high-level point of view, RAPM trains the student model by mimicing the teacher's fine-grainded trajectories by matching both the relative and absolute positions per coarse time-slot. 

<a href="URL_REDIRECT" target="blank"><img align="center" src="https://github.com/guoqiang-zhang-x/RAPM/blob/main/image_examples/RAPM_demo.png" width="500" /></a>


In the experiment, the single GPU being used was A6000 with 48GB. The rank of the Lora adaptive was set to 1. As a result, the lora parameters takes only a very small memory (about 7MB).    

<p href="URL_REDIRECT" target="blank"><img align="center" src="https://github.com/guoqiang-zhang-x/RAPM/blob/main/image_examples/my_awesome.gif" width="460" /></p>
<p><em> Distillating SDXL by RAPM with 4-step sampling 
over 20K iterations with batchsize=1 
</em>
</p>


### FID and CLIP scores by using 30K text-image pairs from COCO 2014


<table style="width:100%">
  <tr>
    <th></th>
    <th>GPUs for training</th>
    <th>Training batch size</th>
    <th>training time +overhead </th>
    <th>FID</th>
    <th>CLIP</th>
    <th>Use real images</th>
  </tr>
  <tr>
    <td> PCM  </td>
    <td> 8 A100  </td>
    <th>16</th>
    <th> - </th>
    <td>22.84</td>
    <td>30.36</td>
    <th>Yes</th>
  </tr>
  <tr>
    <td>DMD2 </td>
    <td> 64 A00  </td>
    <th>128 </th>
    <th> 60 hours </th>
    <td>18.24 </td>
    <td>30.85</td>
    <th>Yes</th>
  </tr>
  <tr>
    <td>RAPM (our) </td>
    <td> 1 A6000  </td>
    <th> 1 </th>
    <th> 31.5 hours </th>
    <td>19.21</td>
    <td>30.50</td>
    <th>No</th>
  </tr>  
</table>


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


