import sys
import os
import random


import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPFeatureExtractor, CLIPProcessor, CLIPVisionModel
from peft import PeftModel, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from model_i2t import Image2Token

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)



def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default", cache_dir="huggingface/hub", local_files_only=True
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_name_or_path, torch_dtype=dtype, requires_safety_checker=False, cache_dir=cache_dir, local_files_only=local_files_only
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe


def load_adapter(pipe, ckpt_dir, adapter_name):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    pipe.unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)


def set_adapter(pipe, adapter_name):
    pipe.unet.set_adapter(adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.set_adapter(adapter_name)


def merging_lora_with_base(pipe, ckpt_dir, adapter_name="default"):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if isinstance(pipe.unet, PeftModel):
        pipe.unet.set_adapter(adapter_name)
    else:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)
    pipe.unet = pipe.unet.merge_and_unload()

    if os.path.exists(text_encoder_sub_dir):
        if isinstance(pipe.text_encoder, PeftModel):
            pipe.text_encoder.set_adapter(adapter_name)
        else:
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )
        pipe.text_encoder = pipe.text_encoder.merge_and_unload()

    return pipe


def create_weighted_lora_adapter(pipe, adapters, weights, adapter_name="default"):
    pipe.unet.add_weighted_adapter(adapters, weights, adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.add_weighted_adapter(adapters, weights, adapter_name)

    return pipe


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    parser.add_argument('--lora_path', type=str, default="texdreamer_u128_t16_origin", help='Lora path')
    parser.add_argument('--cache_dir', type=str, default="huggingface", help='Cache directory for Huggingface models')
    parser.add_argument('--test_folder', type=str, default="data/input", help='Path to test folder')
    parser.add_argument('--output_folder', type=str, default="output/i2uv", help='Output folder for generated images')

    args = parser.parse_args()


    myseed = args.seed
    MODEL_NAME = "stabilityai/stable-diffusion-2-1"  
    CLIP_NAME="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    lora_path = args.lora_path
    i2uv_path = "texdreamer_u128_t16_origin/i2uv"
    local_files_only=True
    uv_mask = Image.open("data/smpl_uv_mask.png").convert("L")
        
    
    processor = CLIPProcessor.from_pretrained(CLIP_NAME, cache_dir=cache_dir, local_files_only=local_files_only)
    pipe = get_lora_sd_pipeline(lora_path, base_model_name_or_path=MODEL_NAME, adapter_name="hutex")
    set_adapter(pipe, adapter_name="hutex")
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker=None
    
    
    i2t_decoder = Image2Token()
    i2t_decoder.load_state_dict(torch.load(os.path.join(i2uv_path, "i2t_decoder.pth")))
    i2t_decoder.eval()
    i2t_decoder.to(pipe.device)
    
    i2uv_vision_encoder_path = os.path.join(i2uv_path, 'vision_encoder')
    if os.path.exists(i2uv_vision_encoder_path):
        CLIP_NAME=i2uv_vision_encoder_path
    vision_encoder = CLIPVisionModel.from_pretrained(CLIP_NAME, cache_dir=cache_dir, local_files_only=local_files_only)
    vision_encoder.eval()
    vision_encoder.to(pipe.device)
    
        
    test_folder = args.test_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    
    for im_file in os.listdir(test_folder):
        
        if os.path.isdir(test_folder):
            folder_path = test_folder
            save_path = output_folder
            os.makedirs(save_path, exist_ok=True)
        
        if im_file.endswith('png'):
        
            im_pil = Image.open(os.path.join(folder_path, im_file))
            w,h=im_pil.size
            max_size = max(w,h)
            crop = Image.new("RGB", (max_size, max_size))
            crop.paste(im_pil, ((max_size-w)//2, (max_size-h)//2))

            with torch.no_grad():
                encoder_hidden_states = i2t_decoder(vision_encoder(processor(images=crop, return_tensors="pt")["pixel_values"].to(pipe.device)).last_hidden_state)
                set_seed(myseed)
                image = pipe(prompt_embeds=encoder_hidden_states, height=1024, width=1024, num_inference_steps=32, guidance_scale=2).images[0]
                image.save(os.path.join(save_path, im_file.replace('.jpg', '.png')))
                
                crop=crop.resize((1024,1024))
                show_img = Image.new("RGB", (1024*2,1024))
                show_img.paste(crop)
                show_img.paste(image, (1024,0))
                filename, extension = os.path.splitext(im_file)
                show_img.save(os.path.join(save_path, f"{filename}_a{extension}"))
                
 

    

    
    
