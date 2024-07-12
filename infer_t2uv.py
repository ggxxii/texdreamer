import sys
import os
from pathlib import Path
from typing import Optional
import argparse

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
from transformers import AutoTokenizer, PretrainedConfig, CLIPFeatureExtractor
from peft import PeftModel, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

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
    parser.add_argument('--save_path', type=str, default="output/t2uv", help='Save path for generated images')
    parser.add_argument('--test_list', type=str, default="data/sample_prompts.txt", help='Path to input txt file')

    args = parser.parse_args()
    
    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    check_min_version("0.10.0.dev0")

    logger = get_logger(__name__)
    
    myseed = args.seed
    MODEL_NAME = "stabilityai/stable-diffusion-2-1"  
    lora_path = args.lora_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    uv_mask = Image.open("data/smpl_uv_mask.png").convert("L")

    positive_prompt = ", natural lighting, photo-realistic, 4k"
    negative_prompt = "overexposed, shadow, reflection, low quality, teeth, open mouth, eyes closed"

    pipe = get_lora_sd_pipeline(lora_path, base_model_name_or_path=MODEL_NAME, adapter_name="hutex")

    set_adapter(pipe, adapter_name="hutex")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    ###################genrate from .txt file###################
    test_list = args.test_list
    idx = 0
    with open(test_list, 'r') as f:
        for line in f.readlines():
            prompt = 'hutex, ' + line.strip()
            with torch.no_grad():
                set_seed(myseed)
                images = pipe(prompt + positive_prompt, height=1024, width=1024, num_inference_steps=32, guidance_scale=7.5,
                              negative_prompt=negative_prompt, num_images_per_prompt=1).images

            image = images[0]
            image.putalpha(uv_mask)
            image.save(os.path.join(save_path, '{:04d}.png'.format(idx)))

            idx += 1

    
    
