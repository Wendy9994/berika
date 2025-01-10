import numpy as np
from tqdm import tqdm
from PIL import Image

# PyTorch backend
import torch
from torch import autocast

# Transformers containing tools
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Accessing our pipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers import StableDiffusionImg2ImgPipeline


# Creating a class for customized Diffusion pipeline
class DiffusionPipeline:

    def __init__(self, 
                 vae, 
                 tokenizer, 
                 text_encoder, 
                 unet, 
                 scheduler):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_text_embeds(self, text):
        text_input = self.tokenizer(text, 
                                    padding='max_length', 
                                    max_length=self.tokenizer.model_max_length, 
                                    truncation=True, 
                                    return_tensors='pt')
        with torch.no_grad():
            text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeds

    def get_prompt_embeds(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]
        cond_embeds = self.get_text_embeds(prompt)
        uncond_embeds = self.get_text_embeds([''] * len(prompt))
        prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
        return prompt_embeds

    def decode_img_latents(self, img_latents):
        img_latents = 1 / self.vae.config.scaling_factor * img_latents
        with torch.no_grad():
            img = self.vae.decode(img_latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        return img

    def transform_img(self, img):
        img = (img * 255).round().astype('uint8')
        img = [Image.fromarray(i) for i in img]
        return img

    def encode_img_latents(self, img, latent_timestep):
        if not isinstance(img, list):
            img = [img]
        img = np.stack([np.array(i) for i in img], axis=0)
        img = 2 * ((img / 255.0) - 0.5)
        img = torch.from_numpy(img).float().permute(0, 3, 1, 2).to(self.device)
        img_latents_dist = self.vae.encode(img)
        img_latents = img_latents_dist.latent_dist.sample()
        img_latents = self.vae.config.scaling_factor * img_latents
        noise = torch.randn(img_latents.shape).to(self.device)
        img_latents = self.scheduler.add_noise(img_latents, noise, latent_timestep)
        return img_latents


class Depth2ImgPipeline(DiffusionPipeline):
    def __init__(self, 
                 vae, 
                 tokenizer, 
                 text_encoder, 
                 unet, 
                 scheduler, 
                 depth_feature_extractor, 
                 depth_estimator):
        super().__init__(vae, tokenizer, text_encoder, unet, scheduler)
        self.depth_feature_extractor = depth_feature_extractor
        self.depth_estimator = depth_estimator

    def get_depth_mask(self, img):
        if not isinstance(img, list):
            img = [img]
        width, height = img[0].size
        pixel_values = self.depth_feature_extractor(img, return_tensors="pt").pixel_values
        with autocast('cuda'):
            depth_mask = self.depth_estimator(pixel_values).predicted_depth
        depth_mask = torch.nn.functional.interpolate(depth_mask.unsqueeze(1),
                                                     size=(height//8, width//8),
                                                     mode='bicubic',
                                                     align_corners=False)
        depth_min = torch.amin(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_mask = 2.0 * (depth_mask - depth_min) / (depth_max - depth_min) - 1.0
        depth_mask = depth_mask.to(self.device)
        depth_mask = torch.cat([depth_mask] * 2)
        return depth_mask

    def denoise_latents(self, 
                        img,
                        prompt_embeds,
                        depth_mask,
                        strength,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        height=512, width=512):
        strength = max(min(strength, 1), 0)
        self.scheduler.set_timesteps(num_inference_steps)
        init_timestep = int(num_inference_steps * strength)
        t_start = num_inference_steps - init_timestep
        timesteps = self.scheduler.timesteps[t_start:]
        num_inference_steps -= t_start
        latent_timestep = timesteps[:1].repeat(1)
        latents = self.encode_img_latents(img, latent_timestep)
        with autocast('cuda'):
            for t in tqdm(timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)['sample']
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents

    def __call__(self, 
                 prompt, 
                 img, 
                 strength=0.8,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 height=512, width=512):
        prompt_embeds = self.get_prompt_embeds(prompt)
        depth_mask = self.get_depth_mask(img)
        latents = self.denoise_latents(img, prompt_embeds, depth_mask, strength,
                                       num_inference_steps, guidance_scale, height, width)
        img = self.decode_img_latents(latents)
        img = self.transform_img(img)
        return img


# Setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='vae').to(device)
tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='text_encoder').to(device)
unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='unet').to(device)
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)
depth_estimator = DPTForDepthEstimation.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='depth_estimator')
depth_feature_extractor = DPTImageProcessor.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='feature_extractor')

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth").to(device)

depth2img = Depth2ImgPipeline(vae, tokenizer, text_encoder, unet, scheduler, depth_feature_extractor, depth_estimator)

# Utility functions
import urllib.parse as parse
import os
import requests

def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
